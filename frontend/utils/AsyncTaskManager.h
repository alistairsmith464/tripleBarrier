#pragma once
#include <QObject>
#include <QFuture>
#include <QFutureWatcher>
#include <QtConcurrent>
#include <QProgressDialog>
#include <QTimer>
#include <functional>
#include <memory>

template<typename T>
class AsyncTask : public QObject {
    Q_OBJECT
    
public:
    using TaskFunction = std::function<T()>;
    using ProgressCallback = std::function<void(int)>;
    using CompletionCallback = std::function<void(const T&)>;
    using ErrorCallback = std::function<void(const QString&)>;
    
    AsyncTask(QObject* parent = nullptr) : QObject(parent) {
        m_watcher = std::make_unique<QFutureWatcher<T>>();
        connect(m_watcher.get(), &QFutureWatcher<T>::finished, this, &AsyncTask::onFinished);
        connect(m_watcher.get(), &QFutureWatcher<T>::progressValueChanged, this, &AsyncTask::onProgressChanged);
    }
    
    void run(TaskFunction task) {
        m_future = QtConcurrent::run(task);
        m_watcher->setFuture(m_future);
    }
    
    void setProgressCallback(ProgressCallback callback) { m_progressCallback = callback; }
    void setCompletionCallback(CompletionCallback callback) { m_completionCallback = callback; }
    void setErrorCallback(ErrorCallback callback) { m_errorCallback = callback; }
    
    void cancel() {
        if (m_future.isRunning()) {
            m_future.cancel();
        }
    }
    
    bool isRunning() const { return m_future.isRunning(); }
    bool isFinished() const { return m_future.isFinished(); }
    
signals:
    void progressChanged(int percentage);
    void completed(const T& result);
    void error(const QString& message);
    void cancelled();
    
private slots:
    void onFinished() {
        if (m_future.isCanceled()) {
            emit cancelled();
            return;
        }
        
        try {
            T result = m_future.result();
            if (m_completionCallback) {
                m_completionCallback(result);
            }
            emit completed(result);
        } catch (const std::exception& ex) {
            QString errorMsg = QString("Task failed: %1").arg(ex.what());
            if (m_errorCallback) {
                m_errorCallback(errorMsg);
            }
            emit error(errorMsg);
        }
    }
    
    void onProgressChanged(int progress) {
        if (m_progressCallback) {
            m_progressCallback(progress);
        }
        emit progressChanged(progress);
    }
    
private:
    QFuture<T> m_future;
    std::unique_ptr<QFutureWatcher<T>> m_watcher;
    ProgressCallback m_progressCallback;
    CompletionCallback m_completionCallback;
    ErrorCallback m_errorCallback;
};

class AsyncTaskManager : public QObject {
    Q_OBJECT
    
public:
    static AsyncTaskManager& instance();
    
    template<typename T>
    std::shared_ptr<AsyncTask<T>> createTask() {
        auto task = std::make_shared<AsyncTask<T>>(this);
        return task;
    }
    
    // Convenience method for simple tasks with progress dialog
    template<typename T>
    void runWithProgressDialog(
        typename AsyncTask<T>::TaskFunction task,
        const QString& title,
        QWidget* parent = nullptr,
        typename AsyncTask<T>::CompletionCallback onComplete = nullptr,
        typename AsyncTask<T>::ErrorCallback onError = nullptr
    ) {
        auto asyncTask = createTask<T>();
        
        // Create progress dialog
        auto progressDialog = std::make_unique<QProgressDialog>(title, "Cancel", 0, 100, parent);
        progressDialog->setWindowModality(Qt::WindowModal);
        progressDialog->setMinimumDuration(500); // Show after 500ms
        
        // Connect progress
        connect(asyncTask.get(), &AsyncTask<T>::progressChanged, 
                progressDialog.get(), &QProgressDialog::setValue);
        
        // Connect cancel
        connect(progressDialog.get(), &QProgressDialog::canceled,
                asyncTask.get(), &AsyncTask<T>::cancel);
        
        // Connect completion
        connect(asyncTask.get(), &AsyncTask<T>::completed, 
                [progressDialog = std::move(progressDialog), onComplete](const T& result) {
                    progressDialog->close();
                    if (onComplete) {
                        onComplete(result);
                    }
                });
        
        // Connect error
        connect(asyncTask.get(), &AsyncTask<T>::error,
                [progressDialog = progressDialog.get(), onError](const QString& error) {
                    progressDialog->close();
                    if (onError) {
                        onError(error);
                    }
                });
        
        // Connect cancelled
        connect(asyncTask.get(), &AsyncTask<T>::cancelled,
                [progressDialog = progressDialog.get()]() {
                    progressDialog->close();
                });
        
        asyncTask->run(task);
        
        // Keep the task alive
        m_activeTasks.push_back(std::static_pointer_cast<QObject>(asyncTask));
        
        // Clean up completed tasks
        connect(asyncTask.get(), &AsyncTask<T>::completed,
                this, [this, asyncTask]() { removeTask(asyncTask); });
        connect(asyncTask.get(), &AsyncTask<T>::error,
                this, [this, asyncTask]() { removeTask(asyncTask); });
        connect(asyncTask.get(), &AsyncTask<T>::cancelled,
                this, [this, asyncTask]() { removeTask(asyncTask); });
    }
    
    void cancelAllTasks();
    int activeTaskCount() const { return m_activeTasks.size(); }
    
signals:
    void allTasksCompleted();
    
private:
    AsyncTaskManager(QObject* parent = nullptr) : QObject(parent) {}
    
    void removeTask(std::shared_ptr<QObject> task);
    
    std::vector<std::shared_ptr<QObject>> m_activeTasks;
};

#include "AsyncTaskManager.moc"
