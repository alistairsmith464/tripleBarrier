#pragma once
#include <QObject>
#include <QTimer>
#include <QProgressDialog>
#include <QVariant>
#include <QMetaObject>
#include <functional>
#include <memory>
#include <thread>
#include <future>
#include <atomic>
#include <chrono>

class AsyncTaskBase : public QObject {
    Q_OBJECT
    
public:
    AsyncTaskBase(QObject* parent = nullptr) : QObject(parent) {}
    virtual ~AsyncTaskBase() = default;
    
public slots:
    virtual void cancel() = 0;
    
signals:
    void progressChanged(int progress);
    void cancelled();
    void completed(const QVariant& result);
    void error(const QString& message);
    
protected slots:
    void onProgressChanged(int progress) {
        emit progressChanged(progress);
    }
};

template<typename T>
class AsyncTask : public AsyncTaskBase {
public:
    using TaskFunction = std::function<T()>;
    using ProgressCallback = std::function<void(int)>;
    using CompletionCallback = std::function<void(const T&)>;
    using ErrorCallback = std::function<void(const QString&)>;
    
    AsyncTask(QObject* parent = nullptr) : AsyncTaskBase(parent), m_running(false) {}
    
    void run(TaskFunction task) {
        if (m_running) return;
        
        m_running = true;
        m_future = std::async(std::launch::async, [this, task]() {
            try {
                T result = task();
                QMetaObject::invokeMethod(this, [this, result]() {
                    m_running = false;
                    emit completed(QVariant::fromValue(result));
                    if (m_completionCallback) {
                        m_completionCallback(result);
                    }
                }, Qt::QueuedConnection);
                return result;
            } catch (const std::exception& e) {
                QMetaObject::invokeMethod(this, [this, e]() {
                    m_running = false;
                    QString errorMsg = QString::fromStdString(e.what());
                    emit error(errorMsg);
                    if (m_errorCallback) {
                        m_errorCallback(errorMsg);
                    }
                }, Qt::QueuedConnection);
                throw;
            }
        });
    }
    
    void setProgressCallback(ProgressCallback callback) { m_progressCallback = callback; }
    void setCompletionCallback(CompletionCallback callback) { m_completionCallback = callback; }
    void setErrorCallback(ErrorCallback callback) { m_errorCallback = callback; }
    
    void cancel() override {
        m_running = false;
        emit cancelled();
    }
    
    bool isRunning() const { return m_running; }
    bool isFinished() const { return m_future.valid() && m_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }
    
private:
    std::future<T> m_future;
    std::atomic<bool> m_running;
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
        connect(asyncTask.get(), &AsyncTaskBase::progressChanged, 
                progressDialog.get(), &QProgressDialog::setValue);
        
        // Connect cancel
        connect(progressDialog.get(), &QProgressDialog::canceled,
                asyncTask.get(), &AsyncTaskBase::cancel);
        
        // Connect completion
        connect(asyncTask.get(), &AsyncTaskBase::completed, 
                [progressDialog = std::move(progressDialog), onComplete](const QVariant& variantResult) {
                    progressDialog->close();
                    if (onComplete) {
                        T result = variantResult.value<T>();
                        onComplete(result);
                    }
                });
        
        // Connect error
        connect(asyncTask.get(), &AsyncTaskBase::error,
                [progressDialog = progressDialog.get(), onError](const QString& error) {
                    progressDialog->close();
                    if (onError) {
                        onError(error);
                    }
                });
        
        // Connect cancelled
        connect(asyncTask.get(), &AsyncTaskBase::cancelled,
                [progressDialog = progressDialog.get()]() {
                    progressDialog->close();
                });
        
        asyncTask->run(task);
        
        // Keep the task alive
        m_activeTasks.push_back(std::static_pointer_cast<QObject>(asyncTask));
        
        // Clean up completed tasks
        connect(asyncTask.get(), &AsyncTaskBase::completed,
                this, [this, asyncTask]() { removeTask(asyncTask); });
        connect(asyncTask.get(), &AsyncTaskBase::error,
                this, [this, asyncTask]() { removeTask(asyncTask); });
        connect(asyncTask.get(), &AsyncTaskBase::cancelled,
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
