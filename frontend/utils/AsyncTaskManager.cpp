#include "AsyncTaskManager.h"
#include <algorithm>

AsyncTaskManager& AsyncTaskManager::instance() {
    static AsyncTaskManager instance;
    return instance;
}

void AsyncTaskManager::cancelAllTasks() {
    for (auto& taskObj : m_activeTasks) {
        // Try to cast to different task types and cancel
        if (auto task = std::dynamic_pointer_cast<AsyncTask<int>>(taskObj)) {
            task->cancel();
        } else if (auto task = std::dynamic_pointer_cast<AsyncTask<double>>(taskObj)) {
            task->cancel();
        } else if (auto task = std::dynamic_pointer_cast<AsyncTask<QString>>(taskObj)) {
            task->cancel();
        }
        // Add more types as needed
    }
    
    m_activeTasks.clear();
    emit allTasksCompleted();
}

void AsyncTaskManager::removeTask(std::shared_ptr<QObject> task) {
    auto it = std::find(m_activeTasks.begin(), m_activeTasks.end(), task);
    if (it != m_activeTasks.end()) {
        m_activeTasks.erase(it);
        
        if (m_activeTasks.empty()) {
            emit allTasksCompleted();
        }
    }
}
