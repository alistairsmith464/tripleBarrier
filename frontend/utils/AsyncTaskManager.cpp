#include "AsyncTaskManager.h"
#include <algorithm>

AsyncTaskManager& AsyncTaskManager::instance() {
    static AsyncTaskManager instance;
    return instance;
}

void AsyncTaskManager::cancelAllTasks() {
    for (auto& taskObj : m_activeTasks) {
        if (auto task = std::dynamic_pointer_cast<AsyncTaskBase>(taskObj)) {
            task->cancel();
        }
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
