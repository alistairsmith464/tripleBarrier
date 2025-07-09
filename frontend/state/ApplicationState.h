#pragma once

#include <vector>
#include <QString>

// Forward declarations
struct PreprocessedRow;
struct LabeledEvent;

// Application state management
class ApplicationState {
public:
    // Data state
    void setData(const std::vector<PreprocessedRow>& rows, const std::vector<LabeledEvent>& events);
    
    const std::vector<PreprocessedRow>& getRows() const { return m_rows; }
    const std::vector<LabeledEvent>& getEvents() const { return m_events; }
    
    bool hasData() const { return !m_rows.empty() && !m_events.empty(); }
    void clearData();
    
    // UI state
    void setStatusMessage(const QString& message) { m_statusMessage = message; }
    const QString& getStatusMessage() const { return m_statusMessage; }
    
    // Singleton access
    static ApplicationState& instance();
    
private:
    ApplicationState() = default;
    
    std::vector<PreprocessedRow> m_rows;
    std::vector<LabeledEvent> m_events;
    QString m_statusMessage;
};
