#pragma once

#include <vector>
#include <QString>

struct PreprocessedRow;
struct LabeledEvent;

class ApplicationState {
public:
    void setData(const std::vector<PreprocessedRow>& rows, const std::vector<LabeledEvent>& events);
    
    const std::vector<PreprocessedRow>& getRows() const { return m_rows; }
    const std::vector<LabeledEvent>& getEvents() const { return m_events; }
    
    bool hasData() const { return !m_rows.empty() && !m_events.empty(); }
    void clearData();
    
    void setStatusMessage(const QString& message) { m_statusMessage = message; }
    const QString& getStatusMessage() const { return m_statusMessage; }
    
    static ApplicationState& instance();
    
private:
    ApplicationState() = default;
    
    std::vector<PreprocessedRow> m_rows;
    std::vector<LabeledEvent> m_events;
    QString m_statusMessage;
};
