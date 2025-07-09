#include "ApplicationState.h"
#include "../backend/data/PreprocessedRow.h"
#include "../backend/data/LabeledEvent.h"

ApplicationState& ApplicationState::instance() {
    static ApplicationState instance;
    return instance;
}

void ApplicationState::setData(const std::vector<PreprocessedRow>& rows, const std::vector<LabeledEvent>& events) {
    m_rows = rows;
    m_events = events;
}

void ApplicationState::clearData() {
    m_rows.clear();
    m_events.clear();
    m_statusMessage.clear();
}
