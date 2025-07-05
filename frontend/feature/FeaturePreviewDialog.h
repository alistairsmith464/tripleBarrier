#pragma once
#include <QDialog>
#include <vector>
#include <map>
#include <QString>
#include <QSet>
#include "../backend/data/PreprocessedRow.h"
#include "../backend/data/LabeledEvent.h"

class FeaturePreviewDialog : public QDialog {
    Q_OBJECT
public:
    FeaturePreviewDialog(const QSet<QString>& selectedFeatures,
                        const std::vector<PreprocessedRow>& rows,
                        const std::vector<LabeledEvent>& labeledEvents,
                        QWidget* parent = nullptr);
};
