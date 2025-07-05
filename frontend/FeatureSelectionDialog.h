#pragma once
#include <QDialog>
#include <QCheckBox>
#include <QVBoxLayout>
#include <QDialogButtonBox>
#include <QMap>
#include <QString>
#include <QSet>

class FeatureSelectionDialog : public QDialog {
    Q_OBJECT
public:
    FeatureSelectionDialog(QWidget* parent = nullptr);
    QSet<QString> selectedFeatures() const;
private:
    QMap<QString, QCheckBox*> featureCheckboxes;
};
