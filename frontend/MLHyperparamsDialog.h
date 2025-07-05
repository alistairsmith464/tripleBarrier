#pragma once
#include <QDialog>
#include <QSpinBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLineEdit>

class MLHyperparamsDialog : public QDialog {
    Q_OBJECT
public:
    MLHyperparamsDialog(QWidget* parent = nullptr);
    int nRounds() const;
    int maxDepth() const;
    int nThread() const;
    QString objective() const;
private:
    QSpinBox* m_nRounds;
    QSpinBox* m_maxDepth;
    QSpinBox* m_nThread;
    QComboBox* m_objective;
};
