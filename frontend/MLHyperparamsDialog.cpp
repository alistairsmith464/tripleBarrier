#include "MLHyperparamsDialog.h"

MLHyperparamsDialog::MLHyperparamsDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle("XGBoost Hyperparameters");
    QFormLayout* layout = new QFormLayout(this);
    m_nRounds = new QSpinBox(this);
    m_nRounds->setRange(1, 1000);
    m_nRounds->setValue(20);
    m_maxDepth = new QSpinBox(this);
    m_maxDepth->setRange(1, 20);
    m_maxDepth->setValue(3);
    m_nThread = new QSpinBox(this);
    m_nThread->setRange(1, 32);
    m_nThread->setValue(4);
    
    layout->addRow("Boosting Rounds", m_nRounds);
    layout->addRow("Max Depth", m_maxDepth);
    layout->addRow("Threads", m_nThread);
    
    QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layout->addWidget(buttons);
}
int MLHyperparamsDialog::nRounds() const { return m_nRounds->value(); }
int MLHyperparamsDialog::maxDepth() const { return m_maxDepth->value(); }
int MLHyperparamsDialog::nThread() const { return m_nThread->value(); }
