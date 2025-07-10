#include "MLHyperparamsDialog.h"
#include "ui/UIStrings.h"
#include <QPushButton>

MLHyperparamsDialog::MLHyperparamsDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(UIStrings::ML_HYPERPARAMS_TITLE);
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
    
    layout->addRow(UIStrings::N_ROUNDS, m_nRounds);
    layout->addRow(UIStrings::MAX_DEPTH, m_maxDepth);
    layout->addRow(UIStrings::THREADS, m_nThread);
    
    QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    buttons->button(QDialogButtonBox::Ok)->setText(UIStrings::OK);
    buttons->button(QDialogButtonBox::Cancel)->setText(UIStrings::CANCEL);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layout->addWidget(buttons);
}
int MLHyperparamsDialog::nRounds() const { return m_nRounds->value(); }
int MLHyperparamsDialog::maxDepth() const { return m_maxDepth->value(); }
int MLHyperparamsDialog::nThread() const { return m_nThread->value(); }
