#include "BaseDialog.h"
#include <QSpacerItem>
#include <QSizePolicy>
#include <QFont>

BaseDialog::BaseDialog(QWidget* parent, const QString& title)
    : QDialog(parent), m_fixedSize(false) {
    if (!title.isEmpty()) {
        setWindowTitle(title);
    }
    setupLayout();
}

void BaseDialog::setupLayout() {
    m_mainLayout = new QVBoxLayout(this);
    m_formLayout = new QFormLayout();
    m_mainLayout->addLayout(m_formLayout);
    
    // Status label for messages
    m_statusLabel = new QLabel(this);
    m_statusLabel->setWordWrap(true);
    m_statusLabel->hide();
    m_mainLayout->addWidget(m_statusLabel);
    
    // Error label for validation errors
    m_errorLabel = new QLabel(this);
    m_errorLabel->setWordWrap(true);
    styleErrorLabel();
    m_errorLabel->hide();
    m_mainLayout->addWidget(m_errorLabel);
    
    // Add spacer before buttons
    m_mainLayout->addStretch();
    
    setLayout(m_mainLayout);
}

void BaseDialog::styleErrorLabel() {
    m_errorLabel->setStyleSheet(
        "QLabel {"
        "  color: #e74c3c;"
        "  background-color: #fdf2f2;"
        "  border: 1px solid #f5c6cb;"
        "  border-radius: 4px;"
        "  padding: 8px;"
        "  margin: 4px 0px;"
        "  font-size: 12px;"
        "}"
    );
}

void BaseDialog::addFormRow(const QString& label, QWidget* widget, const QString& tooltip) {
    QLabel* labelWidget = new QLabel(label, this);
    addFormRow(labelWidget, widget, tooltip);
}

void BaseDialog::addFormRow(QWidget* labelWidget, QWidget* widget, const QString& tooltip) {
    m_formLayout->addRow(labelWidget, widget);
    
    if (!tooltip.isEmpty()) {
        labelWidget->setToolTip(tooltip);
        widget->setToolTip(tooltip);
    }
}

void BaseDialog::addSection(const QString& sectionTitle) {
    QLabel* sectionLabel = new QLabel(sectionTitle, this);
    QFont font = sectionLabel->font();
    font.setBold(true);
    font.setPointSize(font.pointSize() + 1);
    sectionLabel->setFont(font);
    sectionLabel->setStyleSheet("QLabel { margin-top: 10px; margin-bottom: 5px; }");
    
    m_formLayout->addRow(sectionLabel);
}

void BaseDialog::addSpacer(int height) {
    QSpacerItem* spacer = new QSpacerItem(20, height, QSizePolicy::Minimum, QSizePolicy::Fixed);
    m_formLayout->addItem(spacer);
}

void BaseDialog::addWidget(QWidget* widget) {
    m_formLayout->addRow(widget);
}

void BaseDialog::setupStandardButtons(QDialogButtonBox::StandardButtons buttons) {
    m_buttonBox = new QDialogButtonBox(buttons, this);
    m_mainLayout->addWidget(m_buttonBox);
    
    connect(m_buttonBox, &QDialogButtonBox::accepted, this, &BaseDialog::accept);
    connect(m_buttonBox, &QDialogButtonBox::rejected, this, &BaseDialog::onReject);
}

void BaseDialog::addCustomButton(const QString& text, std::function<void()> callback) {
    if (!m_buttonBox) {
        setupStandardButtons(QDialogButtonBox::NoButton);
    }
    
    QPushButton* button = m_buttonBox->addButton(text, QDialogButtonBox::ActionRole);
    connect(button, &QPushButton::clicked, [callback]() { callback(); });
}

QPushButton* BaseDialog::getButton(QDialogButtonBox::StandardButton button) {
    if (m_buttonBox) {
        return m_buttonBox->button(button);
    }
    return nullptr;
}

void BaseDialog::addValidator(std::function<ValidationResult()> validator) {
    m_validators.push_back(validator);
}

ValidationResult BaseDialog::validateInput() {
    for (const auto& validator : m_validators) {
        ValidationResult result = validator();
        if (!result.isValid) {
            return result;
        }
    }
    return ValidationResult::success();
}

void BaseDialog::showValidationError(const ValidationResult& result) {
    if (!result.isValid) {
        QString message = result.errorMessage;
        if (!result.suggestions.isEmpty()) {
            message += "\n\nSuggestions:\n• " + result.suggestions.join("\n• ");
        }
        m_errorLabel->setText(message);
        m_errorLabel->show();
    } else if (!result.warningMessage.isEmpty()) {
        QString message = result.warningMessage;
        if (!result.suggestions.isEmpty()) {
            message += "\n\nSuggestions:\n• " + result.suggestions.join("\n• ");
        }
        m_errorLabel->setText(message);
        m_errorLabel->setStyleSheet(
            "QLabel {"
            "  color: #856404;"
            "  background-color: #fff3cd;"
            "  border: 1px solid #ffeaa7;"
            "  border-radius: 4px;"
            "  padding: 8px;"
            "  margin: 4px 0px;"
            "  font-size: 12px;"
            "}"
        );
        m_errorLabel->show();
    } else {
        clearValidationError();
    }
}

void BaseDialog::clearValidationError() {
    m_errorLabel->hide();
    styleErrorLabel(); // Reset to error style
}

void BaseDialog::setStatusMessage(const QString& message, bool isError) {
    m_statusLabel->setText(message);
    
    if (isError) {
        m_statusLabel->setStyleSheet("color: #e74c3c; font-weight: bold;");
    } else {
        m_statusLabel->setStyleSheet("color: #27ae60; font-weight: bold;");
    }
    
    m_statusLabel->show();
}

void BaseDialog::clearStatusMessage() {
    m_statusLabel->hide();
}

void BaseDialog::setMinimumDialogSize(const QSize& size) {
    setMinimumSize(size);
    if (m_fixedSize) {
        setFixedSize(size);
    }
}

void BaseDialog::accept() {
    clearValidationError();
    ValidationResult result = validateInput();
    
    if (result.isValid) {
        onAccept();
        QDialog::accept();
    } else {
        showValidationError(result);
    }
}

void BaseDialog::onAccept() {
    // Override in derived classes
}

void BaseDialog::onReject() {
    QDialog::reject();
}
