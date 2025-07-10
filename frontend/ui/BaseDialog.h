#pragma once
#include <QDialog>
#include <QFormLayout>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDialogButtonBox>
#include <QLabel>
#include <QWidget>
#include <QPushButton>
#include <QSize>
#include "../utils/InputValidator.h"
#include <functional>

class BaseDialog : public QDialog {
    Q_OBJECT
    
public:
    explicit BaseDialog(QWidget* parent = nullptr, const QString& title = "");
    virtual ~BaseDialog() = default;
    
protected:
    // Layout management
    void addFormRow(const QString& label, QWidget* widget, const QString& tooltip = "");
    void addFormRow(QWidget* labelWidget, QWidget* widget, const QString& tooltip = "");
    void addSection(const QString& sectionTitle);
    void addSpacer(int height = 10);
    void addWidget(QWidget* widget);
    
    // Button management
    void setupStandardButtons(QDialogButtonBox::StandardButtons buttons = 
                             QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    void addCustomButton(const QString& text, std::function<void()> callback);
    QPushButton* getButton(QDialogButtonBox::StandardButton button);
    
    // Validation
    void addValidator(std::function<ValidationResult()> validator);
    virtual ValidationResult validateInput();
    
    // Error display
    void showValidationError(const ValidationResult& result);
    void clearValidationError();
    
    // Status display
    void setStatusMessage(const QString& message, bool isError = false);
    void clearStatusMessage();
    
    // Configuration
    void setFixedSize(bool fixed) { m_fixedSize = fixed; }
    void setMinimumDialogSize(const QSize& size);
    
protected slots:
    virtual void accept() override;
    virtual void onAccept();
    virtual void onReject();
    
private:
    void setupLayout();
    void styleErrorLabel();
    
protected:
    QVBoxLayout* m_mainLayout;
    QFormLayout* m_formLayout;
    QDialogButtonBox* m_buttonBox;
    QLabel* m_errorLabel;
    QLabel* m_statusLabel;
    
private:
    std::vector<std::function<ValidationResult()>> m_validators;
    bool m_fixedSize;
};
