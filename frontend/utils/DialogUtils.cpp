#include "DialogUtils.h"
#include <QMessageBox>

namespace DialogUtils {
void showInfo(QWidget* parent, const QString& title, const QString& message) {
    QMessageBox::information(parent, title, message);
}
void showWarning(QWidget* parent, const QString& title, const QString& message) {
    QMessageBox::warning(parent, title, message);
}
void showError(QWidget* parent, const QString& title, const QString& message) {
    QMessageBox::critical(parent, title, message);
}
}
