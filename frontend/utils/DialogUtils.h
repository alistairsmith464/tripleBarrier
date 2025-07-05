#pragma once
#include <QString>
#include <QWidget>

namespace DialogUtils {
    void showInfo(QWidget* parent, const QString& title, const QString& message);
    void showWarning(QWidget* parent, const QString& title, const QString& message);
    void showError(QWidget* parent, const QString& title, const QString& message);
}
