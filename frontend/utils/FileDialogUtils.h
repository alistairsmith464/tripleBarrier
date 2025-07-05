#pragma once
#include <QString>
#include <QWidget>

namespace FileDialogUtils {
    QString getOpenCSVFile(QWidget* parent);
    QString getSaveCSVFile(QWidget* parent, const QString& defaultName = "labeled_output.csv");
}
