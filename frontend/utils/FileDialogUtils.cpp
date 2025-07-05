#include "FileDialogUtils.h"
#include <QFileDialog>

namespace FileDialogUtils {
QString getOpenCSVFile(QWidget* parent) {
    return QFileDialog::getOpenFileName(
        parent,
        "Select CSV File",
        "",
        "CSV Files (*.csv);;All Files (*.*)"
    );
}
QString getSaveCSVFile(QWidget* parent, const QString& defaultName) {
    return QFileDialog::getSaveFileName(
        parent,
        "Export Labeled Data to CSV",
        defaultName,
        "CSV Files (*.csv);;All Files (*.*)"
    );
}
}
