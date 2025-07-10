#include "FileDialogUtils.h"
#include "../ui/UIStrings.h"
#include <QFileDialog>

namespace FileDialogUtils {
QString getOpenCSVFile(QWidget* parent) {
    return QFileDialog::getOpenFileName(
        parent,
        UIStrings::SELECT_CSV_FILE,
        "",
        QString("%1;;%2").arg(UIStrings::CSV_FILTER, UIStrings::ALL_FILES_FILTER)
    );
}
QString getSaveCSVFile(QWidget* parent, const QString& defaultName) {
    return QFileDialog::getSaveFileName(
        parent,
        UIStrings::EXPORT_CSV_TITLE,
        defaultName,
        QString("%1;;%2").arg(UIStrings::CSV_FILTER, UIStrings::ALL_FILES_FILTER)
    );
}
}
