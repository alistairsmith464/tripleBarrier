#include "FileHandler.h"
#include <QFile>
#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>
#include <QDateTime>

FileHandler::FileHandler()
{
}

FileHandler::~FileHandler()
{
}

bool FileHandler::uploadFile(const QString& sourceFilePath, const QString& destinationDir)
{
    if (!QFile::exists(sourceFilePath)) {
        return false;
    }

    // Ensure upload directory exists
    ensureUploadDirectory(destinationDir);

    QFileInfo sourceInfo(sourceFilePath);
    QString fileName = sourceInfo.fileName();
    
    // Create destination path with timestamp to avoid conflicts
    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss_");
    QString destinationFilePath = destinationDir + "/" + timestamp + fileName;

    // Copy the file
    if (QFile::copy(sourceFilePath, destinationFilePath)) {
        m_lastUploadedFile = destinationFilePath;
        return true;
    }

    return false;
}

bool FileHandler::fileExists(const QString& filePath) const
{
    return QFile::exists(filePath);
}

QString FileHandler::getFileInfo(const QString& filePath) const
{
    if (!fileExists(filePath)) {
        return "File does not exist";
    }

    QFileInfo info(filePath);
    QString fileInfo = QString("File: %1\n")
                      .arg(info.fileName());
    fileInfo += QString("Size: %1 bytes\n")
               .arg(info.size());
    fileInfo += QString("Last Modified: %1\n")
               .arg(info.lastModified().toString());
    fileInfo += QString("Path: %1")
               .arg(info.absoluteFilePath());

    return fileInfo;
}

QString FileHandler::getLastUploadedFile() const
{
    return m_lastUploadedFile;
}

void FileHandler::ensureUploadDirectory(const QString& directory)
{
    QDir dir;
    if (!dir.exists(directory)) {
        dir.mkpath(directory);
    }
}
