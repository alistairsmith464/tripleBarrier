#pragma once

#include <QString>
#include <QFileInfo>

class FileHandler
{
public:
    FileHandler();
    ~FileHandler();

    // Upload a file to a designated directory
    bool uploadFile(const QString& sourceFilePath, const QString& destinationDir = "uploads");
    
    // Check if file exists
    bool fileExists(const QString& filePath) const;
    
    // Get file information
    QString getFileInfo(const QString& filePath) const;
    
    // Get the last uploaded file path
    QString getLastUploadedFile() const;

private:
    QString m_lastUploadedFile;
    void ensureUploadDirectory(const QString& directory);
};
