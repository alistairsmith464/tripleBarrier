#pragma once
#include <QSize>
#include <QString>
#include <QSettings>
#include <memory>
#include "../../backend/ml/MLPipeline.h"
#include "../../backend/data/BarrierConfig.h"

class ApplicationConfig {
public:
    static ApplicationConfig& instance();
    
    QSize defaultWindowSize() const { return m_defaultWindowSize; }
    void setDefaultWindowSize(const QSize& size);
    
    QString defaultDataPath() const { return m_defaultDataPath; }
    void setDefaultDataPath(const QString& path);
    
    QString lastUsedDataPath() const { return m_lastUsedDataPath; }
    void setLastUsedDataPath(const QString& path);
    
    MLPipeline::UnifiedPipelineConfig defaultMLConfig() const { return m_defaultMLConfig; }
    void setDefaultMLConfig(const MLPipeline::UnifiedPipelineConfig& config);
    
    BarrierConfig defaultBarrierConfig() const { return m_defaultBarrierConfig; }
    void setDefaultBarrierConfig(const BarrierConfig& config);
    
    bool enableLogging() const { return m_enableLogging; }
    void setEnableLogging(bool enable);
    
    QString logFilePath() const { return m_logFilePath; }
    void setLogFilePath(const QString& path);
    
    int maxRecentFiles() const { return m_maxRecentFiles; }
    void setMaxRecentFiles(int count);
    
    QStringList recentFiles() const { return m_recentFiles; }
    void addRecentFile(const QString& filePath);
    void clearRecentFiles();
    
    int workerThreadCount() const { return m_workerThreadCount; }
    void setWorkerThreadCount(int count);
    
    bool enableCaching() const { return m_enableCaching; }
    void setEnableCaching(bool enable);
    
    int maxCacheSize() const { return m_maxCacheSize; }
    void setMaxCacheSize(int sizeInMB);
    
    void save();
    void load();
    void reset();
    
private:
    ApplicationConfig();
    ~ApplicationConfig() = default;
    
    ApplicationConfig(const ApplicationConfig&) = delete;
    ApplicationConfig& operator=(const ApplicationConfig&) = delete;
    
    void initializeDefaults();
    void loadSettings();
    void saveSettings();
    
    QSize m_defaultWindowSize;
    QString m_defaultDataPath;
    QString m_lastUsedDataPath;
    
    MLPipeline::UnifiedPipelineConfig m_defaultMLConfig;
    BarrierConfig m_defaultBarrierConfig;
    
    bool m_enableLogging;
    QString m_logFilePath;
    int m_maxRecentFiles;
    QStringList m_recentFiles;
    
    int m_workerThreadCount;
    bool m_enableCaching;
    int m_maxCacheSize;
    
    std::unique_ptr<QSettings> m_settings;
};
