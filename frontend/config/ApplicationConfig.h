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
    
    // UI Configuration
    QSize defaultWindowSize() const { return m_defaultWindowSize; }
    void setDefaultWindowSize(const QSize& size);
    
    QString defaultDataPath() const { return m_defaultDataPath; }
    void setDefaultDataPath(const QString& path);
    
    QString lastUsedDataPath() const { return m_lastUsedDataPath; }
    void setLastUsedDataPath(const QString& path);
    
    // ML Configuration
    MLPipeline::UnifiedPipelineConfig defaultMLConfig() const { return m_defaultMLConfig; }
    void setDefaultMLConfig(const MLPipeline::UnifiedPipelineConfig& config);
    
    BarrierConfig defaultBarrierConfig() const { return m_defaultBarrierConfig; }
    void setDefaultBarrierConfig(const BarrierConfig& config);
    
    // Application Settings
    bool enableLogging() const { return m_enableLogging; }
    void setEnableLogging(bool enable);
    
    QString logFilePath() const { return m_logFilePath; }
    void setLogFilePath(const QString& path);
    
    int maxRecentFiles() const { return m_maxRecentFiles; }
    void setMaxRecentFiles(int count);
    
    QStringList recentFiles() const { return m_recentFiles; }
    void addRecentFile(const QString& filePath);
    void clearRecentFiles();
    
    // Performance Settings
    int workerThreadCount() const { return m_workerThreadCount; }
    void setWorkerThreadCount(int count);
    
    bool enableCaching() const { return m_enableCaching; }
    void setEnableCaching(bool enable);
    
    int maxCacheSize() const { return m_maxCacheSize; } // in MB
    void setMaxCacheSize(int sizeInMB);
    
    // Persistence
    void save();
    void load();
    void reset(); // Reset to defaults
    
private:
    ApplicationConfig();
    ~ApplicationConfig() = default;
    
    // Prevent copying
    ApplicationConfig(const ApplicationConfig&) = delete;
    ApplicationConfig& operator=(const ApplicationConfig&) = delete;
    
    void initializeDefaults();
    void loadSettings();
    void saveSettings();
    
    // UI Settings
    QSize m_defaultWindowSize;
    QString m_defaultDataPath;
    QString m_lastUsedDataPath;
    
    // ML Settings  
    MLPipeline::UnifiedPipelineConfig m_defaultMLConfig;
    BarrierConfig m_defaultBarrierConfig;
    
    // Application Settings
    bool m_enableLogging;
    QString m_logFilePath;
    int m_maxRecentFiles;
    QStringList m_recentFiles;
    
    // Performance Settings
    int m_workerThreadCount;
    bool m_enableCaching;
    int m_maxCacheSize;
    
    std::unique_ptr<QSettings> m_settings;
};
