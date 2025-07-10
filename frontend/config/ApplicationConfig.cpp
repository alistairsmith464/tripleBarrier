#include "ApplicationConfig.h"
#include <QStandardPaths>
#include <QDir>
#include <QCoreApplication>
#include <QThread>

ApplicationConfig& ApplicationConfig::instance() {
    static ApplicationConfig instance;
    return instance;
}

ApplicationConfig::ApplicationConfig() {
    QString configPath = QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation);
    QDir().mkpath(configPath);
    
    m_settings = std::make_unique<QSettings>(
        configPath + "/triplebarrier.ini", 
        QSettings::IniFormat
    );
    
    initializeDefaults();
    load();
}

void ApplicationConfig::initializeDefaults() {
    m_defaultWindowSize = QSize(1200, 800);
    m_defaultDataPath = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    m_lastUsedDataPath = m_defaultDataPath;
    
    // Initialize default ML config
    m_defaultMLConfig = MLPipeline::UnifiedPipelineConfig{};
    
    // Initialize default barrier config
    m_defaultBarrierConfig = BarrierConfig{};
    m_defaultBarrierConfig.profit_multiple = 2.0;
    m_defaultBarrierConfig.stop_multiple = 1.0;
    m_defaultBarrierConfig.vertical_window = 20;
    
    m_enableLogging = true;
    m_logFilePath = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + "/logs/app.log";
    m_maxRecentFiles = 10;
    
    m_workerThreadCount = std::max(1, static_cast<int>(QThread::idealThreadCount()) - 1);
    m_enableCaching = true;
    m_maxCacheSize = 500; // 500 MB
}

void ApplicationConfig::setDefaultWindowSize(const QSize& size) {
    m_defaultWindowSize = size;
}

void ApplicationConfig::setDefaultDataPath(const QString& path) {
    m_defaultDataPath = path;
}

void ApplicationConfig::setLastUsedDataPath(const QString& path) {
    m_lastUsedDataPath = path;
}

void ApplicationConfig::setDefaultMLConfig(const MLPipeline::UnifiedPipelineConfig& config) {
    m_defaultMLConfig = config;
}

void ApplicationConfig::setDefaultBarrierConfig(const BarrierConfig& config) {
    m_defaultBarrierConfig = config;
}

void ApplicationConfig::setEnableLogging(bool enable) {
    m_enableLogging = enable;
}

void ApplicationConfig::setLogFilePath(const QString& path) {
    m_logFilePath = path;
}

void ApplicationConfig::setMaxRecentFiles(int count) {
    m_maxRecentFiles = std::max(0, count);
}

void ApplicationConfig::addRecentFile(const QString& filePath) {
    m_recentFiles.removeAll(filePath); // Remove if already exists
    m_recentFiles.prepend(filePath);
    
    // Limit to max recent files
    while (m_recentFiles.size() > m_maxRecentFiles) {
        m_recentFiles.removeLast();
    }
}

void ApplicationConfig::clearRecentFiles() {
    m_recentFiles.clear();
}

void ApplicationConfig::setWorkerThreadCount(int count) {
    m_workerThreadCount = std::max(1, count);
}

void ApplicationConfig::setEnableCaching(bool enable) {
    m_enableCaching = enable;
}

void ApplicationConfig::setMaxCacheSize(int sizeInMB) {
    m_maxCacheSize = std::max(0, sizeInMB);
}

void ApplicationConfig::save() {
    saveSettings();
}

void ApplicationConfig::load() {
    loadSettings();
}

void ApplicationConfig::reset() {
    m_settings->clear();
    initializeDefaults();
}

void ApplicationConfig::loadSettings() {
    m_settings->beginGroup("UI");
    m_defaultWindowSize = m_settings->value("defaultWindowSize", m_defaultWindowSize).toSize();
    m_defaultDataPath = m_settings->value("defaultDataPath", m_defaultDataPath).toString();
    m_lastUsedDataPath = m_settings->value("lastUsedDataPath", m_lastUsedDataPath).toString();
    m_settings->endGroup();
    
    m_settings->beginGroup("ML");
    // Load ML config values
    m_defaultMLConfig.test_size = m_settings->value("testSize", m_defaultMLConfig.test_size).toDouble();
    m_defaultMLConfig.val_size = m_settings->value("valSize", m_defaultMLConfig.val_size).toDouble();
    m_defaultMLConfig.n_rounds = m_settings->value("nRounds", m_defaultMLConfig.n_rounds).toInt();
    m_defaultMLConfig.max_depth = m_settings->value("maxDepth", m_defaultMLConfig.max_depth).toInt();
    m_defaultMLConfig.learning_rate = m_settings->value("learningRate", m_defaultMLConfig.learning_rate).toDouble();
    m_settings->endGroup();
    
    m_settings->beginGroup("Barrier");
    m_defaultBarrierConfig.profit_multiple = m_settings->value("profitMultiple", m_defaultBarrierConfig.profit_multiple).toDouble();
    m_defaultBarrierConfig.stop_multiple = m_settings->value("stopMultiple", m_defaultBarrierConfig.stop_multiple).toDouble();
    m_defaultBarrierConfig.vertical_window = m_settings->value("verticalWindow", m_defaultBarrierConfig.vertical_window).toInt();
    m_settings->endGroup();
    
    m_settings->beginGroup("Application");
    m_enableLogging = m_settings->value("enableLogging", m_enableLogging).toBool();
    m_logFilePath = m_settings->value("logFilePath", m_logFilePath).toString();
    m_maxRecentFiles = m_settings->value("maxRecentFiles", m_maxRecentFiles).toInt();
    m_recentFiles = m_settings->value("recentFiles", m_recentFiles).toStringList();
    m_settings->endGroup();
    
    m_settings->beginGroup("Performance");
    m_workerThreadCount = m_settings->value("workerThreadCount", m_workerThreadCount).toInt();
    m_enableCaching = m_settings->value("enableCaching", m_enableCaching).toBool();
    m_maxCacheSize = m_settings->value("maxCacheSize", m_maxCacheSize).toInt();
    m_settings->endGroup();
}

void ApplicationConfig::saveSettings() {
    m_settings->beginGroup("UI");
    m_settings->setValue("defaultWindowSize", m_defaultWindowSize);
    m_settings->setValue("defaultDataPath", m_defaultDataPath);
    m_settings->setValue("lastUsedDataPath", m_lastUsedDataPath);
    m_settings->endGroup();
    
    m_settings->beginGroup("ML");
    m_settings->setValue("testSize", m_defaultMLConfig.test_size);
    m_settings->setValue("valSize", m_defaultMLConfig.val_size);
    m_settings->setValue("nRounds", m_defaultMLConfig.n_rounds);
    m_settings->setValue("maxDepth", m_defaultMLConfig.max_depth);
    m_settings->setValue("learningRate", m_defaultMLConfig.learning_rate);
    m_settings->endGroup();
    
    m_settings->beginGroup("Barrier");
    m_settings->setValue("profitMultiple", m_defaultBarrierConfig.profit_multiple);
    m_settings->setValue("stopMultiple", m_defaultBarrierConfig.stop_multiple);
    m_settings->setValue("verticalWindow", m_defaultBarrierConfig.vertical_window);
    m_settings->endGroup();
    
    m_settings->beginGroup("Application");
    m_settings->setValue("enableLogging", m_enableLogging);
    m_settings->setValue("logFilePath", m_logFilePath);
    m_settings->setValue("maxRecentFiles", m_maxRecentFiles);
    m_settings->setValue("recentFiles", m_recentFiles);
    m_settings->endGroup();
    
    m_settings->beginGroup("Performance");
    m_settings->setValue("workerThreadCount", m_workerThreadCount);
    m_settings->setValue("enableCaching", m_enableCaching);
    m_settings->setValue("maxCacheSize", m_maxCacheSize);
    m_settings->endGroup();
    
    m_settings->sync();
}
