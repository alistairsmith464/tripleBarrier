#pragma once
#include <QString>

namespace UIStrings {
    constexpr const char* APP_TITLE = "Triple Barrier Analysis Tool";
    constexpr const char* WINDOW_TITLE = "Triple Barrier - File Upload";
    
    constexpr const char* LOAD_SUCCESS = "✓ Data loaded successfully!";
    constexpr const char* LOAD_FAILED = "✗ Data load failed";
    constexpr const char* PROCESSING = "Processing...";
    constexpr const char* READY = "Ready";
    constexpr const char* LOADING_CSV = "Loading CSV data...";
    constexpr const char* LABELING_CANCELLED = "Labeling cancelled.";
    
    constexpr const char* UPLOAD_DATA = "Upload Data";
    constexpr const char* CLEAR = "Clear";
    constexpr const char* RUN_ML = "Run ML Pipeline";
    constexpr const char* BROWSE = "Browse...";
    constexpr const char* CANCEL = "Cancel";
    constexpr const char* OK = "OK";
    constexpr const char* APPLY = "Apply";
    constexpr const char* RESET = "Reset";
    
    constexpr const char* BARRIER_CONFIG_TITLE = "Barrier Configuration";
    constexpr const char* FEATURE_SELECTION_TITLE = "Feature Selection";
    constexpr const char* FEATURE_PREVIEW_TITLE = "Feature Preview";
    constexpr const char* ML_HYPERPARAMS_TITLE = "ML Hyperparameters";
    constexpr const char* ERROR_TITLE = "Error";
    constexpr const char* WARNING_TITLE = "Warning";
    constexpr const char* INFO_TITLE = "Information";
    
    constexpr const char* PROFIT_MULTIPLE = "Profit Multiple:";
    constexpr const char* STOP_MULTIPLE = "Stop Multiple:";
    constexpr const char* VERTICAL_WINDOW = "Vertical Window:";
    constexpr const char* LABELING_TYPE = "Labeling Type:";
    constexpr const char* PLOT_MODE = "Plot Mode:";
    constexpr const char* TEST_SIZE = "Test Size:";
    constexpr const char* VALIDATION_SIZE = "Validation Size:";
    constexpr const char* LEARNING_RATE = "Learning Rate:";
    constexpr const char* MAX_DEPTH = "Max Depth:";
    constexpr const char* N_ROUNDS = "Number of Rounds:";
    constexpr const char* THREADS = "Threads";
    constexpr const char* CUSUM_THRESHOLD = "CUSUM Threshold:";
    constexpr const char* VOLATILITY_WINDOW = "Volatility Window:";
    constexpr const char* TTBM_DECAY_TYPE = "TTBM Decay Type:";
    constexpr const char* USE_CUSUM = "Use CUSUM Event Detection";
    constexpr const char* AUTO_TUNE_HYPERPARAMS = "Auto-tune hyperparameters (grid search)";
    
    constexpr const char* HARD_BARRIER = "Hard Barrier";
    constexpr const char* TTBM_BARRIER = "TTBM (Time-to-Barrier Modification)";
    constexpr const char* TIME_SERIES = "Time Series";
    constexpr const char* HISTOGRAM = "Histogram";
    constexpr const char* TTBM_TIME_SERIES = "TTBM Time Series";
    constexpr const char* TTBM_DISTRIBUTION = "TTBM Distribution";
    
    constexpr const char* PROFIT_TOOLTIP = "Multiplier for the profit-taking barrier (e.g., 2.0 = 2x volatility above entry).";
    constexpr const char* STOP_TOOLTIP = "Multiplier for the stop-loss barrier (e.g., 1.0 = 1x volatility below entry).";
    constexpr const char* VERTICAL_TOOLTIP = "Maximum holding period in bars (time steps) before exit.";
    constexpr const char* CUSUM_TOOLTIP = "Enable CUSUM event detection for volatility-based event filtering.";
    constexpr const char* CUSUM_THRESHOLD_TOOLTIP = "Sensitivity for CUSUM filter (higher = fewer events).";
    constexpr const char* VOLATILITY_WINDOW_TOOLTIP = "Window size for volatility calculation (e.g., 20 = 20 bars).";
    
    constexpr const char* NO_DATA_ERROR = "No labeled events available. Please upload and label data first.";
    constexpr const char* NO_FEATURES_ERROR = "No features selected. Please select at least one feature.";
    constexpr const char* INVALID_CONFIG_ERROR = "Invalid configuration. Please check your settings.";
    constexpr const char* FILE_NOT_FOUND_ERROR = "File not found. Please select a valid file.";
    constexpr const char* PROCESSING_ERROR = "An error occurred during processing.";
    constexpr const char* INVALID_FILE_SELECTION = "Invalid file selection";
    
    constexpr const char* CONFIG_SAVED = "Configuration saved successfully.";
    constexpr const char* ML_COMPLETED = "Machine learning analysis completed.";
    constexpr const char* DATA_PROCESSED = "Data processed successfully.";
    constexpr const char* CONFIG_VALIDATED = "Configuration validated successfully.";
    
    constexpr const char* SELECT_CSV_FILE = "Select CSV File";
    constexpr const char* EXPORT_CSV_TITLE = "Export CSV File";
    constexpr const char* CSV_FILTER = "CSV Files (*.csv)";
    constexpr const char* ALL_FILES_FILTER = "All Files (*)";
    constexpr const char* SAVE_MODEL = "Save Model";
    constexpr const char* LOAD_MODEL = "Load Model";
    constexpr const char* MODEL_FILTER = "Model Files (*.model)";
    
    constexpr const char* EXTRACTING_FEATURES = "Extracting features...";
    constexpr const char* TRAINING_MODEL = "Training model...";
    constexpr const char* EVALUATING_MODEL = "Evaluating model...";
    constexpr const char* RUNNING_SIMULATION = "Running portfolio simulation...";
    constexpr const char* TUNING_HYPERPARAMS = "Tuning hyperparameters...";
    
    constexpr const char* LABELING_CONFIG_SECTION = "Labeling Configuration";
    constexpr const char* BARRIER_PARAMS_SECTION = "Barrier Parameters";
    constexpr const char* EVENT_DETECTION_SECTION = "Event Detection";
    constexpr const char* TTBM_CONFIG_SECTION = "TTBM Configuration";
    constexpr const char* VOLATILITY_CALC_SECTION = "Volatility Calculation";
    
    constexpr const char* EXPONENTIAL_DECAY = "Exponential Decay (Fixed Parameters)";
    constexpr const char* LINEAR_DECAY = "Linear Decay (Fixed Parameters)";
    constexpr const char* HYPERBOLIC_DECAY = "Hyperbolic Decay (Fixed Parameters)";
    
    constexpr const char* FAILED_PROCESS_CSV = "Failed to process CSV file";
    constexpr const char* CHECK_FILE_FORMAT = "Please check the file format and try again";
}

class UIStringHelper {
public:
    static QString uploadSuccessMessage(const QString& filePath);
    static QString loadErrorMessage(const QString& error);
    static QString featureCountMessage(int count);
    static QString dataPointsMessage(int count);
    static QString accuracyMessage(double accuracy);
    static QString processingTimeMessage(int seconds);
};
