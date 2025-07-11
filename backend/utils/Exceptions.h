#pragma once
#include <stdexcept>
#include <string>
#include <memory>

namespace TripleBarrier {

class BaseException : public std::exception {
public:
    explicit BaseException(const std::string& message, 
                          const std::string& context = "",
                          int error_code = 0)
        : message_(message), context_(context), error_code_(error_code) {}
    
    const char* what() const noexcept override {
        return message_.c_str();
    }
    
    const std::string& context() const noexcept { return context_; }
    int error_code() const noexcept { return error_code_; }
    
    std::string full_message() const {
        std::string full = message_;
        if (!context_.empty()) {
            full += " [Context: " + context_ + "]";
        }
        if (error_code_ != 0) {
            full += " [Error Code: " + std::to_string(error_code_) + "]";
        }
        return full;
    }

private:
    std::string message_;
    std::string context_;
    int error_code_;
};

class DataException : public BaseException {
public:
    explicit DataException(const std::string& message, 
                          const std::string& context = "",
                          int error_code = 1000)
        : BaseException(message, context, error_code) {}
};

class DataLoadException : public DataException {
public:
    explicit DataLoadException(const std::string& filename, 
                              const std::string& details = "")
        : DataException("Failed to load data from file: " + filename, details, 1001) {}
};

class DataValidationException : public DataException {
public:
    explicit DataValidationException(const std::string& message, 
                                   const std::string& field = "")
        : DataException("Data validation failed: " + message, field, 1002) {}
};

class DataProcessingException : public DataException {
public:
    explicit DataProcessingException(const std::string& message, 
                                   const std::string& step = "")
        : DataException("Data processing failed: " + message, step, 1003) {}
};

class MLException : public BaseException {
public:
    explicit MLException(const std::string& message, 
                        const std::string& context = "",
                        int error_code = 2000)
        : BaseException(message, context, error_code) {}
};

class ModelTrainingException : public MLException {
public:
    explicit ModelTrainingException(const std::string& message, 
                                  const std::string& model_type = "")
        : MLException("Model training failed: " + message, model_type, 2001) {}
};

class ModelPredictionException : public MLException {
public:
    explicit ModelPredictionException(const std::string& message, 
                                    const std::string& model_type = "")
        : MLException("Model prediction failed: " + message, model_type, 2002) {}
};

class FeatureExtractionException : public MLException {
public:
    explicit FeatureExtractionException(const std::string& message, 
                                       const std::string& feature_name = "")
        : MLException("Feature extraction failed: " + message, feature_name, 2003) {}
};

class HyperparameterException : public MLException {
public:
    explicit HyperparameterException(const std::string& message, 
                                   const std::string& parameter = "")
        : MLException("Invalid hyperparameter: " + message, parameter, 2004) {}
};

class ConfigException : public BaseException {
public:
    explicit ConfigException(const std::string& message, 
                           const std::string& config_key = "",
                           int error_code = 3000)
        : BaseException(message, config_key, error_code) {}
};

class InvalidConfigException : public ConfigException {
public:
    explicit InvalidConfigException(const std::string& message, 
                                  const std::string& config_key = "")
        : ConfigException("Invalid configuration: " + message, config_key, 3001) {}
};

class ResourceException : public BaseException {
public:
    explicit ResourceException(const std::string& message, 
                             const std::string& resource_type = "",
                             int error_code = 4000)
        : BaseException(message, resource_type, error_code) {}
};

class ResourceAllocationException : public ResourceException {
public:
    explicit ResourceAllocationException(const std::string& resource_type, 
                                       const std::string& details = "")
        : ResourceException("Failed to allocate resource: " + resource_type, details, 4001) {}
};

class PortfolioException : public BaseException {
public:
    explicit PortfolioException(const std::string& message, 
                              const std::string& context = "",
                              int error_code = 5000)
        : BaseException(message, context, error_code) {}
};

class InvalidTradeException : public PortfolioException {
public:
    explicit InvalidTradeException(const std::string& message, 
                                 const std::string& trade_details = "")
        : PortfolioException("Invalid trade: " + message, trade_details, 5001) {}
};

namespace ExceptionUtils {
    
    inline std::unique_ptr<BaseException> convertException(const std::exception& e, 
                                                          const std::string& context = "") {
        if (const auto* base_ex = dynamic_cast<const BaseException*>(&e)) {
            return std::make_unique<BaseException>(base_ex->what(), 
                                                  base_ex->context().empty() ? context : base_ex->context(),
                                                  base_ex->error_code());
        }
        
        const std::string msg = e.what();
        
        if (dynamic_cast<const std::invalid_argument*>(&e)) {
            return std::make_unique<DataValidationException>(msg, context);
        } else if (dynamic_cast<const std::out_of_range*>(&e)) {
            return std::make_unique<DataValidationException>("Out of range: " + msg, context);
        } else if (dynamic_cast<const std::runtime_error*>(&e)) {
            return std::make_unique<DataProcessingException>(msg, context);
        } else if (dynamic_cast<const std::logic_error*>(&e)) {
            return std::make_unique<ConfigException>("Logic error: " + msg, context);
        } else if (dynamic_cast<const std::bad_alloc*>(&e)) {
            return std::make_unique<ResourceAllocationException>("Memory", msg);
        }
        
        return std::make_unique<BaseException>(msg, context, 9999);
    }
    
    inline std::string formatException(const std::exception& e, 
                                      const std::string& operation = "") {
        std::string formatted = "Exception";
        if (!operation.empty()) {
            formatted += " in " + operation;
        }
        formatted += ": " + std::string(e.what());
        
        if (const auto* base_ex = dynamic_cast<const BaseException*>(&e)) {
            if (!base_ex->context().empty()) {
                formatted += " [Context: " + base_ex->context() + "]";
            }
            if (base_ex->error_code() != 0) {
                formatted += " [Code: " + std::to_string(base_ex->error_code()) + "]";
            }
        }
        
        return formatted;
    }
}

} // namespace TripleBarrier
