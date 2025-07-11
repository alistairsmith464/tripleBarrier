#pragma once
#include "Exceptions.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <iostream> // For std::cerr

namespace TripleBarrier {

template<typename T>
class ResourceGuard {
public:
    ResourceGuard(T* resource, std::function<void(T*)> cleanup)
        : resource_(resource), cleanup_(cleanup) {}
    
    ~ResourceGuard() {
        if (resource_ && cleanup_) {
            try {
                cleanup_(resource_);
            } catch (const std::exception& e) {
                std::cerr << "Resource cleanup failed: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "Resource cleanup failed: Unknown exception" << std::endl;
            }
        }
    }
    
    ResourceGuard(ResourceGuard&& other) noexcept
        : resource_(other.resource_), cleanup_(std::move(other.cleanup_)) {
        other.resource_ = nullptr;
        other.cleanup_ = nullptr;
    }
    
    ResourceGuard& operator=(ResourceGuard&& other) noexcept {
        if (this != &other) {
            if (resource_ && cleanup_) {
                cleanup_(resource_);
            }
            resource_ = other.resource_;
            cleanup_ = std::move(other.cleanup_);
            other.resource_ = nullptr;
            other.cleanup_ = nullptr;
        }
        return *this;
    }
    
    ResourceGuard(const ResourceGuard&) = delete;
    ResourceGuard& operator=(const ResourceGuard&) = delete;
    
    T* get() const { return resource_; }
    T* operator->() const { return resource_; }
    T& operator*() const { return *resource_; }
    
    void release() {
        resource_ = nullptr;
        cleanup_ = nullptr;
    }
    
private:
    T* resource_;
    std::function<void(T*)> cleanup_;
};

template<typename T>
class Result {
public:
    static Result<T> success(T&& value) {
        return Result<T>(std::move(value));
    }
    
    static Result<T> failure(const std::string& error, 
                           const std::string& context = "",
                           int error_code = 0) {
        return Result<T>(error, context, error_code);
    }
    
    static Result<T> failure(const BaseException& exception) {
        return Result<T>(exception.what(), exception.context(), exception.error_code());
    }
    
    bool is_success() const { return has_value_; }
    bool is_failure() const { return !has_value_; }
    
    const T& value() const {
        if (!has_value_) {
            throw std::runtime_error("Attempted to access value of failed result");
        }
        return value_;
    }
    
    T& value() {
        if (!has_value_) {
            throw std::runtime_error("Attempted to access value of failed result");
        }
        return value_;
    }
    
    const std::string& error() const { return error_; }
    const std::string& context() const { return context_; }
    int error_code() const { return error_code_; }
    
    std::string full_error() const {
        std::string full = error_;
        if (!context_.empty()) {
            full += " [Context: " + context_ + "]";
        }
        if (error_code_ != 0) {
            full += " [Code: " + std::to_string(error_code_) + "]";
        }
        return full;
    }
    
private:
    explicit Result(T&& value) : has_value_(true), value_(std::move(value)) {}
    
    Result(const std::string& error, const std::string& context, int error_code)
        : has_value_(false), error_(error), context_(context), error_code_(error_code) {}
    
    bool has_value_;
    T value_;
    std::string error_;
    std::string context_;
    int error_code_;
};

template<typename Func>
class SafeFunction {
public:
    explicit SafeFunction(Func func, const std::string& operation_name = "")
        : func_(func), operation_name_(operation_name) {}
    
    template<typename... Args>
    auto operator()(Args&&... args) noexcept -> Result<decltype(func_(std::forward<Args>(args)...))> {
        try {
            return Result<decltype(func_(std::forward<Args>(args)...))>::success(
                func_(std::forward<Args>(args)...)
            );
        } catch (const BaseException& e) {
            return Result<decltype(func_(std::forward<Args>(args)...))>::failure(e);
        } catch (const std::exception& e) {
            auto converted = ExceptionUtils::convertException(e, operation_name_);
            return Result<decltype(func_(std::forward<Args>(args)...))>::failure(*converted);
        } catch (...) {
            return Result<decltype(func_(std::forward<Args>(args)...))>::failure(
                "Unknown exception occurred", operation_name_, 9999
            );
        }
    }
    
private:
    Func func_;
    std::string operation_name_;
};

template<typename Func>
SafeFunction<Func> makeSafe(Func func, const std::string& operation_name = "") {
    return SafeFunction<Func>(func, operation_name);
}

namespace Validation {
    
    inline void validateNotNull(const void* ptr, const std::string& name) {
        if (ptr == nullptr) {
            throw DataValidationException("Null pointer", name);
        }
    }
    
    inline void validateNotEmpty(const std::string& str, const std::string& name) {
        if (str.empty()) {
            throw DataValidationException("Empty string", name);
        }
    }
    
    template<typename Container>
    inline void validateNotEmpty(const Container& container, const std::string& name) {
        if (container.empty()) {
            throw DataValidationException("Empty container", name);
        }
    }
    
    inline void validateRange(double value, double min, double max, const std::string& name) {
        if (value < min || value > max) {
            throw DataValidationException(
                "Value " + std::to_string(value) + " not in range [" + 
                std::to_string(min) + ", " + std::to_string(max) + "]", name
            );
        }
    }
    
    inline void validatePositive(double value, const std::string& name) {
        if (value <= 0.0) {
            throw DataValidationException(
                "Value " + std::to_string(value) + " must be positive", name
            );
        }
    }
    
    inline void validateNonNegative(double value, const std::string& name) {
        if (value < 0.0) {
            throw DataValidationException(
                "Value " + std::to_string(value) + " must be non-negative", name
            );
        }
    }
    
    inline void validateFinite(double value, const std::string& name) {
        if (!std::isfinite(value)) {
            throw DataValidationException("Value is not finite", name);
        }
    }
    
    template<typename Container>
    inline void validateSizeMatch(const Container& c1, const Container& c2, 
                                 const std::string& name1, const std::string& name2) {
        if (c1.size() != c2.size()) {
            throw DataValidationException(
                "Size mismatch: " + name1 + " (" + std::to_string(c1.size()) + 
                ") vs " + name2 + " (" + std::to_string(c2.size()) + ")"
            );
        }
    }
    
    template<typename Container1, typename Container2>
    inline void validateSizeMatch(const Container1& c1, const Container2& c2, 
                                 const std::string& name1, const std::string& name2) {
        if (c1.size() != c2.size()) {
            throw DataValidationException(
                "Size mismatch: " + name1 + " (" + std::to_string(c1.size()) + 
                ") vs " + name2 + " (" + std::to_string(c2.size()) + ")"
            );
        }
    }
}

class ErrorAccumulator {
public:
    void addError(const std::string& error, const std::string& context = "") {
        errors_.push_back({error, context});
    }
    
    void addError(const BaseException& exception) {
        errors_.push_back({exception.what(), exception.context()});
    }
    
    bool hasErrors() const { return !errors_.empty(); }
    
    size_t errorCount() const { return errors_.size(); }
    
    std::vector<std::string> getErrors() const {
        std::vector<std::string> result;
        for (const auto& error : errors_) {
            std::string formatted = error.first;
            if (!error.second.empty()) {
                formatted += " [" + error.second + "]";
            }
            result.push_back(formatted);
        }
        return result;
    }
    
    std::string getAllErrors() const {
        std::string result;
        for (const auto& error : getErrors()) {
            if (!result.empty()) result += "; ";
            result += error;
        }
        return result;
    }
    
    void clear() { errors_.clear(); }
    
private:
    std::vector<std::pair<std::string, std::string>> errors_;
};

} // namespace TripleBarrier
