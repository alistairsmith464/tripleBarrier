#pragma once

#ifndef TYPE_CONVERSION_ADAPTER_H
#define TYPE_CONVERSION_ADAPTER_H

// Standard library includes
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <type_traits>

// Qt includes
#include <QString>
#include <QStringList>
#include <QSet>

namespace TypeConversion {

/**
 * @brief Adapter class to handle type conversions between Qt and STL types
 * 
 * This class provides safe conversion utilities to bridge the gap between
 * Qt container types (QSet, QStringList, QString) and STL types 
 * (std::set, std::vector, std::string) to avoid runtime type safety issues.
 */
class TypeAdapter {
public:
    // QString <-> std::string conversions
    static std::string toStdString(const QString& qstr) {
        return qstr.toStdString();
    }
    
    static QString fromStdString(const std::string& str) {
        return QString::fromStdString(str);
    }
    
    // QStringList <-> std::vector<std::string> conversions
    static std::vector<std::string> toStdVector(const QStringList& qlist) {
        std::vector<std::string> result;
        result.reserve(qlist.size());
        for (const QString& item : qlist) {
            result.push_back(item.toStdString());
        }
        return result;
    }
    
    static QStringList fromStdVector(const std::vector<std::string>& vec) {
        QStringList result;
        result.reserve(vec.size());
        for (const std::string& item : vec) {
            result.append(QString::fromStdString(item));
        }
        return result;
    }
    
    // QSet<QString> <-> std::set<std::string> conversions
    static std::set<std::string> toStdSet(const QSet<QString>& qset) {
        std::set<std::string> result;
        for (const QString& item : qset) {
            result.insert(item.toStdString());
        }
        return result;
    }
    
    static QSet<QString> fromStdSet(const std::set<std::string>& stdset) {
        QSet<QString> result;
        for (const std::string& item : stdset) {
            result.insert(QString::fromStdString(item));
        }
        return result;
    }
    
    // Template-based safe container size checking
    template<typename Container1, typename Container2>
    static bool haveSameSize(const Container1& c1, const Container2& c2) {
        return c1.size() == c2.size();
    }
    
    // Template-based safe container empty checking
    template<typename Container>
    static bool isEmpty(const Container& container) {
        return container.empty();
    }
    
    // Specialization for QSet
    static bool isEmpty(const QSet<QString>& qset) {
        return qset.isEmpty();
    }
    
    // Specialization for QStringList
    static bool isEmpty(const QStringList& qlist) {
        return qlist.isEmpty();
    }
    
    // Safe type-aware validation helpers
    template<typename QtContainer, typename StdContainer>
    static bool validateContainerConversion(const QtContainer& qt_container, 
                                          const StdContainer& std_container) {
        return qt_container.size() == std_container.size();
    }
    
    // Convert Qt container to STL equivalent with validation
    template<typename QtContainer>
    static auto convertToStd(const QtContainer& qt_container) 
        -> std::enable_if_t<std::is_same_v<QtContainer, QSet<QString>>, std::set<std::string>> {
        return toStdSet(qt_container);
    }
    
    template<typename QtContainer>
    static auto convertToStd(const QtContainer& qt_container) 
        -> std::enable_if_t<std::is_same_v<QtContainer, QStringList>, std::vector<std::string>> {
        return toStdVector(qt_container);
    }
    
    // Convert STL container to Qt equivalent with validation
    template<typename StdContainer>
    static auto convertToQt(const StdContainer& std_container) 
        -> std::enable_if_t<std::is_same_v<StdContainer, std::set<std::string>>, QSet<QString>> {
        return fromStdSet(std_container);
    }
    
    template<typename StdContainer>
    static auto convertToQt(const StdContainer& std_container) 
        -> std::enable_if_t<std::is_same_v<StdContainer, std::vector<std::string>>, QStringList> {
        return fromStdVector(std_container);
    }
};

/**
 * @brief RAII wrapper for safe type conversions
 * 
 * This class ensures that type conversions are performed safely and 
 * provides automatic cleanup if conversion fails.
 */
template<typename FromType, typename ToType>
class SafeTypeConverter {
public:
    SafeTypeConverter(const FromType& from) : original_(from), converted_(false) {}
    
    ToType convert() {
        if constexpr (std::is_same_v<FromType, QSet<QString>> && 
                      std::is_same_v<ToType, std::set<std::string>>) {
            result_ = TypeAdapter::toStdSet(original_);
        } else if constexpr (std::is_same_v<FromType, QStringList> && 
                             std::is_same_v<ToType, std::vector<std::string>>) {
            result_ = TypeAdapter::toStdVector(original_);
        } else if constexpr (std::is_same_v<FromType, QString> && 
                             std::is_same_v<ToType, std::string>) {
            result_ = TypeAdapter::toStdString(original_);
        } else {
            static_assert(std::is_same_v<FromType, ToType>, 
                         "Unsupported type conversion requested");
        }
        
        converted_ = true;
        return result_;
    }
    
    bool isConverted() const { return converted_; }
    
    const ToType& getResult() const { 
        if (!converted_) {
            throw std::runtime_error("Conversion not performed yet");
        }
        return result_; 
    }
    
private:
    const FromType& original_;
    ToType result_;
    bool converted_;
};

} // namespace TypeConversion

#endif // TYPE_CONVERSION_ADAPTER_H
