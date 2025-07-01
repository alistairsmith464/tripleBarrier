#include "CSVDataSource.h"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <vector>
#include <functional>

namespace {
    void checkFileExtension(const std::string& filename);
    std::ifstream openFile(const std::string& filename);
    std::vector<std::string> parseHeaderRow(const std::string& line);
    std::unordered_map<std::string, size_t> buildHeaderIndex(const std::vector<std::string>& headers);
    void validateRequiredHeaders(const std::unordered_map<std::string, size_t>& headerIndex);
    std::vector<std::string> parseFields(const std::string& line);
    using OptionalFieldSetter = std::function<void(DataRow&, const std::string&, size_t)>;
    std::vector<std::pair<size_t, OptionalFieldSetter>> buildOptionalFieldSetters(const std::unordered_map<std::string, size_t>& headerIndex);
    DataRow parseDataRow(const std::vector<std::string>& fields, const std::unordered_map<std::string, size_t>& headerIndex, const std::vector<std::pair<size_t, OptionalFieldSetter>>& optionalSetters, size_t rowNum);
    std::string trim(const std::string& s);
}

std::vector<DataRow> CSVDataSource::loadData(const std::string& filename) {
    checkFileExtension(filename);
    std::ifstream file = openFile(filename);
    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error("CSV file is empty: " + filename);
    }

    std::vector<std::string> headers = parseHeaderRow(line);
    for (auto& h : headers) h = trim(h);
    auto headerIndex = buildHeaderIndex(headers);
    validateRequiredHeaders(headerIndex);
    auto optionalSetters = buildOptionalFieldSetters(headerIndex);
    std::vector<DataRow> data;
    size_t rowNum = 1;
    size_t requiredFields = headers.size();

    while (std::getline(file, line)) {
        ++rowNum;
        std::vector<std::string> fields = parseFields(line);
        for (auto& f : fields) f = trim(f);
        if (fields.size() < requiredFields) {
            throw std::runtime_error("Malformed row (too few fields) at row " + std::to_string(rowNum));
        }
        data.push_back(parseDataRow(fields, headerIndex, optionalSetters, rowNum));
    }
    if (data.empty()) {
        throw std::runtime_error("CSV file has no data rows: " + filename);
    }
    return data;
}

namespace {
    void checkFileExtension(const std::string& filename) {
        if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".csv") {
            throw std::runtime_error("File is not a CSV: " + filename);
        }
    }

    std::ifstream openFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        return file;
    }

    std::vector<std::string> parseHeaderRow(const std::string& line) {
        std::vector<std::string> headers;
        std::istringstream ss(line);
        std::string col;
        while (std::getline(ss, col, ',')) {
            headers.push_back(col); // trimming is done in loadData
        }
        if (headers.empty()) {
            throw std::runtime_error("CSV file has no header row");
        }
        return headers;
    }

    std::unordered_map<std::string, size_t> buildHeaderIndex(const std::vector<std::string>& headers) {
        std::unordered_map<std::string, size_t> headerIndex;
        for (size_t i = 0; i < headers.size(); ++i) {
            std::string h = headers[i];
            std::transform(h.begin(), h.end(), h.begin(), ::tolower);
            headerIndex[h] = i;
        }
        return headerIndex;
    }

    void validateRequiredHeaders(const std::unordered_map<std::string, size_t>& headerIndex) {
        if (headerIndex.find("timestamp") == headerIndex.end() || headerIndex.find("price") == headerIndex.end()) {
            throw std::runtime_error("CSV missing required columns: timestamp and price");
        }
    }

    std::vector<std::string> parseFields(const std::string& line) {
        std::vector<std::string> fields;
        std::istringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) {
            fields.push_back(field); // trimming is done in loadData
        }
        return fields;
    }

    std::vector<std::pair<size_t, OptionalFieldSetter>> buildOptionalFieldSetters(const std::unordered_map<std::string, size_t>& headerIndex) {
        std::vector<std::pair<size_t, OptionalFieldSetter>> setters;
        struct FieldInfo { const char* name; OptionalFieldSetter setter; };
        std::vector<FieldInfo> infos = {
            {"open",   [](DataRow& row, const std::string& val, size_t rowNum) { if (!val.empty()) try { row.open = std::stod(val); } catch (...) { throw std::runtime_error("Invalid value for 'open' at row " + std::to_string(rowNum)); } }},
            {"high",   [](DataRow& row, const std::string& val, size_t rowNum) { if (!val.empty()) try { row.high = std::stod(val); } catch (...) { throw std::runtime_error("Invalid value for 'high' at row " + std::to_string(rowNum)); } }},
            {"low",    [](DataRow& row, const std::string& val, size_t rowNum) { if (!val.empty()) try { row.low = std::stod(val); } catch (...) { throw std::runtime_error("Invalid value for 'low' at row " + std::to_string(rowNum)); } }},
            {"close",  [](DataRow& row, const std::string& val, size_t rowNum) { if (!val.empty()) try { row.close = std::stod(val); } catch (...) { throw std::runtime_error("Invalid value for 'close' at row " + std::to_string(rowNum)); } }},
            {"volume", [](DataRow& row, const std::string& val, size_t rowNum) { if (!val.empty()) try { row.volume = std::stod(val); } catch (...) { throw std::runtime_error("Invalid value for 'volume' at row " + std::to_string(rowNum)); } }},
        };
        for (const auto& info : infos) {
            auto it = headerIndex.find(info.name);
            if (it != headerIndex.end()) {
                setters.emplace_back(it->second, info.setter);
            }
        }
        return setters;
    }

    DataRow parseDataRow(const std::vector<std::string>& fields, const std::unordered_map<std::string, size_t>& headerIndex, const std::vector<std::pair<size_t, OptionalFieldSetter>>& optionalSetters, size_t rowNum) {
        DataRow row;
        row.timestamp = fields[headerIndex.at("timestamp")];
        try {
            row.price = std::stod(fields[headerIndex.at("price")]);
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid price at row " + std::to_string(rowNum));
        }
        for (const auto& [idx, setter] : optionalSetters) {
            setter(row, fields[idx], rowNum);
        }
        return row;
    }

    std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        size_t end = s.find_last_not_of(" \t\r\n");
        return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    }
}
