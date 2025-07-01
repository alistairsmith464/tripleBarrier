#pragma once
#include <vector>
#include "DataRow.h"

class DataSource {
public:
    virtual std::vector<DataRow> loadData(const std::string& source) = 0;
    virtual ~DataSource() = default;
};
