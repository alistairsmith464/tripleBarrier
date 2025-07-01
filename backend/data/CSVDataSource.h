#pragma once
#include "DataSource.h"
#include <string>
#include <vector>

class CSVDataSource : public DataSource {
public:
    std::vector<DataRow> loadData(const std::string& filename) override;
};
