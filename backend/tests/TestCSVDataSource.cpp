#include <gtest/gtest.h>
#include "../data/CSVDataSource.h"
#include <fstream>
#include <cstdio>

template<typename... Lines>
std::string createTempCSV(Lines... lines) {
    std::string filename = "test_temp.csv";
    std::ofstream ofs(filename);
    ((ofs << lines << "\n"), ...);
    ofs.close();
    return filename;
}

TEST(CSVDataSourceTest, ParsesValidCSVWithAllColumns) {
    std::string filename = createTempCSV(
        "timestamp,price,open,high,low,close,volume",
        "2023-01-01 09:30:00,101.45,101.0,102.0,100.5,101.5,1000"
    );
    CSVDataSource src;
    auto rows = src.loadData(filename);
    ASSERT_EQ(rows.size(), 1);
    EXPECT_EQ(rows[0].timestamp, "2023-01-01 09:30:00");
    EXPECT_DOUBLE_EQ(rows[0].price, 101.45);
    EXPECT_TRUE(rows[0].open.has_value());
    EXPECT_DOUBLE_EQ(rows[0].open.value(), 101.0);
    EXPECT_TRUE(rows[0].volume.has_value());
    std::remove(filename.c_str());
}

TEST(CSVDataSourceTest, ThrowsOnMissingRequiredColumns) {
    std::string filename = createTempCSV(
        "timestamp,open,high,low,close,volume",
        "2023-01-01 09:30:00,101.0,102.0,100.5,101.5,1000"
    );
    CSVDataSource src;
    EXPECT_THROW(src.loadData(filename), std::runtime_error);
    std::remove(filename.c_str());
}

TEST(CSVDataSourceTest, ThrowsOnInvalidPrice) {
    std::string filename = createTempCSV(
        "timestamp,price",
        "2023-01-01 09:30:00,not_a_number"
    );
    CSVDataSource src;
    EXPECT_THROW(src.loadData(filename), std::runtime_error);
    std::remove(filename.c_str());
}

TEST(CSVDataSourceTest, HandlesMissingOptionalColumns) {
    std::string filename = createTempCSV(
        "timestamp,price",
        "2023-01-01 09:30:00,101.45"
    );
    CSVDataSource src;
    auto rows = src.loadData(filename);
    ASSERT_EQ(rows.size(), 1);
    EXPECT_FALSE(rows[0].open.has_value());
    std::remove(filename.c_str());
}
