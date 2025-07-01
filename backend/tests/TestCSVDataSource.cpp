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

TEST(CSVDataSourceTest, ParsesMultipleRows) {
    std::string filename = createTempCSV(
        "timestamp,price,open,high,low,close,volume",
        "2023-01-01 09:30:00,101.45,101.0,102.0,100.5,101.5,1000",
        "2023-01-01 09:31:00,102.00,101.5,102.5,101.0,102.0,1100"
    );
    CSVDataSource src;
    auto rows = src.loadData(filename);
    ASSERT_EQ(rows.size(), 2);
    EXPECT_EQ(rows[1].timestamp, "2023-01-01 09:31:00");
    EXPECT_DOUBLE_EQ(rows[1].price, 102.00);
    std::remove(filename.c_str());
}

TEST(CSVDataSourceTest, IgnoresExtraColumns) {
    std::string filename = createTempCSV(
        "timestamp,price,foo,bar",
        "2023-01-01 09:30:00,101.45,abc,xyz"
    );
    CSVDataSource src;
    auto rows = src.loadData(filename);
    ASSERT_EQ(rows.size(), 1);
    EXPECT_EQ(rows[0].timestamp, "2023-01-01 09:30:00");
    std::remove(filename.c_str());
}

TEST(CSVDataSourceTest, ThrowsOnEmptyFile) {
    std::string filename = createTempCSV("");
    CSVDataSource src;
    EXPECT_THROW(src.loadData(filename), std::runtime_error);
    std::remove(filename.c_str());
}

TEST(CSVDataSourceTest, ThrowsOnHeaderOnly) {
    std::string filename = createTempCSV("timestamp,price");
    CSVDataSource src;
    EXPECT_THROW(src.loadData(filename), std::runtime_error);
    std::remove(filename.c_str());
}

TEST(CSVDataSourceTest, HandlesWhitespace) {
    std::string filename = createTempCSV(
        "timestamp, price ",
        " 2023-01-01 09:30:00 , 101.45 "
    );
    CSVDataSource src;
    auto rows = src.loadData(filename);
    ASSERT_EQ(rows.size(), 1);
    EXPECT_EQ(rows[0].timestamp, "2023-01-01 09:30:00");
    EXPECT_DOUBLE_EQ(rows[0].price, 101.45);
    std::remove(filename.c_str());
}

TEST(CSVDataSourceTest, ThrowsOnMalformedRow) {
    std::string filename = createTempCSV(
        "timestamp,price",
        "2023-01-01 09:30:00"
    );
    CSVDataSource src;
    EXPECT_THROW(src.loadData(filename), std::runtime_error);
    std::remove(filename.c_str());
}

TEST(CSVDataSourceTest, ThrowsOnNonexistentFile) {
    CSVDataSource src;
    EXPECT_THROW(src.loadData("no_such_file.csv"), std::runtime_error);
}

TEST(CSVDataSourceTest, ParsesRowsWithSomeOptionalColumns) {
    std::string filename = createTempCSV(
        "timestamp,price,open,close",
        "2023-01-01 09:30:00,101.45,101.0,101.5",
        "2023-01-01 09:31:00,102.00,101.5,102.0"
    );
    CSVDataSource src;
    auto rows = src.loadData(filename);
    ASSERT_EQ(rows.size(), 2);
    EXPECT_TRUE(rows[0].open.has_value());
    EXPECT_TRUE(rows[0].close.has_value());
    EXPECT_FALSE(rows[0].high.has_value());
    EXPECT_FALSE(rows[0].low.has_value());
    EXPECT_FALSE(rows[0].volume.has_value());
    std::remove(filename.c_str());
}

TEST(CSVDataSourceTest, ParsesRowsWithNoOptionalColumns) {
    std::string filename = createTempCSV(
        "timestamp,price",
        "2023-01-01 09:30:00,101.45",
        "2023-01-01 09:31:00,102.00"
    );
    CSVDataSource src;
    auto rows = src.loadData(filename);
    ASSERT_EQ(rows.size(), 2);
    EXPECT_FALSE(rows[0].open.has_value());
    EXPECT_FALSE(rows[1].close.has_value());
    std::remove(filename.c_str());
}

TEST(CSVDataSourceTest, ParsesRowsWithMixedOptionalColumns) {
    std::string filename = createTempCSV(
        "timestamp,price,open,close,volume",
        "2023-01-01 09:30:00,101.45,101.0,101.5,1000",
        "2023-01-01 09:31:00,102.00,,,"
    );
    CSVDataSource src;
    auto rows = src.loadData(filename);
    ASSERT_EQ(rows.size(), 2);
    EXPECT_TRUE(rows[0].open.has_value());
    EXPECT_TRUE(rows[0].close.has_value());
    EXPECT_TRUE(rows[0].volume.has_value());
    EXPECT_FALSE(rows[1].open.has_value());
    EXPECT_FALSE(rows[1].close.has_value());
    EXPECT_FALSE(rows[1].volume.has_value());
    std::remove(filename.c_str());
}

TEST(CSVDataSourceTest, ParsesManyRows) {
    std::string filename = createTempCSV(
        "timestamp,price",
        "2023-01-01 09:30:00,101.45",
        "2023-01-01 09:31:00,102.00",
        "2023-01-01 09:32:00,102.50",
        "2023-01-01 09:33:00,103.00",
        "2023-01-01 09:34:00,103.50"
    );
    CSVDataSource src;
    auto rows = src.loadData(filename);
    ASSERT_EQ(rows.size(), 5);
    EXPECT_EQ(rows[4].timestamp, "2023-01-01 09:34:00");
    EXPECT_DOUBLE_EQ(rows[4].price, 103.50);
    std::remove(filename.c_str());
}
