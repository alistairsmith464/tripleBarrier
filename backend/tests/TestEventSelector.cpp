#include <gtest/gtest.h>
#include "../data/EventSelector.h"
#include "../data/DataRow.h"

TEST(EventSelectorTest, SelectEventsInterval) {
    std::vector<DataRow> rows(10);
    for (int i = 0; i < 10; ++i) rows[i].timestamp = std::to_string(i);
    int interval = 3;
    auto events = EventSelector::selectEvents(rows, interval);
    EXPECT_EQ(events.size(), 4);
    EXPECT_EQ(events[0].index, 0);
    EXPECT_EQ(events[1].index, 3);
    EXPECT_EQ(events[2].index, 6);
    EXPECT_EQ(events[3].index, 9);
}

TEST(EventSelectorTest, EmptyRows) {
    std::vector<DataRow> rows;
    auto events = EventSelector::selectEvents(rows, 2);
    EXPECT_TRUE(events.empty());
}

TEST(EventSelectorTest, IntervalLargerThanRows) {
    std::vector<DataRow> rows(5);
    for (int i = 0; i < 5; ++i) rows[i].timestamp = std::to_string(i);
    int interval = 10;
    auto events = EventSelector::selectEvents(rows, interval);
    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events[0].index, 0);
}

TEST(EventSelectorTest, IntervalIsOne) {
    std::vector<DataRow> rows(5);
    for (int i = 0; i < 5; ++i) rows[i].timestamp = std::to_string(i);
    int interval = 1;
    auto events = EventSelector::selectEvents(rows, interval);
    ASSERT_EQ(events.size(), 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(events[i].index, i);
        EXPECT_EQ(events[i].timestamp, std::to_string(i));
    }
}

TEST(EventSelectorTest, LargeDataSet) {
    const int N = 100000;
    const int interval = 1000;
    std::vector<DataRow> rows(N);
    for (int i = 0; i < N; ++i) rows[i].timestamp = std::to_string(i);
    auto events = EventSelector::selectEvents(rows, interval);
    EXPECT_EQ(events.size(), N / interval + (N % interval ? 1 : 0));
    EXPECT_EQ(events[0].index, 0);
    EXPECT_EQ(events.back().index, (N / interval) * interval < N ? (N / interval) * interval : N - interval);
}
