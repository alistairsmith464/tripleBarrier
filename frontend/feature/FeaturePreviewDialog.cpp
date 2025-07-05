#include "FeaturePreviewDialog.h"
#include <QVBoxLayout>
#include <QTableWidget>
#include <QHeaderView>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QFileDialog>
#include "../utils/CSVExportUtils.h"
#include "../utils/DialogUtils.h"
#include "../backend/data/FeatureCalculator.h"
#include <map>
#include <set>

FeaturePreviewDialog::FeaturePreviewDialog(const QSet<QString>& selectedFeatures,
                                           const std::vector<PreprocessedRow>& rows,
                                           const std::vector<LabeledEvent>& labeledEvents,
                                           QWidget* parent)
    : QDialog(parent)
{
    setWindowTitle("Feature Preview");
    QVBoxLayout* vbox = new QVBoxLayout(this);
    QMap<QString, std::string> featureMap = {
        {"Close-to-close return for the previous day", FeatureCalculator::CLOSE_TO_CLOSE_RETURN_1D},
        {"Return over the past 5 days", FeatureCalculator::RETURN_5D},
        {"Return over the past 10 days", FeatureCalculator::RETURN_10D},
        {"Rolling standard deviation of daily returns over the last 5 days", FeatureCalculator::ROLLING_STD_5D},
        {"EWMA volatility over 10 days", FeatureCalculator::EWMA_VOL_10D},
        {"5-day simple moving average (SMA)", FeatureCalculator::SMA_5D},
        {"10-day SMA", FeatureCalculator::SMA_10D},
        {"20-day SMA", FeatureCalculator::SMA_20D},
        {"Distance between current close price and 5-day SMA", FeatureCalculator::DIST_TO_SMA_5D},
        {"Rate of Change (ROC) over 5 days", FeatureCalculator::ROC_5D},
        {"Relative Strength Index (RSI) over 14 days", FeatureCalculator::RSI_14D},
        {"5-day high minus 5-day low (price range)", FeatureCalculator::PRICE_RANGE_5D},
        {"Current close price relative to 5-day high", FeatureCalculator::CLOSE_OVER_HIGH_5D},
        {"Slope of linear regression of close prices over 10 days", FeatureCalculator::SLOPE_LR_10D},
        {"Day of the week", FeatureCalculator::DAY_OF_WEEK},
        {"Days since last event", FeatureCalculator::DAYS_SINCE_LAST_EVENT}
    };
    std::set<std::string> backendFeatures;
    for (const QString& feat : selectedFeatures) {
        if (featureMap.contains(feat)) backendFeatures.insert(featureMap[feat]);
    }
    std::vector<double> prices;
    std::vector<std::string> timestamps;
    std::vector<int> eventIndices;
    for (size_t i = 0; i < rows.size(); ++i) {
        prices.push_back(rows[i].price);
        timestamps.push_back(rows[i].timestamp);
    }
    for (const auto& e : labeledEvents) {
        auto it = std::find_if(rows.begin(), rows.end(), [&](const PreprocessedRow& r) { return r.timestamp == e.entry_time; });
        if (it != rows.end()) eventIndices.push_back(int(std::distance(rows.begin(), it)));
    }
    std::vector<std::map<std::string, double>> allFeatures;
    for (size_t i = 0; i < eventIndices.size(); ++i) {
        allFeatures.push_back(FeatureCalculator::calculateFeatures(prices, timestamps, eventIndices, int(i), backendFeatures));
    }
    QTableWidget* table = new QTableWidget(int(eventIndices.size()), int(backendFeatures.size()), this);
    QStringList headers;
    for (const QString& feat : selectedFeatures) headers << feat;
    table->setHorizontalHeaderLabels(headers);
    int col = 0;
    for (const QString& feat : selectedFeatures) {
        std::string backendId = featureMap[feat];
        for (int row = 0; row < int(eventIndices.size()); ++row) {
            double val = allFeatures[row].count(backendId) ? allFeatures[row][backendId] : NAN;
            table->setItem(row, col, new QTableWidgetItem(QString::number(val)));
        }
        ++col;
    }
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    vbox->addWidget(table);
    QPushButton* exportBtn = new QPushButton("Export to CSV", this);
    vbox->addWidget(exportBtn);
    connect(exportBtn, &QPushButton::clicked, this, [=]() {
        QString fileName = QFileDialog::getSaveFileName(this, "Export Features to CSV", "features_output.csv", "CSV Files (*.csv);;All Files (*.*)");
        if (fileName.isEmpty()) return;
        QFile file(fileName);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            DialogUtils::showError(this, "Export Error", "Could not open file for writing.");
            return;
        }
        QTextStream out(&file);
        QStringList headers;
        for (const QString& feat : selectedFeatures) headers << feat;
        out << headers.join(",") << "\n";
        for (int row = 0; row < table->rowCount(); ++row) {
            QStringList rowVals;
            for (int col = 0; col < table->columnCount(); ++col) {
                QTableWidgetItem* item = table->item(row, col);
                rowVals << (item ? item->text() : "");
            }
            out << rowVals.join(",") << "\n";
        }
        file.close();
        DialogUtils::showInfo(this, "Export Complete", "Features exported to " + fileName);
    });
    QDialogButtonBox* box = new QDialogButtonBox(QDialogButtonBox::Ok, this);
    connect(box, &QDialogButtonBox::accepted, this, &QDialog::accept);
    vbox->addWidget(box);
}
