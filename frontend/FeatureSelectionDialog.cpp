#include "FeatureSelectionDialog.h"
#include "ui/UIStrings.h"
#include <QVBoxLayout>
#include <QDialogButtonBox>
#include <QPushButton>

FeatureSelectionDialog::FeatureSelectionDialog(QWidget* parent)
    : QDialog(parent)
{
    setWindowTitle(UIStrings::FEATURE_SELECTION_TITLE);
    QVBoxLayout* layout = new QVBoxLayout(this);
    QStringList features = {
        "Close-to-close return for the previous day",
        "Return over the past 5 days",
        "Return over the past 10 days",
        "Rolling standard deviation of daily returns over the last 5 days",
        "EWMA volatility over 10 days",
        "5-day simple moving average (SMA)",
        "10-day SMA",
        "20-day SMA",
        "Distance between current close price and 5-day SMA",
        "Rate of Change (ROC) over 5 days",
        "Relative Strength Index (RSI) over 14 days",
        "5-day high minus 5-day low (price range)",
        "Current close price relative to 5-day high",
        "Slope of linear regression of close prices over 10 days",
        "Day of the week",
        "Days since last event"
    };
    for (const QString& feat : features) {
        QCheckBox* cb = new QCheckBox(feat, this);
        layout->addWidget(cb);
        featureCheckboxes[feat] = cb;
    }
    QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    buttonBox->button(QDialogButtonBox::Ok)->setText(UIStrings::OK);
    buttonBox->button(QDialogButtonBox::Cancel)->setText(UIStrings::CANCEL);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layout->addWidget(buttonBox);
}

QSet<QString> FeatureSelectionDialog::selectedFeatures() const {
    QSet<QString> selected;
    for (auto it = featureCheckboxes.begin(); it != featureCheckboxes.end(); ++it) {
        if (it.value()->isChecked())
            selected.insert(it.key());
    }
    return selected;
}
