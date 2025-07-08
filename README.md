# Triple Barrier Labeling with Probabilistic Extension

## Summary

This project implements and extends the Triple Barrier Method from *Advances in Financial Machine Learning* (López de Prado, 2018). It provides a framework for labeling financial time series by detecting significant price movements (using a CUSUM filter) and assigning labels based on which of three barriers—profit, stop-loss, or holding period—is hit first.  

We extend the original method with **probabilistic barrier labeling**, generating continuous targets in the range [-1, +1] to reflect the likelihood of price hitting each barrier. This enables richer machine learning models that support dynamic position sizing and improved risk management.

Upload time series data (timestamp, price), compute features, apply the triple barrier logic, and train predictive models directly in the app.

----



## 1. Introduction

Financial time series are inherently noisy, non-stationary, and prone to sudden regime changes, which makes supervised machine learning particularly challenging in this domain. As Marcos López de Prado highlights in *Advances in Financial Machine Learning (2018)*:

> _“Labels are the weakest link in supervised learning, and in finance, traditional labeling techniques are often inadequate.”_

To address this, Prado introduced the **Triple Barrier Method**, which improves upon fixed time horizon labeling by allowing for dynamic exit conditions based on price movements and holding periods.  

This project implements the Triple Barrier Method as a pre-processing step for machine learning and **extends it** by incorporating probabilistic barrier outcomes. The goal is to provide more informative targets for predictive modeling, enabling refined position sizing and risk management in quantitative trading strategies.

---

### 1.1 The Triple Barrier Method

The Triple Barrier Method defines a framework for labeling financial data based on the outcome of trade-like events within a specified time window.  

At the start of an event window, three barriers are set:  

1. **Upper horizontal barrier** – profit-taking level  
2. **Lower horizontal barrier** – stop-loss level  
3. **Vertical barrier** – maximum holding period  

The method assigns a label to each event based on which barrier the price touches first:  

- `+1` if the upper barrier is breached (profitable movement)  
- `-1` if the lower barrier is crossed (loss)  
- `0` if neither horizontal barrier is hit before the vertical barrier (neutral outcome)  

Prado describes this approach as:  

> _“A generalization of fixed-time horizon labeling, where early exits are allowed if prices hit specified thresholds before the end of the event window.”_

This methodology effectively accounts for volatility and provides a more realistic labeling mechanism for financial time series, where price paths are not uniformly distributed over time.  

---

### 1.2 Event Definition and CUSUM Filter

A critical component of the Triple Barrier Method is the definition of **event windows**, which determine the points in time where the labeling process begins.  

In financial time series, consecutive events often overlap, leading to label leakage and dependence between samples. To address this, López de Prado proposes using a **Cumulative Sum (CUSUM) filter** to identify significant price movements and define non-overlapping event windows.  

The CUSUM filter detects structural shifts in the price series by accumulating incremental price changes until they exceed a specified threshold. As Prado explains:  

> _“The CUSUM filter is designed to avoid triggering a new event for small movements, ensuring that only meaningful price changes initiate labeling windows.”_

By applying the CUSUM filter, the dataset is reduced to a series of timestamps where significant directional changes occur, each serving as the initial timestamp for the triple barrier logic. This approach mitigates noise in high-frequency data and creates more reliable and independent labels for machine learning.

---

### 1.3 Extending the Theory: Probabilistic Barrier Labeling

While the Triple Barrier Method provides a robust framework for labeling, its **hard assignment** of `+1`, `-1`, or `0` does not fully capture the uncertainty and path dependency inherent in financial markets. Small changes in price dynamics can lead to different barrier outcomes, making binary labels potentially misleading for machine learning models.  

This project introduces an extension: **Probabilistic Barrier Labeling**.  

Instead of assigning a discrete label based on the first barrier breached, we compute a continuous value that represents the **expected outcome** of each event:  

- `+1` – strong bias toward the upper barrier being reached first  
- `-1` – strong bias toward the lower barrier being reached first  
- `0` – neutral outcome (vertical barrier hit) or balanced historical tendency  

For example:  
A label of `-0.3` suggests that, under comparable historical scenarios, the price tended to hit the lower barrier first slightly more often than the upper barrier.  

This probabilistic labeling creates a **continuous target variable** well-suited for regression models. It allows the machine learning pipeline to learn not only directional signals but also their relative strengths.

This approach supports:  

1. **Dynamic position sizing** – scaling trade sizes based on prediction confidence  
2. **Improved calibration** – more nuanced risk management  
3. **Alignment with modern ML practices** – probabilistic outputs improve interpretability  

This extension builds on Prado’s original concept while enhancing its applicability in real-world quantitative trading systems.

---
