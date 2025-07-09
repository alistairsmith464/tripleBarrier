# Triple Barrier Labeling with Time-Decayed Labels

## Summary

This project implements and extends the Triple Barrier Method from *Advances in Financial Machine Learning* (López de Prado, 2018). It provides a framework for labeling financial time series by detecting significant price movements (using a CUSUM filter) and assigning labels based on which of three barriers—profit, stop-loss, or holding period—is hit first.  

We extend the original method with **time-decayed labels**, which start at `+1` or `-1` when a barrier is hit immediately and decay toward `0` as time passes without hitting any barrier. This allows models to learn both **directional bias** and **signal confidence over time**, enabling more nuanced decision-making in trading strategies.

Upload time series data (timestamp, price), compute features, apply the triple barrier logic, and train predictive models directly in the app.

---

## 1. Introduction

Financial time series are inherently noisy, non-stationary, and prone to sudden regime changes, which makes supervised machine learning particularly challenging in this domain. As Marcos López de Prado highlights in *Advances in Financial Machine Learning (2018)*:

> “Labels are the weakest link in supervised learning, and in finance, traditional labeling techniques are often inadequate.”

To address this, Prado introduced the **Triple Barrier Method**, which improves upon fixed time horizon labeling by allowing for dynamic exit conditions based on price movements and holding periods.  

This project implements the Triple Barrier Method as a pre-processing step for machine learning and **extends it** by incorporating a time-decayed label methodology. The goal is to provide more informative targets for predictive modeling, enabling refined position sizing and risk management.

---

## 1.1 The Triple Barrier Method

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

> “A generalization of fixed-time horizon labeling, where early exits are allowed if prices hit specified thresholds before the end of the event window.”

This methodology effectively accounts for volatility and provides a more realistic labeling mechanism for financial time series, where price paths are not uniformly distributed over time.  

---

## 1.2 Event Definition and CUSUM Filter

A critical component of the Triple Barrier Method is the definition of **event windows**, which determine the points in time where the labeling process begins.  

In financial time series, consecutive events often overlap, leading to label leakage and dependence between samples. To address this, López de Prado proposes using a **Cumulative Sum (CUSUM) filter** to identify significant price movements and define non-overlapping event windows.  

The CUSUM filter detects structural shifts in the price series by accumulating incremental price changes until they exceed a specified threshold. As Prado explains:  

> “The CUSUM filter is designed to avoid triggering a new event for small movements, ensuring that only meaningful price changes initiate labeling windows.”

By applying the CUSUM filter, the dataset is reduced to a series of timestamps where significant directional changes occur, each serving as the initial timestamp for the triple barrier logic. This approach mitigates noise in high-frequency data and creates more reliable and independent labels for machine learning.

---

## 1.3 Extending the Theory: Time-Decayed Labeling

While the Triple Barrier Method provides a robust framework for labeling, its **hard assignment** of `+1`, `-1`, or `0` does not fully capture the dynamics of financial markets. Two paths that hit the same barrier may differ greatly in how quickly they do so, which is critical information for risk and trade management.  

This project introduces an extension: **Time-Decayed Labeling**.  

In this approach:  

- The label is `+1` if the upper barrier is hit immediately.  
- The label is `-1` if the lower barrier is hit immediately.  
- Otherwise, the label decays toward `0` as time progresses without hitting any barrier, reflecting diminishing confidence in the signal.  

This decay is computed using one of three configurable decay functions:  

1. **Exponential Decay** – Applies a decay factor `f(t) = exp(-λ * t / T)`, where `t` is the time elapsed since event start, `T` is the vertical barrier length, and `λ` controls the decay rate.  
2. **Linear Decay** – Reduces the label linearly with `f(t) = max(0, 1 - α * t / T)`, where `α` defines the slope of decay.  
3. **Hyperbolic Decay** – Decays following `f(t) = 1 / (1 + β * t / T)`, where `β` governs the curve’s steepness.  

Here, `t/T` is the **normalized time ratio** (fraction of the maximum holding period elapsed). As time approaches the vertical barrier without hitting a horizontal barrier, the decay function asymptotically brings the label closer to `0`.  

This creates a **continuous target variable** suitable for regression models, enabling machine learning pipelines to learn both **directional bias** and **signal strength over time**.

This approach supports:  

1. **Dynamic position sizing** – scale trades based on prediction confidence and urgency  
2. **Early exit strategies** – weaker signals naturally decay to near-zero  
3. **Enhanced interpretability** – models learn to distinguish fast, strong moves from slow, weak trends  

This extension builds on Prado’s original concept, making it more actionable for real-world trading systems.

---
