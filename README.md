# Triple Barrier Labeling with Time-Decayed Labels

## Summary

This project implements and extends the Triple Barrier method from *Advances in Financial Machine Learning* (López de Prado, 2018). It provides a framework for labeling financial time series by detecting significant price movements (using a CUSUM filter) and assigning labels based on which of three barriers—profit, stop-loss, or holding period—is hit first.  

We extend the original method with **time-decayed labels**, which start at `+1` or `-1` when a barrier is hit immediately and decay toward `0` as time passes without hitting any barrier. This allows models to learn both **directional bias** and **signal confidence over time**, enabling more nuanced decision-making in trading strategies.

Upload time series data (timestamp, price), compute features, apply the triple barrier logic, and train predictive models directly in the app. This project compares the relative performance of the original **Triple Barrier** method with the **Time to barrier modification** introduced in this project.

---

## 1. Introduction

Financial time series are inherently noisy, non-stationary, and prone to sudden regime changes, which makes supervised machine learning particularly challenging in this domain. As Marcos López de Prado highlights in *Advances in Financial Machine Learning (2018)*:

> “Labels are the weakest link in supervised learning, and in finance, traditional labeling techniques are often inadequate.”

To address this, Prado introduced the **Triple Barrier** method, which improves upon fixed time horizon labeling by allowing for dynamic exit conditions based on price movements and holding periods.  

This project implements the Triple Barrier Method as a pre-processing step for machine learning and extends it by incorporating a time-decayed label methodology. The goal is to provide more informative targets for predictive modeling, enabling refined position sizing and risk management.

### 1.1 The Triple Barrier Method

The Triple Barrier method defines a framework for labeling financial data based on the outcome of trade-like events within a specified time window.  

At the start of an event window, three barriers are set:  

1. **Upper horizontal barrier – profit-taking level**  
2. **Lower horizontal barrier – stop-loss level**  
3. **Vertical barrier – maximum holding period**  

The horizontal barriers are calculated relative to the price at the event’s start and scaled by the estimated volatility of the time series:  

- **Upper barrier**: `Pₜ * (1 + σ * mₚ)`  
- **Lower barrier**: `Pₜ * (1 - σ * mₛ)`  

Where:  
- `Pₜ` = price at the event start  
- `σ` = volatility estimate  
- `mₚ` = profit multiple  
- `mₛ` = stop multiple  

The volatility (`σ`) is typically computed as an exponentially weighted moving standard deviation of log returns to capture changes in market dynamics.  

The **vertical barrier** is a fixed time horizon (e.g., 20 bars) after which the event expires if neither horizontal barrier is hit.  

The method assigns a label to each event based on which barrier the price touches first:  

- `+1` if the upper barrier is breached (profitable movement)  
- `-1` if the lower barrier is crossed (loss)  
- `0` if neither horizontal barrier is hit before the vertical barrier (neutral outcome)  

Prado describes this approach as:  

> “A generalization of fixed-time horizon labeling, where early exits are allowed if prices hit specified thresholds before the end of the event window.”

This approach accounts for volatility and provides a more realistic labeling mechanism for financial time series, where price paths are not uniformly distributed over time.  
 

### 1.2 Event Definition and CUSUM Filter

A critical component of the Triple Barrier Method is the definition of **event windows**, which determine the points in time where the labeling process begins.  

In financial time series, consecutive events often overlap, leading to label leakage and dependence between samples. To address this, López de Prado proposes using a **Cumulative Sum (CUSUM) filter** to identify significant price movements and define non-overlapping event windows.  

The CUSUM filter detects structural shifts in the price series by accumulating incremental price changes until they exceed a specified threshold. As Prado explains:  

> “The CUSUM filter is designed to avoid triggering a new event for small movements, ensuring that only meaningful price changes initiate labeling windows.”

By applying the CUSUM filter, the dataset is reduced to a series of timestamps where significant directional changes occur, each serving as the initial timestamp for the triple barrier logic. This approach mitigates noise in high-frequency data and creates more reliable and independent labels for machine learning.

### 1.3 Extending the Theory: Time-Decayed Labeling

While the Triple Barrier Method provides a robust framework for labeling, its **hard assignment** of `+1`, `-1`, or `0` does not fully capture the dynamics of financial markets. Two paths that hit the same barrier may differ greatly in how quickly they do so, which is critical information for risk and trade management.  

This project introduces an extension: **Time to barrier modification**.  

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

---

## 2. Methodology

### 2.1 Data upload and labelling

When a user uploads financial data to the system, the file is ingested and parsed into a structured format. Each row is represented internally as a `PreprocessedRow` object, capable of holding a set of features. This flexibility allows future extension to datasets with additional indicators such as volume, technical signals, or macroeconomic variables. In this report, however, we focus on the timestamp and price columns. This simplification keeps the focus on exploring the labeling methodology itself, without the confounding effects of additional features.

After the dataset is loaded, the system presents an configuration dialog where users can adjust the parameters of the barrier labeling process. These parameters are defined in the `BarrierConfig` structure and control how events are detected and labeled:

- **profit_multiple** – Sets the threshold for profit-taking. Determines how much the price must rise (relative to volatility) before a profit barrier is triggered.  
- **stop_multiple** – Sets the threshold for stop-loss. Determines how much the price must fall (relative to volatility) before a stop barrier is triggered.  
- **vertical_window** – Defines the maximum holding period for an event. If neither profit nor stop barrier is hit, the event closes after this many periods.  
- **use_cusum** – Enables the CUSUM filter for event detection. When true, only events passing the CUSUM threshold are considered.  
- **cusum_threshold** – The sensitivity threshold for the CUSUM filter. Higher values mean only larger, more significant events are detected.  
- **labeling_type** – Selects the labeling method:  
  - Hard: Assigns discrete labels (+1, -1, 0) based on which barrier is hit first.  
  - TTBM: Assigns continuous labels using a decay function, reflecting both direction and timing.  
- **ttbm_decay_type** – Chooses the decay function for TTBM labeling, controlling how quickly the label magnitude decays over time (Exponential, Linear, or Hyperbolic). In this implementation, the decay parameters (`lambda`, `alpha`, `beta`) are hardcoded to fixed values. An extension would be to optimise these parameters.

The interaction between the CUSUM filter and TTBM is particularly important. When `use_cusum` is enabled, the system first applies the CUSUM filter to identify points in the price series where significant structural changes occur. Only these filtered points are used as candidate events for labeling. This reduces noise and prevents the TTBM from being applied indiscriminately across the entire dataset. TTBM then assigns labels to these events based on which barrier is breached first and adjusts label magnitudes using the selected decay function. If `use_cusum` is disabled, TTBM evaluates all rows in the dataset as potential events, which may result in a denser set of labels but with increased sensitivity to noise.

Once configured, the system processes the data, calculates the barriers for each event, and assigns labels. For each event, the output includes the start and end timestamps, entry and exit prices, hard label, TTBM label (if applicable), time ratio, and decay factor. These results can be visualized on charts to show how the labeling aligns with the price series, helping users evaluate and fine-tune their configuration.

This workflow ensures a consistent, traceable process from raw data ingestion to fully labeled datasets, ready for downstream machine learning or trading strategy development.

### 2.2 Feature extraction and machine learning

Once the financial data has been cleaned and preprocessed, the system moves on to feature extraction, transforming raw time series inputs into structured vectors that characterize the market environment around each event. This step draws on statistical measures such as volatility, returns, and other domain-specific indicators to build a compact, informative representation of the data. The `FeatureExtractor` class handles this process, ensuring that each event is encoded as a vector suitable for classification or regression tasks.

With these feature vectors prepared, the system applies the configured labeling method—either hard barriers or the time to barrier modification (TTBM). Hard barrier labeling assigns discrete values (+1, -1, or 0) based on which threshold—profit, stop, or maximum holding period—is hit first. In contrast, TTBM uses a decay function to produce continuous labels, capturing not only the direction of price movement but also the timing and confidence associated with barrier hits.

The labeled feature vectors are then fed into the machine learning pipeline, where models are trained and evaluated. Crucially, the aim of this step is not to maximize predictive accuracy or to fully model the complex dynamics of financial markets. Instead, it is designed to compare the relative effectiveness of TTBM labels versus hard barrier labels. The codebase uses XGBoost as the core machine learning model for both hard barrier and TTBM approaches. For hard barriers XGBoost is configured for classification tasks, using the "multi:softmax" objective, allowing the model to predict one of the 3 defined outcomes. For TTBM XGBoost is set up for regression with the "reg:squarederror" objective, this allows the model to predict anywhere within the `[-1, 1]` range. The data split type for training and evaluation is hardcoded as chronological, meaning the data is divided into training, validation, and test sets in temporal order to prevent lookahead bias. The pipeline supports feature preprocessing and hyperparameter tuning, but the underlying model for both approaches remains XGBoost, with the main difference being the objective function and label type (classification for hard barriers, regression for TTBM).

Various features are available within the model to act as the explanatory variables for the training. These are all functions of price given the nature of the data being used and include return, moving averages and volatility measures. As mentioned previously, this project looks to show the relative performance of the two methods, so we will not go into detail on the features used.

---

---

## 3. Output and results

In this section, we compare the performance of hard barrier labels and time to barrier modification (TTBM) labels when used to train identical machine learning models. By evaluating the resulting trading signals and portfolio metrics, we highlight the practical differences between these two labeling approaches. We use illustrative data generated using a Brownian motion process which can be tailored to show different market scenarions and the output of the barrier types in them. Multiple simplifying assumptions are also made when looking at the data, however all of these could be dealt with as an extension to the project:

- Data is input as by minute data, however this could be generalised to any interval over which you can trade in and out of an asset
- No transaction costs or issues with being able to go long or short the asset

The first dataset is for a hypothetical asset over a week, with prices every minute. There is no clear trend up or down and there is a fair amount of volatility. To begin with, the hard barrier implementation is applied to the dataset setting the vertical barrier at 30 intervals (30 minutes in this case). The profit and stop-loss multiples are set to be the same and trigger when there is a spike equal to 3 times the volatility at the start of the period. Given the number of events, the performance is hidden slightly, but we can see the green dots showing in periods of increase and then red in downturn. In periods where there is less certain movement, we see the blue dots representing the vertical barrier being hit.

<img width="1762" height="575" alt="image" src="https://github.com/user-attachments/assets/a85f0559-aacd-4abc-b994-b3768763884b" />

The same settings are then applied to generate the TTBM labels. The chart is the same in that green dots represent the profit barrier being hit and the red represent the stop loss being hit. However, now the shade and size of the dots represent the decay experienced before the barrier was hit. The blue vertical baarier hits seen previously now show as small white dots. A similar story is told, however we can immediately see more detail

<img width="1792" height="497" alt="image" src="https://github.com/user-attachments/assets/5e939211-7eee-4625-9ebf-ef34a97f0712" />




