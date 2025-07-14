# Triple Barrier Labelling with Time-Decayed Labels

## Summary

This project implements and extends the triple barrier method from *Advances in Financial Machine Learning* (López de Prado, 2018). It provides a framework for labelling financial time series by detecting significant price movements (using a CUSUM filter) and assigning labels based on which of three barriers (profit, stop-loss, or holding period) is hit first.  

It extends the original method with time-decayed labels, which start at `+1` or `-1` when a barrier is hit immediately and decay toward `0` as time passes without hitting any barrier. This allows models to learn both directional bias and signal confidence over time, enabling more nuanced decision-making in trading strategies.

The project allows for the upload of time series data (timestamp, price), the application of the triple barrier logic, computation of features, and the training of predictive models. This short report aims to then compare the relative performance of the original triple barrier method with the time to barrier modification introduced in this project.

---

## 1. Introduction

Financial time series are inherently noisy, non-stationary, and prone to sudden regime changes, which makes supervised machine learning particularly challenging in this domain. As Marcos López de Prado highlights in *Advances in Financial Machine Learning (2018)*:

> “Labels are the weakest link in supervised learning, and in finance, traditional labelling techniques are often inadequate.”

To address this, Prado introduced the triple barrier method, which improves upon fixed time horizon labelling by allowing for dynamic exit conditions based on price movements and holding periods. This project implements the triple barrier Method as a pre-processing step for machine learning and extends it by incorporating a time-decayed label methodology. The goal is to provide more informative targets for predictive modeling, enabling refined position sizing and risk management.

### 1.1 The Triple Barrier Method

The triple barrier method defines a framework for labelling financial data based on the outcome of trade-like events within a specified time window.  

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

The volatility (`σ`) is computed as an exponentially weighted moving standard deviation of log returns to capture changes in market dynamics.  

The **vertical barrier** is a fixed time horizon (e.g., 20 bars) after which the event expires if neither horizontal barrier is hit.  

The method assigns a label to each event based on which barrier the price touches first:  

- `+1` if the upper barrier is breached (profitable movement)  
- `-1` if the lower barrier is crossed (loss)  
- `0` if neither horizontal barrier is hit before the vertical barrier (neutral outcome)  

Prado describes this approach as:  

> “A generalization of fixed-time horizon labelling, where early exits are allowed if prices hit specified thresholds before the end of the event window.”

This approach accounts for volatility and provides a more realistic labelling mechanism for financial time series, where price paths are not uniformly distributed over time.  

 ---

### 1.2 Event Definition and CUSUM Filter

A critical component of the Triple Barrier Method is the definition of event windows, which determine the points in time where the labelling process begins. In financial time series, consecutive events often overlap, leading to label leakage and dependence between samples. To address this, López de Prado proposes using a cumulative sum (CUSUM) filter to identify significant price movements and define non-overlapping event windows.  

The CUSUM filter detects structural shifts in the price series by accumulating incremental price changes until they exceed a specified threshold. As Prado explains:  

> “The CUSUM filter is designed to avoid triggering a new event for small movements, ensuring that only meaningful price changes initiate labelling windows.”

By applying the CUSUM filter, the dataset is reduced to a series of timestamps where significant directional changes occur, each serving as the initial timestamp for the triple barrier logic. This approach mitigates noise in high-frequency data and creates more reliable and independent labels for machine learning.

---

### 1.3 Extending the Theory: Time-Decayed Labelling

While the Triple Barrier Method provides a robust framework for labelling, its hard assignment of `+1`, `-1`, or `0` does not fully capture the dynamics of financial markets. Two paths that hit the same barrier may differ greatly in how quickly they do so, which is critical information for risk and trade management.  

This project introduces an extension: the time to barrier modification.  

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

When a user uploads financial data to the system, the file is ingested and parsed into a structured format. Each row is represented internally as a `PreprocessedRow` object, capable of holding a set of features. This flexibility allows future extension to datasets with additional indicators such as volume, technical signals, or macroeconomic variables. In this report, however, the focus is on the timestamp and price columns. This simplification keeps the focus on exploring the labelling methodology itself, without the confounding effects of additional features.

After the dataset is loaded, the system presents an configuration dialog where users can adjust the parameters of the barrier labelling process. These parameters are defined in the `BarrierConfig` structure and control how events are detected and labeled:

- **profit_multiple** – Sets the threshold for profit-taking. Determines how much the price must rise (relative to volatility) before a profit barrier is triggered.  
- **stop_multiple** – Sets the threshold for stop-loss. Determines how much the price must fall (relative to volatility) before a stop barrier is triggered.  
- **vertical_window** – Defines the maximum holding period for an event. If neither profit nor stop barrier is hit, the event closes after this many periods.  
- **use_cusum** – Enables the CUSUM filter for event detection. When true, only events passing the CUSUM threshold are considered.  
- **cusum_threshold** – The sensitivity threshold for the CUSUM filter. Higher values mean only larger, more significant events are detected.  
- **labelling_type** – Selects the labelling method:  
  - Hard: Assigns discrete labels (+1, -1, 0) based on which barrier is hit first.  
  - TTBM: Assigns continuous labels using a decay function, reflecting both direction and timing.  
- **ttbm_decay_type** – Chooses the decay function for TTBM labelling, controlling how quickly the label magnitude decays over time (exponential, linear, or hyperbolic). In this implementation, the decay parameters (`lambda`, `alpha`, `beta`) are hardcoded to fixed values. An extension would be to optimise these parameters.

The interaction between the CUSUM filter and TTBM is particularly important. When `use_cusum` is enabled, the system first applies the CUSUM filter to identify points in the price series where significant structural changes occur. Only these filtered points are used as candidate events for labelling. This reduces noise and prevents the TTBM from being applied indiscriminately across the entire dataset. TTBM then assigns labels to these events based on which barrier is breached first and adjusts label magnitudes using the selected decay function. If `use_cusum` is disabled, TTBM evaluates all rows in the dataset as potential events, which may result in a denser set of labels but with increased sensitivity to noise.

Once configured, the system processes the data, calculates the barriers for each event, and assigns labels. For each event, the output includes the start and end timestamps, entry and exit prices, hard label, TTBM label (if applicable), time ratio, and decay factor. These results can be visualized on charts to show how the labelling aligns with the price series, helping users evaluate and fine-tune their configuration.

This workflow ensures a consistent, traceable process from raw data ingestion to fully labeled datasets, ready for downstream machine learning or trading strategy development.

---

### 2.2 Feature extraction and machine learning

Once the financial data has been cleaned and preprocessed, the system moves on to feature extraction, transforming raw time series inputs into structured vectors that characterize the market environment around each event. This step draws on statistical measures such as volatility, returns, and other domain-specific indicators to build a compact, informative representation of the data. The `FeatureExtractor` class handles this process, ensuring that each event is encoded as a vector suitable for classification or regression tasks.

With these feature vectors prepared, the system applies the configured labelling method—either hard barriers or the time to barrier modification (TTBM). Hard barrier labelling assigns discrete values (+1, -1, or 0) based on which threshold—profit, stop, or maximum holding period—is hit first. In contrast, TTBM uses a decay function to produce continuous labels, capturing not only the direction of price movement but also the timing and confidence associated with barrier hits.

The labeled feature vectors are then fed into the machine learning pipeline, where models are trained and evaluated. Crucially, the aim of this step is not to maximize predictive accuracy or to fully model the complex dynamics of financial markets. Instead, it is designed to compare the relative effectiveness of TTBM labels versus hard barrier labels. The codebase uses XGBoost as the core machine learning model for both hard barrier and TTBM approaches. For hard barriers XGBoost is configured for classification tasks, using the "multi:softmax" objective, allowing the model to predict one of the 3 defined outcomes. For TTBM XGBoost is set up for regression with the "reg:squarederror" objective, this allows the model to predict anywhere within the `[-1, 1]` range. The data split type for training and evaluation is hardcoded as chronological, meaning the data is divided into training, validation, and test sets in temporal order to prevent lookahead bias. The pipeline supports feature preprocessing and hyperparameter tuning, but the underlying model for both approaches remains XGBoost, with the main difference being the objective function and label type (classification for hard barriers, regression for TTBM).

Various features are available within the model to act as the explanatory variables for the training. These are all functions of price given the nature of the data being used and include return, moving averages and volatility measures. As mentioned previously, this project looks to show the relative performance of the two methods, so this report will not go into detail on the features used.

---

---

## 3. Results: Comparing Hard Barrier and TTBM labelling Approaches

In this section, a comparison of the performance of hard barrier labels and the time to barrier modification (TTBM) labels when used to train identical machine learning models is performed. By evaluating the resulting trading signals and portfolio metrics, it highlights the practical differences between these two labelling approaches. Illustrative data generated from a Brownian motion process is used to provide data. This controlled setup allows us to explore the outputs of both barrier types in isolation, though it is important to note the following simplifying assumptions:

- **Data Granularity**: The input data consists of minute-by-minute price observations. However, the methodology generalizes naturally to other timeframes (e.g., hourly, daily) provided sufficient data resolution.  
- **Transaction Costs**: Assume no transaction costs, slippage, or market frictions, enabling a focus on the labelling mechanics rather than realistic trading conditions.  
- **Market Access**: Positions can be opened and closed freely without liquidity constraints.  
- **Volatility Estimation**: Volatility is measured using a rolling window of log returns and is assumed to be accurately estimated at each event start.  

The first dataset covers a hypothetical asset over a week, with prices sampled every minute. The simulated price path shows no strong trend but exhibits moderate volatility throughout.  

The hard barrier implementation applies a vertical barrier after 30 intervals (corresponding to a holding period of 30 minutes) and sets equal profit and stop-loss multiples (five times the estimated volatility at each event’s start). THe multiples are set fairly high given the large amount of volatility shown. 

In the first chart, the labelling outcomes are visualized:

- **Green dots** mark events where the profit barrier was hit first.  
- **Red dots** indicate events where the stop-loss barrier was hit.  
- **Blue dots** denote events where neither horizontal barrier was breached, and the vertical barrier closed the position.  

The following key estimations can be made: 
- Profit hits (green) cluster in periods of upward price movement, while stop-losses (red) dominate during downtrends.  
- Blue vertical barrier hits tend to appear during periods of sideways or low-volatility price action, where neither threshold was reached before the holding period expired.  
- The discrete labelling hides subtle differences in how quickly events reach their respective barriers. Two identical labels may represent very different dynamics (e.g., immediate profit hit vs. late profit hit).  

<img width="1770" height="522" alt="image" src="https://github.com/user-attachments/assets/5e42e974-dd7c-4800-b223-05db7cd34a92" />

---

The same dataset and barrier parameters are then used to generate **TTBM labels**. Here, the labelling process retains the same barrier definitions, but instead of assigning a discrete label at the point of barrier breach, it applies a **decay function**  to encode both the direction and time-to-barrier into a continuous value. The user can select between exponential, linear or hyperbolic decay (hyperbolic was chosen here).

In the second chart:

- **Green dots** still represent profit barrier hits, and **red dots** indicate stop-loss hits.  
- Dot size and opacity now encode the **decay factor**, with larger and brighter dots representing events where the barrier was reached quickly and smaller, faded dots indicating slower hits.  
- **White dots** replace the blue vertical barrier markers. These represent events that expired without hitting either horizontal barrier, assigned a decayed value reflecting the elapsed time relative to the vertical barrier.  

The following key observations can be made:
- Compared to the hard barrier labels, TTBM captures a richer picture of event dynamics. Early barrier hits stand out as prominent markers, while late barrier hits are visually subdued.  
- In flat or choppy market conditions, most events decay to smaller values, suggesting lower confidence in the directional signal.  
- Vertical barrier expirations (white dots) clearly dominate periods of sideways price movement, reinforcing the idea that the market offered few actionable opportunities.  

<img width="1768" height="477" alt="image" src="https://github.com/user-attachments/assets/7608558e-a1f8-42ab-8c46-c33504baea96" />

---

Continuing we look to compare the two strategies using machine learning and a hypothetical portfolio simulation. The features chosen as the explanatory variables for the machine learning models are the exponentially-weighted moving average of the 10 intervals prior to the event start and the trend in the price over the previous 10 intervals. These features do not aim to predict all of the variation in price, however provide enough explanation to allow us to compare the two methodologies.

The simulation interprets the signals predicted by the machine learning models according to the selected strategy: hard barrier signals are mapped to fixed long, short, or neutral positions, while regression outputs determine position sizes proportionally. For each event in the test split, the simulation executes trades based on the model’s signals and the actual asset returns, updating the portfolio’s capital accordingly. For a comprehensive evaluation, further robustness checks, out-of-sample validation across multiple timeframes, and consideration of real-world trading constraints would be necessary; however this initial analysis is useful for comparing the relative performance of the two methodologies.

For this scenario we assume initial capital of £1,000. The hard barrier methodology will take a fixed 25% (of capital) position long or short depending on the predicted signal, or not make a trade if the prediction is that the vertical barrier will be hit. For the TTBM barriers, the size of the trade will be linearly scaled by the strength of the signal (a signal of +0.5 will lead to a 12.5% position being taken). Signals with an absolute value of less than 0.25 are considered to be too weak to make a trade. 

The results in this case show that the TTBM methodology led to a higher return (2.24% vs 1.48%), however there are few more interesting observations. As expected, the two models almost always signal in the same direction given they were modelled using the same explanatory variables, however the changes in portfolio values under TTBM are a lot less volatile and unpredictable displaying a stronger upward trend. Even in the cases where both models made incorrect predictions (leading to a decrease in portfolio value), the signal given by TTBM was a lot weaker leading to a smaller drop in portfolio value.

<img width="1237" height="437" alt="image" src="https://github.com/user-attachments/assets/5bf2441f-de27-4631-8fbf-3e7258c2a445" />

---

### 3.1 Limitations of the analysis

- **Simplified trading strategy assumptions:**   The portfolio simulation uses a fixed position size (25% of portfolio) for hard barrier signals and a scaled position size for TTBM signals. This does not account for more sophisticated position sizing methods (e.g., Kelly criterion, volatility targeting).
- **Simplistic features**  To reduce complexity, the explanatory features used in the machine learning process are mainly functions of price. Introducing more sophisticated features could increase the explanatory power and reduce the chance of multicollinearity.
- **No transaction costs or slippage:**  The analysis assumes frictionless trading, ignoring bid-ask spreads, commissions, and slippage, which can materially impact real-world performance.
- **Hardcoded TTBM decay parameters:**  The decay functions (exponential, linear, hyperbolic) use fixed parameters (`lambda`, etc.), which may not be optimal for all assets or market conditions.
- **Single asset focus:**  Results are based on one asset or a limited set of assets. Findings may not generalize across different markets (e.g., equities, FX, crypto).
- **Limited barrier configurations explored:**  Only a few combinations of profit/stop multiples and vertical windows are tested. Different configurations might lead to different relative performances.
- **No out-of-sample validation:**  The same dataset is used for feature extraction, labelling, and simulation, which may lead to overfitting or optimistic performance estimates.
- **Simplistic volatility adjustment:**  Barrier thresholds are defined relative to volatility, but the volatility measure itself (e.g., rolling standard deviation) may not fully capture market dynamics.
- **Market regime shifts ignored:**  The analysis assumes stationary behavior, but financial markets are subject to regime changes that could impact the relative effectiveness of labelling methods.




