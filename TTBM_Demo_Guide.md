# TTBM Demo Script
# This script demonstrates how to use the Time-to-Barrier Modification functionality

## Overview
The Time-to-Barrier Modification (TTBM) extends the traditional Triple Barrier Method by incorporating the time taken to reach the first barrier as a scaling factor for the label.

## Key Features

### 1. **UI Controls in Barrier Configuration Dialog**
- **Labeling Type**: Choose between "Hard Barrier" and "TTBM (Time-to-Barrier Modification)"
- **TTBM Decay Type**: 
  - Exponential Decay: f(t) = e^(-λ * t_b/t_v)
  - Linear Decay: f(t) = 1 - α * t_b/t_v  
  - Hyperbolic Decay: f(t) = 1 / (1 + β * t_b/t_v)
- **Decay Parameters**:
  - Lambda (λ): Exponential decay rate (higher = faster decay)
  - Alpha (α): Linear decay factor (0-1, higher = more decay)
  - Beta (β): Hyperbolic decay steepness (higher = faster decay)

### 2. **Enhanced Visualization Options**
- **TTBM Time Series**: Price chart with events colored by TTBM label strength and sized by speed
- **TTBM Distribution**: Histogram showing distribution of continuous TTBM labels (30 bins)
- **Color Coding**: 
  - Red/Purple: Negative labels (stop losses)
  - Purple: Neutral labels 
  - Green: Positive labels (profits)
  - Intensity: Based on time-to-barrier (faster = more intense)
- **Marker Size**: Larger markers = faster barrier hits

### 3. **TTBM Label Calculation**
```
Original TBM Label: y ∈ {-1, 0, +1}
Time to Barrier: t_b (periods elapsed)
Vertical Barrier: t_v (max holding period)
Decay Function: f(t_b/t_v)

TTBM Label: y' = y * f(t_b/t_v)
```

### 4. **Example Scenarios**
- **Quick Profit Hit**: 
  - Hard Label: +1
  - Time Ratio: 0.1 (hit barrier in 10% of max time)
  - Exponential Decay (λ=1): f(0.1) ≈ 0.905
  - TTBM Label: +0.905

- **Slow Stop Hit**:
  - Hard Label: -1  
  - Time Ratio: 0.8 (hit barrier in 80% of max time)
  - Exponential Decay (λ=1): f(0.8) ≈ 0.449
  - TTBM Label: -0.449

- **Vertical Barrier**: 
  - Hard Label: 0
  - Time Ratio: 1.0 (max time elapsed)
  - TTBM Label: 0 (always 0 regardless of decay)

## How to Use

### 1. **Setup TTBM Configuration**
1. Upload your CSV data (timestamp, price columns)
2. Click "Upload & Label Data"
3. In the Barrier Config Dialog:
   - Set "Labeling Type" to "TTBM (Time-to-Barrier Modification)"
   - Choose your preferred decay type (Exponential recommended)
   - Adjust decay parameters (default λ=1.0 works well)
   - Set profit/stop multiples and vertical window as usual

### 2. **Visualize Results**
1. After labeling, use the plot mode dropdown:
   - "TTBM Time Series": See events with color/size encoding
   - "TTBM Distribution": View the continuous label distribution
2. Compare with traditional hard labels using "Time Series" and "Histogram" modes

### 3. **Interpret Results**
- **TTBM labels close to ±1.0**: Strong, fast-moving signals
- **TTBM labels close to 0.0**: Weak signals or neutral outcomes
- **Large markers**: Quick barrier hits (high confidence)
- **Small markers**: Slow barrier hits (lower confidence)

## Benefits of TTBM
1. **Continuous Targets**: Better suited for regression models
2. **Signal Strength**: Distinguishes between strong and weak signals
3. **Risk-Aware**: Naturally favors faster, more decisive price movements
4. **ML Compatible**: Drop-in replacement for traditional hard labels

## Technical Implementation
- New `TTBMLabeler` class implementing `IBarrierLabeler` interface
- Extended `LabeledEvent` structure with TTBM fields:
  - `ttbm_label`: Continuous label in [-1, +1]
  - `time_to_barrier_ratio`: Normalized time to first barrier
  - `decay_factor`: Applied decay function value
- Enhanced visualization with gradient coloring and size scaling
- Comprehensive test suite with various decay scenarios
