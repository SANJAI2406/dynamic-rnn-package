# DYNAMIC FREQUENCY PREDICTION SYSTEM - USER GUIDE

## 📋 Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Input Data Format](#input-data-format)
4. [Model Selection Logic](#model-selection-logic)
5. [Using the System](#using-the-system)
6. [Understanding Results](#understanding-results)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This enhanced system supports **TWO prediction modes**:

### **Mode 1: SCALAR (Existing)**
- **Use case**: Static outputs (single value per configuration)
- **Model**: OptimizedSVR (Support Vector Regression)
- **Example**: "What is the transmission error for this design?"

### **Mode 2: DYNAMIC (NEW)**
- **Use case**: Frequency-dependent outputs (4000 values per configuration)
- **Model**: DynamicFrequencyModel (GPR + PCA)
- **Example**: "What is the transmission error across 0-4000 Hz for this design?"

**The system AUTOMATICALLY detects** which mode to use based on your input file!

---

## 🏗️ System Architecture

### Dynamic Model Architecture (GPR + PCA)

```
TRAINING PHASE:
┌─────────────────────────────────────────────────────────────┐
│  1. Load Data                                               │
│     - 165 configurations × 4000 frequencies × 4 outputs     │
│     = 660,000 total data points                             │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. PCA Dimensionality Reduction                            │
│     - Input:  [165 × 16,000] (4000 freq × 4 outputs)       │
│     - Output: [165 × 50] latent codes                       │
│     - Captures >99% of variance in 50 components            │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Train GPR Models                                        │
│     - 50 separate Gaussian Process models                   │
│     - Each: 14 inputs → 1 latent component                  │
│     - Provides uncertainty quantification                    │
└─────────────────────────────────────────────────────────────┘

PREDICTION PHASE:
┌─────────────────────────────────────────────────────────────┐
│  New Design [1 × 14 parameters]                             │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  50 GPR models predict latent codes [1 × 50]               │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Inverse PCA transform [1 × 50] → [1 × 16,000]             │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Reshape to [4000 frequencies × 4 outputs]                  │
│  = Complete frequency response prediction!                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Input Data Format

### DYNAMIC Data Format (Frequency-Dependent)

Your file must have this exact structure:

#### **Column Layout:**
```
Column 0:      Frequency (Hz)
Columns 1-14:  Input parameters (14 design parameters)
Columns 15+:   Output parameters (4 outputs)
```

#### **Row Structure:**
```
Configuration 1:
  Row 1:    1 Hz    | [14 inputs - CONSTANT] | [4 outputs - vary]
  Row 2:    2 Hz    | [14 inputs - CONSTANT] | [4 outputs - vary]
  Row 3:    3 Hz    | [14 inputs - CONSTANT] | [4 outputs - vary]
  ...
  Row 4000: 4000 Hz | [14 inputs - CONSTANT] | [4 outputs - vary]

Configuration 2:
  Row 4001: 1 Hz    | [14 inputs - NEW]      | [4 outputs - vary]
  Row 4002: 2 Hz    | [14 inputs - NEW]      | [4 outputs - vary]
  ...
  Row 8000: 4000 Hz | [14 inputs - NEW]      | [4 outputs - vary]

... (repeat for all 165 configurations)

Total rows: 165 × 4000 = 660,000 rows
```

#### **Example CSV (first few rows):**
```csv
Frequency,Lead_Crowning_Pinion,Lead_Crowning_Gear,...,Transmission_Error,Axial_Force,...
1.0,6.061,6.498,...,0.1485,2.528,...
2.0,6.061,6.498,...,0.1502,2.534,...
3.0,6.061,6.498,...,0.1519,2.541,...
...
4000.0,6.061,6.498,...,0.2145,2.987,...
1.0,7.809,7.318,...,0.2042,6.629,...  ← New configuration
2.0,7.809,7.318,...,0.2058,6.637,...
```

#### **Key Requirements:**
✅ First column MUST be frequency
✅ Frequency MUST repeat identically for each configuration
✅ Input parameters (columns 1-14) MUST be constant within each 4000-row block
✅ Output parameters vary with frequency
✅ No missing values (NaN)

#### **Flexible Parameters:**
- **Frequency range**: Can be 0-5000 Hz, 10-1000 Hz, etc.
- **Frequency resolution**: Can be 0.5 Hz, 1 Hz, 10 Hz, etc.
- **Number of points**: Can be 100, 1000, 10000, etc.
- **Spacing**: Can be linear or logarithmic

---

### SCALAR Data Format (Static)

For non-frequency-dependent data:

```csv
Lead_Crowning_Pinion,Lead_Crowning_Gear,...,Transmission_Error,Axial_Force,...
6.061,6.498,...,0.1485,2.528,...
7.809,7.318,...,0.2042,6.629,...
...
```

Total rows: 165 (one per configuration)

---

## 🤖 Model Selection Logic

The system **automatically** determines the model type:

```python
if "Frequency resets detected":
    mode = DYNAMIC
    model = DynamicFrequencyModel(GPR + PCA)
else:
    mode = SCALAR
    model = OptimizedSVR
```

**Detection Method:**
1. Load first column (should be frequency)
2. Calculate `diff()` between consecutive values
3. If any negative jumps detected → DYNAMIC mode
4. Otherwise → SCALAR mode

---

## 🚀 Using the System

### Step 1: Launch Application

```bash
python FINAL_ENHANCED_WITH_DYNAMIC_RNN.py
```

On first run, the system automatically generates a template file:
- **Path**: `/home/claude/dynamic_data_template.csv`
- **Contents**: Example dynamic data with 165 configs × 4000 frequencies

### Step 2: Load Your Data

1. Click **"📁 Load Data File"**
2. Select your CSV or Excel file
3. System displays data info:
   ```
   Data Type: DYNAMIC
   Configurations: 165
   Input Parameters: 14
   Frequency Points: 4000
   Freq Range: 1.0 - 4000.0 Hz
   Output Parameters: 4
   ```

### Step 3: Select Outputs to Train

In the **Output Channels** list:
- Each row shows: Number, Name, Model Type, Build checkbox, R²
- **Model Type** is auto-detected:
  - "Dynamic (GPR+PCA)" for frequency data
  - "Scalar (SVR)" for static data
- Check the outputs you want to train
- Default: All outputs selected

### Step 4: Build Models

1. Click **"🔨 Build Models"**
2. Training progress displays:
   ```
   === Building DYNAMIC Models (GPR + PCA) ===
   
   Training model for: Transmission_Error
   Input: 165 configs × 14 params
   Output: 4000 freq points × 1 outputs
   Flattened output shape: (165, 4000)
   Applying PCA (target: 50 components)...
   PCA Results:
     - Actual components: 50
     - Explained variance: 99.85%
   Training 50 GPR models...
     Training GPR 1/50...
     Training GPR 11/50...
     ...
   
   Model Performance:
     Overall R²: 0.9876
     R² per frequency - Mean: 0.9823, Std: 0.0142
   ```

3. Models are saved automatically

### Step 5: Visualize Results

Navigate to **"Visualization"** tab:

#### Available Plot Types:

**1. Frequency Response**
- Shows actual vs predicted frequency responses
- Displays first 5 configurations
- Each subplot: one configuration
- Blue line: Actual data
- Red dashed: Predicted

**2. Component Importance**
- Top plot: Bar chart of PCA component variance
- Bottom plot: Cumulative variance
- Shows how many components capture 95% of variance

**3. Uncertainty**
- Similar to Frequency Response
- Adds shaded confidence intervals
- Shows ±2σ prediction uncertainty
- Quantifies model confidence

**4. Prediction vs Actual**
- Scatter plot of all predictions
- Perfect predictions lie on diagonal line
- Spread indicates prediction error
- Color density shows data concentration

### Step 6: Analyze Performance

Check the **R² values** in the Output Channels list:

- **R² > 0.95**: Excellent fit
- **R² > 0.90**: Good fit
- **R² > 0.80**: Acceptable fit
- **R² < 0.80**: May need more data or model tuning

For dynamic models, also check:
- **R² per frequency**: Should be consistent across frequency range
- **Explained variance**: Should capture >95% with 50 components

---

## 📊 Understanding Results

### Dynamic Model Outputs

For each configuration, you get a **complete frequency response**:

```python
# Example: Predict for a new design
new_design = np.array([[6.5, 7.0, 11.2, -6.1, ...]])  # 14 parameters

# Prediction
freq_response = model.predict(new_design)
# Shape: (1, 4000, 4) = 1 config × 4000 frequencies × 4 outputs

# Extract specific output
transmission_error_vs_freq = freq_response[0, :, 0]  # Shape: (4000,)

# You now have TE for every frequency from 1-4000 Hz!
```

### With Uncertainty:

```python
freq_response, uncertainty = model.predict(new_design, return_std=True)

# At 500 Hz:
freq_idx = 499  # 0-indexed
te_at_500hz = freq_response[0, freq_idx, 0]
uncertainty_at_500hz = uncertainty[0, freq_idx, 0]

print(f"TE at 500 Hz: {te_at_500hz:.4f} ± {2*uncertainty_at_500hz:.4f}")
# Output: "TE at 500 Hz: 0.2145 ± 0.0087"
```

### Key Metrics

**Overall R²**: Measures fit quality across all frequencies
- Treats all 4000 frequency points as independent predictions
- Weighted by actual frequency response variance

**R² per Frequency**: 
- Individual R² for each frequency point
- Identifies problematic frequency ranges
- Example: Low R² at resonance peaks may indicate underfitting

**Explained Variance**:
- Shows how much info each PCA component captures
- First component typically captures 40-60% of variance
- First 10 components usually capture 90%+

---

## 🔧 Troubleshooting

### Issue 1: "Data structure inconsistent!"

**Cause**: Total rows ≠ n_configs × n_freq

**Fix**:
```python
# Check your data
n_configs = 165
n_freq = 4000
expected_rows = n_configs * n_freq  # Should be 660,000

# Verify in Excel/CSV:
# - Count total rows
# - Count unique combinations of input parameters
# - Ensure frequency column repeats exactly n_configs times
```

### Issue 2: Low R² (<0.80)

**Possible causes**:
1. **Insufficient data**: 165 configs may be borderline for complex systems
   - Solution: Collect more training data (300-500 configs ideal)

2. **Too few PCA components**: 50 may not capture all variance
   - Solution: Increase n_components to 100 or 150

3. **High noise**: Frequency responses have high measurement noise
   - Solution: Increase GPR white noise kernel parameter

4. **Nonlinear patterns**: GPR kernel may be inadequate
   - Solution: Try different kernels (Matern, RationalQuadratic)

### Issue 3: Memory Error

**Cause**: Large datasets (1M+ rows) may exceed RAM

**Fix**:
```python
# Option 1: Reduce frequency resolution
# Instead of 4000 points, use 1000 points (every 4th point)
n_freq_reduced = 1000

# Option 2: Reduce PCA components
n_components = 30  # Instead of 50

# Option 3: Process outputs separately
# Train one model at a time instead of all 4 together
```

### Issue 4: Very Slow Training

**Cause**: GPR scales O(n³) with training samples

**Expected times**:
- 165 configs: ~5-10 minutes total
- 500 configs: ~30-60 minutes total

**If slower**:
- Reduce `n_restarts_optimizer` from 3 to 1
- Reduce PCA components
- Use a faster computer with more RAM

### Issue 5: Unrealistic Predictions

**Symptoms**:
- Negative forces
- Extremely smooth responses (no peaks)
- Values outside training range

**Fixes**:
1. **Check extrapolation**: Are you predicting outside training bounds?
   ```python
   # Check if new design is within training range
   for i, param in enumerate(input_names):
       new_val = new_design[0, i]
       train_min = X[:, i].min()
       train_max = X[:, i].max()
       if new_val < train_min or new_val > train_max:
           print(f"WARNING: {param} = {new_val} outside training range [{train_min}, {train_max}]")
   ```

2. **Add physical constraints**: Post-process predictions
   ```python
   # Ensure non-negative forces
   freq_response[freq_response < 0] = 0
   ```

3. **Increase training data**: Cover more of design space

---

## 📈 Performance Optimization

### For Faster Training:

```python
# 1. Reduce PCA components
model = DynamicFrequencyModel(n_components=30)  # Instead of 50

# 2. Reduce GPR restarts
# In DynamicFrequencyModel.fit(), change:
n_restarts_optimizer=1  # Instead of 3

# 3. Use coarser frequency grid
# In your data generation, use 2 Hz steps instead of 1 Hz
# This halves the data size: 2000 points instead of 4000
```

### For Better Accuracy:

```python
# 1. Increase PCA components
model = DynamicFrequencyModel(n_components=100)

# 2. Use more sophisticated kernel
from sklearn.gaussian_process.kernels import Matern
kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.1)
model = DynamicFrequencyModel(n_components=50, gpr_kernel=kernel)

# 3. Collect more training data
# Aim for 300-500 configurations instead of 165
```

---

## 🎓 Technical Deep Dive

### Why PCA?

**Problem**: 
- 4000 frequency points × 4 outputs = 16,000 dimensional output
- GPR cannot handle 16,000 outputs directly (memory & computational limits)

**Solution**:
- PCA finds 50 "latent modes" that capture 99%+ of the variance
- These modes represent the dominant patterns in frequency responses
- Example modes:
  - Mode 1: Overall amplitude scaling
  - Mode 2: Shift in resonance frequency
  - Mode 3: Damping factor
  - Modes 4-50: Fine details

**Result**:
- Instead of predicting 16,000 values, predict 50 latent codes
- Reconstruct the full 16,000 values from these 50 codes
- 320× reduction in dimensionality!

### Why GPR (Gaussian Process)?

**Advantages**:
1. **Uncertainty quantification**: Tells you prediction confidence
2. **Smooth interpolation**: Perfect for continuous functions like frequency responses
3. **Small data friendly**: Works well with 100-500 samples
4. **No architecture tuning**: Unlike neural networks, minimal hyperparameters

**Disadvantages**:
1. **Computational cost**: O(n³) scaling
2. **Memory intensive**: Stores full covariance matrix
3. **Limited extrapolation**: Poor predictions far from training data

### Alternative Approaches (Future Work)

**1. Neural Networks (1D-CNN)**
- Better for large datasets (1000+ configs)
- Faster prediction
- Requires more careful tuning

**2. Random Forests**
- Simpler, more interpretable
- No uncertainty quantification
- May struggle with smooth functions

**3. Reduced Order Modeling**
- Physics-informed basis functions
- Requires domain expertise
- Can work with very little data

---

## 📦 File Outputs

The system generates:

1. **dynamic_data_template.csv**
   - Example template with synthetic data
   - Use as reference for formatting your own data

2. **Models** (in memory during session)
   - Trained DynamicFrequencyModel objects
   - Can be saved with pickle for reuse

3. **Visualizations** (exportable)
   - Frequency response plots
   - PCA analysis plots
   - Uncertainty plots

---

## 🆘 Support & Contact

For issues or questions:

1. Check this guide first
2. Review the troubleshooting section
3. Examine the example template file
4. Check console output for error messages

Common Error Messages:

```
"Data structure inconsistent!"
→ Row count doesn't match expected n_configs × n_freq

"No frequency resets detected"
→ System thinks this is scalar data; check frequency column

"Model must be fitted before prediction"
→ Train models before trying to predict

"Insufficient data for PCA"
→ Need at least 50 configurations for 50 components
```

---

## ✅ Quick Checklist

Before running:
- [ ] Data file is .csv or .xlsx format
- [ ] First column contains frequency values
- [ ] Frequency repeats identically for each configuration
- [ ] Input parameters (14 columns) are constant within each config
- [ ] Output parameters vary with frequency
- [ ] No missing values (NaN) in data
- [ ] Total rows = n_configs × n_freq

For best results:
- [ ] At least 165 configurations (300+ ideal)
- [ ] Frequency range covers your application needs
- [ ] Input parameters span the full design space
- [ ] Multiple configs explore parameter combinations
- [ ] Data quality checked (no outliers or errors)

---

**Version**: 1.0
**Last Updated**: November 2025
**Author**: Enhanced RNN System Development Team

