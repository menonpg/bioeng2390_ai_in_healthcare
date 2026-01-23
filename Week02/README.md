# Week 02: Feature Engineering & Statistical Analysis of EEG Signals
### BIOENG 2390: AI in Healthcare - Spring 2026

**Instructor:** Professor Prahlad Menon, PhD, PMP  
**University of Pittsburgh, Department of Bioengineering**

---

## 🎯 Week 02 Overview

This week we transition from basic signal processing to **feature engineering** and **statistical modeling**. We'll learn why simply looking at signal amplitude isn't enough to detect seizures, and how frequency-domain features can reveal patterns invisible in the time domain.

---

## 📺 Lecture Recordings & Notes

### Lecture 3 - January 20, 2026 (98 minutes)
**Focus:** Time-Frequency Analysis & Feature Engineering

- **[Watch Recording](https://fathom.video/share/2W4R9-EMc4yJhHeK9vsiZhzyJzw5KrZy)**

**Topics Covered:**
- Loading MATLAB .mat files in Python using `scipy.io`
- Understanding spectrograms and frequency decomposition
- Windowing strategies: 50% overlap with 1-second windows
- Feature engineering: Delta, Theta, Alpha, Beta frequency bands
- Statistical testing: Why mean values fail to distinguish seizure states
- T-test vs Wilcoxon rank-sum test
- Box plots and density plots for distribution comparison

**Key Insight:** 
> "The average normalized signal amplitude in seizure and non-seizure states is **not statistically different** (p=0.94), but the **shapes of their distributions are different** (Wilcoxon p=0.059). This means we need frequency-domain features to distinguish between these states!"

---

### Lecture 4 - January 22, 2026 (86 minutes)
**Focus:** Confusion Matrices, Regression Theory & OLS Derivation

- **[Watch Recording](https://fathom.video/share/otwZPqGCaqtb7yxzfrsbZ3DYK47agk5A)**
- **[Read Detailed Lecture Notes](Lecture04_Notes_Jan22_2026.md)** ← Complete transcript with all concepts

**Topics Covered:**
- Confusion matrices: TP, TN, FP, FN definitions and interpretation
- Performance metrics: Sensitivity, specificity, precision, accuracy
- 2SD vs 1SD control rules and threshold selection
- Sensitivity-precision tradeoff (COVID test analogy)
- Linear regression theory and matrix algebra
- OLS derivation: β = (AᵀA)⁻¹AᵀY
- Implementation in R (`lm` function) and MATLAB (matrix operations)
- Introduction to logistic regression with GLM
- Why single amplitude features fail for seizure detection

**Key Finding:**
> "Logistic regression using only normalizedValue as predictor has p=0.931 - NOT significant. This confirms we need frequency-domain features from windowed analysis!"

---

## 🎯 Week Learning Objectives

By the end of this week, you will be able to:

1. ✅ Load and process MATLAB .mat files in Python
2. ✅ Understand time-domain vs frequency-domain representations
3. ✅ Create spectrograms with overlapping windows
4. ✅ Engineer frequency-band features (Delta, Theta, Alpha, Beta)
5. ✅ Apply statistical tests to compare distributions
6. ✅ Understand why certain features work better than others
7. ✅ Build linear regression models using OLS
8. ✅ Implement models in Python, R, and MATLAB

---

## 📁 Week 02 Files

### Core Analysis Pipeline

#### 1. `ReadMAT_ConvertToSignalvsTime_EngineerWINDOWEDFeatures.ipynb` 🌟
**The main notebook from Tuesday's lecture**

This comprehensive Jupyter/Colab notebook demonstrates the complete feature engineering pipeline:

**Part 1: Data Loading**
- Mount Google Drive and copy `.mat` file to `/content`
- Load MATLAB file using `scipy.io.loadmat()`
- Navigate nested MATLAB structures in Python
- Extract raw EEG signal (32,001 samples at 256 Hz)

**Part 2: Time Series Visualization**
```python
# Plot raw signal
plot.plot(DF.S)
plot.xlabel('Sample')
plot.ylabel('Amplitude')
```

**Part 3: Spectrogram Analysis**
- **Window size**: `samplingFrequency` (256 samples = 1 second)
- **Overlap**: `samplingFrequency // 2` (128 samples = 50% overlap)
- **Total windows**: 250 windows
- Creates time-frequency representation showing how frequency content changes

**Understanding Spectrograms:**
A spectrogram decomposes each window into frequency components:
- Each time window analyzed separately
- Signal decomposed into sine waves of different frequencies
- Weights (α₁, α₂, α₃...) represent power at each frequency
- Color-coded to show frequency content over time

**Part 4: Creating Time Axis**
```python
def time_axis(DF, samplingFrequency):
    return np.arange(0, len(DF)/samplingFrequency, 1/samplingFrequency)

DF['time'] = time_axis(DF, samplingFrequency)
```

**Part 5: Window Segmentation**
- Split signal into overlapping 1-second windows
- Each sample belongs to 1-2 windows (except edges)
- Visualize all 250 windows superimposed on common time axis
- Interactive Plotly visualization to explore each window

**Part 6: Frequency Feature Engineering** ⭐
Extract power in standard EEG frequency bands:

```python
def extract_features(segment, fs):
    # Compute Power Spectral Density using Welch method
    freq, psd = welch(segment, window='hamming', fs=fs, nfft=fs)
    total_energy = np.trapz(psd, freq)
    
    # Extract frequency band powers
    delta_power = np.trapz(psd[(freq >= 1) & (freq < 4)]) / total_energy   # 1-4 Hz
    theta_power = np.trapz(psd[(freq >= 4) & (freq < 7)]) / total_energy   # 4-7 Hz
    alpha_power = np.trapz(psd[(freq >= 7) & (freq < 12)]) / total_energy  # 7-12 Hz
    beta_power = np.trapz(psd[(freq >= 12) & (freq < 20)]) / total_energy  # 12-20 Hz
    
    return [delta_power, theta_power, alpha_power, beta_power]
```

**EEG Frequency Bands:**
- **Delta (1-4 Hz)**: Deep sleep, unconscious processes
- **Theta (4-7 Hz)**: Drowsiness, meditation, creativity
- **Alpha (7-12 Hz)**: Relaxed awareness, eyes closed
- **Beta (12-20 Hz)**: Active thinking, focus, anxiety

**Output:**
- DataFrame with 249-250 windows
- Features: delta, theta, alpha, beta powers
- Labels: seizure (1) or non-seizure (0) based on time

**Current Issue (discussed in class):**
The feature extraction produces ~125 windows (without overlap) or 249 windows (with overlap attempting 250). The slight discrepancy from 250 is due to boundary conditions in the windowing algorithm. This is acceptable for our analysis purposes.

---

#### 2. `AnalyzeSignalSofT.R` 🌟
**Statistical analysis of EEG signals (Part 1 covered Tuesday)**

**Tuesday's Content:**

```r
library(readr)
library(dplyr)

# Load data from Week 01
s_of_t_subset_CLEAN <- read_csv("s_of_t_subset_CLEAN.csv",
    col_types = cols(GT = col_factor(levels = c("Normal", "Seizure")))
)

# Box plot comparison
boxplot(s_of_t_subset_CLEAN$normalizedValue ~ s_of_t_subset_CLEAN$GT,
        xlab = "Class", ylab = "Signal",
        main = "Boxplot of Signal by Class"
)
```

**Key Finding 1: T-test (Comparing Means)**
```r
t.test(s_of_t_subset_CLEAN$normalizedValue ~ s_of_t_subset_CLEAN$GT)
```
- **Result**: p-value = 0.9409
- **Interpretation**: 94% probability the two distributions have the **same mean**
- **Conclusion**: Mean amplitude alone **cannot** distinguish seizure from normal

**Key Finding 2: Wilcoxon Rank-Sum Test (Comparing Shapes)**
```r
wilcox.test(s_of_t_subset_CLEAN$normalizedValue ~ s_of_t_subset_CLEAN$GT)
```
- **Result**: p-value = 0.059
- **Interpretation**: The **shapes** of the distributions are marginally different
- **Conclusion**: Distribution patterns matter, not just means!

**Density Plot Visualization:**
```r
plot(density(s_of_t_subset_CLEAN$normalizedValue[s_of_t_subset_CLEAN$GT == "Normal"]), 
     col = "blue", main = "EEG Signal Depicting Seizure")
lines(density(s_of_t_subset_CLEAN$normalizedValue[s_of_t_subset_CLEAN$GT == "Seizure"]), 
     col = "red")
```

**Visual Observation:**
- Blue curve (Normal) and Red curve (Seizure) have **similar means**
- But they have **different shapes** and **different spreads**
- This explains why frequency features are needed!

**Thursday's Content (Preview):**
- Statistical Process Control (SPC)
- 2 Standard Deviation (2SD) rule
- 1 Standard Deviation (1SD) rule  
- Identifying in-control vs out-of-control signals
- Creating control charts

---

#### 3. `linearRegressionFit.R` 🌟
**Linear regression modeling (Thursday's lecture)**

Demonstrates linear regression in R using the `lm()` function:

```r
# Fit linear model: normalizedValue ~ value
model <- lm(normalizedValue ~ value, data = s_of_t_subset_CLEAN)
summary(model)
```

**Results from Thursday's Class:**
```
Coefficients:
              Estimate    Std. Error  Pr(>|t|)    
(Intercept)  -0.063866   0.001369    < 2e-16 ***
value         0.014400   0.000031    < 2e-16 ***

R-squared: 1.000
```

**Interpretation:**
- **β₁ = 0.0144** = 1/σ → **σ = 69.44** (standard deviation)
- **β₀ = -0.0639** = -μ/σ → **μ = 4.44** (mean)
- **Perfect fit** (R² = 1.0) because normalization IS a linear transformation!

**Part 2: Logistic Regression (Thursday's content)**
```r
# Create binary indicator
s_of_t_subset_CLEAN$GT_binary <- ifelse(s_of_t_subset_CLEAN$GT == "Seizure", 1, 0)

# Fit logistic model
logit_model <- glm(GT_binary ~ normalizedValue,
                   data = s_of_t_subset_CLEAN,
                   family = binomial(link = "logit"))
summary(logit_model)
```

**Results:**
- **normalizedValue coefficient**: p = 0.931 (**NOT significant**)
- **Conclusion**: Single amplitude values cannot predict seizure state
- **Reinforces Tuesday's finding**: Need frequency features!

---

#### 4. `OLSsolution.m` 🌟
**Ordinary Least Squares derivation in MATLAB (Thursday's lecture)**

Demonstrates the matrix algebra approach to linear regression with simulated noisy data:

```matlab
% True parameters
beta1 = 5;
beta0 = 10;

% Generate data with noise
X = linspace(0, 1, 100);
Y = beta1*X + beta0 + (rand(size(X))-0.5)*100;

% Matrix formulation
Y = Y';
A = [ones(100,1), X'];

% OLS Solution (two methods)
beta_formula = inv(A'*A)*A'*Y;    % β = (AᵀA)⁻¹AᵀY
beta_builtin = A\Y;                % MATLAB backslash operator

disp(beta_formula);  % [β₀; β₁] ≈ [10; 5]
```

**Results (example run):**
- Estimated: β₀ ≈ 10.31, β₁ ≈ 4.92
- True values: β₀ = 10, β₁ = 5
- **Close despite noise!**

**Key Lesson:**
> "As long as noise is well-behaved (normally distributed), you can recover parameters close to true values using OLS."

---

## 🔬 Key Concepts from Tuesday

### 1. Time Domain vs Frequency Domain

**Time Domain:**
- Signal amplitude vs time
- Shows when events occur
- Limited for pattern recognition

**Frequency Domain:**
- Power vs frequency
- Shows what frequencies are present
- Better for distinguishing signal types

### 2. Signal Decomposition

Any signal can be represented as a sum of sine waves:
```
s(t) = α₁·sin(ω₁t) + α₂·sin(ω₂t) + α₃·sin(ω₃t) + ...
```

Where:
- ω = frequency
- α = weight/power at that frequency
- Fourier transform finds the α values

### 3. Windowing with Overlap

**Why 50% overlap?**
- Captures transient events at window boundaries
- Increases temporal resolution
- Standard practice in signal processing
- Doubles the number of feature vectors

**Window Parameters:**
- Size: 1 second (256 samples at 256 Hz)
- Step: 0.5 seconds (128 samples)
- Number: ~250 windows for 32,000 samples

### 4. Statistical Testing

**Parametric Tests (T-test):**
- Assumes normal distribution
- Compares means
- Sensitive to outliers
- **Result for our data**: No significant difference (p=0.94)

**Non-Parametric Tests (Wilcoxon):**
- No distribution assumption
- Compares ranks/shapes
- More robust
- **Result for our data**: Marginally significant (p=0.059)

### 5. Why Frequency Features Matter

The fundamental discovery from Tuesday:
1. Visual inspection: Seizure signals **look different**
2. Mean comparison: Seizure signals **are not different** (statistically)
3. Shape comparison: Seizure signals **are marginally different**
4. Conclusion: Need better features → **frequency domain!**

---

## 💻 How to Run Tuesday's Code

### Google Colab Notebook

1. **Setup:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Copy data file:**
   ```python
   !cp "/content/drive/MyDrive/.../session4_train_2018.mat" /content
   ```
   *(Update path to match your Google Drive structure)*

3. **Run all cells sequentially**
   - Data loading and exploration
   - Spectrogram generation
   - Window segmentation
   - Feature engineering

4. **Common issues:**
   - **Path errors**: Update the file path to match your Google Drive
   - **Quotes in paths**: Use double quotes if folder names have spaces
   - **249 vs 250 windows**: This is expected due to boundary conditions

### R Script

1. **Open RStudio or Posit Cloud**

2. **Load required libraries:**
   ```r
   library(readr)
   library(dplyr)
   ```

3. **Run `AnalyzeSignalSofT.R` through the first section** (up to density plots)

4. **Observe the outputs:**
   - Box plots showing similar medians
   - T-test showing p=0.94
   - Wilcoxon test showing p=0.059
   - Density plots showing different shapes

---

## 📝 Assignment 0

**Due:** Thursday, January 22 (soft deadline)

**Task:** Adapt the Tuesday notebook to use `EEG_sleep.mat` instead of `session4_train_2018.mat`

**Steps:**
1. Copy `ReadMAT_ConvertToSignalvsTime_EngineerWINDOWEDFeatures.ipynb`
2. Update the file path to load `EEG_sleep.mat`
3. Adjust the data extraction code for the different structure
4. Run the complete pipeline:
   - Load data
   - Create spectrogram
   - Generate windows with 50% overlap
   - Extract frequency features
5. Submit your modified notebook

**Learning Goals:**
- Practice adapting code to different data formats
- Understand MATLAB structure navigation in Python
- Apply feature engineering to new dataset
- Document your changes

**Hints:**
- The data structure in `EEG_sleep.mat` may be nested differently
- Sampling frequency might be different (check the .mat file)
- You may need to experiment with array indexing `[0][0][0]...`
- Use GitHub Copilot or Claude to help debug!

---

## 🎯 Practice Exercises

### Exercise 1: Understanding Spectrograms (Beginner)
1. Run the notebook and generate the spectrogram
2. Identify where the seizure begins visually
3. Describe how the frequency content changes
4. Compare time-domain signal to spectrogram

### Exercise 2: Statistical Testing (Intermediate)
1. Run both t-test and Wilcoxon test in R
2. Explain why they give different results
3. Calculate effect size (difference in means / pooled SD)
4. Create additional visualizations (histograms, Q-Q plots)

### Exercise 3: Feature Engineering (Advanced)
1. Modify the feature extraction to add more frequency bands:
   - Gamma (20-40 Hz)
   - High Gamma (40-100 Hz)
2. Compare features from seizure vs non-seizure windows
3. Calculate which features show the biggest differences
4. Visualize feature distributions with box plots

---

## 🔬 Key Concepts from Thursday

### 1. Confusion Matrix Analysis

**Visual Example from Lecture:**
- Red dots = Out-of-control (predicted seizure)
- Black dots = In-control (predicted normal)
- Before 12s = Actually normal
- After 12s = Actually seizure

**Confusion Matrix (2-Sigma Rule):**
```
                    ACTUAL
              Normal    Seizure
PREDICTED  
In-Control   5,984     3,220    (TN)  (FN)
Out-Control     17       779    (FP)  (TP)
```

**Performance Metrics:**
- **Sensitivity (Recall)**: 19.48% - Catching only 1 in 5 seizures!
- **Specificity**: 99.72% - Rarely false alarms
- **Precision (PPV)**: 97.86% - When we predict seizure, usually correct
- **NPV**: 65.00% - Many false negatives
- **Accuracy**: 67.63% - Overall correct rate

### 2. The Sensitivity-Precision Tradeoff

**1-Sigma Rule Results:**
- Sensitivity: ↑ 39% (up from 19%)
- Precision: ↓ (more false alarms)
- **Use case dependent!**

**COVID Test Analogy:**
> "Would you rather miss infected people (false negatives) or quarantine healthy people (false positives)?"

**Answer:** Decrease sigma → Increase sensitivity → Accept more false positives

### 3. Linear Regression via Matrix Algebra

**The Transformation:**
```
normalizedValue = (value - μ) / σ
Y = (1/σ)·X - (μ/σ)
Y = β₁·X + β₀
```

Where β₁ = 1/σ and β₀ = -μ/σ

**Matrix Formulation:**
```
⎡Y₁⎤   ⎡X₁  1⎤   ⎡β₁⎤
⎢Y₂⎥ = ⎢X₂  1⎥ · ⎢β₀⎥
⎣Yₙ⎦   ⎣Xₙ  1⎦   ⎣  ⎦

Y = A·β
```

**OLS Solution:**
```
β = (AᵀA)⁻¹AᵀY
```

**From R Results:**
- β₁ = 0.0144 → σ = 69.44
- β₀ = -0.0639 → μ = 4.44
- R² = 1.000 (perfect fit!)

### 4. Classification vs Regression

**Regression:**
- Continuous response (normalized value)
- Method: `lm()` in R
- Perfect fit for normalization (R²=1.0)

**Classification:**
- Categorical response (Normal/Seizure)
- Method: `glm()` with logit link
- **Failed** for single amplitude (p=0.931)
- **Why?** Means are identical between classes!

**The Solution:** Frequency features from Tuesday's windowed analysis!

---

## 🔗 Important Links

- **Course Google Drive**: Contains all course materials
- **Google Sheet Lesson Plan**: Week-by-week schedule
- **GitHub Repository**: Spring2026 branch
- **Canvas**: Assignment submissions
- **Fathom**: Lecture recordings with searchable transcripts

---

## 🙋 Frequently Asked Questions

**Q: Why do we use 50% overlap in windowing?**  
**A:** Overlap ensures we don't miss transient events that occur at window boundaries. It's a standard practice in signal processing that provides better temporal resolution without losing information.

**Q: What's the difference between spectrogram and FFT?**  
**A:** FFT (Fast Fourier Transform) analyzes frequency content of the entire signal at once. A spectrogram applies FFT to multiple time windows, showing how frequency content **changes over time**.

**Q: Why are there 249 windows instead of 250?**  
**A:** This is due to boundary conditions in the windowing algorithm. The difference is negligible for our analysis purposes. It occurs when the last window would extend slightly beyond the data length.

**Q: Can I use VS Code instead of Google Colab?**  
**A:** Yes! You can run Jupyter notebooks locally in VS Code if you have Python and required libraries installed. Colab is just easier for getting started.

**Q: Why did the t-test fail but Wilcoxon test show some difference?**  
**A:** The t-test compares **means** (which are nearly identical). Wilcoxon compares **distributions** (which have different shapes). This tells us the **variation patterns** differ, not the central tendency.

**Q: What are Delta, Theta, Alpha, Beta bands used for?**  
**A:** These are standard EEG frequency bands associated with different brain states:
- Delta: Deep sleep
- Theta: Relaxation, meditation  
- Alpha: Calm wakefulness
- Beta: Active thinking, alertness
Different neurological events (like seizures) show characteristic patterns in these bands.

---

## 📚 Additional Resources

### Python/Signal Processing:
- [SciPy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [Welch's Method for PSD](https://en.wikipedia.org/wiki/Welch%27s_method)
- [Understanding Spectrograms](https://pnsn.org/spectrograms/what-is-a-spectrogram)

### Statistics:
- [T-test vs Wilcoxon Test](https://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/t-test-vs-wilcoxon/)
- [Understanding P-values](https://www.nature.com/articles/d41586-019-00857-9)
- [Box Plot Interpretation](https://www.wellbeingatschool.org.nz/information-sheet/understanding-and-interpreting-box-plots)

### EEG Analysis:
- [EEG Frequency Bands](https://www.sciencedirect.com/topics/neuroscience/eeg-frequency-bands)
- [Seizure Detection with EEG](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6723315/)

---

## 🎓 Professor's Notes

This week represents a critical transition in your understanding of AI in healthcare. We're moving beyond simple signal viewing to **feature extraction** and **regression modeling** - the foundations of all machine learning.

**Week 02 Key Takeaways:**

**From Tuesday:**
1. **Visual inspection isn't enough** - we need quantitative features
2. **Simple statistics can mislead** - mean values were identical!
3. **Domain knowledge matters** - EEG frequency bands aren't arbitrary
4. **Frequency features reveal hidden patterns** - spectrograms show the difference

**From Thursday:**
1. **Confusion matrices organize predictions** - TP, TN, FP, FN framework
2. **Metrics tell different stories** - sensitivity vs precision vs accuracy
3. **Context determines "good"** - COVID test needs high sensitivity
4. **Matrix algebra enables solutions** - OLS derivation powers regression
5. **Feature choice is critical** - single amplitude failed, need frequency

**Professor's Closing Thought:**
> "I know this is a little bit complicated, getting into the weeds of this, but linear regression and logistic regression are your basic regression and classification models that we really need to understand through and through."

Remember: In healthcare AI, the goal isn't just to build models - it's to build models that **make sense** clinically and can be **trusted** by medical professionals.

**Professor Prahlad Menon, PhD, PMP**  
*Office Hours: By appointment*  
*Email: prm44@pitt.edu*

---

*"The appropriate threshold for a given statistical test is greatly a function of the purpose of that test."*

---

## 📋 Week 02 Completion Checklist

- [ ] Clone/pull Week 02 content from GitHub (`git pull`)
- [ ] Watch both lecture recordings (Tuesday + Thursday)
- [ ] Run `ReadMAT_ConvertToSignalvsTime_EngineerWINDOWEDFeatures.ipynb` in Colab
- [ ] Run complete `AnalyzeSignalSofT.R` in R/Posit Cloud
- [ ] Run `linearRegressionFit.R` and verify coefficients
- [ ] Run `OLSsolution.m` in MATLAB Online
- [ ] Understand confusion matrix metrics (TP, TN, FP, FN)
- [ ] Understand sensitivity vs precision tradeoff
- [ ] Complete Assignment 0 (EEG_sleep.mat adaptation)
- [ ] Review OLS derivation: β = (AᵀA)⁻¹AᵀY

---

## 🎬 Next Week Preview: Week 03

**Tuesday, January 27, 2026:**
- Using frequency features for classification
- Building classification models with multiple predictors
- Logistic regression deep dive (logit link function explained)
- ROC curves and threshold optimization
- Feature importance and selection

**Get ready to finally build a good seizure detector using the frequency features from Week 02!**

---

**Have a great weekend! Work on Assignment 0 and review this week's concepts. See you Tuesday!** 🚀
