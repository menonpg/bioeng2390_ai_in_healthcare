# BIOENG-2390 Spring 2026 - Lecture 5
## January 27, 2026

**Instructor:** Professor Prahlad G. Menon, PhD, PMP  
**Email:** menon.prahlad@gmail.com  
**Recording:** [View on Fathom](https://fathom.video/share/qKD5GxbQFJxwR33zUkY5ZsaburtGiDXw)  
**Duration:** 95 minutes

---

## 📋 Lecture Overview

Today's class focused on:
1. AutoML with H2O for seizure classification
2. Comparing frequency features (Delta, Theta, Alpha, Beta)
3. Edge-to-edge vs overlapping windowing strategies
4. Non-parametric testing (Mann-Whitney U test)
5. Building multiple model types automatically
6. Model ensembles and stacked learning
7. Variable importance analysis
8. Area Under Curve (AUC) as performance metric
9. Setting up H2O Flow UI in Google Colab

---

## 🎯 Key Concepts

### 1. **Assignment 0 Status**

**Submissions:** Only 4 out of class completed

**Common Issue:** Memory crashes in Colab
- **Solution:** Subset to first 20 seconds of signal (like in class with EEG_sleep.mat)
- Don't process entire 32,000-sample signal at once

**Important Note:**
> "The solution to Assignment 0 appears in this week's `Process_EEG_sleep.ipynb` notebook. Even though the answer is provided, please try it yourself for the learning experience!"

---

### 2. **Windowing Strategies Explained**

#### Edge-to-Edge Windows (No Overlap)
```
Window 1: [----]
Window 2:      [----]
Window 3:           [----]
```
- Total windows = Signal length / Window size
- ~125 windows for our data
- No redundancy
- Might miss transient events at boundaries

#### 50% Overlap Windows
```
Window 1: [----]
Window 2:   [----]
Window 3:     [----]
```
- Total windows ≈ 2× edge-to-edge
- ~250 windows for our data
- Better temporal resolution
- Captures boundary events
- **Standard practice in signal processing**

#### 75% Overlap Windows
```
Window 1: [----]
Window 2:  [----]
Window 3:   [----]
```
- Even more windows
- Higher temporal resolution
- More computational cost

**Challenge with Ground Truth Labeling:**
> "When you have overlap, the boundary between seizure and non-seizure becomes a little blurry."

**Strategies for assigning labels to overlapping windows:**
1. **Any seizure content:** If window contains ANY seizure time, label as seizure
2. **Majority vote:** If >50% of window is seizure, label as seizure  
3. **Time-based:** Use timestamp of window center
4. **Conservative:** Label only if entirely within seizure period

**What we did in class:**
- Divided signal at 50% point (time-based)
- First half windows = Normal (label 0)
- Second half windows = Seizure (label 1)
- Simple but not perfect for real-world applications

---

### 3. **Feature Comparison Results**

**From Box Plots and Statistical Tests:**

#### Delta Power (1-4 Hz)
- **Visual:** Some median separation
- **Mann-Whitney U:** Significant (p < 0.001)
- **T-test:** Significant
- **Conclusion:** Weakest of the significant features
- **Pattern:** Broader range in Normal, narrower in Seizure

#### Theta Power (4-7 Hz)
- **Visual:** Slight median separation
- **Mann-Whitney U:** NOT significant (p = 0.074)
- **T-test:** NOT significant
- **Conclusion:** Not a strong discriminator
- **Pattern:** Distributions overlap significantly

#### Alpha Power (7-12 Hz) ⭐
- **Visual:** Good median separation
- **Mann-Whitney U:** Highly significant (p < 10⁻¹⁰)
- **T-test:** Highly significant
- **Conclusion:** Strong feature!
- **Pattern:** Higher in Normal, lower in Seizure

#### Beta Power (12-20 Hz) ⭐⭐
- **Visual:** Best median separation
- **Mann-Whitney U:** Most significant (p < 10⁻¹³)
- **T-test:** Most significant  
- **Conclusion:** Strongest feature!
- **Pattern:** Higher in Normal, much lower in Seizure

**Ranking:** Beta > Alpha > Delta > Theta

**Key Insight:**
> "Beta and Alpha especially, which are the higher frequency power areas, are very interesting features to consider when trying to identify seizure versus non-seizure states."

---

### 4. **Parametric vs Non-Parametric Testing**

#### Why Non-Parametric Tests Matter

**Problem:** Our frequency band distributions are **skewed** (asymmetric)

**Skewness Types:**
- **Right-skewed:** Mean > Median (long tail to right)
- **Left-skewed:** Mean < Median (long tail to left)
- **Symmetric:** Mean ≈ Median (normal distribution)

**T-test (Parametric):**
- Assumes symmetric (normal/T) distributions
- Fits best symmetric curve to actual data
- Compares means
- **Works when:** Data is approximately normal
- **Our result:** Still worked because skewness not severe

**Mann-Whitney U Test (Non-Parametric):**
- NO distribution assumptions
- Compares distribution shapes/medians
- Ranks all values and compares ranks
- **Works when:** Any distribution shape
- **Our result:** Confirms t-test findings

**Overlap Analysis:**
The larger the overlap between distributions, the harder to distinguish them:
```
Small overlap = Highly distinguishable → Low p-value
Large overlap = Hard to distinguish → High p-value
```

---

### 5. **H2O AutoML Introduction**

**What is H2O?**
- Automatic Machine Learning framework
- Builds multiple model types simultaneously
- Optimizes hyperparameters automatically
- Creates ensemble models
- Provides user-friendly interface (H2O Flow)

#### Installation in Colab:
```python
!pip install h2o
```

#### Starting H2O Cluster:
```python
import h2o
h2o.init(max_mem_size="2G")
```

#### Accessing H2O Flow UI in Colab:
```python
# Google Colab's built-in port forwarding
from google.colab import output
output.serve_kernel_port_as_window(54321)
```

**Note:** ng-rock and local-tunnel no longer work due to Colab restrictions. Use Colab's built-in method!

---

### 6. **H2O AutoML Workflow**

#### Step 1: Import Data
```python
# Via H2O Flow UI:
# 1. Upload CSV file or provide path
# 2. Parse file (H2O detects CSV structure)
# 3. Set column types (numeric vs categorical/enum)
# 4. Import
```

**Our data:** `segmentDF_with_frequency_features.csv`
- **Features:** delta, theta, alpha, beta (numeric)
- **Response:** seizure (categorical: 0 or 1)

#### Step 2: Split Data
- **Training:** 75% of windows (randomly selected)
- **Validation:** 25% of windows
- **Purpose:** Train on majority, test on unseen data

**Important:** Random split acceptable because windows are already aggregated features, not sequential time points.

#### Step 3: Run AutoML
```python
# In H2O Flow UI:
# 1. Click "Run AutoML"
# 2. Select training frame (75% split)
# 3. Set response column: "seizure"
# 4. Set validation frame (25% split)  
# 5. Enable cross-validation (5-fold)
# 6. Select algorithms to run
# 7. Set max runtime (optional)
# 8. Start
```

**Algorithms Available:**
- ✅ **GLM** (Generalized Linear Model) - Logistic regression
- ✅ **DRF** (Distributed Random Forest) - Ensemble of decision trees
- ✅ **GBM** (Gradient Boosting Machine) - Boosted trees
- ✅ **XGBoost** - Extreme gradient boosting
- ⬜ **DeepLearning** - Neural networks (disabled - too slow)
- ✅ **StackedEnsemble** - Meta-learner combining other models

**Cross-Validation:** 5-fold
- Splits training data into 5 parts
- Trains on 4 parts, validates on 1 part
- Repeats 5 times with different validation folds
- Averages performance across folds
- **Purpose:** Detect overfitting and data sensitivity

---

### 7. **Model Results from Today**

#### Best Overall Model: Stacked Ensemble
- **Type:** StackedEnsemble_BestOfFamily
- **Meta-learner:** Gradient Boosting Machine (GBM)
- **Base models:** XRT, DRF, GLM, GBM, XGBoost
- **Training AUC:** ~1.0 (perfect on training data)
- **Validation AUC:** ~1.0 (perfect on validation data)
- **Cross-validation AUC:** High (exact value not shown, but excellent)

**What is a Stacked Ensemble?**
- Trains multiple different model types
- Uses predictions from base models as inputs
- Trains a **meta-learner** to weight base model predictions
- Often outperforms individual models

**This Ensemble Used:**
1. Extremely Randomized Trees (XRT)
2. Distributed Random Forest (DRF)
3. Generalized Linear Model (GLM)
4. Gradient Boosting Machine (GBM)
5. XGBoost

Meta-learner (GBM) learned optimal weights for combining these 5 models!

#### Best Individual Model: GLM (Generalized Linear Model)
- **Training AUC:** 0.75
- **Validation AUC:** 0.75
- **Type:** Logistic regression
- **Advantage:** Highly interpretable!

**Variable Importance (from GLM):**
1. **Beta** (most important) - Negative correlation
2. **Alpha** - Negative correlation
3. **Theta** - Less important
4. **Delta** - Least important

**Interpretation:**
- When Beta ↓ → Seizure likelihood ↑
- When Alpha ↓ → Seizure likelihood ↑
- **Matches our box plot observations!**

---

### 8. **Understanding AUC (Area Under Curve)**

**What is AUC?**
- Metric ranging from 0 to 1
- Based on ROC (Receiver Operating Characteristic) curve
- **1.0** = Perfect classifier (no errors)
- **0.5** = Random guessing (useless)
- **Higher = Better**

**ROC Curve Preview (Thursday's topic):**
- Plots True Positive Rate vs False Positive Rate
- Shows performance across all probability thresholds
- AUC = Area under this curve

**From Today:**
- GLM alone: AUC = 0.75 (good)
- Stacked ensemble: AUC ≈ 1.0 (excellent!)

---

### 9. **Linear vs Non-Linear Models**

**Linear Models:**
- Features combined with simple addition/multiplication
- Example: `log(odds) = β₀ + β₁·beta + β₂·alpha + β₃·theta + β₄·delta`
- **Pros:** Interpretable, fast, stable
- **Cons:** Limited expressiveness

**Non-Linear Models:**
- Features can be squared, multiplied together, etc.
- Examples: Random Forest, Gradient Boosting, Neural Networks
- **Pros:** Can learn complex patterns
- **Cons:** Less interpretable ("black box")

**Today's Finding:**
- Linear (GLM): AUC = 0.75
- Non-linear (XGBoost): Better performance
- Ensemble (Stack): Best performance!

**Lesson:** Non-linear models capture patterns linear models miss!

---

### 10. **H2O Flow Interface**

**Key Features:**
1. **No-code AutoML:** Click-based model building
2. **Data exploration:** View distributions, statistics
3. **Model comparison:** Automatic leaderboard
4. **Variable importance:** Which features matter most
5. **Model export:** Save models for later use
6. **Cross-validation:** Built-in CV support

**Exporting Models:**
```python
# Copy model from /content/models/ to Google Drive
!cp /content/models/StackedEnsemble_BestOfFamily_... /content/drive/MyDrive/.../Week03/
```

---

## 💻 Code from Today's Lecture

### Setting up H2O in Colab

```python
# Install H2O
!pip install h2o

# Import and initialize
import h2o
h2o.init(max_mem_size="2G")

# Access H2O Flow UI
from google.colab import output
output.serve_kernel_port_as_window(54321)
```

### Loading Data into H2O

```python
# Option 1: From path
data = h2o.import_file("/content/drive/MyDrive/.../segmentDF_with_frequency_features.csv")

# Option 2: Via Flow UI
# - Click "Import Files"
# - Select or upload CSV
# - Parse and import
```

### Running AutoML Programmatically

```python
from h2o.automl import H2OAutoML

# Split data
train, valid = data.split_frame(ratios=[0.75], seed=42)

# Define predictors and response
x = ["delta", "theta", "alpha", "beta"]
y = "seizure"

# Run AutoML
aml = H2OAutoML(max_runtime_secs=600,  # 10 minutes max
                nfolds=5,              # 5-fold cross-validation
                seed=42)

aml.train(x=x, y=y, training_frame=train, validation_frame=valid)

# View leaderboard
lb = aml.leaderboard
print(lb)

# Best model
best_model = aml.leader
```

### Extracting Model Information

```python
# Variable importance
varimp = best_model.varimp(use_pandas=True)
print(varimp)

# Performance metrics
perf = best_model.model_performance(valid)
print(perf)

# Confusion matrix
print(perf.confusion_matrix())
```

---

## 🔬 Statistical Testing Results

### Mann-Whitney U Test (Non-Parametric)

**Results:**
```
Feature    p-value        Interpretation
----------------------------------------------
Delta      < 0.001        Significant
Theta      0.074          NOT significant
Alpha      < 10⁻¹⁰        Highly significant
Beta       < 10⁻¹³        Most significant
```

### T-Test (Parametric)

**Results:**
```
Feature    p-value        Interpretation
----------------------------------------------
Delta      < 0.001        Significant
Theta      Not sig        NOT significant  
Alpha      < 10⁻¹⁰        Highly significant
Beta       < 10⁻¹³        Most significant
```

**Observation:**
Both tests agree! This suggests distributions, while skewed, aren't severely non-normal.

**Contrast with Week 02:**
- Amplitude alone: T-test p=0.94, Wilcoxon p=0.059 (disagreement!)
- Frequency features: Both tests agree (better features!)

---

## 📊 Box Plot Interpretation

### Understanding Box Plots

```
     |----[====|====]----| 
     ^    ^    ^    ^    ^
    min   Q1   med  Q3   max

□ Box: 25th-75th percentile (IQR)
| Line in box: Median (50th percentile)
— Whiskers: Extend to min/max (or 1.5×IQR)
● Dots: Outliers beyond whiskers
```

**For Skewed Distributions:**
- Median ≠ Mean
- Longer whisker/tail on one side
- Outliers more common on skewed side

**Our Observations:**
- **Beta:** Clear separation, minimal overlap
- **Alpha:** Good separation, some overlap
- **Delta:** Slight separation, significant overlap
- **Theta:** Poor separation, large overlap

---

## 🎯 Key Findings

### 1. Frequency Features Work!

**Compared to Week 02 amplitude-only analysis:**
- **Amplitude:** p=0.94 (no difference in means)
- **Frequency bands:** Multiple features p<0.001 (huge differences!)

### 2. Higher Frequencies More Informative

**Why Beta and Alpha are best:**
- Seizures involve rapid, chaotic neural firing
- This creates high-frequency oscillations
- Normal brain activity more organized
- Lower high-frequency power in normal states

### 3. Ensemble Models Outperform

**Individual models:**
- GLM: AUC = 0.75 (interpretable but limited)
- XGBoost: Better than GLM
- GBM, DRF: Also better than GLM

**Ensemble:**
- AUC ≈ 1.0 (near perfect!)
- Combines strengths of multiple models
- More robust to different data patterns

### 4. Feature Engineering is Critical

**Timeline:**
- Week 01: Tried using raw amplitude → Failed
- Week 02 Tuesday: Engineered frequency features → Success!
- Week 02 Thursday: Confirmed amplitude features fail (p=0.931)
- Week 03 Tuesday: Built successful classifiers with frequency features!

**Lesson:**
> "The right features make all the difference. Domain knowledge (EEG frequency bands) + feature engineering = successful AI models."

---

## 🎬 Thursday Preview (January 29)

### Topics to Cover:

1. **Logistic Regression Deep Dive**
   - Deriving the logit link function
   - Understanding log-odds
   - Why we can't use linear regression for classification
   - Connecting GLM math to Week 02's OLS

2. **ROC Curves Explained**
   - True Positive Rate vs False Positive Rate
   - How to interpret ROC curves
   - Why AUC is a good metric
   - Selecting optimal probability thresholds

3. **Cross-Validation**
   - K-fold CV in depth
   - Why we need CV (prevent overfitting)
   - Interpreting CV metrics
   - When to trust your model

4. **Model Evaluation**
   - Beyond accuracy: precision, recall, F1-score
   - Confusion matrices for multi-threshold analysis
   - Calibration curves
   - Model diagnostics

### Homework for Thursday:
- [ ] Complete Assignment 0 (even if solution is available)
- [ ] Run today's H2O notebook
- [ ] Review confusion matrix concepts from Week 02
- [ ] Think about: What does "log-odds" mean?
- [ ] Try building H2O models with different feature combinations

---

## 🔑 Important Concepts

### 1. Train-Validation-Test Split

**Why split data?**
- **Training:** Learn patterns (75% of data)
- **Validation:** Tune and select best model (25% of data)
- **Test:** Final evaluation (not used yet)

**Our approach:**
- Training: 75% of windows
- Validation: 25% of windows
- No separate test set yet (will add later)

### 2. AutoML Advantages

**Traditional approach:**
1. Choose algorithm
2. Code implementation
3. Tune hyperparameters (trial and error!)
4. Repeat for different algorithms
5. Compare results manually

**H2O AutoML approach:**
1. Load data
2. Click "Run AutoML"
3. Get leaderboard of best models
4. Done!

**Saves:** Hours/days of manual coding and tuning

### 3. Ground Truth Assignment

**Our simple rule:** Based on time (first half = normal, second half = seizure)

**Better approaches:**
- Label based on actual seizure annotations
- Use medical expert labels
- Consider window overlap in labeling
- Use probabilistic labels for boundary windows

---

## 🙋 Questions from Class

**Q: What are the most common methods of assigning ground truth to a window?**  
**A:** 
1. **Any seizure:** If window contains any seizure time → label as seizure
2. **Majority:** If >50% of window is seizure → label as seizure
3. **Time-based:** Use timestamp of window start/center/end
4. **Expert-defined:** Medical professionals label each window

**Q: How do I know the scatterplot corresponds to one window?**  
**A:** The `extract_features()` function analyzes one segment/window at a time. Each call processes a single window and returns features for that window.

**Q: Are EEG_sleep.mat and session4_train_2018.mat related?**  
**A:** No! They are completely different datasets. Just similar in structure (both EEG data). Different:
- Number of samples
- Sampling frequencies
- Internal structure/nesting
- Seizure timing

---

## 💡 Troubleshooting

### Memory Issues in Colab
- **Problem:** "Runtime crashed" when processing large .mat files
- **Solution:** Subset to first 20 seconds (10,000 samples) like in class
- **Code:** `raw = raw[:10000]`

### H2O Flow Access
- **Problem:** Can't access localhost:54321
- **Old solutions:** ng-rock, local-tunnel (no longer work in Colab)
- **New solution:** `output.serve_kernel_port_as_window(54321)`

### Git Pull
- **Where:** Open terminal in VS Code
- **Command:** `cd` to repository folder, then `git pull`
- **Purpose:** Get Week03 content

---

## 📚 Files in Week 03

### 1. `Process_session4_train_2018.ipynb` 🌟
**Main notebook from today's lecture**

**Contents:**
- Simple signal example (sum of sinusoids)
- Power spectral density visualization for one window
- Box plot comparison of all four frequency features
- Mann-Whitney U tests
- T-tests
- H2O AutoML setup and execution

**Purpose:** Demonstrates classification with frequency features

### 2. `Process_EEG_sleep.ipynb`
**Solution to Assignment 0**

Adapts the Week 02 feature engineering pipeline to `EEG_sleep.mat` dataset. Contains same analysis but with different data file.

**⚠️ Note:** This contains the assignment solution. Try Assignment 0 yourself first before looking!

### 3. `windowing.ipynb`
**Review material**

Re-demonstrates windowing concepts from Week 02:
- Creating overlapping windows
- Plotting windows on common time axis
- Interactive visualization

**Purpose:** Supplementary review, no new content

---

## 🎓 Professor's Insights

**On Assignment 0:**
> "Even though the solution is provided in this week's folder, try it independently for the learning experience. It's not about getting the answer—it's about learning the process!"

**On Feature Selection:**
> "Beta and Alpha are strong features because seizures involve rapid, chaotic neural firing creating high-frequency oscillations."

**On Model Complexity:**
> "Linear models are interpretable and tell us Beta is most important. Non-linear models perform better but are 'black boxes.' Ensembles give us the best of both worlds."

**For Next Class:**
> "We're going to talk about log-odds and derive this thing. We're going to talk about how to understand ROC curves. Finally, we'll talk about cross-validation in a very deliberate way."

---

## 📋 Week 03 Tuesday Checklist

- [ ] `git pull` to get Week 03 content
- [ ] Run `Process_session4_train_2018.ipynb`
- [ ] Install H2O in Colab (`!pip install h2o`)
- [ ] Access H2O Flow UI
- [ ] Build AutoML models with frequency features
- [ ] Compare box plots of Delta, Theta, Alpha, Beta
- [ ] Understand why Beta and Alpha are strongest features
- [ ] Complete Assignment 0 (if not done)
- [ ] Review confusion matrix concepts for Thursday

---

**Next Class:** Thursday, January 29, 2026  
**Topics:** Log-odds derivation, ROC curves, Cross-validation explained

---

*"Understanding what the model has learned and how best to use it and operationalize it - these are the things we will learn."*

— Professor Prahlad G. Menon, PhD, PMP

---

## 🔑 Key Takeaways

1. ✅ Frequency features (Beta, Alpha) successfully distinguish seizure states
2. ✅ Non-parametric tests appropriate for skewed distributions
3. ✅ H2O AutoML builds multiple models automatically
4. ✅ Ensemble models outperform individual models
5. ✅ Variable importance confirms Beta most important
6. ✅ AUC is a key performance metric (higher = better)
7. ✅ 50% overlap windowing is standard practice
8. ✅ Ground truth labeling strategy matters
9. ✅ Feature engineering was the breakthrough (vs Week 02 amplitude failure)
10. ✅ Next: Understanding the math behind these models!
