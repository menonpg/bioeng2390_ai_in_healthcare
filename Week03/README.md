# Week 03: AutoML & Classification with Frequency Features
### BIOENG 2390: AI in Healthcare - Spring 2026

**Instructor:** Professor Prahlad Menon, PhD, PMP  
**University of Pittsburgh, Department of Bioengineering**

---

## 🎯 Week 03 Overview

This week we transition from feature engineering to **automated model building** and **classification**. We'll use H2O AutoML to build multiple model types simultaneously and discover which frequency features are most important for seizure detection. This week demonstrates the payoff from Week 02's feature engineering!

---

## 📺 Lecture Recordings & Notes

### Lecture 5 - January 27, 2026 (95 minutes)
**Focus:** AutoML, Feature Comparison & Classification Success

- **[Watch Recording](https://fathom.video/share/qKD5GxbQFJxwR33zUkY5ZsaburtGiDXw)**
- **[Read Detailed Lecture Notes](Lecture05_Notes_Jan27_2026.md)** ← Complete transcript

**Topics Covered:**
- Windowing strategies: Edge-to-edge vs 50% overlap vs 75% overlap
- Box plot comparison of frequency features (Delta, Theta, Alpha, Beta)
- Mann-Whitney U test (non-parametric) vs T-test (parametric)
- H2O AutoML installation and setup in Google Colab
- Building multiple model types automatically (GLM, GBM, XGBoost, RF, Stacked Ensembles)
- Variable importance analysis
- Understanding AUC (Area Under Curve) metric
- Train/validation split strategies

**Key Discovery:**
> "Beta and Alpha frequency features show highly significant differences between seizure and normal states (p < 10⁻¹³ and p < 10⁻¹⁰), while simple amplitude failed (p=0.94). Feature engineering was the breakthrough!"

**Model Results:**
- **Best Individual Model:** GLM with AUC = 0.75
- **Best Overall Model:** Stacked Ensemble with AUC ≈ 1.0 (near perfect!)
- **Variable Importance:** Beta > Alpha > Delta > Theta

---

### Lecture 6 - January 29, 2026 (89 minutes)
**Focus:** Logistic Regression Derivation, ROC Curves & Threshold Optimization

- **[Watch Recording](https://fathom.video/share/xTNg5omfgG1KEP6erJbD7zZcQBTFxQn9)**
- **[Read Detailed Lecture Notes](Lecture06_Notes_Jan29_2026.md)** ← Complete derivation and concepts

**Topics Covered:**
- Complete log-odds derivation: ln(P/(1-P)) = β₀ + Σβᵢ·Xᵢ
- Understanding sigmoid function as activation/transfer function
- ROC curves: plotting TPR vs FPR across all thresholds
- AUC interpretation (GLM achieved 0.81)
- Train/Validation/Test split (70/20/10) strategy
- Optimal threshold selection from validation set (0.56-0.58)
- Building GLM with feature interactions in H2O
- Variable importance: Alpha > Theta > Interactions > Beta > Delta
- Making predictions on test data
- IID assumption and why random splits are valid

**Key Mathematics:**
> "We transformed the probability problem into log-odds so we could use OLS from Week 02 to solve it. Then we convert back to probabilities using the sigmoid function!"

**Threshold Results:**
- Training: 0.61, Validation: 0.56, Cross-Val: 0.58
- **Chosen:** 0.57-0.58 for optimal balance
- Test performance: TPR=1.0 (perfect sensitivity!), FPR=0.54

---

## 🎯 Week Learning Objectives

By the end of this week, you will be able to:

1. ✅ Compare statistical distributions with parametric and non-parametric tests
2. ✅ Understand windowing overlap strategies and tradeoffs
3. ✅ Use H2O AutoML to build multiple models automatically
4. ✅ Interpret variable importance rankings
5. ✅ Understand AUC as a performance metric
6. ✅ Build ensemble models that outperform individual models
7. ✅ Explain why frequency features succeed where amplitude failed
8. ✅ Set up H2O Flow UI in Google Colab

---

## 📁 Week 03 Files

### 1. `Process_session4_train_2018.ipynb` 🌟
**Main notebook from Tuesday's lecture**

This notebook builds on Week 02's feature engineering and demonstrates successful classification:

**Part 1: Simple Signal Example**
- Creates sum of two sinusoids: `y = sin(2π·10·t) + sin(2π·20·t)`
- Visualizes single-window power spectral density
- Demonstrates how Welch method works
- Shows frequency decomposition concept

**Part 2: Feature Comparison**
Box plots comparing all four frequency features:

**Beta Power (12-20 Hz):** ⭐⭐ **Best Feature**
- Clear median separation
- Minimal distribution overlap
- p < 10⁻¹³ (Mann-Whitney U)
- **Pattern:** Much higher in Normal than Seizure

**Alpha Power (7-12 Hz):** ⭐ **Second Best**
- Good median separation
- Some distribution overlap
- p < 10⁻¹⁰ (Mann-Whitney U)
- **Pattern:** Higher in Normal than Seizure

**Delta Power (1-4 Hz):** Weakest Significant Feature
- Slight median separation
- Significant overlap
- p < 0.001 (Mann-Whitney U)
- **Pattern:** Broader range in Normal

**Theta Power (4-7 Hz):** Not Significant
- Poor median separation
- Large overlap
- p = 0.074 (Mann-Whitney U)
- **Pattern:** Similar in both states

**Part 3: Statistical Testing**
```python
from scipy.stats import mannwhitneyu, ttest_ind

# Compare each feature between seizure and non-seizure
for feature in ['delta', 'theta', 'alpha', 'beta']:
    stat, p = mannwhitneyu(seizure_data[feature], normal_data[feature])
    print(f"{feature}: p={p:.2e}")
```

**Part 4: H2O AutoML**
- Install H2O: `!pip install h2o`
- Initialize cluster: `h2o.init(max_mem_size="2G")`
- Access Flow UI: `output.serve_kernel_port_as_window(54321)`
- Import frequency features CSV
- Split 75% train / 25% validation
- Run AutoML with 5-fold cross-validation
- Compare GLM, GBM, XGBoost, DRF, Stacked Ensembles

**Results:**
- GLM (Linear): AUC = 0.75
- Non-linear models: Better than GLM
- Stacked Ensemble: AUC ≈ 1.0 (excellent!)

**Variable Importance from GLM:**
1. Beta (negative correlation: ↓Beta → ↑Seizure)
2. Alpha (negative correlation)
3. Theta
4. Delta

---

### 2. `Process_EEG_sleep.ipynb`
**Solution to Assignment 0** ⚠️

This notebook adapts Week 02's feature engineering to `EEG_sleep.mat`:
- Loads EEG_sleep.mat (500 Hz, 10,000 samples)
- Creates spectrogram
- Generates 250 windows with 50% overlap
- Extracts Delta, Theta, Alpha, Beta features
- Prepares data for classification

**Note from Professor:**
> "Even though the solution is provided, please try Assignment 0 yourself for the learning experience!"

**Why use this:**
- Reference for adapting code to different .mat structures
- Example of proper ground truth labeling
- Complete working pipeline for EEG_sleep.mat

---

### 3. `ReadMAT_ConvertToSignalvsTime_EngineerWINDOWEDFeatures.ipynb`
**Week 02 content (for reference)**

Same as Week 02 version - demonstrates feature engineering with session4_train_2018.mat.

---

### 4. `Seizure Model from Windowed Frequency Features.flow`
**H2O Flow export**

Saved H2O Flow session from Tuesday's class:
- Pre-configured AutoML run
- Model comparisons
- Can be loaded back into H2O for review

---

## 🔬 Key Concepts from Tuesday

### 1. **Windowing Strategies**

**Visual Representation:**
```
Edge-to-Edge:
[----][----][----][----]

50% Overlap:
[----]
  [----]
    [----]
      [----]

75% Overlap:
[----]
 [----]
  [----]
   [----]
```

**Tradeoffs:**
- More overlap → More windows → Better time resolution → Blurrier boundaries
- Less overlap → Fewer windows → Sharper boundaries → Might miss events

**Our Choice:** 50% overlap (standard in signal processing)

### 2. **Feature Ranking**

**Statistical Significance:**
1. **Beta**: p < 10⁻¹³ (most significant)
2. **Alpha**: p < 10⁻¹⁰ (highly significant)
3. **Delta**: p < 0.001 (significant)
4. **Theta**: p = 0.074 (NOT significant)

**Clinical Interpretation:**
- **High-frequency bands** (Beta, Alpha) capture seizure chaos
- **Low-frequency bands** (Delta, Theta) less informative
- Matches neuroscience literature!

### 3. **Parametric vs Non-Parametric Tests**

**When Both Agree (our frequency features):**
- Data is approximately normal (not severely skewed)
- Both tests give similar p-values
- Robust finding!

**When They Disagree (Week 02 amplitude):**
- Data is non-normal or has outliers
- T-test misled us (p=0.94)
- Wilcoxon was more appropriate (p=0.059)

**Lesson:** Always check with both tests when unsure about distribution!

### 4. **H2O AutoML Workflow**

**Setup in Google Colab:**
```python
# Install
!pip install h2o

# Initialize
import h2o
h2o.init(max_mem_size="2G")

# Access UI (NEW METHOD - ng-rock no longer works!)
from google.colab import output
output.serve_kernel_port_as_window(54321)
```

**In H2O Flow:**
1. Import CSV with frequency features
2. Split data (75% train, 25% validate)
3. Run AutoML:
   - Response: "seizure"
   - Predictors: delta, theta, alpha, beta
   - Cross-validation: 5-fold
   - Algorithms: GLM, GBM, XGBoost, DRF, Ensembles
4. Review leaderboard (sorted by AUC)
5. Examine best model
6. Save models to Google Drive

**Export Models:**
```python
!cp /content/models/StackedEnsemble_* /content/drive/MyDrive/.../Week03/
```

### 5. **Model Types Explained**

**GLM (Generalized Linear Model):**
- Logistic regression with frequency features
- Interpretable variable importance
- Linear combination of features
- AUC = 0.75

**DRF (Distributed Random Forest):**
- Ensemble of decision trees
- Each tree uses random subset of features
- Averages predictions across trees
- Better than GLM

**GBM (Gradient Boosting Machine):**
- Sequential tree building
- Each tree corrects previous tree's errors
- Powerful non-linear model
- Better than DRF

**XGBoost:**
- Extreme gradient boosting
- Optimized implementation of GBM
- Very popular in competitions
- Often best individual model

**Stacked Ensemble:**
- Trains all above models
- Uses GBM as meta-learner to weight predictions
- Combines strengths of all models
- **Best overall:** AUC ≈ 1.0

### 6. **Understanding AUC**

**AUC (Area Under ROC Curve):**
- Ranges from 0 to 1
- **1.0** = Perfect classifier
- **0.75** = Good classifier
- **0.5** = Random guessing
- **< 0.5** = Worse than random (something's inverted!)

**Why AUC is Better than Accuracy:**
- Works with imbalanced datasets
- Threshold-independent
- Shows performance across all operating points
- More comprehensive than single confusion matrix

**Thursday:** We'll derive ROC curves and understand AUC deeply!

---

## 🎯 Practice Exercises

### Exercise 1: Run H2O AutoML (Beginner)
1. Open `Process_session4_train_2018.ipynb` in Colab
2. Install H2O and initialize
3. Access Flow UI
4. Import the frequency features CSV
5. Run AutoML with default settings
6. Examine the leaderboard
7. Identify which model performed best

### Exercise 2: Feature Analysis (Intermediate)
1. Create box plots for each feature
2. Run both t-test and Mann-Whitney U test
3. Calculate effect sizes
4. Rank features by statistical significance
5. Explain why Beta and Alpha are strongest

### Exercise 3: Model Comparison (Advanced)
1. Run AutoML with different feature combinations:
   - Only Beta and Alpha
   - All four features
   - Add Delta only
2. Compare AUC scores
3. Determine if Theta and Delta add value
4. Document your findings

---

## 💻 How to Run Tuesday's Code

### Google Colab Setup

1. **Mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Copy .mat file (if needed):**
   ```python
   !cp "/content/drive/MyDrive/.../session4_train_2018.mat" /content
   ```

3. **Install H2O:**
   ```python
   !pip install h2o
   ```

4. **Initialize and access UI:**
   ```python
   import h2o
   h2o.init(max_mem_size="2G")
   
   from google.colab import output
   output.serve_kernel_port_as_window(54321)
   ```

5. **Run notebook cells or use Flow UI**

### Common Issues

**Memory crashes:**
- Subset data to first 20 seconds: `raw = raw[:fs*20]`
- Reduce window size or overlap

**Can't access H2O Flow:**
- Use `output.serve_kernel_port_as_window(54321)` (NEW method)
- ng-rock and local-tunnel no longer work in Colab

**File path errors:**
- Update paths to match your Google Drive structure
- Use double quotes if folder names have spaces

---

## 📝 Assignment 0 Status

**Due:** Was January 22 (soft deadline extended)

**Submissions:** Only 4 students completed

**Task:** Adapt Week 02 notebook to `EEG_sleep.mat`

**Solution Available:** `Process_EEG_sleep.ipynb` in this folder

**Professor's Advice:**
> "Try it yourself even though the solution is provided. It's about the learning process, not just getting the answer!"

**If you had memory issues:**
- Subset to 20 seconds like in class
- Don't process full 32,000-sample signal

---

## 🔬 Key Concepts from Thursday

### 1. **Logistic Regression Mathematics**

**Complete Log-Odds Derivation:**
```
Start:  P(Y=1) = 1 / (1 + e^(-(β₀ + Σβᵢ·Xᵢ)))
Step 1: 1/P = 1 + e^(-(β₀ + Σβᵢ·Xᵢ))
Step 2: e^(-(β₀ + Σβᵢ·Xᵢ)) = (1-P)/P
Step 3: -(β₀ + Σβᵢ·Xᵢ) = ln((1-P)/P)
Final:  ln(P/(1-P)) = β₀ + Σβᵢ·Xᵢ  ← Log-odds!
```

**Why this matters:**
- Left side = log-odds (continuous variable we can model)
- Right side = linear combination (we can solve with OLS!)
- **Connects to Week 02's matrix algebra**

**From Probability to Prediction:**
1. Solve for β using log-odds transformation
2. Compute P(Y=1) using sigmoid: `1 / (1 + e^(-(β₀ + Σβᵢ·Xᵢ)))`
3. Apply threshold to get binary decision

### 2. **ROC Curves & Optimal Thresholds**

**ROC Curve:**
- X-axis: False Positive Rate (FPR)
- Y-axis: True Positive Rate (TPR) = Sensitivity
- Each point = different probability threshold
- AUC = Area under this curve

**Our GLM Results:**
- Training AUC: 0.81
- Validation AUC: 0.81
- Cross-validation AUC: 0.81
- **Optimal thresholds:** Train=0.61, Validation=0.56, CV=0.58

**Selected threshold:** 0.57-0.58 (from validation/CV)

**Test set performance at threshold=0.57:**
- TPR = 1.0 (catching all seizures!)
- FPR = 0.54 (some false alarms, but acceptable)

### 3. **Train/Validation/Test Strategy**

**70/20/10 Split:**
- **Training (70%):** Learn β coefficients
- **Validation (20%):** Optimize threshold/operating point
- **Test (10%):** Final unbiased evaluation

**Why three splits?**
> "Train the model, use validation to find optimal threshold, then apply both to test set for true out-of-sample performance."

### 4. **IID Assumption**

**Question:** Does H2O respect temporal nature of samples?

**Answer:** No, assumes windows are IID (Independent and Identically Distributed)

**Why this works:**
- Time already handled in feature engineering (windowing)
- Each window is independent observation of brain state
- Random shuffling into train/val/test is valid
- **Different from time-series forecasting!**

### 5. **Feature Interactions in GLM**

**With interactions enabled:**
- H2O creates: delta×theta, theta×alpha, alpha×beta, etc.
- Captures combined effects
- Example: Low Alpha AND Low Beta together → stronger indicator

**Thursday's results:**
- Alpha × Alpha interaction was important
- Main effects still dominated (Alpha, Theta, Beta)
- Interactions provided marginal improvement

### 6. **Cross-Validation Explained**

**5-Fold CV:**
- Split training data into 5 equal parts
- Train on 4 folds → Validate on 1 fold
- Repeat 5 times (different validation fold each time)
- Average results

**Purpose:**
- Detect overfitting
- Check data sensitivity
- Get robust performance estimate
- **Science needs generalization, not lucky patterns!**

**Professor's Stock Market Example:**
> "I made a model in 2016 that worked for a year then failed. It learned a special pattern from one fold. Science needs patterns that work across ALL folds."

---

## 📊 Comparison: Week 02 vs Week 03

| Aspect | Week 02 (Amplitude Only) | Week 03 (Frequency Features) |
|--------|-------------------------|------------------------------|
| **Features** | normalizedValue | Delta, Theta, Alpha, Beta |
| **T-test p-value** | 0.94 (not significant) | Beta: < 10⁻¹³ (highly significant!) |
| **Logistic Regression** | p=0.931 (failed) | AUC=0.75+ (success!) |
| **Best Model** | None worked | Stacked Ensemble AUC≈1.0 |
| **Conclusion** | Single amplitude insufficient | Frequency engineering works! |

**The Breakthrough:**
Week 02 showed us amplitude alone doesn't work. Week 03 proves frequency features DO work!

---

## 🔑 Important Concepts

### 1. Ensemble Learning

**What is a Stacked Ensemble?**
1. Train multiple base models (GLM, GBM, XGBoost, DRF, XRT)
2. Collect predictions from all base models
3. Train **meta-learner** (another GBM) on base model predictions
4. Meta-learner learns optimal weights for combining base models
5. Final prediction = weighted combination of base model predictions

**Why it works:**
- Different models capture different patterns
- Ensemble captures all patterns
- More robust than any single model
- Reduces overfitting risk

### 2. Ground Truth Labeling for Windows

**Our Simple Approach (Tuesday):**
```python
# Divide at 50% point
half_windows = num_windows // 2
non_seizure_segments = windows[0:half_windows]        # Label 0
seizure_segments = windows[half_windows:num_windows]  # Label 1
```

**Better Approaches:**
- Check if window center time > seizure_start_time
- Use majority vote (>50% of window in seizure)
- Label any window touching seizure as positive
- Use expert annotations if available

### 3. Linear vs Non-Linear Models

**Linear (GLM):**
```
log(odds) = β₀ + β₁·delta + β₂·theta + β₃·alpha + β₄·beta
```
- Simple addition of weighted features
- Interpretable coefficients
- Limited expressiveness
- AUC = 0.75

**Non-Linear (GBM, XGBoost, RF):**
- Can learn: feature interactions, thresholds, non-linear relationships
- More powerful pattern recognition
- Less interpretable
- AUC > 0.75

**Best of Both:**
- Use GLM to understand feature importance
- Use ensemble for best performance
- Report both results!

---

## 🙋 Frequently Asked Questions

**Q: Why are there only 4 submissions for Assignment 0?**  
**A:** Common issue was memory crashes in Colab when processing full dataset. Solution: subset to first 20 seconds like in class examples.

**Q: The solution is already provided - should I still do Assignment 0?**  
**A:** Yes! Learning comes from the struggle, not from seeing the answer. Try it yourself, then check your solution.

**Q: Why does H2O Flow say "localhost" but we can't access it directly?**  
**A:** Colab runs in Google's cloud servers. The `output.serve_kernel_port_as_window()` function creates a secure tunnel to access the UI.

**Q: What's the difference between edge-to-edge and overlapping windows?**  
**A:** 
- **Edge-to-edge:** Windows don't overlap - might miss boundary events
- **50% overlap:** Each window overlaps half with previous - standard practice
- **75% overlap:** Even more overlap - better resolution but more computation

**Q: Why did both parametric and non-parametric tests agree this time?**  
**A:** Our frequency features have distributions that aren't severely skewed, so both test types work. With amplitude (Week 02), distributions were problematic.

**Q: What if I want to use only Beta and Alpha features?**  
**A:** Great experiment! In H2O, just select only those columns as predictors. Likely will still get good performance since they're the strongest features.

**Q: How do I export H2O models?**  
**A:** Click "Download Model" in Flow UI, or use command:
```python
!cp /content/models/ModelName /content/drive/MyDrive/Week03/
```

---

## 📚 Additional Resources

### H2O Documentation:
- [H2O AutoML Guide](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
- [H2O Flow UI Tutorial](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/flow.html)
- [H2O Python API](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/index.html)

### Statistical Testing:
- [Mann-Whitney U Test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)
- [When to Use Parametric vs Non-Parametric Tests](https://www.statisticshowto.com/parametric-vs-non-parametric/)
- [Understanding Box Plots](https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51)

### Machine Learning:
- [Ensemble Learning Explained](https://towardsdatascience.com/ensemble-learning-stacking-blending-voting-b37737c4f483)
- [ROC and AUC Explained](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)

---

## 🎓 Professor's Notes

**The Week 03 Story:**

We started Week 01 trying to detect seizures from raw EEG signals. We failed.

Week 02 Tuesday, we engineered frequency features. Week 02 Thursday, we proved amplitude alone fails (p=0.931).

Week 03 Tuesday, **we succeeded!** Using Beta and Alpha frequency features, we built models with AUC ≈ 1.0.

**What changed?** Feature engineering. 

**The lesson:** In AI/ML, **features matter more than algorithms**. You can have the fanciest deep learning model, but if your features are poor, you'll fail. Conversely, with good features, even simple models (GLM) perform well.

**This is why we spent two weeks on feature engineering before jumping to modeling!**

**For Thursday:**
> "We're going to talk about log-odds and derive this thing. We're going to talk about how to understand ROC curves. Finally, we'll talk about cross-validation in a very deliberate way. Understanding what the model has learned and how best to use it and operationalize it - these are the things we will learn."

**Week 03 Key Takeaways:**

**From Tuesday:**
1. Frequency features (Beta, Alpha) are highly significant (p < 10⁻¹³)
2. H2O AutoML builds multiple models automatically
3. Stacked Ensemble achieved AUC ≈ 1.0 (near perfect!)
4. Windowing with 50% overlap is standard practice

**From Thursday:**
1. Log-odds transformation enables us to use OLS for classification
2. Sigmoid function converts linear predictions → probabilities
3. ROC curves show performance across all thresholds
4. Validation set determines optimal operating point (0.57-0.58)
5. IID assumption justifies random train/val/test splits
6. Cross-validation detects overfitting and ensures generalization

**Professor's Final Thought:**
> "Understanding the math helps you use it correctly. It's not just clicking buttons - it's understanding what those buttons do mathematically!"

**Professor Prahlad Menon, PhD, PMP**  
*Office Hours: By appointment*  
*Email: prm44@pitt.edu*

---

*"Understanding what the model has learned and how best to use it and operationalize it - these are the things we will learn."*

---

## 📋 Week 03 Completion Checklist

- [ ] `git pull` to get Week 03 content
- [ ] Watch both lecture recordings (Tuesday + Thursday)
- [ ] Run `Process_session4_train_2018.ipynb` completely
- [ ] Install and configure H2O in Colab
- [ ] Access H2O Flow UI successfully
- [ ] Build GLM with feature interactions
- [ ] Understand log-odds derivation
- [ ] Interpret ROC curves and select optimal threshold
- [ ] Understand train/validation/test split strategy
- [ ] Complete Assignment 0 (even with solution available)
- [ ] Complete Assignment 1 (feature engineering)
- [ ] Complete Assignment 2 (H2O classification)
- [ ] Write reports for all assignments

---

## 🎬 Next Week Preview: Week 04

**Topics:**
- Dimensionality reduction (PCA, LDA)
- Visualizing high-dimensional feature spaces
- Feature selection vs feature extraction
- Competition dataset introduction
- Advanced ensemble methods

**Assignments:**
- Assignment 3: Will involve dimensionality reduction
- Ongoing: Assignments 0, 1, 2 (if not completed)

---

**Have a great weekend! Work on Assignments 0, 1, and 2! See you next week for dimensionality reduction!** 🚀
