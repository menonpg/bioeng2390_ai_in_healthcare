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

### Lecture 6 - January 29, 2026
**Focus:** ROC Curves, Logistic Regression Math & Cross-Validation

**Planned Topics:**
- Deriving the logit link function
- Understanding log-odds and probabilities
- ROC curves and AUC interpretation
- Threshold selection strategies
- K-fold cross-validation explained
- Model evaluation beyond accuracy

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

## 🎬 Thursday Preview

### Topics for January 29:

**1. Logistic Regression Mathematics**
- Deriving the logit link function
- Understanding log-odds: `log(p/(1-p)) = β₀ + β₁X`
- Why linear regression fails for binary outcomes
- Connecting to Week 02's OLS derivation

**2. ROC Curves Explained**
- What is TPR vs FPR?
- How to read ROC curves
- Why AUC = area under ROC curve
- Selecting optimal probability thresholds
- Sensitivity-specificity tradeoff revisited

**3. Cross-Validation Deep Dive**
- K-fold CV step-by-step
- Why we need CV (overfitting detection)
- Interpreting CV standard deviations
- When to trust model performance

**4. Model Interpretability**
- Understanding GLM coefficients
- Variable importance plots
- Partial dependence plots (if time permits)
- Making model predictions actionable

### Prepare by:
- Completing Assignment 0
- Running H2O AutoML yourself
- Reviewing Week 02 confusion matrix concepts
- Thinking about: What does "probability of seizure" mean clinically?

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

**Professor Prahlad Menon, PhD, PMP**  
*Office Hours: By appointment*  
*Email: prm44@pitt.edu*

---

*"Linear models are interpretable. Non-linear models perform better. Ensembles give us the best of both worlds."*

---

## 📋 Week 03 Tuesday Checklist

- [ ] `git pull` to get Week 03 content
- [ ] Run `Process_session4_train_2018.ipynb` in Colab
- [ ] Install H2O: `!pip install h2o`
- [ ] Initialize H2O and access Flow UI
- [ ] Import frequency features CSV
- [ ] Run AutoML with 5-fold CV
- [ ] Examine leaderboard and compare models
- [ ] Review variable importance from GLM
- [ ] Understand why Beta and Alpha are strongest
- [ ] Complete Assignment 0 (if not done)
- [ ] Save H2O models to Google Drive

**Ready for Thursday?** Review confusion matrices and think about probability thresholds!

---

**Have a great week! See you Thursday for ROC curves and the math behind logistic regression!** 🚀
