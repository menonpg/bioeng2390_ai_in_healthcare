# BIOENG-2390 Spring 2026 - Lecture 4
## January 22, 2026

**Instructor:** Professor Prahlad G. Menon, PhD, PMP  
**Email:** menon.prahlad@gmail.com  
**Recording:** [View on Fathom](https://fathom.video/share/otwZPqGCaqtb7yxzfrsbZ3DYK47agk5A)  
**Duration:** 86 minutes

---

## 📋 Lecture Overview

Today's class focused on:
1. Confusion matrices and performance metrics
2. Understanding true/false positives/negatives
3. Sensitivity, specificity, precision, and accuracy
4. The tradeoff between sensitivity and precision (2SD vs 1SD rules)
5. Linear regression theory and matrix derivation
6. Ordinary Least Squares (OLS) solution
7. Implementing regression in R and MATLAB
8. Introduction to logistic regression for classification

---

## 🎯 Key Concepts

### 1. **Confusion Matrix**

A cross-tabulation of **Actual (Ground Truth)** vs **Predicted** outcomes.

**Important:** The layout can vary! Sometimes actuals are rows, sometimes columns. Always check the labels!

**Our Layout (from Tuesday's R analysis):**
```
                    ACTUAL
              Normal    Seizure
PREDICTED  
In-Control   5,984     3,220    (TN)  (FN)
Out-Control     17       779    (FP)  (TP)
```

**Terminology:**
- **True Positives (TP)**: 779 - Correctly identified seizures
- **True Negatives (TN)**: 5,984 - Correctly identified normal states  
- **False Positives (FP)**: 17 - Normal incorrectly flagged as seizure
- **False Negatives (FN)**: 3,220 - Seizures missed by the rule

**Critical Insight:**
> "Whether it's predicted positive or predicted negative, as long as the actuals row has the word 'seizure' in it, it is an actual seizure event."

---

### 2. **Performance Metrics**

#### Sensitivity (Recall / True Positive Rate)
**Formula:** TP / (TP + FN) = 779 / (779 + 3,220) = **19.48%**

**Meaning:** Of all actual seizures, how many did we catch?
- **Low sensitivity** = Missing many seizures (bad for COVID test!)
- **High sensitivity** = Catching most seizures (better for screening)

#### Specificity (True Negative Rate)
**Formula:** TN / (TN + FP) = 5,984 / (5,984 + 17) = **99.72%**

**Meaning:** Of all actual normal states, how many did we correctly identify as normal?
- **High specificity** = Few false alarms (good!)

#### Precision (Positive Predictive Value - PPV)
**Formula:** TP / (TP + FP) = 779 / (779 + 17) = **97.86%**

**Meaning:** When we predict seizure, how often are we correct?
- **Fishing analogy:** "If I threw a line in the water, I'd catch a fish 97.86% of the time"
- **High precision** = Predictions are reliable

#### Negative Predictive Value (NPV)
**Formula:** TN / (TN + FN) = 5,984 / (5,984 + 3,220) = **65.00%**

**Meaning:** When we predict normal, how often are we correct?
- **Lower NPV** = Many false negatives slip through

#### Accuracy
**Formula:** (TP + TN) / Total = (779 + 5,984) / 10,000 = **67.63%**

**Meaning:** Overall, how often is the model correct?

---

### 3. **The Sensitivity-Precision Tradeoff**

**Professor's COVID Example:**
> "In a pandemic, I'd rather be safe than sorry. I'd rather see a large number of false positives and let people shelter in place at home than let a bunch of people think they're okay (false negatives) but actually they're positive."

**Question posed:** If this signal represented a COVID test instead of seizure detection, what would you do?

**Answer:** **Decrease sigma** (e.g., from 2σ to 0.5σ) to flag MORE patients

**Why?**
- Increases sensitivity (catch more true cases)
- Decreases precision (more false alarms)
- Better to quarantine healthy people than miss infected ones!

**Results from 1-Sigma Rule:**
- **Sensitivity**: Improved from 19% → 39% ✓
- **Precision**: Decreased from 97.86% → lower (more false positives)
- **Tradeoff**: More true positives caught, but at cost of more false alarms

**Key Insight:**
> "The appropriate threshold for a given statistical test is greatly a function of the purpose of that test."

---

### 4. **Linear Regression Theory**

**The Normalization Equation:**
```
normalizedValue = (value - mean(value)) / sd(value)
```

**Rewritten as Linear Equation:**
```
Y = (1/σₓ)·X - (μₓ/σₓ)
Y = β₁·X + β₀
```

Where:
- `β₁ = 1/σₓ` (slope)
- `β₀ = -μₓ/σₓ` (intercept)
- X = raw value
- Y = normalized value

**The Regression Problem:**
Given observations (X₁, Y₁), (X₂, Y₂), ..., (Xₙ, Yₙ), estimate β₁ and β₀.

---

### 5. **Matrix Formulation**

**System of Equations:**
```
Y₁ = β₁·X₁ + β₀
Y₂ = β₁·X₂ + β₀
⋮
Yₙ = β₁·Xₙ + β₀
```

**Matrix Form:**
```
⎡Y₁⎤   ⎡X₁  1⎤   ⎡β₁⎤
⎢Y₂⎥ = ⎢X₂  1⎥ · ⎢β₀⎥
⎢⋮ ⎥   ⎢⋮   ⋮⎥   ⎣  ⎦
⎣Yₙ⎦   ⎣Xₙ  1⎦

Y = A · β
```

Where:
- **Y**: n×1 vector of responses (observed outputs)
- **A**: n×2 matrix of predictors (observed inputs + ones column)
- **β**: 2×1 vector of coefficients (unknowns to solve for)

---

### 6. **Ordinary Least Squares (OLS) Solution**

**Formula:**
```
β = (AᵀA)⁻¹AᵀY
```

**This works for:**
- Exact fit (when points lie perfectly on a line)
- **Best fit** (when data has noise/scatter)

**When you have more equations than unknowns** (n > 2), OLS finds the best-fit solution that minimizes squared errors.

---

### 7. **Implementation in R**

**File:** `linearRegressionFit.R`

```r
# Fit linear model: normalizedValue ~ value
model <- lm(normalizedValue ~ value, data = s_of_t_subset_CLEAN)
summary(model)
```

**Results:**
```
Coefficients:
              Estimate    Std. Error  Pr(>|t|)    
(Intercept)  -0.063866   0.001369    < 2e-16 ***
value         0.014400   0.000031    < 2e-16 ***
```

**Interpretation:**
- **β₁ (slope)** = 0.0144 = 1/σ  
  → σ = 1/0.0144 = **69.44** (standard deviation)
  
- **β₀ (intercept)** = -0.0639 = -μ/σ  
  → μ = -(-0.0639) × 69.44 = **4.44** (mean)

**Verification:**
```r
mean(s_of_t_subset_CLEAN$value)  # Should be ~4.44
sd(s_of_t_subset_CLEAN$value)    # Should be ~69.44
```

**Model Quality:**
- **R-squared**: 1.000 (perfect fit!)
- **p-values**: < 2e-16 (highly significant)
- **Conclusion**: Normalization is a perfect linear transformation

---

### 8. **Implementation in MATLAB**

**File:** `OLSsolution.m`

**Simulate Data with Noise:**
```matlab
% True parameters
beta1 = 5;
beta0 = 10;

% Generate X values
X = linspace(0, 1, 100);

% Generate Y with noise
Y = beta1*X + beta0 + (rand(size(X))-0.5)*100;

% Prepare matrices
Y = Y';  % Transpose to column vector
A = [ones(100,1), X'];  % Design matrix

% OLS Solution (two methods)
beta_solution = inv(A'*A)*A'*Y;      % Formula method
beta_builtin = A\Y;                   % MATLAB built-in

% Results
disp(beta_solution);  % [β₀; β₁] ≈ [10; 5]
```

**Results (example run):**
```
beta_solution = [10.3051; 4.9177]
```

Close to true values [10; 5] despite noise!

**Key Insight:**
> "As long as the noise is well-behaved (e.g., normally distributed white noise), you will be able to get estimates fairly close to your actual values."

---

### 9. **Logistic Regression (Introduction)**

**The Classification Problem:**
Predict categorical outcome (Normal/Seizure) from continuous predictor (normalizedValue).

**Challenge:** Can't directly use Y = β₁X + β₀ for binary outcomes (0/1)

**Solution:** Logit Link Function
- Transform binary outcome → continuous log-odds
- Fit regression on log-odds
- Convert back to probability

**R Implementation:**
```r
# Create binary indicator: 1 = Seizure, 0 = Normal
s_of_t_subset_CLEAN$GT_binary <- ifelse(s_of_t_subset_CLEAN$GT == "Seizure", 1, 0)

# Fit logistic regression
logit_model <- glm(GT_binary ~ normalizedValue, 
                   data = s_of_t_subset_CLEAN,
                   family = binomial(link = "logit"))

summary(logit_model)
```

**Results:**
```
Coefficients:
                 Estimate   Std. Error  Pr(>|t|)    
(Intercept)     -0.405882   0.020156    <2e-16 ***
normalizedValue  0.001764   0.020153    0.931
```

**Interpretation:**
- **Intercept**: Highly significant (p < 2e-16)
- **normalizedValue**: **NOT significant** (p = 0.931 = 93.1%)

**Conclusion:**
> "The probability that this number (0.001764) is statistically the same as 0 is 93.1%. That means it's a terrible fit - it's really not a very good predictor at all."

**Why?**
- We already knew from Tuesday: means are identical between Normal and Seizure
- Single time-point amplitude can't distinguish states
- Need windowed features (frequency domain) instead!

---

## 📚 Regression vs Classification

### Regression
- **Predicts**: Continuous variable
- **Example**: Predicting normalized value from raw value
- **Method**: Linear regression (OLS)
- **Evaluation**: R-squared, RMSE, residual plots

### Classification  
- **Predicts**: Categorical variable
- **Example**: Predicting Normal vs Seizure from amplitude
- **Method**: Logistic regression (GLM with logit link)
- **Evaluation**: Confusion matrix, sensitivity, specificity, precision, accuracy

**Both use the same underlying math** (matrix algebra, OLS), but with different link functions!

---

## 🔑 Key Equations

### OLS Solution (General Form)
```
β = (AᵀA)⁻¹AᵀY
```

### Linear Model
```
Y = Xβ + ε
```
Where:
- Y = response vector
- X = design matrix (predictors + intercept column)
- β = coefficients to estimate
- ε = error/noise

### Logistic Regression (Preview)
```
log(p/(1-p)) = β₀ + β₁X
```
Where:
- p = probability of positive class
- p/(1-p) = odds
- log(odds) = log-odds (continuous!)

---

## 💻 Code from Today's Lecture

### R: Linear Regression
```r
# Load data
library(readr)
s_of_t_subset_CLEAN <- read_csv("s_of_t_subset_CLEAN.csv",
    col_types = cols(GT = col_factor(levels = c("Normal", "Seizure")))
)

# Fit linear model
model <- lm(normalizedValue ~ value, data = s_of_t_subset_CLEAN)
summary(model)

# Extract coefficients
beta1 <- coef(model)[2]  # Slope = 1/σ
beta0 <- coef(model)[1]  # Intercept = -μ/σ

# Back-calculate mean and SD
sigma <- 1 / beta1
mu <- -beta0 * sigma
```

### R: Logistic Regression
```r
# Create binary indicator
s_of_t_subset_CLEAN$GT_binary <- ifelse(s_of_t_subset_CLEAN$GT == "Seizure", 1, 0)

# Fit logistic model
logit_model <- glm(GT_binary ~ normalizedValue,
                   data = s_of_t_subset_CLEAN,
                   family = binomial(link = "logit"))

summary(logit_model)
```

### MATLAB: OLS with Simulated Data
```matlab
% Create data with noise
X = linspace(0, 1, 100);
beta1 = 5;
beta0 = 10;
Y = beta1*X + beta0 + (rand(size(X)) - 0.5)*100;

% Transpose to column vectors
Y = Y';
A = [ones(100,1), X'];

% OLS Solution (two methods)
beta_formula = inv(A'*A)*A'*Y;    % Using formula
beta_builtin = A\Y;                % Using MATLAB backslash

% Display results
disp('Formula method:'); disp(beta_formula);
disp('Built-in method:'); disp(beta_builtin);
```

---

## 🎓 Important Insights from Lecture

### 1. Single Time-Point Classification is Problematic

**Observation from EEG data:**
- Some time points during seizure have "normal-looking" amplitudes (false negatives)
- Some time points during normal state exceed 2σ threshold (false positives)

**Why?**
- Seizures don't uniformly increase amplitude at every instant
- Natural variation causes overlap between classes
- **Need windowed analysis** with frequency features!

### 2. Choosing Thresholds Depends on Application

**2-Sigma Rule (Conservative):**
- Sensitivity: 19.48% (low - misses many seizures)
- Precision: 97.86% (high - predictions very reliable)
- **Use when:** Cost of false alarm is high

**1-Sigma Rule (Aggressive):**
- Sensitivity: 39% (improved - catches more seizures)
- Precision: Lower (more false alarms)
- **Use when:** Cost of missed case is high (COVID!)

**Professor's Advice:**
> "If you made the threshold 0.5 sigma, you would see precision going even lower and sensitivity going even higher."

### 3. Well-Behaved Noise Enables Good Fits

**From MATLAB demonstration:**
- True parameters: β₁=5, β₀=10
- Added random noise: `(rand() - 0.5)*100`
- Recovered parameters: β₁≈4.92, β₀≈10.31

**Conclusion:**
> "As long as noise is kind of more or less well-behaved (normally distributed), you will be able to get a number fairly close to your estimate."

---

## 📊 Regression vs Classification Summary

| Aspect | Regression | Classification |
|--------|-----------|----------------|
| **Response** | Continuous | Categorical |
| **Example** | Normalized value from raw value | Normal vs Seizure |
| **Method** | Linear Regression (LM) | Logistic Regression (GLM) |
| **Link Function** | Identity (Y = Xβ) | Logit (log-odds) |
| **Evaluation** | R², RMSE | Confusion matrix, accuracy |
| **Today's Result** | R²=1.0, perfect fit | p=0.931, terrible fit |

**Why the difference?**
- Normalization IS a perfect linear transformation
- Single amplitude values CAN'T distinguish seizure states
- Need better features (frequency domain from Tuesday!)

---

## 🔬 Matrix Algebra Review

### Why Matrices?

**Professor explained:**
When you have:
- Multiple observations (n data points)
- Multiple predictors (p features)
- You get a system of n equations with p unknowns

**Matrix notation** lets us solve this elegantly:
```
Y = Aβ
β = (AᵀA)⁻¹AᵀY
```

This works whether:
- n = p (exact solution)
- n > p (overdetermined → least squares best fit)
- n < p (underdetermined → need regularization)

---

## 🎯 For Next Class (Tuesday, January 27)

### Topics to Cover:
1. Complete feature engineering discussion
2. Apply frequency features to classification
3. Build classification models using engineered features
4. Logistic regression deep dive (logit link function)
5. ROC curves and threshold selection

### Homework:
- [ ] Complete Assignment 0 (adapt notebook to EEG_sleep.mat)
- [ ] Run `AnalyzeSignalSofT.R` completely
- [ ] Run `linearRegressionFit.R` and verify coefficients
- [ ] Run `OLSsolution.m` in MATLAB Online
- [ ] Review confusion matrix concepts
- [ ] Understand sensitivity vs precision tradeoff

### Questions to Think About:
1. Why does logistic regression need a link function?
2. How do you choose between sensitivity and precision?
3. What happens if you use 3-sigma rule? 0.1-sigma rule?
4. Can you back-calculate mean and SD from the regression coefficients?

---

## 📝 Assignment 0 Reminder

**Task:** Adapt `ReadMAT_ConvertToSignalvsTime_EngineerWINDOWEDFeatures.ipynb` to use `EEG_sleep.mat`

**Main challenge:** Adjusting the `scipy.io.loadmat()` data extraction
- Different .mat structure
- Different nesting levels
- Different sampling frequency (500 Hz vs 256 Hz)

**Tip:** Use GitHub Copilot to help navigate the structure!

---

## 🙋 Questions from Class

**Q: Where do I find the R files?**  
**A:** Two places:
1. Git repository (Week02 folder) - use `git pull`
2. Google Drive (Week 02 folder)

**Q: How do I get files from Git to Posit Cloud?**  
**A:** 
1. Right-click file in VS Code
2. "Reveal in Finder" (Mac) or "Reveal in Explorer" (Windows)
3. Copy file to your Posit Cloud project folder

**Q: Can confusion matrices have different layouts?**  
**A:** Yes! Sometimes actuals are rows (our case), sometimes columns. Always check labels to identify TP, TN, FP, FN correctly!

**Q: Why does MATLAB backslash `\` work for OLS?**  
**A:** MATLAB's `A\Y` internally computes the least squares solution, equivalent to `inv(A'*A)*A'*Y` but more numerically stable.

---

## 🔗 Important Links

- **Today's Recording**: https://fathom.video/share/otwZPqGCaqtb7yxzfrsbZ3DYK47agk5A
- **COVID Breath Test Paper** (Professor's Lancet paper): Uses similar classification metrics
- **Course GitHub**: Spring2026 branch (all code pushed)
- **Google Drive**: Week 02 folder (all materials)

---

## 📚 Additional Context

### About the Lancet Paper
Professor Menon's research used volatile organic compounds (VOC) mass spectrometry from breath samples to predict COVID-19. This real-world application demonstrates:
- Importance of high sensitivity in pandemic screening
- Trade-offs in diagnostic test design
- How machine learning applies to clinical decision-making

### Why This Week Matters
We're learning the **fundamentals** that underpin all machine learning:
1. **Metrics matter**: Accuracy isn't everything
2. **Features matter**: Single variables often insufficient  
3. **Math matters**: Matrix algebra enables efficient solutions
4. **Context matters**: Application determines what "good" means

---

## 🎓 Professor's Final Thoughts

**On Complexity:**
> "I know this is a little bit complicated, getting into the weeds of this, but linear regression and logistic regression are your basic regression and classification models that we really need to understand through and through."

**Study Tips:**
- Watch the recording again
- Look at the code and run it yourself
- Make sure everything makes sense
- All code pushed to Git repository
- README will be updated

**Remember:**
- **Regression**: Continuous → Continuous
- **Classification**: Continuous/Categorical → Categorical
- Both use similar math (OLS, matrix algebra)
- Performance metrics differ

---

**Next Class:** Tuesday, January 27, 2026  
**Topics:** Feature engineering + Classification models with frequency features

---

*"The appropriate threshold for a given statistical test is greatly a function of the purpose of that test."*

— Professor Prahlad G. Menon, PhD, PMP

---

## 📋 Key Takeaways

1. ✅ Confusion matrices organize TP, TN, FP, FN
2. ✅ Sensitivity = recall = TP rate
3. ✅ Specificity = TN rate
4. ✅ Precision = PPV = prediction reliability
5. ✅ Application context determines optimal threshold
6. ✅ OLS solution: β = (AᵀA)⁻¹AᵀY
7. ✅ Regression fits continuous responses
8. ✅ Classification fits categorical responses
9. ✅ Logistic regression uses logit link function
10. ✅ Single amplitude features fail for seizure detection

**Have a great weekend! Work on Assignment 0! See you Tuesday!** 🚀
