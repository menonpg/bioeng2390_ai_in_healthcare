# BIOENG-2390 Spring 2026 - Lecture 6
## January 29, 2026

**Instructor:** Professor Prahlad G. Menon, PhD, PMP  
**Email:** menon.prahlad@gmail.com  
**Recording:** [View on Fathom](https://fathom.video/share/xTNg5omfgG1KEP6erJbD7zZcQBTFxQn9)  
**Duration:** 89 minutes

---

## 📋 Lecture Overview

Today's class focused on:
1. Deriving logistic regression mathematically (log-odds transformation)
2. Understanding ROC curves and AUC interpretation
3. Train/Validation/Test splits (70/20/10)
4. Optimal threshold selection using validation set
5. Building GLM models with feature interactions in H2O
6. Variable importance and coefficient interpretation
7. Model predictions on test data
8. Comparing training, validation, and test performance

---

## 🎯 Key Mathematical Derivation: Logistic Regression

### The Complete Derivation

**Starting Point: Sigmoid Function**
```
P(Y=1|X) = 1 / (1 + e^(-x))
```

**Step 1: Express for our model**
```
P(Y=1|X₁,X₂,...,Xₙ) = 1 / (1 + e^(-(β₀ + Σβᵢ·Xᵢ)))
```

**Step 2: Invert both sides**
```
1/P = 1 + e^(-(β₀ + Σβᵢ·Xᵢ))
```

**Step 3: Isolate exponential term**
```
e^(-(β₀ + Σβᵢ·Xᵢ)) = (1/P) - 1 = (1-P)/P
```

**Step 4: Take natural log**
```
-(β₀ + Σβᵢ·Xᵢ) = ln((1-P)/P)
```

**Step 5: Multiply by -1**
```
β₀ + Σβᵢ·Xᵢ = -ln((1-P)/P) = ln(P/(1-P))
```

**Final Result: LOG-ODDS**
```
ln(P/(1-P)) = β₀ + Σβᵢ·Xᵢ
```

**Where:**
- **P/(1-P)** = Odds ratio
- **ln(P/(1-P))** = Log-odds
- **Right side** = Linear combination of predictors (like OLS!)

**Why this matters:**
> "We know how to solve equations that look like β₀ + Σβᵢ·Xᵢ using OLS from Week 02! We just transformed the probability problem into a linear problem we can solve."

---

## 📊 Understanding the Sigmoid Function

### The Transfer Function

**Equation:**
```
σ(x) = 1 / (1 + e^(-x))
```

**Properties:**
- Output ranges from 0 to 1 (perfect for probabilities!)
- S-shaped curve
- Smooth transition
- Steepness can be adjusted

**What it does:**
- Converts continuous values (-∞ to +∞) → probabilities (0 to 1)
- **Activates** predictions into binary decisions
- Also called: **activation function**, **transfer function**

**Threshold Selection:**
```
If P(Y=1) > threshold → Predict Seizure (1)
If P(Y=1) ≤ threshold → Predict Normal (0)
```

**Common thresholds:**
- 0.5 (default - balanced)
- 0.4 (favor sensitivity)
- 0.6 (favor specificity)
- **Optimal:** Determined from validation set!

---

## 🎯 Train/Validation/Test Split

### The 70/20/10 Split

**Training Set (70%):**
- Used to **learn** model parameters (β₀, β₁, β₂...)
- Fit the betas using OLS-style optimization
- Largest portion of data

**Validation Set (20%):**
- Used to **optimize** threshold/operating point
- Select best model from multiple candidates
- Tune hyperparameters
- **Key use:** Find optimal probability cutoff

**Test Set (10%):**
- **Final evaluation** on completely unseen data
- Apply both model AND threshold from validation
- True out-of-sample performance
- Only use once at the end!

**Why three splits?**
> "I train a model and then I apply the model to the validation set to see what optimal threshold I could use. Then from there, I might identify the threshold as 0.4. That 0.4 is the operating point. This operating point, as well as the model together, are used to determine the performance in the out-of-sample test set."

---

## 📈 ROC Curves Explained

### What is an ROC Curve?

**ROC = Receiver Operating Characteristic**

**Axes:**
- **X-axis:** False Positive Rate (FPR) = FP / (FP + TN)
- **Y-axis:** True Positive Rate (TPR) = TP / (TP + FN) = Sensitivity

**How it's created:**
1. For each possible threshold (0.0, 0.01, 0.02, ..., 0.99, 1.0)
2. Create confusion matrix
3. Calculate TPR and FPR
4. Plot point (FPR, TPR)
5. Connect all points

**Interpretation:**
```
Perfect Classifier: Hugs top-left corner (TPR=1, FPR=0)
Random Guessing: Diagonal red line (45° angle)
Our GLM: Blue curve between them
```

**AUC (Area Under Curve):**
- **1.0** = Perfect (blue curve reaches top-left)
- **0.8-0.9** = Excellent
- **0.7-0.8** = Good
- **0.5** = Random (diagonal line)
- **< 0.5** = Worse than random (model inverted!)

**From Today's GLM:**
- Training AUC: 0.81
- Validation AUC: ~0.81
- Test AUC: Similar

---

## 🎯 Threshold Selection Strategy

### Results from Thursday's GLM

**Training Set Optimal Threshold:**
- Value: 0.61 (max min-per-class accuracy)
- Criterion: Balances accuracy for both classes

**Validation Set Optimal Threshold:**
- Value: 0.56
- Better estimate (unseen data)

**Cross-Validation Optimal Threshold:**
- Value: 0.58
- Most robust (averaged across 5 folds)

**Final Choice: 0.57-0.58**
> "Somewhere between 0.56 and 0.59 is a good threshold. Maybe 0.58 is the best threshold from cross-validation."

**Applied to Test Set:**
At threshold ≈ 0.57:
- **TPR**: 1.0 (100% - catching all seizures!)
- **FPR**: 0.54 (54% - some false alarms)

**Trade-off:**
- Catching all true seizures (excellent for patient safety!)
- But generating false alarms (~50%)
- Acceptable for medical screening (better safe than sorry)

---

## 🔬 Feature Interactions

### What are Interactions?

**Without Interactions:**
```
log(odds) = β₀ + β₁·delta + β₂·theta + β₃·alpha + β₄·beta
```

**With Interactions:**
```
log(odds) = β₀ + β₁·delta + β₂·theta + β₃·alpha + β₄·beta
           + β₅·(delta×theta) + β₆·(theta×alpha) + β₇·(alpha×beta) + ...
```

**Why add interactions?**
- Capture combined effects of features
- Example: Maybe low Alpha AND low Beta together = stronger seizure indicator
- Can improve model performance
- More features = more complex model

**In H2O Flow:**
- Check "Interactions" checkbox
- Select "pairwise" or specify pairs
- H2O automatically creates interaction terms

**From Today:**
- Alpha × Alpha interaction was important
- Several other interactions contributed
- Variable importance showed both main effects and interactions

---

## 💻 H2O GLM Configuration

### Model Settings from Thursday

```python
# Model name
model_id = "GLM01"

# Data splits
training_frame = train_data      # 70%
validation_frame = valid_data    # 20%
# test set held out for final evaluation

# Response and predictors
y = "seizure"
x = ["delta", "theta", "alpha", "beta"]

# Model options
nfolds = 5                      # 5-fold cross-validation
compute_p_values = True         # Get statistical significance
interactions = True              # Include interaction terms
max_runtime_secs = 600          # 10 minutes max
```

**Export Model:**
```python
# In H2O Flow: Click "Download Gen Model"
# Or export to specific location

# Copy to Google Drive (via terminal/code cell):
!cp /content/models/GLM01 "/content/drive/MyDrive/.../Week03/"
```

**Note on Spaces in Paths:**
- Use double quotes if folder names contain spaces
- H2O may not like exporting directly to Google Drive
- Export to `/content` first, then copy to Drive

---

## 📊 Model Results from Thursday

### GLM with Interactions

**Performance Metrics:**
```
Dataset        AUC    Optimal Threshold
-----------------------------------------
Training      0.81         0.61
Validation    0.81         0.56
Cross-Val     0.81         0.58
Test          ~0.81        (applied 0.57)
```

**Variable Importance (Top Features):**
1. **Alpha** (main effect) - Negative correlation
2. **Theta** - Positive correlation
3. **Alpha × Alpha** (interaction)
4. **Other interactions**
5. **Beta**, **Delta**

**Coefficient Signs:**
- **Alpha ↑** → Seizure likelihood ↓ (negative β)
- **Theta ↑** → Seizure likelihood ↑ (positive β)
- Matches box plot observations from Tuesday!

**Confusion Matrix on Test Set (threshold=0.57):**
- True Positives: High (TPR = 1.0)
- False Positives: Moderate (FPR = 0.54)
- Excellent sensitivity, acceptable specificity for medical screening

---

## 🔑 Key Concepts

### 1. IID Assumption (Independent and Identically Distributed)

**Question from Class:** "Does H2O respect the temporal nature of samples?"

**Answer:** No, because we assume windows are IID!

**Why this is okay:**
- We already handled time in feature engineering
- Each 1-second window analyzed for frequency content
- Windows treated as independent observations
- **Assumption:** One window's seizure state doesn't depend on previous window

**When IID fails:**
- Sequential time series forecasting
- When next observation depends on previous
- Then: Use time-series specific methods (ARIMA, LSTMs)

**For our seizure detection:**
- Windows ARE independent observations of brain state
- IID assumption is reasonable
- Random shuffling into train/val/test is valid

### 2. Operating Point vs Model

**The Model:**
- Learns β coefficients
- Predicts probabilities P(Y=1)
- **Output:** Continuous value from 0 to 1

**The Operating Point:**
- Threshold for converting P → binary decision
- Example: threshold = 0.58
- **Output:** Binary classification (0 or 1)

**Why separate?**
- Model is fixed once trained
- Operating point can be adjusted for different use cases
- COVID screening: Lower threshold (favor sensitivity)
- Seizure confirmation: Higher threshold (favor specificity)

### 3. Cross-Validation Purpose

**5-Fold Cross-Validation:**
1. Split training data into 5 equal parts
2. Train on folds {1,2,3,4} → Validate on fold {5}
3. Train on folds {2,3,4,5} → Validate on fold {1}
4. Train on folds {3,4,5,1} → Validate on fold {2}
5. Train on folds {4,5,1,2} → Validate on fold {3}
6. Train on folds {5,1,2,3} → Validate on fold {4}
7. Average performance across all 5 experiments

**Why?**
- Detect if model is overfitting
- Check if performance depends on specific data fold
- More robust estimate of generalization
- **Science seeks generalization, not lucky patterns!**

**Professor's Stock Market Story:**
> "I made a model in 2016 that identified head-and-shoulders patterns and made money for a year, then stopped working. That was learning a SPECIAL pattern from one fold. In science, we want GENERAL patterns that work across all folds."

---

## 💡 Assignment Expectations

### Assignment 0, 1, 2 Requirements

**What to Submit:**
1. ✅ **Jupyter Notebook** with code and outputs
2. ✅ **H2O Flow file** (.flow export) if used
3. ✅ **Written Report** (Word/PDF)

**Report Must Include:**
- What you understood at each step
- Explanation of your choices
- Results and interpretation
- Challenges faced and solutions

**Professor's Emphasis:**
> "No report, no points! You need to explain what you learned and how you executed it. Even though I don't explicitly write this in assignment instructions, this IS the instruction."

### Assignment Status

**Assignment 0:** ~7 students completed (out of class)
- **Tip:** Subset to first 20 seconds to avoid memory crashes
- **Solution:** Available in `Process_EEG_sleep.ipynb` (but try yourself first!)

**Assignment 1:** Available now
- Feature engineering with EEG_sleep.mat
- First 20 seconds only (~10,000 samples)
- Extract Delta, Theta, Alpha, Beta features

**Assignment 2:** Available now
- Use Assignment 1 features
- Build H2O models
- Classify seizure states
- Evaluate performance

**No Quiz:** Despite earlier confusion, no quiz this week

---

## 🔬 Complete Logistic Regression Framework

### From Probability to Log-Odds

**1. Start with desired probability model:**
```
P(Y=1) = 1 / (1 + e^(-(β₀ + Σβᵢ·Xᵢ)))
```

**2. Problem:** Can't solve directly with OLS

**3. Transform to log-odds:**
```
ln(P/(1-P)) = β₀ + Σβᵢ·Xᵢ
```

**4. Now we can use OLS!**
- Left side = log-odds (continuous)
- Right side = linear combination of predictors
- Solve for β using (AᵀA)⁻¹AᵀY from Week 02!

**5. Convert back to probability:**
```
P = 1 / (1 + e^(-(β₀ + Σβᵢ·Xᵢ)))
```

**The Circle:**
```
Linear Regression → Can solve but wrong for binary
Log-Odds Transform → Makes problem solvable
OLS Solution → Find β coefficients
Sigmoid Transform → Convert back to probabilities
Threshold → Make binary decision
```

---

## 🎓 Practical Implementation in H2O

### Loading Data into H2O (Three Methods)

**Method 1: Upload CSV via Flow UI**
```
1. Click "Import Files"
2. Browse or paste path
3. Parse CSV
4. Set column types (numeric vs enum)
5. Import
```

**Method 2: Programmatic with h2o.import_file()**
```python
data = h2o.import_file(path, 
                       col_types={'seizure': 'enum'})
```

**Method 3: Convert from Pandas**
```python
segments_h2o = h2o.H2OFrame(segments_df)
segments_h2o['seizure'] = segments_h2o['seizure'].asfactor()
```

**Pro Tip:** Specify `col_types` to avoid manual enum conversion!

---

### Data Splitting

**Programmatic Split:**
```python
# 70/20/10 split
train, valid, test = segments_h2o.split_frame(
    ratios=[0.7, 0.2],
    destination_frames=['train_data', 'valid_data', 'test_data'],
    seed=42
)
```

**Why seed=42?**
- Makes split reproducible
- Same data goes to same split each time
- Different seed = different random assignment

**Result:**
- Training: 176 windows
- Validation: 48 windows  
- Test: 25 windows

---

### Building GLM with Interactions

**In H2O Flow UI:**
1. Click "Build Model" → "Generalized Linear Model"
2. Set model_id: "GLM01"
3. Select training_frame: train_data
4. Select validation_frame: valid_data
5. Response column: seizure
6. Predictor columns: delta, theta, alpha, beta
7. Enable interactions: Check "interactions" box
8. Set nfolds: 5 (for cross-validation)
9. Enable compute_p_values: Yes
10. Set max_runtime_secs: 600
11. Build Model!

**What H2O Does:**
- Fits logistic regression
- Creates interaction terms automatically
- Performs 5-fold CV on training data
- Computes optimal threshold on validation data
- Generates ROC curves for all sets
- Calculates variable importance

---

## 📊 Understanding Model Output

### Variable Importance

**From Today's GLM:**
```
Rank  Feature          Coefficient  Interpretation
----------------------------------------------------
1     alpha           -2.9         ↑Alpha → ↓Seizure
2     theta           +2.6         ↑Theta → ↑Seizure  
3     alpha×alpha     varies       Interaction effect
4     beta            negative     ↑Beta → ↓Seizure
5     delta           varies       Weaker effect
```

**Negative Coefficient:**
- When feature increases, seizure probability decreases
- Alpha and Beta have negative coefficients
- **Makes sense:** Higher high-frequency power in normal states

**Positive Coefficient:**
- When feature increases, seizure probability increases
- Theta has positive coefficient

---

### Optimal Thresholds

**Finding the Best Threshold:**

**Training Set:** 0.61
- Based on max min-per-class-accuracy
- Balances accuracy for both classes

**Validation Set:** 0.56 ⭐
- Most trustworthy (unseen data)
- Use this for final model!

**Cross-Validation:** 0.58
- Average across 5 folds
- Robust estimate
- Right between training (0.61) and validation (0.56)

**Recommendation:** Use 0.57-0.58 as operating point

**At threshold = 0.57 on test set:**
- TPR = 1.0 (100% sensitivity!)
- FPR = 0.54 (54% false positive rate)
- **Trade-off:** Perfect seizure detection, but some false alarms

---

## 🔬 Linear vs Non-Linear Models

### Why GLM (Linear) Had AUC=0.81, Not 1.0

**GLM Assumes:**
```
log(odds) = β₀ + β₁·delta + β₂·theta + β₃·alpha + β₄·beta + interactions
```

**This is still linear** even with interactions!
- Features added together (with weights)
- Interactions are pre-computed products
- Can't learn: thresholds, decision boundaries, complex patterns

**Non-Linear Models (from Tuesday):**
- **XGBoost, GBM, Random Forest:** Can learn complex decision rules
- **Stacked Ensemble:** Combines multiple non-linear models
- **Result:** AUC ≈ 1.0 (much better!)

**Lesson:**
> "Non-linear models can capture the non-linear effects of predictors on response better than linear models with or without interactions."

---

## 🎯 Making Predictions

### Using the Trained Model

**In H2O Flow:**
1. Select your GLM model
2. Click "Predict"
3. Select test data frame
4. Run prediction
5. View results

**Output:**
- **predict:** Binary decision (0 or 1) using default/optimal threshold
- **p0:** Probability of class 0 (Normal)
- **p1:** Probability of class 1 (Seizure)

**Interpreting Results:**
```
Window  p0     p1     predict  actual  correct?
------------------------------------------------
1       0.95   0.05   0        0       ✓
2       0.30   0.70   1        1       ✓
3       0.45   0.55   1        0       ✗ (FP)
4       0.60   0.40   0        1       ✗ (FN)
```

**Adjust Threshold:**
- Lower threshold (0.4) → More predictions of "1" → Higher sensitivity
- Higher threshold (0.7) → Fewer predictions of "1" → Higher specificity

---

## 🙋 Questions from Class

**Q: Should we still do Assignment 0 if the solution is provided?**  
**A:** YES! The learning comes from the struggle, not from seeing the answer. Try it yourself, then verify against the solution.

**Q: Why use 70/20/10 split instead of just train/test?**  
**A:** 
- **With validation:** Can optimize threshold AND evaluate fairly
- **Without validation:** Either optimize on test (unfair) or use default threshold (suboptimal)
- Validation set lets us tune without "peeking" at test set

**Q: What if I want to change the threshold after seeing test results?**  
**A:** Don't! That would be cheating. Test set should only be used once for final evaluation. If you want to tune more, use validation set.

**Q: Why does Gemini/AI sometimes give wrong code?**  
**A:** LLMs can "hallucinate" functions that don't exist (like `assign_key()`). Always test code and verify against documentation. AI is a tool, not infallible.

**Q: Can I run H2O on my local computer instead of Colab?**  
**A:** Yes! Install with `pip install h2o`, then access Flow UI at `http://localhost:54321` in your browser. Much faster than Colab!

---

## 💡 Practical Tips from Lecture

### Notebook Organization

**Use Markdown Headers to Collapse Sections:**
```markdown
# Main Heading (one hashtag)
## Subheading (two hashtags)  
### Sub-subheading (three hashtags)
```

**Benefit:**
- Click arrow to collapse entire section
- Easier navigation in long notebooks
- Better organization

### H2O Frame Management

**Delete unwanted frames:**
```python
# In Flow UI:
# 1. Data → Get Frames
# 2. Select frames to delete (click checkboxes)
# 3. "Delete Selected Frames"
```

**Name your frames meaningfully:**
```python
h2o.H2OFrame(df, destination_frame="EEG_segments_70_20_10_split")
```

### Troubleshooting H2O Flow

**Problem:** localhost:54321 link doesn't open  
**Solution:** Click the link (don't copy-paste). It auto-generates a new URL.

**Problem:** Using `as_window` parameter as iframe  
**Solution:** Keep it as "window" (default). iframe embedding may not work in Colab.

**Problem:** Can't export to Google Drive  
**Solution:** Export to `/content/models/` first, then copy with `!cp` command.

---

## 📚 Connecting the Weeks

### The Complete Journey

**Week 01:**
- Setup environments
- Load EEG data
- Visualize signals
- **Problem:** Can't distinguish seizures visually at single time points

**Week 02:**
- Engineer frequency features (Delta, Theta, Alpha, Beta)
- Prove amplitude alone fails (p=0.94, p=0.931)
- Learn OLS regression math
- **Discovery:** Frequency features show promise

**Week 03:**
- Build classification models with frequency features
- **Tuesday:** AutoML finds best models (Ensemble AUC≈1.0)
- **Thursday:** Understand the math (log-odds derivation)
- **Success:** Can detect seizures with high accuracy!

**The Lesson:**
1. Domain knowledge → Feature engineering
2. Feature engineering → Successful models
3. Understanding math → Proper model use
4. Validation strategy → Trustworthy results

---

## 🎬 Next Week Preview

**Week 04:**
- Dimensionality reduction (PCA, LDA)
- Visualizing high-dimensional data
- Feature selection strategies
- More advanced modeling techniques
- Competition dataset introduction

---

## 📋 Week 03 Thursday Checklist

- [ ] Understand log-odds derivation
- [ ] Understand sigmoid transformation
- [ ] Know how to interpret ROC curves
- [ ] Understand AUC metric
- [ ] Know difference between train/validation/test
- [ ] Understand IID assumption
- [ ] Can build GLM with interactions in H2O
- [ ] Can select optimal threshold from validation set
- [ ] Complete Assignments 0, 1, and 2
- [ ] Write reports explaining your understanding

---

## 🎓 Professor's Final Thoughts

**On Mathematics:**
> "We spent time deriving log-odds because understanding WHY logistic regression works helps you use it correctly. It's not just clicking buttons - it's understanding what those buttons do!"

**On Assignments:**
> "Your assignments have liberty in what you do specifically, as long as you get the task done. But you MUST provide code, H2O flow, and a report explaining your understanding."

**On Learning:**
> "Even though solutions are provided, the learning comes from trying it yourself. Don't rob yourself of the learning experience!"

**Professor Prahlad Menon, PhD, PMP**  
*Office Hours: By appointment*  
*Email: prm44@pitt.edu*

---

*"Understanding what the model has learned and how best to use it and operationalize it - these are the things we will learn."*

---

## 🔑 Key Takeaways

1. ✅ Log-odds = ln(P/(1-P)) = linear combination of predictors
2. ✅ Sigmoid function converts linear → probability
3. ✅ ROC curves plot TPR vs FPR across all thresholds
4. ✅ AUC measures overall classification performance
5. ✅ Validation set determines optimal threshold
6. ✅ Test set gives final unbiased performance
7. ✅ IID assumption justifies random data splits
8. ✅ Cross-validation detects overfitting
9. ✅ Feature interactions can improve models
10. ✅ Understanding math enables proper model use

**Have a great weekend! Work on Assignments 0, 1, and 2! See you next week!** 🚀
