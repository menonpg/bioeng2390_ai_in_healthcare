# Week 04: Model Inference, ROC Analysis & Future Prediction
### BIOENG 2390: AI in Healthcare - Spring 2026

**Instructor:** Professor Prahlad Menon, PhD, PMP  
**University of Pittsburgh, Department of Bioengineering**

---

## 🎯 Week 04 Overview

This week we move from model building to **model deployment** and **advanced evaluation**. We'll learn how to load saved models, run inference on new data, deeply understand ROC curves in different clinical contexts, and explore the distinction between seizure **detection** vs **prediction**. We'll also tackle the bonus challenge: Can we predict seizures before they happen?

---

## 📺 Lecture Recordings & Notes

### Lecture 7 - February 3, 2026 (67 minutes)
**Focus:** Model Loading, ROC Interpretation & Future Prediction

- **[Watch Recording](https://fathom.video/share/Esz_6RSP-DviJ8EPzufrv4dgvnbYS_1z)**
- **[Read Detailed Lecture Notes](Lecture07_Notes_Feb03_2026.md)** ← Complete transcript

**Topics Covered:**
- H2O version compatibility and management
- Loading pre-calculated features efficiently
- Loading saved H2O models for inference
- Deep ROC curve interpretation for different use cases (COVID, seizure, insurance)
- Threshold selection strategies: MaxF1, MaxF2, balanced accuracy
- Running inference (forward pass) on trained models
- Making predictions with custom thresholds
- Context-dependent operating point selection
- VS Code Colab extension for GPU access
- **Bonus Challenge:** Future seizure prediction (50 points!)

**Key Insight:**
> "Detection is nice, but prediction is what we really want. Can we use current window features to predict seizures 5-10 windows in advance? That's the real clinical value."

**Three Use Cases for ROC Thresholds:**
1. **COVID Screening:** Low threshold (0.05-0.10) → Max sensitivity, accept false positives
2. **Seizure Detection:** Medium threshold (0.55-0.60) → Balanced F1
3. **Insurance Cost Reduction:** High threshold (0.75-0.80) → Max specificity

---

### Lecture 8 - February 5, 2026 (88 minutes)
**Focus:** Overfitting, Cross-Validation & Non-Linear Models

- **[Watch Recording](https://fathom.video/share/wzY-sAzurZDhzdxsUjG-j6uojkkC3pJw)**
- **[Read Detailed Lecture Notes](Lecture08_Notes_Feb05_2026.md)** ← Complete concepts with whiteboard explanations

**Topics Covered:**
- Cross-validation deep dive: 3-fold, 5-fold, leave-one-out (LOO-CV)
- Overfitting vs underfitting with visual examples
- Curse of dimensionality: N observations > P parameters rule
- Variance inflation and feature independence (pizza analogy!)
- Non-linearity from features vs model complexity
- TensorFlow Playground demonstrations (spiral, circular, blob patterns)
- Forward pass/inference examples: LLMs, avatars, text-to-speech
- ROC curves from ANY continuous variable (not just probabilities)
- LazyPredict for quick model competitions
- K-Nearest Neighbors (KNN) introduction

**Key Discoveries:**
> "If you torture the data enough, it will confess to anything." (on overfitting)

> "ROC curves aren't just for probabilities - you can threshold ANY continuous variable and create an ROC curve!"

**LazyPredict Results:**
- KNN: AUC = 0.73 (best non-linear)
- Logistic Regression: AUC = 0.66 (linear baseline)
- **Conclusion:** Our seizure problem is non-linear!

---

## 🎯 Week Learning Objectives

By the end of this week, you will be able to:

1. ✅ Manage H2O versions for model compatibility
2. ✅ Load pre-calculated features efficiently
3. ✅ Load and reuse saved H2O models
4. ✅ Interpret ROC curves for different clinical contexts
5. ✅ Select optimal thresholds based on use case
6. ✅ Run inference/forward pass on new data
7. ✅ Make predictions with custom probability thresholds
8. ✅ Distinguish between detection and prediction
9. ✅ Understand when to use MaxF1 vs MaxF2 vs balanced accuracy

---

## 📁 Week 04 Files

### 1. `buildCompetitionModels.ipynb` 🌟
**Main notebook from Tuesday's lecture**

This notebook demonstrates the complete model deployment workflow:

**Part 1: H2O Version Management**
- Installing specific H2O version (3.46.0.9)
- Why version pinning matters for reproducibility
- Finding H2O version URLs from AWS

**Part 2: Efficient Data Loading**
- Copy pre-calculated features from Week02/Week03
- Load CSV directly (skip feature engineering)
- Convert to H2O frame with correct types

**Part 3: Loading Saved Models**
```python
# Copy model from Google Drive
!mkdir /content/models
!cp "/content/drive/MyDrive/Week03/GLM01" /content/models/

# Load into H2O
model = h2o.load_model("/content/models/GLM01")
```

**Part 4: ROC Analysis**
- Reviewing training ROC curves
- Understanding perfect vs random classifiers
- Identifying optimal thresholds for different contexts

**Part 5: Running Inference**
```python
# Make predictions
predictions = model.predict(test_h2o)

# Apply custom threshold
review_df = pd.DataFrame({
    'actual': test_h2o['seizure'],
    'prob_seizure': predictions['p1'],
    'predicted': predictions['p1'] > threshold
})

# Export results
review_df.to_csv("predictions.csv")
```

**Part 6: Future Prediction (Bonus)**
- Lagging response variable by k windows
- Training models to predict k steps ahead
- Evaluating predictive horizon

---

### 2. `buildKNNModel.R`
**K-Nearest Neighbors in R (Thursday's topic)**

Demonstrates non-linear classification using KNN algorithm.

---

### 3. `ReadMAT_ConvertToSignalvsTime_EngineerWINDOWEDFeatures.ipynb`
**Week 02 reference**

Same feature engineering notebook from Week 02, included for reference.

---

## 🔬 Key Concepts from Tuesday

### 1. **H2O Version Specificity**

**The Problem:**
- Model saved in h2o-3.46.0.9 won't load in h2o-3.47 or h2o-3.45
- `pip install h2o` gets latest (may break old models)

**The Solution:**
```python
# Uninstall any version
!pip uninstall h2o -y

# Install specific version
!pip install https://h2o-release.s3.amazonaws.com/h2o/rel-3.46/0.9/Python/h2o-3.46.0.9-py2.py3-none-any.whl
```

**When to use:**
- Production deployments
- Reproducible research
- Loading old models

### 2. **ROC Curves in Context**

**Perfect Classifier:**
```
TPR = 1, FPR = 0 at all thresholds
ROC hugs top-left corner
AUC = 1.0
```

**Random Classifier:**
```
TPR = FPR at all thresholds
ROC is 45° diagonal line
AUC = 0.5
```

**Our GLM (AUC = 0.81):**
- Between perfect and random
- Good but not excellent
- Room for improvement with non-linear models

**Interpretation Depends on Application:**

**COVID Pandemic:**
- **Goal:** Catch all infected (minimize FN)
- **Threshold:** 0.05-0.10 (very low)
- **Accept:** High false positive rate (85%)
- **Result:** TPR = 100%, Specificity = 15%
- **Rationale:** "Better safe than sorry - quarantine healthy people rather than miss infected ones"

**Balanced Medical Test:**
- **Goal:** Equal performance both classes
- **Threshold:** 0.55-0.60 (medium)
- **Result:** TPR ≈ 75%, Specificity ≈ 75%
- **Metrics:** MaxF1, balanced accuracy

**Insurance Cost Reduction:**
- **Goal:** Minimize expensive false hospital visits
- **Threshold:** 0.75-0.90 (high)
- **Accept:** Missing some true seizures
- **Result:** TPR = 30%, Specificity = 90%+
- **Ethics:** "Not very kind" but realistic business perspective

### 3. **F1 vs F2 Scores**

**F1 Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Harmonic mean of Precision and Recall
- Equal weight to both
- **Use:** When FP and FN costs are similar

**F2 Score:**
```
F2 = 5 × (Precision × Recall) / (4×Precision + Recall)
```
- Weighs Recall 2× more than Precision
- Favors sensitivity
- **Use:** When FN cost > FP cost (COVID, cancer screening)

**MaxF1 Threshold:**
- Balanced approach
- Our GLM: 0.5531-0.5938
- Good default choice

**MaxF2 Threshold:**
- Sensitivity-focused
- Lower than MaxF1
- Better for screening applications

### 4. **Inference vs Training**

**Training:**
- Learning model parameters (β coefficients)
- Optimizing on training data
- Result: Trained model

**Inference (Forward Pass):**
- Applying trained model to new data
- No learning/parameter updates
- Result: Predictions

**Code Pattern:**
```python
# Training (done previously)
model = h2o.train(...)

# Inference (today's focus)
predictions = model.predict(new_data)
```

### 5. **Industrial Best Practices**

**From Professor's Experience:**

**Feature Engineering in Production:**
- Don't recompute every time!
- Save to database (SQL, Delta Lake)
- Giant Eagle example: $1000s per feature run on Spark
- Load when needed for model building

**Model Management:**
- Pin software versions
- Save model artifacts
- Document hyperparameters
- Version control everything

**Procedural Skills:**
> "Knowing what to click where and what needs to be run where - that's incredibly important in industry. It might seem trivial, but it's a valuable skill you'll use daily."

---

## 🚀 Bonus Challenge: Future Seizure Prediction

### The Challenge (50 Bonus Points!)

**Current Limitation:**
- Our models detect seizures as they occur
- Limited clinical utility (can see seizures visually)

**The Goal:**
- Predict seizures BEFORE they occur
- Enable preventive intervention
- Give clinicians lead time

**Implementation:**

**Concept: Lag the Response**
```python
# Current: Window N features → Seizure state at window N
X = features  # All windows
y = labels    # [0,0,0,1,1,1,...]

# Future: Window N features → Seizure state at window N+k
k = 5  # Predict 5 windows ahead

y_future = labels[k:]      # Shift labels
X_current = features[:-k]  # Remove last k windows

# Train model: Current features → Future state!
model.train(X_current, y_future)
```

**Example with k=5:**
```
Time  Window  Features      Current State  Future State (t+5)
--------------------------------------------------------------
0     W0      [d,t,a,b]          Normal        Normal
1     W1      [d,t,a,b]          Normal        Normal
2     W2      [d,t,a,b]          Normal        Seizure  ← Predicting 5 ahead!
3     W3      [d,t,a,b]          Normal        Seizure
4     W4      [d,t,a,b]          Normal        Seizure
5     W5      [d,t,a,b]          Seizure       Seizure
```

**Questions to Answer:**
1. Can we predict 1 window ahead? 5? 10?
2. How does AUC degrade with prediction horizon?
3. Which features best predict future seizures?
4. What's the maximum useful prediction horizon?
5. Do we need different features for prediction vs detection?

**What to Submit (for 50 bonus points):**
- Modified Assignment 2 with future prediction
- Comparison: k=0 (detection) vs k=1,5,10 (prediction)
- AUC vs prediction horizon plot
- Analysis of which features matter for prediction
- Clinical interpretation
- Report discussing feasibility

---

## 💻 How to Run Tuesday's Code

### Setup

**1. Install specific H2O version:**
```python
!pip uninstall h2o -y
!pip install https://h2o-release.s3.amazonaws.com/h2o/rel-3.46/0.9/Python/h2o-3.46.0.9-py2.py3-none-any.whl
```

**2. Initialize H2O:**
```python
import h2o
h2o.init(max_mem_size="2G")

# Access Flow UI
from google.colab import output
output.serve_kernel_port_as_window(54321)
```

**3. Load pre-computed features:**
```python
!cp "/content/drive/MyDrive/Week02/segmentDF_with_frequency_features.csv" /content
import pandas as pd
df = pd.read_csv("segmentDF_with_frequency_features.csv")
df['seizure'] = df['seizure'].astype('category')
```

**4. Load saved model:**
```python
!mkdir /content/models
!cp "/content/drive/MyDrive/Week03/GLM01" /content/models/
model = h2o.load_model("/content/models/GLM01")
```

**5. Run inference:**
```python
predictions = model.predict(test_h2o)
```

---

## 📝 Assignments 0, 1, 2 Status

### Assignment 0: ~7 submissions
- Adapt notebook to EEG_sleep.mat
- **Key challenge:** Array indexing

### Assignment 1: 1 submission  
- Feature engineering with EEG_sleep.mat
- Extract frequency features

### Assignment 2: 0 submissions
- Build H2O models with Assignment 1 features
- **NEW:** Can add future prediction for +50 bonus!

### Critical Reminder

**Must submit ALL THREE components:**
1. ✅ Jupyter Notebook
2. ✅ H2O Flow export (.flow file)
3. ✅ **Written Report** (Word/PDF)

> **"No report, no points!"** - Professor Menon

**Datasets:**
- **Class examples:** session4_train_2018.mat
- **Your assignments:** EEG_sleep.mat (different structure!)

---

## 🎯 Practice Exercises

### Exercise 1: Load and Analyze Model (Beginner)
1. Load GLM01 model from Week 03
2. Access H2O Flow UI
3. View ROC curves for train/validation/CV
4. Identify MaxF1, MaxF2, and balanced accuracy thresholds
5. Compare threshold values across datasets

### Exercise 2: Threshold Exploration (Intermediate)
1. Run inference with threshold = 0.3, 0.5, 0.7
2. Calculate TPR, FPR for each
3. Plot on ROC curve
4. Determine which threshold you'd use for:
   - COVID screening
   - General seizure detection
   - Cost-minimizing deployment

### Exercise 3: Future Prediction (Advanced/Bonus)
1. Implement k-step-ahead prediction (k=1,5,10)
2. Train separate models for each k
3. Compare AUC across prediction horizons
4. Analyze which features degrade vs remain predictive
5. Write clinical interpretation

---

## 🔑 Important Concepts

### 1. Industrial ML Workflow

**Feature Engineering (Expensive):**
- Done once (or periodically)
- Saved to database/storage
- Giant Eagle example: $1000s per run on Databricks

**Model Training (Moderate Cost):**
- Load pre-computed features
- Train models
- Save model artifacts

**Inference (Cheap):**
- Load saved model
- Make predictions on new data
- Deploy in production

**Lesson:** Separate concerns, optimize each stage

### 2. Operating Point Selection Framework

**Framework for choosing threshold:**

**Step 1: Define Cost Function**
- Cost of False Positive (CFP)
- Cost of False Negative (CFN)

**Step 2: Determine Ratio**
- CFN >> CFP → Low threshold (MaxF2)
- CFN ≈ CFP → Medium threshold (MaxF1)
- CFN << CFP → High threshold (Max Specificity)

**Step 3: Use Validation Set**
- Never use training set
- Prefer cross-validation
- Report on test set once

**Examples:**
```
Application         CFN vs CFP    Threshold  Metric
-----------------------------------------------------
COVID Screening     CFN >>> CFP   0.05-0.10  MaxF2
Seizure Detection   CFN ≈ CFP     0.55-0.60  MaxF1
Insurance Model     CFN < CFP     0.75-0.90  Max Spec
```

### 3. Detection vs Prediction

**Detection (Current):**
```python
# Window at time t → State at time t
model.predict(features[t]) → seizure[t]
```
- Simultaneous with event
- Limited clinical value
- What we've built so far

**Prediction (Future Work):**
```python
# Window at time t → State at time t+k
model.predict(features[t]) → seizure[t+k]
```
- Advance warning (k windows)
- High clinical value!
- **Bonus challenge**

**Clinical Impact:**
- Detection: "Patient is seizing" (obvious)
- Prediction: "Patient will seize in 5 seconds" (actionable!)

---

## 🙋 Frequently Asked Questions

**Q: Why do saved models need exact H2O versions?**  
**A:** Internal model format changes between versions. A model from 3.46 won't load in 3.47. Always pin versions in production!

**Q: Can I use pre-computed features from class for my assignments?**  
**A:** No! Assignments use EEG_sleep.mat (different data). You must engineer features yourself. But you CAN reuse YOUR features across assignments.

**Q: Where should I choose my threshold on the ROC curve?**  
**A:** Depends on application:
- **Screening:** Top of curve (high sensitivity)
- **Balanced:** Northwest corner (MaxF1)  
- **Confirmatory:** Left of curve (high specificity)

**Q: What's the difference between MaxF1 and balanced accuracy?**  
**A:** 
- **Balanced Accuracy:** (Sensitivity + Specificity) / 2
- **MaxF1:** Harmonic mean of Precision and Recall
- Usually similar, but MaxF1 better for imbalanced data

**Q: How do I know if my model can predict the future?**  
**A:** Implement lagged prediction! If AUC stays high with k>0, you have predictive power. If AUC drops to 0.5, no future predictability.

**Q: Is the VS Code Colab extension better than browser Colab?**  
**A:** 
- **Pros:** Copilot access, familiar IDE
- **Cons:** No Drive mounting, must stay connected
- **Use:** When you need GPU + Copilot together

---

## 📚 Additional Resources

### H2O Resources:
- [H2O Model Management](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/save-and-load-model.html)
- [H2O Version Archives](https://h2o-release.s3.amazonaws.com/h2o/index.html)
- [H2O Predict API](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2o.model.ModelBase.predict)

### ROC and Metrics:
- [ROC Curves Explained](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [F-Score](https://en.wikipedia.org/wiki/F-score)
- [Choosing Operating Points](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)

### Time Series Prediction:
- [Time Series Forecasting](https://otexts.com/fpp3/)
- [Lagged Features](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)

---

## 🎓 Professor's Notes

**On Professional Skills:**
> "The biggest thing I learned from working in industry vs academia was knowing procedural stuff - what to click where, what needs to run where. It seems trivial but it's SO important. You'll do it so many times it becomes second nature."

**On Cost-Benefit in Healthcare:**

**Real-world ROC decisions aren't just technical - they're ethical:**
- COVID: Society accepts inconvenience (false positives) for safety
- Cancer screening: Prioritize catching cases over false alarms
- Insurance models: Prioritize cost over catching all events (concerning!)

**Our GLM achieved:**
- AUC = 0.81 (good)
- MaxF1 threshold = 0.56-0.58
- At optimal threshold: 100% sensitivity, 54% false positive rate

**This is excellent for screening! But could we do better with:**
1. Non-linear models (Thursday)
2. Better features (future weeks)
3. Prediction instead of detection (bonus challenge)

**For Next Class:**
> "We'll build models programmatically, understand F1/F2 mathematics, and explore K-Nearest Neighbors in R as our first non-linear classifier."

**Professor Prahlad Menon, PhD, PMP**  
*Office Hours: By appointment*  
*Email: prm44@pitt.edu*

---

*"Detection is nice, but prediction is what we really want clinically."*

---

## 📋 Week 04 Tuesday Checklist

- [ ] Run `buildCompetitionModels.ipynb`
- [ ] Install correct H2O version (3.46.0.9)
- [ ] Load pre-calculated features
- [ ] Load GLM01 model from Week 03
- [ ] Review ROC curves in H2O Flow
- [ ] Run inference on test data
- [ ] Understand MaxF1 vs MaxF2 vs balanced accuracy
- [ ] Explore thresholds for different use cases
- [ ] Complete Assignments 0, 1, 2
- [ ] (BONUS) Implement future prediction challenge

---

## 🔬 Key Concepts from Thursday

### 1. **Cross-Validation Deep Dive**

**Textbook Analogy:**
> "If you wanted to ensure every textbook teaches Newton's Laws the same way, would you test 3 or 5 textbooks? More tests = more confidence the knowledge is consistent!"

**K-Fold Cross-Validation:**
- **3-Fold:** 3 models, 3 trials
- **5-Fold:** 5 models, 5 trials (more robust)
- **Leave-One-Out:** N models, N trials (best but expensive)

**Purpose:** Verify model learns generalizable patterns, not data-specific quirks

### 2. **Overfitting vs Underfitting**

**Overfitting (Model Too Complex):**
- Training performance: Excellent
- Test performance: Poor
- **Causes:**
  - Too many parameters (P > N observations)
  - Too many features (M > N observations)
  - Too complex model for simple pattern
- **Visual:** Squiggly boundary with disconnected "islands"

**Good Fit:**
- Training ≈ Test performance
- **Sweet spot:** Right complexity for data
- **Visual:** Smooth boundary capturing true pattern

**Underfitting (Model Too Simple):**
- Training performance: Poor
- Test performance: Poor
- **Cause:** Linear model for non-linear pattern
- **Visual:** Straight line through curved data

**Rule:** N (observations) > P (parameters) > M (features)

### 3. **Variance Pizza Analogy** 🍕

**Goal:** Explain all variance in Y (response)

**Good Features (Independent):**
- X₁ explains slices A, B, C
- X₂ explains slices D, E, F
- X₃ explains slices G, H
- **Result:** Full pizza covered, no overlap ✓

**Bad Features (Variance Inflation):**
- X₁ explains slices A, B, C, D
- X₂ explains slices C, D, E, F
- **Problem:** Slices C,D counted twice (inflated)
- **Problem:** Slices G,H unexplained
- **Solution:** Remove redundant feature

### 4. **Non-Linearity: Two Approaches**

**Via Feature Engineering:**
- Add X₁², X₂², sin(X₁), etc.
- Linear model can capture curved boundaries
- **We did this with frequency features!**

**Via Model Complexity:**
- Neural networks with multiple neurons
- Each neuron learns piece of boundary
- **TensorFlow Playground:** 4 neurons = spiral

**Lesson:** Feature engineering + simple model often beats simple features + complex model!

### 5. **Forward Pass Examples**

**What is Inference?**
- Using pre-trained model
- No learning/training
- Just predictions

**Professor's Examples:**
- **ChatGPT:** Forward pass on every query
- **Text-to-Speech:** Qwen3-TTS model
- **Virtual CFI:** Chained models (speech→text→LLM→TTS→avatar)

### 6. **LazyPredict Results**

**Quick Model Competition:**
```
KNN:                    AUC = 0.73 ⭐ (best)
Random Forest:          AUC = 0.71
SVC:                    AUC = 0.70
Logistic Regression:    AUC = 0.66 (baseline)
```

**Conclusion:** Seizure detection is a **non-linear** problem!

### 7. **ROC from Any Variable!**

**Mind-blowing insight:**
- Don't need model probabilities
- Can threshold ANY continuous variable
- Create ROC by varying threshold
- **Use:** Evaluate single features directly

---

## 🎬 Next Week Preview

**Tuesday, February 10:**
- Elastic Net regularization (L1, L2)
- Multi-class classification
- KNN in R (`buildKNNModel.R`)
- Advanced evaluation metrics

**Prepare by:**
- Running `buildCompetitionModels.ipynb`
- Trying TensorFlow Playground
- Understanding overfitting vs underfitting
- Completing all assignments

---

**Have a great week! Work on assignments and explore the 50-point bonus challenge!** 🚀
