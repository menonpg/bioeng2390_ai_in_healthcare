# BIOENG-2390 Spring 2026 - Lecture 7
## February 3, 2026

**Instructor:** Professor Prahlad G. Menon, PhD, PMP  
**Email:** menon.prahlad@gmail.com  
**Recording:** [View on Fathom](https://fathom.video/share/Esz_6RSP-DviJ8EPzufrv4dgvnbYS_1z)  
**Duration:** 67 minutes

---

## 📋 Lecture Overview

Today's class focused on:
1. Loading pre-calculated features and saved models
2. H2O version compatibility and management
3. Deep ROC curve interpretation for different use cases
4. Threshold selection strategies (MaxF1, MaxF2, balanced accuracy)
5. Running inference/forward pass on trained models
6. Making predictions with custom thresholds
7. Context-dependent operating point selection
8. VS Code Colab extension for GPU access
9. Future seizure prediction bonus challenge

---

## 🎯 Assignment Status & Expectations

### Current Submissions

**Assignment 0:** ~7 students
- Adapt notebook to EEG_sleep.mat
- Main challenge: Array indexing for different .mat structure

**Assignment 1:** 1 submission
- Feature engineering with EEG_sleep.mat
- Extract Delta, Theta, Alpha, Beta features
- First 20 seconds only

**Assignment 2:** 0 submissions
- Use Assignment 1 features with H2O
- Build classification models
- Evaluate performance

### Clarification on Datasets

**In Class:** Used `session4_train_2018.mat` for most Python/R demonstrations

**For Assignments:** Use `EEG_sleep.mat` (different structure!)

**Why different?**
> "Since assignment 0, 1, and 2 are pretty much reruns of what we have done in class, I want to make sure you use a different dataset than you used in class."

**Key Difference (from Bhavya):**
> "You will not get the output if you just replace what is done in class and load this file. You need to look at the contents because they're not the same. Make sure what you're plotting is the spectral data and not something else which is just a single variable."

### Assignment Submission Requirements

**MUST include all three:**
1. ✅ Jupyter Notebook (code + outputs)
2. ✅ H2O Flow file (.flow export)
3. ✅ **Written Report** (Word/PDF)

**Professor's Emphasis:**
> "No report, no points! Your assignments have liberty in what you do, as long as you get the task done. But you MUST explain what you learned and how you executed it."

**Report should include:**
- What you understood at each step
- Explanation of your design choices
- Results and interpretation
- Challenges faced and solutions
- Screenshots/figures from H2O Flow

---

## 🔬 H2O Version Management

### Version Specificity

**Important:** Models built in one H2O version only work with that exact version!

**Current Version:** h2o-3.46.0.9

**Generic Installation:**
```python
!pip install h2o  # Gets latest version
```

**Problem:** Future updates will break compatibility with old models

**Solution: Install Specific Version**
```python
# Uninstall current version
!pip uninstall h2o -y

# Install specific version
!pip install https://h2o-release.s3.amazonaws.com/h2o/rel-3.46/0.9/Python/h2o-3.46.0.9-py2.py3-none-any.whl
```

**Finding Version URLs:**
1. Google search: "H2O version 3.46.0.9"
2. Visit H2O documentation archives
3. Find AWS-hosted wheel file URL
4. Use in pip install command

**Why this matters:**
- Can reload old models months/years later
- Reproducible research
- Production deployments need version pinning

---

## 💻 Loading Pre-Calculated Features

### Workflow: Skip Feature Engineering

**Problem:** Feature engineering is computationally expensive

**Solution:** Load previously saved features

**Code:**
```python
# Instead of running all feature engineering cells...
# Just copy the pre-calculated CSV

!cp "/content/drive/MyDrive/.../segmentDF_with_frequency_features.csv" /content

# Load into pandas
import pandas as pd
segments_df = pd.read_csv("segmentDF_with_frequency_features.csv")

# Ensure categorical type
segments_df['seizure'] = segments_df['seizure'].astype('category')

# Load into H2O
import h2o
h2o.init(max_mem_size="2G")
segments_h2o = h2o.H2OFrame(segments_df)
segments_h2o['seizure'] = segments_h2o['seizure'].asfactor()
```

**Real-World Example (from lecture):**

**Professor's Giant Eagle Fraud Detection:**
- Retail transaction data (billions of records)
- Features: Recency, Frequency, Monetary value
- Computed on Databricks Spark cluster
- Cost: Thousands of dollars per feature engineering run!
- **Solution:** Save to SQL database/Delta Lake, reload when needed

**Lesson:**
> "In industry, you don't recompute features every time. You save them to a database and reload for model building."

---

## 🔄 Loading Saved H2O Models

### Step-by-Step Process

**1. Copy model from Google Drive to Colab:**
```python
# Create models directory
!mkdir /content/models

# Copy saved model
!cp "/content/drive/MyDrive/Week03/GLM01" /content/models/
```

**2. Load model into H2O:**
```python
# Import model
model = h2o.load_model("/content/models/GLM01")
```

**3. Verify in H2O Flow:**
```
- Click "Admin" → "Get Models"
- Should see GLM01 listed
- Click to view full details
```

**What's in a saved model:**
- Model coefficients (β values)
- **Metadata:** Model type, hyperparameters
- Training metrics and confusion matrices
- ROC curves for train/validation/test
- Variable importance
- Cross-validation results
- Optimal thresholds (MaxF1, MaxF2, balanced accuracy)

**Not just rules - complete model with all evaluation!**

---

## 📈 Deep Dive: ROC Curve Interpretation

### Understanding Every Point on ROC

**Axes:**
- **X:** False Positive Rate (FPR) = FP / (FP + TN)
- **Y:** True Positive Rate (TPR) = TP / (TP + FN) = Recall = Sensitivity

**Each point represents:**
- Different probability threshold (0.0 to 1.0)
- Resulting confusion matrix at that threshold
- Trade-off between TPR and FPR

**Perfect Classifier:** 
```
   ^ TPR
 1 |•
   |  ╲
   |    ╲  ← Perfect ROC
0.5|      ╲
   |        ╲
 0 |__________•
   0        0.5        1 → FPR
```
Hugs top-left corner (TPR=1, FPR=0)

**Random Classifier:**
```
   ^ TPR
 1 |        •
   |      ╱  
   |    ╱    ← Random guess (45° line)
0.5|  ╱      ← AUC = 0.5
   |╱        
 0 |•
   0        0.5        1 → FPR
```
Diagonal line (no better than coin flip)

**Our GLM:**
- AUC = 0.81-0.82
- Between perfect and random
- Good, not excellent

---

### Context-Dependent Threshold Selection

#### Use Case 1: COVID Screening

**Goal:** Minimize missed infections (false negatives)

**Strategy:** Maximize TPR (Sensitivity)

**Where on ROC?** Top part of curve

**Example from lecture:**
- Threshold: 0.05-0.10 (very low!)
- TPR: 100% (catch all infected people!)
- Specificity: 15% (low - many false alarms)
- FPR: 85% (very high false positives)

**Trade-off:**
> "I don't care if I have a high false positive rate, as long as very few people slip through the cracks. In this case, no people slip through."

**Result:** Healthy people quarantine (inconvenient but safe)

#### Use Case 2: Seizure Detection (Current Model)

**Goal:** Balanced performance

**Strategy:** Use balanced accuracy or MaxF1

**Where on ROC?** Northwest corner

**Thresholds:**
- Balanced accuracy: 0.60
- MaxF1: ~0.59
- Training: 0.61, Validation: 0.56, CV: 0.58

**Trade-off:**
- ~75% sensitivity
- ~75% specificity
- Balanced false positives and false negatives

#### Use Case 3: Insurance Company Perspective

**Goal:** Minimize false hospital visits (cost reduction)

**Strategy:** Maximize specificity

**Where on ROC?** Lower-left part of curve

**Example from lecture:**
- Threshold: 0.75-0.80 (high!)
- TPR: ~30% (catch only 3 in 10 seizures)
- Specificity: 90%+ (few false alarms)

**Trade-off:**
> "If they have a few seizures that go unnoticed, that's fine. I'd rather reduce the cost of the healthcare system."

**Ethics:** Professor notes this is "not very kind" but realistic

---

### F1 and F2 Scores Explained

**F1 Score:**
- Harmonic mean of Precision and Recall
- Balances both metrics equally
- **Use when:** Both false positives and false negatives matter equally

**F2 Score:**
- Weighs Recall higher than Precision
- **Use when:** False negatives more costly than false positives
- Better for COVID-type screening

**MaxF1 Threshold:**
- Optimizes F1 score
- Balanced approach
- Good default choice

**MaxF2 Threshold:**
> "More similar to the operating point you might choose if you were trying to make a model to predict COVID. You don't want too many false negatives."

---

## 🎯 Running Inference (Forward Pass)

### What is Inference?

**Inference = Forward Pass:**
- Taking a trained model
- Running it on new data
- Getting predictions
- **No training/learning occurs**

**Code from Thursday:**
```python
# Make predictions
predictions_h2o = model.predict(test_h2o)

# Extract results to pandas
predictions_df = predictions_h2o.as_data_frame()

# Columns:
# - predict: Binary decision (0 or 1)
# - p0: Probability of class 0 (Normal)
# - p1: Probability of class 1 (Seizure)
```

### Custom Threshold Inference

**Using specific threshold (e.g., MaxF1 = 0.5531):**
```python
import pandas as pd

# Get predictions and actuals
review_df = pd.DataFrame({
    'actual_seizure': test_h2o['seizure'].as_data_frame(),
    'predicted_prob': predictions_h2o['p1'].as_data_frame(),
    'predicted_seizure': (predictions_h2o['p1'] > 0.5531).as_data_frame()
})

# Add comparison
review_df['correct'] = review_df['actual_seizure'] == review_df['predicted_seizure']

# View results
print(review_df.head())
```

**Output format:**
```
   actual  prob   predicted  correct
0    0     0.12      0         ✓
1    1     0.87      1         ✓
2    0     0.65      1         ✗ (FP)
3    1     0.42      0         ✗ (FN)
```

**Export predictions:**
```python
review_df.to_csv("/content/drive/MyDrive/Week04/predictions.csv", index=False)
```

---

## 🚀 Advanced Topic: Future Seizure Prediction

### From Detection to Prediction

**Current Models:** Detect seizures as they occur
- Use window features to classify current state
- **Problem:** Not very useful clinically
- Can detect with eyes/clinical observation

**Better Goal:** Predict seizures BEFORE they occur
- Give clinicians time to intervene
- Administer medication prophylactically
- Alert caregivers

### How to Implement (Bonus Challenge - 50 points!)

**Concept: Lag the Response**

```python
# Current approach: window N predicts seizure state at window N
y_current = seizure_labels  # [0,0,0,1,1,1...]

# Future prediction: window N predicts seizure state at window N+k
k = 5  # Predict 5 windows ahead
y_future = seizure_labels[k:]  # Shift labels backward
X_current = features[:-k]       # Remove last k windows

# Now: Current window features → Future seizure state!
```

**Example:**
```
Window  Features         Current Label  Future Label (k=5)
--------------------------------------------------------------
0       [d,t,a,b]            0              0
1       [d,t,a,b]            0              0  
2       [d,t,a,b]            0              1  ← Predicting 5 windows ahead!
3       [d,t,a,b]            0              1
...
```

**Questions to Explore:**
1. Can we predict 1 window ahead? 5 windows? 10 windows?
2. At what prediction horizon does performance degrade?
3. Which features are best for future prediction?
4. Does increasing prediction horizon require different features?

**Professor's Challenge:**
> "For those doing assignment 1 and 2, you can try predicting the seizure occurring about 5 or 10 windows in the future. If you do establish future predictability, you'll get 50 bonus points!"

---

## 💻 VS Code Colab Extension

### Running Colab Notebooks in VS Code

**Benefits:**
- Use GitHub Copilot within Colab notebooks
- GPU/TPU access from VS Code interface
- Consistent IDE experience

**Setup:**
1. Open VS Code Extensions (Cmd/Ctrl + Shift + X)
2. Search "Colab"
3. Install Google Colab extension
4. Reload VS Code

**Usage:**
1. Open .ipynb file in VS Code
2. Click "Select Kernel" (top-right)
3. Choose "Colab" → "Auto-connect" (CPU) or select GPU/TPU
4. Select language: Python 3
5. Notebook now runs on Colab VM!

**Verification:**
```python
!pwd  # Shows /content (Colab) vs local path
```

**Limitations:**
- Google Drive mounting not supported (`drive.mount()` fails)
- Need to keep computer on (closing lid disconnects)
- GPU usage drains credits if left connected

**Disconnecting:**
```
1. Cmd/Ctrl + Shift + P
2. Type "Developer: Restart Extension Host"
3. Extension restarts → disconnects from Colab
```

**Professor's Experience:**
> "I accidentally left a GPU notebook connected overnight and ran out of all my GPU credits! Make sure to disconnect when done."

---

## 🔑 Key Concepts

### 1. Array Indexing in Python for .mat Files

**Question from Class:** What do the bracket zeros mean?

**Answer:** Each `[0]` descends one level in nested structure

**Example:**
```python
data['EEG']         # Access EEG structure
data['EEG'][0]      # First level of nesting
data['EEG'][0][0]   # Second level
data['EEG'][0][0][0]  # Third level
data['EEG'][0][0][0][0]  # Fourth level → actual data array!
```

**Visualization:**
```
data = {
  'header': {...},
  'EEG': [[[[array([...]),    ← Target data
            62,               ← Seizure start
            124,              ← Seizure end
            'seconds',        ← Units
            256]]]]           ← Sampling rate
}
```

**To access array:**
```python
raw = data['EEG'][0][0][0][0]  # Navigate 4 levels deep
```

**Tip:** 
- Open .mat file in MATLAB to see structure
- Use variable workspace to understand nesting
- Different .mat files have different nesting levels!

---

### 2. ROC Curve Operating Points

**Standard Operating Points:**

**1. Balanced Accuracy (Min-Max Per-Class Accuracy)**
- Equal weight to TPR and TNR
- Threshold typically ~0.50-0.60
- **Use:** General-purpose, no strong preference

**2. MaxF1**
- Optimizes F1 score (harmonic mean of Precision and Recall)
- Balanced between precision and recall
- Threshold typically ~0.55-0.60
- **Use:** When both false positives and false negatives matter equally

**3. MaxF2**
- Optimizes F2 score (weighs recall 2× more than precision)
- Favors sensitivity over specificity
- Threshold typically ~0.30-0.45 (lower)
- **Use:** When false negatives more costly (COVID, cancer screening)

**4. Custom (e.g., Insurance)**
- Maximize specificity (minimize false positives)
- High threshold (0.70-0.90)
- **Use:** When false positives are very costly
- **Ethics:** Questionable for healthcare!

**From Our GLM:**
```
Operating Point          Threshold    TPR    FPR
----------------------------------------------------
Balanced Accuracy        0.60         0.75   0.25
MaxF1 (Training)         0.5938       0.76   0.24
MaxF1 (Validation)       0.5531       varies
MaxF1 (Cross-Val)        0.58         varies
```

---

### 3. Training vs Validation vs Test for Thresholds

**NEVER use training set for threshold selection!**
- Training set optimizes model coefficients (β)
- Would be overfitting to also optimize threshold on same data

**Use Validation Set:**
- Independent from training
- Fair estimate of threshold performance
- Our validation MaxF1: 0.56

**Use Cross-Validation:**
- Most robust
- Average across 5 folds
- Our CV MaxF1: 0.58

**Final Evaluation on Test Set:**
- Apply chosen threshold (0.57-0.58 from validation/CV)
- Get unbiased performance estimate
- **Only use test set once!**

---

## 🎓 Practical Skills from Lecture

### 1. Industrial Skills

**Professor on tool proficiency:**
> "When you get a job, knowing how to use tools and what to click where and what needs to be run where - that's SO important. The biggest thing I learned from working in industry vs academia was knowing procedural stuff."

**Skills practiced today:**
- Loading pre-computed data efficiently
- Managing model versions
- Navigating H2O Flow UI
- Running inference programmatically
- Exporting predictions

**Transferable to any industrial ML setting!**

### 2. Understanding Array Addressing

**Bhavya's Tip:**
> "If you work a lot with MATLAB, open the file in MATLAB first. You'll see the structure clearly, then know how many bracket-zeros you need."

**For EEG_sleep.mat:**
- Structure may be different than session4_train_2018.mat
- Might need different number of `[0]` levels
- Sampling rate might be different (500 Hz vs 256 Hz)
- **This is Assignment 0!**

---

## 🎬 For Next Class (Thursday, Feb 5)

### Topics to Cover:

1. **Building Models Programmatically (No AutoML)**
   - H2O Python API
   - Scikit-learn comparison
   - When to use AutoML vs manual

2. **F1 and F2 Score Mathematics**
   - Derivation and interpretation
   - When to use which metric

3. **Non-Linear Models in R**
   - Using `buildKNNModel.R`
   - K-Nearest Neighbors algorithm
   - Comparison with linear models

4. **Model Evaluation Metrics**
   - Beyond AUC: Precision-Recall curves
   - Calibration plots
   - Lift and gain charts

### Homework:

- [ ] Run today's notebook (loading models + inference)
- [ ] Explore ROC curve with different thresholds
- [ ] Complete Assignment 0 (if not done)
- [ ] Complete Assignment 1 (feature engineering)
- [ ] Complete Assignment 2 (H2O modeling)
- [ ] **BONUS:** Try future prediction (50 points!)
- [ ] Write reports for all assignments

---

## 🔑 Critical Insights from Lecture

### 1. Context Determines "Good"

**COVID Screening:**
- High sensitivity critical (catch all infections)
- Accept high false positive rate
- Low threshold (0.05-0.10)

**Seizure Detection:**
- Balanced approach
- Equal importance to sensitivity and specificity
- Medium threshold (0.55-0.60)

**Cost Minimization:**
- High specificity critical (reduce false hospital visits)
- Accept lower sensitivity
- High threshold (0.70-0.90)

**Lesson:**
> "There's no universal 'best' threshold. It depends on the cost of false positives vs false negatives in your specific application."

### 2. Detection vs Prediction

**Detection (what we've done):**
- Model identifies seizures as they occur
- Limited clinical value
- Can see seizures visually anyway

**Prediction (bonus challenge):**
- Model predicts seizures before they occur
- HIGH clinical value!
- Enables preventive intervention
- **Implementation:** Lag response by k windows

**Question to explore:**
> "Is current frequency information a future predictor of a seizure event?"

---

## 🙋 Questions from Class

**Q: What does each [0] in data['EEG'][0][0][0][0] represent?**  
**A:** Each [0] descends one level in the nested structure. MATLAB .mat files store data in nested arrays. Different files have different nesting levels - you must inspect each file's structure.

**Q: Why specify exact H2O version?**  
**A:** Models built in version 3.46.0.9 won't load in version 3.47 or 3.45. For reproducibility and production, always pin versions.

**Q: Where on ROC curve should I choose threshold?**  
**A:** Depends on application:
- COVID → High on curve (max sensitivity)
- Balanced → Northwest corner (max F1)
- Cost reduction → Lower on curve (max specificity)

**Q: Can I change threshold after seeing test results?**  
**A:** No! That's cheating. Choose threshold using validation/CV only. Test set is for final evaluation once.

**Q: How do I know if my model is overfitting?**  
**A:** Compare training AUC vs validation/CV AUC:
- Training: 0.90, Validation: 0.60 → Overfitting!
- Training: 0.82, Validation: 0.81 → Good generalization

---

## 💡 Practical Tips

### Efficient Workflow

**Don't recompute everything every time:**
1. Run feature engineering once
2. Save features to CSV
3. For modeling: Load CSV directly
4. Save trained models
5. For inference: Load model + make predictions

**This saves:**
- Time (minutes to hours)
- Compute resources
- Money (in production settings)

### Model Management

**Organize your models:**
```
/models/
  GLM01
  GLM_with_interactions
  XGBoost_best
  StackedEnsemble_final
  /archived/
    GLM_old_version
```

**Name models meaningfully:**
- Include algorithm type
- Include key hyperparameters
- Include date/version
- Example: `GLM_interactions_F1_020326`

---

## 📚 Bonus Challenge Details

### Future Seizure Prediction (50 Bonus Points)

**Task:**
Modify Assignment 2 to predict seizures k windows in advance

**Implementation hints:**
1. After feature engineering, shift labels:
   ```python
   k = 5  # Predict 5 windows ahead
   y_future = y[k:]  # Remove first k labels
   X_current = X[:-k]  # Remove last k feature windows
   ```

2. Train models as usual on (X_current, y_future)

3. Interpretation:
   - Window at time t predicts seizure at time t+k

4. Compare performance:
   - k=0 (current): Baseline
   - k=1, 5, 10: Future prediction

5. Report findings:
   - At what k does model fail?
   - Which features best predict future?
   - Is future prediction clinically feasible?

**Expected in report:**
- Methodology explanation
- Results for multiple k values
- Performance comparison (AUC vs k)
- Clinical interpretation
- Limitations and future work

---

## 📋 Week 04 Lecture 7 Checklist

- [ ] Understand H2O version management
- [ ] Can load pre-calculated features
- [ ] Can load saved H2O models
- [ ] Understand ROC curve operating points
- [ ] Know how to select threshold for different use cases
- [ ] Can run inference on trained models
- [ ] Understand difference between detection and prediction
- [ ] Complete Assignments 0, 1, 2
- [ ] (Optional) Try future prediction bonus challenge

---

## 🎓 Professor's Final Thoughts

**On Procedural Knowledge:**
> "Sometimes this procedural stuff seems trivial - just clicking around. But knowing what to click where is incredibly important in industry. You'll do this so many times it becomes second nature, and that's a valuable skill."

**On ROC Interpretation:**
> "We need to really understand this curve. It's so important! You can spend hours exploring different thresholds and understanding trade-offs."

**On Future Work:**
> "Detection is nice, but prediction is what we really want. Can we use current window features to predict seizures 5-10 windows in advance? That's the real clinical value."

**Professor Prahlad Menon, PhD, PMP**  
*Office Hours: By appointment*  
*Email: prm44@pitt.edu*

---

*"The best operating point depends on whether you're screening for COVID, detecting seizures, or minimizing healthcare costs. Context is everything."*

---

## 🔑 Key Takeaways

1. ✅ H2O models are version-specific - pin versions in production
2. ✅ Load pre-computed features to save time/cost
3. ✅ ROC curves show TPR vs FPR across all thresholds
4. ✅ Operating point selection depends on application context
5. ✅ MaxF1 balances precision and recall
6. ✅ MaxF2 prioritizes recall over precision
7. ✅ Use validation/CV for threshold selection, NOT training
8. ✅ Inference = forward pass = predictions on new data
9. ✅ Detection ≠ Prediction (prediction is more valuable clinically)
10. ✅ Industrial skills include tool proficiency, not just math

**Next Class:** Building models programmatically, F1/F2 derivations, KNN in R

**See you Thursday! Work on assignments and explore the bonus challenge!** 🚀
