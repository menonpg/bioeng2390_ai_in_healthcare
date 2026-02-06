# BIOENG-2390 Spring 2026 - Lecture 8
## February 5, 2026

**Instructor:** Professor Prahlad G. Menon, PhD, PMP  
**Email:** menon.prahlad@gmail.com  
**Recording:** [View on Fathom](https://fathom.video/share/wzY-sAzurZDhzdxsUjG-j6uojkkC3pJw)  
**Duration:** 88 minutes

---

## 📋 Lecture Overview

Today's class was highly conceptual with extensive whiteboarding covering:
1. Cross-validation deep dive (3-fold, 5-fold, leave-one-out)
2. Overfitting vs underfitting explained
3. Curse of dimensionality and parameter count
4. Variance inflation and feature independence
5. Non-linearity from features vs model complexity
6. TensorFlow Playground demonstrations
7. Forward pass/inference with LLM examples (avatars, text-to-speech)
8. ROC curves from any continuous variable (not just probabilities!)
9. LazyPredict for model competitions
10. Introduction to K-Nearest Neighbors (KNN)

---

## 🎯 Key Concepts

### 1. **Cross-Validation Explained**

**The Goal of Cross-Validation:**
> "To determine if the model is sensitive to the training data choice - does it learn generalizable science or just lucky patterns?"

**K-Fold Cross-Validation:**

**3-Fold Example:**
```
Data split into 3 parts: [1] [2] [3]

Model A: Train on {2,3} → Predict {1}
Model B: Train on {1,3} → Predict {2}
Model C: Train on {1,2} → Predict {3}

Combine predictions from all models → ROC curve
```

**5-Fold Example:**
```
Data split into 5 parts: [1] [2] [3] [4] [5]

Model 1: Train on {2,3,4,5} → Predict {1}
Model 2: Train on {1,3,4,5} → Predict {2}
Model 3: Train on {1,2,4,5} → Predict {3}
Model 4: Train on {1,2,3,5} → Predict {4}
Model 5: Train on {1,2,3,4} → Predict {5}

Combine predictions → More robust ROC curve
```

**Question from Class:** Is 5-fold better than 3-fold?

**Answer:** YES! More folds = more trials = more confidence

**Analogy:**
> "If you wanted to make sure every textbook teaches Newton's Laws the same way, would you test 3 textbooks or 5 textbooks? More tests = more confidence that the knowledge is consistent!"

**Leave-One-Out Cross-Validation (LOO-CV):**
```
If you have N observations:
- Create N models
- Each model trains on N-1 observations
- Each model predicts 1 held-out observation
- Combine all N predictions

This is the BEST case! But computationally expensive.
```

**When to use:**
- Small datasets (hundreds of observations)
- When you need maximum confidence
- Research/critical applications

**Not practical for:**
- Large datasets (thousands/millions of observations)
- Limited compute resources
- Time-constrained projects

---

### 2. **Overfitting vs Underfitting**

**The Three Scenarios:**

#### Scenario 1: Underfitting
```
Model: y = β₀ + β₁·x₁ + β₂·x₂  (Linear)
Data: Circular blobs (needs curved boundary)

Result: Straight line through curved data
Performance: Poor on train AND test
Problem: Model too simple for data complexity
```

**Visual:** Linear boundary trying to separate circular clusters

**Solution:** Add non-linear features (x₁², x₂²) or use non-linear model

#### Scenario 2: Good Fit
```
Model: 4-6 neurons with appropriate features
Data: Spiral pattern

Result: Smooth spiral boundary
Performance: Good on train AND test
Sweet spot: Right complexity for data
```

**Visual:** Smooth curved boundary matching spiral pattern

#### Scenario 3: Overfitting
```
Model: 10+ neurons with complex features
Data: Simple pattern

Result: Squiggly, disconnected boundaries
Performance: Excellent on train, poor on test
Problem: Model too complex for data
```

**Visual:** Decision boundary with "islands" disconnected from main regions

**From MATLAB Documentation Example:**
- **Underfit:** Straight line for curved data (misses structure)
- **Good fit:** Curved line capturing general pattern
- **Overfit:** Squiggly line perfectly fitting training noise

---

### 3. **Causes of Overfitting**

**Three Main Causes:**

**1. Too Many Features, Too Few Observations**
```
M features > N observations → Overfitting risk!
```

**Example:**
- 50 features (gene expression)
- 10 patient observations
- Can find combinations that "work" on these 10
- Won't generalize to new patients

**Rule of Thumb:** N > M (more observations than features)

**2. Too Many Parameters in Model**
```
Parameters > Observations → Overfitting!
```

**Example from class:**
- Dog vs cat classifier
- 10 billion parameters
- **Need:** > 10 billion images!
- Otherwise: Model memorizes training data

**3. Too Complicated Model for Simple Problem**
```
100-neuron network for 2-class circular data → Overkill!
```

**Lesson:**
> "If you torture the data enough, it will confess to anything." - Professor's professor

---

### 4. **Variance Inflation (Pizza Analogy)**

**The Pizza Metaphor:**

**Goal:** Explain all variation in Y (the whole pizza)

**Scenario 1: Good Features**
```
🍕 Pizza (Y variance)
────────────────
X₁ explains: Slices A, B, C
X₂ explains: Slices D, E, F  
X₃ explains: Slices G, H

Result: All 8 slices covered, minimal overlap
This is GOOD! Features are independent.
```

**Scenario 2: Variance Inflation**
```
🍕 Pizza (Y variance)
────────────────
X₁ explains: Slices A, B, C, D
X₂ explains: Slices C, D, E, F  ← Overlap with X₁!

Result: C and D counted twice (inflated)
Some slices (G, H) not explained
This is BAD! Features are redundant.
```

**Variance Inflation Factor (VIF):**
- Measures how much features overlap in explained variance
- High VIF = redundant features
- **Solution:** Remove one of the correlated features

**Example:**
- Beta and Alpha both high-frequency bands
- Some overlap in what they explain
- But each also explains unique variance
- Both are worth keeping

---

### 5. **Non-Linearity: Features vs Model Complexity**

**Two Ways to Achieve Non-Linear Decision Boundaries:**

#### Method 1: Feature Engineering
```
Original features: X₁, X₂
Add non-linear features: X₁², X₂², X₁·X₂, sin(X₁), etc.
Model: Still linear! y = β₀ + β₁X₁² + β₂X₂²

But decision boundary is curved (circle, parabola)
```

**Example from TensorFlow Playground:**
- Linear model + (X₁², X₂²) features
- Captures circular decision boundary
- Model is linear, boundary is not!

**This is what we did with frequency features!**
- Added Delta, Theta, Alpha, Beta (non-linear transforms of signal)
- Even with linear GLM, achieved curved boundaries in feature space

#### Method 2: Model Complexity
```
Original features: X₁, X₂ (no engineering)
Model: Neural network with multiple neurons

Each neuron: Can learn piece of complex boundary
Together: Combine to form overall non-linear boundary
```

**From TensorFlow Playground:**
- 1 neuron → Straight line (linear)
- 4 neurons → Curved line (good fit)
- 10 neurons → Squiggly line (overfit)

**Lesson:**
> "Feature engineering with simple models often beats complex models with simple features!"

---

### 6. **TensorFlow Playground Demonstrations**

**Website:** [playground.tensorflow.org](https://playground.tensorflow.org)

**Experiment 1: Linear Separation (2 Blobs)**
- **Data:** Two separate circular blobs
- **1 Neuron (Linear):** Straight line → Good enough!
- **Activation:** Sigmoid
- **Result:** Converged in 204 iterations
- **Learning:** Sometimes simple is sufficient

**Experiment 2: Circular Pattern**
- **Data:** Orange ring around blue blob
- **1 Neuron:** Straight line → Severe underfit
- **Add Features:** X₁², X₂² → Captures circle with linear model!
- **Learning:** Feature engineering can replace model complexity

**Experiment 3: Spiral Pattern**
- **Data:** Interleaved orange/blue spiral
- **4 Neurons:** Smooth spiral boundary → Good fit
- **6 Neurons:** Some disconnected regions → Slight overfit
- **10 Neurons:** Many islands → Severe overfit
- **Learning:** More isn't always better!

**Key Parameters:**
- **Learning Rate:** How fast to adjust weights
- **Activation:** Sigmoid, ReLU (Rectified Linear Unit), etc.
- **Regularization:** L1, L2, elastic net (penalize large weights)
- **Neurons:** Trade-off between expressiveness and overfitting

---

### 7. **Forward Pass / Inference Examples**

**What is Inference?**
> "Using a pre-trained model to make predictions on new data. No training/learning occurs."

**Also called:** Forward pass, prediction, model serving

#### Example 1: Large Language Models

**When you use ChatGPT/Gemini:**
```
1. Model already trained (billions of parameters learned)
2. You provide input (prompt/question)
3. Model runs forward pass
4. Output generated (text completion)
```

**No training happens** when you chat!

#### Example 2: Text-to-Speech (Qwen3-TTS)

**Professor demonstrated:**
- Input: Text string
- Model: Pre-trained TTS model
- Forward pass: Text → Audio waveform
- Output: Synthesized speech

**Multiple forward passes chained:**
```
Voice Input → Speech-to-Text Model (Forward Pass 1)
     ↓
Text → LLM Model (Forward Pass 2) 
     ↓
Text Response → Text-to-Speech Model (Forward Pass 3)
     ↓
Audio Output
```

#### Example 3: Virtual CFI (Aviation Assistant)

**Professor's Project:**
1. Pilot asks question (voice)
2. Speech → Text (Whisper model)
3. Text + Weather API → Answer (LLM)
4. Answer → Voice (TTS model)
5. Avatar lip-sync (HeyGen model)

**All forward passes** - no training!

**Example interaction:**
```
Pilot: "What's the crosswind limitation for Icon A5 at Brooksville runway 27?"
System: [Fetches METAR, calculates crosswind]
Avatar: "November 161 Bravo Alpha, winds 290 at 15 gusting 29. 
         Crosswind about 10 knots with gusts, well within your 
         12 knot demonstrated limit."
```

**Technologies chained:**
- ASR (Automatic Speech Recognition)
- RAG (Retrieval Augmented Generation) with web access
- TTS (Text-to-Speech)
- Avatar generation

**All use pre-trained models!**

---

### 8. **ROC Curves from Any Continuous Variable**

**Mind-Blowing Insight:**
> "ROC stands for Receiver Operating Characteristic. It's nothing but TPR vs FPR. You can dichotomize ANY continuous variable based on a threshold. It is not just probability!"

**Standard ROC (from probabilities):**
```python
# Threshold probability at different values
for threshold in [0.0, 0.1, 0.2, ..., 0.9, 1.0]:
    predictions = (probabilities > threshold)
    TPR, FPR = calculate_rates(predictions, actual)
    plot_point(FPR, TPR)
```

**ROC from raw feature (e.g., X₂):**
```python
# Threshold X₂ directly!
for threshold in [min(X₂), ..., max(X₂)]:
    predictions = (X₂ > threshold)
    TPR, FPR = calculate_rates(predictions, actual)
    plot_point(FPR, TPR)
```

**What this means:**
- Can evaluate any single feature's predictive power
- Don't need a model!
- Just threshold the feature at different values
- See how well it separates classes

**Use case:**
- Feature selection
- Understanding which features matter
- Comparing feature importance

---

### 9. **LazyPredict for Model Competitions**

**What is LazyPredict?**
- Automatically trains many model types
- No hyperparameter tuning
- Quick comparison
- Identifies if problem is linear vs non-linear

**Code:**
```python
from lazypredict.Supervised import LazyClassifier

# Prepare data
X = df[['delta', 'theta', 'alpha', 'beta']]
y = df['seizure'].astype('category')

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Run competition
clf = LazyClassifier()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)  # Leaderboard!
```

**Results from Class:**
```
Model                      Accuracy  Balanced Acc   AUC
-------------------------------------------------------
KNeighborsClassifier        0.68        0.73       0.73
RandomForestClassifier      0.65        0.70       0.71
SVC (Support Vector)        0.64        0.69       0.70
...
LogisticRegression          0.60        0.65       0.66
```

**Key Finding:**
> "Nonlinear classifiers (KNN, Random Forest) were better than logistic regression (linear). This tells us our problem is nonlinear in nature!"

---

### 10. **K-Nearest Neighbors (KNN)**

**What is KNN?**
- Non-linear supervised classifier
- Classifies based on "neighbors" in feature space
- No explicit model equation!
- Decision boundary emerges from data structure

**How it works:**
1. Store all training data
2. For new point, find K nearest neighbors
3. Majority vote determines class
4. No parameters to learn (lazy learning)

**Why it's non-linear:**
- Can capture arbitrarily complex boundaries
- Adapts to local data structure
- Not constrained to straight lines

**Implementation (in notebook):**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Build KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Cross-validation
scores = cross_val_score(knn, X_train, y_train, cv=5)
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Make predictions
y_pred = knn.predict(X_test)
```

**Our results:** KNN best model with ~73% AUC

---

## 📊 Understanding Model Complexity

### The Parameters vs Observations Rule

**Safe Zone:**
```
N (observations) > P (parameters)
```

**Example Questions from Lecture:**

**Q:** Dog vs cat classifier with 10 billion parameters. How many images needed?

**A:** > 10 billion images!

**Why?**
- Fewer observations than parameters = overfitting risk
- Model can "memorize" training data
- Won't generalize to new data

**Q:** Model with 10 features. How many parameters should model have?

**A:** < 10 parameters ideally

**Why?**
- More parameters than features = too complex
- Will find spurious patterns
- Overfitting likely

---

### The Complexity Trade-Off

**From TensorFlow Playground Spiral Example:**

**2 Neurons:**
- Underfit (can't capture spiral)
- Too simple

**4 Neurons:** ✓
- Good fit (smooth spiral boundary)
- Right complexity

**6 Neurons:**
- Slight overfit (small disconnected regions)
- Starting to memorize noise

**10 Neurons:**
- Severe overfit (many isolated islands)
- Memorizing training set
- Poor generalization

**Lesson:**
> "More parameters isn't always better. Find the sweet spot where model is complex enough to capture true pattern, but simple enough to avoid memorizing noise."

---

## 🍕 The Variance Pizza Analogy

### Understanding Feature Independence

**The Pizza = Total variance in Y (response variable)**

**Each slice = One source of variation**

**Scenario 1: Good Independent Features**
```
🍕 8-slice pizza

X₁ explains: Slices 1, 2, 3 (⅜ of variance)
X₂ explains: Slices 4, 5, 6 (⅜ of variance)
X₃ explains: Slices 7, 8    (¼ of variance)

Total explained: 8/8 = 100% ✓
Overlap: Minimal ✓
This is IDEAL!
```

**Scenario 2: Variance Inflation (Redundant Features)**
```
🍕 8-slice pizza

X₁ explains: Slices 1, 2, 3, 4
X₂ explains: Slices 3, 4, 5, 6  ← Overlap!

Slices 3,4 counted twice (inflated)
Slices 7,8 unexplained
This is PROBLEMATIC!
```

**Variance Inflation Factor (VIF):**
- Measures redundancy between features
- High VIF = correlated features
- **Action:** Remove one of the correlated pair

**Scenario 3: Underfitting (Insufficient Features)**
```
🍕 8-slice pizza

X₁ explains: Slices 1, 2, 3
X₂ explains: Slices 1, 2, 3  ← Same as X₁!

Only 3/8 slices explained
Most variance unexplained
Model will be poor!
```

**Key Insight:**
> "You want features that explain DIFFERENT parts of the variance pie, together covering ALL parts, with minimal overlap."

---

## 🎯 Supervised vs Unsupervised Learning

**Supervised Learning (what we've been doing):**
- **Known:** Structure in response (orange vs blue, seizure vs normal)
- **Known:** Labeled training data
- **Goal:** Learn mapping from features → labels
- **Examples:** Classification, regression

**Unsupervised Learning (coming soon):**
- **Unknown:** Structure in response
- **No:** Labeled data
- **Goal:** Discover structure in features alone
- **Examples:** Clustering, dimensionality reduction

**Why the distinction matters:**
> "If you know the structure of the data (labels), you can fit a supervised model. If you don't know the structure, you use unsupervised learning to discover it."

**Coming in future weeks:**
- Clustering (K-means, GMM)
- PCA (Principal Component Analysis)
- t-SNE (non-linear dimensionality reduction)

---

## 💻 Code from Today

### Loading Pre-Computed Data

```python
# Skip feature engineering, load CSV directly
!cp "/content/drive/MyDrive/Week02/segmentDF_with_frequency_features.csv" /content

import pandas as pd
df = pd.read_csv("segmentDF_with_frequency_features.csv")
df['seizure'] = df['seizure'].astype('category')

# Load into H2O
import h2o
h2o.init(max_mem_size="2G")
df_h2o = h2o.H2OFrame(df)
df_h2o['seizure'] = df_h2o['seizure'].asfactor()
```

### LazyPredict Competition

```python
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

# Prepare data
X = df[['delta', 'theta', 'alpha', 'beta']]
y = df['seizure']

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                     test_size=0.2, 
                                                     random_state=42)

# Run all models
clf = LazyClassifier(verbose=0, predictions=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# View leaderboard
print(models.sort_values('Balanced Accuracy', ascending=False))
```

### KNN with scikit-learn

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc

# Build model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='roc_auc')
print(f"CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Predictions
y_pred_proba = knn.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label=f'KNN (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'r--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

---

## 🔑 Decision Boundary Concepts

### What is a Decision Boundary?

**In 2D Feature Space:**
```
     X₂
      ^
    1 |     • • •        ← Class 1 (Blue)
      |   • • • •
  0.5 | ─────────────    ← Decision boundary
      | • • •
    0 | • • •            ← Class 0 (Orange)
      └──────────────> X₁
        0    0.5    1
```

**The boundary line represents:**
- Probability = threshold
- Classification decision point
- In higher dimensions: hyperplane or surface

**Linear Boundary:**
- Straight line (2D), plane (3D), hyperplane (N-D)
- Created by linear models
- Equation: β₀ + β₁X₁ + β₂X₂ = 0

**Non-Linear Boundary:**
- Curved line, circle, spiral, etc.
- Created by non-linear models OR non-linear features
- Can fit complex patterns

**Optimal Boundary:**
- Captures true underlying pattern
- Not too simple (underfit)
- Not too complex (overfit)
- Generalizes to new data

---

## 🎓 Practical Lessons

### 1. Model Selection Strategy

**Start Simple:**
1. Try linear model (logistic regression)
2. Check performance
3. If poor → Add complexity

**Add Complexity Via:**
- **Feature engineering** (preferred - interpretable)
- **Non-linear model** (powerful but black box)
- **Both** (powerful but risk overfitting)

**Check Generalization:**
- Always use cross-validation
- Compare train vs validation performance
- Large gap → overfitting

### 2. Feature Engineering Principles

**Goals:**
- ✅ Cover all variance in response (whole pizza)
- ✅ Minimal redundancy between features (low VIF)
- ✅ Independent features (different pizza slices)
- ✅ Sufficient but not excessive (sweet spot)

**Red Flags:**
- ❌ More features than observations (M > N)
- ❌ High correlation between features
- ❌ Features all explain same variance
- ❌ Large unexplained variance remaining

---

## 📝 Assignment Clarifications

### Assignment 0

**Report Required?** Optional but recommended
- Canvas says not required
- README suggests including short report
- Professor: "If you can write a short report, that'd be very nice"
- **No penalty** if missing for Assignment 0
- **Helps learning** by reinforcing concepts

### Assignments 1 & 2

**Report Required:** YES!
> "No report, no points!"

**Dataset:** EEG_sleep.mat (NOT session4_train_2018.mat!)

**Why different structure matters:**
- Different nesting levels in .mat file
- Different sampling frequency
- Different array indexing needed
- **Can't just copy-paste class code!**

---

## 🎬 For Next Class (Tuesday, Feb 10)

### Topics:

1. **Elastic Net Regularization**
   - L1 (Lasso), L2 (Ridge), Elastic Net
   - Preventing overfitting
   - Feature selection via regularization

2. **Multi-Class Classification**
   - One-vs-rest
   - Softmax function
   - Confusion matrices for >2 classes

3. **K-Nearest Neighbors in R**
   - `buildKNNModel.R` walkthrough
   - Choosing optimal K
   - Distance metrics

4. **Advanced Evaluation Metrics**
   - Precision-Recall curves
   - Calibration plots
   - Lift charts

### Homework:

- [ ] Run `buildCompetitionModels.ipynb`
- [ ] Try TensorFlow Playground experiments
- [ ] Understand overfitting vs underfitting
- [ ] Complete Assignments 0, 1, 2
- [ ] Write reports for Assignments 1 & 2
- [ ] (BONUS) Implement future prediction

---

## 🙋 Questions from Class

**Q: Is 5-fold better than 3-fold cross-validation?**  
**A:** YES! More folds = more trials = more confidence that model generalizes. Best case is leave-one-out (N-fold) but computationally expensive.

**Q: How does more features lead to overfitting if more data usually helps?**  
**A:** Curse of dimensionality! If M (features) > N (observations), you can "torture data until it confesses anything." Need N > M for safe zone.

**Q: Can you explain variance inflation?**  
**A:** If two features explain the SAME variance in Y, that variance is "counted twice" (inflated). This wastes features and can cause overfitting. Want features explaining DIFFERENT parts of variance.

**Q: Do we need reports for Assignment 0?**  
**A:** Optional for Assignment 0 (no penalty). But REQUIRED for Assignments 1 & 2. Writing reports helps reinforce learning!

**Q: Why did you use session4_train_2018.mat in class but assign EEG_sleep.mat?**  
**A:** "Since assignments are reruns of class work, I want you to use a different dataset to ensure you understand the process, not just copy code."

---

## 🔑 Key Takeaways

1. ✅ Cross-validation tests if model generalizes across data folds
2. ✅ Leave-one-out CV is best but computationally expensive
3. ✅ Overfitting caused by: too many features, too many parameters, too complex model
4. ✅ Underfitting caused by: too simple model, insufficient features
5. ✅ Need: N observations > P parameters > M features (ideally)
6. ✅ Variance inflation = redundant features (bad!)
7. ✅ Non-linearity via feature engineering OR model complexity
8. ✅ TensorFlow Playground shows overfitting visually
9. ✅ Forward pass = inference = prediction (no training)
10. ✅ ROC curves work for ANY continuous variable, not just probabilities!
11. ✅ LazyPredict reveals if problem is linear vs non-linear
12. ✅ KNN is non-linear, lazy learning algorithm

---

*"If you torture the data enough, it will confess to anything."*

— Professor's professor (on overfitting)

---

**Next Class:** Tuesday, February 10 - Elastic Net, Multi-Class Classification, KNN in R

**Keep working on assignments! See you Tuesday!** 🚀
