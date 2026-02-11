# BIOENG-2390 Spring 2026 - Lecture 9
## February 10, 2026

**Instructor:** Professor Prahlad G. Menon, PhD, PMP  
**TA:** Bhavya Iyer  
**Recording:** [View on Fathom](https://fathom.video/share/xQ7iyUP__2_szfK_DqgEDLnG_8nvayER)  
**Duration:** 72 minutes

---

## 📋 Lecture Overview

Today's class focused on:
1. TA session on assignment expectations and grading
2. K-Nearest Neighbors (KNN) algorithm deep dive
3. How KNN works: distance calculation and voting mechanism
4. Variable importance in non-linear models (experimental approach)
5. Introduction to K-means clustering (unsupervised learning)
6. Principal Component Analysis (PCA) preview
7. Elastic Net regularization (L1 Lasso + L2 Ridge)
8. Structure vs variance in data
9. Supervised vs unsupervised learning distinction
10. Quiz announcement for next Thursday (Feb 19)

---

## 👩‍🏫 TA Session (First 15 minutes)

### Assignment Requirements Clarification

**From Bhavya (TA):**

**For ALL Assignments:**
1. ✅ Jupyter Notebook (.ipynb)
2. ✅ H2O Flow file (.flow) if used
3. ✅ **Written Report** (Word/PDF)

**What the report should include:**
- Summary of what you did
- What you understood from the data
- Explanation of figures/outputs
- Answers to conceptual questions
- Comparison of raw vs engineered data (for Assignment 1)
- Description of which features predict best

**Assignment 0:**
- Report was optional (everyone got credit regardless)
- But recommended for learning reinforcement

**Assignments 1 & 2:**
- **Report is REQUIRED**
- "No report, no points"
- Can include screenshots of outputs if code doesn't run on grader's system

**Assignment 1 Specific Deliverables:**
- Descriptive statistics of raw data
- Descriptive statistics of engineered data
- Comparison: which is better for prediction?
- Answer conceptual questions (e.g., what if window size changes?)
- Identify which features (Delta/Theta/Alpha/Beta) predict best

**Tips from Bhavya:**
> "This class has a very diverse background. It can be overwhelming because you're learning eigenvalues, vectors, math, statistics. Use AI to help with syntax! The focus is understanding concepts, not memorizing code."

---

## 🎯 K-Nearest Neighbors (KNN) Explained

### How KNN Works

**Visual Explanation:**

**Setup:**
```
Feature Space (2D):
     X₂
      ^
      |  • • •        ← Green class
      |• • • • •
      |  ? (purple)  ← New observation
      |• • •          ← Red class
      |_____________> X₁
```

**Algorithm Steps:**

**1. Calculate Distances**
```
For new observation (purple dot):
- Calculate distance to EVERY training point
- Distance formula: √[(x₁-x₁')² + (x₂-x₂')²]
```

**2. Sort by Distance**
```
Sort all distances in ascending order:
Point 1: distance = 0.15
Point 2: distance = 0.23
Point 3: distance = 0.31
...
```

**3. Select K Nearest**
```
If K=5, select 5 closest points:
Point 1: Green
Point 2: Green
Point 3: Red
Point 4: Green
Point 5: Red
```

**4. Majority Vote**
```
Count classes:
Green: 3 votes
Red: 2 votes

Prediction: Green (majority wins!)
```

**Key Parameter: K**
- Default in scikit-learn: K=5
- Can customize: `KNeighborsClassifier(n_neighbors=3)`
- Too small K → Overfitting (sensitive to noise)
- Too large K → Underfitting (over-smoothing)

---

### Why KNN is Non-Linear

**No Equation!**
- Can't write: `y = β₀ + β₁x₁ + β₂x₂`
- Decision boundary emerges from data structure
- Adapts to local patterns

**What's "Saved" in Model:**
- Entire training dataset
- Distance calculation method
- K parameter

**Prediction Process:**
- Find K nearest neighbors
- Majority vote
- No matrix multiplication!

**Decision Boundary:**
- Can be arbitrarily complex
- Follows data clusters
- Not constrained to straight lines or simple curves

---

### Variable Importance in KNN

**Problem:** No β coefficients to interpret!

**Solution:** Experimental Approach

**Method:**
1. Train KNN on all features
2. Make predictions, measure accuracy
3. Remove one feature (e.g., remove Alpha)
4. Retrain and predict again
5. If accuracy drops significantly → feature was important!
6. Repeat for each feature

**From Today's Notebook:**
```
Feature Importance (from permutation):
1. Alpha: Most important
2. Theta: Second most important
3. Beta: Less important  
4. Delta: Least important
```

**Contrast with Linear Models:**
- **Linear:** Read β coefficients directly
- **Non-linear:** Run experiments to discover importance

**Professor's Point:**
> "You can't determine variable importance from an equation in non-linear models. You determine it by experimental observation - trying different feature combinations and seeing what works."

---

## 🎯 K-Means Clustering (Unsupervised)

### The Difference: No Labels!

**K-Nearest Neighbors (Supervised):**
- **Known:** Class labels in training data
- **Goal:** Classify new observations
- **Method:** Find similar labeled points, vote

**K-Means Clustering (Unsupervised):**
- **Unknown:** Class labels
- **Goal:** Discover natural groups in data
- **Method:** Group by distance, identify clusters

---

### How K-Means Works

**Algorithm:**

**1. Initialize K cluster centers randomly**

**2. Assign each point to nearest center**
```
For each point:
  distances = [dist to center1, dist to center2, ..., dist to centerK]
  assign to cluster with minimum distance
```

**3. Recalculate cluster centers**
```
For each cluster:
  new_center = mean of all points in cluster
```

**4. Repeat steps 2-3 until convergence**

**Output:**
- K clusters (groups)
- Each point assigned to one cluster
- No guarantee clusters match true classes!

---

### Structure vs Variance

**Critical Assumption:**
> "Variance must be a predictor of structure"

**What this means:**
- If data points cluster naturally (variance patterns)
- AND those clusters correspond to true classes (structure)
- THEN K-means will work!

**When it fails:**
- Variance doesn't match structure
- Clusters found ≠ true classes
- Example: Red/green labels scattered randomly

**Visual Example:**
```
Clustered Data (K-means works):
   •••         •••
  • • •       • • •  ← Two natural clusters
   •••         •••
  Cluster1    Cluster2

Random Data (K-means fails):
  • • • • • •  
 • • • • • • •  ← No natural clusters
  • • • • • •
```

**From Lecture:**
> "Sometimes variation in your data does not indicate structure. Even if you cluster your data, you may not fully identify classes."

---

## 📐 Introduction to Principal Components

### The Setup

**Given:** Data in original feature space (X₁, X₂)

**Observation:** Points spread more in one direction

**Visual:**
```
     X₂
      ^
      |    /
      |  /••••  ← Data spreads diagonally
      |/•••••
      |_______> X₁
      
      PC₁ direction: /
      PC₂ direction: ⊥ to PC₁
```

**Principal Components:**
- **PC₀ (First PC):** Direction of maximum variance
- **PC₁ (Second PC):** Perpendicular to PC₀, next most variance
- Also called: **Eigenvectors** of covariance matrix

**Key Insight:**
> "The principal or eigen mode of variation of the data might be more aligned with one original feature than another. That's why some features are more important for prediction!"

**Example from Class:**
- X₂ aligned with principal direction of variance
- X₁ less aligned
- **Result:** X₂ more important predictor than X₁

---

## 🔧 Elastic Net Regularization

### The Three Regularization Methods

**1. Lasso (L1) - "The Eliminator"**
```
Loss = (Y - Xβ)² + λ·Σ|βⱼ|
```
- **Penalty:** Sum of absolute values of β
- **Effect:** Forces some β to exactly zero
- **Use:** Feature selection (eliminates useless features)
- **Harsh:** Picks winners, eliminates losers

**2. Ridge (L2) - "The Shrinker"**
```
Loss = (Y - Xβ)² + λ·Σ(βⱼ²)
```
- **Penalty:** Sum of squared β values
- **Effect:** Shrinks all β toward zero (but not to zero)
- **Use:** When all features somewhat useful
- **Gentle:** Reduces all coefficients evenly

**3. Elastic Net - "Best of Both"**
```
Loss = (Y - Xβ)² + λ₁·Σ|βⱼ| + λ₂·Σ(βⱼ²)
      = (Y - Xβ)² + α·(L1_ratio·Σ|βⱼ| + (1-L1_ratio)·Σβⱼ²)
```
- **Combines:** Lasso + Ridge
- **Effect:** Select features AND shrink coefficients
- **Use:** Best general-purpose regularization
- **Used in:** H2O GLM models automatically!

---

### Why Regularization Matters

**Problem Without Regularization:**
```
β values widely distributed:
β₁ = 100  ← Very large weight
β₂ = 0.01 ← Tiny weight
β₃ = -50  ← Large negative weight
β₄ = 0.1  ← Small weight
```
- Some features dominate
- Small changes in dominant features = huge prediction changes
- Model unstable

**Solution With Regularization:**
```
β values narrowly distributed:
β₁ = 2.3
β₂ = 1.8  ← All similar magnitude
β₃ = 2.1
β₄ = 1.9
```
- Balanced feature contributions
- More robust predictions
- Better generalization

**Training Wheels Analogy:**
> "Think of regularization like training wheels on a bike. It prevents the model from going too wild and falling over by making very large weights for some features."

---

## 🔬 Supervised vs Unsupervised Learning

### Supervised Learning (What we've done so far)

**Requirements:**
- **Labeled training data** (know which is orange, which is blue)
- **Known structure** (know the classes exist)

**Goal:** Learn function mapping features → labels

**Examples:**
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Random Forests, XGBoost
- Neural Networks (classification/regression)

**Key:** Structure is KNOWN

---

### Unsupervised Learning (Starting Thursday)

**NO labeled data!**
- Don't know which is orange, which is blue
- Don't know how many classes exist
- Don't even know if classes exist!

**Goal:** Discover structure from variance alone

**Examples:**
- K-Means Clustering
- Principal Component Analysis (PCA)
- t-SNE, UMAP
- Autoencoders

**Key Assumption:**
> "Variance must be a predictor of structure"

**What this means:**
- If points cluster naturally (variance pattern)
- Those clusters might correspond to meaningful groups (structure)
- But no guarantee!

---

## 💻 Code from Today's Notebook

### KNN Implementation

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

# Initialize KNN
knn = KNeighborsClassifier(n_neighbors=5)  # Default K=5

# 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics
aucs = []
sensitivities = []
specificities = []
precisions = []

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train
    knn.fit(X_train, y_train)
    
    # Predict probabilities
    y_pred_proba = knn.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    aucs.append(auc)
    # ... calculate sensitivity, specificity, precision

print(f"Mean AUC: {np.mean(aucs):.3f} (+/- {np.std(aucs):.3f})")
```

### Saving KNN Model

```python
import joblib

# Train on all data
knn.fit(X, y)

# Save model (pickle file)
joblib.dump(knn, 'knn_model.pkl')
```

### Loading and Using Saved Model

```python
# Load model
knn_loaded = joblib.load('knn_model.pkl')

# Make predictions
predictions = knn_loaded.predict(X_new)
probabilities = knn_loaded.predict_proba(X_new)[:, 1]
```

### Feature Importance (Permutation Method)

```python
from sklearn.inspection import permutation_importance

# Calculate importance
result = permutation_importance(knn, X_test, y_test, 
                                n_repeats=10, 
                                random_state=42)

# Get importances
importances = result.importances_mean

# Visualize
import matplotlib.pyplot as plt
features = ['delta', 'theta', 'alpha', 'beta']
plt.barh(features, importances)
plt.xlabel('Importance')
plt.title('KNN Feature Importance')
plt.show()
```

---

## 🔑 Key Concepts

### 1. **Distance Metrics in KNN**

**Euclidean Distance (default):**
```
d = √[(x₁-x₁')² + (x₂-x₂')² + ... + (xₙ-xₙ')²]
```

**Other options:**
- **Manhattan:** Sum of absolute differences
- **Minkowski:** Generalization of Euclidean/Manhattan
- **Cosine:** Angle between vectors

**Impact of scale:**
- Features with large ranges dominate distance
- **Solution:** Normalize/standardize features first!

---

### 2. **Choosing K**

**Small K (e.g., K=1):**
- Very flexible boundary
- Sensitive to noise
- **Risk:** Overfitting

**Large K (e.g., K=100):**
- Smooth boundary
- Less sensitive to noise
- **Risk:** Underfitting

**Optimal K:**
- Use cross-validation!
- Try K = 1, 3, 5, 7, 9, ...
- Select K with best CV performance

**From Class:** K=5 is common default (works well often)

---

### 3. **Principal Components Intuition**

**Original Features:** X₁, X₂ (coordinate system)

**Principal Components:** PC₀, PC₁ (rotated coordinate system)

**Why rotate?**
- PC₀ captures maximum variance
- PC₁ captures second-most variance (perpendicular to PC₀)
- Often: PCs more aligned with class separation than original features

**Benefits:**
- Reduce dimensions (use only PC₀, ignore PC₁)
- Remove correlated features
- Visualize high-dimensional data

**Thursday:** Full PCA mathematical derivation!

---

### 4. **Regularization Prevents Overfitting**

**Problem:**
```
Without regularization:
β = [100, 0.01, -50, 0.1]  ← Widely distributed!
```

**Solution:**
```
With elastic net:
β = [2.3, 1.8, 2.1, 1.9]  ← Narrowly distributed ✓
```

**How it works:**
```
Traditional OLS: Minimize (Y - Xβ)²
With Lasso:      Minimize (Y - Xβ)² + λ·Σ|β|
With Ridge:      Minimize (Y - Xβ)² + λ·Σβ²
With Elastic Net: Minimize (Y - Xβ)² + λ₁·Σ|β| + λ₂·Σβ²
```

**When minimizing:**
- Must balance fitting data AND keeping β small
- Trade-off controlled by λ (lambda)
- Larger λ → More regularization → Smaller β

**Note on L1 (from class question):**
> "Do we take magnitude of β? Yes! L1 penalty is Σ|βⱼ| because β can be positive or negative."

---

## 🎓 Important Insights from Lecture

### 1. Non-Linear Models are Data-Sensitive

**From KNN Cross-Validation Results:**
```
Fold    Sensitivity  Specificity  AUC
---------------------------------------
1       0.65         0.80         0.78
2       0.78         0.72         0.82
3       0.70         0.75         0.81
4       0.82         0.68         0.79
5       0.68         0.78         0.80
```

**Large variation across folds!**

**Why?**
> "Non-linear methods are more susceptible to dataset choice. A fold missing certain patterns will perform differently than one that includes them."

**Lesson:** Always use cross-validation for non-linear models!

---

### 2. Separability and Principal Directions

**Dallas's Question:** Would X₂ have larger importance?

**Answer:** YES!

**Why?**
```
     X₂ (important!)
      ^
      |    ↗ Principal direction
      |  ↗
      |↗••••  ← Data spreads more along X₂
      |•••••
      |_______> X₁ (less important)
```

**General Rule:**
- Features aligned with principal variance direction
- → More important for prediction
- Features perpendicular to variance
- → Less important

**This motivates PCA!**
- Rotate to principal component axes
- Use components as features
- Often better than original features

---

### 3. Structure vs Variance

**Variance:**
- How data points are spread out
- Measured by covariance, standard deviation
- **Captured by:** Features (X₁, X₂, ...)

**Structure:**
- True underlying classes/groups
- Pattern in response variable (Y)
- **What we want to predict**

**Supervised Learning:**
- Use variance to predict structure
- Know structure (labels) during training

**Unsupervised Learning:**
- Discover structure from variance alone
- Assume: high variance regions = different classes
- **Risky:** Assumption may not hold!

---

## 📝 Pickle Files (.pkl)

**What are they?**
- Binary files storing Python objects
- Can save any variable type
- Better than CSV for complex objects

**Use cases:**
- Save trained models
- Save feature engineering results
- Save preprocessors (scalers, encoders)

**Code:**
```python
import joblib

# Save
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
```

**Advantages over CSV:**
- Preserves exact Python object
- Can store multiple objects
- Faster for large data
- Model artifacts in one file

---

## 🎬 For Next Class (Thursday, Feb 12)

### Topics:

**1. Dimensionality Reduction Deep Dive**
- PCA mathematical derivation
- Eigenvalues and eigenvectors
- Covariance matrix
- Choosing number of components

**2. PCA for Visualization**
- Reducing 4D data (our features) to 2D
- Scatter plots in PC space
- Interpreting principal components

**3. PCA vs Other Methods**
- Linear (PCA, LDA)
- Non-linear (t-SNE, UMAP)
- When to use which

### Homework:

- [ ] Run `knn_feature_importances.ipynb` completely
- [ ] Understand KNN voting mechanism
- [ ] **Complete Assignments 0, 1, 2 before dimensionality reduction!**
- [ ] Review eigenvectors/eigenvalues (linear algebra)
- [ ] Watch Lecture 8 recording (cross-validation, overfitting)

### Important Deadline:

**Quiz 1:** Thursday, February 19 (next week!)
- Open book, open notes
- Covers fundamentals through dimensionality reduction
- "Forcing function to think in controlled setting"
- Focus on concepts, not memorization

---

## 📋 Week 05 Lecture 9 Checklist

- [ ] `git pull` to get Week 05 content
- [ ] Run `knn_feature_importances.ipynb`
- [ ] Understand KNN algorithm (distance + voting)
- [ ] Understand variable importance in non-linear models
- [ ] Understand K-means clustering concept
- [ ] Understand elastic net regularization
- [ ] Understand structure vs variance distinction
- [ ] Complete Assignments 0, 1, 2
- [ ] Prepare for Quiz 1 (Feb 19)

---

## 🙋 Questions from Class

**Q: Do we need reports for Assignment 0?**  
**A:** Optional (no penalty if missing). But recommended for learning. Assignments 1 & 2: **Required!**

**Q: What should the report include?**  
**A (from Bhavya):** 
- Summary of code
- What each figure means
- What you understood
- Comparison of raw vs engineered data
- Which features predict best
- Answers to conceptual questions in assignment prompt

**Q: Can we use AI to help with coding syntax?**  
**A (from Bhavya):** YES! 
> "The whole point of this class is to use AI to code. You don't have to know how to code. The challenge is understanding the data and concepts, not syntax."

**Q: For L1 penalty, do we take magnitude of β?**  
**A:** YES! Because β can be positive or negative. L1 = Σ|βⱼ|

**Q: Why does X₂ have larger importance?**  
**A:** Because data spreads more along X₂ direction. Features aligned with principal variance directions are more predictive.

---

## 🎓 Professor's Notes

**On Non-Linear Models:**
> "K-Nearest Neighbors is our first non-linear classifier. It works by voting, not equations. You can't write y = β·X for KNN. The decision boundary emerges from the data structure itself."

**On Regularization:**
> "Lasso eliminates, Ridge shrinks, Elastic Net does both. In H2O GLM, this happens automatically to ensure your β coefficients don't go wild."

**On Supervised vs Unsupervised:**
> "If you know the structure (labels), use supervised learning (KNN, logistic regression). If you don't know structure, use unsupervised (K-means, PCA) to discover it. But unsupervised assumes variance predicts structure - not always true!"

**On Assignment Deadlines:**
> "Try to finish Assignment 2 this week before we dive into dimensionality reduction. It's a whole new ballgame - you can easily forget supervised learning concepts if you don't practice them now!"

**For Quiz:**
> "Open book, open notes. It's a forcing function to think in a controlled setting about fundamentals."

**Professor Prahlad Menon, PhD, PMP**  
**TA: Bhavya Iyer**  
*Office Hours: By appointment*  
*Email: prm44@pitt.edu*

---

*"Variance must be a predictor of structure."*

— Core assumption of unsupervised learning

---

## 🔑 Key Takeaways

1. ✅ KNN classifies by distance + K-neighbor voting
2. ✅ KNN is non-linear (no equation)
3. ✅ Variable importance in non-linear models requires experiments
4. ✅ K-means clustering = unsupervised KNN
5. ✅ Principal components = directions of maximum variance
6. ✅ Lasso (L1) eliminates features, Ridge (L2) shrinks all
7. ✅ Elastic Net combines Lasso + Ridge (best of both)
8. ✅ Supervised = known labels, Unsupervised = discover labels
9. ✅ Structure vs variance are different concepts
10. ✅ Quiz 1 next Thursday (Feb 19) - fundamentals + dim reduction

**Next Class:** PCA deep dive with eigenvalue/eigenvector math!

**Happy Valentine's Day weekend! Finish those assignments!** 💝🚀
