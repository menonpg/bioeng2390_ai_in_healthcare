# Quiz 1 Solutions: AI in Healthcare Fundamentals
## INSTRUCTOR USE ONLY - DO NOT DISTRIBUTE

**Course:** BIOENG 2390 - Spring 2026  
**Scoring:** Answer 2 questions from each of 6 sections  
**Base Points:** ~90 points  
**Maximum Possible:** ~150 points (with all bonuses)

---

## Section 1: Feature Engineering (20 points)

### Question 1.1 Solution (10 points)

**a) Why amplitude alone failed:**
- T-test p=0.94 indicated seizure and normal states had nearly identical means
- Box plots showed overlapping distributions
- Wilcoxon test p=0.059 suggested shapes differed slightly but not enough
- Single time-point amplitudes can't distinguish states - need windowed analysis

**b) Frequency features that worked:**
- **Delta (1-4 Hz):** p < 0.001 (significant but weakest)
- **Theta (4-7 Hz):** p = 0.074 (NOT significant)
- **Alpha (7-12 Hz):** p < 10⁻¹⁰ (highly significant)
- **Beta (12-20 Hz):** p < 10⁻¹³ (most significant)

**Why effective:** Seizures involve high-frequency chaotic neural firing. Beta/Alpha capture this while Delta/Theta less informative.

**c) Window calculation:**
- Window size: 256 samples (1 second at 256 Hz)
- Step size: 128 samples (50% overlap)
- Formula: # windows = (N - window_size)/step_size + 1
- N = 30 × 256 = 7,680 samples
- Windows = (7,680 - 256)/128 + 1 = 58 + 1 = **59 windows**

---

### Question 1.2 Solution (10 points)

**a) Two-step process:**

**Segmentation:**
- Identify cardiac cycles (R-peak to R-peak)
- Extract each cycle as separate observation
- Align starts and ends

**Resampling:**
- Upsample 250 Hz → 500 Hz using interpolation
- Or downsample 500 Hz → 250 Hz (less preferred)
- Ensure all segments have same number of points

**b) Upsampling preferred:**
- Downsampling can cause **aliasing** - high frequencies appear as low frequencies
- Information loss from discarding samples
- Upsampling uses interpolation to estimate intermediate points
- No information loss (just estimation)

**c) Fourier method:**
- Convert time → frequency domain (FFT)
- Frequency domain has same # points as time domain
- Keep desired frequencies, discard others
- Inverse FFT → resampled signal
- Natural anti-aliasing filter (keeps only representable frequencies)

---

## Section 2: Classification & Model Evaluation (25 points)

### Question 2.1 Solution (10 points)

**Given:**
```
           Normal  Seizure
Normal       850      120   (TN=850, FN=120)
Seizure       50      180   (FP=50, TP=180)
```

**a) Sensitivity = TP/(TP+FN) = 180/(180+120) = 180/300 = 0.60 = 60%**

**b) Specificity = TN/(TN+FP) = 850/(850+50) = 850/900 = 0.944 = 94.4%**

**c) Precision = TP/(TP+FP) = 180/(180+50) = 180/230 = 0.783 = 78.3%**

**d) Accuracy = (TP+TN)/Total = (180+850)/1200 = 1030/1200 = 0.858 = 85.8%**

**e) COVID test threshold:**
- Set LOW threshold (0.05-0.10)
- Maximize sensitivity (catch all infected)
- Accept high false positive rate
- Rational: Better to quarantine healthy people than miss infected ones
- On ROC: Choose point at top of curve (high TPR, accept high FPR)

---

### Question 2.2 Solution (8 points)

**a) AUC = 0.81 means:**
- Area under ROC curve = 81% of maximum possible area
- Model performs better than random (AUC > 0.5)
- "Good" but not excellent (0.8-0.9 range)
- 81% of time, model ranks random positive higher than random negative

**b) Threshold = 0.57 means:**
- Probability cutoff for classification decision
- If P(seizure) ≥ 0.57 → Predict seizure
- If P(seizure) < 0.57 → Predict normal
- This is the "operating point" of the model
- Selected from validation set to optimize performance

**c) Why validation set:**
- Training set already used to fit β coefficients
- Optimizing threshold on training = overfitting
- Validation set = independent data for fair threshold selection
- Prevents "double dipping" in same data

---

### Question 2.3 Solution (7 points)

**Derivation:**

**a) Start:** P = 1/(1 + e^(-(β₀ + β₁X)))

**b) Steps:**
```
1/P = 1 + e^(-(β₀ + β₁X))
1/P - 1 = e^(-(β₀ + β₁X))
(1-P)/P = e^(-(β₀ + β₁X))
ln((1-P)/P) = -(β₀ + β₁X)
-ln((1-P)/P) = β₀ + β₁X
ln(P/(1-P)) = β₀ + β₁X  ✓
```

**c) Why necessary:**
- Left side (log-odds) is continuous (−∞ to +∞)
- Right side is linear combination we can solve with OLS!
- Connects logistic regression to linear regression
- We already know how to solve β₀ + β₁X from Week 02
- Transform probability problem → solvable linear problem

---

## Section 3: Overfitting & Regularization (20 points)

### Question 3.1 Solution (12 points)

**a) Overfitting risks:**
- **MAJOR RISK:** 10,000 features >> 50 observations (M >> N)
- Violates N > P rule from Week 04
- Can "torture data until it confesses anything"
- Model will memorize training set
- Poor generalization to new patients

**b) Three causes in decision trees:**
1. **Too many features:** Tree can split on spurious gene correlations
2. **Too deep tree:** With 50 samples, even depth=10 creates 2¹⁰=1024 leaf nodes (>50!)
3. **No regularization:** No constraints on complexity

**c) Two solutions:**
1. **Dimensionality reduction (PCA):**
   - Reduce 10,000 genes → ~10 principal components
   - Keep 90-95% explained variance
   - Now N=50 > P=10 ✓

2. **Regularization:**
   - Set max_depth limit (e.g., max_depth=3)
   - Require min_samples_split (e.g., ≥10)
   - Use pruning after tree grows
   - Or use Elastic Net in regression

---

### Question 3.2 Solution (8 points)

**a) Pizza slice meaning:**
- Each slice = one source of variation in Y (response)
- Whole pizza = total variance to explain
- Goal: Cover all slices with features

**b) Variance inflation:**
- When 2+ features explain SAME variance (same slices)
- "Counted twice" - redundant features
- Example: Slice 3 explained by both Feature A and Feature B
- **Bad because:** Wastes features, can cause overfitting with redundancy

**c) What to do:**
- **Remove one feature!** (A or B)
- Keep the one more aligned with other unique slices
- Or combine features (A+B)/2
- Check VIF (Variance Inflation Factor)

---

## Section 4: Cross-Validation & Model Selection (15 points)

### Question 4.1 Solution (8 points)

**a) 5-fold CV:**
- Split data into 5 equal parts
- Train on 4 folds, validate on 1 fold
- Repeat 5 times (each fold used for validation once)
- **5 models trained total**
- Combine all out-of-sample predictions for ROC

**b) Leave-one-out CV:**
- N-fold CV where N = number of observations
- Train on N-1, validate on 1
- Repeat N times
- **Best CV** but computationally expensive
- Use when: Small dataset, need maximum confidence

**c) Why CV better:**
- Tests model on different data folds
- Detects if model sensitive to data selection
- More robust performance estimate
- Single split could be "lucky" or "unlucky"

---

### Question 4.2 Solution (7 points)

**a) Calculations:**
- Mean = (0.78 + 0.82 + 0.70 + 0.85 + 0.72)/5 = 3.87/5 = **0.774**
- Std = √(Σ(x-mean)²/N) = √((0.006² + 0.046² + 0.074² + 0.076² + 0.054²)/5)
- Std ≈ **0.057**

**b) High variation indicates:**
- Model is sensitive to training data selection
- Performance depends on which fold used
- Typical for non-linear models (KNN)
- Some folds might be missing key patterns

**c) Production trust:**
- **Moderate trust:** Mean AUC 0.774 is decent
- **Concern:** High std dev (0.057) suggests instability
- **Recommendation:** Need more data or simpler model
- Would deploy with caution and monitoring

---

## Section 5: Dimensionality Reduction (20 points)

### Question 5.1 Solution (10 points)

**a) Structure:**
- Class labels (colors, categories)
- Example: Seizure vs Normal, Cancer vs Healthy
- What we want to predict
- Visual: Red dots vs green dots

**b) Variance:**
- How data points spread/distributed
- Example: Standard deviation, covariance
- Measured numerically
- Visual: Shape of scatter plot

**c) Supervised without structure?**
- **NO!** Need Y labels to learn X → Y mapping
- Without labels, nothing to predict

**Supervised without variance?**
- **NO!** If all features identical, can't distinguish classes
- Need variation in X to predict Y

**d) Unsupervised assumption:**
- **"Variance must predict structure"**
- Assumes: Natural clusters (variance) = true classes (structure)
- **Fails when:** Data scattered randomly despite having true classes
- Example: Cancer cells mixed randomly with normal cells

---

### Question 5.2 Solution (10 points)

**a) Explained variance:**
- PC0: 120/500 = 0.24 = **24%**
- PC1: 80/500 = 0.16 = **16%**
- Combined: (120+80)/500 = 200/500 = 0.40 = **40%**

**b) Is this enough?**
- **NO!** Only 40% variance explained
- Losing 60% of information
- **Should keep more:** Aim for 90-95%
- Need ~10-15 components likely
- Severe dimensionality reduction loses too much

**c) Information lost in 3D → 2D:**
- Variance along 3rd dimension (perpendicular to plane)
- Points far apart in 3D may appear close in 2D
- Separability in excluded dimension lost
- Example: Two points at different heights but same x,y

**d) t-SNE non-reproducible:**
- **Why not:** Random initialization, no fixed transformation
- **No .transform() method** in sklearn
- **Workaround:** Train regression models
  - Predict t-SNE_x from original features
  - Predict t-SNE_y from original features
  - Apply to new data via regression
- **Limitation:** Quality depends on regression R²

---

## Section 6: Conceptual Understanding (30 points)

### Question 6.1 Solution (5 points)
**Purpose of dimensionality reduction:**
- Visualize high-dimensional data in 2D/3D
- Remove redundant/correlated features
- Reduce overfitting risk (fewer features)
- Speed up computation
- Find most important features/directions
- Enable clustering when N features >> M observations

### Question 6.2 Solution (5 points)
**Structure vs Variance:**
- **Structure:** Class labels (categories). Example: Red dots = cancer, blue dots = normal
- **Variance:** How data spreads. Example: Standard deviation, covariance between features
- **Key:** Structure is what we predict (Y). Variance is in features (X).
- Unsupervised assumes variance patterns reveal structure

### Question 6.3 Solution (5 points)
**Techniques:**
- **Aligns to structure:** LDA (Linear Discriminant Analysis) - supervised, maximizes class separation
- **Aligns to variance:** PCA (Principal Component Analysis) - unsupervised, maximizes variance
- LDA needs labels, PCA doesn't
- LDA finds directions that separate classes, PCA finds directions of maximum spread

### Question 6.4 Solution (5 points)
**Linear vs Non-linear:**
- **Linear:** Output is linear combination of inputs. Example: y = β₀ + β₁x₁ + β₂x₂
- **Non-linear:** Output involves interactions, powers, etc. Example: KNN (voting), trees (thresholds)
- Linear = straight line/plane boundary
- Non-linear = curved/complex boundary
- Can achieve non-linearity via: features (x²) OR model complexity (neural nets)

### Question 6.5 Solution (5 points)
**Symptoms of non-linear unsupervised clustering:**
- Non-reproducible (random initialization)
- Distance-based (like KNN)
- No .transform() method
- Different results each run
- Can't easily apply to new data

**Two methods:**
1. **t-SNE:** Best for visualization, preserves local structure
2. **Isomap:** Geodesic distances, manifold learning

(Also accept: LLE, MDS if explained correctly)

### Question 6.6 Solution (5 points)
**Three binary classifier metrics:**
1. **Sensitivity/Recall:** TP/(TP+FN) - catching positive cases
2. **Specificity:** TN/(TN+FP) - correctly identifying negatives
3. **Precision/PPV:** TP/(TP+FP) - reliability of positive predictions

(Also accept: Accuracy, F1-score, AUC with explanations)

### Question 6.7 Solution (5 points)
**Number of PCs generated:**
- **Answer: N principal components** (where N = number of features/dimensions)
- For M vectors, each N-dimensional: Generate N PCs
- Can select top K where K < N
- Example: 400 genes → 400 PCs possible, but use top 2-10

### Question 6.8 Solution (5 points)
**Knowing if PCA will work for clustering:**
- **Check explained variance:** If top 2 PCs explain >90%, likely works
- **Visual inspection:** See if clusters visible in PC space
- **Scree plot:** Sharp elbow suggests good 2D reduction
- **Domain knowledge:** If variance = structure assumption holds
- **Validation:** Run K-means, check cluster quality metrics

---

## Bonus Section Solutions (+20 points)

### Option A: Grokking
**Expected elements:**
- Screenshot showing training/validation curves
- Identification of grokking point (where validation suddenly improves)
- Connection to Week 04 overfitting concepts
- Discussion of 777 parameters vs larger models
- Reference to N > P rule

### Option B: Mini-Project
**Expected elements:**
- Code showing feature extraction (e.g., Delta and Beta power)
- Classifier trained (any method acceptable)
- AUC reported with confusion matrix
- Discussion of train vs test performance
- Assessment: "Did I overfit?" with evidence
- If train >> test accuracy → overfitting

---

## 📊 Grading Notes

**Full credit criteria:**
- Shows work for calculations
- References specific lectures
- Demonstrates understanding, not just memorization
- Makes connections across concepts
- Provides reasoning for answers

**Partial credit:**
- Correct method but calculation errors
- Right idea but incomplete explanation
- Missing lecture references but correct concept

**No credit:**
- No attempt
- Completely incorrect understanding
- Copied without comprehension

---

**This solutions file is for instructor use only. Do not distribute to students before quiz deadline.**

