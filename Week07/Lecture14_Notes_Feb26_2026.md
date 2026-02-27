# BIOENG-2390 Spring 2026 - Lecture 14
## February 26, 2026

**Instructor:** Professor Prahlad G. Menon, PhD, PMP  
**Recording:** [View on Fathom](https://fathom.video/share/zkCJkr8yzADPsqqvvHMgqkdfn1fL-Ymb)  
**Duration:** 93 minutes

---

## 📋 Lecture Overview

Today's class covered:
1. **Administrative Updates** - Zoom AI companion replacing Fathom, quiz grading clarification
2. **Quiz 1 Review** - Solutions posted, grading out of 90 base points (up to 150 with bonuses)
3. **Grokking Deep Dive** - Detailed exploration of the grokking phenomenon in neural networks
4. **Neural Network Transformers** - 777-parameter transformer for learning addition
5. **Blog Post Discussion** - Professor's three blog posts on grokking experiments
6. **Gaussian Mixture Models (GMM)** - Alternative to K-means clustering with probabilistic boundaries
7. **K-Means Clustering Theory** - Combinatorics of decision boundaries
8. **Linear Discriminant Analysis (LDA)** - Supervised dimensionality reduction
9. **MNIST Dataset** - 2D image dimensionality reduction (28×28 → 784D → reduced space)
10. **t-SNE Perplexity** - Understanding the perplexity parameter
11. **Project Team Formation** - Final push to form teams and identify datasets

**Key Focus:** Understanding generalization vs memorization through the lens of grokking, and expanding dimensionality reduction toolkit with GMM, LDA, and nonlinear methods.

---

## 🎯 Main Topics

### Part 1: Administrative & Quiz Review (0:00-17:00)
- Fathom recording changes
- Quiz 1 grading structure clarification
- Quiz solutions posted to repository

### Part 2: Grokking Phenomenon (17:00-35:00)
- What is grokking? (Generalization after memorization)
- 777-parameter transformer learning addition
- Blog post walkthrough: three attempts at reproducing grokking
- Connection to overfitting and the N > P rule

### Part 3: Transformer Architecture (35:00-48:00)
- Attention mechanism basics
- Forward pass through neural network
- Training loop with Adam optimizer
- Cosine learning rate scheduling
- Batch training (512 examples at a time)

### Part 4: Dimensionality Reduction Extensions (48:00-75:00)
- Gaussian Mixture Models (GMM) vs K-Means
- Linear Discriminant Analysis (LDA) - supervised dimensionality reduction
- MNIST dataset introduction (28×28 pixel handwritten digits)
- Reshaping 2D images to 1D vectors (784 dimensions)
- Multiple methods comparison: PCA, NMF, Isomap, LLE, t-SNE

### Part 5: Project Planning (75:00-93:00)
- Team formation status check
- Dataset identification urgency
- Assignment 4 deadline discussion (postponed to after spring break)
- 30 minutes per class dedicated to projects starting next week

---

## 📝 Detailed Notes

## 1. Administrative Updates

### 1.1 Recording Platform Changes

**Announcement:**
> "It turns out that Fathom is now suddenly going to be blocked from Zoom. So hopefully, we will still be able to get our meeting transcripts and all in the future. But we have some AI companion in Zoom as well."

**Impact:**
- May affect future lecture recordings
- Zoom's built-in AI may replace Fathom
- Professor monitoring the situation

### 1.2 Quiz 1 Grading Structure

**Student Question (Michael):**
> "Regarding the points for the quiz, I got kind of confused with the numbers. It's out of 100, right?"

**Professor's Clarification:**
- **Base points:** 90 (answering 2 questions from each of 6 core sections)
- **Maximum possible:** ~150 (with all bonuses)
- **Canvas showing:** 30 (error - will be fixed to 90)

**Grading Breakdown:**
```
6 sections × 2 questions × variable points = 90 base
Section 7 (bonus) = additional points
Grokking exercise = bonus points
Total possible ≈ 150 points
```

**Note:** Canvas denominator needs updating from 30 to 90.

---

## 2. Grokking Phenomenon Deep Dive

### 2.1 What is Grokking?

**Definition:**
> "Grokking is the colloquial term for generalization of a learning exercise."

**Key Characteristics:**
1. **Phase 1:** Model quickly memorizes training data (training loss → 0)
2. **Phase 2:** After many more epochs, sudden jump in test accuracy
3. **"Aha Moment":** Model discovers underlying concept/pattern
4. **Delayed:** Can happen after 10,000+ training epochs

**Analogy:**
> "It's a little bit like finding a snow leopard. You don't expect to see it all the time, but you will be able to see it."

**Connection to Course Concepts:**
- Overfitting = memorization (Week 4)
- N > P rule (more observations than parameters prevents overfitting)
- Training vs validation accuracy gap
- Generalization as the ultimate goal

### 2.2 The 777-Parameter Transformer

**Problem:** Learn the operation of addition
- Input: Two numbers (A, B)
- Output: Their sum (C = A + B)
- Constraint: Use modular arithmetic (C mod P) to simplify

**Why This is Hard:**
- Infinite possible input pairs
- Cannot memorize all examples
- Must learn the concept of addition
- Generative model (produces output, not just classifies)

**Model Size:**
- **777 parameters** (intentionally small)
- Compare to large language models: billions of parameters
- Similar complexity to decision tree or linear model with 777 features

**Why Small Models?**
> "By making smaller numbers of parameters, you will be in better shape to prevent overfitting or prevent memorization of your training data."

**N > P Rule Application:**
- Training examples > 1,000
- Parameters = 777
- N > P ✓ (should prevent overfitting)
- But grokking still delayed!

### 2.3 Professor's Blog Post Series

**Blog:** [blog.themenonlab.com](https://blog.themenonlab.com)

**Three Posts on Grokking:**

**Post 1:** Vision Transformers (ViT)
- Applied to MNIST digit classification (28×28 pixels)
- Classification task (0-9)
- Model worked well
- NOT generative (just classifying)

**Post 2:** Replicating 777-Parameter Addition Model
- Attempted to reproduce published results
- Trained for ~10,000 epochs
- Achieved 100% training accuracy (memorization ✓)
- Test accuracy stuck at ~15% (grokking ✗)
- **Five attempts** through the night (until 3 AM!)
- Never achieved grokking event

**Post 3:** Understanding Grokking Factors
- Grokking can be delayed by:
  - Choice of GPU
  - Number of training epochs
  - Learning rate
  - Batch size
  - Random initialization
  - Numerical stability

**Professor's Hypothesis:**
> "I was just hypothesizing that maybe 20,000 or 50,000 might eventually work, or 100,000."

**Takeaway:** Grokking is real but elusive - like finding a rare phenomenon in nature!

### 2.4 Transformer Architecture Explained

**Paper:** "Attention is All You Need" (2017)
- Foundation of all modern transformers
- GPT, BERT, ChatGPT all based on this
- Available on arXiv

**Key Components:**

1. **Multi-Head Attention**
   - Core innovation
   - Allows model to focus on different parts of input
   - Multiple attention heads work in parallel

2. **Feed-Forward Networks**
   - Connect neuron layers
   - Apply transformations between attention blocks

3. **Add & Normalize**
   - Residual connections (yellow blocks in diagram)
   - Help with training stability
   - Add input to output of each layer

4. **Activation Functions**
   - Logit (from logistic regression - we know this!)
   - Softmax (multi-class generalization)
   - GELU (Gaussian Error Linear Unit) - used in code

**Professor's Implementation:**
- Modified transformer architecture
- Simplified for learning addition
- Uses PyTorch framework
- Includes all key transformer concepts

---

## 3. Neural Network Training Process

### 3.1 Forward Pass

**What Happens:**
```python
# Input: A, B (two numbers)
# Pass through transformer blocks
# Output: Predicted sum C
```

**Transformer Block Structure:**
1. Input embedding
2. Multi-head attention
3. Add & normalize
4. Feed-forward network
5. Add & normalize
6. Output

### 3.2 Training Loop

**Key Parameters:**

**Learning Rate:** 0.02 (2%)
> "Instead of changing the weights by large amounts, you change them by the amount that you want to change them multiplied by the learning rate."

- Typical: 0.001 (0.1%)
- Here: 0.02 (large for experimentation)
- Controls how fast weights update

**Batch Size:** 512
- Send 512 training examples at once
- Model learns from batch
- Updates weights
- Repeat with next 512 examples

**Epochs:** 10,000+
- One epoch = one pass through all training data
- Grokking may require 20,000-100,000 epochs
- Professor stopped at 10,000

**Optimizer:** Adam
- Alternative to Ordinary Least Squares (OLS) we learned
- Better for neural networks
- Adaptive learning rates per parameter

**Cosine LR Scheduling:**
- Prevents numerical instability
- Gradually reduces learning rate
- Helps convergence
- Prevents "explosion" of gradients

### 3.3 Loss Function

**What is Loss?**
> "At the end of every batch of training, I save the loss, which is kind of the residual. Remember when you said AX equal to B, AX minus B was called a residual."

**In Neural Networks:**
- **Training loss:** Error on training data
- **Test loss:** Error on held-out test data
- **Goal:** Minimize both
- **Overfitting:** Training loss ↓↓, test loss stays high
- **Grokking:** Test loss suddenly ↓ after long plateau

**Expected Grokking Graph:**
```
Test Accuracy
    |            ___________  ← Grokking event!
    |           /
    |__________/
    |
    +---------------------------→ Epochs
       0      10k     20k?
```

**Professor's Results:**
- Training accuracy: 100% (memorization achieved)
- Test accuracy: ~15% (no generalization)
- No sudden jump observed

---

## 4. Gaussian Mixture Models (GMM)

### 4.1 Concept Overview

**K-Means Clustering (Review):**
- Hard assignments (point belongs to ONE cluster)
- Decision boundaries based on distance
- Linear boundaries between clusters

**Gaussian Mixture Models:**
- **Soft assignments** (probability of belonging to each cluster)
- Fits Gaussian (normal) distributions to clusters
- **Quadratic decision boundaries** (curved, not straight)
- More flexible than K-Means

**Key Difference:**
> "Gaussian mixture modeling defines a classification paradigm that is nonlinear as a result."

### 4.2 How GMM Works

**Mathematical Approach:**
1. Define K Gaussian distributions (one per cluster)
2. Each has mean (μ) and covariance (Σ)
3. Fit parameters to maximize likelihood
4. Decision boundary = where probabilities are equal

**Decision Boundary:**
- Not straight lines (unlike K-means)
- Ellipses or curves
- Defined by equal probability contours

**Example from Notebook:**
```python
from sklearn.mixture import GaussianMixture

# Fit GMM with 2 components
gmm = GaussianMixture(n_components=2)
gmm.fit(X_tsne)  # X_tsne = t-SNE projections

# Get cluster assignments
labels = gmm.predict(X_tsne)

# Get probabilities
probabilities = gmm.predict_proba(X_tsne)
```

**Output:**
- Cluster labels (0, 1, 2, ...)
- Probabilities for each cluster
- Can visualize with ellipses

### 4.3 GMM vs K-Means Comparison

| Feature | K-Means | GMM |
|---------|---------|-----|
| Assignment | Hard (0 or 1) | Soft (probabilities) |
| Boundaries | Linear | Quadratic/Curved |
| Shape | Circular clusters | Elliptical clusters |
| Output | Labels only | Labels + probabilities |
| Speed | Fast | Slower |
| Flexibility | Less | More |

**When to Use Which:**
- **K-Means:** Fast, clear spherical clusters, hard assignments needed
- **GMM:** Elliptical clusters, need probabilities, more complex shapes

---

## 5. K-Means Clustering Theory

### 5.1 Decision Boundary Combinatorics

**Question from Class:**
> "Is there an ordering to the decision boundaries?"

**Professor's Answer:**
- No inherent ordering
- Number of boundaries = combinatorics problem
- "Like shaking hands at a party"

**Formula:**
```
K clusters → C(K, 2) decision boundaries
C(K, 2) = K! / (2! × (K-2)!)
```

**Examples:**
- 2 clusters → 1 boundary
- 3 clusters → 3 boundaries (C(3,2) = 3)
- 5 clusters → 10 boundaries (C(5,2) = 10)

**Visualization:**
- Each pair of clusters needs one boundary
- Boundaries separate nearest neighbors
- Total = "choose 2 from K" combinations

### 5.2 K-Means Algorithm Steps

**Step 1:** Select K (number of clusters)
- Default in libraries: often 5
- User must choose based on problem
- No guarantee K is optimal

**Step 2:** Initialize cluster centers (seeds)
- Random initialization
- Or use heuristics (K-means++)
- Initial choice affects final result

**Step 3:** Assign points to nearest center
- Calculate distance to each center
- Assign to closest

**Step 4:** Update cluster centers
- Recalculate mean of points in each cluster
- Move center to new mean

**Step 5:** Repeat until convergence
- Iterate steps 3-4
- Stop when centers don't move significantly

**Limitations:**
- Requires knowing K
- Sensitive to initialization
- Assumes spherical clusters
- Hard assignments only

---

## 6. Linear Discriminant Analysis (LDA)

### 6.1 LDA vs PCA

**PCA (Review):**
- **Unsupervised** (no class labels needed)
- Aligns with variance in data
- May or may not align with class structure
- Linear transformation

**LDA:**
- **Supervised** (requires class labels!)
- Maximizes class separability
- Finds direction that best discriminates classes
- Feature engineering tool

**Key Insight:**
> "Linear discriminant analysis brings some supervision to that process and says, let me try and find a direction in my scatterplot... that will help you maximize separability of your classes."

### 6.2 How LDA Works

**Scenario:**
Imagine 2D data with two classes (red vs blue)

**PCA Would:**
1. Find direction of maximum variance (PC1)
2. Find orthogonal direction (PC2)
3. May or may not separate classes well

**LDA Does:**
1. Find direction that maximizes separation of class means
2. Consider within-class variance (scatter)
3. Consider between-class variance (separation)
4. Optimal: Large between-class, small within-class

**Mathematical Goal:**
```
Maximize: J(w) = ||μ₁ - μ₂||²
```
Where:
- μ₁ = mean of class 1 projected onto w
- μ₂ = mean of class 2 projected onto w
- w = projection direction (what we're solving for)

**Result:** Direction that best separates classes!

### 6.3 LDA as Feature Engineering

**Use Case:**
1. Start with high-dimensional data (e.g., 34 features)
2. Apply LDA to find 1-5 best discriminative directions
3. Project data onto these directions
4. Use projections as NEW features for classification
5. Train simpler model on reduced space

**Example from Notebook:**
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Fit LDA (supervised!)
lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)  # Note: y_train required!

# Apply to test data
X_test_lda = lda.transform(X_test)

# Train classifier on LDA features
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train_lda, y_train)

# Predict
y_pred = clf.predict(X_test_lda)
```

**Benefits:**
- Reduces dimensionality
- Improves class separability
- Speeds up downstream classification
- Can handle more than 2 classes!

**Note on Multi-Class:**
> "Logistic regression as a function generalizes to more than two classes, because we've talked about logistic regression and talked about a sigmoid activation, right? But we always talked about it in the context of binary classification. But this exercise shows you how logistic regression can be applied to more than two classes as well."

---

## 7. MNIST Dataset & Image Dimensionality Reduction

### 7.1 MNIST Dataset Introduction

**What is MNIST?**
- **M**odified **N**ational **I**nstitute of **S**tandards and **T**echnology
- Created by Yann LeCun and Corinna Cortez (1993)
- Standard benchmark for ML algorithms

**Dataset Details:**
- **Training:** 60,000 images
- **Testing:** 10,000 images
- **Size:** 28×28 pixels
- **Format:** Grayscale (0-255 intensity values)
- **Classes:** Digits 0-9 (10 classes)

**Historical Significance:**
> "Yann LeCun is the chief AI officer at Facebook. MNIST was created by Yann LeCun... and they made a model called LeNet-5, which is a convolutional neural network."

**Original Use:** Check digit recognition in banks!

### 7.2 Reshaping Images to Vectors

**Problem:** Images are 2D, but ML algorithms expect 1D vectors

**Solution:** Reshape (flatten) the image

**Math:**
```
28 × 28 = 784 dimensions
```

**Process:**
```python
# Original: 28×28 grid
[[p₁₁, p₁₂, ..., p₁₂₈],
 [p₂₁, p₂₂, ..., p₂₂₈],
 ...
 [p₂₈₁, p₂₈₂, ..., p₂₈₂₈]]

# Reshaped: 784×1 vector
[p₁₁, p₁₂, ..., p₁₂₈, p₂₁, p₂₂, ..., p₂₈₂₈]
```

**Illustration from Lecture:**
> "Take a 3×3 grid. Take first column, then second column and stack underneath, then third column. Result: 9×1 vector from 3×3 image."

**Key Point:** 
- As long as you do it consistently, the method doesn't matter
- All images must use same reshaping approach
- Each pixel becomes one feature

### 7.3 PCA on MNIST

**Dimensionality:**
- Start: 784 dimensions (one per pixel)
- PCA: Can get up to 784 principal components
- But we only use a sample (1,000 images)
- So we get 1,000 PCs of size 784 each

**Unusual Approach:**
> "Here, we're doing principal component analysis the other way around. We're saying, okay, if you have 1,000 observations, let's get 1,000 principal components, and each of those principal components are 784 in size."

**Visualizing PCs:**
- Each PC is 784 numbers
- Reshape back to 28×28 for visualization
- Shows patterns of variation

**Observation:**
- **PC0** (lowest): Looks like average digit shape (circular)
- **Higher PCs:** Show variations (vertical strokes, horizontal strokes, etc.)

**Magic of PCA:**
> "Some linear combination of these principal components will give you every digit. Alpha 1 times this plus alpha 2 times this plus alpha 3 times this... is going to give me a new picture that could be any digit."

### 7.4 Multiple Dimensionality Reduction Methods

**Methods Applied to MNIST:**

1. **PCA** (Principal Component Analysis)
   - Linear
   - Reproducible
   - Fast

2. **NMF** (Non-Negative Matrix Factorization)
   - Linear
   - All values positive
   - Similar to PCA but constrained

3. **MDS** (Multi-Dimensional Scaling)
   - Linear
   - Preserves pairwise distances

4. **Isomap**
   - Non-linear
   - Preserves geodesic distances
   - Manifold learning

5. **LLE** (Locally Linear Embedding)
   - Non-linear
   - Preserves local neighborhood structure

6. **t-SNE**
   - Non-linear
   - Best for visualization
   - Non-reproducible

**Comparison:**
> "Non-negative matrix factorization and PCA have a very similar effect... but they tend to not be very powerful in determining compressing the classification value of a multi-class problem... Whereas t-SNE, which is a nonlinear method, is quite good at it."

### 7.5 t-SNE Perplexity Parameter

**What is Perplexity?**
- Hyperparameter specific to t-SNE
- Roughly: "expected number of neighbors per point"
- Balances local vs global structure

**Effect of Perplexity:**
- **Low perplexity:** Many small, tight clusters
- **High perplexity:** Fewer, larger clusters
- Typical range: 5-50

**Professor's Explanation:**
> "Perplexity identifies how many observations form a cluster. The higher the perplexity, the fewer clusters you will find, and the more points you will have per cluster. The lower the perplexity, you will have smaller clusters and more of them."

**Visualization from Notebook:**
- Shows t-SNE with multiple perplexity values
- Can see digit clusters forming
- Different perplexities reveal different structures

**Limitation of t-SNE:**
> "You can't project a new observation into this space unless it was coupled with all the old observations and you make a new projection."

**Workaround:**
1. Use t-SNE to find structure
2. See if clusters exist
3. If yes, train supervised model on t-SNE labels
4. Use supervised model for new data

**Use Case:**
> "People will use t-SNE before they go down a real deep rabbit hole to try to classify some data to see, is it even possible to classify the data?"

---

## 8. Project Team Formation & Planning

### 8.1 Current Status

**From Class Roster Check:**
- 11 students confirmed in teams
- 18 total students in class
- **7 students still need teams!**

**Professor's Request:**
> "We need to make sure everybody is part of a team, okay? So by the time next Tuesday comes around, project summary needs to be starting to come together."

### 8.2 Assignment 4 Timeline

**Student Question (Michael):**
> "Is assignment 4 due before or after spring break?"

**Spring Break:** Starts March 7th (Saturday, next week from lecture)

**Professor's Decision:**
- **Formal submission:** After spring break (relaxed)
- **Progress checkpoint:** Before break (expected)
- **Goal:** Know what you're doing before break
- **Iteration:** Can refine after break

**Professor's Words:**
> "I would like for the project proposals to be in before then, and then you can iterate on it... Tuesday next week, which is the first week of March, we should have everything more or less buttoned down."

### 8.3 Project Requirements Reminder

**Team Structure:**
- 3 students per team (ideal)
- Max 4 students
- Everyone must be in a team

**Deliverables:**
1. **Project proposal** (Assignment 4)
2. **Dataset identified and accessible**
3. **Methods planned**
4. **Kanban board set up**

**Time Allocation Starting Next Week:**
> "Tuesday next week, we will spend like 30 minutes each class talking at least about projects."

### 8.4 Dataset Priority

**Critical Point:**
> "Form your teams and find your datasets!"

**Why Dataset First:**
- Can't do project without data
- Need to verify access
- Need to understand data structure
- Can pivot early if needed

---

## 9. Connection to Previous Concepts

### 9.1 Overfitting → Memorization → Grokking

**Week 4 (Overfitting):**
- Training accuracy high, test accuracy low
- Model too complex for data
- N < P problem

**Week 7 (Grokking):**
- Same phenomenon but extended
- Phase 1: Memorization (overfitting)
- Phase 2: Eventually generalizes (grokking)
- Shows learning can continue past overfitting!

### 9.2 Feature Engineering Tools Summary

**Learned So Far:**

1. **Windowed Features** (Week 2-3)
   - For time series signals
   - Frequency domain features
   - Prevented overfitting on EEG data

2. **PCA** (Week 5-6)
   - Unsupervised
   - Linear
   - Based on variance

3. **t-SNE** (Week 6)
   - Unsupervised
   - Non-linear
   - For visualization

4. **K-Means** (Week 6)
   - Unsupervised clustering
   - Hard assignments

5. **GMM** (Week 7)
   - Unsupervised clustering
   - Soft assignments
   - Probabilistic

6. **LDA** (Week 7)
   - Supervised
   - Linear
   - Maximizes class separability

**Choosing the Right Tool:**
- Have labels? → LDA or supervised learning
- No labels? → PCA, t-SNE, K-Means, GMM
- Need probabilities? → GMM
- Need reproducibility? → PCA, LDA (not t-SNE)
- Need nonlinear? → t-SNE, GMM
- Fast computation? → PCA, K-Means

---

## 10. Code Examples from Lecture

### 10.1 Loading MNIST Data

```python
from sklearn.datasets import fetch_openml

# Fetch MNIST from OpenML repository
mnist = fetch_openml('mnist_784', version=1, parser='auto')

# mnist_784 = 28×28 = 784 pixels
X = mnist.data  # Shape: (70000, 784)
y = mnist.target  # Shape: (70000,) - digit labels 0-9

# Sample 1000 random observations
import numpy as np
idx = np.random.choice(70000, 1000, replace=False)
X_sample = X[idx]
y_sample = y[idx]
```

### 10.2 Gaussian Mixture Model

```python
from sklearn.mixture import GaussianMixture

# After doing PCA or t-SNE
X_reduced = # your reduced dimensional data

# Fit GMM with K components
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_reduced)

# Get cluster assignments
labels = gmm.predict(X_reduced)

# Get probabilities
probs = gmm.predict_proba(X_reduced)
# probs[i,j] = probability that point i belongs to cluster j
```

### 10.3 Linear Discriminant Analysis

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit LDA (supervised - needs y!)
lda = LDA(n_components=1)  # Reduce to 1 dimension
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Train classifier on LDA features
clf = LogisticRegression()
clf.fit(X_train_lda, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_lda)

# Compare to baseline (no LDA)
clf_baseline = LogisticRegression()
clf_baseline.fit(X_train, y_train)
y_pred_baseline = clf_baseline.predict(X_test)

# LDA version is typically faster and often more accurate!
```

### 10.4 Visualizing t-SNE with Different Perplexities

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Try multiple perplexity values
perplexities = [5, 10, 30, 50]

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

for idx, perplexity in enumerate(perplexities):
    # Fit t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_sample)
    
    # Plot
    ax = axes[idx // 2, idx % 2]
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                        c=y_sample, cmap='tab10', alpha=0.7)
    ax.set_title(f'Perplexity = {perplexity}')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()
```

---

## 11. Helper Functions from Notebooks

### 11.1 Plot MNIST Digits

```python
def plot_digits(data, n_rows=5, n_cols=10):
    """
    Plot a grid of MNIST digits
    data: array of shape (n_samples, 784)
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
    
    for i, ax in enumerate(axes.flat):
        if i < len(data):
            # Reshape 784 → 28×28
            img = data[i].reshape(28, 28)
            ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
```

### 11.2 Plot Explained Variance

```python
def plot_explained_variance(pca):
    """
    Plot cumulative explained variance for PCA
    """
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumsum) + 1), cumsum, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    plt.legend()
    plt.show()
    
    # Find number of components for 95% variance
    n_components_95 = np.argmax(cumsum >= 0.95) + 1
    print(f"Components needed for 95% variance: {n_components_95}")
```

---

## 🔑 Key Takeaways

### 1. Understanding Grokking
- ✅ Grokking = delayed generalization after memorization
- ✅ Demonstrates that learning can continue past initial overfitting
- ✅ Requires extreme patience (10,000+ epochs)
- ✅ Rare phenomenon but important for understanding neural network learning
- ✅ Small models (N > P) help but don't guarantee quick grokking

### 2. Transformer Architecture Basics
- ✅ Foundation of modern AI (GPT, BERT, ChatGPT)
- ✅ "Attention is All You Need" paper (2017) - seminal work
- ✅ Key components: Multi-head attention, feed-forward networks, add & normalize
- ✅ Can learn complex operations like addition (generatively, not just classification)
- ✅ Training requires: Learning rate, batch size, optimizer (Adam), LR scheduling

### 3. Dimensionality Reduction Toolkit Expanded

**Unsupervised Methods:**
- ✅ **PCA:** Linear, reproducible, fast, variance-based
- ✅ **t-SNE:** Nonlinear, non-reproducible, best for visualization, perplexity parameter
- ✅ **NMF:** Linear, non-negative values
- ✅ **Isomap/LLE:** Nonlinear manifold learning

**Supervised Methods:**
- ✅ **LDA:** Maximizes class separability, requires labels, feature engineering tool

### 4. Clustering Methods

**K-Means:**
- Hard assignments (0 or 1)
- Spherical clusters assumed
- Fast but less flexible
- C(K,2) decision boundaries

**Gaussian Mixture Models (GMM):**
- Soft assignments (probabilities)
- Elliptical clusters
- Quadratic decision boundaries
- More flexible, outputs probabilities

### 5. MNIST Dataset
- ✅ 28×28 pixel handwritten digits → 784 dimensions
- ✅ Reshape 2D images to 1D vectors for ML algorithms
- ✅ Can apply all dimensionality reduction methods
- ✅ PCA components visualize as "eigen-digits"
- ✅ Standard benchmark dataset created by Yann LeCun (1993)

### 6. Choosing the Right Method

**Decision Tree:**
```
Have class labels?
├─ YES → LDA (supervised dimensionality reduction)
└─ NO
   ├─ Need reproducibility?
   │  ├─ YES → PCA, K-Means
   │  └─ NO → t-SNE (visualization only)
   ├─ Need probabilities?
   │  └─ YES → GMM
   └─ Need nonlinear?
      └─ YES → t-SNE, GMM, Isomap, LLE
```

### 7. Project Planning Urgency
- ✅ Form teams NOW (7 students still unassigned!)
- ✅ Dataset identification is THE priority
- ✅ Assignment 4 due after spring break (flexible formal deadline)
- ✅ But proposals should be ready BEFORE break
- ✅ 30 minutes per class dedicated to projects starting next week

### 8. Connection to Course Themes

**Overfitting Arc:**
- Week 2-3: EEG amplitude overfits, frequency features generalize
- Week 4: N > P rule, variance pizza, regularization
- Week 7: Grokking shows memorization → eventual generalization

**Feature Engineering Arc:**
- Week 2-3: Windowed frequency features
- Week 5-6: PCA, K-Means clustering
- Week 7: GMM, LDA, multiple methods comparison

---

## 📚 Resources from Lecture

### Blog Posts
- [Professor Menon's Lab Blog](https://blog.themenonlab.com)
- Search for "Grokking" to find all three posts
- Vision Transformers for MNIST
- Replicating 777-parameter addition model
- Understanding grokking factors

### Key Paper
- "Attention is All You Need" (2017) - Transformer architecture foundation
- Available on arXiv
- Basis for GPT, BERT, and all modern LLMs

### Notebooks to Run
- `DimRed--NonLinear--GMM--DecisionTrees.ipynb` - GMM examples
- `2d-DimRed--MNIST01.ipynb` - MNIST dimensionality reduction
- `LDA.ipynb` - Linear Discriminant Analysis examples

### PDFs Available
- `GMMvsKmeans.pdf` - Comparison of clustering methods
- `KMeansClustering.pdf` - K-Means algorithm details
- `LDA.pdf` - Linear Discriminant Analysis theory

---

## 🎯 Action Items for Students

### Immediate (Before Next Class)
- [ ] Form project teams or join existing team
- [ ] Update class roster (Column G to 'Y')
- [ ] Identify potential datasets
- [ ] Review Quiz 1 solutions
- [ ] Run dimensionality reduction notebooks

### This Week
- [ ] Finalize team formation
- [ ] Select and verify dataset access
- [ ] Draft project proposal outline
- [ ] Set up Kanban board for team
- [ ] Run LDA and GMM notebooks

### Before Spring Break (March 7)
- [ ] Complete Assignment 4 draft (project proposal)
- [ ] Verify dataset is accessible and usable
- [ ] Outline project methods and timeline
- [ ] Identify any potential roadblocks

---

## 🙋 Questions from Class

**Q: Is the quiz out of 100 or 90 points?**  
**A:** Base points = 90 (6 sections × 2 questions). Maximum with all bonuses ≈ 150. Canvas will be updated to show 90.

**Q: Is there ordering to decision boundaries in K-means?**  
**A:** No inherent ordering. Number of boundaries = C(K, 2) = combinatorics (like handshakes at a party).

**Q: Is Assignment 4 due before or after spring break?**  
**A:** Formal submission after break is fine, but proposals should be substantially ready BEFORE break so teams know their direction.

**Q: Can PCA on MNIST give more than 784 components?**  
**A:** No, maximum = min(n_samples, n_features). With 1,000 samples and 784 features, we get 784 components max. When using 1,000 samples, we get 1,000 components of size 784 (transposed approach).

**Q: Why can't we apply t-SNE to new data like PCA?**  
**A:** t-SNE has random initialization and no .transform() method. Must fit all data together. Workaround: Use t-SNE to verify clusters exist, then train supervised model on t-SNE labels.

---

## 💻 Additional Code Snippets

### Comparing All Dimensionality Reduction Methods

```python
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding as LLE, MDS
import matplotlib.pyplot as plt

# Assume X_sample is your data (e.g., 1000 MNIST images)

methods = {
    'PCA': PCA(n_components=2),
    'NMF': NMF(n_components=2, init='random', random_state=42),
    'MDS': MDS(n_components=2, random_state=42),
    'Isomap': Isomap(n_components=2),
    'LLE': LLE(n_components=2, random_state=42),
    't-SNE': TSNE(n_components=2, random_state=42, perplexity=30)
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, (name, method) in enumerate(methods.items()):
    ax = axes[idx // 3, idx % 3]
    
    # Transform data
    if name == 't-SNE':
        X_transformed = method.fit_transform(X_sample)
    else:
        X_transformed = method.fit_transform(X_sample)
    
    # Plot
    scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
                        c=y_sample, cmap='tab10', alpha=0.7)
    ax.set_title(f'{name}')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

plt.tight_layout()
plt.show()
```

### Training Supervised Classifier on t-SNE Embeddings

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE

# Step 1: Use t-SNE to find structure (on training data)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_train_tsne = tsne.fit_transform(X_train)

# Step 2: Train decision tree on t-SNE coordinates
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train_tsne, y_train)

# Step 3: For test data - CANNOT use t-SNE directly!
# Workaround: Train classifier on ORIGINAL features
dt_original = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_original.fit(X_train, y_train)  # Train on original X
y_pred = dt_original.predict(X_test)  # Predict on original X_test

# The t-SNE was just for VISUALIZATION to verify clusters exist!
```

---

## 📖 Next Class Preview

**Topics for Next Tuesday (March 4):**
- 30 minutes dedicated to project discussions
- Team presentations of dataset/proposal ideas
- Advanced topics continuation
- Possible: Support Vector Machines (SVM)
- Possible: Decision Trees and ensemble methods

**Before Next Class:**
- ✅ Be in a team!
- ✅ Have a dataset identified!
- ✅ Draft project proposal outline

**Remember:**
> "Form your teams and find your datasets!" - Professor Menon

---

**Keep exploring grokking if you're curious - it's a fascinating phenomenon!** 🔬🚀
