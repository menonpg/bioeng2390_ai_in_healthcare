# BIOENG-2390 Spring 2026 - Lecture 10
## February 12, 2026

**Instructor:** Professor Prahlad G. Menon, PhD, PMP  
**Recording:** [View on Fathom](https://fathom.video/share/azJWBGcZC6A8WNeY6pxuTwc7J5sw5saD)  
**Duration:** 94 minutes

---

## 📋 Lecture Overview

Today's lecture was highly conceptual with extensive whiteboarding on Principal Component Analysis (PCA). Topics covered:

1. Structure vs variance in data (critical distinction!)
2. Why dimensionality reduction is necessary
3. Gene expression example (cells × genes matrix)
4. Visualization limits (can't visualize >3 dimensions)
5. Combinatorial explosion of 3D plots
6. PCA mathematical framework  
7. Eigenvectors and eigenvalues intuition
8. Basis sets and orthonormal vectors
9. Projection concept (dot product)
10. Information loss and explained variance
11. PCA with sklearn demonstration

**Supplementary Material:** `PCAbasics.pdf` added to Week05 folder

---

## 🎯 Critical Concept: Structure vs Variance

### The Distinction

**Variance:**
- How data points are spread/distributed
- Measured numerically (std dev, covariance)
- **Always exists** for continuous variables
- Visual: Shape of scatter plot

**Structure:**
- Class labels (red vs green, seizure vs normal)
- What we want to predict
- **May not be known!** (unlabeled data)
- Visual: Colors of points

**From Class:**

**Professor's Question:** "Does this unlabeled scatter plot have structure?"

**Answer:** NO! Without labels, no structure.

**But:** "Does it have variance?"

**Answer:** YES! Can calculate spread numerically.

**Critical Insight:**
> "The structure is the colors of the dots. The colors indicate structure. The labels of the data indicate structure. The shape of the scatterplot is the variance."

---

## 🧬 Gene Expression Example

### The Setup

**Data Matrix:**
```
         Gene1  Gene2  Gene3  ...  Gene400
Cell1     2.3    5.1    1.2  ...    4.5
Cell2     2.1    5.3    1.1  ...    4.7
Cell3     8.9    2.1    9.3  ...    1.2  ← Different cluster?
...
CellM     2.2    5.2    1.3  ...    4.6
```

**Dimensions:**
- M cells (observations/rows)
- N genes (features/columns)
- Example: M=100 cells, N=400 genes

**Goal:** Identify which genes separate cancer vs normal cells

**Problem:** Can't visualize 400-dimensional space!

---

### Why We Can't Just Make 3D Plots

**Idea:** Visualize triplets of genes in 3D

**How many triplets?**
```
nCk = n! / (k! × (n-k)!)

For 4 genes, pick 3:
4C3 = 4!/(3!×1!) = 4 plots (manageable)

For 400 genes, pick 3:
400C3 = 400!/(3!×397!)
      = (400 × 399 × 398)/(3 × 2 × 1)
      = 10,586,800 plots!
```

**From Lecture:**
> "You could be making three-dimensional plots and looking at them for the rest of your life! I'd rather be on the beach, go jet skiing. I don't want to spend my life looking at 3D plots."

**Solution:** Dimensionality reduction!

---

## 🎯 Use Cases for Unsupervised Learning

### 1. Unlabeled Data

**When you receive data without labels:**
- Don't know which cells are cancerous
- Don't know which neural spikes correspond to activities
- Can still discover natural clusters!

### 2. Rare Events

**Example: Bridge Collapse Prediction**

**Problem:**
- Only 1 bridge collapse event (not enough for supervised learning!)
- Years of normal operation data
- Need to define "normal" vs "anomaly"

**Solution:**
- Cluster normal operation patterns (unsupervised)
- Define boundary around normal cluster
- Points outside boundary = anomalies = potential collapse risk

**Professor's Pittsburgh Example:**
> "City of Pittsburgh hired to predict bridge collapses. Only one observation of collapse (near Stanley Park). Use unsupervised clustering to define 'normal operation' from years of seismograph data. Outliers = anomaly detection."

### 3. Feature Selection

**When engineering many features:**
- Don't know which features useful
- Not enough labeled data yet
- Use unsupervised clustering to test feature combinations
- Features that create clear clusters = good candidates!

---

## 📐 PCA Fundamentals

### The Goal

**Start:** N-dimensional feature space (e.g., 400 genes)
**End:** K-dimensional space (e.g., K=2 for visualization)
**Where:** K << N (much smaller)

**Mathematical Expression:**
```
X (M × N) → Transform → Y (M × K)

Where:
- M = number of observations (cells)
- N = number of features (genes)
- K = reduced dimensions (2 or 3)
```

---

### The Two Steps

**Step 1: Find the Magic Plane**
- Identify principal directions of variation
- PC0 = direction of maximum variance
- PC1 = direction of second-most variance (⊥ to PC0)
- These define a plane in high-D space

**Step 2: Project Points onto Plane**
- Flatten 3D → 2D (or 400D → 2D!)
- Use dot product (projection)
- Get coordinates in PC space

---

### Visual Intuition (From Whiteboard)

**3D Data (F1, F2, F3):**
```
      F3
       ^
       |    •••
       |  •••  ← Data spreads diagonally
       | •••
       |_______> F2
      /
    F1

Principal directions:
PC0: Diagonal (main spread direction)
PC1: ⊥ to PC0 (secondary spread)
```

**Define Plane:**
- PC0 and PC1 are two lines
- Two lines always lie in a plane (geometry!)
- This plane = "magic viewing angle"

**Project Points:**
```
3D Point: (F1, F2, F3)
         ↓ projection
2D Point: (PC0, PC1)
```

**Key Insight:**
> "Two lines must lie on a plane. If you have two points, they can be connected by a line. If you have two lines, they must lie on a plane."

---

## 🔬 PCA Mathematical Framework

### The Steps (from PCAbasics.pdf)

**Step 1: Represent Data**
```
Have M vectors (cells)
Each vector is N×1 (gene expressions)
X₁, X₂, ..., Xₘ
```

**Step 2: Compute Mean**
```
X̄ = (1/M) Σ Xᵢ
```
Average gene expression across all cells

**Step 3: Zero-Center (Mean Subtraction)**
```
Φᵢ = Xᵢ - X̄
```
Subtract mean from each cell vector

**Step 4: Form Matrix A**
```
A = [Φ₁  Φ₂  ...  Φₘ]
Size: N × M
```

**Step 5: Compute Covariance Matrix**
```
C = (1/M) × A × Aᵀ
Size: N × N
```

**Step 6: Eigendecomposition**
```
Find eigenvalues and eigenvectors of C
Sort by eigenvalue magnitude
```

**Step 7: Select K Components**
```
Pick top K eigenvectors (largest eigenvalues)
These are your principal components!
```

**Step 8: Project Data**
```
For each observation Xᵢ:
Yᵢ = Uₖᵀ × (Xᵢ - X̄)

Where Uₖ = [PC₀  PC₁  ...  PCₖ₋₁]
```

---

## 🎓 Eigenvectors and Eigenvalues Explained

### What are Eigenvectors?

**Definition (from lecture):**
> "Eigenvectors are a set of orthogonal vectors that define a space completely."

**3D Example:**
```
i-hat = [1, 0, 0] → X-direction
j-hat = [0, 1, 0] → Y-direction  
k-hat = [0, 0, 1] → Z-direction
```

**Properties:**
- **Orthogonal:** At right angles to each other
- **Unit length:** Magnitude = 1 (orthonormal)
- **Basis set:** Any point can be expressed as combination

**Any 3D point:**
```
Point = (x₁, y₁, z₁)
      = x₁·[1,0,0] + y₁·[0,1,0] + z₁·[0,0,1]
      = x₁·i + y₁·j + z₁·k
```

### Connection to PCA

**In PCA:**
- Eigenvectors = principal component directions
- Eigenvalues = amount of variance along each PC
- **Largest eigenvalue** → PC0 (most variance)
- **Second largest** → PC1 (second-most variance)

**Selecting K Components:**
- Sort eigenvectors by eigenvalue
- Pick top K
- These capture most variance!

---

### Google Search Connection

**Surprising Application:**

**Professor mentioned:**
> "Google search was originally based on this! Instead of cells, you have websites. The eigenvectors corresponding to the top eigenvalues identify the most important websites - the most well-connected nodes."

**How:**
- Websites = observations
- Links = features
- Eigenvectors of link matrix = PageRank!
- Top eigenvalues = most important pages

---

## 💡 Information Loss and Explained Variance

### The Trade-Off

**Reducing N → K dimensions:**
- ✅ Can visualize (K=2 or K=3)
- ✅ Remove noise
- ✅ Reduce overfitting
- ❌ Lose information!

**What's Lost:**
- Variance in excluded principal components
- Separability perpendicular to chosen plane
- Example: Points close in PC0-PC1 plane might be far apart along PC2

**From Lecture:**
> "Two points that are close in the projected 2D space might actually be separated in the 3D space along an axis normal to the plane. You're going to lose that information by virtue of this flattening exercise."

---

### How Much Information is Lost?

**Explained Variance Ratio:**
```
Information Retained = Σ(selected eigenvalues) / Σ(all eigenvalues)
```

**Example:**
```
All eigenvalues: [50, 30, 10, 5, 3, 2] (N=6)
Total: 100

Select top 2: [50, 30]
Explained variance = (50+30)/100 = 80%
Information retained: 80%
Information lost: 20%
```

**Rule of Thumb:**
- Aim for 90-95% explained variance
- Don't reduce so much you lose critical information
- Trade visualization benefit vs information loss

**Applications:**
- **Face recognition:** Don't reduce so much you lose face features
- **Gene expression:** Keep enough PCs to preserve cell type differences

---

## 🔑 Key Terminology

**Transformation Matrix T:**
- Size: K × N (or N × K depending on formulation)
- Contains K eigenvectors (principal components)
- Maps high-D → low-D

**Projection:**
- Mathematical operation: dot product
- Takes point in N-D space
- Flattens to K-D plane
- Results in K coordinates

**Covariance Matrix:**
- Size: N × N
- Captures variance and correlations between features
- Eigendecomposition reveals principal directions

**Basis Set:**
- Collection of orthogonal unit vectors
- Defines coordinate system
- Original: F1, F2, F3
- New (PCA): PC0, PC1, PC2

---

## 💻 PCA with sklearn (From Notebook)

### Implementation

```python
from sklearn.decomposition import PCA
import numpy as np

# Assuming X is M×N matrix (cells × genes)

# Create PCA object (reduce to 2 dimensions)
pca = PCA(n_components=2)

# Fit and transform
X_pca = pca.fit_transform(X)

# X_pca is now M×2 (cells in PC space)

# Check explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", sum(pca.explained_variance_ratio_))

# Get principal components (eigenvectors)
components = pca.components_  # 2×N matrix

# Get eigenvalues
eigenvalues = pca.explained_variance_
```

### Visualize in PC Space

```python
import matplotlib.pyplot as plt

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC0 (First Principal Component)')
plt.ylabel('PC1 (Second Principal Component)')
plt.title('Data in Principal Component Space')
plt.show()
```

### Choosing Number of Components

```python
# Try different values of K
for k in [2, 3, 5, 10]:
    pca = PCA(n_components=k)
    pca.fit(X)
    print(f"K={k}: {sum(pca.explained_variance_ratio_):.2%} variance explained")
```

---

## 🎓 Important Insights from Lecture

### 1. Matrix Dimensions in PCA

**Question from Class:** Determining transformation matrix size

**Challenge:**
```
X (M×N) × T (?, ?) = Y (M×K)
```

**Matrix multiplication rule:**
- Inner dimensions must match!
- Columns of first = Rows of second

**Solution involves transpose:**
```
Xᵀ (N×M) × T (N×K) → Yᵀ (M×K)
Then transpose result
```

**Key:** Must respect matrix multiplication rules

### 2. Cells vs Genes Orientation

**Question:** "Wouldn't X₁ be the column?"

**Answer:** NO (in our formulation)

**Our Setup:**
- Rows = Cells (observations)
- Columns = Genes (features)
- X₁ = First cell (row vector of gene expressions)
- Could transpose, but changes problem!

**Why This Way:**
- Goal: Cluster CELLS (not genes)
- Using genes as FEATURES
- Standard ML convention: observations in rows

### 3. Information Always Lost

**Unavoidable:**
- Reducing 400D → 2D loses information
- Points that were far apart might appear close
- Perpendicular separability lost

**Acceptable:**
- If explained variance > 90-95%
- Trade visualization for slight information loss
- Often noise is what's lost (good!)

---

## 📚 Real-World Applications

### 1. Face Recognition
- Each pixel = feature (thousands!)
- Reduce to "eigenfaces" (PCs)
- Keep enough to preserve identity
- Used in early face recognition systems

### 2. Gene Expression Analysis
- Thousands of genes measured
- Reduce to 2-3 PCs for visualization
- Identify gene "signatures" for diseases
- Feature selection for clinical tests

### 3. Anomaly Detection (Bridge Example)
- Define normal operation cluster
- Set boundary (e.g., 3σ from center)
- Points outside = anomalies
- **Rare event prediction without failure examples!**

### 4. Feature Selection
- Engineer many features
- Use PCA to see which create separability
- Original features aligned with top PCs = important
- Guided feature selection

---

## 🔑 Key Equations

### Covariance Matrix
```
C = (1/M) × A × Aᵀ

Where A = [Φ₁  Φ₂  ...  Φₘ]
And Φᵢ = Xᵢ - X̄ (mean-subtracted)
```

### Eigendecomposition
```
C × v = λ × v

Where:
- v = eigenvector (principal component direction)
- λ = eigenvalue (variance along that direction)
```

### Projection
```
Yᵢ = Uₖᵀ × (Xᵢ - X̄)

Where:
- Uₖ = matrix of K selected eigenvectors
- Yᵢ = coordinates in K-dimensional PC space
```

### Explained Variance
```
Explained Variance = Σ(selected λ) / Σ(all λ)

Typically aim for > 0.90 (90%)
```

---

## 🎬 For Next Class (Thursday Quiz, then Next Tuesday)

### Immediate:

**Quiz 1 - Thursday, February 19:**
- Open book, open notes
- Covers fundamentals through dimensionality reduction
- Focus on concepts, not memorization

### Upcoming Topics:

1. **More PCA:**
   - Scree plots
   - Choosing optimal K
   - Interpreting principal components
   - PCA vs LDA

2. **Non-Linear Dimensionality Reduction:**
   - t-SNE
   - UMAP  
   - When linear (PCA) isn't enough

3. **Clustering:**
   - K-means implementation
   - Choosing K
   - Evaluating clusters

### Homework:

- [ ] Run `Dimensionality Reduction for Time Series Classification.ipynb`
- [ ] Review PCAbasics.pdf
- [ ] Understand structure vs variance distinction
- [ ] Complete Assignments 0, 1, 2 before quiz
- [ ] Review eigenvectors/eigenvalues (Gilbert Strang book/videos)
- [ ] Prepare for Quiz 1 (Feb 19)

---

## 🙋 Questions from Class

**Q: What is structure in data?**  
**A:** Class labels (colors, seizure/normal, cancer/not). Without labels, no structure!

**Q: Can we do supervised learning without structure?**  
**A:** NO! Need labels (Y) to learn X → Y mapping. Without Y, use unsupervised.

**Q: How to visualize 4+ dimensions?**  
**A:** Can't directly. Either:
1. Make many 3D plots (combinatorial explosion!)
2. Use PCA to reduce to 2-3D (smart solution)

**Q: Is X₁ a row or column?**  
**A:** In our formulation, X₁ = first cell = row of gene values. Could transpose, but changes problem from clustering cells to clustering genes.

**Q: How does PCA know which directions are important?**  
**A:** Eigenvalues! Larger eigenvalue = more variance in that direction = more important.

**Q: Why orthogonal (perpendicular) components?**  
**A:** Eliminates redundancy. Each PC captures DIFFERENT variance (like pizza slices from Week04!).

---

## 📋 Week 05 Thursday Checklist

- [ ] Understand structure vs variance distinction
- [ ] Know why we need dimensionality reduction
- [ ] Understand PCA finds principal variation directions
- [ ] Understand eigenvectors = basis set
- [ ] Understand eigenvalues = variance magnitude
- [ ] Know projection = flattening to lower-D plane
- [ ] Understand information is lost (explained variance)
- [ ] Can run sklearn PCA
- [ ] Prepare for Quiz 1 (Feb 19)

---

## 🎓 Professor's Final Thoughts

**On Conceptual Understanding:**
> "Dimensionality reduction can feel like magic. When I first learned it, I really struggled. With deliberate understanding of fundamentals, we'll all be in good shape."

**On Matrix Confusion:**
> "This is where the math gets crazy, guys. The inner dimensions must match for matrix multiplication. That's the trick."

**On Eigenvalues:**
> "Eigenvectors corresponding to the top eigenvalues are most aligned with your most well-connected nodes. This underpins Google search!"

**On Assignments:**
> "Try to keep pace. The assignments are often just 'run this notebook and write a short report.' Don't delay - it gets harder to catch up when you're trying to remember what we did weeks ago."

**For Next Class:**
> "We'll look at the math more deliberately. How to programmatically do PCA. How to interpret results. Then we'll prepare for the quiz."

**Professor Prahlad Menon, PhD, PMP**  
*Office Hours: By appointment*  
*Email: prm44@pitt.edu*

---

*"I'd rather be on the beach, go jet skiing, than spend the rest of my life looking at three-dimensional plots."*

— On why we need automated dimensionality reduction

---

## 🔑 Key Takeaways

1. ✅ Structure = labels, Variance = spread (CRITICAL distinction!)
2. ✅ Unsupervised learning discovers structure from variance alone
3. ✅ Can't visualize >3D (combinatorial explosion of plots)
4. ✅ PCA finds "magic plane" of maximum variance
5. ✅ Eigenvectors = principal component directions
6. ✅ Eigenvalues = variance magnitude along each PC
7. ✅ Top K eigenvectors = keep most information
8. ✅ Projection = flattening high-D → low-D (dot product)
9. ✅ Information loss unavoidable but measurable (explained variance)
10. ✅ Applications: gene expression, face recognition, anomaly detection, feature selection

**Next:** Quiz preparation, then deeper PCA implementation!

**Study hard for Quiz 1! See you Thursday!** 📚🚀
