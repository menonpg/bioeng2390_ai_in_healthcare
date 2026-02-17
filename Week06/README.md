# Week 06: PCA Applications, Clustering & Project Planning
### BIOENG 2390: AI in Healthcare - Spring 2026

**Instructor:** Professor Prahlad G. Menon, PhD, PMP  
**University of Pittsburgh, Department of Bioengineering**

---

## 🎯 Week 06 Overview

This week we apply PCA to real neural data, introduce K-means clustering, and start planning final projects. We'll see how to save/load models with pickle, understand the difference between PCA (linear/reproducible) and t-SNE (non-linear/non-reproducible), and learn about agile project management with Kanban Flow.

---

## 📺 Lecture Recordings & Notes

### Lecture 11 - February 17, 2026 (65 minutes)
**Focus:** Projects, Datasets & Practical PCA/K-Means

- **[Watch Recording](https://fathom.video/share/GHbQG-wscK9FsAauEVrKcKscHDHp5ryj)**
- **[Read Detailed Lecture Notes](Lecture11_Notes_Feb17_2026.md)** ← Complete transcript

**Topics Covered:**
- Final project planning and team formation (teams of 3, max 4)
- Kanban Flow for agile project management
- Dataset resources: HuggingFace, PhysioNet, Kaggle, Google Dataset Search, Stanford AIMI
- Neural spike data analysis (fMRI/EEG context)
- Z-score normalization per time point
- Practical PCA with sklearn (34D → 5D → 2D)
- K-means clustering in PC space
- Saving models with pickle (PCA, K-means, normalization params)
- t-SNE introduction (non-linear, non-reproducible)
- Applying models trained on 50 to all 3,636 observations

## 1. Administrative Updates
- Currently in Week 6 of the course
- Assignments status:
  - Assignments 1 & 2 should be completed
  - Assignment 3 & 4 relate to dimensionality reduction
  - Assignment 4 is the project proposal due first week of March

### Project Requirements
- Teams of up to 3 members
- Focus areas:
  - Time series data
  - Images
  - Signal data (2D, 3D, N-dimensional)
- Methods should include:
  - Unsupervised learning
  - Supervised learning
  - Dimensionality reduction
  - Feature engineering

## 2. Principal Component Analysis (PCA)

### 2.1 Core Concepts
1. **Purpose**:
   - Reduce dimensionality while preserving variation
   - Identify key directions of variation
   - Create features aligned with variation directions

2. **Mathematical Foundation**:
   - Eigenvectors and eigenvalues
   - Covariance matrix analysis
   - Linear transformation from high to low dimensional space

### 2.2 PCA Process
1. **Data Preparation**:
   - Center data by subtracting mean
   - Standardization recommended before PCA
   - Create covariance matrix

2. **Eigenvalue Decomposition**:
   - Calculate eigenvalues and eigenvectors
   - Sort eigenvalues in decreasing order
   - Each eigenvector represents a direction in data

3. **Variance Explanation**:
   - Percentage variance explained = \[\frac{\lambda_i}{\sum \lambda_j}\]
   - Higher eigenvalues indicate more important principal components

### 2.3 Key Points About PCA
- Linear and reversible dimensionality reduction
- Number of principal components equals number of features
- Units and scale affect PCA results
- Standardization is important before PCA

## 3. Types of Data Analysis

### 3.1 Cross-sectional Data
- No time resolution
- Each observation is a single point
- Features are individual values

### 3.2 Longitudinal Data
- Time resolution exists
- Features are vectors over time
- More complex dimensional structure

## 4. Practical Example Discussed
- 2×2 matrix example: \[A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}\]
- Eigenvalues: λ₁ = 3, λ₂ = 1
- Principal components:
  - PC0: [1, 1] (explains 75% variance)
  - PC1: [1, -1] (explains 25% variance)

## 5. Important Considerations
- Normalization is crucial before PCA
- More observations can improve clustering results
- PCA depends on:
  - Units of measurement
  - Original range of values
  - Scale of variables

## Next Lecture Preview
- Application of eigenvalues to base classification
- Linear vs nonlinear clustering methods
- Reversibility concepts



# Week 6 Lecture 12: Nonlinear Dimensionality Reduction

## Key Topics
1. Review of PCA and Introduction to Nonlinear Methods
2. TSNE (t-Distributed Stochastic Neighbor Embedding)
3. Practical Implementation and Comparison

## 1. Principal Component Analysis (PCA) Review

### Limitations of PCA
- PCA discovers intrinsic variance in data, not necessarily structure
- Structure may not always be related to variance
- Features used for PCA may not be good descriptors of the underlying structure

### Types of Dimensionality Reduction

#### By Supervision:
- **Supervised**:
  - Fischer Linear Discriminant Analysis (LDA)
  - Neural network techniques (e.g., variational autoencoders)
- **Unsupervised**:
  - PCA
  - Independent Component Analysis
  - TSNE
  - ISOMAP

#### By Linearity:
- **Linear Methods**:
  - Reversible transformations
  - Can map between physical and reduced domains
  - Useful for feature engineering in machine learning pipelines
- **Nonlinear Methods**:
  - Not reversible
  - Better for visualization and understanding data structure
  - Examples: TSNE, ISOMAP

## 2. TSNE (t-Distributed Stochastic Neighbor Embedding)

### Key Characteristics
- Aims to solve PCA's limitations with nonlinear scaling
- Offers optimal separation in reduced dimensions
- Uses pairwise distances between points
- Includes hyperparameter "perplexity" (expected number of neighbors)

### How TSNE Works
1. Measures pairwise distances between points
2. Uses perplexity to scale distance matrix
3. Randomly scatters points and iteratively adjusts positions
4. Optimizes point distances to match similarity matrix

### TSNE vs PCA
- TSNE axes have no physical meaning
- Distances are not meaningful in physical units
- Focus is on structure visualization
- Results may vary between runs due to non-deterministic nature

## 3. Practical Implementation

### Workflow Demonstrated
1. Apply dimensionality reduction (PCA/TSNE)
2. Cluster in reduced space
3. Learn cluster assignments
4. Apply to new data

### Learning from TSNE Results
- Create supervised learning dataset from TSNE labels
- Train decision tree classifier
- Apply to full dataset
- Visualize results with original data

### Code Implementation Notes
- Used Python libraries: sklearn, pandas
- Demonstrated normalization and unnormalization
- Visualization of clustering results
- Decision tree interpretation of learned rules

## Important Notes for Students
1. Assignments 1, 2, and 3 are due
2. Assignment 4: Prepare dataset discussion for Tuesday
3. Project proposals due first week of March
4. No strict deadlines but recommended to complete assignments when relevant to current topics

## Technical Requirements
- Required Python libraries:
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - DtreeViz (for decision tree visualization)

This lecture effectively bridged the gap between linear and nonlinear dimensionality reduction techniques, providing both theoretical understanding and practical implementation guidance.