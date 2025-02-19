# BIOENG 2390: AI in Healthcare

**University of Pittsburgh, Department of Bioengineering**

## Course Overview
Welcome to BIOENG 2390: AI in Healthcare. 

This course provides an introduction to artificial intelligence (AI) and machine learning (ML) in healthcare. The course covers fundamental concepts in AI/ML, data preprocessing, model development, evaluation, and deployment. The course also includes hands-on labs and assignments to apply these concepts to healthcare datasets.


# Week 6 Lecture Notes, Lecture 11 
## Lecture Notes: Principal Component Analysis and Project Discussion
Date: February 18, 2025

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