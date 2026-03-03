# BIOENG 2390: AI in Healthcare

**University of Pittsburgh, Department of Bioengineering**

## Course Overview
Welcome to BIOENG 2390: AI in Healthcare. This course includes comprehensive introduction to programming in Python, R, and MATLAB, as well as setting up development environments and using Integrated Development Environments (IDEs) to develop linear, nonlinear, and deep learning models for healthcare applications, starting with time series data, image data, and text data.


# Week 8  Lecture Notes, Lecture 15 [Lecture 16: PCA Quiz + Completion of Assignment-4 pre-submission project review]
Here are the key notes from Week 8's lecture on Linear Discriminant Analysis (LDA) and dimensionality reduction:

## Linear Discriminant Analysis (LDA)

### Key Concepts
1. LDA is a supervised dimensionality reduction technique that:
   - Finds optimal orientations/dimensions that best separate different classes
   - Works by identifying directions that maximize separation between class means
   - Number of LDA dimensions possible = number of classes - 1

### Comparison with Other Methods
1. **LDA vs PCA**:
   - PCA finds directions of maximum variance
   - LDA finds directions that best separate classes
   - PCA is unsupervised, LDA is supervised

2. **LDA vs TSNE**:
   - TSNE can capture non-linear relationships
   - TSNE is not reproducible (different runs give different results)
   - LDA provides reproducible linear separations

### Important Properties
1. **Limitations**:
   - Can only have (n_classes - 1) dimensions
   - For binary classification, only 1 LDA dimension possible
   - Tends to underfit rather than overfit due to linear nature

2. **Usage with TSNE**:
   - Can use TSNE for initial clustering/labeling
   - Then apply LDA on original data for reproducible classification
   - Useful when TSNE reveals structure but isn't practical for deployment

## Practical Implementation
1. **Workflow**:
   ```python
   # Basic LDA workflow
   1. Get data and labels
   2. Split into train/test
   3. Fit LDA transform
   4. Transform data using LDA
   5. Train classifier on transformed data
   6. Evaluate using ROC curves
   ```

2. **Performance Evaluation**:
   - Use ROC curves to evaluate performance
   - Adjust probability thresholds for classification
   - Consider sensitivity/specificity tradeoffs

## Additional Topics Covered
1. **Dimensionality Reduction Review**:
   - Principal Component Analysis (PCA)
   - Eigenvalues and eigenvectors
   - Variance explained ratios

2. **Image Data Processing**:
   - MedMNIST dataset examples
   - 2D image to 1D vector conversion
   - Spatial relationships in image data

## Important Reminders
1. **Upcoming Tasks**:
   - Quiz on dimensionality reduction concepts
   - Project progress reviews
   - Assignment 4 completion
   - Assignment 5 on 2D image clustering, on Med-MNIST dataset of student's choice

2. **Review Areas**:
   - Linear vs non-linear dimensionality reduction
   - Mathematical concepts behind LDA
   - ROC curve interpretation
   - Feature engineering considerations