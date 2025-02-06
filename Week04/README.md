# BIOENG 2390: AI in Healthcare

**University of Pittsburgh, Department of Bioengineering**

## Course Overview
Welcome to BIOENG 2390: AI in Healthcare. 

This course provides an introduction to artificial intelligence (AI) and machine learning (ML) in healthcare. The course covers fundamental concepts in AI/ML, data preprocessing, model development, evaluation, and deployment. The course also includes hands-on labs and assignments to apply these concepts to healthcare datasets.


# Week 4 Lecture Notes, Lecture 7 - Machine Learning Model Development and Evaluation (Part 2)

## Key Topics Covered

### Cross-Validation & Model Evaluation
- Cross-validation helps determine if a problem/response variable is generalizable
- Types of cross-validation:
  - K-fold cross-validation (e.g. 5-fold, 10-fold)
  - Leave-one-out cross-validation (most extensive/expensive)
- Cross-validation metrics being high indicates:
  - Features are useful regardless of data source
  - Problem is generalizable

### ROC Curves & Model Performance
- Components:
  - True Positive Rate (TPR) = Sensitivity = Recall
  - False Positive Rate (FPR) 
  - Specificity = 1 - FPR
- Optimal threshold typically found at:
  - Northwest corner of ROC curve
  - Where sensitivity = specificity
- For rare events, optimal threshold ≈ prevalence of positive class
- For balanced datasets, optimal threshold ≈ 50%

### Model Competition Analysis on Windowed EEG Features Dataset Engineered Last Class 
- Used lazy_predict library to compare multiple models
- Key findings:
  - KNN classifier performed best (~68% accuracy)
  - Nonlinear classifiers outperformed linear models
  - Balanced accuracy ~73%
  - AUC ~73%
  - F1 score ~69%

### Implementation Details
- Data preparation:
  - Split data into training (80%) and testing (20%)
  - Convert response variable to categorical
- Model evaluation metrics:
  - Accuracy
  - Balanced accuracy 
  - Sensitivity/Specificity
  - Precision
  - AUC

## Key Takeaways
1. Cross-validation is crucial for understanding model generalizability
2. Optimal threshold selection depends on data balance and problem type
3. Multiple model comparison helps identify best approach
4. ROC curves provide comprehensive performance visualization

## Next Steps
1. Complete Assignment 1
2. Begin work on Assignment 2 (model building using H2O)
3. Next class: Implementation in R




# Week 4 Lecture Notes, Lecture 8

## Administrative Updates
- Assignments 7.1 and 7.2 are now available
- Three students have submitted assignment 1
- Content for assignments has been covered in class
- H2O experience should help with assignment 2

## Key Concepts Covered

### 1. Types of Classification Problems
- **Based on Response Variable Type**
  - Continuous → Regression models
  - Categorical → Classification models

- **Types of Classification**
  1. Binary Classification (2 classes)
  2. Multi-class Single Label
  3. Multi-class Multi-label
  4. Binary Multi-label

### 2. Real-world Examples
1. **ECG Classification**
   - Binary: Normal vs Abnormal
   - Multi-class: Normal, AFib, VT, ST elevation, etc.
   - Multi-label: Multiple conditions in one ECG

2. **MRI Brain Lesion Detection**
   - Multi-class: Different types of lesions
   - Multi-label: Multiple lesions in one image

3. **Medical Coding**
   - Multi-class: Thousands of ICD-10 codes
   - Multi-label: Multiple conditions per chart

### 3. ROC Curve Operating Points
- **Three key operating scenarios:**
  1. **Balanced Dataset**
     - Optimal threshold where sensitivity ≈ specificity
     - Typically at northwest corner of ROC curve

  2. **COVID-type Situation**
     - Higher sensitivity preferred
     - Lower threshold
     - Accept more false positives
     - "Better safe than sorry" approach

  3. **Cancer Drug Situation**
     - Higher specificity preferred
     - Higher threshold
     - Minimize false positives
     - "Better careful than sorry" approach

### 4. Model Implementation in R
- Libraries used: tidyverse, caret, pROC, e1071
- Key steps:
  1. Data preparation
  2. Cross-validation setup
  3. Model training
  4. Performance evaluation
  5. ROC curve plotting

### 5. Linear vs Nonlinear Classification
- **Understanding Through TensorFlow Playground**
  1. Single neuron → Linear decision boundary
  2. Multiple neurons → Nonlinear decision boundaries
  3. Feature engineering impact
     - Adding nonlinear features (x², y², sin(x), etc.)
     - Combining with model complexity

### 6. Overfitting Concepts
- Too many parameters can lead to overfitting
- Balance needed between:
  - Model complexity
  - Feature engineering
  - Number of neurons/layers
  - Learning rate
  - Regularization

## Key Takeaways
1. Choose classification approach based on problem structure
2. Select ROC operating point based on use case
3. Balance model complexity with data requirements
4. Consider both feature engineering and model architecture
5. Watch for overfitting when increasing model complexity