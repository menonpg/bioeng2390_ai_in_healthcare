# BIOENG 2390: AI in Healthcare

**University of Pittsburgh, Department of Bioengineering**

## Course Overview
Welcome to BIOENG 2390: AI in Healthcare. 

This course provides an introduction to artificial intelligence (AI) and machine learning (ML) in healthcare. The course covers fundamental concepts in AI/ML, data preprocessing, model development, evaluation, and deployment. The course also includes hands-on labs and assignments to apply these concepts to healthcare datasets.


# Week 4 Lecture Notes - Machine Learning Model Development and Evaluation (Part 2)

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