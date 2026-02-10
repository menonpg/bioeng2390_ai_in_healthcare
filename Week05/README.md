# BIOENG 2390: AI in Healthcare

**University of Pittsburgh, Department of Bioengineering**

## Course Overview
Welcome to BIOENG 2390: AI in Healthcare. 

This course provides an introduction to artificial intelligence (AI) and machine learning (ML) in healthcare. The course covers fundamental concepts in AI/ML, data preprocessing, model development, evaluation, and deployment. The course also includes hands-on labs and assignments to apply these concepts to healthcare datasets.


# Week 5 Lecture Notes, Lecture 10 - Understanding features selection using supervised learning, dimensional reduction and clustering

## Key Topics Covered

#### **Overview & Recap**
- **Catch-Up Session**: This class aimed to get back on track after a missed lecture (Lecture 9).
- **Assignments**: Emphasis on completing Assignment 1 (feature engineering) and starting Assignment 2.
- **Review of Topics Covered**:
  - **Feature Engineering**: Transforming raw time-series data into engineered features.
  - **Classification Techniques**: Binary, multi-class, and multi-label classifications, regression.
  - **Model Performance Metrics**: Sensitivity, specificity, precision, recall, AUC, ROC, and threshold optimization.
  - **AutoML**: Using H2O AutoML to build and evaluate models.

#### **Concept of Overfitting**
- **Overfitting Risk**: When the number of features exceeds the number of observations (e.g., 100 features, 50 observations), models, especially complex ones (e.g., decision trees), may overfit.
  - **Less Data than Features**: Increases the risk of models fitting noise.
  - **More Features than Observations**: A situation that demands dimensionality reduction to avoid overfitting.

#### **Dimensionality Reduction**
- **Why Needed?**: 
  - **Feature Explosion**: After feature engineering, we often have a large number of features (e.g., 100).
  - **Problem**: More features than observations often leads to overfitting and poor generalization.
  
- **Techniques for Dimensionality Reduction**:
  1. **Feature Selection**: 
     - **Goal**: Choose the most informative features while reducing redundancy (e.g., variance inflation).
     - **Feature Selection Criteria**: 
       - **Variance**: Maximize variance explained by selected features.
       - **Variance Inflation**: Minimize features explaining overlapping variance.
  2. **Dimensionality Reduction Methods**:
     - **Principal Component Analysis (PCA)**: Projects the data into a new feature space where variance is maximized.
     - **Non-negative Matrix Factorization (NMF)**, **MDS**, **t-SNE**: Alternative techniques to PCA, with t-SNE being nonlinear and not reversible.
     - **Goal**: Reduce the data dimensionality while retaining the most critical information for analysis.

#### **Feature Selection Process**
- **Mathematical Framework**: 
  - We discussed the scenario where you have 100 features but fewer observations.
  - By performing **Feature Selection**, we identify the key features that best explain the variance in the data.
  - This could be done manually (e.g., using box plots for exploratory analysis) or via models like KNN.

#### **Supervised vs Unsupervised Feature Selection**
- **Supervised Feature Selection**: Involves training a model first (using all features) and then selecting the most important features based on the model's evaluation (e.g., variable importance).
- **Unsupervised Feature Selection**: Involves using techniques like PCA without knowing the class labels. The focus is on the data structure (clusters) instead of predicting a specific outcome.

#### **Dimensionality Reduction Using PCA**
- **PCA in Action**:
  - Given 34 features (e.g., EEG data), PCA can reduce the features down to just the most important ones (e.g., 5), while maintaining the majority of the data's variance.
  - After performing PCA, the data points are projected into principal components, making it easier to detect clusters or separable regions in the data.
  - **Benefits**: PCA makes dimensionality reduction **intelligent**, as it aligns with the variance in the data.

#### **Clustering with Reduced Dimensions**
- **K-Means Clustering**:
  - K-Means can be applied after dimensionality reduction (PCA) to create clusters in the data.
  - **Objective**: Find two (or more) distinct clusters in the data using the reduced dimensions.
  - **Unsupervised Learning**: By reducing the dimensionality first and then applying clustering, you can detect hidden patterns even without knowing the exact classes.

#### **Feature Selection in Practice**
- **Exploration**: From previous lectures, features like **Alpha** and **Theta** were found to be the most important for classifying seizure vs non-seizure data.
  - **Delta** and **Beta** were less important, and removing them reduced the risk of overfitting.
  - **Feature Importance**: By analyzing the impact of each feature, we can reduce the feature space while retaining predictive power.

---

### **Key Concepts Covered**
1. **Overfitting**: A problem when too many features are used compared to the number of observations. Dimensionality reduction helps mitigate this.
2. **Dimensionality Reduction**: Techniques like PCA reduce features to the most important ones, based on variance, thus simplifying the model without losing too much information.
3. **Feature Selection**: A process of picking the most relevant features either manually or using statistical models, ensuring better generalization and avoiding overfitting.
4. **Supervised vs Unsupervised Feature Selection**: Supervised methods use model evaluation to select features, whereas unsupervised methods focus on data structure (e.g., clustering).

#### **Next Steps**
- **Assignment Focus**: Continue to work on **Assignment 1** (feature engineering), and **start Assignment 2**.
- **Practical Exercises**: 
  - **Use PCA** on time-series data to reduce dimensions.
  - **Perform Clustering** (e.g., using K-Means) to understand the data structure.

---

Feel free to reach out if you have any specific questions on dimensionality reduction, feature selection, or any other topics covered in today's lecture.