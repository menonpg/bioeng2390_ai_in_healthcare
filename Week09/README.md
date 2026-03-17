# BIOENG 2390: AI in Healthcare

**University of Pittsburgh, Department of Bioengineering**

## Course Overview
Welcome to BIOENG 2390: AI in Healthcare. This course includes comprehensive introduction to programming in Python, R, and MATLAB, as well as setting up development environments and using Integrated Development Environments (IDEs) to develop linear, nonlinear, and deep learning models for healthcare applications, starting with time series data, image data, and text data.


# Week 9 Lecture Notes, Lecture 17
This lecture focused on building and analyzing a color classification model.

## 1. Environment Setup
- Started with Git repository setup
- Created folder structure for week 9
- Emphasized importance of maintaining organized development environment

## 2. Data Collection Exercise
- **Task**: Collect RGB color data for purple/non-purple classification
- **Tools Used**: 
  - Google Sheets for collaborative data collection
  - RapidTables website for RGB color values
- **Data Structure**:
  - R, G, B values as features
  - Binary classification (purple/not purple)
  - Included contributor names

## 3. Data Access & Preparation
- Demonstrated two approaches:
  1. Google Sheets API (encountered some access issues)
  2. Google Colab integration (successful approach)
- Data Cleaning Steps:
  - Converted R,G,B to float type
  - Converted class to categorical
  - Handled missing values
  - Standardized class labels (purple/not purple)

## 4. Model Building with H2O
- Setup:
  - Initialized H2O with 2GB memory
  - Split data into train/test sets
- Model Training:
  - Used AutoML approach
  - Built multiple models including:
    - Gradient Boosting Machine (GBM)
    - Extreme Randomized Trees
    - Stacked Ensembles
- Model Performance:
  - Best model achieved high accuracy
  - Confusion matrix analysis
  - Used optimal threshold (0.784) for predictions

## 5. Model Analysis
- Variable Importance:
  - Red channel most important
  - Followed by Green and Blue
- Error Analysis:
  - Identified misclassified cases
  - Attempted to implement LIME for explanation (ran into technical issues)

## 6. Key Technical Concepts Covered
- Data preprocessing
- Model building automation
- Ensemble methods
- Model interpretation techniques
- API integration
- Version control practices

## 7. Tools & Libraries Used
- H2O AutoML
- Pandas
- Google Sheets API
- LIME (attempted)
- Git for version control

## Important Takeaways
1. Importance of organized development environment
2. Value of collaborative data collection
3. Need for proper data cleaning and preprocessing
4. Benefits of ensemble models
5. Importance of model interpretability

## Notes on Technical Challenges
- Google Sheets API authentication issues
- LIME integration with H2O models CODE UPDATED in CreateModelsFromCSV_h2o.ipynb  
- Importance of proper data format conversion between tools

The lecture emphasized practical implementation and real-world challenges in machine learning projects, from data collection to model interpretation.