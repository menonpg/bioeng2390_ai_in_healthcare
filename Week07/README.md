# BIOENG 2390: AI in Healthcare

**University of Pittsburgh, Department of Bioengineering**

## Course Overview
Welcome to BIOENG 2390: AI in Healthcare. This course includes comprehensive introduction to programming in Python, R, and MATLAB, as well as setting up development environments and using Integrated Development Environments (IDEs) to develop linear, nonlinear, and deep learning models for healthcare applications, starting with time series data, image data, and text data.


# Week 7 Lecture Notes, Lecture 13
## Class Overview
Today's class focused on project discussions and planning. Students presented their project proposals, received feedback, and discussed implementation strategies. The class was primarily dedicated to reviewing project ideas rather than introducing new content.

## Project Discussions

### Project 1: Microglia Classification
- **Objective**: Classify microglia (brain immune cells) based on morphology to determine active vs. resting states
- **Data**: Two-photon microscope images (2D max projections from 3D)
- **Approach**: Extract features from images to analyze branching patterns using unsupervised learning
- **Key Features**: Cell soma size/shape, process branching patterns, branch length, branch number
- **Discussion Points**:
  - Importance of local vs. global features in image analysis
  - Need for skeletonization and pre-processing steps
  - Suggestion to document manual labeling process before clustering

### Project 2: MRI Image Segmentation
- **Objective**: Segment lesions in brain MRI images
- **Data**: 2D slices from MRI scans
- **Approach**: U-Net or simplified version for segmentation
- **Discussion Points**:
  - Computational requirements and potential use of Google Colab
  - Need to consider morphological analysis of segmented regions
  - Suggestion to explore SimpleITK library for morphological analysis

### Project 3: Protein Analysis for Glaucoma Prediction
- **Objective**: Analyze protein data from aqueous humor to identify biomarkers for glaucoma
- **Data**: 349 samples with protein measurements and clinical metadata
- **Approach**: Clustering, dimensionality reduction, and classification
- **Discussion Points**:
  - Feature selection strategies
  - Data engineering tasks to combine protein data with clinical metadata
  - Approaches for dichotomizing gene expression data

### Project 4: EEG Analysis
- **Objective**: Classify stroke patients or analyze sleep stages using EEG data
- **Data Options**: 
  1. 16-channel EEG from stroke patients (pending approval)
  2. Public sleep EEG data from PhysioNet (40GB)
- **Approach**: Extract features and use unsupervised learning for classification
- **Discussion Points**:
  - Alternative project plan if primary data access not approved
  - Categorization of sleep stages using objective measures

## Administrative Items

### Assignment Feedback
- Reminder that students should submit both notebooks AND written reports for assignments
- Reports should include:
  - Description of methodology
  - Key findings and results
  - Interpretation of results
  - Figures and tables as needed

### Assignment Clarifications
- Discussion about which dataset to use for assignments
- Clarification on using EEG sleep data with specific labels

### Next Steps
- Students to update project proposals based on feedback
- Complete and resubmit assignments with proper reports
- Next class will focus on two-dimensional analysis of clustering

## Upcoming Schedule
- Next class: Two-dimensional analysis of clustering
- Following class: In-depth project review and further discussion

## Resources Mentioned
- SimpleITK library for image morphological analysis
- G-power for statistical power analysis
- Google Colab for computational resources


# Week 7, Lecture 14  
# Machine Learning Class - Week 7, Lecture 14

## Class Overview
Today's class began with brief project updates from each team, followed by an in-depth exploration of dimensionality reduction techniques applied to two-dimensional image data. The instructor emphasized the transition from analyzing time series data (where each time point was treated independently) to analyzing image data where global features are more meaningful than individual pixel values.

## Dimensionality Reduction for Image Data

### MNIST Dataset Introduction
- Dataset consists of handwritten digits (0-9)
- Each image is 28×28 pixels (784 pixels total)
- Original dataset contains 70,000 images
- For the exercise, a random subset of 1,000 images was used

### Key Concepts Covered:

1. **Difference from Time Series Analysis**:
   - In time series data: Each time point treated as independent predictor
   - In image data: Individual pixel values are poor independent predictors
   - Need to consider global features of images rather than individual pixels

2. **Linear vs. Nonlinear Dimensionality Reduction**:
   - **PCA (Linear)**: 
     - Creates eigenvectors that represent global features of intensity
     - Each image is a linear combination of these eigenvectors
     - Showed limited ability to separate digit classes

   - **NMF (Non-negative Matrix Factorization)**:
     - Similar to PCA but constrains values to be positive
     - Also showed limited separation ability

   - **t-SNE (Nonlinear)**:
     - Produced much better clustering of digit classes
     - Demonstrates that nonlinear combinations of global features are needed
     - Different perplexity values affect clustering results

3. **Perplexity in t-SNE**:
   - Controls how many neighbors are considered when forming clusters
   - Lower perplexity: More small clusters
   - Higher perplexity: Fewer, larger clusters
   - For 1,000 observations with 10 classes, perplexity around 50-100 works well

4. **Visualization of Principal Components**:
   - Each principal component represents a global feature of the images
   - Images are reconstructed through linear combinations of these components

### Clustering Methods
- Brief introduction to K-means clustering and Gaussian Mixture Models (GMM)
- These will be covered in more theoretical detail in the next class

## Administrative Items

1. **Assignments**:
   - Reminder to complete assignments 1-3 this week
   - For assignments submitted without reports, students need to create and submit reports

2. **Spring Break**:
   - Next week (March 2-9) is spring break
   - By March 11 (first class after break), all teams should have:
     - Completed project proposals
     - Filled Excel sheets with specific aims
     - Started preliminary data analysis

3. **Future Topics**:
   - After spring break: Review of classification metrics, H2O, and working with tabular and image data
   - Following that: Introduction to neural networks

## Resources Shared
- SimpleITK notebooks for image segmentation and analysis
- Additional notebooks on K-means clustering and Gaussian Mixture Models
