# BIOENG 2390: AI in Healthcare

**University of Pittsburgh, Department of Bioengineering**

## Course Overview
Welcome to BIOENG 2390: AI in Healthcare. This course includes comprehensive introduction to programming in Python, R, and MATLAB, as well as setting up development environments and using Integrated Development Environments (IDEs) to develop linear, nonlinear, and deep learning models for healthcare applications, starting with time series data, image data, and text data.


# Week 10 Lecture Notes, Lecture 19
Here are the lecture notes for Week 10, Lecture 19:

# Week 10 Lecture 19: Model Interpretability and Support Vector Machines

## 1. Project Management with Kanban
- Students using Kanbanflow.com for project management
- Features:
  - Create boards and invite team members
  - Track tasks through different stages (To Do → Do Today → In Progress → Done)
  - Add descriptions, links, and content to tasks
  - Can organize as user stories or individual tasks
  - Similar to industry tools like Jira, Atlassian, Notion

## 2. Model Interpretability Tools

### LIME (Local Interpretable Model-agnostic Explanations)
- Helps understand why models make specific predictions
- Creates an "explainer" that:
  - Takes a specific decision to explain
  - Makes small variations around that decision
  - Observes how model responds to variations
  - Creates simple interpretable explanations
- Example with house price prediction:
  - 3 bedrooms → +$100,000
  - Close to downtown → +$150,000
  - New roof → +$50,000

### SHAP (SHapley Additive exPlanations)
- More rigorous mathematical approach based on game theory
- Characteristics:
  - More global in scope
  - More thorough and mathematically rigorous
  - Computationally intensive
  - Generally more accurate than LIME
- Uses force plots to show feature contributions
- Can generate violin plots to show feature impact distributions

## 3. Types of Machine Learning

### Learning Paradigms
1. Supervised Learning
   - Uses labeled data
   - Examples: Classification, regression

2. Unsupervised Learning
   - Uses unlabeled data
   - Examples: Clustering, dimensionality reduction

3. Reinforcement/Reward-based Learning
   - Learns incrementally over time
   - More relevant to modern large language models

### Classification Methods
1. Numerical Classification
   - Linear classification
   - Logistic regression
   - Support Vector Machines
   - Neural networks/perceptrons

2. Parametric Classification
   - Assumes underlying distribution (e.g., normal distribution)
   - Examples: Naive Bayes, Gaussian mixture models

3. Non-parametric Instance-based
   - k-Nearest Neighbors
   - Kernel regression methods
   - Kernel density estimation

## 4. Support Vector Machines (SVM)
- Can handle nonlinear decision boundaries
- Works by:
  - Creating higher dimensional feature space
  - Finding linear separation in higher dimensions
  - Converting nonlinear boundaries to linear ones in higher space
- Applications:
  - One-class classification (anomaly detection)
  - Two-class classification
  - Can handle complex decision boundaries

## Key Takeaway
- No Free Lunch Theorem: No single classifier works best for all problems
- Simple models can be very effective when:
  - Using novel feature spaces
  - Applying domain knowledge
  - Having clear objective functions
  - Understanding the specific problem context

## Homework (Ungraded) Assignment
- Review notebook "svm_lets_classify_purple"
- Run through one-class and two-class SVM examples
- Not graded but recommended for understanding next class material