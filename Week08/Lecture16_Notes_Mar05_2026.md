# BIOENG-2390 Spring 2026 - Lecture 16
## March 5, 2026 (Thursday)

**Instructor:** Professor Prahlad G. Menon, PhD, PMP  
**Recording:** Zoom AI Transcript Available  
**Duration:** ~90 minutes

---

## 📋 Lecture Overview

Today's class finalized all project proposals and reviewed Linear Discriminant Analysis (LDA) through code examples. Each team presented their refined project plans with datasets verified, specific aims structured, and Kanban boards created.

**Topics Covered:**
1. **Project Finalization** - All 7 teams presented updated proposals
2. **LDA Code Review** - Walkthrough of LDA.ipynb notebook
3. **LDA vs PCA vs t-SNE** - When to use each method
4. **Multi-class Logistic Regression** - Extension beyond binary classification
5. **ROC Curve Analysis** - Performance evaluation techniques
6. **Assignment 5 Preview** - MedMNIST 2D image clustering
7. **Spring Break Planning** - Final preparations

**Key Focus:** Grounding project proposals in verified datasets and understanding supervised dimensionality reduction.

---

## 🎯 Finalized Project Teams

Based on discussions and the [Project Planning Spreadsheet](https://docs.google.com/spreadsheets/d/1zx0zyhs9P8LcYy8nOV064GZCjnf1McctYfpQntquJsQ/edit?pli=1&gid=1300895358#gid=1300895358):

### Team 1: Sepsis Prediction - Single vs Multi-Center Generalization
**Members:** CJ Shores, Carter Jones

**Final Hypothesis:**
> "We hypothesize that an algorithm trained on the vast multi-center network of eICU will exhibit significantly less performance degradation on predicting sepsis upon external validation compared to a MIMIC-IV model, proving that dataset heterogeneity is beneficial for strong prediction models."

**Datasets:**
- MIMIC-IV (single-center, Beth Israel)
- eICU (multi-center, 200 hospitals)
- VitalDB (additional validation)

**Specific Aims:**

**Aim 1:** Create overlapping feature space
- Gain access to eICU and MIMIC-IV via PhysioNet
- Download datasets and upload to Google Drive
- Load data in Colab
- Match features in both datasets (exclude unique features)
- Resample data to common time frame and sampling rate
- Add binary sepsis variable to random 70% using medically defined criteria

**Aim 2:** Develop benchmark models
- Using H2O Flow, develop XGBoost model
- 70% training, 10% validation, 20% test split
- Separate models for single-center and multi-center

**Aim 3:** Compare generalization performance
- Run MIMIC-IV model on its own test data
- Run MIMIC-IV model on eICU test data (cross-validation)
- Run eICU model on its own test data
- Run eICU model on MIMIC-IV test data (cross-validation)
- Compare AUC, sensitivity, average precision
- Draw conclusions about generalization

**Kanban:** [View Board](https://kanbanflow.com/board/7hNKePV)

---

### Team 2: ECG Arrhythmia Classification (Pivoted from PPG)
**Members:** Dibyasankha Kundu (DK), Anurag Kulkarni

**Final Hypothesis:**
> "A 1D CNN-LSTM model trained on raw ECG waveform segments from the MIT-BIH Arrhythmia Database can learn distinguishing temporal features of cardiac signals and accurately classify normal and arrhythmic heartbeats."

**Dataset:** MIT-BIH Arrhythmia Database
- Two-lead ECG recordings
- Beat-level arrhythmia annotations
- Multi-class or binary classification possible
- Classes: Normal, PVC (Premature Ventricular Contractions), APB (Atrial Premature Beats)

**Pivot Rationale:**
- Original PPG/heart failure too complex to match clinical data
- ECG arrhythmia well-defined problem
- Dataset readily accessible
- Clear labels available

**Specific Aims:**

**Aim 1:** Prepare and preprocess ECG waveform data
- Load MIT-BIH Arrhythmia Database
- Segment into individual beats or fixed windows
- Normalize and standardize signals
- Create labeled dataset for supervised learning

**Aim 2:** Develop deep learning model
- Implement 1D CNN architecture (or CNN-LSTM hybrid)
- Learn features directly from time-series inputs
- Train on segmented ECG beats
- Handle class imbalance if needed

**Aim 3:** Evaluate performance and reliability
- Assess using accuracy, precision, recall, F1-score
- Generate confusion matrices on held-out test dataset
- Analyze failure modes
- Compare to baseline models

**Alternative Focus (noted):**
> "Estimating heart failure classification (NYHA I-IV) from PPG/ECG dataset."

**Kanban:** [View Board](https://kanbanflow.com/board/VpQb3BS)

---

### Team 3: EEG-Based Attention Detection During Eye-Typed Communication
**Members:** Michael Christofidis, Kyle Thrush, Shaaz Nadeem, Aakash Kottakota

**Final Hypothesis:**
> "EEG spectral features extracted from time windows aligned with eye-tracking events will differ between attention (active key selection) and inattention periods, allowing machine learning models to reliably classify user attention during gaze-based spelling tasks."

**Dataset:** EEGET-ALS Dataset
- EEG recordings (EEG.edf)
- Eye-tracking data (ET.csv)
- Metadata (eeg.json, scenario.json)
- ALS patients and healthy controls using eye-typing system

**Project Summary:**
> "This project aims to determine whether EEG signals alone can detect when a user is actively paying attention during gaze-based typing. Eye-tracking data will be used to identify moments of key selection, and corresponding EEG segments will be analyzed using frequency-based features and dimensionality reduction techniques."

**Specific Aims:**

**Aim 1:** Generate attention labels
- Load EEG recordings and metadata
- Read eye-tracking data and timestamps from ET.csv
- Synchronize ET and EEG data using timestamps (compensate for sampling frequency)
- Identify periods of attention using gaze coordinates and character typing events
- Create labeled segments representing attention & inattention time windows

**Aim 2:** Extract spectral EEG features
- Segment EEG signals into fixed-length windows (2s tentatively)
- Compute spectral features (band power across EEG frequencies)
- Construct feature dataset from extracted EEG features
- Apply PCA to reduce dimensionality and visualize patterns

**Aim 3:** Train classification model
- Split feature dataset into training and testing sets (80/20 split)
- Use 5-fold cross-validation during training
- Train machine learning model (Kernel SVM tentatively)
- Evaluate performance: accuracy, precision, F1, specificity, sensitivity
- Visualize results and summarize for final report

**Kanban:** [View Board](https://kanbanflow.com/board/2LfZqFw)

---

### Team 4: Intracranial Aneurysm Detection from MRA
**Members:** Dallas B, Laura Claytor, Yuanzhe Huang, Lingyun Wang

**Final Hypothesis:**
> "We hypothesize that both age and sex may be associated with differences in volume or the prevalence of aneurysm location."

**Datasets:**
- Kaggle RSNA: Intracranial Aneurysm Detection AI Challenge
- MONAI: OpenNeuro dataset

**Project Summary:**
> "Anomaly detection of blood vessels in the brain specifically targeting vessels at risk of aneurysm. AIM1: Vessel Segmentation; AIM2: Aneurysm Detection"

**Specific Aims:**

**Aim 1:** Train U-Net vessel segmentation
- Implement simple U-NET architecture
- Develop preprocessing pipeline to crop images to 512×512 focused on target region
- Input: TOF (Time of Flight) MRA images
- Output: Associated 3D binary mask
- Match U-Net output with Vessel Mapper segmentation to identify aneurysm location

**Aim 2:** Demographic and modality analysis
- Load Kaggle dataset
- Segment based on modality feature (MRA), age, and sex
- Bin age into ranges and one-hot encode
- 80:20 split for train/test with 70:30 train vs validate split
- Implement supervised learning and ensemble models
- Evaluate predictive performance of log odds of aneurysm detection based on demographics and imaging modality

**Aim 3:** Determine correlations
- Analyze if aneurysm volume correlates with age, sex, or other factors
- Statistical analysis of associations
- Clinical interpretation

**Kanban:** [View Board](https://kanbanflow.com/board/KK1ks6F)

---

### Team 5: SEEG-Based Seizure Onset Zone Classification
**Members:** Jingxiao Sun, Michael Edwards

**Final Hypothesis:**
> "There are significant differences in the strength of information connectivity between SOZ (Seizure Onset Zone) and non-SOZ brain regions, and we are able to distinguish these features and classify SOZ versus non-SOZ regions using models such as SVM or random forest."

**Dataset:** Lab data (15 patients with SEEG recordings)
- Each patient: One 5-minute segment during sleep
- One 5-minute segment during awake state
- Interictal period (between seizures)
- Clinically annotated SOZ information

**Project Summary:**
> "This project utilizes SEEG data from epilepsy patients, combined with clinically annotated seizure onset zone (SOZ) information, to develop and optimize automatic SOZ identification methods. Functional connectivity matrices will be computed, and the strength of information connections between brain regions will be extracted as features to distinguish SOZ from non-SOZ areas."

**Specific Aims:**
- Extract functional connectivity features from SEEG
- Quantify information flow between brain regions
- Train SVM or Random Forest classifier
- Distinguish SOZ from non-SOZ regions
- Validate on held-out patients

**Alternative Dataset (if lab data inaccessible):**
- Neuromatch fMRI dataset (from March 3 discussion)
- Human Connectome Project

**Kanban:** [View Board](https://kanbanflow.com/board/w2aNYWH)

---

### Team 6: Biomechanics - Wheelchair Propulsion Analysis
**Members:** Marcel Oliart

**Datasets:**
- Markerless Motion Analysis System
- IMU/EMG sensors
- TU Delft wheelchair propulsion dataset

**Focus:** Gait analysis or biomechanics for optimal wheelchair propulsion

**Status:** Individual project, may need additional team members

**Kanban:** [View Board](https://kanbanflow.com/board/VhL9bdD)

---

### Team 7: Decoding Finger Kinematics from Intracortical Neural Activity
**Members:** Joshua Daniel

**Tentative Hypothesis:**
> "Applying dimensionality reduction and regression models to decode finger kinematics and evaluate generalization for neuroprosthetic control."

**Dataset:** LINK - Long Term Intracortical Neural Activity and Kinematics
- Source: DANDI Archive
- Intracortical recordings
- Finger movement kinematics
- Neuroprosthetic control application

**Focus:**
- Decode finger movements from neural signals
- Regression to predict kinematics
- Evaluate generalization across sessions
- Dimensionality reduction on neural features

**Status:** Individual project or seeking team members

---

## 📚 Linear Discriminant Analysis (LDA) - Code Review

### Overview from LDA.ipynb

**Purpose:** Supervised dimensionality reduction that maximizes class separability

**Key Concept:**
- Unlike PCA (finds variance), LDA finds directions that best separate classes
- Requires labels (supervised method)
- Linear transformation (reproducible)

### Mathematical Foundation

**Goal:** Maximize between-class variance while minimizing within-class variance

**Formula:**
```
Maximize: J(w) = (w^T S_B w) / (w^T S_W w)
```

Where:
- w = projection direction
- S_B = between-class scatter matrix
- S_W = within-class scatter matrix

**Result:** 
- Projection onto direction that maximizes class separability
- Can have max (n_classes - 1) dimensions

### Implementation from Notebook

**Using Spike Data (from Week 6-7):**

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load normalized spike data
# Shape: 3636 observations × 34 features

# Use GMM labels from t-SNE clustering as "ground truth"
# (Since we don't have actual labels for spike data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    spike_data_norm, gmm_labels, test_size=0.2, random_state=42
)

# Fit LDA
lda = LDA(n_components=1)  # For 2-class: max 1 component
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Train classifier on LDA features
clf = LogisticRegression()
clf.fit(X_train_lda, y_train)
y_pred = clf.predict(X_test_lda)

# Measure time taken (very fast!)
# ~0.007 seconds for fitting
```

**Key Observations:**
1. LDA requires labels (used GMM clusters as pseudo-labels)
2. Very fast computation
3. Single dimension for binary problem
4. Can apply to multi-class (max n_classes - 1 dimensions)

### LDA vs PCA Comparison

**From the Notebook:**

| Aspect | PCA | LDA |
|--------|-----|-----|
| **Type** | Unsupervised | Supervised |
| **Optimizes** | Variance | Class separability |
| **Labels Needed** | No | Yes |
| **Max Components** | n_features | n_classes - 1 |
| **Reproducible** | Yes | Yes |
| **Linear** | Yes | Yes |
| **Speed** | Fast | Fast |
| **Best For** | Exploration | Classification |

**When to Use LDA:**
- ✅ Have labeled data
- ✅ Want to maximize class separation
- ✅ Need feature engineering for classification
- ✅ Multi-class problem with clear boundaries

**When to Use PCA:**
- ✅ No labels available
- ✅ Exploratory data analysis
- ✅ Want to preserve variance
- ✅ Need many components

**When to Use t-SNE:**
- ✅ Visualization of complex structure
- ✅ Nonlinear relationships
- ⚠️ Not reproducible
- ⚠️ Cannot transform new data

### Multi-Class Logistic Regression

**Important Insight from Notebook:**
> "Logistic regression generalizes to more than two classes. We've always talked about sigmoid activation for binary classification, but logistic regression can be applied to more than two classes as well."

**How It Works:**
- Uses one-vs-rest or multinomial approach
- Each class gets its own set of coefficients
- Softmax activation for probability distribution
- Output: probability for each class

**Example with 3 Classes (from spike data + GMM):**
```python
# Logistic regression handles multi-class automatically
clf = LogisticRegression(multi_class='ovr')  # or 'multinomial'
clf.fit(X_train_lda, y_train)  # y_train can have 3+ classes

# Predictions
y_pred = clf.predict(X_test_lda)

# Probabilities
y_proba = clf.predict_proba(X_test_lda)
# Shape: (n_samples, n_classes)
```

### ROC Curve Analysis

**From LDA Notebook:**

**For Binary Classification:**
```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get prediction probabilities
y_proba = clf.predict_proba(X_test_lda)[:, 1]  # Probability of class 1

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - LDA + Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()
```

**Key Points:**
- AUC = Area Under Curve (ranges 0-1, higher better)
- 0.5 = random guessing
- >0.7 = acceptable
- >0.8 = good
- >0.9 = excellent

**Adjusting Thresholds:**
- Default threshold = 0.5
- Can adjust for sensitivity/specificity tradeoff
- Higher threshold → fewer positives (higher precision)
- Lower threshold → more positives (higher recall)

---

## 🔬 LDA in Practice - Key Lessons

### 1. Dimension Limitation

**Critical Constraint:**
```python
# For K classes, max LDA components = K - 1
lda = LDA(n_components=min(K-1, n_features))
```

**Examples:**
- Binary (K=2) → Max 1 LDA component
- 3 classes (K=3) → Max 2 LDA components
- 10 classes (K=10) → Max 9 LDA components
- MNIST digits (K=10) → Max 9 LDA components

**Why This Matters:**
- Can't use LDA for massive dimensionality reduction on binary problems
- For binary, limited to single discriminative direction
- Multi-class gives more LDA dimensions to work with

### 2. LDA with Unsupervised Labels

**Clever Workflow:**
1. Use t-SNE to discover structure (unsupervised, nonlinear)
2. Apply clustering (K-means or GMM) to get labels
3. Train LDA on ORIGINAL data using t-SNE cluster labels
4. Result: Reproducible linear transformation approximating nonlinear structure!

**From Notebook:**
```python
# Step 1: t-SNE clustering
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(spike_data_norm)

# Step 2: Cluster in t-SNE space
gmm = GaussianMixture(n_components=3)
pseudo_labels = gmm.fit_predict(X_tsne)

# Step 3: Train LDA on ORIGINAL data
lda = LDA(n_components=2)  # 3 classes → max 2 components
X_lda = lda.fit_transform(spike_data_norm, pseudo_labels)

# Now can transform NEW data with LDA (unlike t-SNE!)
X_new_lda = lda.transform(new_spike_data)
```

**Advantage:**
- Combines t-SNE's nonlinear discovery with LDA's reproducibility
- Can apply to new data (t-SNE can't do this!)
- Linear approximation of nonlinear structure

### 3. Confusion Matrix Analysis

**Multi-Class Confusion Matrix:**
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# For 3-class problem
cm = confusion_matrix(y_test, y_pred)

# cm[i,j] = number of samples with true label i predicted as j
# Diagonal = correct predictions
# Off-diagonal = misclassifications

# Visualize
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
```

**Interpretation:**
- High values on diagonal = good performance
- Off-diagonal = confusion between classes
- Can identify which classes are hard to distinguish

---

## 💻 Code Snippets from Lecture

### Complete LDA Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load data (assuming spike_data_norm and labels exist)
X = spike_data_norm  # Shape: (n_samples, 34)
y = gmm_labels  # From previous clustering

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit LDA
n_classes = len(np.unique(y))
lda = LDA(n_components=min(n_classes-1, X.shape[1]))
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Train classifier
clf = LogisticRegression(random_state=42)
clf.fit(X_train_lda, y_train)

# Predict
y_pred = clf.predict(X_test_lda)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))

# Visualize (if 2D)
if X_train_lda.shape[1] >= 2:
    plt.figure(figsize=(10, 6))
    for class_val in np.unique(y_train):
        mask = y_train == class_val
        plt.scatter(X_train_lda[mask, 0], X_train_lda[mask, 1], 
                   label=f'Class {class_val}', alpha=0.6)
    plt.xlabel('LDA Component 1')
    plt.ylabel('LDA Component 2')
    plt.title('Data in LDA Space')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### Comparing Classification Performance

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

# Baseline: Logistic Regression on original features
start = time.time()
clf_baseline = LogisticRegression()
clf_baseline.fit(X_train, y_train)
baseline_time = time.time() - start
y_pred_baseline = clf_baseline.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred_baseline)

# With LDA: Logistic Regression on LDA features
start = time.time()
clf_lda = LogisticRegression()
clf_lda.fit(X_train_lda, y_train)
lda_time = time.time() - start
y_pred_lda = clf_lda.predict(X_test_lda)
lda_acc = accuracy_score(y_test, y_pred_lda)

# Compare
print(f"Baseline: {baseline_acc:.3f} accuracy in {baseline_time:.4f}s")
print(f"With LDA: {lda_acc:.3f} accuracy in {lda_time:.4f}s")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_baseline = confusion_matrix(y_test, y_pred_baseline)
disp1 = ConfusionMatrixDisplay(cm_baseline)
disp1.plot(ax=axes[0])
axes[0].set_title('Baseline')

cm_lda = confusion_matrix(y_test, y_pred_lda)
disp2 = ConfusionMatrixDisplay(cm_lda)
disp2.plot(ax=axes[1])
axes[1].set_title('With LDA')

plt.tight_layout()
plt.show()
```

**Typical Results:**
- LDA often faster (fewer features)
- Accuracy similar or better
- Especially effective when classes well-separated

---

## 🎯 Assignment 5 Preview: MedMNIST

### What is MedMNIST?

**Extension of MNIST to medical imaging:**
- Multiple medical imaging datasets
- All standardized to 28×28 or similar small sizes
- Various modalities: X-ray, CT, ultrasound, etc.
- 10+ different datasets available

**Assignment 5 Focus:**
> "2D image clustering on MedMNIST dataset of student's choice"

**Possible Datasets:**
- PathMNIST: Pathology images
- DermaMNIST: Dermatology images
- BloodMNIST: Blood cell images
- PneumoniaMNIST: Chest X-rays
- And more!

**What You'll Do:**
1. Select a MedMNIST dataset
2. Apply dimensionality reduction (PCA, t-SNE)
3. Cluster in reduced space (K-means, GMM)
4. Visualize clusters
5. Train classifier on clusters (optional: LDA)

**Connection to Projects:**
- Practice for image-based projects
- Reinforces dimensionality reduction concepts
- Prepares for deep learning on images later

---

## 🔑 Key Takeaways

### 1. All Projects Now Finalized!
- ✅ 7 teams with defined hypotheses
- ✅ Datasets identified (varying accessibility)
- ✅ Specific aims structured
- ✅ Kanban boards created
- ✅ Ready for implementation after spring break

### 2. LDA Concepts Solidified
- ✅ Supervised dimensionality reduction maximizes separability
- ✅ Max dimensions = n_classes - 1 (critical limitation!)
- ✅ Can use with t-SNE labels for reproducibility
- ✅ Much faster than training on original high-D data
- ✅ Logistic regression works for multi-class

### 3. Project Diversity
- **Tabular:** Sepsis classification (Team 1)
- **Signal:** ECG arrhythmia (Team 2), EEG attention (Team 3), SEEG SOZ (Team 5)
- **Image:** Brain aneurysm MRA (Team 4)
- **Biomechanics:** Wheelchair/IMU (Team 6)
- **Neural:** Finger kinematics decoding (Team 7)

### 4. Common Themes Across Projects
- All require data wrangling → model-ready dataset
- All use classification or regression
- All apply dimensionality reduction or feature engineering
- All evaluate with standard metrics (AUC, accuracy, F1)
- All address generalization (overfitting prevention)

### 5. Technical Challenges Identified
- **Signal data:** Resampling, segmentation, synchronization
- **Image data:** File sizes, 3D processing, visualization
- **Lab data:** Privacy, accessibility, documentation
- **Multi-dataset:** Feature alignment, unit standardization

---

## 📊 Project Comparison Matrix

| Team | Data Type | Supervision | Main Challenge | Key Method |
|------|-----------|-------------|----------------|------------|
| 1 | Tabular | Supervised | Cross-dataset generalization | XGBoost |
| 2 | Signal (ECG) | Supervised | Temporal feature learning | 1D CNN-LSTM |
| 3 | Signal (EEG) | Supervised | Multi-modal sync | Spectral features + SVM |
| 4 | Image (MRA) | Supervised | 3D processing | U-Net / 3D CNN |
| 5 | Signal (SEEG) | Supervised | Connectivity features | SVM / Random Forest |
| 6 | Sensor (IMU/EMG) | TBD | Motion analysis | TBD |
| 7 | Neural | Supervised | Kinematics decoding | Regression + Dim Reduction |

---

## 🗓️ Updated Timeline

### This Week (Before Spring Break)
- **Tuesday (March 3):** Initial consultations ✅
- **Thursday (March 5):** Finalized proposals ✅

### Spring Break (March 7-14)
- Refine specific aims documents
- Verify dataset accessibility
- Begin preliminary data exploration
- Update Kanban boards

### After Spring Break
- **Tuesday (March 17):** Assignment 4 DUE
  - Specific aims page (1 page)
  - Kanban board link
- Begin active implementation
- Weekly progress check-ins

### Mid-April
- Final project presentations
- Deliverables due

---

## 📚 Resources & Tools

### Datasets Mentioned
- **MIMIC-IV, MIMIC-III:** MIT PhysioNet (ICU data)
- **eICU:** Multi-center ICU data (PhysioNet)
- **VitalDB:** Surgical patient monitoring
- **MIT-BIH:** Arrhythmia ECG database
- **EEGET-ALS:** EEG + eye-tracking (Springer Nature)
- **Kaggle RSNA:** Brain aneurysm challenge
- **BIDMC CHF:** Congestive heart failure (PhysioNet)
- **CAPNObase:** PPG + clinical data
- **Neuromatch:** fMRI datasets with notebooks
- **LINK (DANDI):** Intracortical neural + kinematics

### Tools & Libraries
- **Kanban Flow:** Project management
- **Mermaid:** Flowchart generation
- **soul.py:** AI agent creation
- **MONAI:** 3D medical imaging
- **ParaView:** 3D visualization
- **H2O Flow:** AutoML platform
- **scikit-learn:** LDA, PCA, classifiers

### Notebooks to Review
- `LDA.ipynb` - Complete LDA implementation
- Previous dimensionality reduction notebooks from Weeks 5-7

---

## 🙋 Questions from Students

**Q: Can I use my research lab's private data?**
**A:** Yes! Just ensure you have permission and proper documentation. Team 5 is doing this.

**Q: What if dataset is too large (200GB)?**
**A:** Sample strategically! 10-20 examples for initial work, not the entire dataset.

**Q: Can I pivot my project focus?**
**A:** Yes, multiple teams pivoted (Team 2, Team 5). But do it NOW before spring break!

**Q: How do I handle different units across datasets?**
**A:** Standardize to common unit. Example: convert all heights to centimeters. That's feature engineering!

**Q: Is vessel segmentation required for aneurysm project?**
**A:** No - make binary classification the primary aim. Segmentation is optional stretch goal.

---

## 💡 Professor's Key Advice (Repeated Throughout)

### On Datasets
> "The last thing we want is to work hard on a proposal then have to change the dataset."

### On Scope
> "Make the first aim classification. That way you have a bird in hand."

### On Feasibility
> "Make sure you can do this with some samples - that's the best thing you can do to make it sound reasonable."

### On Collaboration
> "Keep a local copy of your project description - Google Sheets is collaborative but risky!"

### On Implementation
> "The goal is to show you can do something, not to write a publication."

### On Signal Processing
> "You will 100% have to resample. Every dataset from different labs will have different sampling rates."

---

## 📝 Homework for Spring Break

### All Students:
- [ ] Finalize specific aims page (1 page document)
- [ ] Complete Kanban board with all tasks
- [ ] Create workflow flowchart (Mermaid recommended)
- [ ] Verify dataset samples download and load correctly
- [ ] Begin data exploration if possible
- [ ] Prepare for implementation starting March 17

### Team-Specific:
**Team 1:** Verify MIMIC-IV and eICU access, identify overlapping features  
**Team 2:** Download MIT-BIH, test CNN-LSTM architecture  
**Team 3:** Verify EEGET-ALS download, test EEG-ET synchronization  
**Team 4:** Sample Kaggle RSNA data (don't download all 200GB!), test ParaView  
**Team 5:** Explore Neuromatch colab notebook, verify lab data access  
**Team 6:** Check in with professor  
**Team 7:** Verify DANDI archive access

---

## 🎓 Learning Outcomes Achieved

By completing this week, students have:

1. ✅ Structured scientific projects using Agile methodology
2. ✅ Developed testable hypotheses grounded in datasets
3. ✅ Created specific aims with detailed task breakdowns
4. ✅ Understood LDA as supervised dimensionality reduction
5. ✅ Recognized when to use LDA vs PCA vs t-SNE
6. ✅ Applied multi-class logistic regression
7. ✅ Interpreted ROC curves and confusion matrices
8. ✅ Identified technical challenges for different data types
9. ✅ Practiced collaborative project planning
10. ✅ Prepared for independent project implementation

---

**Have a productive spring break! See you March 17 with finalized proposals!** 🌴📚

**Critical Reminder:** Verify your dataset works before break ends!
