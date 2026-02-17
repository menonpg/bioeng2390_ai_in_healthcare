# BIOENG-2390 Spring 2026 - Lecture 11
## February 17, 2026

**Instructor:** Professor Prahlad G. Menon, PhD, PMP  
**Recording:** [View on Fathom](https://fathom.video/share/GHbQG-wscK9FsAauEVrKcKscHDHp5ryj)  
**Duration:** 65 minutes

---

## 📋 Lecture Overview

Today's class focused on:
1. Final project planning and team formation
2. Kanban Flow for agile project management
3. Dataset resources for projects (HuggingFace, PhysioNet, Kaggle, Google Dataset Search)
4. Neural spike data analysis (fMRI/EEG context)
5. Practical PCA application with sklearn
6. K-means clustering in reduced dimensional space
7. Saving/loading PCA models with pickle
8. Introduction to t-SNE (non-linear dimensionality reduction)
9. PCA vs t-SNE: reproducibility differences
10. Normalizing time series data (z-score normalization)

**Note:** Professor was under the weather - class ended early

---

## 🎯 Final Project Planning

### Team Formation

**Requirements:**
- **Team size:** 3 students (max 4 allowed)
- **Deadline:** Form teams by Thursday, Feb 19
- **Tracking:** Update Google Sheet (Column G) with 'Y' when team formed

**Team Deliverables:**
1. **Assignment 4:** Project proposal (due soon)
2. **Mid-April:** Final project delivery

**Project Scope:**
- Select a dataset (priority!)
- Define analysis goals
- Can extend beyond class material
- Work like real organization team

---

### Kanban Flow for Project Management

**Tool:** [kanbanflow.com](https://kanbanflow.com) (free)

**Why Kanban?**
> "Literally every organization today, small or large, follows some form of agile or project tracking. Knowing how to do that is a good idea."

**Kanban Board Structure:**
```
Backlog → In Progress (This Week/Sprint) → Done
```

**Task Examples:**
- Import data
- Select models
- Create features
- Analyze models
- Midterm presentation
- Final presentation
- Code organization
- Paper/poster

**Features:**
- Assign tasks to individuals
- Color-code task types
- Track progress
- Collaborate as team

**Professor's Examples:**
- Echocardiography image segmentation project
- Tasks assigned to team members
- Progression from backlog → done

---

## 📊 Dataset Resources for Projects

### 1. HuggingFace

**Website:** [huggingface.co/datasets](https://huggingface.co/datasets)

**Data Types:**
- 3D, Audio, Documents
- Geospatial, Tabular, Image
- Signal, Text, Time Series

**Features:**
- Dataset Studio (visualize before downloading)
- Multi-class, multi-label support
- Code examples included
- Search by type or keyword

**Professor demonstrated:** ECG deepfake generator (Spaces)
- Pre-trained model generates synthetic ECGs
- Can create training data from models!
- Code available in `app.py` file

---

### 2. PhysioNet

**Website:** [physionet.org](https://physionet.org)

**Highlighted:** CAP Sleep Study Database
- Multi-lead EEG data
- Sleep stage recordings
- Disease labels (insomnia, etc.)
- Thousands of recordings
- Requires registration (can list professor as PI)

**Benefits:**
- Well-documented
- Bibliography of papers using dataset
- Sample code sometimes included
- Don't have to replicate exact study!

**Example Use:**
> "Take insomnia labels and time series data. Use 100 observations. Do your own analysis."

---

### 3. Kaggle

**Why Kaggle:**
- Competition datasets
- Active forums
- Many people asked questions
- Community solutions
- Well-curated

**Advantage:**
> "Because many people participate, many have asked questions. You can look into forums and help yourself to that information."

---

### 4. Google Dataset Search

**Website:** [datasetsearch.research.google.com](https://datasetsearch.research.google.com)

**What it does:**
- Aggregates datasets from all sources
- Searches HuggingFace, Kaggle, Roboflow, etc.
- One search interface

**Professor demonstrated:** Plant pathology datasets
- Found on HuggingFace, Roboflow
- Medical imaging not required!
- Can use botany, zoology data

---

### 5. Stanford AIMI

**Center for Artificial Intelligence in Medicine & Imaging**
- Curated medical imaging datasets
- High quality, well-documented
- Research-grade data

---

## 🧠 Neural Spike Data Analysis

### The Dataset

**Data:** `spike_data.csv`
- **Shape:** 3,636 observations × 34 time points
- **Type:** Standardized neural signal amplitudes
- **Source:** Brain activity measurements (EEG/fMRI equivalent)

**Context:**

**Invasive recording:**
- Mouse with skull removed
- Surface of brain exposed
- Direct neural signal measurement
- Activity during specific tasks

**Non-invasive recording (fMRI):**
- Blood Oxygen Level Dependent (BOLD) signal
- Oxygenated vs deoxygenated blood
- When brain region activated → blood oxygenates
- Intensity change captured over time

---

### Visual Inspection

**Before Normalization:**
```
Amplitude vs Time:
- Some signals clearly different
- Visual clusters apparent
- Time point ~9 shows separation
- Can distinguish by eye
```

**Key Assumption:**
> "Each individual time point amplitude is an independent predictor of activation state"

**Different from EEG seizure problem:**
- EEG: Individual points NOT good predictors
- Had to use windowed frequency features
- Neural spikes: Individual amplitudes ARE predictive
- Context-dependent!

---

## 🔬 Normalization Process

### Z-Score Normalization

**Method:** At each time point, across all signals

```python
# For each of 34 time points:
mean_t = np.mean(data[:, t])  # Mean across all signals
std_t = np.std(data[:, t])     # Std dev across all signals

# Normalize
data_norm[:, t] = (data[:, t] - mean_t) / std_t
```

**What this does:**
- Each time point now has mean=0, std=1
- Values represent "# of std devs from mean"
- Brings out subtle differences

**Result:**
- Original strong features (time=9) still visible
- NEW features become apparent
- Better for clustering

**Saved to pickle:**
```python
import joblib

# Save normalization parameters
joblib.dump({'mean': means, 'std': stds}, 'normalization_params.pkl')

# Can apply same normalization to new data!
```

---

## 📐 PCA Application

### Choosing Number of Components

**Original dimensions:** 34 time points

**How many PCs exist?**
- Maximum = N features = 34 PCs
- But don't need all!

**Selected:** K = 5 components

**Why 5?**
> "From the naked eye, I can see about 1, 2, 3, 4, 5 regions/time points that seem to have good separability of these lines."

**Visual inspection guides K selection!**

---

### Implementation

```python
from sklearn.decomposition import PCA

# Create PCA with 5 components
pca = PCA(n_components=5)

# Fit on sample (50 signals)
X_sample = data_norm[:50, :]
X_pca = pca.fit_transform(X_sample)

# X_pca shape: 50 × 5 (reduced from 50 × 34)

# Get components
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_  # 5 × 34 matrix

# Sort by eigenvalue
idx = eigenvalues.argsort()[::-1]
eigenvalues_sorted = eigenvalues[idx]
eigenvectors_sorted = eigenvectors[idx]
```

---

### Visualization in PC Space

```python
import matplotlib.pyplot as plt

# Plot first two PCs
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC0 (1st Principal Component)')
plt.ylabel('PC1 (2nd Principal Component)')
plt.title('Neural Signals in PC Space')
plt.show()
```

**Observation:**
- Two distinct clusters!
- PC0 vs PC1 shows clear separation
- Even without labels, clusters visible

---

## 🎯 K-Means Clustering

### Unsupervised Labeling

```python
from sklearn.cluster import KMeans

# Cluster in PC space (2D or 5D)
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_pca[:, :2])  # Use PC0, PC1

# Labels: 0 or 1 (we don't know which is "activated")
# But we know there are 2 groups!
```

**What we learned:**
- Cluster 0: One type of neural response
- Cluster 1: Different type
- **Don't know which is "activated" vs "not activated"**
- Would need expert validation

---

### Applying to All Data

**Trained on 50 signals, apply to all 3,636:**

```python
# 1. Transform all data with fitted PCA
X_all_pca = pca.transform(data_norm)  # Use same PCA model!

# 2. Predict clusters with fitted K-means
all_labels = kmeans.predict(X_all_pca[:, :2])

# Now all 3,636 signals have cluster labels!
```

**Key Insight:**
> "I can give a class assignment to all data based on a model fit to just 50 observations!"

**Two models saved:**
1. PCA transformation (reproducible)
2. K-means cluster centers (reproducible)

---

## 🌟 t-SNE Introduction

### What is t-SNE?

**t-Distributed Stochastic Neighbor Embedding**

**Type:** Non-linear dimensionality reduction

**How it differs from PCA:**

**PCA (Linear):**
- Finds linear projection (plane)
- Reproducible (same input → same output)
- Can apply to new data easily
- Fast

**t-SNE (Non-Linear):**
- Finds non-linear manifold
- **NOT reproducible** (random initialization)
- **Cannot directly apply to new data**
- Slow but powerful

---

### Implementation Teaser

```python
from sklearn.manifold import TSNE

# Non-linear reduction
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(data_norm)

# Different from PCA!
# - No .transform() method
# - Must fit_transform all data at once
# - Can't add new points later
```

**From Notebook:**
- Multiple methods shown: Isomap, LLE, MDS, t-SNE, NMF
- Linear: PCA, NMF, MDS
- Non-linear: t-SNE, Isomap, LLE
- t-SNE often best for visualization

**Next Class:**
- Deep dive into t-SNE
- Why non-reproducible
- When to use vs PCA
- How to work around limitations

---

## 🔑 Key Concepts from Lecture

### 1. Project Dataset Selection

**Priority:** Start with dataset!
- Ensure data is available
- Understand data structure
- Read papers using dataset
- Don't have to replicate exact study

**Can use:**
- Public datasets
- Your research lab data
- Generated data (from models!)

### 2. Pickle Files for Model Persistence

**What to save:**
```python
import joblib

# Save PCA model
joblib.dump(pca, 'pca_model.pkl')

# Save normalization params
joblib.dump({'mean': means, 'std': stds}, 'norm_params.pkl')

# Save K-means model
joblib.dump(kmeans, 'kmeans_model.pkl')
```

**Why:**
- Reproducibility
- Apply same transformations to new data
- Share with team
- Production deployment

### 3. Assumption Check

**EEG Seizure (Weeks 2-3):**
- Individual amplitudes NOT good predictors
- Needed windowed frequency features

**Neural Spikes (Week 6):**
- Individual amplitudes ARE good predictors
- Can use directly (after normalization)
- **Always verify assumptions!**

---

## 📚 Real-World Examples

### 1. fMRI Brain Imaging

**What it measures:**
- Blood oxygen levels
- Deoxygenated → Oxygenated = activation
- Non-invasive brain activity

**Example:**
> "You're reading a book and you giggle. The giggle activates a brain region. That region converts deoxygenated blood to oxygenated. The intensity increase is captured."

**Signal pattern:**
- Activation → Intensity ↑
- Return to baseline → Intensity ↓
- Over ~34 time points

### 2. Face Recognition with PCA

**Application mentioned:**
- Each pixel = feature (thousands!)
- PCA reduces to "eigenfaces"
- Used in early face recognition
- Datasets available (old but useful)

### 3. Plant Pathology

**Demonstrated from dataset search:**
- Identify plant diseases
- Bounding boxes around lesions
- Available on HuggingFace, Roboflow
- Shows: Not just medical data!

---

## 🎯 For Next Class (Thursday, Feb 19)

### Topics:

**Quiz 1 likely postponed or adjusted**

1. **t-SNE Deep Dive**
   - Why non-linear methods needed
   - t-SNE algorithm overview
   - Reproducibility issues
   - When to use vs PCA

2. **Other Dimensionality Reduction Methods**
   - Isomap, LLE, MDS
   - Linear vs non-linear comparison
   - Choosing the right method

3. **Interpreting Clusters**
   - Validating unsupervised labels
   - Connecting to domain knowledge
   - Supervised learning on unsupervised labels

### Homework:

- [ ] Run `Dimensionality Reduction for Time Series Classification.ipynb`
- [ ] Form project teams (update Google Sheet!)
- [ ] Browse datasets (HuggingFace, PhysioNet, Kaggle)
- [ ] Think about project ideas
- [ ] Sign up for Kanban Flow
- [ ] Complete pending assignments (0, 1, 2)

---

## 🙋 Questions from Class

**Q: How many people per team?**  
**A:** 3 students ideal, max 4 allowed. Class small enough for teams of 3.

**Q: Does project have to be biomedical/human subjects?**  
**A:** NO! Can use plant data, animal data, any domain you're interested in. Or your research lab data.

**Q: Do we have to replicate the exact study from dataset papers?**  
**A:** NO! Read the papers for context, but design your own analysis. Example: "Take insomnia labels and do your own thing."

**Q: What if dataset requires PI signature?**  
**A:** Can list Professor Menon, but inform him first. He may already have access.

**Q: Why normalize each time point separately?**  
**A:** Makes values comparable across time. Each time point gets mean=0, std=1. Values = "# std devs from mean at that time."

**Q: Can we apply PCA model to new data?**  
**A:** YES! Save with pickle, load anytime, use `.transform()` on new observations. This is why PCA is reproducible.

**Q: Why can't we do that with t-SNE?**  
**A:** t-SNE has random initialization, no `.transform()` method. Must fit all data at once. Next class: workarounds using supervised classification.

---

## 💻 Code from Today

### Loading and Normalizing Spike Data

```python
import pandas as pd
import numpy as np

# Load data
spike_data = pd.read_csv('spike_data.csv')
# Shape: 3636 × 34

# Normalize each time point
data = spike_data.values
means = np.mean(data, axis=0)  # 34 means
stds = np.std(data, axis=0)    # 34 std devs

# Z-score normalization
data_norm = (data - means) / stds

# Save normalization params
import joblib
joblib.dump({'mean': means, 'std': stds}, 'normalization.pkl')
```

### PCA + K-Means Pipeline

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. PCA on sample
sample_idx = np.random.choice(3636, 50, replace=False)
X_sample = data_norm[sample_idx, :]

pca = PCA(n_components=5)
X_pca_sample = pca.fit_transform(X_sample)

# 2. K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
labels_sample = kmeans.fit_predict(X_pca_sample[:, :2])

# 3. Apply to all data
X_pca_all = pca.transform(data_norm)
labels_all = kmeans.predict(X_pca_all[:, :2])

# Save models
joblib.dump(pca, 'pca_model.pkl')
joblib.dump(kmeans, 'kmeans_model.pkl')
```

---

## 📋 Week 06 Lecture 11 Checklist

- [ ] Form project team (update Google Sheet)
- [ ] Sign up for Kanban Flow
- [ ] Browse datasets (HuggingFace, PhysioNet, Kaggle)
- [ ] Run dimensionality reduction notebook
- [ ] Understand PCA + K-means pipeline
- [ ] Understand normalization (z-score per time point)
- [ ] Understand pickle for model persistence
- [ ] Prepare for t-SNE deep dive (Thursday)

---

## 🔑 Key Takeaways

1. ✅ Projects require datasets - find them first!
2. ✅ HuggingFace, PhysioNet, Kaggle, Google Dataset Search = main resources
3. ✅ Kanban Flow for agile project management
4. ✅ Teams of 3 (max 4) forming by Thursday
5. ✅ Neural spike data: individual time points ARE predictive (unlike EEG)
6. ✅ Normalize per time point for comparability
7. ✅ PCA reduces 34D → 5D → visualize 2D
8. ✅ K-means finds 2 clusters in PC space
9. ✅ Trained on 50, apply to 3,636 (model reusability!)
10. ✅ t-SNE preview: non-linear, non-reproducible (next class)

**Next:** t-SNE deep dive, more clustering methods, project discussions

**Form your teams and find your datasets!** 🚀

---

**Note:** Professor was under the weather. Get well soon, Professor Menon! 💊
