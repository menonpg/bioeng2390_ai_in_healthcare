# BIOENG-2390 Spring 2026 - Lecture 12
## February 19, 2026

**Instructor:** Professor Prahlad G. Menon, PhD, PMP  
**Recording:** [View on Fathom](https://fathom.video/share/KsFonyWyPqsb9ubznFJ-onPKHsyNoceJ)  
**Duration:** 68 minutes

---

## 📋 Lecture Overview

Today's class focused on:
1. Project team formation and dataset discussions
2. Individual project pitches and feedback
3. Dataset resources for specific topics
4. Signal standardization (resampling + segmentation)
5. t-SNE non-reproducibility problem
6. Workaround: Training regression models on t-SNE embeddings
7. Comparison of dimensionality reduction methods (PCA, NMF, MDS, Isomap, LLE, t-SNE)
8. Assignment 4 deadline: Project proposal due March 6

---

## 🎯 Project Discussions

### Team 1: CJ & Carter
**Topic:** Pacemaker lead placement optimization  
**Idea:** Using ECG leads to classify good vs bad lead placement

**Challenge:**  
- No public datasets found for lead placement
- Very proprietary data
- **Action:** Need to pivot to available data or find alternative

**Professor's Guidance:**
> "This may be the best project in the world, but if there's no data, you can't do anything in a reasonable amount of time. You've got to find another dataset or topic."

**Recommended approach:** Start with dataset, then define project

---

### Team 2: Michael, Kyle, Shaz & Akash  
**Topic:** ALS motor unit loss prediction  
**Data:** HD-EMG (High-Density Electromyography)

**Hypothesis:**  
> "What features best predict motor unit loss between ALS and control patients?"

**Data Type:**
- Muscle EMG (like neural spikes)
- Surface level recording
- Some noise present
- Fast vs slow-twitch muscle fiber delineation

**Features:**
- Amplitude peaks
- Frequency data (like EEG frequency analysis from class)
- Disease progression metrics

**Datasets Provided:**
- EEG and eye tracking in ALS patients (PhysioNet)
- Public ALS EMG datasets available

**Status:** Checking with PI for lab data, have public alternatives

---

### Team 3: Dallas
**Topic:** Social Determinants of Health (SDOH)  
**Analysis:** Propensity to advance through healthcare stages

**Data Type:**
- Healthcare referrals (food, housing, transportation)
- Social barriers to healthcare
- 1115 waiver program (New York State)

**Hypothesis:**
> "Social determinants of health predict regional hospital rates"

**Datasets Provided:**
- Duke SDOH repository
- CMS mapping disparities data
- California SDOH & hospitalization study

**Status:** Awaiting compliance approval for work data, have public alternatives

---

### Team 4: Jingxiao Sun
**Topic:** Lower back pain detection from ultrasound  
**Data:** Multimodal imaging (ultrasound + CT)

**Analysis:**  
Extract image features → Predict pain

**Approach Options:**
1. Transfer learning (pre-trained neural networks)
2. Traditional image features (GLCM, etc.)
3. SAM/SAM2/SAM3 for embeddings

**Datasets Provided:**
- Lumbar spine ultrasound with CT ground truth (Nature paper)
- Spinal cord ultrasound dataset
- Open Data Commons spinal cord injury longitudinal imaging

**Status:** Good data availability, clear path forward

---

### Team 5: Marcel
**Topic:** Wheelchair propulsion strategy classification  
**Data:** IMU sensors, EMG (shoulder), gait analysis

**Focus:**
- Manual wheelchair propulsion
- Different surfaces
- Motion capture

**Tools:**
- OpenPose (markerless motion capture)
- SAM/SAM2 for human pose detection
- CMU-based landmark tracking

**Application:** Caregiver posture analysis during transfers

**Dataset Provided:**
- Wheelchair gait analysis data (various sources)

**Professor's Encouragement:**
> "High schoolers made OpenPose work with copilot help. With Gen AI, it's so easy now!"

---

## 🔬 Signal Standardization

### The Problem

**Scenario:** Multiple data sources with different characteristics

**Site 1:**
- Sampling rate: 500 Hz (500 samples/second)
- Patient: High heart rate
- Signal length: Variable

**Site 2:**
- Sampling rate: 250 Hz (250 samples/second)
- Patient: Low heart rate
- Signal length: Different

**Challenge:** How to combine into one dataset?

---

### The Solution: Resampling + Segmentation

**Two Key Steps:**

#### 1. Segmentation
**Goal:** Identify repeating patterns (e.g., cardiac cycles)

**Process:**
- Use features to identify start/end of cycles
- ECG: R-peak to R-peak = one cycle
- EEG: Seizure window boundaries
- Segment data into consistent units

**Example:**
```
Long signal: ─────▲─────▲─────▲─────
             Cycle1 Cycle2 Cycle3

Extract each cycle as separate observation
```

#### 2. Resampling
**Goal:** Standardize number of samples per segment

**Upsampling (5 Hz → 10 Hz):**
```
Original:  •     •     •     •     •
                ↓ linear interpolation
Upsampled: • • • • • • • • • •
```

**Downsampling (500 Hz → 250 Hz):**
```
Original:  • • • • • • • • • •
                ↓ decimate
Decimated: •   •   •   •   •
```

**Methods:**
- Linear interpolation
- Spline fitting
- Fourier transform method

---

### Fourier Transform Resampling

**The Trick:**

**Step 1:** Convert time → frequency domain
```
Signal (N points) → FFT → Frequency (N points)
```

**Step 2:** Modify frequency content
```
Keep N/2 frequencies → Discard high frequencies
```

**Step 3:** Inverse transform
```
Reduced frequencies → IFFT → Signal (N/2 points)
```

**Advantage:**
- Natural anti-aliasing
- Preserves low-frequency content
- Mathematically sound

**From Lecture:**
> "Discrete Time Fourier Transform always has same number of points in time and frequency domain. Cut frequencies, reconstruct → downsample!"

---

### Aliasing Problem

**What is Aliasing?**

**Original signal:** High-frequency oscillation
```
     /\  /\  /\  /\  
────/  \/  \/  \/  \───
```

**Downsample too aggressively:**
```
Sample points: •   •   •   •
Connected:     ────────────  ← Looks like DC!
```

**Lost high-frequency information!**

**Solution:**
- Upsample when possible
- Use anti-aliasing filters
- Fourier method naturally handles this

---

## 🎯 t-SNE Deep Dive

### Why t-SNE is Non-Reproducible

**The Problem:**

**PCA:**
```python
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_train)

# Later, on new data:
X_new_reduced = pca.transform(X_new)  # Works!
```

**t-SNE:**
```python
tsne = TSNE(n_components=2)
X_reduced = tsne.fit_transform(X_train)

# Later, on new data:
X_new_reduced = tsne.transform(X_new)  # ERROR! No .transform()
```

**Why?**
- Random initialization
- Iterative optimization
- Different run → different embedding
- No fixed transformation matrix

**From Lecture:**
> "Every time you run t-SNE, even on the same data, you get a different shape of clusters. It's because it's coming from a different embedding space every time."

---

### The Workaround

**Idea:** Train supervised model on t-SNE embeddings

**Process:**

**Step 1:** Get t-SNE embeddings for training data
```python
X_tsne = tsne.fit_transform(X_train)
# X_tsne shape: N × 2 (t-SNE coordinates)
```

**Step 2:** Train regression models to predict t-SNE coordinates
```python
from sklearn.ensemble import RandomForestRegressor

# Predict t-SNE_x from original features
model_x = RandomForestRegressor()
model_x.fit(X_train, X_tsne[:, 0])

# Predict t-SNE_y from original features
model_y = RandomForestRegressor()
model_y.fit(X_train, X_tsne[:, 1])
```

**Step 3:** Apply to new data
```python
# Predict t-SNE coordinates for new data
X_new_tsne_x = model_x.predict(X_new)
X_new_tsne_y = model_y.predict(X_new)
X_new_tsne = np.column_stack([X_new_tsne_x, X_new_tsne_y])
```

**Limitation:**
- Depends on regression model R²
- If R² low → poor approximation
- Never as good as native t-SNE
- But better than nothing!

**From Lecture:**
> "If you have higher R-squared with t-SNE dimension prediction, you can reproducibly get those values. If not, projection won't work very well either."

---

## 📊 Dimensionality Reduction Methods Comparison

### Linear Methods

**1. PCA (Principal Component Analysis)**
- **Speed:** Fast
- **Reproducible:** YES
- **Finds:** Maximum variance directions
- **Best for:** Quick exploration, when variance = structure
- **Limitation:** Poor for severe dimensionality reduction (784D → 2D)

**2. NMF (Non-negative Matrix Factorization)**
- **Like PCA but:** All values remain positive
- **Use case:** When features are inherently positive (images, counts)
- **Limitation:** Still linear

**3. MDS (Multi-Dimensional Scaling)**
- **Actually:** Can be non-linear!
- **Preserves:** Pairwise distances
- **Like:** Making a map from distance table
- **Good for:** When distances matter more than axes

---

### Non-Linear Methods

**All distance-based (like KNN!):**

**1. t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- **Best for:** Visualization of complex data
- **Preserves:** Local structure (neighborhoods)
- **Use case:** Handwritten digits, complex clusters
- **Limitation:** Not reproducible, slow

**2. Isomap (Isometric Mapping)**
- **Measures:** Geodesic distances (along curved surfaces)
- **Preserves:** Global geometry
- **Use case:** Manifold learning

**3. LLE (Locally Linear Embedding)**
- **Analyzes:** Local neighborhood relationships
- **Preserves:** Local linearity
- **Fast:** Faster than t-SNE

**Key Commonality:**
> "All these MDS, Isomap, LLE, t-SNE are based on regional distances and analysis of distances of observations next to each other, much like K-nearest neighbors."

---

## 💡 Practical Insights

### When to Use Which Method?

**PCA:**
- ✅ First choice always
- ✅ Reproducible
- ✅ Fast
- ✅ Interpretable (eigenvectors have meaning)
- ❌ Poor for severe reduction

**t-SNE:**
- ✅ Best visualization
- ✅ Handles complex patterns
- ✅ Separates overlapping clusters
- ❌ Slow
- ❌ Not reproducible
- ❌ Axes have no physical meaning

**Use Both:**
1. PCA for initial exploration
2. t-SNE for final visualization
3. Train model on t-SNE labels if needed

---

## 📚 Key Terminology

**Pseudo-Labeling:**
- Using clustering to create labels
- Labels from unsupervised learning
- Use for downstream supervised learning

**Embeddings:**
- Fixed-length numerical representations
- From images, text, signals → vectors
- Enable tabular ML methods on complex data

**R-Squared (R²):**
- Goodness of fit metric
- Percentage variance explained by predictors
- Range: 0 (bad) to 1 (perfect)
- **Formula:** R² = 1 - (SS_residual / SS_total)

---

## 🎬 For Next Class & Beyond

### Assignment 4: Project Proposal

**Due:** March 6, 2026 (FIRM DEADLINE!)

**Requirements:**
1. One-page project summary
2. Team members listed
3. Dataset identified and accessible
4. Hypothesis statement
5. Specific aims
6. Kanban board URL

**Process:**
- Tuesday (Feb 24): More project discussions
- Thursday (Feb 26): Generate proposals with Gen AI
- By March 6: Submit finalized proposals

**Timeline:**
- March 6: Proposals due
- March 6 - Mid-April: Execute project
- Mid-April: Final deliverables

### Week 7 Structure

**Lecture 13 (Feb 24):** Self-paced quiz
- Take-home or in-class
- +20 bonus for completing during class
- Full points if submitted before Thursday

**Lecture 14 (Feb 26):** Resume normal lectures

---

## 🙋 Questions from Class

**Q: How many people per team?**  
**A:** 3 students maximum (4 allowed if necessary)

**Q: Can we use data from our research labs?**  
**A:** Yes! But need PI approval. Also have public alternatives ready.

**Q: What if we can't find a dataset for our idea?**  
**A:** Must pivot quickly! Start with dataset, then define project around it.

**Q: Is downsampling preferred over upsampling?**  
**A:** NO! Downsampling can cause aliasing (information loss). Upsample when possible.

**Q: Why does t-SNE give different results each time?**  
**A:** Random initialization + iterative optimization. Each run starts from different point, converges to different solution.

**Q: How do we apply t-SNE to new data?**  
**A:** Can't directly. Workaround: Train regression model to predict t-SNE coordinates. Quality depends on R².

---

## 💻 Code Concepts from Notebook

### Complete Pipeline

```python
# 1. Load and normalize
data = pd.read_csv('spike_data.csv')
mean = data.mean(axis=0)
std = data.std(axis=0)
data_norm = (data - mean) / std

# 2. PCA on sample
X_sample = data_norm[:50]
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_sample)

# 3. K-means clustering
kmeans = KMeans(n_clusters=2)
labels_sample = kmeans.fit_predict(X_pca[:, :2])

# 4. Apply to all data
X_all_pca = pca.transform(data_norm)  # Reproducible!
labels_all = kmeans.predict(X_all_pca[:, :2])

# 5. t-SNE (non-reproducible)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_sample)  # Can't reuse!

# 6. Workaround: Train regression
from sklearn.ensemble import RandomForestRegressor
model_x = RandomForestRegressor().fit(X_sample, X_tsne[:, 0])
model_y = RandomForestRegressor().fit(X_sample, X_tsne[:, 1])

# Predict t-SNE for new data
X_new_tsne_x = model_x.predict(X_new)
X_new_tsne_y = model_y.predict(X_new)
```

---

## 🎓 Professor's Guidance

**On Dataset Selection:**
> "Lead with a dataset. If you have an idea, look for the dataset. If you can't find it quickly, switch topics. Don't get stuck at data collection stage."

**On Industry Skills:**
> "Johnson & Johnson changed recruitment from asking people to write code to giving them a dataset and asking them to analyze it. If you can spin up H2O and analyze data, you have valuable skills!"

**On Project Management:**
> "Some companies want students who can join a scrum team, get assigned stories on Kanban board, and hit the ground running. This isn't just academic - it's real industry practice."

**On t-SNE:**
> "t-SNE is best for visualizing complex data like handwritten digits. But it's not reproducible. That's the only drawback."

**Assignment 4 Deadline:**
> "This is the ONLY assignment with a firm deadline: March 6th. Everything else is flexible, but the proposal must be submitted by then."

**Professor Prahlad Menon, PhD, PMP**  
*Office Hours: By appointment*  
*Email: prm44@pitt.edu*

---

*"Start with the dataset, then define the project around it."*

— Core advice for project success

---

## 🔑 Key Takeaways

1. ✅ Teams of 3 forming (4 if needed)
2. ✅ Project ideas emerging across diverse topics
3. ✅ Dataset availability is CRITICAL - must verify first
4. ✅ Resampling + segmentation standardizes variable-length signals
5. ✅ Upsampling preferred over downsampling (avoid aliasing)
6. ✅ Fourier transform method for resampling
7. ✅ PCA is reproducible, t-SNE is not
8. ✅ t-SNE best for visualization, PCA for production
9. ✅ Workaround: Train regression to predict t-SNE embeddings
10. ✅ Assignment 4 (proposal) due March 6 - FIRM DEADLINE

**Next:** Self-paced quiz (Lecture 13), then project proposal generation (Lecture 14)

**Form teams, find datasets, start planning!** 🚀
