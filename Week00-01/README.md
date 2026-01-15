# Week 00-01: Introduction to Programming and Environment Setup
### BIOENG 2390: AI in Healthcare - Spring 2026

**Instructor:** Professor Prahlad Menon, PhD, PMP  
**University of Pittsburgh, Department of Bioengineering**

---

## 🎯 Welcome to Week 00-01!

Hello and welcome! I'm Professor Prahlad Menon, and I'm thrilled to have you in BIOENG 2390. This first week is all about laying a strong foundation. Think of this as preparing your toolkit before we start building amazing AI applications in healthcare.

---

## 📺 Lecture Recordings & Notes

### Lecture 1 - January 13, 2026 (81 minutes)
**Focus:** Course Setup & Introduction to AI in Healthcare

- **[Watch Recording](https://fathom.video/share/B2xPtF4xDdwWipoxVLtfx2KXH4utRqze)**
- **[Read Detailed Lecture Notes](Lecture01_Notes_Jan13_2026.md)** ← Complete transcript with all links

**Topics Covered:**
- Course logistics (Canvas, Zoom, GitHub, Google Drive)
- Real-world AI in Healthcare: AATS Risk Calculator example  
- Key ML concepts: response variables, predictive variables, prevalence, bias
- Development environment setup (VS Code, Git, GitHub Copilot, Miniconda)
- First hands-on coding with Jupyter notebooks
- Working with Python to generate synthetic patient data

---

### Lecture 2 - January 15, 2026
**Focus:** EEG Data Processing Pipeline (MATLAB → R)

**Topics Covered:**
- MATLAB: Loading and exporting EEG signal data
- R: Complete EEG data preprocessing workflow
- Creating time axes for signal data
- Signal normalization (z-score)
- Adding ground truth labels for classification
- Kernel density visualization
- Multi-language data pipelines

**Files Created:**
- `MATtoCSV.m` - MATLAB script for EEG data export
- `ExploreEEGData_1152026.R` ⭐ **NEW!** - Complete R preprocessing pipeline

---

## 🎯 Week Learning Objectives

By the end of this week, you will be able to:

1. ✅ Set up development environments for Python, R, and MATLAB
2. ✅ Write and execute basic programs in all three languages
3. ✅ Understand when to use each language in healthcare applications
4. ✅ Work with real biomedical data (EEG signals)
5. ✅ Process data through multi-language pipelines
6. ✅ Use version control (Git/GitHub) for your projects
7. ✅ Navigate between different IDEs and cloud platforms

---

## 📁 Week 00-01 Files & Data Processing Pipeline

### Complete EEG Data Workflow

#### Step 1: Generate Synthetic Data (Python)
**File:** `generate_csv.py`
- Generates synthetic patient data (age, height, weight, gender)
- Demonstrates pandas DataFrames and CSV export
- **Usage:** `python generate_csv.py myFile.csv`

#### Step 2: Process EEG Signals (MATLAB) 
**File:** `MATtoCSV.m`
- Loads `EEG_sleep.mat` containing brain wave data
- Visualizes signal at different sampling rates
- Exports 10,000 data points (500 Hz, 20 seconds)
- **Output:** `s_of_t_subset.csv`

#### Step 3: Analyze & Preprocess (R) - **Lecture 2 Content** ⭐
**File:** `ExploreEEGData_1152026.R` **NEW!**

This complete R script demonstrates the full preprocessing pipeline:

**Key Features:**
- Imports EEG data from CSV (output of MATtoCSV.m)
- Creates time axis: `time = (row_number - 1) / 500` (500 Hz sampling)
- Normalizes signal: `normalizedValue = (s - mean(s)) / sd(s)` (z-score)
- Adds ground truth labels: "Seizure" if time > 12s, else "Normal"
- Generates visualizations:
  - Time series plot of raw EEG signal
  - Time series plot of normalized signal
  - Kernel density plots (unnormalized vs normalized)
  - Density plot of normalized signal only

**Dependencies:**
```r
library(readr)   # For reading CSV files
library(dplyr)   # For data manipulation
```

**Output Files:**
- `s_of_t_subset_withTimeAxis.csv` - EEG data with time column
- `s_of_t_subset_CLEAN.csv` - Complete processed dataset with:
  - `value`: Raw signal values
  - `time`: Time in seconds  
  - `normalizedValue`: Z-score normalized signal
  - `GT`: Ground truth labels (Seizure/Normal)

**Usage:**
```r
# After running MATtoCSV.m in MATLAB:
source("ExploreEEGData_1152026.R")
```

#### Step 4: Advanced Analysis & Outlier Detection (Python/Colab) - **Lecture 2 Content** ⭐
**File:** `Week01_2390_Notebook01.ipynb` **NEW!**

This comprehensive Jupyter/Colab notebook demonstrates advanced data analysis techniques:

**Key Features:**
- **Google Drive integration** for cloud-based data access
- **Loads processed EEG data** from R output (`s_of_t_subset_CLEAN.csv`)
- **Exploratory Data Analysis (EDA):**
  - DataFrame inspection with `.info()`, `.head()`, `.describe()`
  - Grouped statistics by GT (Ground Truth) labels
  - Boxplots comparing Normal vs Seizure states
- **Outlier Detection:**
  - IQR (Interquartile Range) method with customizable margin
  - Separate outlier removal for Normal and Seizure groups
- **Data Normalization:**
  - Z-score normalization: `(x - mean) / std`
  - Min-Max normalization: `(x - min) / (max - min)`
- **Interactive Visualization:**
  - Plotly line plots for time series exploration
  - Seaborn boxplots for distribution comparison
- **Multi-language Integration:**
  - Uses `rpy2` to run R code within Python notebook
  - Demonstrates seamless Python ↔ R interoperability

**Dependencies:**
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
# For R integration:
%load_ext rpy2.ipython
```

**Output Files:**
- `s_of_t_subset_CLEAN_outliersRemoved.csv` - EEG data with outliers removed from both Normal and Seizure groups

**Usage:**
```python
# In Google Colab:
# 1. Mount Google Drive
# 2. Run all cells sequentially
# 3. Explore interactive visualizations
```

**What You'll Learn:**
- How to connect Google Colab to Google Drive
- Advanced pandas data manipulation
- Statistical outlier detection methods
- Multiple normalization techniques
- Creating publication-quality visualizations
- Running R code from Python notebooks

---

#### Step 5: Basic Interactive Exploration (Jupyter)
**File:** `myFirstNotebook.ipynb`
- Simple interactive Python notebook environment
- Demonstrates generating synthetic patient data
- Introduction to pandas DataFrames and CSV export
- Can be used to practice basic Python concepts

---

### Additional Files

#### `loadData.R` (Legacy)
Earlier version of the R data processing script. Use `ExploreEEGData_1152026.R` for the updated Spring 2026 version with complete pipeline.

---

## 📖 Self-Paced Interactive Learning Guide

*The following sections provide additional self-paced learning materials to supplement the lectures. Work through these at your own pace to deepen your understanding.*

---

### Part 1: Understanding the Healthcare Context

Before we jump into code, let me share why this week's material is so important. In healthcare AI, we don't just build models - we work with real patient data, real signals from the human body, and real decisions that can impact lives.

This week, we're working with **EEG (Electroencephalography) data** - brain signals that can help detect seizures. Why is this important? Because automated seizure detection can:
- Alert caregivers to dangerous seizure events
- Help neurologists diagnose epilepsy
- Enable better treatment monitoring
- Potentially save lives through early intervention

---

### Part 2: The Healthcare Challenge

Imagine a patient in an intensive care unit, connected to multiple monitors tracking heart rate, brain activity, oxygen levels, and more. Every second, these devices generate thousands of data points. Now imagine trying to manually watch all these signals for hundreds of patients. Impossible, right?

This is where AI comes in. But before we can build intelligent systems to help doctors and nurses, we need to understand:
1. How to work with different programming languages
2. How to process and visualize data
3. How to extract meaningful patterns from signals

---

### Part 3: Setting Up Your Workspace

#### Option A: Google Colab (Easiest - No installation needed!)
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account
3. Click "New Notebook"
4. You're ready to code! 🎉

#### Option B: Local Setup with VS Code (Recommended for the course)
1. **Install Python**: Download from [python.org](https://python.org)
2. **Install VS Code**: Download from [code.visualstudio.com](https://code.visualstudio.com)
3. **Install Extensions**:
   - Python (by Microsoft)
   - Jupyter (by Microsoft)
   - GitHub Copilot
4. **Install R**: Download from [r-project.org](https://www.r-project.org/)
5. **Install MATLAB**: Use your Pitt credentials at [matlab.mathworks.com](https://matlab.mathworks.com)
6. **Install Miniconda**: Use GitHub Copilot agent mode to install automatically

#### Option C: Cloud Alternatives
- **RStudio Cloud** (for R): [posit.cloud](https://posit.cloud)
- **MATLAB Online**: [matlab.mathworks.com](https://matlab.mathworks.com)
- **Lightning AI**: [lightning.ai](https://lightning.ai)

---

### Part 4: Your First Python Program

**🎯 Learning Goal**: Generate synthetic patient data and save it as CSV

**File**: `generate_csv.py`

```python
import pandas as pd
import numpy as np
import sys

# Set random seed for reproducibility
np.random.seed(42)

# Generate 20 patients
n_patients = 20
data = {
    'age': np.random.randint(18, 80, n_patients),
    'height': np.random.uniform(150, 190, n_patients),
    'weight': np.random.uniform(50, 100, n_patients),
    'gender': np.random.choice(['M', 'F'], n_patients)
}

# Create DataFrame and save
df = pd.DataFrame(data)
filename = sys.argv[1] if len(sys.argv) > 1 else 'myFile.csv'
df.to_csv(filename, index=False)
print(f"✅ Generated {filename} with {n_patients} patients!")
```

**Try It Yourself:**
```bash
python generate_csv.py myFile.csv
```

---

### Part 5: Working with Real Medical Data - MATLAB

**🎯 Learning Goal**: Process EEG brain signals

**File**: `MATtoCSV.m`

The EEG data we're using:
- **Sampling Rate**: 500 Hz (500 measurements per second)
- **Duration**: 20 seconds (10,000 data points)
- **Signal**: Brain electrical activity in microvolts (μV)

**Key MATLAB Operations:**
```matlab
% Load the data
load('EEG_sleep.mat');

% Visualize
plot(s, 'b', 'LineWidth', 1.5);
title('EEG Signal - Full Resolution');

% Export to CSV
subset = s(1:10000);
csvwrite('s_of_t_subset.csv', subset);
```

---

### Part 6: Data Analysis with R

**🎯 Learning Goal**: Create complete preprocessing pipeline

**File**: `ExploreEEGData_1152026.R` (covered in Lecture 2)

**Key R Operations:**
```r
library(readr)
library(dplyr)

# Read CSV
s_of_t_subset <- read_csv("s_of_t_subset.csv", col_names = c("s"))

# Create time axis
s_of_t_subset <- s_of_t_subset %>% 
  mutate(time = (row_number() - 1) / 500)

# Normalize
s_of_t_subset$normalizedValue <- (s_of_t_subset$s - mean(s_of_t_subset$s)) / sd(s_of_t_subset$s)

# Add labels
s_of_t_subset$GT <- ifelse(s_of_t_subset$time > 12, "Seizure", "Normal")

# Visualize
plot(s_of_t_subset$time, s_of_t_subset$normalizedValue,
     main = "Normalized EEG Signal Depicting Seizure",
     xlab = "Time (seconds)", ylab = "Normalized Signal", type = "l")
```

---

## 🎯 Practice Exercises

### Exercise 1: Modify Python Script (Beginner)
Add a `bmi` column to the patient data:
- Formula: BMI = weight (kg) / (height (m))²
- Hint: height is in cm, divide by 100 first!

### Exercise 2: Explore MATLAB Visualization (Intermediate)
- Create a histogram of the EEG signal values
- Calculate basic statistics (mean, std, min, max)
- Export data with time column included

### Exercise 3: Enhance R Analysis (Advanced)
- Calculate rolling statistics (moving window mean and std)
- Create a simple classifier using amplitude thresholds
- Calculate accuracy, precision, and recall
- Plot ROC curve

---

## 🎯 Week 00-01 Challenges

### Challenge 1: The Patient Database (Beginner)
Create a more realistic patient dataset:
- Add patient ID, admission date, diagnosis codes
- Include missing values (realistic!)
- Generate 100 patients instead of 20
- **Bonus**: Add family relationships with correlated features

### Challenge 2: Seizure Detector (Intermediate)
Improve the seizure detection:
- Use multiple features for classification
- Evaluate performance with confusion matrix
- **Bonus**: Try different thresholds and plot ROC curve

### Challenge 3: Multi-language Pipeline (Advanced)
Create a complete automated pipeline:
1. Python: Generate synthetic EEG data with known seizure times
2. MATLAB: Process and visualize
3. R: Analyze, classify, and evaluate
4. Jupyter: Create report with all findings
- **Bonus**: Add command-line scripts to run entire pipeline automatically

---

## 📊 Learning Checkpoints

By the end of Week 00-01, you should be able to:

- [ ] Set up Python, R, and MATLAB environments
- [ ] Write basic scripts in all three languages
- [ ] Load and save data in CSV format
- [ ] Create visualizations (line plots, histograms, density plots)
- [ ] Understand sampling rates and signal processing basics
- [ ] Perform z-score normalization
- [ ] Calculate summary statistics
- [ ] Add ground truth labels to datasets
- [ ] Use Jupyter notebooks for interactive analysis
- [ ] Work with multi-language data pipelines

---

## 📚 Additional Resources

### Python Resources:
- [Python for Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) (Free online)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Tutorial](https://numpy.org/doc/stable/user/quickstart.html)

### R Resources:
- [R for Data Science](https://r4ds.had.co.nz/) (Free online)
- [ggplot2 Documentation](https://ggplot2.tidyverse.org/)
- [dplyr Documentation](https://dplyr.tidyverse.org/)

### MATLAB Resources:
- [MATLAB Onramp](https://www.mathworks.com/learn/tutorials/matlab-onramp.html) (Free tutorial)
- [Signal Processing Toolbox](https://www.mathworks.com/products/signal.html)

### Healthcare AI:
- [MIT HST.953: Collaborative Data Science in Medicine](https://www.youtube.com/playlist?list=PLUl4u3cNGP60B0PQXVQyGNdCyCTDU1Q5j)
- [Deep Learning in Medical Imaging](https://www.coursera.org/learn/ai-for-medical-diagnosis)

### Git & GitHub:
- [Git Tutorial](https://git-scm.com/doc)
- [GitHub Learning Lab](https://lab.github.com/)

---

## 🎬 Next Week Preview

**Week 02: Feature Engineering for Time Series**

We'll dive deeper into signal processing:
- Windowing techniques
- Feature extraction (statistical, frequency domain)
- Linear regression models
- Model evaluation

Get ready to build your first machine learning model! 🚀

---

## 💬 Discussion Forum

Have questions? Want to share your work? Use the discussion forum!

**This Week's Topics:**
1. Environment setup help
2. Code debugging
3. Share your modifications to the scripts
4. Discuss applications of AI in your area of interest

---

## 📝 Tasks

### Task 1: Environment Setup
- [ ] Install Python, R, and MATLAB (or set up cloud alternatives)
- [ ] Run all provided scripts successfully
- [ ] Take screenshots of outputs
- [ ] Submit a 1-page reflection on the setup process

### Task 2: Complete the EEG Pipeline
- [ ] Run `MATtoCSV.m` in MATLAB
- [ ] Run `ExploreEEGData_1152026.R` in R
- [ ] Review all generated CSV files
- [ ] Understand each step of the preprocessing pipeline

### Task 3: Code Modifications
- [ ] Complete Exercise 1 (Python - add BMI column)
- [ ] Complete Exercise 2 (MATLAB - histogram and stats)
- [ ] Complete Exercise 3 (R - enhanced analysis)

---

## 🙋 Frequently Asked Questions

**Q: I'm not a programmer. Can I still succeed in this course?**  
**A:** Absolutely! This course assumes no prior programming experience. We start from the basics and build up gradually. The key is to practice consistently and ask questions when you're stuck.

**Q: Which language should I focus on?**  
**A:** All three have their strengths! Python is the most popular for AI/ML, R is excellent for statistics, and MATLAB is powerful for signal processing. Learning all three makes you versatile.

**Q: Why do we use multiple languages?**  
**A:** In real-world healthcare AI, you'll often work with teams using different tools. MATLAB might be used for signal acquisition, R for statistical analysis, and Python for machine learning. Understanding all three makes you a better collaborator and problem-solver.

**Q: How much time should I spend on this course weekly?**  
**A:** Plan for 8-10 hours per week: 3 hours of lecture/guided work, 5-7 hours of practice and assignments.

**Q: I'm stuck on an exercise. What should I do?**  
**A:** 
1. Read the error message carefully
2. Check the documentation
3. Google the error (Stack Overflow is your friend!)
4. Ask in the discussion forum
5. Come to office hours

**Q: Can I use AI tools like ChatGPT or GitHub Copilot to help with coding?**  
**A:** Yes, with caveats! AI tools can be great for learning, but:
- Understand the code it generates
- Don't just copy-paste without learning
- You still need to explain your work
- On exams, you won't have AI assistance

---

## 🎓 Professor's Final Thoughts

Welcome to the journey! Healthcare AI is one of the most impactful applications of artificial intelligence today. Every skill you learn in this course can help save lives, improve patient care, and advance medical research.

Remember:
- **Everyone starts somewhere** - don't compare your beginning to someone else's middle
- **Errors are learning opportunities** - you'll spend more time debugging than writing code, and that's normal!
- **Collaboration is key** - help each other, share knowledge, grow together
- **Stay curious** - ask "why" and "what if" constantly

I'm excited to see what you'll create! Let's make this a fantastic semester.

**Professor Prahlad Menon, PhD, PMP**  
*Office Hours: Scheduled via email*  
*Email: prm44@pitt.edu*

---

*"In God we trust. All others must bring data." - W. Edwards Deming*

---

**🚀 Ready to begin? Start with Lecture 1 materials and work through the pipeline step-by-step!**
