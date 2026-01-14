# BIOENG-2390 Spring 2026 - Lecture 1
## January 13, 2026

**Instructor:** Professor Prahlad G. Menon, PhD, PMP  
**Email:** menon.prahlad@gmail.com  
**Recording:** [View on Fathom](https://fathom.video/share/B2xPtF4xDdwWipoxVLtfx2KXH4utRqze)  
**Duration:** 81 minutes

---

## 📋 Lecture Overview

Today's class focused on:
1. Course setup and logistics (Canvas, Zoom, Google Drive, GitHub)
2. Introduction to AI in Healthcare through real-world examples
3. Setting up development environments (VS Code, Git, Python, Miniconda)
4. Understanding key ML concepts through practical examples
5. First hands-on coding with Jupyter notebooks

---

## 🎯 Key Concepts Introduced

### 1. **Response Variable**
The variable you are trying to predict in your model.
- **Example:** Operative mortality (whether a patient survives surgery)

### 2. **Predictive Variables (Features)**
The variables used to predict the response.
- **Examples:** Age, gender, laboratory values, cardiac function, surgical history

### 3. **Prevalence**
The underlying ratio in which a certain outcome exists in a dataset.
- **Example:** If operative mortality happens in 2% of cases, the natural prevalence is 2%
- **Important:** A risk score should be compared to prevalence, not just to 50%

### 4. **Bias**
Systematic error introduced into your model by selection or other factors.
- **Selection Bias:** Occurs when your data sample doesn't represent the population
- **Example:** Using only patients from one hospital vs. diverse hospitals nationwide

### 5. **Dataset**
A collection of data that may or may not be representative of the population.
- **Stratified Random Sample:** Better representation of key variations
- **Convenience Sample:** Easier to collect but potentially biased

---

## 💡 Real-World Example: AATS Risk Calculator

### What is it?
The **American Association for Thoracic Surgery (AATS) Risk Calculator** is a tabular data-driven AI system for pre-surgical planning in cardiovascular procedures.

**Website:** [riskcalculator.aatsqualitygateway.org](http://riskcalculator.aatsqualitygateway.org)

### Dataset
- **50,000+ patients** with known outcomes
- Cardiovascular surgical procedures
- Data collected over 5+ years through AATS Quality Gateway

### Eight Risk Models
1. Operative Mortality
2. Mortality and Morbidity
3. Long Hospital Stays (>14 days)
4. Extended Intubation
5. Deep Sternal Wound Infection
6. Stroke
7. Reoperation
8. Renal Failure

### Key Insight: Understanding Risk Scores

**Question:** Is a 6% mortality risk high or low?
- **Average risk** across all patients: 6.15%
- **Natural prevalence** of mortality: 1-2%
- **Interpretation:** 6% is higher than natural prevalence (concerning), but average for the dataset

**ROC Analysis (Receiver Operating Characteristic):**
- Small differences in risk scores can be significant
- Compare predicted risk to natural prevalence, not arbitrary thresholds
- A 0.36% risk when prevalence is 2% is actually good news!

---

## 🗂️ Course Structure

### Data Types We'll Analyze
1. **Signal Data (1D):** Time series - EEG, ECG, vital signs
2. **Tabular Data (2D):** Excel-like data with records and attributes
3. **Image Data:** Medical imaging - X-rays, MRI, CT scans
4. **Text Data:** Clinical notes, research papers

### Course Format
- **Project-Based:** Work in teams on real healthcare data
- **Assignments:** Individual/collaborative work building to final assignment
- **Remote + In-Person:** Mostly Zoom classes with some in-person sessions
- **Tools:** Python, R, MATLAB, multiple IDEs

---

## 🛠️ Development Environment Setup

### Tools Installed Today

#### 1. **Visual Studio Code (VS Code)**
- Main IDE for the course
- Download: [code.visualstudio.com](https://code.visualstudio.com)
- **Key Shortcut:** `Cmd/Ctrl + Shift + P` - Command palette
- **Theme:** Can switch between light/dark modes

#### 2. **GitHub & Git**
- **Version control** for code and collaboration
- **GitHub Education:** Free benefits for students
- **Apply:** [github.com/education](https://github.com/education)
- **Repository:** [github.com/menonpg/bioeng2390_ai_in_healthcare](https://github.com/menonpg/bioeng2390_ai_in_healthcare)
- **Branch:** Spring2026

#### 3. **GitHub Copilot**
- AI coding assistant (free for students via GitHub Education)
- **Extensions to install:**
  - GitHub Copilot
  - GitHub Copilot Chat
  - Cline (optional but helpful)

#### 4. **Homebrew (Mac Only)**
- Package manager for Mac
- Install from [brew.sh](https://brew.sh)
- Use to install Git: `brew install git`

#### 5. **Miniconda**
- Python package manager
- Installs Python environments
- Use GitHub Copilot to install: "Control the terminal and install Miniconda on my system"

---

## 📝 Step-by-Step Setup Guide

### Part 1: Visual Studio Code Setup

```bash
# 1. Download and install VS Code
# 2. Open VS Code
# 3. Press Cmd/Ctrl + Shift + P
# 4. Type "Preferences: Color Theme" to change theme
```

### Part 2: GitHub Account & Repository

```bash
# 1. Create GitHub account (if you don't have one)
# 2. Apply for GitHub Education benefits
# 3. Create a dedicated folder for the class
# 4. Open folder in VS Code
# 5. Open Terminal (Terminal > New Terminal)
# 6. Clone the repository:

git clone https://github.com/menonpg/bioeng2390_ai_in_healthcare.git
cd bioeng2390_ai_in_healthcare

# 7. Switch to Spring2026 branch:
# Click on bottom-right where it says "main"
# Select "Spring2026" branch
```

### Part 3: Python Environment Setup

**Using GitHub Copilot Agent:**
1. Open Copilot Chat (agent mode)
2. Ask: "Control the terminal and install Miniconda on my system. Then conda activate base environment."
3. Let Copilot download and install Miniconda
4. Verify: You should see `(base)` prefix in terminal

**Manual verification:**
```bash
conda --version  # Check Conda is installed
python --version # Check Python is installed
```

### Part 4: Running Your First Notebook

1. Open `Week00-01/myFirstNotebook.ipynb`
2. Select kernel: Click "Select Kernel" → "Python Environments" → "base"
3. Install ipykernel if prompted
4. Run cells with Shift + Enter or click play button
5. Install pandas/numpy if needed: `pip install pandas numpy`

---

## 🌐 Cloud Development Environments

### Google Colab
- **URL:** [colab.research.google.com](https://colab.research.google.com)
- **Pros:** Zero setup, free GPU access, cloud storage
- **Setup:** 
  - Install "Collaboratory" app in Google Drive (New → More → Connect More Apps)
  - Create shortcuts to course folder in your Drive
  - Open .ipynb files directly in Colab

**GPU Access:**
- Runtime → Change runtime type → Hardware accelerator → GPU
- Free T4 GPU available

### Lightning AI
- **URL:** [lightning.ai](https://lightning.ai)
- **Pros:** VS Code-like interface, free credits, multiple GPU options (T4, L4, A100)
- **Features:** Built-in AI copilot, cloud storage, collaborative coding

**When to use cloud IDEs:**
- Local machine lacks disk space
- Need GPU for deep learning
- Want to avoid local environment setup hassles
- Working on large datasets

---

## 📊 Today's Code Example: Generating Patient Data

### What it does:
Creates a CSV file with synthetic patient data (age, height, weight, gender) using Python and Pandas.

### The Code:
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

### Key Concepts:
- **Pandas DataFrame:** Like Excel in Python - rows and columns
- **NumPy:** Fast numerical operations
- **Random seed (42):** Makes "random" results reproducible
- **sys.argv:** Command-line arguments

---

## 🤖 Introduction to RAG (Retrieval Augmented Generation)

### What is RAG?
A technique that combines:
1. **Knowledge Base:** Indexed documents/data
2. **Embeddings:** Fixed-length numerical representations of text
3. **Vector Database:** Stores embeddings for similarity search
4. **LLM:** Generates answers based on retrieved context

### How it Works:
1. Documents are converted to embeddings (vectors)
2. User query is converted to same embedding space
3. **Dot product** finds most similar documents (cosine similarity)
4. Retrieved documents sent to chatbot
5. Chatbot generates factual answer based on retrieved data

### Connection to Linear Algebra:
- Vectors and dot products are fundamental
- Cosine similarity measures vector similarity
- This is why we'll study linear algebra in this course!

### Example:
The AATS Risk Calculator bot uses RAG to answer questions about the quality gateway system using indexed documentation.

---

## 🎓 Course Resources

### Google Drive
- **Access:** Via Canvas → Lecture 1, Lecture 2 module
- **Contents:**
  - Lecture notes and recordings
  - Assignments folder
  - Class roster and project teams
  - Supplementary materials

**Important:** Create a shortcut to course folder in your Drive!
1. Navigate to course folder
2. Click folder name → Organize → Add Shortcut
3. Add to your personal Drive folder
4. This allows Colab to access files

### GitHub Repository
- **Main Branch:** Last year's content (read-only)
- **Spring2026 Branch:** This semester's content (active)
- **Structure:** Week-by-week folders with code and assignments

---

## 💻 Platform-Specific Notes

### Mac Users
- Use Homebrew for package management
- Terminal commands work directly
- GitHub Copilot installation straightforward

### Windows Users
- May need to enable PowerShell execution policies
- Git Bash recommended for Git operations
- Miniconda installation can be trickier
- **Tip:** Let GitHub Copilot handle the setup!

**Common Windows issue:**
```powershell
# If "git" command not recognized:
# 1. Install Git from git-scm.com
# 2. Restart VS Code
# 3. Verify: git --version
```

---

## 🔑 Key Takeaways

1. **AI in Healthcare is Real:** AATS Risk Calculator saves lives through better surgical planning
2. **Data Quality Matters:** Bias and selection affect model performance
3. **Understand Your Metrics:** Risk scores must be interpreted relative to prevalence
4. **Modern Development is Assisted:** Use AI tools (Copilot) to ease setup pain
5. **Multiple IDEs Available:** Local (VS Code) and Cloud (Colab, Lightning AI)
6. **Version Control is Essential:** Git/GitHub for collaboration and tracking changes

---

## 📝 For Next Class

### To Do Before Thursday:
1. ✅ Complete VS Code setup
2. ✅ Install Git and clone repository
3. ✅ Switch to Spring2026 branch
4. ✅ Install Miniconda and Python environment
5. ✅ Run `myFirstNotebook.ipynb` successfully
6. ✅ Set up Google Colab access
7. ✅ Create shortcuts to course Drive folder
8. ✅ Apply for GitHub Education benefits

### Come Prepared to Discuss:
- Any setup challenges you faced
- Questions about Git workflow
- Initial thoughts on project ideas

### Next Topics:
- Data aggregation techniques
- Working with EEG signal data
- Linear algebra fundamentals
- Feature engineering for ML models

---

## 🙋 Common Questions from Class

**Q: Do we have to use all three IDEs (VS Code, Colab, Lightning AI)?**  
A: VS Code is primary. Use cloud IDEs when you need GPU or have limited local resources.

**Q: Why is Windows setup harder than Mac?**  
A: Windows has different security policies and path configurations. Using Copilot helps!

**Q: How much does GitHub Copilot cost?**  
A: Free for students via GitHub Education. Otherwise $10/month (worth it!).

**Q: What if I can't get Miniconda to work?**  
A: Use Google Colab for now - zero setup required. We'll troubleshoot your local setup.

**Q: Is the Spring2026 branch read-only?**  
A: Yes, for course materials. Create your own repos for personal projects.

**Q: When do we form project teams?**  
A: Soon! Will be updated in class roster spreadsheet.

---

## 🔗 Important Links

- **Canvas:** Course main page
- **Zoom:** Check Canvas for link (same link all semester)
- **Fathom Notes:** Auto-transcription of lectures
- **GitHub Repo:** https://github.com/menonpg/bioeng2390_ai_in_healthcare
- **Google Drive:** Link in Canvas
- **AATS Risk Calculator:** http://riskcalculator.aatsqualitygateway.org
- **VS Code:** https://code.visualstudio.com
- **GitHub Education:** https://education.github.com
- **Homebrew:** https://brew.sh
- **Google Colab:** https://colab.research.google.com
- **Lightning AI:** https://lightning.ai

---

## 📚 Additional Context

### About Professor Menon's Research
- Board member of American Association for Thoracic Surgery (AATS)
- Editor for Journal of Cardiovascular Thoracic Surgery (JTCVS)
- 15+ years in cardiovascular and thoracic surgery research
- Focus: Diagnostics, surgical planning, image-guided surgery, patient-specific care

### Course Philosophy
*"The goal isn't just to teach theory - it's to make you confident practitioners who can apply AI to improve patient outcomes."*

### On Using AI Tools for Learning
- **Encouraged:** Use ChatGPT, Copilot, etc. to help understand concepts
- **Required:** You must understand the code and be able to explain it
- **Exams:** No AI assistance allowed - must demonstrate knowledge

---

**Next Class:** Thursday (remote via Zoom)  
**Office Hours:** TBA  
**Questions?** Post in Canvas discussion or email professor

---

*"Every skill you learn in this course can help save lives, improve patient care, and advance medical research."*

— Professor Prahlad G. Menon, PhD, PMP
