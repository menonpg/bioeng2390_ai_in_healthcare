# Quiz 1: AI in Healthcare Fundamentals
## Covering Weeks 1-6: Fundamentals through Dimensionality Reduction

**Course:** BIOENG 2390 - Spring 2026  
**Instructor:** Professor Prahlad G. Menon, PhD, PMP  
**Date:** February 24, 2026  
**Format:** Open book, open notes  
**Time:** Self-paced (recommended 60-90 minutes)

**Bonus:** +20 points if completed and submitted during class time (5:30-7:00 PM)  
**Deadline:** Thursday, February 26, 2026 before class

---

## 📋 Instructions

1. Answer all questions to the best of your ability
2. You may reference lecture notes, notebooks, and online resources
3. Show your work and reasoning
4. Submit as PDF or Word document via Canvas
5. Include your name and date

**Total Points:** 100 (+ 20 bonus possible)

---

## Section 1: Feature Engineering (20 points)

### Question 1.1 (10 points)
**In Week 02, we discovered that raw EEG amplitude features failed to distinguish seizure from normal states (p=0.94 for t-test).**

**a) Explain WHY amplitude alone failed. Reference the box plots and statistical tests.**

**b) What frequency-domain features DID work? List them and explain why they were more effective.**

**c) Calculate: If you have a 256 Hz signal and use 1-second windows with 50% overlap, how many windows will you get from a 30-second recording? Show your work.**

---

### Question 1.2 (10 points)
**You're analyzing ECG data from two hospitals:**
- Hospital A: 500 Hz sampling rate, average heart rate 70 bpm
- Hospital B: 250 Hz sampling rate, average heart rate 90 bpm

**a) Explain the two-step process (resampling + segmentation) to standardize these signals for ML.**

**b) Why is upsampling preferred over downsampling? What is aliasing?**

**c) Describe the Fourier transform method for resampling. Why does it naturally prevent aliasing?**

---

## Section 2: Classification & Model Evaluation (25 points)

### Question 2.1 (10 points)
**Given this confusion matrix for a seizure detector:**

```
                ACTUAL
            Normal  Seizure
PREDICTED
Normal       850      120
Seizure       50      180
```

**Calculate:**
- a) Sensitivity (Recall)
- b) Specificity  
- c) Precision (PPV)
- d) Accuracy

**e) If this were a COVID test instead, where on the ROC curve would you set the threshold? Explain your reasoning.**

---

### Question 2.2 (8 points)
**Our GLM model achieved AUC = 0.81 with frequency features.**

**a) What does AUC = 0.81 mean? Is this good?**

**b) We selected threshold = 0.57 from cross-validation. What does this threshold represent?**

**c) Why use validation set for threshold selection instead of training set?**

---

### Question 2.3 (7 points)
**Derive the log-odds transformation (Week 03):**

**a) Start with logistic function: P(Y=1) = 1/(1 + e^(-(β₀ + β₁X)))**

**b) Show algebraic steps to arrive at: ln(P/(1-P)) = β₀ + β₁X**

**c) Why is this transformation necessary for logistic regression?**

---

## Section 3: Overfitting & Regularization (20 points)

### Question 3.1 (12 points)
**You have a gene expression dataset:**
- 50 patient samples
- 10,000 genes measured per patient
- Binary outcome: cancer vs normal

**a) Identify the overfitting risks. Reference the N > P rule from Week 04.**

**b) If you train a decision tree, what are THREE causes of overfitting in this scenario?**

**c) Propose TWO solutions to reduce overfitting. Be specific about implementation.**

---

### Question 3.2 (8 points)
**Explain the "variance pizza" analogy from Week 04 Lecture 8:**

**a) What does each pizza slice represent?**

**b) What is variance inflation? Why is it bad?**

**c) If Feature A and Feature B both explain slices 3, 4, and 5, what should you do?**

---

## Section 4: Cross-Validation & Model Selection (15 points)

### Question 4.1 (8 points)
**Compare cross-validation approaches:**

**a) Explain 5-fold cross-validation. How many models are trained?**

**b) What is leave-one-out cross-validation? When would you use it?**

**c) Why is cross-validation better than a single train/test split?**

---

### Question 4.2 (7 points)
**From Week 05, our KNN model showed:**
```
Fold  AUC
1     0.78
2     0.82  
3     0.70
4     0.85
5     0.72
```

**a) Calculate mean and standard deviation of AUC.**

**b) What does the high variation tell you about the model?**

**c) Would you trust this model in production? Why or why not?**

---

## Section 5: Dimensionality Reduction (20 points)

### Question 5.1 (10 points)
**Distinguish between structure and variance (Week 05-06):**

**a) Define "structure" in data. Give an example.**

**b) Define "variance" in data. Give an example.**

**c) Can you do supervised learning without structure? Can you do it without variance? Explain.**

**d) What is the core assumption of unsupervised learning? When does it fail?**

---

### Question 5.2 (10 points)
**PCA reduces 400 genes to 2 principal components.**

**Eigenvalues:** [120, 80, 30, 20, 15, 10, ...] (sum to 500)

**a) Calculate explained variance for PC0 and PC1.**

**b) Is this enough? Should you keep more components? Justify your answer.**

**c) What information is LOST when you project 3D → 2D? Be specific.**

**d) Why can't you directly apply a fitted t-SNE model to new data? How do you work around this?**

---

## Section 6: Practical Application (Bonus: +20 points)

### Complete ONE of the following:

**Option A: Run Grokking Code**
- Clone and run the 777-parameter transformer
- Include screenshot of training/validation curves
- Describe when "grokking" occurred
- Connect to overfitting concepts from Week 04

**Option B: Implement Mini-Project**
- Load EEG_sleep.mat
- Extract 2 frequency features (your choice)
- Build simple classifier (any method)
- Report AUC and confusion matrix
- Discuss: Did you overfit? How do you know?

---

## 📊 Grading Rubric

**Section 1 (Feature Engineering):** 20 points
- Correct calculations
- Clear explanations
- Connections to lecture material

**Section 2 (Classification):** 25 points
- Accurate metric calculations
- Understanding of ROC/thresholds
- Correct derivations

**Section 3 (Overfitting):** 20 points
- Identifies risks correctly
- Proposes viable solutions
- Demonstrates understanding

**Section 4 (Cross-Validation):** 15 points
- Explains methods accurately
- Interprets results correctly
- Makes sound judgments

**Section 5 (Dim Reduction):** 20 points
- Distinguishes structure vs variance
- Calculates correctly
- Explains limitations

**Bonus Section:** +20 points
- Demonstrates hands-on skills
- Connects theory to practice

---

## 💡 Tips for Success

1. **Reference specific lectures:** "In Week 04 Lecture 8..."
2. **Show your work:** Especially for calculations
3. **Be concise:** Quality over quantity
4. **Make connections:** Link concepts across weeks
5. **Use your notes:** This is open book!

---

## 🎯 Learning Outcomes Assessed

By completing this quiz, you demonstrate:
- ✅ Understanding of feature engineering principles
- ✅ Ability to calculate and interpret classification metrics
- ✅ Knowledge of overfitting causes and prevention
- ✅ Understanding of cross-validation purpose
- ✅ Grasp of dimensionality reduction concepts
- ✅ Ability to apply concepts to new scenarios

---

**Good luck! This quiz consolidates 6 weeks of intensive learning!** 📚🚀

**Remember:** +20 bonus if submitted during class today (Feb 24, 5:30-7:00 PM)
