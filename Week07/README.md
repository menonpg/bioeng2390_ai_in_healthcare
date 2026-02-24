# Week 07: Self-Paced Learning & Advanced Topics
### BIOENG 2390: AI in Healthcare - Spring 2026

**Instructor:** Professor Prahlad G. Menon, PhD, PMP  
**University of Pittsburgh, Department of Bioengineering**

---

## 🎯 Week 07 Overview

This week features **self-paced learning** with a take-home exercise on grokking and small transformers, plus supplemental reading materials on topics we've covered. No traditional lecture on Tuesday (Feb 24) - use the time to work on the exercise and complete pending assignments!

---

## 📚 Week 07 Structure

### Lecture 13 - February 24, 2026 (Self-Paced)
**Mode:** Quiz + Supplemental Exercise

📁 **Two Components:**

**1. Quiz Folder:** `Quiz/`
- **[Quiz 1: Fundamentals Through Dimensionality Reduction](Quiz/Quiz1_Fundamentals_Through_DimReduction.md)** ← Main assessment
  - 100 points (+ 20 bonus)
  - Covers Weeks 1-6 material
  - Open book, open notes
  - 6 sections: Feature Engineering, Classification, Overfitting, Cross-Validation, Dim Reduction, Bonus
  - **Bonus:** +20 points if submitted during class (5:30-7:00 PM)
  - **Deadline:** Before Thursday (Feb 26)

**2. Supplemental Exercise:** `Lecture13_SelfPaced/`
- **[Grokking Exercise](Lecture13_SelfPaced/Grokking_Exercise.md)** ← Additional learning
  - Read blog post on 777-parameter transformers
  - Clone and run repository
  - 3 reflection questions for discussion Thursday
  - Connects overfitting to memorization
  - Explores small model advantages

**What to Do:**
1. **Complete Quiz 1** (required)
2. **Explore Grokking** (prepares for Thursday discussion)
3. **Review supplemental PDFs** (optional)

---

### Lecture 14 - February 26, 2026
**Mode:** Discussion + Project Proposal Work

**Planned Topics:**
- Discussion of grokking insights
- Project proposal generation with Gen AI
- Advanced topics preview
- Q&A on pending assignments

---

## 📁 Folder Organization

### `Lecture13_SelfPaced/`
**Self-paced materials for Tuesday:**
- `Grokking_Exercise.md` - Main assignment with reflection questions

### `SupplementalReading/`
**Optional PDFs for deeper understanding:**
- `GMMvsKmeans.pdf` - Gaussian Mixture Models vs K-Means clustering
- `KMeansClustering.pdf` - K-Means algorithm details
- `LDA.pdf` - Linear Discriminant Analysis

**These PDFs supplement previous lectures:**
- GMM: Extends Week 05-06 clustering discussion
- K-Means: Deepens Week 06 understanding  
- LDA: Alternative to PCA (supervised dimensionality reduction)

### `Lecture14_Materials/`
**For Thursday's class:**
- Notebooks and materials for advanced topics (TBD)

---

## 🎯 Learning Objectives

By the end of this week, you will:

1. ✅ Understand the grokking phenomenon
2. ✅ Connect small model success to overfitting prevention
3. ✅ Draw analogies between memorization and overfitting
4. ✅ Apply N > P rule to transformer models
5. ✅ Understand trade-offs in model selection
6. ✅ Reference course concepts in new contexts
7. ✅ Make progress on project proposals

---

## 📝 Assignments Due

### Assignment 4: Project Proposal
**Due:** March 6, 2026 (FIRM DEADLINE!)

**Requirements:**
- One-page summary
- Team members listed
- Dataset identified and verified
- Hypothesis statement
- Specific aims
- Kanban board URL

**Timeline:**
- Feb 24 (today): Self-paced work
- Feb 26 (Thursday): Proposal generation with Gen AI
- Mar 6: Final submissions due

### Pending Assignments
**Complete before proposal:**
- Assignment 0: EEG_sleep.mat adaptation
- Assignment 1: Feature engineering
- Assignment 2: H2O modeling

---

## 🔑 Key Concepts to Review

**From the Grokking Exercise:**

**Overfitting = Memorization:**
- Training loss low, validation loss high
- Model learns training data patterns
- Fails to generalize to new data

**Small Models Prevent Overfitting:**
- Fewer parameters = less capacity to memorize
- Forces learning of general patterns
- 777 parameters < 1,000 observations = safe zone

**Grokking:**
- Phase 1: Quick memorization
- Phase 2: Slow generalization (the "grokking" moment)
- Extended training needed

**Connection to Our Course:**
- Week 02: Amplitude overfits, frequency generalizes
- Week 04: N > P rule, variance pizza
- Week 05-06: Explained variance, information loss

---

## 💡 Tips for Success

**For the Grokking Exercise:**
1. **Read carefully:** The blog post is dense but rewarding
2. **Run the code:** Seeing grokking happen is powerful
3. **Make connections:** Reference specific lectures
4. **Be specific:** "Week 04 Lecture 8 showed..." not "we learned..."
5. **Think critically:** Don't just summarize - analyze!

**For Project Proposals:**
1. **Dataset first:** Verify access before committing
2. **Start simple:** Can always extend later
3. **Be realistic:** 6 weeks is not long!
4. **Use what you know:** Build on course material

---

## 🙋 FAQ

**Q: Do I have to finish during class time for the bonus?**  
**A:** No! Bonus is optional. Full credit if submitted before Thursday.

**Q: How long should my answers be?**  
**A:** 1-2 paragraphs per sub-question. Quality over quantity.

**Q: Can I work with my project team?**  
**A:** Yes, but submit individually. Each person's answers should be unique.

**Q: What if I can't run the code?**  
**A:** Focus on the blog post and reflection questions. Code is optional (10 points).

**Q: Will this be on the final?**  
**A:** There is no final exam! This is a graded assignment.

---

## 📚 Supplemental Reading

**Located in `SupplementalReading/` folder:**

**GMM vs K-Means (GMMvsKmeans.pdf):**
- Gaussian Mixture Models = probabilistic K-Means
- When to use which
- Soft vs hard clustering

**K-Means Clustering (KMeansClustering.pdf):**
- Algorithm details
- Choosing K (elbow method)
- Initialization strategies

**LDA (LDA.pdf):**
- Supervised dimensionality reduction
- Comparison with PCA
- When to use LDA vs PCA

**Note:** These are supplemental - not required for the exercise!

---

**Enjoy the self-paced learning! See you Thursday for discussion!** 🚀
