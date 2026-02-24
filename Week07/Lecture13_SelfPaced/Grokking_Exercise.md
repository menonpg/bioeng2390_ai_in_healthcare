# Lecture 13: Self-Paced Learning - Grokking & Small Transformers
## February 24, 2026

**Mode:** Self-paced (take-home or in-class)  
**Bonus:** +20 points for completing during class time  
**Deadline:** Submit before Lecture 14 (Thursday, Feb 26) for full points

---

## 📚 Reading Assignment

### Understanding Grokking with Small Transformers

**Required Reading:**  
[Grokking a 777-Parameter Transformer](https://blog.themenonlab.com/blog/grokking-777-parameter-transformer/)

**What You'll Learn:**
- How tiny transformers (777 parameters) can learn arithmetic
- The phenomenon of "grokking" (sudden generalization after long training)
- Relationship between model size, memorization, and generalization
- Why small models reduce overfitting risk

---

## 💻 Hands-On Exercise

### Clone and Explore the Repository

**Instructions:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/menonpg/grokking_small_transformer.git
   cd grokking_small_transformer
   ```

2. **Install dependencies:**
   ```python
   pip install torch numpy matplotlib
   ```

3. **Run the training script:**
   ```python
   python train.py
   ```

4. **Observe:**
   - Training loss vs validation loss over time
   - The "grokking" moment when validation suddenly improves
   - How a 777-parameter model learns modular addition

5. **Experiment:**
   - Try different numbers of parameters
   - Compare memorization vs generalization
   - Observe when overfitting occurs

---

## 🤔 Reflection Questions

**Answer these questions based on the reading and your understanding from Weeks 1-6:**

### Question 1: Parameters and Overfitting

**Consider the following scenarios:**

**Scenario A:** 1,000-parameter model, 10,000 training examples  
**Scenario B:** 10,000-parameter model, 1,000 training examples  
**Scenario C:** 777-parameter model, 1,000 training examples

**a) Which scenario has the highest overfitting risk? Why?**

**b) Connect this to the "curse of dimensionality" from Week 04 Lecture 8. How does the N (observations) > P (parameters) rule apply?**

**c) In the grokking blog post, why does a SMALL transformer (777 parameters) succeed where larger models might fail for the same task?**

---

### Question 2: Memorization vs Generalization

**The grokking phenomenon shows two phases:**
1. **Phase 1:** Model memorizes training data quickly (low training loss)
2. **Phase 2:** Model suddenly generalizes (low validation loss) after many more epochs

**a) Draw an analogy between "memorization" in neural networks and "overfitting" in traditional ML models. How are they similar?**

**b) In Week 02, we found that using raw EEG amplitude features led to overfitting (train accuracy high, test accuracy low). In Week 03, frequency features generalized better. Connect this to the grokking phenomenon: What allowed the model to transition from memorization to generalization?**

**c) Why do small models (like 777 parameters) reduce overfitting risk compared to billion-parameter models? Reference the "variance pizza" analogy from Week 04 in your answer.**

---

### Question 3: Practical Implications

**You're designing a seizure detection system for clinical deployment.**

**Option A:** Use a 1-billion parameter transformer (like GPT-3 scale)  
**Option B:** Use a 777-parameter small transformer (like the blog post)  
**Option C:** Use logistic regression with frequency features (like our Week 03 GLM)

**You have 500 labeled EEG recordings.**

**a) Which approach would you choose and why? Consider:**
- Overfitting risk (N vs P)
- Interpretability needs
- Deployment constraints (edge devices)
- Generalization requirements

**b) How does the concept of "explained variance" from PCA (Week 05-06) relate to preventing overfitting? If your features explain 95% of variance in training data, is that good or bad for generalization?**

**c) The blog post shows grokking happens after extended training. In clinical ML, we often can't wait millions of epochs. What strategies from our course (Week 03-04) help models generalize faster?**
- Hint: Think cross-validation, regularization, ensemble methods

---

## 📝 Submission Requirements

**Submit via Canvas:**

1. ✅ **Reflection Questions Document** (PDF or Word)
   - Answer all 3 questions with sub-parts
   - 1-2 paragraphs per sub-question
   - Reference specific lectures and concepts from course
   - Include your own insights and connections

2. ✅ **Code Screenshot** (optional but encouraged)
   - Show you ran the grokking repository
   - Include training/validation loss curves
   - Note any experiments you tried

3. ✅ **Key Takeaways** (1 paragraph)
   - What surprised you about grokking?
   - How does this change your understanding of overfitting?
   - What will you apply to your final project?

---

## 🎯 Grading

**Total:** 100 points

**Distribution:**
- Question 1: 30 points
- Question 2: 30 points
- Question 3: 30 points
- Code exploration: 10 points (optional)
- Bonus: +20 points if submitted during class (Feb 24, 5:30-7:00 PM)

**Criteria:**
- Demonstrates understanding of course concepts
- Makes connections across lectures
- Shows critical thinking
- References specific examples from class
- Clear, well-organized writing

---

## 💡 Discussion Preview (Thursday, Feb 26)

**Be prepared to discuss:**
- Your biggest insight from the grokking phenomenon
- How small models can outperform large ones
- Connections to our course material (Weeks 1-6)
- Implications for your final project

**Bring questions!** This will be an open discussion, not a lecture.

---

## 🔗 Resources

**Required:**
- [Grokking Blog Post](https://blog.themenonlab.com/blog/grokking-777-parameter-transformer/)
- [GitHub Repository](https://github.com/menonpg/grokking_small_transformer)

**Recommended (from our course):**
- Week 04 Lecture 8: Overfitting, curse of dimensionality, variance pizza
- Week 03 Lectures 5-6: Feature engineering success, model comparison
- Week 05 Lecture 10: Explained variance, information loss

---

**Good luck! This is a fascinating topic that ties together everything we've learned!** 🧠✨
