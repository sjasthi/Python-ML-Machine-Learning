# Week 19: Naive Bayes Classifier

**Course:** Python for Machine Learning  
**Instructor:** Siva Jasthi (Siva.Jasthi@metrostate.edu)   
**Topic:** Naive Bayes Classifier

---

## 📚 Learning Objectives

By the end of this week, students will be able to:

1. Understand Bayes' Theorem and how it applies to classification
2. Explain why the algorithm is called "Naive" (feature independence assumption)
3. Walk through a Naive Bayes prediction step by step (prior, likelihood, posterior)
4. Identify the three types of Naive Bayes: Gaussian, Multinomial, and Bernoulli
5. Know when to use each type based on the data
6. Build and evaluate a Naive Bayes classifier using scikit-learn
7. Understand Laplace smoothing and why it's needed
8. Compare Naive Bayes performance against other classifiers learned so far

---

## 📖 Review Materials

👉 **Presentation:**  
[ML_Naive_Bayes_Classifier.md](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_Naive_Bayes_Classifier.md)

👉 **Colab Notebook:**  
[ML_Naive_Bayes_Classifier.ipynb](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Colab_Notebooks/ML_Naive_Bayes_Classifier.ipynb)


---

## 🔑 Key Vocabulary

| Term | Definition |
|------|-----------|
| **Bayes' Theorem** | A formula that calculates the probability of an event based on prior knowledge |
| **Prior Probability** | The initial probability of a class before seeing any features |
| **Likelihood** | The probability of seeing certain features given a specific class |
| **Posterior Probability** | The updated probability of a class after considering the features |
| **Naive Assumption** | The assumption that all features are independent of each other |
| **Gaussian NB** | Used when features are continuous numbers (e.g., height, temperature) |
| **Multinomial NB** | Used when features are counts or frequencies (e.g., word counts) |
| **Bernoulli NB** | Used when features are binary/yes-no (e.g., word present or not) |
| **Laplace Smoothing** | A technique to handle zero-probability problems by adding a small count |

---

## 🤔 Discussion Questions

1. Why is it called "Naive"? Is the independence assumption realistic?
2. Even though the assumption is rarely true, why does Naive Bayes still work well in practice?
3. If you were building a spam email filter, which type of Naive Bayes would you choose and why?
4. How does Naive Bayes compare to Decision Trees in terms of speed and interpretability?
5. Can you think of a situation where Naive Bayes would perform poorly?

---

## 🧭 Where Does Naive Bayes Fit?

| Algorithm | Type | Key Idea |
|-----------|------|----------|
| Linear Regression | Regression | Best fit line |
| Logistic Regression | Classification | Probability via sigmoid |
| KNN | Classification/Regression | Nearest neighbors vote |
| K-Means | Clustering (Unsupervised) | Group by distance to centroids |
| Decision Tree | Classification/Regression | If-then rules (splits) |
| Random Forest | Classification/Regression | Ensemble of many trees |
| **Naive Bayes** | **Classification** | **Probability using Bayes' Theorem** |

---

## 💡 Success Tips

- 🧮 **Practice the math by hand** — Working through one example manually helps you understand what scikit-learn does behind the scenes
- 📊 **Compare classifiers** — Try running the same dataset through Decision Tree, Random Forest, AND Naive Bayes to see which performs better
- 🔤 **Think about data types** — The biggest decision with Naive Bayes is choosing the right variant (Gaussian vs. Multinomial vs. Bernoulli) based on your features
- ⚡ **Speed advantage** — Naive Bayes is one of the fastest classifiers — great for large datasets and real-time predictions
