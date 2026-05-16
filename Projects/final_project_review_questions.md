# Final Project – Industry Expert Q&A Preparation Guide

**Learn and Help | Python Machine Learning Course | Academic Year 2025–2026**

> This guide prepares students for questions from industry experts during final project demonstrations (May 16–23, 2026). Questions are grouped by theme and span the full year's curriculum. For each question, a brief hint is included to help you frame your answer.

---

## Section 1: ML Foundations & Problem Framing
*Weeks 1–6 | Core concepts, types of ML, bias/variance, overfitting*

**Q1. Why did you choose this ML approach for your project? Could you have solved the same problem a different way?**

> *Hint: Explain whether your problem is supervised, unsupervised, or something else — and why that guided your algorithm choice. Show you understand there are always tradeoffs.*

---

**Q2. How did you decide what your features (inputs) and predictions (outputs) are? Did you consider any features you decided NOT to use?**

> *Hint: Feature selection matters. Talk about what data you collected, what you kept, and what you dropped — and why.*

---

**Q3. Did your model overfit or underfit the data? How did you know, and what did you do about it?**

> *Hint: Compare training accuracy vs. test accuracy. If they were very different, you likely overfit. Explain what steps you took to fix it.*

---

**Q4. What is the cost function your model is trying to minimize, and why does that matter?**

> *Hint: Every ML model is trying to get "less wrong" with every iteration. Explain what measure of error your model uses (e.g., mean squared error for regression, log loss for classification).*

---

**Q5. Walk me through how you split your data into training, validation, and test sets. Why is that split important?**

> *Hint: The test set is data your model has never seen — it's the "real-world simulation." Explain why peeking at test data early would be cheating.*

---

## Section 2: Algorithms – Supervised Learning
*Weeks 8–11, 14–20 | Regression, KNN, Decision Trees, Random Forest, Naive Bayes, SVM*

**Q6. If you used regression, how do you interpret the model's coefficients? What story do they tell about your data?**

> *Hint: A coefficient tells you: "for every one unit increase in this feature, the prediction changes by this much." Make it concrete with a number from your project.*

---

**Q7. KNN is called a "lazy learner." Can you explain what that means, and how you picked the right value of K?**

> *Hint: "Lazy" means KNN doesn't actually learn a model — it memorizes the data and looks things up at prediction time. Talk about how you tested different K values and what happened.*

---

**Q8. Explain how a Decision Tree makes a decision. How does it know which feature to split on first?**

> *Hint: Decision Trees pick the split that creates the most "pure" groups — groups where most examples belong to the same class. Mention concepts like Gini impurity or information gain if you used them.*

---

**Q9. Why is a Random Forest generally more accurate than a single Decision Tree?**

> *Hint: Think of it like asking 100 different experts and taking a vote instead of trusting just one expert. The diversity of trees reduces the chance of one bad decision derailing everything.*

---

**Q10. How does a Support Vector Machine (SVM) draw a boundary between classes? What is the "margin" and why does it matter?**

> *Hint: SVM tries to find the widest possible "street" between two groups of data points. A wider margin means the model is more confident and generalizes better to new data.*

---

**Q11. Naive Bayes makes a big assumption about your data. What is it, and when does that assumption hold up in the real world?**

> *Hint: It assumes all features are completely independent of each other — which is rarely true, but the algorithm still works surprisingly well for things like spam detection and text classification.*

---

## Section 3: Algorithms – Unsupervised Learning
*Weeks 12–13 | K-Means Clustering*

**Q12. In K-Means, how does the algorithm decide which cluster a data point belongs to? What does it mean for the algorithm to "converge"?**

> *Hint: Each point gets assigned to its nearest cluster center. After reassigning all points, the center moves to the middle of the new group. This repeats until the centers stop moving — that's convergence.*

---

**Q13. How did you choose the number of clusters (K) in your project? Did you use the Elbow Method?**

> *Hint: Explain the Elbow Method — you plot the error for different values of K and look for the point where adding more clusters stops helping much. Relate this to your actual project.*

---

## Section 4: Model Evaluation & Validation
*Week 18 | Cross-Validation; Weeks 5–6 | Evaluation Metrics*

**Q14. What accuracy did your model achieve, and is that number actually good? How do you know?**

> *Hint: Accuracy alone can be misleading — if 95% of your data is one class, a model that always predicts that class is 95% "accurate" but useless. Talk about precision, recall, F1-score, or a confusion matrix if you used them.*

---

**Q15. What is cross-validation, and why is it a more trustworthy evaluation than a single train/test split?**

> *Hint: Cross-validation rotates which portion of data is used for testing, so every data point gets to be in the test set at least once. It gives you a more reliable picture of real-world performance.*

---

**Q16. What was the biggest source of error in your model? If you had more time, what would you do to improve it?**

> *Hint: This is your chance to show you understand your model's weaknesses. Talk honestly about data quality, class imbalance, feature choices, or algorithm limitations.*

---

## Section 5: Advanced Topics
*Weeks 22–24 | PCA, Recommender Systems, CNNs*

**Q17. If you used PCA, can you explain what it does in plain language? How did you decide how many components to keep?**

> *Hint: PCA compresses many features into fewer "summary" features while losing as little information as possible. You can use a "scree plot" or aim to keep 90–95% of the variance.*

---

**Q18. How does a Convolutional Neural Network "see" an image? What is a filter, and what does pooling do?**

> *Hint: Filters slide across the image looking for patterns (edges, shapes, textures) — like Instagram filters but for finding features. Pooling shrinks the image down so the model focuses on the big picture, not every pixel.*

---

**Q19. If you built a recommender system, how does it know what to recommend to a new user who has no history?**

> *Hint: This is called the "cold start problem." Explain how your system handles it — maybe it recommends popular items, or asks the new user a few questions first.*

---

## Section 6: Tools, Data, and Real-World Considerations
*Week 7 | scikit-learn; Week 21 | Kaggle & Hugging Face*

**Q20. Where did your data come from, and how did you check that it was good quality?**

> *Hint: Talk about how you handled missing values, outliers, or imbalanced classes. Clean data is often more important than a fancy algorithm.*

---

**Q21. Did you use any pre-trained models or datasets from Hugging Face or Kaggle? Why is using pre-trained models so common in industry?**

> *Hint: Training large models from scratch is expensive and slow. Pre-trained models let you benefit from thousands of hours of compute work done by someone else — you just fine-tune them for your specific problem.*

---

**Q22. How would you deploy your model so a real person could use it? What would that look like?**

> *Hint: Think beyond the notebook — a web app, a mobile app, an API. You don't need to have built it, but show you've thought about the "last mile" of putting ML into the world.*

---

## Section 7: Ethics, Impact & Big Picture Thinking
*Applies across the entire course*

**Q23. Could your model make unfair or biased predictions against any group of people? How would you detect that?**

> *Hint: ML models learn from historical data — if that data reflects past biases, the model will too. Talk about what fairness means in the context of your specific project and how you'd test for it.*

---

**Q24. If your model gets a prediction wrong, what is the worst thing that could happen? How did that risk affect your design choices?**

> *Hint: A false positive and a false negative can have very different consequences. For example, a spam filter deleting an important email vs. letting through a phishing attack — which is worse? Relate this to your project.*

---

**Q25. What is one thing you learned this year that surprised you the most about Machine Learning?**

> *Hint: This is your chance to be authentic. Experts love curiosity and self-awareness. Talk about a moment where your expectations were wrong — maybe a simple algorithm outperformed a complex one, or clean data mattered more than the algorithm.*

---

## Quick Reference: Topics by Week

| Question(s) | Topic | Weeks Covered |
|-------------|-------|---------------|
| Q1–Q5 | ML Foundations, Core Concepts | 1–6 |
| Q6 | Linear / Logistic Regression | 8–9 |
| Q7 | K-Nearest Neighbors | 10–11 |
| Q8–Q9 | Decision Trees, Random Forest | 14–17 |
| Q10 | Support Vector Machines | 20 |
| Q11 | Naive Bayes | 19 |
| Q12–Q13 | K-Means Clustering | 12–13 |
| Q14–Q16 | Model Evaluation, Cross-Validation | 6, 18 |
| Q17 | PCA | 22 |
| Q18 | CNNs / Image Classification | 24 |
| Q19 | Recommender Systems | 23 |
| Q20–Q22 | Data, Tools, Deployment | 7, 21 |
| Q23–Q25 | Ethics, Impact, Reflection | All |

---

*Prepared for Learn and Help | www.learnandhelp.com | May 2026*
