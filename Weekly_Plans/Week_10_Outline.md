# Week 10 Outline - Machine Learning Fundamentals

**Course:** Python for Machine Learning  
**Week:** 10  
**Topics:** Feature Engineering, Logistic Regression, ML Metrics & Performance Measures  

---

## 📚 Learning Objectives

By the end of this week, you will be able to:
- Understand and apply feature engineering techniques to improve model performance
- Implement and interpret Logistic Regression models for binary classification
- Evaluate model performance using appropriate metrics and confusion matrices
- Understand the trade-offs between different performance measures
- Apply learned concepts to real-world classification problems

---

## 📖 Materials to Review

### 1. Presentation: Feature Engineering
📊 **Review the Feature Engineering presentation:**  
[ML_Feature_Engineering.md](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_Feature_Engineering.md)

**What to focus on:**
- Feature scaling and normalization techniques
- Creating new features from existing data
- Feature selection methods


**Why it matters:**
Feature engineering is often the difference between a mediocre and an excellent model. Good features can dramatically improve your model's performance.

---

### 2. Presentation: Logistic Regression
📊 **Review the Logistic Regression presentation:**  
[ML_Logistic_Regression.md](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_Logistic_Regression.md)

**What to focus on:**
- The sigmoid function (S-curve) and how it works
- Differences between linear and logistic regression
- Binary classification concepts
- Probability interpretation
- Decision boundaries
- Implementation with scikit-learn

**Key Concepts:**
- Understanding the logit function
- How probabilities are converted to class predictions
- The role of the threshold (typically 0.5)

---

### 3. Presentation: ML Metrics and Performance Measures
📊 **Review the ML Metrics presentation:**  
[ML_Metrics_and_Performance_Measures.md](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_Metrics_and_Performance_Measures.md)

**What to focus on:**
- Accuracy, Precision, Recall, and F1-Score
- When to use each metric
- Understanding the confusion matrix
- ROC curves and AUC scores
- Trade-offs between metrics
- Imbalanced dataset considerations

**Critical Understanding:**
Different business problems require different metrics. For example:
- Medical diagnosis: High recall might be crucial
- Spam detection: High precision might be more important
- Balanced problems: F1-score provides a good balance

---

### 4. Notebook: Python ML with Logistic Regression
💻 **Work through the Jupyter notebook:**  
[ML_Logistic_Regression.ipynb](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Notebooks/ML_Logistic_Regression.ipynb)

**How to use it:**
- Download or open in Google Colab
- Run each cell sequentially
- Pay attention to the data preprocessing steps
- Observe how the model is trained and evaluated
- Experiment with different parameters
- Try modifying the code to deepen understanding

---

## 🎮 Interactive Learning Tool

### Play with the Confusion Matrix
🎯 **Interactive Confusion Matrix Tool:**  
[ML_Metrics_Confusion_Matrix.html](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_Metrics_Confusion_Matrix.html)

**Activities:**
- Explore how changes in True Positives, False Positives, True Negatives, and False Negatives affect metrics
- See real-time calculations of Accuracy, Precision, Recall, and F1-Score
- Understand the relationships between different metrics
- Experiment with imbalanced datasets scenarios
- Visualize how different thresholds affect model performance

**Learning Goals:**
- Develop intuition for metric trade-offs
- Understand when each metric is most important
- See the impact of class imbalance on metrics

---

## ✅ What's Due This Week?

### 📝 Assignment 8: Logistic Regression
**Due Date:** Thursday, November 21, 2025 at 11:59 PM  
**Points:** TBD

**Assignment Details:**  
[Python_ML_Assignment_Logistic_Regression.md](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Assignments/Python_ML_Assignment_Logistic_Regression.md)

**What to expect:**
- Implement a complete logistic regression pipeline
- Perform feature engineering on a dataset
- Train and evaluate your model
- Calculate and interpret performance metrics
- Create visualizations of model performance
- Write a brief analysis of your results

**Deliverables:**
- Colab notebook with complete code and markdown explanations
- All cells must run without errors
- Clear visualizations and interpretations
- Performance metrics and confusion matrix

---


## 💡 Tips for Success

### Before You Start:
- Review all three presentations thoroughly
- Make sure you understand the confusion matrix components
- Have a clear understanding of what metrics mean
- Set up your development environment (Jupyter/Colab)

### While Learning:
- **Don't just read** - run the code examples yourself
- **Experiment** with the interactive confusion matrix tool
- **Take notes** on when to use different metrics
- **Ask questions** in the discussion board if concepts are unclear
- **Compare** logistic regression to linear regression

### For the Assignment:
- Start early - don't wait until Thursday night
- Read the assignment requirements carefully
- Test your code incrementally
- Use meaningful variable names
- Add comments explaining your approach
- Visualize your results
- Interpret what your metrics mean in context

### Common Pitfalls to Avoid:
- ❌ Not scaling features when necessary
- ❌ Using accuracy alone for imbalanced datasets
- ❌ Forgetting to split train/test data properly
- ❌ Not interpreting what the metrics mean
- ❌ Submitting code with errors

---

## 🎯 Key Concepts This Week

### Feature Engineering
- **Scaling:** StandardScaler, MinMaxScaler, RobustScaler
- **Encoding:** One-Hot Encoding, Label Encoding, Ordinal Encoding
- **Creation:** Polynomial features, interaction terms, binning
- **Selection:** Correlation analysis, feature importance

### Logistic Regression
- **Sigmoid Function:** Maps any real number to probability [0,1]
- **Decision Boundary:** Threshold (usually 0.5) for classification
- **Coefficients:** Indicate feature importance and direction
- **Probability Interpretation:** Understanding model confidence

### Performance Metrics
- **Accuracy:** Overall correctness (good for balanced datasets)
- **Precision:** How many predicted positives are actually positive
- **Recall (Sensitivity):** How many actual positives were caught
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Complete picture of classification results
---

## 🔬 Practical Application Examples

### Feature Engineering in Action:
- Creating age groups from continuous age data
- Extracting day/month/year from datetime
- Combining height and weight into BMI
- Creating interaction terms (e.g., age × income)

### Logistic Regression Use Cases:
- Email spam detection (spam vs. not spam)
- Disease diagnosis (positive vs. negative)
- Customer churn prediction (will churn vs. won't churn)
- Credit default prediction (will default vs. won't default)

### Metric Selection:
- **Medical Testing:** High recall to catch all positive cases
- **Email Spam:** High precision to avoid false positives
- **Balanced Dataset:** Accuracy might be sufficient
- **Imbalanced Dataset:** F1-score or AUC-ROC

---

## 📚 Additional Resources

- **Scikit-learn Documentation:** [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- **Confusion Matrix Calculator:** Use the interactive tool provided
- **Feature Engineering Guide:** Review presentation materials
- **Metric Selection Guide:** Review performance measures presentation

---

Good luck with your machine learning journey! 🚀🤖
