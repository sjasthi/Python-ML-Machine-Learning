# Logistic Regression Assignment - Model Building and Evaluation

**Course:** Python for Data Science  
**Due Date:** Thursday, November 21, 2025  
**Points:** 25

---

## Learning Objectives

By completing this assignment, you will:
- Load and explore a real-world dataset from scikit-learn
- Build and train a logistic regression classification model
- Make predictions using your trained model
- Evaluate model performance using confusion matrix
- Calculate and interpret classification metrics (accuracy, precision, recall, F1-score)
- Understand the relationship between confusion matrix and model performance

---

## Dataset: Breast Cancer Wisconsin

For this assignment, you will use the **Breast Cancer Wisconsin** dataset from scikit-learn. This is a binary classification dataset where the goal is to predict whether a tumor is malignant (cancerous) or benign (non-cancerous) based on various features.

**Dataset Information:**
- **Samples:** 569 cases
- **Features:** 30 numerical features computed from cell nuclei images
- **Target:** Binary (0 = malignant, 1 = benign)
- **Task:** Predict tumor diagnosis

**How to load the dataset:**
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

---

## Assignment Requirements

Create a Jupyter notebook named **`assignment_logistic_regression.ipynb`** that includes the following components:

### Required Sections

Your notebook should be organized with clear markdown headers and contain these sections:

1. **Data Loading and Exploration**
2. **Data Preparation**
3. **Model Building**
4. **Model Evaluation**
5. **Confusion Matrix Analysis**

---

## Part 1: Data Loading and Exploration (5 points)

**Tasks:**
- Load the breast cancer dataset from scikit-learn
- Display the dataset description and feature names
- Create a pandas DataFrame with the features and target
- Display basic statistics (shape, info, describe)
- Check for missing values

**What to include:**
- Code to load and explore the data
- Output showing dataset characteristics
- Brief observations about the data

---

## Part 2: Data Preparation (3 points)

**Tasks:**
- Split the data into training and testing sets (use 80-20 or 70-30 split)
- Use `train_test_split` from scikit-learn
- Set a random state for reproducibility (e.g., `random_state=42`)
- Display the shape of training and testing sets

**What to include:**
- Code for splitting the data
- Print statements showing the sizes of train/test sets

---

## Part 3: Model Building (5 points)

**Tasks:**
- Import LogisticRegression from scikit-learn
- Create a logistic regression model instance
- Train (fit) the model on the training data
- Make predictions on the test data

**What to include:**
- Code to create and train the logistic regression model
- Code to generate predictions on test set
- Brief comment explaining what the model is doing

---

## Part 4: Model Evaluation (7 points)

**Tasks:**
- Calculate the confusion matrix using `confusion_matrix` from scikit-learn
- Display the confusion matrix in a readable format
- Visualize the confusion matrix using a heatmap (use seaborn or matplotlib)
- Calculate and display the following metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

**What to include:**
- Confusion matrix output (both raw numbers and visualization)
- All four classification metrics with clear labels
- Use `classification_report` for comprehensive metrics

**Hint:** You can use `classification_report` from scikit-learn to get all metrics at once!

---

## Part 5: Confusion Matrix Analysis (5 points)

**Tasks:**
Write a markdown cell that answers these questions based on your confusion matrix:

1. **What do the four values in the confusion matrix represent?**
   - True Positives (TP)
   - True Negatives (TN)
   - False Positives (FP)
   - False Negatives (FN)

2. **Model Performance Analysis:**
   - How many cases did the model predict correctly?
   - How many cases did the model predict incorrectly?
   - Which type of error is more critical in cancer diagnosis (False Positive or False Negative)? Why?

3. **Metrics Interpretation:**
   - What does the accuracy score tell you?
   - What does precision mean in this context?
   - What does recall mean in this context?
   - Why might you care more about recall than precision in medical diagnosis?

**What to include:**
- A markdown cell with clear, thoughtful answers to all questions
- Reference specific numbers from your confusion matrix
- Demonstrate understanding of the trade-offs between metrics

---

## Grading Rubric

| Section | Points | Criteria |
|---------|--------|----------|
| **Data Loading & Exploration** | 5 | Dataset loaded correctly, exploration complete, clear output |
| **Data Preparation** | 3 | Proper train-test split with correct parameters |
| **Model Building** | 5 | Model created, trained, and predictions generated correctly |
| **Model Evaluation** | 7 | Confusion matrix displayed and visualized, all metrics calculated |
| **Confusion Matrix Analysis** | 5 | Thoughtful answers demonstrating understanding of metrics |
| **Total** | **25** | |

---

## Submission Instructions

1. Create your notebook named `assignment_logistic_regression.ipynb`
2. Include all five required sections with clear markdown headers
3. Ensure all code cells run without errors
4. Add comments to explain your code
5. Include your written analysis in markdown cells
6. Test your notebook: **Restart kernel and run all cells**
7. Submit your notebook through the course submission portal

**Submission Deadline:** Thursday, November 21, 2025 at 11:59 PM

---

## Starter Code Template

Here's a template to help you get started:

```python
# Section 1: Data Loading and Exploration
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

# Load the dataset
data = load_breast_cancer()

# Your exploration code here...

# Section 2: Data Preparation
from sklearn.model_selection import train_test_split

# Your data splitting code here...

# Section 3: Model Building
from sklearn.linear_model import LogisticRegression

# Your model building code here...

# Section 4: Model Evaluation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Your evaluation code here...

# Section 5: Analysis (use markdown cell)
```

---

## Tips for Success

💡 **Before you start:**
- Review logistic regression concepts from class
- Understand what a confusion matrix represents
- Familiarize yourself with scikit-learn documentation

💡 **While coding:**
- Run code cells frequently to catch errors early
- Print intermediate results to verify your work
- Use meaningful variable names
- Add comments to explain your logic

💡 **For the confusion matrix:**
- Remember: rows represent actual values, columns represent predictions
- Use `sns.heatmap()` for a nice visualization
- Label your axes clearly (Actual vs Predicted)

💡 **For the analysis:**
- Think about real-world implications
- In medical diagnosis, which error is worse?
- Connect metrics back to the confusion matrix values

💡 **Before submitting:**
- Restart kernel and run all cells from top to bottom
- Check that all outputs are displayed
- Verify your analysis answers all questions
- Proofread your written responses

---

## Required Libraries

Make sure you have these libraries installed:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

---

## Resources

- **Scikit-learn Logistic Regression:** [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- **Confusion Matrix:** [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- **Classification Metrics:** [https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- **Breast Cancer Dataset:** [https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

---

## Key Concepts to Remember

**Logistic Regression:**
- Used for binary classification problems
- Predicts probability of belonging to a class
- Output is 0 or 1 (using 0.5 threshold by default)

**Confusion Matrix:**
```
                  Predicted
                  0       1
Actual    0      TN      FP
          1      FN      TP
```

**Metrics:**
- **Accuracy** = (TP + TN) / Total
- **Precision** = TP / (TP + FP) - "Of all positive predictions, how many were correct?"
- **Recall** = TP / (TP + FN) - "Of all actual positives, how many did we catch?"
- **F1-Score** = Harmonic mean of precision and recall

---

## Questions?

If you have questions about this assignment:
- Review course materials on logistic regression
- Check the scikit-learn documentation
- Post on the course discussion board
- Attend office hours
- Email through Learn and Help platform

---

**Good luck with your classification model!** 🤖📊🔬
