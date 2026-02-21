# Assignment: Support Vector Machines (SVM)
**Course:** Python for Machine Learning  
**Total Points:** 25  

---

## 📚 Reference Resources

Before starting, review the following:

- 📄 [SVM Concepts](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_Support_Vector_Machine.md)
- 💻 [SVM Colab Notebook](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Colab_Notebooks/ML_Support_Vector_Machines.ipynb)
- 🧠 [SVM Quiz / Playbook](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Play/ML_Support_Vector_Machines.html)

---

## 📝 Instructions

Complete all tasks in a **Google Colab notebook**. Add a **Markdown cell** before each task with the task title and point value. Your notebook should be clean, organized, and well-commented.

Submit your Colab notebook link (make sure sharing is set to **"Anyone with the link can view"**).

---

## Task 1 — Load and Explore the Dataset (3 points)

Use the **Breast Cancer Wisconsin dataset** from `sklearn.datasets`.

- Load the dataset and create a DataFrame
- Print the shape, feature names, and target class names
- Display the first 5 rows
- Check for any missing values

```python
from sklearn.datasets import load_breast_cancer
```

**What to show:** DataFrame head, shape, class names, and missing value check.

---

## Task 2 — Preprocess the Data (3 points)

- Split the dataset into training (80%) and testing (20%) sets using `train_test_split` with `random_state=42`
- Apply **feature scaling** using `StandardScaler`
- Explain in a Markdown cell **why feature scaling is especially important for SVM**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

---

## Task 3 — Train and Evaluate SVM Models with Different Kernels (9 points)

Train **three separate SVM classifiers** using the following kernels:

| Kernel | Points |
|--------|--------|
| Linear | 3 pts  |
| RBF (Radial Basis Function) | 3 pts  |
| Polynomial | 3 pts  |

For **each kernel**, print:
- Accuracy score
- Confusion matrix
- Classification report (precision, recall, F1-score)

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

---

## Task 4 — Compare and Visualize Results (5 points)

- Create a **bar chart** comparing the accuracy scores of all three kernels
- Use `matplotlib` or `seaborn` for visualization
- Label the chart clearly (title, x-axis, y-axis)
- In a Markdown cell, answer: **Which kernel performed best and why do you think that is?**

---

## Task 5 — Tune Hyperparameters with GridSearchCV (5 points)

Use `GridSearchCV` to find the best combination of hyperparameters for the **RBF kernel SVM**.

- Search over: `C = [0.1, 1, 10, 100]` and `gamma = ['scale', 'auto', 0.01, 0.001]`
- Use `cv=5` (5-fold cross-validation)
- Print the best parameters and best cross-validation score
- Re-evaluate on the test set using the best estimator

```python
from sklearn.model_selection import GridSearchCV
```

---

## 📊 Grading Rubric

| Task | Description | Points |
|------|-------------|--------|
| Task 1 | Data loading and exploration | 3 |
| Task 2 | Preprocessing + scaling explanation | 3 |
| Task 3 | Three kernels trained and evaluated | 9 |
| Task 4 | Visualization + written comparison | 5 |
| Task 5 | GridSearchCV hyperparameter tuning | 5 |
| **Total** | | **25** |

---

## ✅ Submission Checklist

- [ ] Notebook runs top-to-bottom without errors
- [ ] Each task has a Markdown cell with task title
- [ ] All outputs (accuracy, confusion matrix, charts) are visible
- [ ] Written explanations are included for Task 2 and Task 4
- [ ] Colab link is shared with view access

---

*Good luck! Refer to the [SVM Colab Notebook](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Colab_Notebooks/ML_Support_Vector_Machines.ipynb) if you need a refresher on any concepts.*
