# Assignment: K-Fold Cross Validation with Decision Trees

**Course:** Python for Machine Learning  
**Program:** Learn and Help (www.learnandhelp.com)  
**Instructor:** Siva R Jasthi  
**Total Points:** 25 points  
**Due Date:** [To be announced]

---

## 📚 Learning Objectives

By completing this assignment, you will:
- Apply K-Fold Cross Validation to real-world datasets
- Understand how the number of folds (K) affects model performance
- Build and evaluate Decision Tree classification models
- Visualize and interpret cross-validation results
- Make data-driven decisions about hyperparameter selection

---

## 📋 Assignment Overview

In this assignment, you will use a toy dataset from scikit-learn to explore how different values of K in K-Fold Cross Validation affect the accuracy of a Decision Tree classifier. You'll determine which K value gives the best performance and present your findings with visualizations.

---

## 🎯 Instructions

### Step 1: Select a Classification Dataset (3 points)

Choose **ONE** of the following classification datasets from scikit-learn:

- **Iris Dataset** (`load_iris`) - Classify iris flower species (3 classes, 150 samples)
- **Wine Dataset** (`load_wine`) - Classify wine varieties (3 classes, 178 samples)  
- **Breast Cancer Dataset** (`load_breast_cancer`) - Classify tumors as malignant/benign (2 classes, 569 samples)

**Requirements:**
- Load the dataset using scikit-learn
- Display basic information about the dataset (number of samples, features, classes)
- Split the data into features (X) and target (y)

**Example Code Structure:**
```python
from sklearn.datasets import load_iris
# Load dataset
data = load_iris()
X = data.data
y = data.target
```

---

### Step 2: Build a Decision Tree Classifier (5 points)

Create a Decision Tree model that will be used for classification.

**Requirements:**
- Import `DecisionTreeClassifier` from sklearn
- Create an instance of the classifier
- Use `random_state=42` for reproducibility

**Example Code Structure:**
```python
from sklearn.tree import DecisionTreeClassifier

# Create model
model = DecisionTreeClassifier(random_state=42)
```

---

### Step 3: Apply K-Fold Cross Validation (8 points)

Test different values of K (number of folds) to see how they affect model accuracy.

**Requirements:**
- Test K values from **2 to 10** (that's 9 different K values)
- Use `cross_val_score` from sklearn.model_selection
- For each K value, calculate the **mean accuracy** across all folds
- Store the results in a list or dictionary for later visualization

**Example Code Structure:**
```python
from sklearn.model_selection import cross_val_score

# Test different K values
k_values = range(2, 11)  # K from 2 to 10
mean_accuracies = []

for k in k_values:
    scores = cross_val_score(model, X, y, cv=k, scoring='accuracy')
    mean_accuracy = scores.mean()
    mean_accuracies.append(mean_accuracy)
    print(f"K={k}: Mean Accuracy = {mean_accuracy:.4f}")
```

---

### Step 4: Find the Optimal K (4 points)

Determine which K value gives the highest mean accuracy.

**Requirements:**
- Identify the best K value
- Report the corresponding accuracy
- Print a clear statement of your findings

**Example Output:**
```
Best K value: 5
Best Mean Accuracy: 0.9667
```

---

### Step 5: Create Visualizations (5 points)

Visualize your results to make them easy to understand.

**Requirements:**
- Create a **line plot** showing:
  - X-axis: K values (2 to 10)
  - Y-axis: Mean Accuracy
  - Mark the best K value with a different color or marker
- Include appropriate:
  - Title (e.g., "K-Fold Cross Validation: K vs Accuracy")
  - Axis labels
  - Grid (optional but recommended)
- Create a **table** showing K values and their corresponding accuracies

**Example Code Structure:**
```python
import matplotlib.pyplot as plt

# Line plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, mean_accuracies, marker='o', linestyle='-', linewidth=2)
plt.xlabel('Number of Folds (K)', fontsize=12)
plt.ylabel('Mean Accuracy', fontsize=12)
plt.title('K-Fold Cross Validation: K vs Accuracy', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 📊 Grading Rubric (25 Points)

| Component | Points | Criteria |
|-----------|--------|----------|
| **Dataset Selection** | 3 | ✅ Correctly loads a scikit-learn classification dataset<br>✅ Displays dataset information (samples, features, classes)<br>✅ Properly separates X and y |
| **Decision Tree Model** | 5 | ✅ Correctly imports DecisionTreeClassifier<br>✅ Creates model instance with random_state=42<br>✅ Code is clean and well-commented |
| **K-Fold Cross Validation** | 8 | ✅ Tests K values from 2 to 10 (2 points)<br>✅ Uses cross_val_score correctly (2 points)<br>✅ Calculates mean accuracy for each K (2 points)<br>✅ Prints results for all K values (2 points) |
| **Optimal K Identification** | 4 | ✅ Correctly identifies best K value (2 points)<br>✅ Reports corresponding accuracy (1 point)<br>✅ Clear statement of findings (1 point) |
| **Visualizations** | 5 | ✅ Line plot with correct axes (2 points)<br>✅ Proper labels, title, and formatting (1 point)<br>✅ Table showing K values and accuracies (2 points) |

---

## 💡 Bonus Opportunities (+3 Extra Credit)

Want to go above and beyond? Try these extensions:

1. **Compare Multiple Datasets** (+1 point)
   - Run the same experiment on 2 different datasets
   - Compare which K works best for each dataset

2. **Add Standard Deviation** (+1 point)
   - Calculate and display standard deviation for each K
   - Add error bars to your plot

3. **Compare with Stratified K-Fold** (+1 point)
   - Run the experiment using StratifiedKFold
   - Compare results with regular K-Fold
   - Explain which one performs better and why

---

## 📤 Submission Requirements

**What to Submit:**
- A Google Colab notebook (.ipynb file) containing:
  - All your code with comments
  - All outputs (print statements, visualizations, tables)
  - A brief conclusion paragraph (3-5 sentences) answering:
    - Which K value worked best for your dataset?
    - Why do you think this K value performed well?
    - Did you notice any patterns as K increased?

**How to Submit:**
1. Make sure your Google Colab notebook runs from top to bottom without errors
2. Download your notebook as a .ipynb file: `File → Download → Download .ipynb`
3. Submit the .ipynb file to Google Classroom
4. **File naming convention:** `LastName_FirstName_KFold_CV.ipynb`

**Before Submitting - Checklist:**
- [ ] My code runs without errors
- [ ] All outputs are visible (don't clear outputs!)
- [ ] My visualizations have proper labels and titles
- [ ] I've included comments explaining my code
- [ ] I've answered the conclusion questions
- [ ] My file is named correctly

---

## 🔧 Helpful Tips

### Getting Started with Google Colab
```python
# First cell - Install required packages (if needed)
# Usually sklearn, matplotlib, numpy are pre-installed in Colab

# Import all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
```

### Common Issues and Solutions

**Problem:** "cross_val_score is giving weird results"  
**Solution:** Make sure you're passing the correct parameters: `cross_val_score(model, X, y, cv=k)`

**Problem:** "My plot looks messy"  
**Solution:** Use `plt.figure(figsize=(10, 6))` to make it bigger and easier to read

**Problem:** "I don't know which dataset to choose"  
**Solution:** Start with Iris - it's the simplest with only 150 samples and 3 classes

### Want to Create a Nice Table?

```python
import pandas as pd

# Create a results dataframe
results_df = pd.DataFrame({
    'K Value': list(k_values),
    'Mean Accuracy': mean_accuracies
})
print(results_df)
```

---

## 📖 Review These Concepts

Before starting, make sure you understand:
- ✅ What is Cross Validation and why we use it
- ✅ How K-Fold Cross Validation works
- ✅ What overfitting means
- ✅ How to interpret accuracy scores
- ✅ The bias-variance tradeoff with different K values

**Reference Materials:**
- Course notes on K-Fold Cross Validation
- Scikit-learn documentation: https://scikit-learn.org/stable/modules/cross_validation.html
- StatQuest Cross Validation video: https://www.youtube.com/watch?v=fSytzGwwBVw

---

## ❓ Frequently Asked Questions

**Q: Which K value should I expect to be best?**  
A: There's no single "correct" answer! It depends on your dataset. Generally, K=5 or K=10 works well, but your results might differ.

**Q: Should I use the entire dataset for cross-validation?**  
A: Yes! Cross-validation uses all the data by rotating through different train/test splits.

**Q: Do I need to manually split my data into train/test?**  
A: No! `cross_val_score` handles all the splitting automatically.

**Q: Can I use a different dataset not listed?**  
A: Stick to the three listed datasets for this assignment. They're specifically chosen to work well with this task.

**Q: How long should this take?**  
A: Most students complete this in 1-2 hours. Don't rush - take time to understand what you're doing!

**Q: Can I work with a partner?**  
A: You can discuss concepts with classmates, but everyone must submit their own unique notebook.

---

## 🎓 Learning Outcomes Assessment

After completing this assignment, you should be able to:
- [ ] Explain what K-Fold Cross Validation is and why it's important
- [ ] Implement K-Fold CV using scikit-learn
- [ ] Interpret cross-validation results
- [ ] Make informed decisions about hyperparameter selection
- [ ] Present data science findings visually

---

## 🌟 Example Expected Output

Your notebook should show something like:

```
Dataset: Iris
Number of samples: 150
Number of features: 4
Number of classes: 3

Testing K-Fold Cross Validation:
K=2: Mean Accuracy = 0.9400
K=3: Mean Accuracy = 0.9533
K=4: Mean Accuracy = 0.9467
K=5: Mean Accuracy = 0.9600
K=6: Mean Accuracy = 0.9533
K=7: Mean Accuracy = 0.9600
K=8: Mean Accuracy = 0.9467
K=9: Mean Accuracy = 0.9533
K=10: Mean Accuracy = 0.9667

Best K value: 10
Best Mean Accuracy: 0.9667

[Line plot showing the trend]
[Table showing all K values and accuracies]

Conclusion:
For the Iris dataset, K=10 gave the highest accuracy of 96.67%. 
This makes sense because with more folds, each fold gets a better 
representation of all three iris species...
```

---

## 📞 Getting Help

If you're stuck:
1. Review the course materials on Cross Validation
2. Check the scikit-learn documentation
3. Post your question in the class discussion forum
4. Attend office hours
5. Email: Siva.Jasthi@metrostate.edu

**Remember:** The goal is to learn, not just to get the right answer. Understanding the concepts is more important than getting a perfect score!

---

## 🏆 Success Criteria

You'll know you've succeeded when:
- ✅ Your code runs without errors
- ✅ You can explain why different K values give different accuracies
- ✅ Your visualizations clearly show the relationship between K and accuracy
- ✅ You can answer: "Which K would you recommend for this dataset and why?"

---

**Good luck! 🚀**

*Last Updated: February 2026*
