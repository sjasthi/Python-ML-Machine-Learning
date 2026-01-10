# Assignment: Decision Tree Classifier (Breast Cancer)

**Course:** Python for Machine Learning  
**Instructor:** Siva Jasthi (Siva.Jasthi@metrostate.edu)  
**Week:** 14  
**Points:** 25

---

## 📋 Assignment Overview

### Breast Cancer Decision Tree Classifier

**Objective:** Build a decision tree classifier to predict breast cancer diagnosis (malignant vs. benign)

**Dataset:** 
- [breast-cancer.csv](https://github.com/sjasthi/Python-DS-Data-Science/blob/main/datasets/breast-cancer.csv)
- Features: Various tumor measurements (radius, texture, perimeter, area, etc.)
- Target: diagnosis (M = Malignant, B = Benign)

---

## 📝 Requirements

### Part 1: Data Exploration (5 points)

1. Load the breast-cancer.csv dataset
2. Display basic information (shape, columns, data types)
3. Check for missing values
4. Display the class distribution (how many M vs. B)
5. Show summary statistics for features

---

### Part 2: Data Preparation (5 points)

1. Separate features (X) and target (y)
2. Encode the target variable if needed (M=1, B=0 or use LabelEncoder)
3. Split data into training (70%) and testing (30%) sets
4. Use random_state=42 for reproducibility

---

### Part 3: Build Decision Tree Models (8 points)

1. Create a baseline decision tree with default parameters
2. Train the model on training data
3. Make predictions on test data
4. Calculate and display:
   - Accuracy score
   - Confusion matrix
   - Classification report (precision, recall, f1-score)

5. Build improved models with different hyperparameters:
   - Model with max_depth=3
   - Model with max_depth=5
   - Model with max_depth=10
   - Model with min_samples_split=20
6. Compare accuracies and identify which performs best

---

### Part 4: Visualization (4 points)

1. Visualize your best decision tree using plot_tree()
   - Make it readable (large figure size)
   - Include feature names
   - Include class names (Malignant, Benign)
   - Use filled=True for colored boxes

2. Create a feature importance bar chart
   - Show top 10 most important features
   - Sort from highest to lowest importance

---

### Part 5: Analysis (3 points)

Write a brief analysis (4-6 sentences) addressing:
1. Which hyperparameter setting worked best and why?
2. What are the top 3 most important features for prediction?
3. Did you observe any overfitting? How can you tell?
4. How confident would you be using this model in a real medical setting?

---

## 📤 Submission Instructions

1. Complete all parts in a Google Colab notebook
2. Add markdown cells with explanations for each step
3. Ensure all outputs are visible
4. Share your notebook with proper permissions
5. Submit the notebook link to the assignment dropbox

**Due Date:** End of Week 14

---

## 📊 Grading Rubric

| Component | Points |
|-----------|--------|
| Data Exploration | 5 |
| Data Preparation | 5 |
| Model Building & Comparison | 8 |
| Visualization | 4 |
| Analysis | 3 |
| **TOTAL** | **25** |

---

## 💡 Tips for Success

- Follow the marketing_data notebook as a template
- Add comments explaining your code
- Use markdown cells to document your process
- Compare multiple models to find the best one
- Make visualizations clear and readable
- Think critically about overfitting

---

## 🎯 Learning Goals

By completing this assignment, you will:

✅ Practice loading and exploring real medical datasets  
✅ Build multiple decision tree models with different configurations  
✅ Compare model performance objectively  
✅ Visualize decision trees and feature importance  
✅ Analyze overfitting vs. underfitting  
✅ Apply ML to a real-world medical diagnosis problem  
✅ Develop critical thinking about model deployment  

---

## 📚 Resources

**Reference Materials:**
- [Decision Trees Presentation](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_Decision_Trees_aka_CART.pdf)
- [Marketing Data Notebook (Template)](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Colab_Notebooks/ML_Decistion_Tree_Classifier_Marketing.ipynb)
- [Breast Cancer Dataset](https://github.com/sjasthi/Python-DS-Data-Science/blob/main/datasets/breast-cancer.csv)

**Scikit-learn Documentation:**
- [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [plot_tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)

---

## ❓ Common Questions

**Q: How do I choose the best model?**  
A: Compare test accuracy scores. The model with the highest test accuracy (not training accuracy) is typically best.

**Q: What if my tree looks too big to visualize?**  
A: Use a larger figure size (e.g., `plt.figure(figsize=(25,15))`) or reduce max_depth.

**Q: Should I remove any features?**  
A: No, use all features initially. Feature importance will show which ones matter most.

**Q: What if I see 100% training accuracy?**  
A: That's overfitting! The model memorized the training data. Use hyperparameters to control this.

**Q: Can I use different hyperparameters than suggested?**  
A: Yes! Feel free to experiment, but make sure to include the required ones for comparison.

---

**Good luck! 🌳**

*Remember: Decision trees are powerful but can overfit easily. Your goal is to find the sweet spot between complexity and generalization!*
