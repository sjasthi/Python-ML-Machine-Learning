# Week 18: Cross Validation

**Course:** Python for Machine Learning  
**Program:** Learn and Help (www.learnandhelp.com)  
**Instructor:** Siva R Jasthi

---

## 📚 Topics Covered

- Introduction to Cross Validation
- Training, Validation, and Test Sets
- Understanding Overfitting
- Types of Cross Validation:
  - Holdout Method
  - K-Fold Cross Validation
  - Stratified K-Fold Cross Validation
  - Leave-One-Out Cross Validation (LOOCV)
  - Leave-P-Out Cross Validation
  - Time Series Cross Validation

---

## 🎯 Learning Objectives

By the end of this week, you will be able to:
- Explain what Cross Validation is and why it's important
- Understand how Cross Validation addresses overfitting
- Implement different types of Cross Validation using scikit-learn
- Choose the appropriate CV method for different types of data
- Apply K-Fold Cross Validation to evaluate machine learning models

---

## 📖 Course Materials

### 📊 Presentation (Markdown)
[ML_Cross_Validation.md](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_Cross_Validation.md)

### 💻 Hands-On Notebook
[ML_Cross_Validation.ipynb](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Colab_Notebooks/ML_Cross_Validation.ipynb)

### 🎞️ PowerPoint Slides
[ML_Cross_Validation.pptx](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_Cross_Validation.pptx)

### 🎥 Recommended Video
**StatQuest: Cross Validation Explained**  
[https://www.youtube.com/watch?v=fSytzGwwBVw](https://www.youtube.com/watch?v=fSytzGwwBVw)  
*Excellent visual explanation by Josh Starmer*

---

## ✅ Activities

### 📝 Quiz (10 points)
[Cross Validation Interactive Quiz](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Play/ML_cross_validation_playbook_n_quiz.html)
- 10 multiple-choice questions
- Immediate feedback
- Take a screenshot and submit to Google Classroom

### 📊 Assignment (25 points)
[K-Fold Cross Validation with Decision Trees](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Assignments/Python_ML_K-Fold_Cross_Validation_with_Decision_Trees.md)
- Apply K-Fold CV to a scikit-learn toy dataset
- Test K values from 2 to 10
- Visualize and analyze results
- Submit Google Colab notebook (.ipynb file)

---

## 🔑 Key Concepts

- **Overfitting** - When a model memorizes training data instead of learning patterns
- **K-Fold CV** - Splitting data into K equal parts and rotating which part is used for validation
- **Stratified K-Fold** - Preserves class distribution in each fold (for imbalanced data)
- **Time Series CV** - Respects temporal order (never shuffle time series data!)
- **Model Generalization** - How well a model performs on unseen data

---

## 💡 Important Reminders

✅ **DO:**
- Always use Cross Validation for model evaluation
- Choose the right CV method based on your data type
- Use stratification for imbalanced datasets
- Respect temporal order for time series data
- Set `random_state` for reproducibility

❌ **DON'T:**
- Use holdout method on small datasets
- Use regular K-Fold on time series data
- Shuffle time series data
- Forget to look at both mean and standard deviation of CV scores

---

## 🎓 Success Criteria

You'll know you've mastered this week's content when you can:
- [ ] Explain why Cross Validation is better than a simple train-test split
- [ ] Implement K-Fold CV using scikit-learn
- [ ] Choose the appropriate CV method for different scenarios
- [ ] Interpret Cross Validation results and make decisions about model performance

---

**Happy Learning! 🚀**

*Last Updated: February 2026*
