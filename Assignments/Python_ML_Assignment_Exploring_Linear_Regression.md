# Assignment 1: Exploring Linear Regression with Scikit-Learn

**Course:** Python Machine Learning  
**Points:** 25  
**Due Date:** [Insert Due Date]

---

## Learning Objectives

By completing this assignment, you will:

1. Understand the fundamentals of Ordinary Least Squares (OLS) linear regression
2. Practice implementing linear regression using scikit-learn
3. Develop skills in data visualization for model evaluation
4. Compare model performance across different features and dataset configurations
5. Analyze the impact of training/test split on model performance

---

## Prerequisites

- Python 3.x
- Libraries: `scikit-learn`, `numpy`, `matplotlib`, `pandas`

Install required packages if needed:
```bash
pip install scikit-learn numpy matplotlib pandas
```

---

## Part 1: Study and Run the Linear Regression Example (8 points)

### Task 1.1: Reproduce the Example (8 points)

1. Visit the scikit-learn documentation: [OLS and Ridge Regression Example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols_ridge.html#sphx-glr-auto-examples-linear-model-plot-ols-ridge-py)

2. Create a Python script or Jupyter notebook that reproduces the **first part** of the example (Linear regression on the diabetes dataset)
   - Data loading and preparation (2 points)
   - Training the Linear Regression model (2 points)
   - Model evaluation with MSE and R² score (2 points)
   - Visualization of results (2 points)

3. Run the code and ensure you get similar results to the documentation

4. Add comments to your code explaining what each section does

**Deliverable:** Your code output with plots displayed and proper documentation

---

## Part 2: Modify and Experiment (10 points)

### Task 2.1: Explore Different Features (4 points)

Train separate Linear Regression models using different features from the diabetes dataset:

1. Train models using feature indices 2, 5, and 8 (3 points)
   - For each feature, calculate MSE and R² score
   - Create a comparison table showing which feature gives the best performance
   
2. Generate plots for all three features (1 point)
   - Create side-by-side visualizations showing train and test predictions

**Deliverable:** Code, comparison table, and visualizations

### Task 2.2: Experiment with Test Set Size (3 points)

Investigate how the train/test split affects model performance:

1. Try different test set sizes: `test_size=10`, `test_size=20`, and `test_size=50` (2 points)
   - Use the same feature (index 2) for consistency
   - Calculate MSE and R² score for each split
   - Create a table summarizing your results

2. Document your observations (1 point)
   - How does test set size affect MSE and R² score?
   - What patterns do you notice?

**Deliverable:** Code with results table and written observations

### Task 2.3: Make Predictions on New Data (3 points)

Apply your trained model to new data points:

1. Create 5 new data points within the range of your training data (1 point)
   - Document the values you chose and why

2. Use your trained model to make predictions on these points (1 point)
   - Print the predicted values

3. Visualize these predictions on your plot (1 point)
   - Mark them differently from training/test points

**Deliverable:** Code showing new data points, predictions, and updated visualization

---

## Part 3: Analysis and Understanding (7 points)

Answer the following questions in your submission. Each question should be answered in 3-5 sentences with clear explanations.

### Question 1 (2 points)
How does Linear Regression (OLS) find the best-fit line? Explain the concept of minimizing squared errors and why we use "squared" errors rather than just errors.

### Question 2 (2 points)
What does the R² score tell you about model performance? What range of values can it take, and what do they mean? Give an example of what an R² of 0.0, 0.5, and 1.0 would indicate.

### Question 3 (2 points)
Compare MSE values across different features (from Task 2.1). Why might one feature be better than another for prediction? What characteristics might make a feature more predictive?

### Question 4 (1 point)
Based on your experiments with different test set sizes (Task 2.2), what trade-offs do you observe between training set size and model evaluation? Which test set size would you recommend and why?

---

## Submission Guidelines

Submit a single ZIP file containing:

1. **Code file(s):** Python script (.py) or Jupyter notebook (.ipynb)
2. **Report document:** PDF or Markdown file with:
   - All plots and visualizations
   - Comparison tables from Tasks 2.1 and 2.2
   - Answers to all analysis questions from Part 3
3. **README:** Brief description of your files and how to run your code

**Naming convention:** `LastName_FirstName_Assignment1_LinearRegression.zip`

---

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| **Part 1** | 8 | Code runs successfully, reproduces example with proper documentation |
| **Task 2.1** | 4 | Models trained on different features with comparison and visualizations |
| **Task 2.2** | 3 | Test set size experiments with results and observations |
| **Task 2.3** | 3 | New predictions with proper visualization |
| **Part 3** | 7 | Thoughtful, detailed answers demonstrating understanding |
| **Total** | **25** | |

**Deductions:**
- Code doesn't run: -5 points
- Missing visualizations: -3 points
- Incomplete submission: -3 points
- Poor documentation/comments: -2 points
- Late submission: [Your late policy]

---

## Helpful Resources

- [Scikit-Learn Linear Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Diabetes Dataset Description](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
- [Train-Test Split Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [Matplotlib Plotting Guide](https://matplotlib.org/stable/tutorials/index.html)

---

**Good luck! Remember to start early and ask questions during office hours if you need clarification.**
