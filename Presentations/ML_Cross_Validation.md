# K-Fold Cross Validation

**Siva R Jasthi**  
Computer Science and Cybersecurity  
Metropolitan State University

---

## Table of Contents

1. [Machine Learning Process](#machine-learning-process)
2. [Training, Validation, and Test Sets](#training-validation-and-test-sets)
3. [Model Performance Assessment](#model-performance-assessment)
4. [The Problem: Overfitting](#the-problem-overfitting)
5. [Cross Validation Introduction](#cross-validation-introduction)
6. [Types of Cross-Validation](#types-of-cross-validation)
7. [Holdout Cross Validation](#holdout-cross-validation)
8. [K-Fold Cross Validation](#k-fold-cross-validation)
9. [Stratified K-Fold Cross Validation](#stratified-k-fold-cross-validation)
10. [Leave-P-Out Cross-Validation](#leave-p-out-cross-validation)
11. [Leave-One-Out Cross-Validation (LOOCV)](#leave-one-out-cross-validation-loocv)
12. [Rolling Cross-Validation for Time Series](#rolling-cross-validation-for-time-series)
13. [Summary](#summary)
14. [References](#references)

---

## Machine Learning Process

The typical machine learning workflow consists of the following key steps:

1. **Data Collection and Preparation**
2. **Feature Selection**
3. **Algorithm Choice**
4. **Parameter and Model Selection**
5. **Training**
6. **Evaluation**

Each step is critical to building effective machine learning models. Cross-validation plays a crucial role in the evaluation phase and helps with model selection and parameter tuning.

---

## Training, Validation, and Test Sets

![Train-Validation-Test Split](image/cross_validation/slide_03_image.png)

### The Three-Way Split

When working with machine learning models, we typically split our data into three distinct sets:

1. **Training Set**: Used to train the model and learn patterns
2. **Validation Set**: Used to evaluate and tune the model during development
3. **Test Set**: Used for final evaluation of the model's performance

### Workflow

1. **Train** the model using the "training set"
2. **Evaluate** the model on the "validation set" and tweak the model as needed
3. **Finally, test** your model with the "test set" to get an unbiased performance estimate

### Common Split Ratios

![Train-Val-Test Example](image/cross_validation/train_val_test_split.png)

A common split ratio is:
- **60%** Training
- **20%** Validation  
- **20%** Test

Other common ratios include 70-15-15 or 80-10-10, depending on dataset size.

---

## Model Performance Assessment

Model evaluation is the process of assessing the performance of a machine learning model on a given dataset.

Depending on the type of problem (classification, regression, clustering) and the model, various metrics/measures come into play.

### Common Evaluation Metrics

#### Classification Metrics

**Basic Metrics:**
- **Accuracy**: The percentage of correctly classified instances out of all instances in the dataset
- **Precision**: The ratio of true positives (correctly predicted positive instances) to the total number of predicted positive instances
- **Recall (Sensitivity)**: The ratio of true positives to the total number of actual positive instances
- **Specificity**: The ratio of true negatives to the total number of actual negative instances
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two metrics

**Advanced Classification Metrics:**
- **AUC-ROC Curve**: The area under the receiver operating characteristic curve, plotting true positive rate against false positive rate
- **Confusion Matrix**: A matrix showing true positives, true negatives, false positives, and false negatives
- **Cohen's Kappa**: Measures agreement between classifiers
- **Matthews Correlation Coefficient (MCC)**: Measures correlation between predicted and actual values
- **Mean Per Class Accuracy**: Average accuracy for each class in multi-class problems
- **Balanced Accuracy**: Average of recall rates for each class, accounting for imbalanced distributions
- **F-beta Score**: Generalized F1 score allowing tuning between precision and recall
- **Top-N Accuracy**: Percentage of times correct answer is in top N predictions
- **Weighted F1 Score**: F1 score for each class weighted by instance count

#### Regression Metrics

- **Mean Squared Error (MSE)**: Average squared difference between predicted and actual values
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **R-squared (R²)**: Measures how well the model fits data by comparing variance
- **Mean Absolute Percentage Error (MAPE)**: Average absolute percentage difference

#### Other Useful Metrics

- **Log Loss**: Measures classifier performance when output is a probability
- **Gini Coefficient**: Measures inequality in probability distributions
- **Mean IoU (Intersection over Union)**: For object detection and segmentation
- **Precision at K**: Precision of top K predictions in recommendation systems
- **Coverage**: Percentage of unique items covered in recommendations
- **Spearman Rank Correlation**: Correlation based on rank order
- **Kullback-Leibler Divergence**: Difference between probability distributions
- **Mean Average Precision (MAP)**: Average precision over recall levels

---

## The Problem: Overfitting

### What is Cross Validation Trying to Solve?

**Cross Validation addresses the problem of Overfitting.**

![Overfitting Illustration](image/cross_validation/overfitting_illustration.png)

### Understanding Overfitting

**Overfitting** happens when a model is too complex and captures the noise or random fluctuations in the training data, instead of the underlying patterns that generalize to new data.

This can lead to:
- Poor performance on the test set
- Model being less useful in practice
- High variance in predictions
- Inability to generalize to new, unseen data

### How Cross-Validation Helps

Cross-validation addresses overfitting by:

1. **Partitioning** the available data into several subsets (folds)
2. **Using each fold** as a validation set while other folds are used for training
3. **Testing the model** on data it has not seen during training
4. **Estimating** the model's generalization performance based on validation set results

This ensures the model learns generalizable patterns rather than memorizing training data.

---

## Cross Validation Introduction

![Cross Validation Concept](image/cross_validation/slide_12_image.png)

Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. It provides a more robust estimate of model performance than a single train-test split.

### Key Benefits

1. **Reliability**: Better estimate of model performance on unseen data
2. **Efficiency**: Makes maximum use of available data
3. **Generalization**: Helps detect overfitting
4. **Model Selection**: Fair comparison between different models
5. **Reduced Bias**: Less dependent on how data was split

---

## Types of Cross-Validation

Cross-validation techniques can be categorized into different types:

### Classification of CV Methods

```
Cross-Validation
├── Non-Exhaustive Methods
│   ├── Holdout Method
│   ├── K-Fold CV
│   └── Stratified K-Fold CV
├── Exhaustive Methods
│   ├── Leave-One-Out (LOO)
│   └── Leave-P-Out (LPO)
└── Rolling Cross-Validation
    └── (For Time Series Data)
```

### Quick Comparison

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| **Holdout** | Very large datasets | Fast, simple | High variance, wastes data |
| **K-Fold** | Most situations | Balanced, reliable | Takes K times longer |
| **Stratified K-Fold** | Imbalanced classes | Preserves class distribution | Classification only |
| **Leave-One-Out** | Very small datasets | Maximum data usage | Extremely slow |
| **Leave-P-Out** | Tiny datasets | Very thorough | Computationally prohibitive |
| **Time Series** | Sequential data | Respects temporal order | Only for time series |

---

## Holdout Cross Validation

![Holdout Method](image/cross_validation/slide_14_image.jpg)

### Overview

The **Holdout Method** is the most basic and simple approach where we split the data into training and testing sets.

### Key Characteristics

- **Simple Split**: Data is divided once into training and testing sets
- **Random Shuffling**: Data is shuffled randomly before splitting
- **Variability**: The model can give different results every time we train it (depending on the split)

### Common Split Ratios

- **70% - 30%** (70% training, 30% testing)
- **75% - 25%** (75% training, 25% testing)
- **80% - 20%** (80% training, 20% testing)

### When to Use

✅ **Best for:**
- Very large datasets (millions of samples)
- Quick initial model building
- Computational efficiency is critical

❌ **Avoid when:**
- Dataset is small (< 10,000 samples)
- Need reliable performance estimates
- Data is imbalanced

### Limitations

- High variance in results (depends heavily on the random split)
- Wastes data (test set never used for training)
- May not be representative if unlucky with the split

---

## K-Fold Cross Validation

![K-Fold Visualization 1](image/cross_validation/slide_15_image.png)

![K-Fold Visualization 2](image/cross_validation/slide_16_image.png)

![K-Fold Diagram](image/cross_validation/kfold_diagram.png)

### Overview

K-Fold Cross-Validation is the most widely used cross-validation technique. It guarantees that the model's score does not depend on how we picked the training and testing datasets.

### How It Works

The dataset is divided into **k** equal-sized subsets (folds), and the holdout method is repeated **k** times.

### Algorithm

1. **Randomly split** your entire dataset into **k** number of folds (subsets)
2. **Pick k-1 folds** to build the model and use the **k-th fold** to test the model
3. **Repeat** step 2, **k times**, each time using a different fold as the test set
4. **Record** the accuracy of the model for each iteration
5. **Average** the accuracy of all k runs to get the final performance metric

### Example: 5-Fold Cross-Validation

```
Fold 1: [TEST] [TRAIN] [TRAIN] [TRAIN] [TRAIN]  → Score: 0.92
Fold 2: [TRAIN] [TEST] [TRAIN] [TRAIN] [TRAIN]  → Score: 0.89
Fold 3: [TRAIN] [TRAIN] [TEST] [TRAIN] [TRAIN]  → Score: 0.91
Fold 4: [TRAIN] [TRAIN] [TRAIN] [TEST] [TRAIN]  → Score: 0.90
Fold 5: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST]  → Score: 0.93
                                        
Average Score: 0.91 (91% accuracy)
```

### Choosing K

**Common Values:**
- **k = 5**: Good balance between bias and variance (most common)
- **k = 10**: More thorough but takes longer
- **k = 3**: Faster but higher variance

**Guidelines:**
- Larger k → lower bias, higher variance, slower computation
- Smaller k → higher bias, lower variance, faster computation

### Advantages

✅ **Every data point is used** for both training and testing
✅ **Results in less biased model** compared to holdout method
✅ **Works best when we have limited data**
✅ **Reduces variance** in performance estimates

### Disadvantages

❌ **Computational cost**: Takes k times as much computation
❌ **Not suitable for** imbalanced datasets (use Stratified K-Fold instead)
❌ **Not suitable for** time series data (use Time Series CV instead)

### Python Example

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# Create K-Fold object
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Create model
model = LogisticRegression()

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=kfold)

print(f"Scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.3f}")
print(f"Std Deviation: {scores.std():.3f}")
```

---

## Stratified K-Fold Cross Validation

### The Problem with Regular K-Fold

Using K-Fold CV can be tricky when dealing with **imbalanced datasets** because it involves random shuffling.

**Problem**: Some folds may become highly imbalanced, resulting in a biased model.

**Example**: A fold might have a majority belonging to one class (say positive), and only a few negative classes.

### What is Stratification?

**Stratification** is the process of rearranging the data to ensure that each fold is a good representative of the whole dataset.

**Goal**: Maintain the same class distribution in each fold as in the original dataset.

### Example

In a binary classification problem where:
- Class A comprises 70% of the data
- Class B comprises 30% of the data

**Stratified K-Fold ensures** that each fold also has approximately:
- 70% Class A samples
- 30% Class B samples

### Visual Comparison

![Stratified Sampling](image/cross_validation/slide_19_image.png)

![Stratified K-Fold Visualization](image/cross_validation/slide_20_image.png)

![Stratified Comparison](image/cross_validation/stratified_comparison.png)

### When to Use

✅ **Must use for:**
- **Imbalanced datasets** (unequal class distributions)
- **Multi-class classification** problems
- **Small datasets** with rare classes

❌ **Cannot use for:**
- **Regression problems** (no classes to stratify)
- **Perfectly balanced datasets** (regular K-Fold is fine)

### Advantages Over Regular K-Fold

1. **Preserves class distribution** in each fold
2. **More reliable estimates** for imbalanced data
3. **Reduces variance** in performance metrics
4. **Fairer model comparison**

### Python Example

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# Create Stratified K-Fold object
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create model
model = LogisticRegression()

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=stratified_kfold)

print(f"Scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.3f}")
print(f"Std Deviation: {scores.std():.3f}")
```

### Real-World Applications

- **Medical diagnosis**: Rare diseases (few positive cases)
- **Fraud detection**: Most transactions are legitimate
- **Spam detection**: Most emails are not spam
- **Credit default prediction**: Most loans are repaid

---

## Leave-P-Out Cross-Validation

![Leave-P-Out Visualization](image/cross_validation/slide_21_image.png)

### Overview

Leave-P-Out (LPO) is an **exhaustive cross-validation** method that tests on all possible combinations.

### How It Works

1. **Select p points** from the total number of data points in the dataset (n)
2. **Train** the model on the remaining **(n - p)** data points
3. **Test** the model on the **p** data points
4. **Repeat** this process for **all possible combinations** of p from the original dataset
5. **Average** the accuracies from all iterations to get final performance

### Number of Iterations

For a dataset with **n** samples and leaving out **p** samples:

**Number of iterations = C(n, p) = n! / (p! × (n-p)!)**

### Example

For n=100 and p=2:
- Number of iterations = C(100, 2) = 4,950 iterations
- Each iteration trains a model!

For n=100 and p=5:
- Number of iterations = C(100, 5) = 75,287,520 iterations
- **Computationally infeasible!**

### Characteristics

✅ **Thorough**: Tests on all possible combinations
✅ **Deterministic**: No randomness involved

❌ **Extremely slow**: Combinatorial explosion
❌ **Rarely practical**: Only feasible for very small p values

### When to Use

**Rarely used in practice** due to computational cost. Only consider when:
- Dataset is very small (< 50 samples)
- p is very small (p ≤ 3)
- Computational resources are abundant
- Need absolutely exhaustive testing

### Python Example

```python
from sklearn.model_selection import LeavePOut, cross_val_score
from sklearn.linear_model import LogisticRegression

# WARNING: Only use small p values!
lpo = LeavePOut(p=2)

model = LogisticRegression()

# This might take a VERY long time!
scores = cross_val_score(model, X, y, cv=lpo)

print(f"Number of iterations: {lpo.get_n_splits(X)}")
print(f"Mean Accuracy: {scores.mean():.3f}")
```

---

## Leave-One-Out Cross-Validation (LOOCV)

![LOOCV Visualization](image/cross_validation/slide_22_image.png)

### Overview

Leave-One-Out Cross-Validation (LOOCV) is a special case of Leave-P-Out where **p = 1**.

### How It Works

1. In each iteration, the model is built using **(N-1)** instances
2. The **single remaining instance** is used to test the model
3. This process is repeated **N times** (once for each data point)
4. Each data point serves as the test set exactly once

### Characteristics

**For a dataset with n samples:**
- Number of iterations = **n**
- Training set size = **n - 1** (99.9% of data)
- Test set size = **1** (single sample)

### Example

For a dataset with 100 samples:
```
Iteration 1:  [X] [√] [√] [√] ... [√]  → Train on 99, test on 1
Iteration 2:  [√] [X] [√] [√] ... [√]  → Train on 99, test on 1
Iteration 3:  [√] [√] [X] [√] ... [√]  → Train on 99, test on 1
...
Iteration 100: [√] [√] [√] [√] ... [X] → Train on 99, test on 1

Final Score = Average of 100 test results
```

### Advantages

✅ **Maximum data usage**: Uses (n-1) samples for training
✅ **Deterministic**: No randomness, same results every time
✅ **No bias** from data splitting
✅ **Good for tiny datasets**: When every sample counts

### Disadvantages

❌ **Computationally expensive**: Trains n models
❌ **High variance**: Single sample test sets are unstable
❌ **Not suitable for large datasets**: Too slow
❌ **Doesn't work well** with complex models (too slow)

### When to Use

✅ **Use LOOCV when:**
- Dataset is very small (< 100 samples)
- Every sample is valuable
- Training is fast
- Need deterministic results

❌ **Avoid LOOCV when:**
- Dataset is large (> 1,000 samples)
- Training is slow
- K-Fold gives similar results faster

### Comparison with K-Fold

| Aspect | LOOCV | 5-Fold | 10-Fold |
|--------|-------|--------|---------|
| Iterations | n | 5 | 10 |
| Training size | n-1 | 80% | 90% |
| Test size | 1 | 20% | 10% |
| Computation | Slowest | Fast | Medium |
| Variance | Highest | Low | Medium |
| Bias | Lowest | Medium | Low |

### Python Example

```python
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression
import time

# Create LOOCV object
loo = LeaveOneOut()

model = LogisticRegression()

# Measure time
start = time.time()
scores = cross_val_score(model, X, y, cv=loo)
elapsed = time.time() - start

print(f"Number of iterations: {loo.get_n_splits(X)}")
print(f"Time taken: {elapsed:.2f} seconds")
print(f"Mean Accuracy: {scores.mean():.3f}")
print(f"Correct predictions: {scores.sum()} out of {len(scores)}")
```

### Pro Tip

**For most practical applications, 5-fold or 10-fold CV is better than LOOCV!**

Use LOOCV only when you have a very small dataset and computational time is not a concern.

---

## Rolling Cross-Validation for Time Series

![Time Series CV](image/cross_validation/slide_23_image.png)

![Time Series Diagram](image/cross_validation/timeseries_cv.png)

### Overview

**Rolling Cross-Validation** (also called **Walk-Forward** or **Rolling Forward**) is a specialized technique applicable **only for time series data**.

### Why Time Series is Different

❌ **Regular K-Fold is WRONG for time series** because:
- It randomly shuffles data
- Training on future data to predict the past violates causality
- Breaks temporal dependencies

✅ **Time Series CV respects temporal order:**
- Always trains on past data
- Always tests on future data
- Maintains chronological sequence

### How It Works

The training set **grows progressively** while the test set moves forward in time:

```
Split 1: [Train ████        ] [Test █]
Split 2: [Train █████       ] [Test █]
Split 3: [Train ██████      ] [Test █]
Split 4: [Train ███████     ] [Test █]
Split 5: [Train ████████    ] [Test █]
         ←─── Past    Future ──→
```

### Algorithm

1. Start with initial training window
2. Predict on next time period (test set)
3. Add test period to training set
4. Predict on next time period
5. Repeat until end of dataset

### Variants

**1. Expanding Window (Cumulative)**
- Training set grows with each iteration
- Uses all historical data
- More stable but slower

**2. Rolling Window (Sliding)**
- Fixed-size training window
- Drops oldest data as new data comes in
- Adapts to recent patterns
- Faster computation

### When to Use

✅ **Must use for:**
- **Stock price prediction**
- **Weather forecasting**
- **Sales forecasting**
- **Demand prediction**
- **Any sequential/temporal data**

❌ **Never use:**
- Regular K-Fold on time series data
- Random shuffling of temporal data
- Testing on past to predict future

### Python Example

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import numpy as np

# Create Time Series CV object
tscv = TimeSeriesSplit(n_splits=5)

model = LinearRegression()

# Iterate through splits
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    # Split data maintaining time order
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train on past
    model.fit(X_train, y_train)
    
    # Predict future
    score = model.score(X_test, y_test)
    
    print(f"Fold {fold}:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")
    print(f"  Score: {score:.3f}")
```

### Key Differences from Regular CV

| Aspect | Regular K-Fold | Time Series CV |
|--------|----------------|----------------|
| Data Order | Random shuffle | Chronological |
| Training Set | Fixed size | Growing size |
| Test Set | Random | Always future |
| Iterations | Independent | Sequential |
| Use Case | General ML | Time series only |

### Real-World Applications

- 📈 **Financial forecasting**: Stock prices, currency exchange rates
- 🌡️ **Weather prediction**: Temperature, rainfall forecasts
- 🏪 **Retail analytics**: Sales forecasting, inventory planning
- 📊 **Web traffic**: User visits, page views prediction
- 💰 **Cryptocurrency**: Price prediction, trading signals
- ⚡ **Energy consumption**: Load forecasting, demand prediction

---

## Summary

### Key Takeaways

1. **Cross-Validation** addresses the problem of **overfitting**
2. Cross-Validation provides a reliable measure of **model performance**
3. Different CV techniques suit different **data characteristics**
4. Proper CV is essential for **model selection** and **hyperparameter tuning**

### Choosing the Right Method

```
Decision Tree:
    
    Is it time series data?
    ├─ YES → Use TimeSeriesSplit
    └─ NO
        ├─ Is dataset very small (< 100)?
        │   ├─ YES → Consider LeaveOneOut
        │   └─ NO → Continue
        ├─ Are classes imbalanced?
        │   ├─ YES → Use Stratified K-Fold
        │   └─ NO → Use K-Fold
        └─ Is dataset very large?
            ├─ YES → Holdout or 3-Fold CV
            └─ NO → 5-Fold or 10-Fold CV
```

### Best Practices

✅ **DO:**
- Always use cross-validation for model evaluation
- Choose CV method based on data characteristics
- Use stratification for imbalanced data
- Respect temporal order for time series
- Set random_state for reproducibility
- Look at both mean and standard deviation

❌ **DON'T:**
- Use holdout method on small datasets
- Use regular K-Fold on imbalanced data
- Shuffle time series data
- Ignore high variance in CV scores
- Forget to scale data properly
- Use too few or too many folds

### Recommended Reading & Viewing

📺 **StatQuest: Cross-Validation Video**  
[https://www.youtube.com/watch?v=fSytzGwwBVw](https://www.youtube.com/watch?v=fSytzGwwBVw)

This excellent video by Josh Starmer provides an intuitive explanation of cross-validation with clear visualizations.

### Next Steps

1. **Practice** with different datasets
2. **Experiment** with various CV techniques
3. **Implement** CV in your projects
4. **Explore** the Jupyter notebook for hands-on examples
5. **Compare** results across different methods

---

## References

### Image Sources

1. V7 Labs - Train/Validation/Test Set: [https://www.v7labs.com/blog/train-validation-test-set](https://www.v7labs.com/blog/train-validation-test-set)

2. KDNuggets - Model Evaluation Metrics: [https://www.kdnuggets.com/2020/05/model-evaluation-metrics-machine-learning.html](https://www.kdnuggets.com/2020/05/model-evaluation-metrics-machine-learning.html)

3. DataAspirant - Cross Validation: [https://dataaspirant.com/cross-validation/](https://dataaspirant.com/cross-validation/)

4. Great Learning - Cross Validation Types: [https://www.mygreatlearning.com/blog/cross-validation/](https://www.mygreatlearning.com/blog/cross-validation/)

5. ResearchGate - K-Fold Example: [https://www.researchgate.net/figure/k-fold-cross-validation-example_fig1_335074407](https://www.researchgate.net/figure/k-fold-cross-validation-example_fig1_335074407)

6. Section.io - K-Fold Implementation: [https://www.section.io/engineering-education/how-to-implement-k-fold-cross-validation/](https://www.section.io/engineering-education/how-to-implement-k-fold-cross-validation/)

7. Analytics Vidhya - Importance of Cross Validation: [https://www.analyticsvidhya.com/blog/2021/05/importance-of-cross-validation-are-evaluation-metrics-enough/](https://www.analyticsvidhya.com/blog/2021/05/importance-of-cross-validation-are-evaluation-metrics-enough/)

8. Baeldung - Stratified Sampling: [https://www.baeldung.com/cs/ml-stratified-sampling](https://www.baeldung.com/cs/ml-stratified-sampling)

9. TSCV Documentation - Time Series CV: [https://tscv.readthedocs.io/en/latest/tutorial/roll_forward.html](https://tscv.readthedocs.io/en/latest/tutorial/roll_forward.html)

### Additional Resources

- Scikit-learn Cross-Validation Guide: [https://scikit-learn.org/stable/modules/cross_validation.html](https://scikit-learn.org/stable/modules/cross_validation.html)
- Python Cross-Validation Tutorial: [https://realpython.com/cross-validation-python/](https://realpython.com/cross-validation-python/)
- Towards Data Science - CV Explained: [https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85](https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85)

---

## Questions?

**Thank you for your attention!**

For questions or feedback, please contact:

**Siva R Jasthi**  
Siva.Jasthi@metrostate.edu  
Metropolitan State University  
Computer Science and Cybersecurity Department

---

*Last Updated: February 2026*
