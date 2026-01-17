# Week 15: Feature Engineering - Making Data ML-Ready

**Course:** Python for Machine Learning  
**Instructor:** Siva Jasthi (Siva.Jasthi@metrostate.edu)  
**Week:** 15  
**Topic:** Feature Engineering - Transforming Raw Data into ML Gold

---

## 📚 Learning Objectives

By the end of this week, students will be able to:

1. Understand what feature engineering is and why it's critical for machine learning
2. Handle missing data using different imputation strategies
3. Convert categorical variables to numerical format using encoding techniques
4. Apply label encoding for ordinal and binary categories
5. Use one-hot encoding for nominal categories
6. Scale features using standardization and normalization
7. Create new features by combining or transforming existing ones
8. Generate polynomial features to capture non-linear relationships
9. Select the most important features to reduce model complexity
10. Build a complete feature engineering pipeline for real-world datasets

---

## 🎯 What is Feature Engineering?

**The Big Idea:** Feature engineering is the process of transforming raw data into features that better represent the underlying problem to machine learning algorithms.

**The Recipe Analogy:**
```
Raw Data = Raw Ingredients (flour, eggs, sugar)
Feature Engineering = Preparing Ingredients (mixing, measuring, combining)
Machine Learning Model = Baking (the actual cooking process)
Predictions = Delicious Cake! 🎂

Just like you need to prepare ingredients before baking, 
you need to engineer features before training models!
```

**Why It Matters:**
- 🎯 Better features → Better predictions
- 📉 Proper scaling → Faster training
- 🎨 Feature creation → Capture hidden patterns
- 🔍 Feature selection → Simpler, faster models

---

## 📋 Topics Covered

### 1. Handling Missing Data

**The Problem:**
Real-world data is messy! You'll often find missing values (NaN, None, blank cells).

**The Solutions:**

| Strategy | What It Does | When To Use | Example |
|----------|--------------|-------------|---------|
| **Drop Rows** | Remove rows with missing values | When you have lots of data and few missing values | Drop customers with missing age |
| **Drop Columns** | Remove features with missing values | When a feature is mostly empty | Drop a column that's 80% empty |
| **Mean/Median Imputation** | Fill with average or middle value | For numerical features | Fill missing age with median age |
| **Mode Imputation** | Fill with most common value | For categorical features | Fill missing country with most common country |
| **Forward/Backward Fill** | Use previous/next value | For time-series data | Fill today's price with yesterday's |
| **Constant Fill** | Fill with a specific value | When missing has meaning | Fill missing income with 0 |

**The Lego Analogy:**
```
Imagine building with Legos but some pieces are missing:
- Drop the row: Skip that part of the building
- Fill with average: Use a standard-sized piece
- Fill with constant: Use a specific replacement piece
```

**Python Example:**
```python
# Drop missing values
df.dropna()  # Drop rows with any missing values
df.dropna(axis=1)  # Drop columns with any missing values

# Fill missing values
df['age'].fillna(df['age'].mean())  # Fill with mean
df['city'].fillna(df['city'].mode()[0])  # Fill with mode
df.fillna(0)  # Fill all missing with 0
```

---

### 2. Label Encoding (Ordinal Encoding)

**What Is It?**
Converting categorical text values into numbers (0, 1, 2, 3...).

**When To Use:**
- ✅ **Ordinal categories** (with natural order): Small < Medium < Large
- ✅ **Binary categories**: Yes/No, True/False, Male/Female
- ❌ **NOT for nominal categories** (no natural order): Red, Blue, Green

**The T-Shirt Size Analogy:**
```
Original: ['Small', 'Medium', 'Large', 'Small', 'Large']
Encoded:  [0, 1, 2, 0, 2]

This works because: Small < Medium < Large (natural order!)
```

**The Problem with Wrong Use:**
```
Original: ['Red', 'Blue', 'Green', 'Red']
Encoded:  [0, 1, 2, 0]

Problem: Model thinks Red(0) < Blue(1) < Green(2)
But there's NO natural order! Red isn't "less than" Blue!
```

**Python Example:**
```python
from sklearn.preprocessing import LabelEncoder

# Education level (ordinal - has order)
education = ['High School', 'Bachelor', 'Master', 'High School', 'PhD']
le = LabelEncoder()
encoded = le.fit_transform(education)
print(encoded)  # [0, 1, 2, 0, 3]
```

---

### 3. One-Hot Encoding

**What Is It?**
Creating separate binary (0/1) columns for each category.

**When To Use:**
- ✅ **Nominal categories** (no natural order): Colors, Countries, Product Types
- ✅ When categories are independent and equal

**The Light Switch Analogy:**
```
Original: Color = ['Red', 'Blue', 'Green', 'Red']

One-Hot Encoded:
Color_Red   Color_Blue   Color_Green
    1           0            0         (Red)
    0           1            0         (Blue)
    0           0            1         (Green)
    1           0            0         (Red)

Each color gets its own "light switch" (1 = on, 0 = off)
```

**Why It's Better for Nominal Data:**
- No false ordering implied (Red ≠ less than Blue)
- Each category is treated equally
- Model can learn independent effects

**Python Example:**
```python
import pandas as pd

# City (nominal - no order)
data = pd.DataFrame({'City': ['NYC', 'LA', 'Chicago', 'NYC']})

# One-hot encoding
encoded = pd.get_dummies(data, columns=['City'], prefix='City')
print(encoded)
#    City_Chicago  City_LA  City_NYC
# 0            0        0         1
# 1            0        1         0
# 2            1        0         0
# 3            0        0         1
```

**Watch Out for the Dummy Variable Trap!**
```python
# Use drop_first=True to avoid multicollinearity
encoded = pd.get_dummies(data, columns=['City'], drop_first=True)
# If City_LA=0 and City_NYC=0, we know it's Chicago!
```

---

### 4. Standardization (Z-Score Normalization)

**What Is It?**
Transforming features to have **mean = 0** and **standard deviation = 1**.

**The Test Score Analogy:**
```
You scored 85 in Math (class avg: 75, std: 10)
You scored 90 in English (class avg: 80, std: 15)

Standardized:
Math: (85-75)/10 = 1.0 (1 std above average)
English: (90-80)/15 = 0.67 (only 0.67 std above average)

You actually did better in Math relatively!
```

**Formula:**
```
z = (x - μ) / σ

Where:
- x = original value
- μ = mean
- σ = standard deviation
- z = standardized value
```

**When To Use:**
- ✅ Features have different units (age vs. income)
- ✅ Algorithms assume normally distributed data (Linear Regression, SVM)
- ✅ You want to preserve the original distribution shape
- ✅ Features have different scales

**Python Example:**
```python
from sklearn.preprocessing import StandardScaler

# Age: 20-80, Income: 20000-100000
data = [[25, 30000], [50, 60000], [35, 45000]]

scaler = StandardScaler()
scaled = scaler.fit_transform(data)
print(scaled)
# Both features now have mean≈0, std≈1
```

---

### 5. Normalization (Min-Max Scaling)

**What Is It?**
Squeezing all values into a fixed range, usually **[0, 1]**.

**The Volume Knob Analogy:**
```
Original sound levels: 10, 50, 100 (different volumes)
Normalized to [0, 1]: 0.0, 0.4, 1.0

All sounds now on the same scale!
Quietest = 0, Loudest = 1
```

**Formula:**
```
X_scaled = (X - X_min) / (X_max - X_min)

Where:
- X = original value
- X_min = minimum value in feature
- X_max = maximum value in feature
```

**When To Use:**
- ✅ Neural networks (need bounded inputs)
- ✅ Image processing (pixel values 0-255 → 0-1)
- ✅ When you need data in a specific range
- ✅ When you know there are no outliers

**⚠️ Warning: Very Sensitive to Outliers!**
```
Values: [1, 2, 3, 4, 1000]
After Min-Max: [0.000, 0.001, 0.002, 0.003, 1.000]

The outlier (1000) squashed everything else near 0!
```

**Python Example:**
```python
from sklearn.preprocessing import MinMaxScaler

data = [[10], [20], [30], [40], [50]]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)
print(scaled)
# [[0.0], [0.25], [0.5], [0.75], [1.0]]
```

---

### 6. Feature Creation

**What Is It?**
Creating new features by combining or transforming existing ones to capture hidden patterns.

**Common Techniques:**

| Technique | Description | Example |
|-----------|-------------|---------|
| **Arithmetic Operations** | Add, subtract, multiply, divide | BMI = weight / height² |
| **Date/Time Features** | Extract components | day_of_week, month, hour |
| **Aggregations** | Group and summarize | avg_purchase_per_customer |
| **Binning** | Convert continuous to categories | age → age_group (young/middle/old) |
| **Domain Knowledge** | Use expertise | is_weekend, is_holiday |

**The Cooking Analogy:**
```
You have: Flour, Eggs, Sugar (original features)
You create: Cake Batter (new feature combining all three)

The batter is MORE useful than individual ingredients!
```

**Examples:**

```python
# 1. BMI from height and weight
df['BMI'] = df['weight'] / (df['height'] ** 2)

# 2. Total price from quantity and unit price
df['total_price'] = df['quantity'] * df['unit_price']

# 3. Age groups from age
df['age_group'] = pd.cut(df['age'], 
                          bins=[0, 18, 35, 60, 100],
                          labels=['child', 'young', 'middle', 'senior'])

# 4. Day of week from date
df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6])

# 5. Time since last purchase
df['days_since_purchase'] = (pd.to_datetime('today') - 
                               pd.to_datetime(df['last_purchase'])).dt.days
```

**Real-World Example:**
```
Original Features: purchase_date, customer_age, product_price
Created Features:
- days_since_last_purchase
- customer_age_group
- is_weekend_purchase
- price_per_age_ratio

These new features often work BETTER than the originals!
```

---

### 7. Polynomial Features

**What Is It?**
Creating interaction terms and higher-degree features to capture **non-linear relationships**.

**The Simple Math:**
```
Original Features: x₁, x₂

Degree 2 Polynomial:
x₁, x₂, x₁², x₂², x₁·x₂

Degree 3 Polynomial:
x₁, x₂, x₁², x₂², x₁·x₂, x₁³, x₂³, x₁²·x₂, x₁·x₂²
```

**The Pizza Analogy:**
```
Original: Size (small/large), Toppings (1-5)

Polynomial Features:
- Size (original)
- Toppings (original)
- Size × Toppings (interaction: large pizza with many toppings!)
- Size² (does doubling size have exponential effect?)
- Toppings² (do many toppings have diminishing returns?)

The interaction term captures: "Large pizzas with lots of toppings 
are EXTRA expensive" - not just additive!
```

**When To Use:**
- ✅ When relationship between features is non-linear
- ✅ When features interact with each other
- ✅ To improve model performance on complex patterns

**⚠️ Warning: Creates Many Features!**
```
2 features, degree 2: 6 features
5 features, degree 2: 21 features
10 features, degree 2: 66 features

Can lead to overfitting and slow training!
```

**Python Example:**
```python
from sklearn.preprocessing import PolynomialFeatures

# Original: [height, weight]
data = [[1.7, 70], [1.8, 80]]

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data)

# New features: [height, weight, height², height×weight, weight²]
print(poly_features)
```

---

### 8. Feature Selection

**What Is It?**
Choosing the **most important features** and removing irrelevant or redundant ones.

**Why Do It?**
- 🚀 **Faster training:** Fewer features = faster models
- 📉 **Reduce overfitting:** Simpler models generalize better
- 💡 **Better interpretability:** Easier to understand
- 💰 **Lower cost:** Less data to collect and store

**The Backpack Analogy:**
```
You're going hiking (training a model).
Your backpack has:
- Water bottle ✅ (essential!)
- Snacks ✅ (important!)
- First aid kit ✅ (crucial!)
- 10 textbooks ❌ (heavy and useless for hiking!)
- Bowling ball ❌ (why would you bring this?!)

Feature selection = removing the textbooks and bowling ball!
```

**Three Main Approaches:**

### **A. Filter Methods** (Statistical Tests)
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top 5 features based on ANOVA F-statistic
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)
```

**Pros:** Fast, independent of model  
**Cons:** Ignores feature interactions

### **B. Wrapper Methods** (Use Model Performance)
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Recursive Feature Elimination
model = LogisticRegression()
rfe = RFE(estimator=model, n_features_to_select=5)
X_selected = rfe.fit_transform(X, y)
```

**Pros:** Considers feature interactions  
**Cons:** Slow, computationally expensive

### **C. Embedded Methods** (Built into Model)
```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest feature importance
model = RandomForestClassifier()
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
important_features = X.columns[importances > 0.05]
```

**Pros:** Fast, considers interactions  
**Cons:** Model-specific

**Comparison Table:**

| Method | Speed | Accuracy | Considers Interactions? |
|--------|-------|----------|------------------------|
| **Filter** | ⚡⚡⚡ Fast | 😐 Moderate | ❌ No |
| **Wrapper** | 🐢 Slow | 😊 High | ✅ Yes |
| **Embedded** | ⚡⚡ Fast | 😊 High | ✅ Yes |

---

## 🎯 Class Activities (60 minutes)

### **Activity 1: The Missing Data Challenge (15 minutes)**

**Hands-On Exercise:**

```python
import pandas as pd
import numpy as np

# Create dataset with missing values
data = {
    'age': [25, np.nan, 35, 40, np.nan, 50],
    'income': [30000, 45000, np.nan, 60000, 55000, np.nan],
    'city': ['NYC', 'LA', 'NYC', np.nan, 'LA', 'Chicago']
}
df = pd.DataFrame(data)

# Challenge: Try different strategies
# 1. Drop rows with any missing values
# 2. Fill age with mean, income with median
# 3. Fill city with mode
# 4. Which strategy loses least information?
```

**Student Tasks:**
1. Identify how many missing values are in each column
2. Calculate what % of data would be lost with dropna()
3. Implement all three filling strategies
4. Discuss: Which strategy is best and why?

**Expected Output:**
```
Strategy 1 (Drop): Only 2 complete rows remain (lost 67%!)
Strategy 2 (Fill): All 6 rows remain
Best choice: Fill with mean/median/mode - preserves data!
```

---

### **Activity 2: Encoding Showdown (15 minutes)**

**Live Coding Demo:**

```python
# Dataset with both ordinal and nominal features
data = pd.DataFrame({
    'education': ['High School', 'Bachelor', 'Master', 'High School', 'PhD'],
    'color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],
    'salary': [30000, 50000, 70000, 35000, 90000]
})

# Challenge: Encode both features
# 1. Use Label Encoding for education (ordinal)
# 2. Use One-Hot Encoding for color (nominal)
# 3. Compare results
```

**Student Tasks:**
1. Explain why education is ordinal and color is nominal
2. Encode education using LabelEncoder
3. Encode color using pd.get_dummies()
4. What happens if you use Label Encoding on color? (Show the problem!)

**Key Learning:**
```
education (ordinal):     HS(0) < Bachelor(1) < Master(2) < PhD(3) ✅
color (nominal):         Red(0) < Blue(1) < Green(2) ❌ WRONG!

One-hot for color:       Color_Red  Color_Blue  Color_Green ✅
```

---

### **Activity 3: Scaling Showdown (15 minutes)**

**Interactive Demo:**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Features with very different scales
data = pd.DataFrame({
    'age': [25, 35, 45, 55],
    'income': [30000, 50000, 70000, 90000]
})

# Compare both scaling methods
std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

scaled_std = std_scaler.fit_transform(data)
scaled_minmax = minmax_scaler.fit_transform(data)

print("Original:")
print(data)
print("\nStandardized:")
print(scaled_std)
print("\nMin-Max Normalized:")
print(scaled_minmax)
```

**Student Tasks:**
1. Which feature (age or income) would dominate without scaling?
2. After standardization, what's the mean and std of each feature?
3. After min-max, what's the range of each feature?
4. Try adding an outlier (age=25, income=1000000) - which scaling handles it better?

**Expected Insights:**
```
Without scaling: Income dominates (30000 >> 25)
Standardized: Both centered at 0, std=1
Min-Max: Both in [0, 1] range
With outlier: Min-Max squashes normal values, Standardization better!
```

---

### **Activity 4: Feature Creation Lab (15 minutes)**

**Creative Challenge:**

```python
# Real estate dataset
data = pd.DataFrame({
    'bedrooms': [2, 3, 4, 2, 5],
    'bathrooms': [1, 2, 2, 1, 3],
    'sqft': [1000, 1500, 2000, 900, 2500],
    'year_built': [1990, 2000, 2010, 1985, 2015],
    'price': [200000, 300000, 400000, 180000, 500000]
})

# Create new features!
# Ideas:
# 1. Price per sqft
# 2. Age of house (2024 - year_built)
# 3. Total rooms (bedrooms + bathrooms)
# 4. Rooms per sqft (density)
```

**Student Tasks:**
1. Create at least 3 new features
2. Use polynomial features to create interactions
3. Which new features might be most useful for predicting price?
4. Try training a simple model with and without new features

**Example Solution:**
```python
# Feature engineering
df['price_per_sqft'] = df['price'] / df['sqft']
df['house_age'] = 2024 - df['year_built']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['room_density'] = df['total_rooms'] / df['sqft']

# These features often improve predictions!
```

---

## 📚 Resources & Materials

### **📓 Class Notebook (Follow Along)**
- **Colab Notebook**: [ML_Feature_Engineering.ipynb](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Colab_Notebooks/ML_Feature_Engineering.ipynb)
- Covers all 8 feature engineering techniques
- Real-world examples with complete code
- Step-by-step walkthroughs

### **🎮 Interactive Playbooks (Study at Home)**

**CART Performance Measures Playbook:**
- [ML_CART_Performance_Measures_Interactive_Playbook.html](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Play/ML_CART_Performance_Measures_Interactive_Playbook.html)
- Review Gini Impurity and Entropy
- Interactive calculators and quizzes
- Helps understand when feature engineering improves tree-based models

**Feature Scaling Playbook:**
- [ML_Feature_Scaling_Interactive_Playbook.html](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Play/ML_Feature_Scaling_Interactive_Playbook.html)
- Deep dive into Standardization, Min-Max, and Robust Scaling
- Interactive calculators for each method
- Compare all methods side-by-side
- 8-question quiz to test understanding

### **📊 Dataset Used**
- Real estate pricing data
- Customer demographics data
- Time-series sales data

---

## 🗓️ What's Due This Week?

### **✅ Assignment: Feature Selection**
- **Due**: End of Week 15
- **Format**: Jupyter Notebook or Interactive Playbook or Markup Document
- **Points**: 25 points
- **Link**: [Python_ML_Assignment_Feature_Selection.md](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Assignments/Python_ML_Assignment_Feature_Selection.md)

---

## 💡 Teaching Tips & Common Challenges

### **Common Student Mistakes:**

1. **Using Label Encoding for Nominal Categories**
   - **Problem**: Student encodes ['Red', 'Blue', 'Green'] as [0, 1, 2]
   - **Fix**: "Is Red less than Blue? No! Use One-Hot Encoding instead!"
   - **Visual**: Draw on board showing false ordering problem

2. **Fitting Scaler on Test Data**
   - **Problem**: `StandardScaler().fit_transform(X_test)`
   - **Fix**: "Fit on training data only! This prevents data leakage!"
   - **Code Example:**
   ```python
   # ✅ CORRECT
   scaler.fit(X_train)
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   
   # ❌ WRONG - Data Leakage!
   X_test_scaled = scaler.fit_transform(X_test)
   ```

3. **Creating Too Many Polynomial Features**
   - **Problem**: Using degree=5 with 20 features → 10,626 features!
   - **Fix**: "Start with degree=2, only use higher degrees if needed"
   - **Show**: Exponential growth visualization

4. **Not Handling Missing Data First**
   - **Problem**: Trying to encode or scale data with NaN values
   - **Fix**: "Always handle missing data FIRST in your pipeline!"
   - **Pipeline Order:**
   ```
   1. Handle Missing Data
   2. Encode Categories  
   3. Scale Features
   4. Create New Features
   5. Select Features
   ```

5. **Removing Too Many Features**
   - **Problem**: Selecting only 2 features when you have 50
   - **Fix**: "Plot model performance vs. number of features"
   - **Rule of Thumb**: Start by keeping 20-30% of features

### **Pacing Recommendations:**
- Spend extra time on encoding - it's conceptually tricky
- Use the "when to use what" decision tree frequently
- Have students predict encoding before showing results
- Show failure cases (what happens when you use wrong encoding)

### **Engagement Strategies:**
- **Real-World Poll**: "What features would you create to predict house prices?"
- **Encoding Game**: Give categories, students vote on label vs. one-hot
- **Feature Creation Contest**: Who can create the most useful feature?
- **Scaling Challenge**: "Add an outlier - which scaler survives?"

### **Visual Aids:**
Create posters for:
- Encoding decision tree (ordinal → label, nominal → one-hot)
- Scaling comparison chart (when to use which)
- Feature engineering pipeline flowchart

---

## 🔍 Key Takeaways

Students should leave this week able to answer:

✅ **What is feature engineering?**
- Transforming raw data into features that better represent patterns to ML models

✅ **How do I handle missing data?**
- Drop (if little data missing), Fill with mean/median/mode, or Fill with constant

✅ **What's the difference between Label and One-Hot Encoding?**
- Label: For ordinal/binary categories (has order)
- One-Hot: For nominal categories (no order)

✅ **When do I use Standardization vs. Normalization?**
- Standardization: For normally distributed data, preserves shape
- Normalization: When you need [0,1] range, no outliers

✅ **What are polynomial features?**
- Interaction terms (x₁ × x₂) and higher powers (x₁²) to capture non-linear relationships

✅ **Why is feature selection important?**
- Faster training, less overfitting, better interpretability, lower cost

✅ **What's the correct order for feature engineering?**
1. Handle missing data
2. Encode categories
3. Scale features
4. Create new features
5. Select features

---

## 📊 Quick Reference: Feature Engineering Cheat Sheet

### **Handling Missing Data**
| Method | Code | When |
|--------|------|------|
| Drop | `df.dropna()` | Few missing, lots of data |
| Mean | `df.fillna(df.mean())` | Numerical, no outliers |
| Median | `df.fillna(df.median())` | Numerical, with outliers |
| Mode | `df.fillna(df.mode()[0])` | Categorical |

### **Encoding**
| Data Type | Method | Code |
|-----------|--------|------|
| Ordinal | Label Encoding | `LabelEncoder()` |
| Nominal | One-Hot | `pd.get_dummies()` |
| Binary | Label Encoding | `LabelEncoder()` |

### **Scaling**
| Method | Range | Use When |
|--------|-------|----------|
| Standardization | Unbounded | Normal dist, different units |
| Min-Max | [0, 1] | Bounded range needed |
| Robust | Unbounded | Outliers present |

### **Feature Creation Examples**
```python
# Arithmetic
df['BMI'] = df['weight'] / (df['height'] ** 2)

# Date/Time
df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 65, 100])

# Polynomial
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
```

### **Feature Selection**
```python
# Filter Method
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=10)

# Wrapper Method
from sklearn.feature_selection import RFE
selector = RFE(estimator=model, n_features_to_select=10)

# Embedded Method
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
importances = model.feature_importances_
```

---

## 🎯 Feature Engineering Pipeline Template

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Define transformers for different column types
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Complete pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Fit and transform
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

---

## 📧 Questions?

If you have questions about this week's material:
- **Office Hours**: [Your office hours here]
- **Email**: Siva.Jasthi@metrostate.edu
- **Discussion Forum**: [Link if applicable]

---

## 🌟 Final Thought

**"Feature engineering is where domain knowledge meets data science. The best features often come from understanding your problem deeply, not just from fancy algorithms!"**

**Remember:**  
🎨 Be creative with feature creation  
🔍 Be thoughtful with feature selection  
⚖️ Be careful with encoding and scaling  
🚀 Good features make good models!

---

**Practice Makes Perfect!** Work through the notebook examples, play with the interactive playbooks, and complete the assignment. Feature engineering is a skill that improves with experience! 💪
