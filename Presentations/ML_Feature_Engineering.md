# Introduction to Feature Engineering

## What is Feature Engineering?

**Feature Engineering** is the art of selecting and transforming the best input variables (features) to help your machine learning model make better predictions.

Think of it like this: if you're trying to predict house prices, is it better to know:
- The number of bedrooms, OR
- The total living space per bedroom?

Both contain information, but the second might be more useful! That's feature engineering.

> "Coming up with features is difficult, time-consuming, requires expert knowledge. 'Applied machine learning' is basically feature engineering." — Andrew Ng

---

## Why Does It Matter?

**Good features can make a simple model perform brilliantly!**

Imagine predicting house prices with linear regression:

```
❌ Poor features → Poor predictions (even with a good model)
✅ Great features → Great predictions (even with a simple model)
```

**Benefits:**
- **Better accuracy**: The model learns more effectively
- **Faster training**: Fewer, better features = less computation
- **Easier interpretation**: Understand what drives predictions
- **Less overfitting**: Remove noise and irrelevant data

---

## The Three Pillars of Feature Engineering

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   1. FEATURE SELECTION  →  2. TRANSFORMATION  →  3. CREATION  │
│   (Choose the best)        (Improve existing)    (Build new)  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 1. Feature Selection: Choosing the Best Features

**Goal:** Keep only the features that actually help predict the target.

### Example: Predicting House Prices

Suppose you have these features:
- Square feet
- Number of bedrooms
- Number of bathrooms
- Square meters (just square feet converted!)
- House color
- Age of house

**Question:** Which features should you keep?

**Analysis:**
- ✅ **Square feet**: Highly correlated with price → KEEP
- ✅ **Bedrooms/Bathrooms**: Important for buyers → KEEP
- ✅ **Age**: Affects value → KEEP
- ❌ **Square meters**: Redundant (same as square feet) → REMOVE
- ❌ **House color**: Probably not predictive → REMOVE

### Visual: Correlation with Price

```
Feature                 Correlation with Price
────────────────────────────────────────────────
Square feet             ████████████ (0.85)
Bedrooms                ████████ (0.65)
Bathrooms               █████████ (0.70)
Age                     ███ (-0.45)
Square meters           ████████████ (0.85)  ← Redundant!
House color             █ (0.05)              ← Not useful!
```

### Simple Python Example

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Original data with 6 features
X_all = df[['square_feet', 'bedrooms', 'bathrooms', 
            'square_meters', 'age', 'house_color']]
y = df['price']

# After feature selection: keep only 4 useful features
X_selected = df[['square_feet', 'bedrooms', 'bathrooms', 'age']]

# Train model
model = LinearRegression()
model.fit(X_selected, y)
```

**Result:** Simpler model, faster training, same (or better) accuracy!

---

## 2. Feature Transformation: Making Features Better

**Goal:** Change features to make them more useful for the model.

### Example 1: Scaling Features

**Problem:** Different features have different scales

```
Before Scaling:
────────────────────────
Square feet: 2000, 1500, 3000  (large numbers)
Bedrooms:    3, 2, 4            (small numbers)
Age:         5, 10, 25          (medium numbers)
```

In linear regression, large numbers dominate! We need to put everything on the same scale.

**Solution: Standardization**

```
After Scaling:
────────────────────────
Square feet: 0.5, -0.8, 1.2   (mean=0, std=1)
Bedrooms:    0.2, -0.9, 1.1   (mean=0, std=1)
Age:         -0.6, 0.1, 1.5   (mean=0, std=1)
```

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Example 2: Polynomial Features

**Problem:** Your data has a curved relationship, but linear regression draws straight lines!

```
Without Polynomial Features:
    Price
      │     ╱
      │   ╱  ·  ·
      │ ╱   ·
      │╱·  ·
      └──────── Square Feet
      (straight line doesn't fit curved data well)

With Polynomial Features (add Square_Feet²):
    Price
      │       ╱─╲
      │     ╱   ╲ · ·
      │   ╱  ·   ╲
      │ ╱  ·      ╲
      └──────────── Square Feet
      (curve fits data better!)
```

**Simple Example:**

```python
# Original feature
square_feet = 2000

# Create polynomial feature
square_feet_squared = 2000²  = 4,000,000

# Now linear regression can learn:
# Price = β₀ + β₁(square_feet) + β₂(square_feet²)
```

---

## 3. Feature Creation: Building New Features

**Goal:** Combine or derive new features that capture important relationships.

### Example: House Price Prediction

**Original Features:**
- Square feet: 2000
- Bedrooms: 4

**Created Features:**

#### Feature 1: Total Rooms
```python
total_rooms = bedrooms + bathrooms
# 4 + 2 = 6
```
*Why useful?* Total space matters for families!

#### Feature 2: Space per Bedroom
```python
space_per_bedroom = square_feet / bedrooms
# 2000 / 4 = 500 sq ft per bedroom
```
*Why useful?* Shows if rooms are spacious or cramped!

#### Feature 3: Is New Construction?
```python
is_new = 1 if age < 5 else 0
# If age = 3 → is_new = 1 (True)
```
*Why useful?* New homes often command premium prices!

#### Feature 4: Interaction Term
```python
sqft_x_bedrooms = square_feet * bedrooms
# 2000 * 4 = 8000
```
*Why useful?* A large house with many bedrooms is especially valuable!

### Visual: Before vs After Feature Creation

```
BEFORE:                      AFTER:
────────────                 ────────────────────────
square_feet                  square_feet
bedrooms                     bedrooms
bathrooms                    bathrooms
age                          age
                             + total_rooms ✨
                             + space_per_bedroom ✨
                             + is_new ✨
                             + sqft_x_bedrooms ✨
```

---

## Putting It All Together: Linear Regression Example

Let's see how feature engineering improves a house price prediction model!

### Scenario

You want to predict house prices using linear regression.

**Dataset:**
- 1000 houses
- Features: square_feet, bedrooms, bathrooms, age
- Target: price

### Step 1: Baseline Model (No Feature Engineering)

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Use original features
X = df[['square_feet', 'bedrooms', 'bathrooms', 'age']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
```

**Performance:**
- R² Score: 0.82
- Average Error: $45,000

### Step 2: With Feature Engineering

```python
# Create new features
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['space_per_room'] = df['square_feet'] / df['total_rooms']
df['is_new'] = (df['age'] < 5).astype(int)

# Select best features
X = df[['square_feet', 'total_rooms', 'space_per_room', 'is_new']]

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y_train)
```

**Performance:**
- R² Score: 0.91 ✨ (improved!)
- Average Error: $28,000 ✨ (much better!)

### Visual Comparison

```
Model Performance
──────────────────────────────────────────────
                  Baseline    With Feature Engineering
R² Score:         0.82        0.91 ⭐
Avg Error:        $45,000     $28,000 ⭐
Training Time:    100ms       95ms ⭐
```

---

## Common Pitfalls to Avoid

### ❌ Don't: Keep redundant features
```python
# Bad: Both measure the same thing!
features = ['square_feet', 'square_meters']
```

### ✅ Do: Remove duplicates
```python
# Good: Keep only one
features = ['square_feet']
```

### ❌ Don't: Create features that leak information
```python
# Bad: Using future information!
df['price_next_month'] = ...  # Can't know this when predicting!
```

### ✅ Do: Use only past/present information
```python
# Good: Historical average
df['neighborhood_avg_price'] = ...  # Based on past sales
```

---

## Key Takeaways

1. **Feature Engineering is crucial**: Good features > Complex models
2. **Three main techniques**:
   - **Selection**: Choose relevant features
   - **Transformation**: Scale, polynomials, etc.
   - **Creation**: Combine features meaningfully
3. **Domain knowledge matters**: Understanding your problem helps create better features
4. **Experiment and iterate**: Try different combinations and measure results

---

## Practice Exercise

Given this house dataset, create 3 new features:

**Original features:**
- `lot_size` (sq ft)
- `house_size` (sq ft)
- `bedrooms`
- `year_built`

**Challenge:** Create features that might improve price prediction!

<details>
<summary>Click to see example solutions</summary>

```python
# 1. Ratio feature
df['house_to_lot_ratio'] = df['house_size'] / df['lot_size']
# Shows how much of the lot is built on

# 2. Time-based feature
current_year = 2024
df['house_age'] = current_year - df['year_built']
# More intuitive than year built

# 3. Categorical feature
df['is_spacious'] = (df['house_size'] / df['bedrooms'] > 500).astype(int)
# Indicates if each bedroom has 500+ sq ft
```

</details>

---

## Next Steps

- Practice with real datasets (Kaggle, UCI ML Repository)
- Learn about advanced techniques (PCA, feature interactions)
- Experiment with different transformations
- Always measure: does your feature engineering actually improve the model?

**Remember:** Feature engineering is both art and science. Keep experimenting! 🚀
