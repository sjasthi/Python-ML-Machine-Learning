# scikit-learn Toy Datasets 📊

## What are Toy Datasets?

Toy datasets are small, built-in datasets that come with scikit-learn. They're perfect for learning and practicing machine learning without having to find or download data from the internet!

Think of them as "practice problems" that help you learn how to use different machine learning algorithms.

## Why Use Toy Datasets?

- **Ready to use**: No downloading or cleaning needed!
- **Perfect for learning**: Small enough to understand quickly
- **Well-documented**: Everyone uses them, so lots of examples available
- **Great for testing**: Try out new algorithms and ideas
- **Free**: No need to search for data online

## Available Toy Datasets

### Classification Datasets

These datasets are used when you want to predict categories or labels.

#### 1. **Iris Dataset** 🌸
- **What it is**: Measurements of 150 iris flowers (3 different species)
- **Features**: Sepal length, sepal width, petal length, petal width
- **Target**: Species (Setosa, Versicolor, or Virginica)
- **Use case**: Learn basic classification

#### 2. **Digits Dataset** 🔢
- **What it is**: Images of handwritten digits (0-9)
- **Features**: 8x8 pixel images (64 features total)
- **Target**: Which digit it is (0-9)
- **Use case**: Image classification, great for neural networks

#### 3. **Wine Dataset** 🍷
- **What it is**: Chemical analysis of 178 wines from 3 different cultivars
- **Features**: 13 chemical measurements (alcohol, acidity, etc.)
- **Target**: Wine type (3 classes)
- **Use case**: Multi-class classification

#### 4. **Breast Cancer Dataset** 🎗️
- **What it is**: Features computed from breast mass images
- **Features**: 30 measurements describing cell nuclei
- **Target**: Malignant or Benign
- **Use case**: Binary classification, important real-world problem

### Regression Datasets

These datasets are used when you want to predict continuous numbers.

#### 5. **Diabetes Dataset** 💉
- **What it is**: Health measurements from diabetes patients
- **Features**: 10 baseline variables (age, BMI, blood pressure, etc.)
- **Target**: Disease progression after one year (a number)
- **Use case**: Learn regression techniques

#### 6. **California Housing Dataset** 🏠
- **What it is**: Housing prices in California districts
- **Features**: 8 features like average rooms, population, median income
- **Target**: Median house value
- **Use case**: Predicting prices with regression

## How to Load a Dataset

Loading a toy dataset is super easy! Here's the basic pattern:

```python
from sklearn import datasets

# Load the dataset
data = datasets.load_iris()

# The dataset object has several parts:
# data.data = the features (input)
# data.target = the labels (output)
# data.feature_names = names of the features
# data.target_names = names of the labels
# data.DESCR = description of the dataset
```

## Complete Example: Iris Dataset

Here's a complete example using the famous Iris dataset:

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features (measurements)
y = iris.target  # Target (species)

# Step 2: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 3: Create and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 4: Make predictions
predictions = model.predict(X_test)

# Step 5: Check accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 6: Look at some details
print(f"\nDataset shape: {X.shape}")
print(f"Feature names: {iris.feature_names}")
print(f"Target names: {iris.target_names}")
```

## Complete Example: Digits Dataset

Here's an example with image data (handwritten digits):

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load the digits dataset
digits = datasets.load_digits()

# Look at one example digit
plt.imshow(digits.images[0], cmap='gray')
plt.title(f"This is a {digits.target[0]}")
plt.show()

# Prepare data
X = digits.data
y = digits.target

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use K-Nearest Neighbors classifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
print(f"Total images: {len(digits.images)}")
print(f"Image size: 8x8 pixels")
```

## Exploring a Dataset

Here's how to explore what's inside a dataset:

```python
from sklearn import datasets

# Load any dataset
data = datasets.load_iris()

# Print the description (very helpful!)
print(data.DESCR)

# See the shape
print(f"Number of samples: {data.data.shape[0]}")
print(f"Number of features: {data.data.shape[1]}")

# See feature names
print(f"Features: {data.feature_names}")

# See the first few samples
print(f"First sample: {data.data[0]}")
print(f"First label: {data.target[0]}")
```

## Quick Reference: Loading Functions

```python
from sklearn import datasets

# Classification datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()

# Regression datasets
diabetes = datasets.load_diabetes()
california_housing = datasets.fetch_california_housing()
```

**Note**: Some datasets use `load_` (already included) while others use `fetch_` (downloaded when first used).

## Practice Challenges

Try these on your own:

1. Load the Wine dataset and train a Logistic Regression model
2. Use the Digits dataset with K-Means clustering
3. Load the Diabetes dataset and create a Linear Regression model
4. Compare the accuracy of different models on the Breast Cancer dataset
5. Visualize some samples from the Digits dataset

## Official Documentation

- **Dataset Documentation**: [https://scikit-learn.org/stable/datasets/toy_dataset.html](https://scikit-learn.org/stable/datasets/toy_dataset.html)
- **All Datasets**: [https://scikit-learn.org/stable/datasets.html](https://scikit-learn.org/stable/datasets.html)

## Tips for Using Toy Datasets

- Always split your data into training and testing sets
- Read the dataset description using `data.DESCR` to understand what you're working with
- Start with small datasets like Iris before moving to larger ones
- Try different algorithms on the same dataset to see which works best
- These datasets are great for homework and projects!

## Fun Fact! 🎉

The Iris dataset is one of the most famous datasets in machine learning history! It was introduced by Ronald Fisher in 1936 and has been used to teach machine learning for almost 90 years!

---

**Remember**: These toy datasets are your training ground. Master them first, then you'll be ready to tackle real-world data! 🚀