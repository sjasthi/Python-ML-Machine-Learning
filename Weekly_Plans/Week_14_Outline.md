# Week 14: Decision Trees (Classification and Regression Trees)

**Course:** Python for Machine Learning  
**Instructor:** Siva Jasthi (Siva.Jasthi@metrostate.edu)  
**Week:** 14  
**Topic:** Introduction to Decision Trees (CART)

---

## 📚 Learning Objectives

By the end of this week, students will be able to:

1. Understand what decision trees are and how they make predictions
2. Explain the difference between classification trees and regression trees
3. Understand key concepts: entropy, information gain, and Gini impurity
4. Visualize and interpret decision tree structures
5. Build decision tree classifiers using scikit-learn
6. Understand how trees split data using decision rules
7. Recognize overfitting in decision trees
8. Use hyperparameters to control tree complexity (max_depth, min_samples_split, etc.)
9. Evaluate decision tree performance using appropriate metrics
10. Apply decision trees to real-world classification problems

---

## 📄 Review: Supervised Learning Journey

### What We've Learned So Far:

**Classification Methods:**
- **K-Nearest Neighbors (KNN):** "Tell me who your neighbors are, and I'll tell you who you are"
  - ✓ Simple, intuitive, no training needed
  - ❌ Slow predictions, struggles with high dimensions

**Previous Unsupervised Methods:**
- **K-Means Clustering (Week 12):** Finding groups in unlabeled data
- **Hierarchical & DBSCAN (Week 13):** Advanced clustering techniques

### This Week's Focus:

**Decision Trees 🌳** - A powerful supervised learning method that:
- Makes decisions like a flowchart
- Easy to understand and visualize
- Works for both classification AND regression
- Handles both numerical and categorical features
- No need for feature scaling!

---

## 📖 Week 14 Topics

### Part 1: Introduction to Decision Trees 🌳

#### 1. What are Decision Trees?
- The flowchart analogy: making decisions step by step
- How trees ask yes/no questions about features
- From root to leaves: the anatomy of a tree
- Real-world decision-making examples:
  - "Should I wear a jacket?" (weather → temperature → wind)
  - "Will I like this movie?" (genre → rating → duration)
  - "Is this email spam?" (sender → keywords → links)

#### 2. Classification vs. Regression Trees
**Classification Trees:**
- Predict categories/classes (spam vs. not spam)
- Leaves contain class labels
- Use majority voting in leaf nodes
- Examples: disease diagnosis, customer churn

**Regression Trees:**
- Predict continuous numbers (price, temperature)
- Leaves contain average values
- Use mean of training samples in leaf nodes
- Examples: house prices, sales forecasting

**This Week:** Focus on Classification Trees

#### 3. Tree Terminology (The Tree Family!)
- **Root Node:** The starting point (top of tree)
- **Internal Nodes:** Decision points (ask questions)
- **Branches:** Connections between nodes (yes/no paths)
- **Leaf Nodes:** End points with predictions (final answers)
- **Depth:** How many levels deep the tree goes
- **Split:** Dividing data based on a feature

### Part 2: How Trees Make Decisions 🤔

#### 4. The Splitting Process
**Step-by-Step Tree Building:**
1. Start with all training data at root
2. Find the best feature to split on
3. Create branches for different feature values
4. Repeat for each branch (recursive splitting)
5. Stop when a stopping criterion is met

**The 20 Questions Analogy:**
- Each question narrows down possibilities
- Best questions eliminate the most uncertainty
- Keep asking until you know the answer

#### 5. Measuring Impurity: How "Mixed" is a Node?

**Concept:** Pure nodes contain only one class (good!), impure nodes are mixed (need more splitting)

**Method 1: Gini Impurity** 🎯
- Measures probability of misclassification
- Range: 0 (pure) to 0.5 (50/50 mix for binary)
- Formula: Gini = 1 - Σ(probability of each class)²
- Default in scikit-learn
- Think: "If I randomly label a point, how often am I wrong?"

**Method 2: Entropy & Information Gain** 📊
- Entropy measures disorder/randomness
- Range: 0 (pure) to 1 (maximum disorder for binary)
- Formula: Entropy = -Σ(p × log₂(p))
- Information Gain = Parent Entropy - Weighted Child Entropy
- Think: "How much certainty do I gain from this split?"

**The Messy Room Analogy:**
- Pure node = organized room (all books on shelf)
- Impure node = messy room (books everywhere)
- Splitting = organizing (books vs. clothes in different places)

#### 6. Finding the Best Split
**The Algorithm's Goal:**
- Try every possible split on every feature
- Calculate information gain (or Gini decrease) for each
- Choose the split with highest gain (or lowest Gini)
- This creates the purest child nodes

**Example with Student Data:**
```
Feature: Study Hours
Split at 5 hours:
  Left: < 5 hours → mostly fail (impure)
  Right: ≥ 5 hours → mostly pass (more pure)
Information Gain = 0.3 (good split!)

Feature: Sleep Hours  
Split at 6 hours:
  Left: < 6 hours → mixed results
  Right: ≥ 6 hours → mixed results
Information Gain = 0.05 (not helpful)

Winner: Study Hours gives more information!
```

### Part 3: Building Trees with Scikit-Learn 💻

#### 7. DecisionTreeClassifier Basics
**Key Parameters:**

**max_depth:**
- Maximum levels in the tree
- Deeper = more complex = can overfit
- Shallow = simpler = may underfit
- Example: max_depth=3 means only 3 levels of questions

**min_samples_split:**
- Minimum samples needed to split a node
- Higher = prevents splitting small groups
- Prevents overfitting to noise
- Default: 2

**min_samples_leaf:**
- Minimum samples required in leaf node
- Ensures predictions based on enough data
- Prevents tiny leaves with 1-2 samples
- Default: 1

**criterion:**
- 'gini': Gini impurity (default, faster)
- 'entropy': Information gain (more interpretable)
- Both usually give similar results

**random_state:**
- Ensures reproducible results
- Important when ties exist in splitting

#### 8. Training a Decision Tree
**Basic Workflow:**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)

# Train (fit)
tree.fit(X_train, y_train)

# Predict
y_pred = tree.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

#### 9. Visualizing Decision Trees
**Why Visualization Matters:**
- See exactly how the tree makes decisions
- Understand which features are most important
- Identify overfitting (too many splits)
- Explain predictions to non-technical people

**Methods:**
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(tree, 
          feature_names=feature_names,
          class_names=class_names,
          filled=True,  # Color by class
          rounded=True)  # Rounded boxes
plt.show()
```

**Reading Tree Diagrams:**
- **Top box (root):** First question asked
- **Color intensity:** How pure the node is
- **samples:** How many training points reached this node
- **value:** Count of each class at this node
- **class:** Predicted class for this node

### Part 4: The Overfitting Problem 🚨

#### 10. Understanding Overfitting in Trees

**The Memorization Problem:**
- Trees can perfectly memorize training data
- Create specific rule for every single training point
- Results in very deep, complex trees
- Fails on new data

**Signs of Overfitting:**
- ✅ Training accuracy = 100%
- ❌ Test accuracy = 70% or worse
- Very deep tree (10+ levels)
- Leaves with only 1-2 samples
- Very specific, detailed rules

**The Study Guide Analogy:**
- Overfitting = memorizing only practice test questions
- Good fitting = understanding concepts to handle any question
- Underfitting = barely studying at all

#### 11. Controlling Overfitting

**Strategy 1: Limit Tree Depth**
```python
# Shallow tree - may underfit
tree = DecisionTreeClassifier(max_depth=3)

# Moderate tree - usually good
tree = DecisionTreeClassifier(max_depth=5)

# Deep tree - may overfit
tree = DecisionTreeClassifier(max_depth=15)
```

**Strategy 2: Minimum Samples**
```python
# Require at least 20 samples to split
tree = DecisionTreeClassifier(min_samples_split=20)

# Require at least 10 samples in leaves
tree = DecisionTreeClassifier(min_samples_leaf=10)
```

**Strategy 3: Pruning** (Advanced - mentioned but not implemented this week)
- Build full tree first
- Remove branches that don't improve validation performance
- "Cost-complexity pruning" in scikit-learn

**Finding the Sweet Spot:**
- Try different max_depth values (3, 5, 7, 10)
- Compare training vs. test accuracy
- Look for the depth where test accuracy peaks
- Use cross-validation for robust evaluation

### Part 5: Feature Importance 🌟

#### 12. What Features Matter Most?

**Feature Importance Scores:**
- Every feature gets a score (0 to 1)
- Higher = more important for predictions
- Based on how much each feature reduces impurity
- Sum of all importances = 1.0

**Accessing Feature Importance:**
```python
importances = tree.feature_importance_
feature_names = ['age', 'income', 'education']

for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.3f}")
```

**Visualization:**
```python
import pandas as pd

# Create dataframe
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Plot
importance_df.plot(x='Feature', y='Importance', kind='barh')
```

**Interpreting Importance:**
- Top features are used for splitting near the root
- These features best separate classes
- Low importance = not useful for prediction
- Can help with feature selection

---

## 📚 Required Materials

### 📊 Presentation
**Decision Trees (CART) Presentation**
- [ML_Decision_Trees_aka_CART.pdf](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_Decision_Trees_aka_CART.pdf)
- Comprehensive guide covering:
  - Introduction to decision trees
  - Classification vs. regression trees
  - Splitting criteria (Gini, Entropy)
  - Tree construction algorithm
  - Overfitting and hyperparameters
  - Feature importance
  - Real-world applications
- **Review carefully before starting the Colab notebook**

### 💻 Hands-On Notebook
**Marketing Data Decision Tree**
- [ML_Decision_Tree_Classifier_Marketing.ipynb](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Colab_Notebooks/ML_Decistion_Tree_Classifier_Marketing.ipynb)
- Interactive examples including:
  - Building decision tree classifier
  - Visualizing tree structure
  - Hyperparameter tuning (max_depth, min_samples)
  - Feature importance analysis
  - Evaluating model performance
  - Comparing different tree configurations
- Work through this notebook to understand concepts

### ✅ Dataset for Class
**marketing_data.xlsx**
- Used in the Colab notebook
- Contains customer marketing campaign data
- Features include demographics, purchase history
- Target: whether customer responded to campaign
- Perfect for learning decision tree classification

---

## 📝 Required Assignment (25 Points)

### Breast Cancer Decision Tree Classifier

**Objective:** Build a decision tree classifier to predict breast cancer diagnosis (malignant vs. benign)

**Dataset:** 
- [breast-cancer.csv](https://github.com/sjasthi/Python-DS-Data-Science/blob/main/datasets/breast-cancer.csv)
- Features: Various tumor measurements (radius, texture, perimeter, area, etc.)
- Target: diagnosis (M = Malignant, B = Benign)

**Requirements:**

**Part 1: Data Exploration (5 points)**
1. Load the breast-cancer.csv dataset
2. Display basic information (shape, columns, data types)
3. Check for missing values
4. Display the class distribution (how many M vs. B)
5. Show summary statistics for features

**Part 2: Data Preparation (5 points)**
1. Separate features (X) and target (y)
2. Encode the target variable if needed (M=1, B=0 or use LabelEncoder)
3. Split data into training (70%) and testing (30%) sets
4. Use random_state=42 for reproducibility

**Part 3: Build Decision Tree Models (8 points)**
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

**Part 4: Visualization (4 points)**
1. Visualize your best decision tree using plot_tree()
   - Make it readable (large figure size)
   - Include feature names
   - Include class names (Malignant, Benign)
   - Use filled=True for colored boxes

2. Create a feature importance bar chart
   - Show top 10 most important features
   - Sort from highest to lowest importance

**Part 5: Analysis (3 points)**
Write a brief analysis (4-6 sentences) addressing:
1. Which hyperparameter setting worked best and why?
2. What are the top 3 most important features for prediction?
3. Did you observe any overfitting? How can you tell?
4. How confident would you be using this model in a real medical setting?

**Submission Instructions:**
1. Complete all parts in a Google Colab notebook
2. Add markdown cells with explanations for each step
3. Ensure all outputs are visible
4. Share your notebook with proper permissions
5. Submit the notebook link to the assignment dropbox

**Due Date:** End of Week 14

**Grading Rubric:**
- Data Exploration: 5 points
- Data Preparation: 5 points  
- Model Building & Comparison: 8 points
- Visualization: 4 points
- Analysis: 3 points
- **Total: 25 points**

**Tips for Success:**
- Follow the marketing_data notebook as a template
- Add comments explaining your code
- Use markdown cells to document your process
- Compare multiple models to find the best one
- Make visualizations clear and readable
- Think critically about overfitting

---

## 🎯 Key Concepts to Master

### ✅ Understanding Checklist:

**Decision Tree Fundamentals:**
- [ ] Can explain what a decision tree is in simple terms
- [ ] Understand the difference between classification and regression trees
- [ ] Know tree terminology (root, node, leaf, branch, depth)
- [ ] Can read and interpret a tree diagram
- [ ] Understand how trees make predictions (follow the path)

**Splitting and Impurity:**
- [ ] Understand what "impurity" means in a node
- [ ] Can explain Gini impurity concept
- [ ] Can explain entropy and information gain concept
- [ ] Know how the tree decides which feature to split on
- [ ] Understand why pure nodes are good

**Building Trees:**
- [ ] Can create a DecisionTreeClassifier in scikit-learn
- [ ] Know how to fit the model to training data
- [ ] Can make predictions on new data
- [ ] Understand the training process (recursive splitting)
- [ ] Can visualize trees using plot_tree()

**Hyperparameters:**
- [ ] Understand what max_depth controls
- [ ] Know what min_samples_split does
- [ ] Understand min_samples_leaf parameter
- [ ] Can explain the criterion parameter (gini vs. entropy)
- [ ] Know when to use each hyperparameter

**Overfitting:**
- [ ] Can recognize signs of overfitting
- [ ] Understand why trees overfit easily
- [ ] Know how to control overfitting with hyperparameters
- [ ] Can compare training vs. test accuracy
- [ ] Understand the bias-variance tradeoff

**Feature Importance:**
- [ ] Can extract feature importance scores
- [ ] Know how to interpret importance values
- [ ] Can visualize feature importance
- [ ] Understand what makes a feature important
- [ ] Can use importance for feature selection

**Evaluation:**
- [ ] Can calculate accuracy for decision trees
- [ ] Know how to create confusion matrices
- [ ] Can interpret precision and recall
- [ ] Understand when accuracy might be misleading
- [ ] Can compare multiple models objectively

---

## 📊 Decision Tree Advantages & Limitations

### ✅ Advantages (Why We Love Trees!)

**1. Easy to Understand and Visualize** 🎨
- Tree diagrams are intuitive flowcharts
- Can explain decisions to non-technical people
- "If this, then that" logic makes sense
- Great for communicating with stakeholders

**2. No Feature Scaling Needed** 📏
- Works directly with original data
- No need to standardize or normalize
- Treats features independently
- Saves preprocessing time

**3. Handles Both Numerical and Categorical** 🔢
- Can mix different feature types
- No need for one-hot encoding (in some implementations)
- Natural handling of categorical splits

**4. Non-Linear Decision Boundaries** 🌊
- Can capture complex patterns
- Not limited to straight lines
- Creates rectangular decision regions
- Handles interactions between features

**5. Feature Importance** ⭐
- Automatic feature ranking
- Identifies key predictors
- Helps understand the problem
- Useful for feature selection

**6. Minimal Data Preparation** ⚡
- Resistant to outliers in features
- Handles missing values (with modifications)
- No assumptions about data distribution
- Less preprocessing needed

### ❌ Limitations (What to Watch Out For!)

**1. Overfitting Tendency** 🚨
- Can create overly complex trees
- Memorizes training data easily
- Poor generalization without constraints
- Requires careful hyperparameter tuning

**2. Instability** 🎲
- Small data changes can completely change tree
- Different training sets may produce very different trees
- High variance in predictions
- Solution: use ensemble methods (next weeks!)

**3. Greedy Algorithm** 🎯
- Makes locally optimal decisions
- May miss globally optimal tree
- Can't backtrack once split is made
- May create suboptimal trees

**4. Biased Toward Dominant Classes** ⚖️
- Favors majority class in imbalanced data
- May ignore minority class
- Need to balance data or use class weights

**5. Axis-Parallel Splits Only** 📐
- Can only split parallel to axes
- Struggles with diagonal patterns
- May need many splits for simple patterns
- Linear relationships can be inefficient

**6. Not Great for Extrapolation** 📉
- Can't predict beyond training data range
- Predictions limited to training set values
- No smooth interpolation like regression

---

## 🔍 When to Use Decision Trees

### ✅ Good Use Cases:

**Medical Diagnosis:**
- Clear decision rules for doctors
- Interpretable predictions
- Example: tumor classification (this week's assignment!)

**Customer Segmentation:**
- Identify customer types
- Create actionable profiles
- Easy to explain to marketing team

**Credit Approval:**
- Transparent lending decisions
- Explainable to regulators
- Fair and auditable

**Fraud Detection:**
- Fast real-time decisions
- Clear rules for suspicious activity
- Easy to update rules

**Quality Control:**
- Manufacturing defect detection
- Clear pass/fail criteria
- Actionable insights for improvement

### ❌ When to Consider Alternatives:

**Very Large Datasets:**
- Trees can become too large
- Consider random forests or gradient boosting
- Or simpler models like logistic regression

**Smooth Relationships:**
- Linear or smooth patterns
- Logistic regression might be better
- Less prone to overfitting

**High-Stakes Predictions:**
- Single tree is too unstable
- Use ensemble methods (Random Forest, XGBoost)
- Better accuracy and reliability

**Need for Uncertainty:**
- Trees give hard classifications
- Consider probabilistic methods
- Logistic regression gives probabilities

---

## 📊 Complete Supervised Learning Comparison

### Models Learned So Far:

| Feature | KNN | Decision Trees |
|---------|-----|----------------|
| **Training Time** | ⚡ Instant (lazy) | 🚗 Moderate |
| **Prediction Time** | 🐌 Slow | ⚡ Very Fast |
| **Interpretability** | ❌ Black box | ✅ Very clear |
| **Feature Scaling** | ⚠️ Required | ✅ Not needed |
| **Overfitting** | With small K | ⚠️ Easy to overfit |
| **Handles Non-Linear** | ✅ Yes | ✅ Yes |
| **Categorical Features** | ❌ Difficult | ✅ Natural |
| **Feature Importance** | ❌ No | ✅ Built-in |
| **Works with Missing Data** | ❌ No | ⚠️ With modifications |
| **Best For** | Small data, simple | Interpretable, mixed features |

---

## 🔄 Activity Checklist

### Class Activities:
- [ ] Reviewed presentation on Decision Trees
- [ ] Walked through marketing_data Colab notebook
- [ ] Built decision tree classifier
- [ ] Visualized tree structure
- [ ] Experimented with different max_depth values
- [ ] Analyzed feature importance
- [ ] Compared training vs. test accuracy
- [ ] Identified overfitting in deep trees

### Assignment Progress:
- [ ] Downloaded breast-cancer.csv dataset
- [ ] Explored the data (shape, columns, class distribution)
- [ ] Prepared features and target
- [ ] Split into train/test sets
- [ ] Built baseline decision tree model
- [ ] Created models with different hyperparameters
- [ ] Compared model performances
- [ ] Visualized best decision tree
- [ ] Created feature importance chart
- [ ] Wrote analysis of results
- [ ] Submitted completed notebook


---

## 🎉 Learning Outcomes

By completing Week 14, you will have:

✅ Understood decision tree fundamentals and structure  
✅ Learned how trees split data using impurity measures  
✅ Mastered Gini impurity and entropy concepts  
✅ Built decision tree classifiers with scikit-learn  
✅ Visualized and interpreted tree diagrams  
✅ Controlled overfitting with hyperparameters  
✅ Analyzed feature importance  
✅ Applied trees to real medical diagnosis problem  
✅ Compared multiple models to find best configuration  
✅ Developed intuition for when to use decision trees  

---

## 📊 Week 14 Summary

### What Makes Decision Trees Special? 🌳

**The Power of Interpretability:**
- Trees are **flowcharts we can see and understand**
- Every decision is **explainable** with clear rules
- Non-technical people can **follow the logic**
- Perfect for **regulated industries** (healthcare, finance)

**The Flexibility:**
- Works with **any type of features** (numbers, categories)
- No need for **feature scaling or normalization**
- Handles **non-linear patterns** naturally
- Can do both **classification and regression**

**The Challenge:**
- **Overfits easily** without proper constraints
- **Unstable** - small data changes create different trees
- **Greedy algorithm** may miss optimal solution
- Needs **careful hyperparameter tuning**

### Key Takeaways:

1. **Decision trees split data recursively** using if-then rules
2. **Gini and entropy** measure how mixed (impure) a node is
3. **Pure nodes** (one class) are the goal of splitting
4. **max_depth** is your primary overfitting control
5. **Feature importance** tells you what matters most
6. **Visualization** makes trees powerful communication tools
7. **Overfitting is common** - always compare train vs. test
8. **Trees are greedy** - first split affects all future splits

### The Decision Tree Algorithm in Plain English:

1. Start with all data at the root
2. Find the feature and split point that best separates classes
3. Create branches for each side of the split
4. Repeat steps 2-3 for each branch (recursively)
5. Stop when nodes are pure, max depth reached, or too few samples
6. Predict: follow path from root to leaf for new data

### Real-World Wisdom:

**Start Simple:**
- Begin with max_depth=3 or 5
- Visualize the tree to understand it
- Check if it makes logical sense

**Then Optimize:**
- Try different max_depth values
- Experiment with min_samples parameters
- Compare training vs. test accuracy
- Choose based on test performance

**Finally Validate:**
- Use cross-validation for robust evaluation
- Check feature importance for sanity
- Visualize decision boundaries if possible
- Test on completely new data

---

## 🆚 Decision Trees vs. KNN

Since you already know KNN, here's a direct comparison:

### When Decision Trees are Better:

✅ **Need interpretability** - can explain every decision  
✅ **Have categorical features** - handles them naturally  
✅ **Want feature importance** - automatic ranking  
✅ **Have mixed feature types** - no scaling needed  
✅ **Need fast predictions** - just follow the path  
✅ **Want automatic feature selection** - uses important ones  

### When KNN is Better:

✅ **Have smooth decision boundaries** - no rectangular regions  
✅ **Small dataset** - no training time  
✅ **Need to update easily** - just add new points  
✅ **Simple relationships** - less prone to overfitting  

### The Big Difference:

**KNN:** "Who are your neighbors? You're probably like them!"
- Instance-based learning
- No explicit model
- Slow predictions

**Decision Trees:** "Let me ask you questions until I know who you are!"
- Model-based learning
- Explicit decision rules
- Fast predictions



---

## 🎯 Final Thoughts

### The Tree Mindset 🌳

**Decision trees teach you to think like a machine learning algorithm:**
- Break problems into yes/no questions
- Find the most informative questions to ask
- Organize knowledge hierarchically
- Make explicit, traceable decisions

**This mindset applies beyond ML:**
- Debugging code (decision tree of what could be wrong)
- Troubleshooting (systematic elimination)
- Planning (breaking goals into steps)
- Problem-solving (divide and conquer)

### Remember:

**Trees are powerful but need care:**
- ⚠️ Will overfit if you let them
- ⚡ But easy to control with hyperparameters
- 🎨 Best for interpretability and explanation
- 🚀 Foundation for ensemble methods coming next

**Great data scientists:**
- Know when trees are the right tool
- Can tune hyperparameters effectively
- Balance complexity with generalization
- Explain model decisions clearly

**You now have that knowledge! 🎓**

---

**Important Reminders:**
- Complete the marketing_data Colab notebook
- Work through the breast cancer assignment (25 points!)
- Experiment with different hyperparameters
- Visualize your trees to build intuition
- Compare training vs. test accuracy
- Ask questions in discussions
- Practice interpreting tree diagrams

---

*Questions? Review the presentation, work through the notebook, and experiment with the breast cancer assignment. Tree mastery comes from seeing how they split data and controlling their complexity!*

---

**Happy Tree Building! 🌳**

*"A decision tree is just a series of smart questions - and you're learning to ask them!"*
