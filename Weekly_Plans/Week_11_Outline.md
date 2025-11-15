# Week 11: K-Nearest Neighbors (KNN) Classification

**Course:** Python for Machine Learning  
**Instructor:** Siva Jasthi (Siva.Jasthi@metrostate.edu)  
**Week:** 11  
**Topic:** K-Nearest Neighbors (KNN) Algorithm

---

## 📚 Learning Objectives

By the end of this week, students will be able to:

1. Understand the fundamental concept of K-Nearest Neighbors algorithm
2. Explain how KNN makes predictions using distance metrics
3. Implement KNN classification using both basic Python and scikit-learn
4. Tune hyperparameters (K value, distance metrics) for optimal performance
5. Evaluate KNN models using accuracy, confusion matrices, and cross-validation
6. Apply KNN to real-world datasets (Breast Cancer, Wine classification)
7. Visualize KNN decision boundaries and classification results

---

## 🔄 Review: ML Algorithms Covered So Far

### Previously Learned:

**Simple Linear Regression**
- Predicting continuous values with one feature
- Understanding slope, intercept, and best-fit lines
- Linear relationships between variables

**Multiple Linear Regression**
- Predicting with multiple features
- Feature importance and coefficients
- Working with multidimensional data

**Logistic Regression**
- Binary classification problems
- Sigmoid function and probability predictions
- Decision boundaries for classification

### This Week's Focus:

**K-Nearest Neighbors (KNN)**
- Instance-based learning (lazy learning)
- Classification using similarity/distance
- Non-parametric algorithm
- Different approach: no training phase, makes decisions at prediction time

---

## 📖 Week 11 Topics

### 1. Introduction to KNN
- What is K-Nearest Neighbors?
- How KNN differs from regression models
- Instance-based vs. model-based learning
- When to use KNN

### 2. Core Concepts

**Distance Metrics:**
- Euclidean Distance (straight-line distance)
- Manhattan Distance (city-block distance)
- Minkowski Distance (generalized metric)

**The K Parameter:**
- What does K represent?
- Choosing the right K value
- Impact of K on model performance
- Why we typically use odd K values

**Classification Process:**
- Finding nearest neighbors
- Majority voting mechanism
- Handling ties in voting

### 3. Implementation

**Basic Python Implementation:**
- Calculating distances manually
- Implementing KNN from scratch
- Understanding the algorithm step-by-step

**Scikit-learn Implementation:**
- Using `KNeighborsClassifier`
- Training and predicting
- Model evaluation techniques

### 4. Model Optimization

**Hyperparameter Tuning:**
- Finding optimal K value through experimentation
- Cross-validation techniques
- Comparing different distance metrics

**Model Evaluation:**
- Accuracy scores
- Confusion matrices
- Classification reports
- Understanding Precision, Recall, F1-score

### 5. Visualization Techniques
- Scatter plots showing data points and neighbors
- Decision boundaries
- K value vs. Accuracy graphs
- Confusion matrix heatmaps
- PCA for high-dimensional visualization

---

## 📚 Required Materials

### 📊 Presentation
**K-Nearest Neighbors Classification**
- [ML_K_Nearest_Neighbors_Classification.pdf](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_K_Nearest_Neighbors_Classification.pdf)
- Comprehensive slides covering KNN theory and examples
- Review before starting hands-on work

### 📝 Introduction Document
**Introduction to KNN**
- [ML_K_Nearest_Neighbors_Introduction.md](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_K_Nearest_Neighbors_Introduction.md)
- Detailed explanation of KNN concepts
- Background reading and fundamentals

### 💻 Hands-On Notebook
**KNN Classifier Colab Notebook**
- [ML_KNN_Classifier.ipynb](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Colab_Notebooks/ML_KNN_Classifier.ipynb)
- Interactive code examples and exercises
- Includes:
  - Basic Python KNN implementation
  - Breast Cancer dataset classification
  - Wine dataset classification
  - Hyperparameter tuning examples
  - Visualization code

### ✅ Required Quiz
**KNN Knowledge Assessment**
- [ML_KNN_Quiz.html](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Play/ML_KNN_Quiz.html)
- Test your understanding of KNN concepts
- **REQUIRED:** Take the quiz and submit a screenshot of your score to the dropbox
- Make sure your screenshot clearly shows your score

---

## 🎯 Key Concepts to Master

### ✅ Understanding Checklist:
- [ ] Can explain what "K" represents in KNN
- [ ] Can calculate Euclidean distance between two points
- [ ] Understand how majority voting works
- [ ] Can explain the difference between K=1, K=3, and K=5
- [ ] Know when to use KNN vs. other algorithms
- [ ] Can interpret a confusion matrix
- [ ] Understand the concept of cross-validation
- [ ] Can tune hyperparameters (K value, distance metric)
- [ ] Recognize the pros and cons of KNN

---

## 💡 Practical Applications

### Where is KNN Used?

1. **Recommendation Systems**
   - "Customers who bought X also bought Y"
   - Finding similar products or users
   - Netflix movie recommendations
   - Amazon product suggestions

2. **Medical Diagnosis**
   - Classifying diseases based on symptoms
   - Cancer detection (as in our Breast Cancer dataset)
   - Predicting patient outcomes

3. **Image Recognition**
   - Handwriting recognition
   - Face detection and recognition
   - Object classification in photos

4. **Credit Scoring**
   - Determining loan eligibility
   - Risk assessment
   - Fraud detection

5. **Pattern Recognition**
   - Anomaly detection
   - Quality control in manufacturing
   - Predictive maintenance

---

## 🤔 Discussion Questions

1. **When would you choose KNN over Logistic Regression?**
   - Think about: dataset size, number of features, interpretability, computational cost

2. **What are the advantages and disadvantages of KNN?**
   - Advantages: Simple to understand, no training phase, works well with non-linear data
   - Disadvantages: Slow prediction, sensitive to irrelevant features, memory intensive

3. **How does the choice of K affect model performance?**
   - Small K (e.g., K=1): More sensitive to noise, overfitting
   - Large K: Smoother boundaries, possible underfitting
   - Explore the bias-variance tradeoff

4. **Why is feature scaling important for KNN?**
   - Consider how distance calculations work
   - What happens when features have different scales?

5. **What happens if K equals the total number of data points?**
   - Try to predict what the model would always output

---

## 💻 Code Snippets to Master

### Basic KNN Implementation
```python
def euclidean_distance(x1, x2):
    """Calculate Euclidean distance between two points"""
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i])**2
    return distance ** 0.5

def knn(data, query, k):
    """Find k nearest neighbors"""
    distances = []
    for point in data:
        dist = euclidean_distance(point[:2], query)
        distances.append((dist, point[2]))
    
    distances.sort()  # Sort by distance
    neighbors = [distances[i][1] for i in range(k)]
    return neighbors

def majority_vote(labels):
    """Determine the most common label"""
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    return max(label_counts, key=label_counts.get)
```

### Scikit-learn Implementation
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
```

### Finding Best K with Cross-Validation
```python
from sklearn.model_selection import cross_val_score

k_values = range(3, 22, 2)
mean_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    mean_scores.append(scores.mean())

# Find best K
best_k = k_values[mean_scores.index(max(mean_scores))]
print(f"Best K: {best_k}")
```

---

## 📚 Additional Resources

### Recommended Reading:
- **Scikit-learn Documentation:** [K-Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- **Visual Introduction to KNN:** Interactive explanations with graphics
- **Distance Metrics Explained:** Understanding different distance calculations

### Video Tutorials:
- StatQuest: K-Nearest Neighbors, Clearly Explained
- Python KNN Tutorial: Building from Scratch
- Choosing the Right K Value

### Practice Datasets:
- **Iris Dataset** (scikit-learn) - Classic 3-class classification
- **Digits Dataset** (scikit-learn) - Handwritten digit recognition
- **Wine Dataset** (included in notebook) - Multi-class wine classification
- **Breast Cancer Dataset** (included in notebook) - Binary medical classification

---

## 🎓 Tips for Success

### 1. Start with Simple Examples
- Begin with 2D visualizations to understand what KNN is doing
- Plot the data points and see which neighbors are closest
- Visualize how decision boundaries form

### 2. Experiment with Parameters
- Try different K values (3, 5, 7, 9, 11, etc.)
- Compare distance metrics (Euclidean, Manhattan, Minkowski)
- Observe how results change
- Keep notes on what works best

### 3. Understand the Math
- Calculate distances by hand for a few examples
- Verify your manual calculations match the code output
- This builds intuition for how the algorithm works

### 4. Use Visualizations
- Create scatter plots to see data distributions
- Plot decision boundaries to understand classification
- Generate confusion matrices to evaluate performance
- Graph K vs. Accuracy to find optimal K

### 5. Compare with Other Algorithms
- How does KNN compare to Logistic Regression on the same dataset?
- When would you choose one over the other?
- What are the tradeoffs?

### 6. Test Your Understanding
- Can you explain KNN to a friend who doesn't know programming?
- Can you implement a simple version from scratch?
- Do you understand why we use cross-validation?

---

## ✅ Week 11 Checklist

By the end of this week, make sure you have:

- [ ] Reviewed the KNN presentation slides
- [ ] Read the KNN introduction document
- [ ] Completed all sections of the Colab notebook
- [ ] Implemented KNN from scratch in basic Python
- [ ] Used scikit-learn's KNeighborsClassifier
- [ ] Worked with the Breast Cancer dataset
- [ ] Experimented with different K values
- [ ] Compared different distance metrics
- [ ] Applied KNN to the Wine dataset
- [ ] Created visualizations (scatter plots, confusion matrices, K vs. Accuracy)
- [ ] Understood the concept of cross-validation
- [ ] **Taken the KNN Quiz and submitted screenshot to dropbox**
- [ ] Can explain KNN concepts in your own words
- [ ] Participated in class discussions

---

## 📝 Required Submission

### KNN Quiz
**Due:** End of Week 11

**Instructions:**
1. Open the quiz: [ML_KNN_Quiz.html](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Play/ML_KNN_Quiz.html)
2. Complete all questions
3. Take a screenshot of your final score
4. **Submit the screenshot to the dropbox**
---

## 🚀 Challenge Problems (Optional)

For students who want extra practice and deeper understanding:

### 1. Implement Weighted KNN
- Closer neighbors have more influence on the prediction
- Use inverse distance weighting: weight = 1 / distance
- Compare performance with standard KNN

### 2. Create KNN for Regression
- Instead of classification, predict continuous values
- Use mean of K nearest neighbors' values
- Test on a housing prices dataset

### 3. Build a Digit Recognition System
- Use the MNIST dataset (handwritten digits 0-9)
- Apply KNN to classify digits
- Experiment with different K values
- Create visualizations of misclassified digits

### 4. Feature Scaling Experiment
- Apply StandardScaler or MinMaxScaler to features
- Compare KNN performance with and without scaling
- Explain why scaling makes a difference

### 5. Distance Metric Comparison Study
- Test Euclidean, Manhattan, Chebyshev, and other metrics
- Create a comparison table
- Analyze which metric works best for which dataset

---

## 🎉 Learning Outcomes

By completing Week 11, you will have:

✅ Built a solid understanding of instance-based learning  
✅ Implemented KNN from scratch and with scikit-learn  
✅ Mastered hyperparameter tuning techniques  
✅ Applied KNN to real-world classification problems  
✅ Created professional visualizations  
✅ Learned to evaluate and compare ML models  
✅ Added another powerful algorithm to your ML toolkit  
✅ Understood when to use KNN vs. other algorithms  

---

## 📊 Week 11 Summary

### What Makes KNN Special?
- **Simple & Intuitive:** Easy to understand and explain
- **No Training Phase:** Lazy learning approach
- **Flexible:** Works with any distance metric
- **Non-parametric:** Makes no assumptions about data distribution

### Key Takeaways:
1. **K matters:** Choosing the right K is crucial for performance
2. **Distance metrics:** Different metrics work better for different data
3. **Scaling is important:** Features should be on similar scales
4. **Computational cost:** Prediction can be slow with large datasets
5. **Visualization helps:** Seeing the data helps understand KNN decisions

### Remember:
KNN is all about finding similar examples and learning from them - just like how we learn by looking at examples and finding patterns!

---

**Happy Learning! 🎓**

*"In KNN, your prediction is only as good as your neighbors. Choose wisely!"*

---

**Important Reminders:**
- Complete all activities in the Colab notebook
- Take the quiz and submit your screenshot
- Ask questions if you're confused
- Help your classmates when you can
- Enjoy exploring this elegant algorithm!
