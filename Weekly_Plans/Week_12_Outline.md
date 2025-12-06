# Week 12: K-Means Clustering

**Course:** Python for Machine Learning  
**Instructor:** Siva Jasthi (Siva.Jasthi@metrostate.edu)  
**Week:** 12  
**Topic:** K-Means Clustering Algorithm

---

## 📚 Learning Objectives

By the end of this week, students will be able to:

1. Understand the fundamental concept of unsupervised learning and clustering
2. Explain how K-Means algorithm groups similar data points
3. Implement K-Means clustering using both basic Python and scikit-learn
4. Apply the Elbow Method to determine the optimal number of clusters
5. Evaluate clustering results using metrics (inertia, silhouette score)
6. Visualize clusters and centroids effectively
7. Apply K-Means to real-world datasets (customer segmentation, image compression)
8. Understand the key differences between supervised (KNN) and unsupervised (K-Means) learning

---

## 📄 Review: ML Algorithms Covered So Far

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

**K-Nearest Neighbors (KNN)**
- Instance-based learning (lazy learning)
- Classification using similarity/distance
- Supervised learning with labeled data
- Majority voting mechanism

### This Week's Focus:

**K-Means Clustering**
- Unsupervised learning (no labels needed!)
- Grouping similar data points together
- Finding natural patterns in data
- Iterative centroid-based algorithm
- **Key Difference:** Discovers groups rather than predicting labels

---

## 📖 Week 12 Topics

### 1. Introduction to Clustering
- What is unsupervised learning?
- Difference between classification (KNN) and clustering (K-Means)
- What is K-Means Clustering?
- When to use K-Means
- Real-world clustering applications

### 2. Core Concepts

**The K-Means Algorithm:**
- What K represents in K-Means (number of clusters)
- Centroids: The center points of clusters
- Algorithm steps: Initialize → Assign → Update → Converge
- Why it's called "K-Means"

**The Iterative Process:**
- Random initialization of centroids
- Assigning points to nearest centroid
- Updating centroids to cluster means
- Convergence criteria

**Distance Calculations:**
- Euclidean distance (same as KNN!)
- Why distance matters in clustering
- Inertia (Within-Cluster Sum of Squares)

### 3. Choosing the Right K

**The Elbow Method:**
- Plotting K vs. Inertia
- Finding the "elbow" point
- Balancing cluster quality and quantity

**Other Considerations:**
- Domain knowledge
- Business requirements
- Silhouette analysis
- Computational constraints

### 4. Implementation

**Basic Python Implementation:**
- Building K-Means from scratch
- Understanding each algorithm step
- Manual centroid calculation

**Scikit-learn Implementation:**
- Using `KMeans` class
- Fitting and predicting clusters
- Accessing centroids and labels
- Model parameters and attributes

### 5. Feature Scaling & Preprocessing
- Why scaling is critical for K-Means
- StandardScaler and normalization
- Impact of feature scales on clustering
- Handling different feature ranges

### 6. Evaluation Metrics

**Inertia (WCSS):**
- Within-Cluster Sum of Squares
- Lower is better
- Used in Elbow Method

**Silhouette Score:**
- Measures cluster quality
- Range: -1 to 1 (higher is better)
- Per-sample and average scores

### 7. Visualization Techniques
- Scatter plots with colored clusters
- Centroid markers
- Decision boundaries
- Elbow curves
- 2D projections of high-dimensional data (PCA)
- Cluster size comparison

---

## 📚 Required Materials

### 📊 Presentation
**K-Means Clustering**
- [ML_K_Means_Clustering.md](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_K_Means_Clustering.md)
- Comprehensive guide covering K-Means theory, examples, and applications
- Review before starting hands-on work

### 💻 Hands-On Notebook
**K-Means Clustering Colab Notebook**
- [ML_K_Means_Clustering.ipynb](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Colab_Notebooks/ML_K_Means_Clustering.ipynb)
- Interactive code examples and exercises
- Includes:
  - Basic Python K-Means implementation
  - Customer segmentation example
  - Image color quantization
  - Elbow Method implementation
  - Feature scaling demonstrations
  - Visualization code

### ✅ Required Quiz
**K-Means Clustering Knowledge Assessment**
- [ML_KMC_Quiz.html](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Play/ML_KMC_Quiz.html)
- Test your understanding of K-Means concepts
- **REQUIRED:** Take the quiz and submit a screenshot of your score to the dropbox
- Make sure your screenshot clearly shows your score

---

## 🎯 Key Concepts to Master

### ✅ Understanding Checklist:
- [ ] Can explain what "K" represents in K-Means
- [ ] Understand the difference between supervised and unsupervised learning
- [ ] Can describe the four steps of K-Means algorithm
- [ ] Know what a centroid is and how it's calculated
- [ ] Can explain the Elbow Method for choosing K
- [ ] Understand why feature scaling is crucial for K-Means
- [ ] Can interpret inertia and silhouette scores
- [ ] Know when to use K-Means vs. KNN
- [ ] Can explain convergence in K-Means
- [ ] Recognize the pros and cons of K-Means

---

## 💡 Practical Applications

### Where is K-Means Used?

1. **Customer Segmentation**
   - Grouping customers by shopping behavior
   - Targeted marketing campaigns
   - Personalized recommendations
   - Understanding customer types

2. **Image Compression**
   - Reducing colors in images
   - Color quantization
   - File size reduction
   - Efficient storage

3. **Document Classification**
   - Organizing articles by topic
   - News categorization
   - Content recommendation
   - Automatic tagging

4. **Anomaly Detection**
   - Fraud detection
   - Network intrusion detection
   - Quality control
   - Outlier identification

5. **Social Network Analysis**
   - Finding communities
   - User grouping
   - Influence analysis
   - Pattern discovery

6. **Medical Imaging**
   - Image segmentation
   - Tissue classification
   - Disease pattern detection
   - Diagnostic support

7. **Genomics**
   - Gene expression analysis
   - Species classification
   - DNA pattern recognition
   - Biological data organization

---

## 🤔 Discussion Questions

1. **What's the key difference between KNN and K-Means?**
   - Think about: labels, purpose, output, when to use each
   - Both use "K" but for very different purposes!

2. **Why is K-Means called "unsupervised" learning?**
   - Consider: what data do we need? What does the algorithm discover?
   - How is this different from classification?

3. **What happens if we choose K=1? K=N (number of points)?**
   - Think about extreme cases
   - What would the clusters look like?

4. **Why might K-Means give different results each time?**
   - Consider the initialization step
   - What role does randomness play?

5. **When would K-Means fail or perform poorly?**
   - Think about: non-spherical clusters, different sized clusters, outliers
   - What assumptions does K-Means make?

6. **How do you decide the "right" number of clusters?**
   - Is the Elbow Method always clear?
   - What other factors should you consider?

---

## 💻 Code Snippets to Master

### Basic K-Means Implementation
```python
import numpy as np

def initialize_centroids(data, k):
    """Randomly select k data points as initial centroids"""
    indices = np.random.choice(len(data), k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    """Assign each point to nearest centroid"""
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(data, labels, k):
    """Calculate new centroids as mean of assigned points"""
    return np.array([data[labels == i].mean(axis=0) for i in range(k)])

def kmeans(data, k, max_iters=100):
    """K-Means clustering algorithm"""
    centroids = initialize_centroids(data, k)
    
    for _ in range(max_iters):
        old_centroids = centroids.copy()
        labels = assign_clusters(data, centroids)
        centroids = update_centroids(data, labels, k)
        
        # Check convergence
        if np.all(centroids == old_centroids):
            break
    
    return labels, centroids
```

### Scikit-learn Implementation
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Access results
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

# Evaluate
silhouette = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {silhouette:.3f}")
```

### Elbow Method
```python
import matplotlib.pyplot as plt

# Test different K values
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method For Optimal K')
plt.grid(True)
plt.show()
```

### Visualization
```python
import matplotlib.pyplot as plt

# Plot clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, 
            cmap='viridis', alpha=0.6, edgecolors='black')

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], 
            c='red', marker='X', s=300, edgecolors='black', 
            linewidth=2, label='Centroids')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering Results')
plt.legend()
plt.colorbar(label='Cluster')
plt.show()
```

---

## 📚 Additional Resources

### Recommended Reading:
- **Scikit-learn Documentation:** [K-Means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- **Understanding Cluster Quality:** Silhouette analysis and validation
- **Feature Scaling Guide:** Why and how to normalize data

### Video Tutorials:
- StatQuest: K-Means Clustering, Clearly Explained
- Visualizing K-Means Algorithm Step by Step
- When K-Means Works (and When It Doesn't)

### Practice Datasets:
- **Iris Dataset** (scikit-learn) - Classic clustering example
- **Mall Customer Dataset** - Customer segmentation practice
- **Wine Dataset** (scikit-learn) - Multi-feature clustering
- **Images** - Color quantization experiments

---

## 🎓 Tips for Success

### 1. Understand the Algorithm Flow
- Watch the centroids move iteration by iteration
- Visualize how points get reassigned
- See when convergence happens
- Build intuition for the process

### 2. Master Feature Scaling
- Always scale features before K-Means!
- Compare results with and without scaling
- Understand why it matters
- Use StandardScaler or MinMaxScaler

### 3. Use the Elbow Method Wisely
- The "elbow" isn't always obvious
- Look for the point of diminishing returns
- Consider business context too
- Try silhouette analysis as well

### 4. Visualize Everything
- Plot your data first (using PCA if needed)
- Color-code clusters
- Show centroids clearly
- Create elbow curves
- Compare different K values visually

### 5. Experiment with K Values
- Try K=2, 3, 4, 5, etc.
- See how clusters change
- Observe stability of results
- Document what you learn

### 6. Compare with KNN
- Both use K, but differently!
- One is supervised, one is unsupervised
- Understand when to use each
- Practice explaining the difference

### 7. Think About Real Applications
- How would you segment customers?
- What features matter for your clusters?
- How many clusters make sense?
- Can you explain clusters to non-technical people?

---

## ✅ Week 12 Checklist

By the end of this week, make sure you have:

- [ ] Reviewed the K-Means Clustering presentation
- [ ] Understood the difference between supervised and unsupervised learning
- [ ] Completed all sections of the Colab notebook
- [ ] Implemented K-Means from scratch in basic Python
- [ ] Used scikit-learn's KMeans class
- [ ] Applied feature scaling before clustering
- [ ] Practiced the Elbow Method to find optimal K
- [ ] Worked with customer segmentation examples
- [ ] Experimented with image color quantization
- [ ] Created visualizations (scatter plots with clusters, elbow curves)
- [ ] Calculated and interpreted inertia and silhouette scores
- [ ] Compared results with different K values
- [ ] **Taken the K-Means Quiz and submitted screenshot to dropbox**
- [ ] Can explain K-Means in your own words
- [ ] Understand when to use K-Means vs. KNN
- [ ] Participated in class discussions

---

## 📝 Required Submission

### K-Means Clustering Quiz
**Due:** End of Week 12

**Instructions:**
1. Open the quiz: [ML_KMC_Quiz.html](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Play/ML_KMC_Quiz.html)
2. Complete all questions
3. Take a screenshot of your final score
4. **Submit the screenshot to the dropbox**

---

## 🚀 Challenge Problems (Optional)

For students who want extra practice and deeper understanding:

### 1. Compare K-Means Initialization Methods
- Try random initialization multiple times
- Implement K-Means++ initialization
- Compare stability of results
- Analyze convergence speed

### 2. Build an Image Color Compressor
- Load a colorful image
- Apply K-Means to RGB values
- Reduce to 16, 8, or 4 colors
- Compare file sizes and visual quality
- Create before/after visualizations

### 3. Customer Segmentation Project
- Find a real customer dataset (Kaggle)
- Clean and prepare the data
- Scale features appropriately
- Use Elbow Method to find optimal K
- Profile each customer segment
- Create business recommendations

### 4. Explore Non-Spherical Clusters
- Generate moon-shaped or ring-shaped data
- Apply K-Means and observe failures
- Try DBSCAN or hierarchical clustering
- Compare different clustering algorithms
- Understand K-Means limitations

### 5. Silhouette Analysis Deep Dive
- Calculate silhouette scores for multiple K values
- Plot silhouette coefficients for each sample
- Create silhouette plots
- Use this to validate your K choice
- Compare with Elbow Method results

### 6. Mini-Batch K-Means
- Learn about mini-batch variation
- Compare speed with standard K-Means
- Test on large datasets
- Analyze accuracy tradeoffs
- Determine when to use which version

---

## 🎉 Learning Outcomes

By completing Week 12, you will have:

✅ Understood the fundamentals of unsupervised learning  
✅ Mastered the K-Means clustering algorithm  
✅ Implemented K-Means from scratch and with scikit-learn  
✅ Learned to choose optimal K using Elbow Method  
✅ Applied proper feature scaling techniques  
✅ Evaluated clustering quality with multiple metrics  
✅ Created professional cluster visualizations  
✅ Applied K-Means to real-world problems  
✅ Distinguished between clustering and classification  
✅ Added unsupervised learning to your ML toolkit  

---

## 📊 Week 12 Summary

### What Makes K-Means Special?
- **Unsupervised:** Discovers patterns without labels
- **Efficient:** Fast and scalable to large datasets
- **Intuitive:** Easy to understand and explain
- **Versatile:** Works across many domains

### Key Takeaways:
1. **K-Means finds groups:** It discovers natural clusters in data
2. **No labels needed:** Unlike KNN, it works with unlabeled data
3. **K matters:** Choosing the right K is both art and science
4. **Scaling is critical:** Always normalize features first
5. **Iterative process:** Algorithm converges to stable clusters
6. **Random initialization:** Results can vary, use multiple runs
7. **Has limitations:** Assumes spherical clusters of similar size

### KNN vs. K-Means Quick Reference:

| Aspect | KNN (Week 11) | K-Means (Week 12) |
|--------|---------------|-------------------|
| **Type** | Supervised | Unsupervised |
| **K Meaning** | # of neighbors | # of clusters |
| **Needs Labels?** | Yes | No |
| **Purpose** | Classification | Grouping |
| **Output** | Class label | Cluster assignment |
| **Training** | Lazy (none) | Iterative |

### Remember:
K-Means helps us discover hidden patterns and natural groupings in data - it's like finding communities without being told they exist!

---

**Happy Learning! 🎓**

*"K-Means doesn't predict classes - it discovers them!"*

---

**Important Reminders:**
- Complete all activities in the Colab notebook
- Take the quiz and submit your screenshot
- Always scale your features before clustering
- Try different K values and use the Elbow Method
- Visualize your clusters to understand results
- Ask questions if you're confused
- Help your classmates when you can
- Enjoy discovering patterns in data!

---

## 🔍 Connecting the Dots: Week 11 → Week 12

### From KNN to K-Means:

**What's Similar:**
- Both use the letter "K"
- Both use distance calculations (Euclidean)
- Both can be visualized with scatter plots
- Both are fundamental ML algorithms

**What's Different:**
- **KNN (Week 11):** "Tell me what this new point is" (classification)
- **K-Means (Week 12):** "Show me what groups exist" (clustering)

**The Big Picture:**
- You now know both supervised and unsupervised learning!
- KNN: Learn from labeled examples → predict new examples
- K-Means: Find patterns in data → discover groups

This week, you're moving from **prediction** to **discovery**! 🎨

---

**Final Thought:**
Clustering is like organizing a messy room - you don't know the categories beforehand, but you group similar items together. K-Means does this automatically with data!
