# Week 7 Outline: Introduction to scikit-learn

## Course Information
**Topics:** 
1. Introduction to scikit-learn
2. Playing with toy datasets

---

## Learning Objectives
By the end of this week, students will be able to:
- Understand what scikit-learn is and why it's important in machine learning
- Navigate the scikit-learn library and documentation
- Load and explore built-in toy datasets
- Differentiate between classification, regression, and clustering problems
- Apply basic scikit-learn functions to work with datasets
- Understand the structure and features of common toy datasets

---

## Topics Covered

### 1. Introduction to scikit-learn
**Reference Material:** [ML scikit-learn Introduction](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_scikit_learn_introduction.md)

Understanding Python's premier machine learning library:
- What is scikit-learn?
- Why scikit-learn is the industry standard for ML in Python
- Key features and capabilities
- Overview of scikit-learn's API and design
- Common use cases and applications

#### Core Components of scikit-learn
- **Estimators:** Objects that learn from data (models)
- **Transformers:** Objects that transform data
- **Predictors:** Objects that make predictions
- **Datasets:** Built-in datasets for learning and testing

### 2. Playing with Toy Datasets
**Reference Material:** [ML scikit-learn Toy Datasets](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_scikit_toy_datasets.md)

Exploring built-in datasets for practice:

#### What are Toy Datasets?
- Small, clean datasets perfect for learning
- Pre-loaded in scikit-learn
- Real-world data with known characteristics
- No need to download or clean data

#### Popular Toy Datasets
- **Iris Dataset:** Classification of iris flowers (3 species)
- **Wine Dataset:** Classification of wine cultivars
- **Breast Cancer Dataset:** Binary classification for diagnosis
- **Diabetes Dataset:** Regression for disease progression
- **Boston Housing Dataset:** Regression for housing prices
- **Digits Dataset:** Classification of handwritten digits

#### Working with Datasets
- Loading datasets using `load_*()` functions
- Understanding dataset structure (data, target, feature_names, target_names)
- Exploring features and targets
- Basic data inspection and visualization

---

## Review Materials

### Core Presentations
1. **[Introduction to scikit-learn](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_scikit_learn_introduction.md)**
   - Library overview and capabilities
   - Installation and setup
   - Basic API structure
   
2. **[Toy Datasets in scikit-learn](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_scikit_toy_datasets.md)**
   - Available datasets
   - How to load and explore datasets
   - Dataset characteristics and use cases

---

## Exploration Activities

### Interactive Exploration: scikit-learn Official Documentation
**Website:** [scikit-learn.org](https://scikit-learn.org/stable/)

#### What to Explore:

1. **Classification Models**
   - Browse available classification algorithms
   - Understand when to use different classifiers
   - Review example applications

2. **Clustering Models**
   - Explore unsupervised learning algorithms
   - Understand grouping and pattern discovery
   - See clustering in action

3. **Regression Models**
   - Review regression algorithms
   - **Special Focus:** Ordinary Least Squares (OLS)
     - Linear regression fundamentals
     - How OLS works
     - When to use linear regression
     - Understanding the math behind OLS

#### Exploration Guidelines:
- Navigate through the scikit-learn homepage
- Click on "User Guide" for detailed explanations
- Check out example galleries for visual understanding
- Read about model assumptions and limitations
- Pay special attention to the Ordinary Least Squares documentation

---

## Assignment Due This Week

### Assignment 5: Exploring scikit-learn Toy Datasets
- **Points:** 25 points
- **Description:** Hands-on exploration of scikit-learn's built-in datasets
- **Assignment Link:** [Python ML Assignment - Exploring scikit-learn Datasets](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Assignments/Python_ML_Assignment_Exploring_scikitlearn_datasets.ipynb)

#### What to Complete:
- Load multiple toy datasets from scikit-learn
- Explore dataset structures and features
- Analyze feature distributions
- Visualize data relationships
- Understand the difference between classification and regression datasets
- Practice basic data manipulation with scikit-learn

---

## Key Takeaways

✓ scikit-learn is the most popular machine learning library in Python  
✓ Toy datasets provide perfect practice data for learning ML concepts  
✓ All scikit-learn datasets have consistent structure (data, target, features)  
✓ Classification predicts categories, regression predicts continuous values  
✓ Clustering finds patterns without labels (unsupervised learning)  
✓ Ordinary Least Squares is a fundamental regression technique  
✓ The scikit-learn API follows a consistent pattern across all models  

---

## Class Activities

### In-Class Exploration
- Live demonstration of loading and exploring toy datasets
- Walkthrough of scikit-learn documentation
- Comparison of different dataset types
- Discussion of real-world applications

### Hands-On Practice
- Load the Iris dataset together
- Explore features and targets
- Create simple visualizations
- Discuss dataset characteristics

### Discussion Questions
- What makes scikit-learn different from other ML libraries?
- Why are toy datasets useful for learning?
- How do you choose between classification and regression?
- What is the difference between supervised and unsupervised learning?
- When would you use Ordinary Least Squares regression?

---

## Next Steps
- Review both presentation materials before class
- Create a scikit-learn account bookmark for future reference
- Explore the scikit-learn website, especially Classification, Clustering, and Regression sections
- Read about Ordinary Least Squares in the Regression section
- Complete Assignment 5 (Exploring scikit-learn Datasets) by October 24
- Install scikit-learn if not already installed: `pip install scikit-learn`
- Think about which toy dataset interests you most and why

---

## Additional Resources
- [scikit-learn Introduction](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_scikit_learn_introduction.md)
- [Toy Datasets Overview](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_scikit_toy_datasets.md)
- [scikit-learn Official Website](https://scikit-learn.org/stable/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [scikit-learn Datasets Documentation](https://scikit-learn.org/stable/datasets.html)
- [Assignment 5 Notebook](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Assignments/Python_ML_Assignment_Exploring_scikitlearn_datasets.ipynb)

---

## Connection to Previous Weeks
This week builds directly on Week 6's concepts:
- The ML workflow includes using libraries like scikit-learn
- Toy datasets help us practice training/validation/test splits
- We'll apply bias/variance concepts to real models
- Understanding datasets is crucial for feature engineering
- scikit-learn makes the entire ML process more accessible
