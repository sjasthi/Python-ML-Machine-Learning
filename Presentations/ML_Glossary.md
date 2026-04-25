# ML Glossary — Key Terms and Concepts
**Learn and Help – Python Machine Learning**
**learnandhelp.com | Academic Year 2025–2026**

---

> This glossary covers the terms and concepts from our 27-week course.
> Terms marked with a **★** are **cross-cutting concepts** — they apply to almost every ML algorithm you will ever use.

---

## How to Use This Glossary

- Use **Ctrl+F** (or Cmd+F on Mac) to search for any term
- Cross-cutting concepts are grouped first — read those carefully, they are the foundation
- Algorithm-specific terms follow, organized by topic
- Every definition includes a plain-English analogy to help it stick

---

## Table of Contents

1. [Cross-Cutting Concepts (The Big Ideas)](#cross-cutting-concepts-the-big-ideas)
2. [Data and Features](#data-and-features)
3. [Model Training and Evaluation](#model-training-and-evaluation)
4. [Regression](#regression)
5. [Classification](#classification)
6. [Clustering](#clustering)
7. [Decision Trees and Ensemble Methods](#decision-trees-and-ensemble-methods)
8. [Neural Networks and Deep Learning](#neural-networks-and-deep-learning)
9. [Natural Language Processing (NLP)](#natural-language-processing-nlp)
10. [Transformers and Large Language Models](#transformers-and-large-language-models)
11. [Tools and Platforms](#tools-and-platforms)
12. [Quick Reference — A to Z Index](#quick-reference--a-to-z-index)

---

## Cross-Cutting Concepts (The Big Ideas)

These concepts appear in almost every ML algorithm. Master these and everything else clicks faster.

---

### ★ Machine Learning (ML)
Teaching a computer to learn patterns from data, rather than writing every rule by hand.

> **Analogy:** Teaching a child to recognize dogs by showing them thousands of dog photos — not by writing a rulebook that says "has four legs, has fur, has a tail."

---

### ★ Algorithm
A step-by-step set of instructions a computer follows to learn patterns and make predictions. Different algorithms work better for different types of problems.

> **Analogy:** A recipe. Just as you might use a different recipe for cake vs. soup, you use different ML algorithms for different problems.

---

### ★ Model
The result of training an algorithm on data. Once trained, the model is what you use to make predictions on new, unseen data.

> **Analogy:** A model is like a student who has studied. The training data is the textbook; the model is the student's understanding after studying.

---

### ★ Feature
An individual measurable input variable used by the model to make a prediction. Also called a **predictor** or **independent variable**.

> **Analogy:** When predicting house prices, features might be: square footage, number of bedrooms, and ZIP code. Each is one piece of information the model uses.

---

### ★ Label / Target
The output the model is trying to predict. Also called the **dependent variable** or **y**.

> **Analogy:** In a housing dataset, if features are the house details, the label is the actual sale price.

---

### ★ Training Data
The portion of the dataset used to teach the model. The model sees both features and labels during training.

> **Analogy:** The practice problems a student studies before a test — they see the questions AND the answers.

---

### ★ Test Data
A separate portion of the dataset the model has never seen, used to evaluate how well it learned.

> **Analogy:** The actual exam — the student has never seen these specific questions before. This is the true measure of how well they learned.

---

### ★ Validation Data
A third split of data used to tune the model during development — separate from both training and test data.

> **Analogy:** Practice tests the student takes before the real exam, used to figure out which study strategies work best.

---

### ★ Overfitting
When a model learns the training data too well — including its noise and quirks — so it performs poorly on new data.

> **Analogy:** A student who memorizes every practice problem word-for-word but cannot solve any problem they have not seen before. They "overfit" the practice material.

---

### ★ Underfitting
When a model is too simple to capture the real patterns in the data, so it performs poorly on both training and test data.

> **Analogy:** A student who barely studied and just guesses "B" on every multiple-choice question. The model is too simple to be useful.

---

### ★ Bias (in ML models)
The error that comes from wrong assumptions in the model. A high-bias model is too simple — it underfits.

> **Analogy:** Assuming every fruit is an apple before you even look at it. You are bringing in a wrong assumption.

---

### ★ Variance (in ML models)
The error that comes from a model being too sensitive to small changes in training data. A high-variance model overfits.

> **Analogy:** A student whose test score changes wildly depending on which exact questions appeared. Their knowledge is unstable.

---

### ★ Bias-Variance Tradeoff
The balance between bias and variance. Reducing one often increases the other. The goal is to find the sweet spot with low error on unseen data.

> **Analogy:** A fishing net — too wide (high bias) and you catch nothing specific; too tight (high variance) and you only catch exactly what you practiced catching.

---

### ★ Cost Function / Loss Function
A mathematical formula that measures how wrong the model's predictions are. Training is the process of minimizing this number.

> **Analogy:** Your score on a golf game — the lower, the better. The model keeps adjusting itself trying to shoot a lower score.

---

### ★ Accuracy
The percentage of predictions the model gets right. Most useful when the classes are balanced.

> **Formula:** Accuracy = (Correct Predictions) / (Total Predictions) × 100%

---

### ★ Precision
Of all the times the model said "Yes," how often was it right?

> **Analogy:** If a metal detector beeps 10 times and finds real metal 8 times, precision = 80%.

---

### ★ Recall (Sensitivity)
Of all the actual "Yes" cases, how many did the model catch?

> **Analogy:** If there are 10 real metal objects and the detector finds 7 of them, recall = 70%.

---

### ★ F1 Score
The harmonic mean of Precision and Recall. Useful when you need to balance both.

> **When to use it:** When false positives AND false negatives both matter (e.g., disease detection).

---

### ★ Confusion Matrix
A table showing how many predictions were correct and incorrect, broken down by class.

```
                Predicted: Yes    Predicted: No
Actual: Yes     True Positive     False Negative
Actual: No      False Positive    True Negative
```

---

### ★ Cross-Validation
A technique to evaluate a model more reliably by training and testing it on multiple different splits of the data.

> **Analogy:** Instead of studying from one textbook and taking one test, you study from 5 different textbooks and take 5 different tests. Your average score is more trustworthy.

---

### ★ Hyperparameter
A setting you choose before training begins that controls how the algorithm learns. Different from parameters, which the model learns on its own.

> **Examples:** The number of neighbors in KNN (`k`), the max depth of a Decision Tree, the learning rate in a Neural Network.
> **Analogy:** The temperature setting on an oven. You set it before cooking; the oven does not choose it for you.

---

### ★ Parameter
A value the model learns automatically from training data (not set by you).

> **Examples:** The slope and intercept in Linear Regression; the weights in a Neural Network.

---

### ★ Generalization
A model's ability to perform well on new, unseen data — not just the data it was trained on. This is the ultimate goal.

---

### ★ Supervised Learning
A type of ML where the model is trained on labeled data (features + correct answers).

> **Examples from class:** Linear Regression, KNN, Decision Trees, Random Forest, Naïve Bayes, SVM, CNNs

---

### ★ Unsupervised Learning
A type of ML where the model finds patterns in data with no labels — no right answers are provided.

> **Examples from class:** K-Means Clustering, PCA

---

### ★ Inference
Using a trained model to make predictions on new data. This is what happens after training is complete.

> **Analogy:** The exam after all the studying is done.

---

## Data and Features

---

### Dataset
A structured collection of data used to train and evaluate ML models. Usually organized as rows (examples) and columns (features).

---

### Tabular Data
Data organized in rows and columns, like a spreadsheet. Most classic ML algorithms (KNN, Decision Trees, SVM) expect tabular data.

---

### Feature Engineering
The process of creating new features from raw data to help the model learn better.

> **Example:** From a "date" column, you might engineer "day of week" and "is weekend" as separate features.

---

### Feature Scaling / Normalization
Adjusting features so they are on the same numerical scale. Important for algorithms like KNN and SVM that use distance.

> **Analogy:** Comparing heights in both inches and centimeters in the same model would confuse it. Scaling converts everything to the same unit.

---

### Missing Values
Data points that are absent from the dataset. Must be handled before training (by filling in a default, the mean, or removing the row).

---

### Categorical Variable
A feature that holds labels or categories, not numbers (e.g., "red," "blue," "green" or "cat," "dog").

---

### One-Hot Encoding
Converting a categorical variable into a set of binary (0/1) columns so ML models can use it.

> **Example:** "Color" with values Red/Blue/Green becomes three columns: `is_red`, `is_blue`, `is_green`.

---

### Train/Test Split
Dividing the dataset into a training set and a test set before training, so you can evaluate the model on data it has never seen.

> **Typical split:** 80% training, 20% testing.

---

### Class Imbalance
When one label is much more common than another in the dataset.

> **Example:** In a fraud detection dataset, 99% of transactions are normal and 1% are fraud. A model that always says "not fraud" would be 99% accurate but completely useless.

---

### Toy Dataset
A small, clean, well-known dataset used for learning and experimentation.

> **Examples from class:** Iris (flower species), Boston Housing, Titanic, MNIST digits.

---

## Model Training and Evaluation

---

### Training
The process of fitting a model to data — the algorithm adjusts its parameters to minimize the cost function.

---

### Prediction
The output the model produces when given new input features.

---

### Regression
Predicting a **continuous numerical value** (e.g., price, temperature, score).

---

### Classification
Predicting a **category or class** (e.g., spam/not spam, cat/dog/bird, 0–9 digit).

---

### Baseline Model
The simplest possible model used as a reference point. If your model cannot beat the baseline, it is not learning anything useful.

> **Example:** For classification, the baseline might be "always predict the most common class."

---

### Learning Curve
A plot showing how model performance changes as more training data is added. Helps diagnose overfitting and underfitting.

---

### ROC Curve / AUC
A graph showing the tradeoff between true positive rate and false positive rate at different thresholds. AUC (Area Under the Curve) summarizes it as a single number — closer to 1.0 is better.

---

### Gradient Descent
An optimization algorithm used to minimize the cost function by iteratively adjusting parameters in the direction of steepest descent.

> **Analogy:** Finding the lowest point in a hilly landscape by always taking a step downhill.

---

### Learning Rate
A hyperparameter that controls how big each step is during gradient descent. Too large = overshooting; too small = very slow learning.

---

## Regression

---

### Linear Regression
An algorithm that fits a straight line to data to predict a continuous output.

> **Formula:** y = mx + b (slope-intercept form)
> **Used in class:** Week 8 — predicting house prices, exam scores.

---

### Multiple Linear Regression
Linear Regression with more than one input feature.

> **Example:** Predicting house price using square footage, number of bedrooms, AND neighborhood — not just one feature.

---

### Logistic Regression
Despite the name, this is a **classification** algorithm. It predicts the probability that an input belongs to a class (output between 0 and 1).

> **Analogy:** Instead of "the temperature is 72 degrees," it says "there is a 90% chance it will rain."
> **Used in class:** Week 9 — email spam detection.

---

### Coefficient
The weight assigned to each feature in a linear model. A higher coefficient means that feature has more influence on the prediction.

---

### Intercept
The value of the prediction when all features are zero. The starting point of the line.

---

### R² Score (R-squared)
A metric for regression models measuring how well the model explains the variation in the data. Ranges from 0 to 1; closer to 1 is better.

---

### Mean Squared Error (MSE)
A common cost function for regression: the average of the squared differences between predicted and actual values.

---

## Classification

---

### K-Nearest Neighbors (KNN)
A classification algorithm that predicts a label by finding the K most similar examples in the training data and taking a majority vote.

> **Analogy:** When you move to a new city, you ask your nearest neighbors what the neighborhood is like. KNN does the same — it asks the nearest data points.
> **Used in class:** Weeks 10–11.

---

### Euclidean Distance
The straight-line distance between two points. Used by KNN to measure "similarity."

---

### k (in KNN)
The hyperparameter controlling how many neighbors to consult. Small k = more complex boundary (risk of overfitting); large k = smoother boundary (risk of underfitting).

---

### Naïve Bayes
A fast classification algorithm based on Bayes' Theorem. It assumes all features are independent of each other (the "naïve" assumption).

> **Great for:** Text classification, spam detection.
> **Used in class:** Week 19.

---

### Bayes' Theorem
A formula for updating the probability of a hypothesis given new evidence.

> **Plain English:** "How likely is this email to be spam, given that it contains the word 'FREE'?"

---

### Prior Probability
The probability of something being true before seeing any evidence.

---

### Posterior Probability
The updated probability after seeing the evidence.

---

### Support Vector Machine (SVM)
A classification algorithm that finds the best decision boundary (hyperplane) that maximally separates two classes.

> **Analogy:** Drawing a line between two groups of points on paper, positioned so it is as far from both groups as possible.
> **Used in class:** Week 20.

---

### Support Vectors
The data points closest to the decision boundary. These are the only points that determine where the boundary goes.

---

### Margin
The distance between the decision boundary and the nearest support vectors. SVM maximizes this margin.

---

### Kernel
A function that transforms data into a higher-dimensional space so it becomes linearly separable. Allows SVM to work on curved or complex boundaries.

> **Common kernels:** Linear, RBF (Radial Basis Function), Polynomial.

---

### Decision Boundary
The line (or curve, or surface) that separates different classes in a classification model.

---

## Clustering

---

### Clustering
An unsupervised ML task of grouping similar data points together without predefined labels.

---

### K-Means Clustering
An algorithm that partitions data into K clusters by iteratively assigning points to the nearest centroid and updating centroids.

> **Analogy:** Sorting a pile of mixed candy into K bowls by color — you keep adjusting which bowl each piece goes in until the groupings stabilize.
> **Used in class:** Weeks 12–13.

---

### Centroid
The center point of a cluster. In K-Means, each cluster is defined by its centroid.

---

### K (in K-Means)
The number of clusters to find. You must choose this before running the algorithm.

---

### Inertia
The sum of squared distances from each point to its assigned centroid. Lower inertia = tighter, more compact clusters.

---

### Elbow Method
A technique for choosing the best K by plotting inertia vs. K and looking for the "elbow" — the point where adding more clusters stops helping much.

---

## Decision Trees and Ensemble Methods

---

### Decision Tree
A model that makes predictions by asking a series of yes/no questions about the features, forming a tree structure.

> **Analogy:** A flowchart for making decisions — like a choose-your-own-adventure book where each question leads to the next.
> **Used in class:** Weeks 14–15.

---

### Root Node
The first question at the top of the decision tree — the most important split.

---

### Leaf Node
A terminal node at the bottom of the tree that gives a final prediction (no more questions).

---

### Splitting
Dividing a node into two branches based on a feature and threshold that best separates the data.

---

### Gini Impurity
A measure of how mixed the classes are at a node. A Gini of 0 means perfectly pure (all one class); higher = more mixed. Decision Trees try to minimize this.

---

### Information Gain / Entropy
Another way to measure the quality of a split. Entropy measures disorder; a good split reduces entropy (brings order to the data).

---

### Tree Depth
How many levels of questions the tree can ask. Deeper trees can overfit; shallower trees may underfit.

---

### Pruning
Trimming branches from a Decision Tree to reduce overfitting by simplifying the model.

---

### Random Forest
An ensemble of many Decision Trees, each trained on a random subset of data and features. The final prediction is a vote across all trees.

> **Analogy:** Instead of asking one expert, you ask 100 experts and go with the majority opinion.
> **Used in class:** Weeks 16–17.

---

### Ensemble Method
A technique that combines multiple models to produce a better prediction than any single model alone.

---

### Bagging (Bootstrap Aggregating)
Training multiple models on different random samples of the training data (with replacement), then averaging their predictions. Random Forest uses bagging.

---

### Boosting
An ensemble technique where models are trained sequentially, with each new model focusing on the mistakes of the previous one.

> **Examples:** Gradient Boosting, XGBoost, AdaBoost.

---

### Feature Importance
A score showing how much each feature contributed to the model's predictions in a tree-based model. Great for understanding what the model learned.

---

## Principal Component Analysis (PCA)

---

### Dimensionality Reduction
Reducing the number of features in a dataset while keeping as much useful information as possible.

> **Analogy:** Summarizing a 500-page book into 10 key bullet points — you lose some detail but keep the essence.

---

### Principal Component Analysis (PCA)
An unsupervised technique that transforms features into a new set of uncorrelated variables (principal components) ordered by how much variance they explain.

> **Used in class:** Week 22.

---

### Principal Component
A new feature created by PCA that is a linear combination of the original features. The first principal component captures the most variance.

---

### Explained Variance
The proportion of the total information in the data captured by each principal component.

---

### Dimensionality
The number of features (columns) in a dataset.

---

### Curse of Dimensionality
The problem where having too many features makes it harder to find patterns, and distances between points become meaningless.

---

## Neural Networks and Deep Learning

---

### Neural Network
A model loosely inspired by the human brain, made up of layers of interconnected nodes (neurons) that learn complex patterns.

> **Analogy:** A relay race — each runner (layer) passes information to the next, transforming it along the way.

---

### Neuron / Node
The basic unit of a neural network. It receives inputs, applies a weight and activation function, and passes output to the next layer.

---

### Layer
A group of neurons in a neural network.

- **Input Layer:** Takes in the raw features.
- **Hidden Layer(s):** Transform and extract patterns.
- **Output Layer:** Produces the final prediction.

---

### Weight
A learnable parameter in a neural network that controls the strength of the connection between two neurons.

---

### Activation Function
A function applied to a neuron's output to introduce non-linearity, allowing neural networks to learn complex patterns.

> **Common ones:** ReLU (`max(0, x)`), Sigmoid (outputs 0–1), Softmax (outputs probabilities across classes).

---

### ReLU (Rectified Linear Unit)
The most commonly used activation function: outputs `x` if x > 0, otherwise outputs 0.

---

### Backpropagation
The algorithm neural networks use to learn — it calculates how much each weight contributed to the error and adjusts them accordingly.

> **Analogy:** A coach reviewing game tape to figure out which player made mistakes, then giving each player specific feedback.

---

### Epoch
One complete pass through the entire training dataset during neural network training.

---

### Batch Size
The number of training examples used in one forward/backward pass before updating weights.

---

### Dropout
A regularization technique where random neurons are temporarily "turned off" during training to prevent overfitting.

---

### Convolutional Neural Network (CNN)
A neural network architecture designed for image data. Uses filters to detect local patterns like edges, shapes, and textures.

> **Used in class:** Week 24.

---

### Convolution
A mathematical operation where a filter slides across an image, detecting a specific pattern (like a horizontal edge or a curve).

---

### Filter / Kernel (CNN)
A small matrix of learned weights that slides over an image to detect a specific feature. Early filters detect edges; later filters detect complex shapes.

---

### Feature Map
The output of applying a filter to an image — highlights where a particular pattern was detected.

---

### Pooling Layer
A layer that reduces the size of feature maps by summarizing regions (e.g., Max Pooling takes the largest value in each region).

---

### Flatten Layer
A layer that converts the 2D output of convolutional layers into a 1D vector before feeding it to a Dense layer.

---

### Dense / Fully Connected Layer
A layer where every neuron is connected to every neuron in the previous layer. Used at the end of CNNs for classification.

---

### Transfer Learning
Using a neural network pre-trained on a large dataset (like ImageNet) as a starting point, then fine-tuning it on your smaller dataset.

> **Analogy:** Hiring a chef who already knows how to cook — you just teach them your restaurant's specific recipes. Much faster than training from scratch.

---

## Natural Language Processing (NLP)

---

### Natural Language Processing (NLP)
A branch of ML focused on enabling computers to understand, interpret, and generate human language.

---

### Corpus
A large collection of text used for training NLP models.

---

### Token
A unit of text — usually a word or subword — that an NLP model processes. Tokenization is the step of breaking text into tokens.

---

### Stop Words
Common words (like "the," "is," "and") that are often removed before NLP processing because they carry little meaning.

---

### TF-IDF (Term Frequency–Inverse Document Frequency)
A numerical representation of text that reflects how important a word is to a specific document relative to a collection of documents.

> **TF:** How often a word appears in this document.
> **IDF:** How rare the word is across all documents. Rare words get higher scores.

---

### Bag of Words
A simple text representation that counts how many times each word appears in a document, ignoring word order.

---

### Word Embedding
A way of representing words as dense vectors of numbers so that similar words have similar vectors.

> **Example:** The vectors for "king" and "queen" are closer together than "king" and "pizza."

---

### Sentiment Analysis
Classifying text by the emotion or opinion it expresses — typically Positive, Negative, or Neutral.

> **Used in class:** Week 25 — IMDB movie reviews.

---

### Tokenization
Splitting text into individual tokens (words or subwords) before feeding it to an NLP model.

---

### Vocabulary
The full set of unique tokens a model knows about.

---

## Transformers and Large Language Models

---

### Transformer
A neural network architecture that uses self-attention to process all parts of an input simultaneously (not one word at a time). The foundation of modern NLP.

> **Used in class:** Week 26.

---

### Attention Mechanism
The key innovation in Transformers — allows the model to focus on the most relevant parts of the input when making each prediction.

> **Analogy:** When reading "The trophy didn't fit in the suitcase because it was too big," attention helps the model connect "it" to "trophy," not "suitcase."

---

### Self-Attention
A type of attention where a model looks at all other words in the same sentence to understand the meaning of each word in context.

---

### BERT (Bidirectional Encoder Representations from Transformers)
A Transformer model pre-trained on massive text data that reads context from both left and right. Great for understanding tasks.

---

### GPT (Generative Pre-trained Transformer)
A Transformer model pre-trained to predict the next word. Great for generating text. The foundation of ChatGPT.

---

### Large Language Model (LLM)
A Transformer model trained on enormous amounts of text data, capable of understanding and generating human-like language.

> **Examples:** GPT-4, Claude, Gemini, LLaMA.

---

### Pre-training
Training a model on a massive general dataset before fine-tuning it for a specific task.

---

### Fine-tuning
Taking a pre-trained model and continuing to train it on a smaller, task-specific dataset.

> **Analogy:** Pre-training = going to medical school. Fine-tuning = doing a residency in cardiology.

---

### Hugging Face
A platform and library (`transformers`) that provides pre-trained models for NLP tasks, ready to use in Python.

---

### Pipeline (Hugging Face)
A high-level function in the Hugging Face library that wraps a pre-trained model for a specific task (sentiment analysis, summarization, Q&A, etc.) in just a few lines of code.

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
classifier("I love this class!")
```

---

### Prompt
The input text given to an LLM to guide its response.

---

### Token (in LLMs)
The unit of text an LLM processes — roughly ¾ of a word on average. LLMs have a maximum number of tokens they can handle at once (the **context window**).

---

### Context Window
The maximum amount of text (in tokens) that an LLM can consider at one time.

---

### Hallucination
When an LLM generates text that sounds confident and fluent but is factually incorrect. A major limitation of current LLMs.

---

## Tools and Platforms

---

### scikit-learn
The main Python library for classic ML algorithms. Provides consistent APIs for training, evaluating, and tuning models.

---

### pandas
A Python library for loading, cleaning, and manipulating tabular data using DataFrames.

---

### NumPy
A Python library for fast numerical computing. Most ML libraries use NumPy arrays under the hood.

---

### matplotlib / seaborn
Python libraries for creating charts and visualizations — bar charts, scatter plots, confusion matrices, etc.

---

### TensorFlow / Keras
Google's deep learning framework. Keras is the high-level API that makes building neural networks straightforward.

---

### Google Colab
A free, cloud-based Jupyter notebook environment with access to free GPUs — our primary coding environment.

---

### Kaggle
A platform for ML competitions, free datasets, and free hosted notebooks (Kaggle Kernels). Great for practice and competitions.

---

### Hugging Face
A platform for sharing and using pre-trained Transformer models. Also hosts Spaces for deploying ML apps.

---

### GitHub
A platform for storing, sharing, and version-controlling code. Our course repository lives here.

---

### Streamlit
A Python library that turns a script into an interactive web app with minimal code — the easiest way to share your ML project.

---

### GPU (Graphics Processing Unit)
Hardware originally designed for graphics that is also extremely fast at the matrix math used in deep learning. Training CNNs and Transformers without a GPU would take hours or days.

---

### API (Application Programming Interface)
A way for one program to communicate with another. Hugging Face and many ML services expose APIs so you can use their models in your own code.

---

## Quick Reference — A to Z Index

| Term | Section |
|---|---|
| Accuracy | Cross-Cutting Concepts |
| Activation Function | Neural Networks |
| API | Tools and Platforms |
| Attention Mechanism | Transformers |
| Backpropagation | Neural Networks |
| Bagging | Decision Trees |
| Bag of Words | NLP |
| Baseline Model | Model Training |
| Bayes' Theorem | Classification |
| BERT | Transformers |
| Bias (model) | Cross-Cutting Concepts |
| Bias-Variance Tradeoff | Cross-Cutting Concepts |
| Boosting | Decision Trees |
| Categorical Variable | Data and Features |
| Centroid | Clustering |
| Class Imbalance | Data and Features |
| Classification | Model Training |
| CNN | Neural Networks |
| Coefficient | Regression |
| Confusion Matrix | Cross-Cutting Concepts |
| Context Window | Transformers |
| Convolution | Neural Networks |
| Corpus | NLP |
| Cost Function | Cross-Cutting Concepts |
| Cross-Validation | Cross-Cutting Concepts |
| Curse of Dimensionality | PCA |
| Dataset | Data and Features |
| Decision Boundary | Classification |
| Decision Tree | Decision Trees |
| Dense Layer | Neural Networks |
| Dimensionality Reduction | PCA |
| Dropout | Neural Networks |
| Elbow Method | Clustering |
| Embedding | NLP |
| Ensemble Method | Decision Trees |
| Entropy | Decision Trees |
| Epoch | Neural Networks |
| Euclidean Distance | Classification |
| Explained Variance | PCA |
| F1 Score | Cross-Cutting Concepts |
| Feature | Cross-Cutting Concepts |
| Feature Engineering | Data and Features |
| Feature Importance | Decision Trees |
| Feature Map | Neural Networks |
| Feature Scaling | Data and Features |
| Filter (CNN) | Neural Networks |
| Fine-tuning | Transformers |
| Flatten Layer | Neural Networks |
| Generalization | Cross-Cutting Concepts |
| Gini Impurity | Decision Trees |
| Google Colab | Tools |
| GPT | Transformers |
| GPU | Tools |
| Gradient Descent | Model Training |
| Hallucination | Transformers |
| Hugging Face | Tools / Transformers |
| Hyperparameter | Cross-Cutting Concepts |
| Information Gain | Decision Trees |
| Inference | Cross-Cutting Concepts |
| Inertia | Clustering |
| Intercept | Regression |
| K (K-Means) | Clustering |
| k (KNN) | Classification |
| Kaggle | Tools |
| Kernel (SVM) | Classification |
| K-Means Clustering | Clustering |
| K-Nearest Neighbors | Classification |
| Label / Target | Cross-Cutting Concepts |
| Layer | Neural Networks |
| Leaf Node | Decision Trees |
| Learning Curve | Model Training |
| Learning Rate | Model Training |
| Linear Regression | Regression |
| LLM | Transformers |
| Logistic Regression | Regression |
| Loss Function | Cross-Cutting Concepts |
| Machine Learning | Cross-Cutting Concepts |
| Margin | Classification |
| Missing Values | Data and Features |
| Model | Cross-Cutting Concepts |
| MSE | Regression |
| Naïve Bayes | Classification |
| Neural Network | Neural Networks |
| NLP | NLP |
| Normalization | Data and Features |
| NumPy | Tools |
| One-Hot Encoding | Data and Features |
| Overfitting | Cross-Cutting Concepts |
| pandas | Tools |
| Parameter | Cross-Cutting Concepts |
| PCA | PCA |
| Pipeline (Hugging Face) | Transformers |
| Pooling Layer | Neural Networks |
| Posterior Probability | Classification |
| Pre-training | Transformers |
| Precision | Cross-Cutting Concepts |
| Principal Component | PCA |
| Prior Probability | Classification |
| Prompt | Transformers |
| Pruning | Decision Trees |
| R² Score | Regression |
| Random Forest | Decision Trees |
| Recall | Cross-Cutting Concepts |
| Regression | Model Training |
| ReLU | Neural Networks |
| ROC / AUC | Model Training |
| Root Node | Decision Trees |
| scikit-learn | Tools |
| Self-Attention | Transformers |
| Sentiment Analysis | NLP |
| Split (tree) | Decision Trees |
| Stop Words | NLP |
| Streamlit | Tools |
| Supervised Learning | Cross-Cutting Concepts |
| Support Vector Machine | Classification |
| Support Vectors | Classification |
| TF-IDF | NLP |
| Token | NLP / Transformers |
| Tokenization | NLP |
| Toy Dataset | Data and Features |
| Train/Test Split | Data and Features |
| Training | Model Training |
| Training Data | Cross-Cutting Concepts |
| Transfer Learning | Neural Networks |
| Transformer | Transformers |
| Tree Depth | Decision Trees |
| Underfitting | Cross-Cutting Concepts |
| Unsupervised Learning | Cross-Cutting Concepts |
| Validation Data | Cross-Cutting Concepts |
| Variance (model) | Cross-Cutting Concepts |
| Vocabulary | NLP |
| Weight | Neural Networks |

---

*Course Repository: https://github.com/sjasthi/Python-ML-Machine-Learning*
*Instructor: Siva | learnandhelp.com*
