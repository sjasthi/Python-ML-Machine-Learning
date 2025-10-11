# Week 6 Outline: ML Workflow and Data Splits

## Course Information
**Topics:** 
1. Machine Learning Workflow (aka Process)
2. Testing, Validation, and Training Data Sets

---

## Learning Objectives
By the end of this week, students will be able to:
- Understand and apply the complete machine learning workflow
- Differentiate between training, validation, and test datasets
- Explain why we split data into different sets
- Apply best practices for dataset splitting
- Understand the role of each dataset in preventing overfitting

---

## Topics Covered

### 1. Machine Learning Workflow (aka Process)
**Reference Material:** [ML Workflow Process](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_Workflow_aka_Process.md)

The step-by-step process of building machine learning models:
- Problem definition and understanding
- Data collection and exploration
- Data preparation and cleaning
- Feature engineering
- Model selection and training
- Model evaluation
- Model deployment and monitoring
- Iteration and improvement

### 2. Testing, Validation, and Training Data Sets
**Reference Material:** [ML Training, Validation, Test Datasets](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_Training_Validation_Test_Datasets.pptx)

Understanding how we split our data:

#### Training Dataset
- Used to train the model
- The model learns patterns from this data
- Largest portion of the data (typically 60-70%)

#### Validation Dataset
- Used to tune model parameters (hyperparameters)
- Helps select the best model
- Prevents overfitting during development
- Typically 15-20% of the data

#### Test Dataset
- Used ONLY at the end to evaluate final model performance
- Provides unbiased evaluation
- Should never be used during training or tuning
- Typically 15-20% of the data

#### Why Split Data?
- Avoid overfitting (memorizing instead of learning)
- Get honest assessment of model performance
- Ensure model works on new, unseen data
- Simulate real-world performance

---

## Review Materials

### Interactive Exploration
**Interactive HTML Tool:** [Explore Training, Validation, and Test Datasets](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Play/ML_Training_Validation_Test_Datasets.html)

**How to use:**
- Download the HTML file from the link above
- Open it in your web browser
- Interact with the visualization to understand dataset splits
- Experiment with different split ratios
- Observe how each dataset is used in the ML process

### Core Concepts Review
**Reference Material:** [ML Core Concepts Part 1](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_core_concepts_part_1.pdf)

Review key concepts from Week 5:
- Features and predictions
- Bias vs. variance
- Overfitting vs. underfitting
- Cost functions

---

## Assignment Due This Week

### Assignment 4: Exploring No Coding ML - Teachable Machine
- **Due Date:** October 17, 2025 (Thursday)
- **Points:** 25 points
- **Description:** Hands-on exploration of machine learning using Google's Teachable Machine
- **Assignment Link:** [Python ML Assignment - Teachable Machine](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Assignments/Python_ML_Assignment_Exploring_NoCoding_Teachable_ML.md)
- **What to Complete:** 
  - Create an image classification model
  - Train your model with multiple classes
  - Test and evaluate your model's performance
  - Reflect on the ML concepts learned

---

## Key Takeaways

✓ The ML workflow is a systematic process from problem definition to deployment  
✓ Data should be split into training, validation, and test sets  
✓ Training data teaches the model, validation data tunes it, test data evaluates it  
✓ Never use test data during model development - save it for final evaluation  
✓ Proper data splitting prevents overfitting and ensures model generalization  
✓ The ML workflow connects all concepts: features, predictions, bias/variance, and cost functions

---

## Class Activities

### In-Class Exploration
- Review the interactive HTML tool together
- Discuss real-world examples of the ML workflow
- Analyze how training/validation/test splits prevent overfitting
- Share and test Teachable Machine models

### Discussion Questions
- Why can't we just use all our data for training?
- What happens if we use the test set multiple times?
- How does the validation set help us avoid overfitting?
- What's the connection between data splits and bias/variance?

---

## Next Steps
- Review both presentation materials before class
- Explore the interactive HTML tool
- Complete Assignment 4 (Teachable Machine) by October 17
- Think about how the ML workflow applies to your Teachable Machine project
- Reflect on how you split your "training data" (examples) in Teachable Machine

---

## Additional Resources
- [ML Workflow Process](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_Workflow_aka_Process.md)
- [Training, Validation, Test Datasets Presentation](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_Training_Validation_Test_Datasets.pptx)
- [Interactive Dataset Explorer](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Play/ML_Training_Validation_Test_Datasets.html)
- [ML Core Concepts Review](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Presentations/ML_core_concepts_part_1.pdf)
