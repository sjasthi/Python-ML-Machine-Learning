# 🚀 Kaggle & Hugging Face — Your ML Launchpad
### Python for Machine Learning | Metropolitan State University

> **Course Goal:** By the end of this course, your capstone project will be **live on Hugging Face** — accessible to the world, on your professional ML portfolio.

---

## 🗺️ The Big Picture

As a machine learning practitioner, you don't work in isolation. The ML community thrives on **shared data, shared models, and shared knowledge**. Two platforms make this possible more than any other:

| Platform | What it gives you | Your main use |
|---|---|---|
| 🏆 **Kaggle** | Datasets, competitions, notebooks, community | Learn, explore data, benchmark your skills |
| 🤗 **Hugging Face** | Pretrained models, datasets, deployment tools | Share your work, use state-of-the-art models |

Think of **Kaggle** as your *training ground* and **Hugging Face** as your *showcase stage*.

---

## 🏆 Part 1: Kaggle — Where ML Practitioners Learn and Compete

### What Is Kaggle?
Kaggle (owned by Google) is the world's largest data science community with over **15 million users**. It's where companies post real-world problems, and data scientists compete to solve them — often for prize money, but always for learning.

### Why You NEED to Know Kaggle

- **Free compute** — GPU/TPU notebooks in the cloud (no local setup needed)
- **10,000+ real-world datasets** — clean, documented, and ready to use
- **Pre-written notebooks** — see how experts approach any problem
- **Competitions** — benchmark yourself against the global ML community
- **Certifications and courses** — free, hands-on ML micro-courses

### 🔍 How to Explore Kaggle (Step-by-Step)

#### Step 1: Create Your Account
1. Go to [https://www.kaggle.com](https://www.kaggle.com)
2. Click **Register** → Sign up with Google or email
3. Complete your profile — add your university and interests

#### Step 2: Explore Datasets
1. Click **Datasets** in the top navigation
2. Search for any topic you're interested in (e.g., `"housing prices"`, `"diabetes"`, `"movies"`)
3. Click a dataset to see:
   - **Overview** tab — what the data means
   - **Data** tab — preview the actual CSV/files
   - **Code** tab — notebooks others wrote using this exact dataset
   - **Discussion** tab — tips, questions, insights from the community

```
💡 Pro Tip: Before building anything from scratch, always check the 
"Code" tab on a Kaggle dataset. You'll find hundreds of notebooks 
showing different approaches to the same problem!
```

#### Step 3: Run Your First Kaggle Notebook
1. Go to **Code** → **New Notebook**
2. You get a free Jupyter-like environment with:
   - 30 hours/week of free GPU
   - Python pre-installed with sklearn, pandas, tensorflow, pytorch
3. Click **+ Add Data** to attach any Kaggle dataset

#### Step 4: Browse Competitions
1. Click **Competitions** in the nav bar
2. Filter by **Getting Started** to find beginner-friendly challenges
3. The legendary **Titanic: Machine Learning from Disaster** is where every ML practitioner starts:
   - [kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)

#### Step 5: Take a Free Course
Kaggle Learn offers bite-sized, free ML courses:
- [kaggle.com/learn](https://www.kaggle.com/learn)
- Recommended path: **Intro to ML → Intermediate ML → Feature Engineering → Intro to Deep Learning**

### 📂 Downloading a Dataset to Use in Google Colab

```python
# Install the Kaggle API
!pip install kaggle

# Upload your kaggle.json API key (from kaggle.com → Account → API → Create New Token)
from google.colab import files
files.upload()  # Upload kaggle.json here

# Set up credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download a dataset (example: Titanic)
!kaggle competitions download -c titanic
!unzip titanic.zip

# Now load it
import pandas as pd
df = pd.read_csv('train.csv')
df.head()
```

### 🏅 Kaggle Progression Tiers
Your Kaggle profile has a public ranking system — a great motivation to keep improving:

```
Novice → Contributor → Expert → Master → Grandmaster
```

Reaching **Expert** tier in any category (Competitions, Datasets, Notebooks, or Discussions) is a strong signal to future employers.

---

## 🤗 Part 2: Hugging Face — The GitHub of Machine Learning

### What Is Hugging Face?
Hugging Face is the central hub for the modern AI/ML ecosystem. It hosts over **900,000 models**, **200,000 datasets**, and **300,000 interactive apps (Spaces)**. It's where researchers and companies share their work — and where **you will share yours**.

### Why Hugging Face Matters for Your Career

- **Industry standard** — companies like Google, Meta, Microsoft, and Amazon all publish models here
- **Free model hosting** — deploy your ML app for the world to use (no server needed)
- **Transformers library** — access state-of-the-art NLP, vision, and multimodal models in 3 lines of code
- **Your public portfolio** — your Hugging Face profile is a living, interactive ML resume

### 🔍 How to Explore Hugging Face (Step-by-Step)

#### Step 1: Create Your Account
1. Go to [https://huggingface.co](https://huggingface.co)
2. Click **Sign Up** — use your university email
3. Set a professional username — this will appear in your portfolio URL:
   `https://huggingface.co/YOUR_USERNAME`

#### Step 2: Explore Models
1. Click **Models** in the top navigation
2. Use the left panel to filter by:
   - **Task** (e.g., Text Classification, Image Classification, Object Detection)
   - **Library** (e.g., PyTorch, TensorFlow, scikit-learn)
   - **Language**
3. Click any model card to see:
   - Model description and how it works
   - **Inference API** — test the model live in your browser, no code!
   - Sample code to use it in Python
   - Training details and performance metrics

```
🔥 Try This Right Now: Go to huggingface.co/models, search for 
"sentiment-analysis", click the top result, and type any sentence 
into the live inference box on the right side. You just used an 
AI model without writing a single line of code!
```

#### Step 3: Explore Datasets
1. Click **Datasets** in the nav bar
2. Search for datasets relevant to your project
3. Each dataset has a **Dataset Viewer** — preview the data directly in browser
4. Click **Use in dataset library** to get the Python code to load it instantly

```python
# Load any Hugging Face dataset in one line
from datasets import load_dataset

dataset = load_dataset("imdb")  # 50,000 movie reviews
print(dataset['train'][0])
```

#### Step 4: Explore Spaces (Live ML Apps!)
1. Click **Spaces** in the nav bar
2. Browse thousands of **live, interactive ML demos** built by the community
3. Each Space is a running web app — try them directly in your browser
4. Click **Files and versions** on any Space to see the source code

```
💡 Your capstone project will be a Space like these — 
a live web app anyone can visit and interact with!
```

#### Step 5: Use a Pretrained Model in Colab

```python
# Install the transformers library
!pip install transformers

from transformers import pipeline

# Sentiment analysis in 3 lines!
classifier = pipeline("sentiment-analysis")
result = classifier("I absolutely loved this machine learning course!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]

# Try other tasks:
# "text-generation", "summarization", "translation_en_to_fr"
# "image-classification", "object-detection", "zero-shot-classification"
```

---

## 🎯 Part 3: Your Capstone — Going Live on Hugging Face Spaces

Your capstone project won't just be a notebook — it will be a **deployed, interactive web application** hosted on Hugging Face Spaces. Here's the roadmap:

### What is a Hugging Face Space?
A Space is a free, hosted web app. You push your code to it (like GitHub), and Hugging Face runs it. Your app gets a public URL you can share with anyone — including future employers.

Supported frameworks:
- **Gradio** ← We will use this (simplest, most powerful for ML)
- Streamlit
- Docker
- Static HTML

### 🛠️ Building Your First Gradio App (Practice This Now!)

```python
# Step 1: Install Gradio
!pip install gradio

# Step 2: Build a simple ML app
import gradio as gr
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Train a model
iris = load_iris()
clf = RandomForestClassifier()
clf.fit(iris.data, iris.target)
labels = iris.target_names

# Define the prediction function
def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = clf.predict(features)[0]
    probabilities = clf.predict_proba(features)[0]
    return {labels[i]: float(probabilities[i]) for i in range(len(labels))}

# Build the UI
app = gr.Interface(
    fn=predict_flower,
    inputs=[
        gr.Slider(4.0, 8.0, value=5.8, label="Sepal Length (cm)"),
        gr.Slider(2.0, 4.5, value=3.0, label="Sepal Width (cm)"),
        gr.Slider(1.0, 7.0, value=4.0, label="Petal Length (cm)"),
        gr.Slider(0.1, 2.5, value=1.2, label="Petal Width (cm)"),
    ],
    outputs=gr.Label(num_top_classes=3),
    title="🌸 Iris Flower Classifier",
    description="Adjust the sliders to classify an iris flower species using a Random Forest model.",
    examples=[[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5], [6.3, 3.3, 6.0, 2.5]]
)

app.launch()
```

### 🚀 Deploying to Hugging Face Spaces

#### Step 1: Create a New Space
1. Log in to [huggingface.co](https://huggingface.co)
2. Click your profile icon → **New Space**
3. Fill in:
   - **Space name** → e.g., `iris-classifier` or your capstone project name
   - **License** → MIT
   - **SDK** → Select **Gradio**
4. Click **Create Space**

#### Step 2: Upload Your Files
Your Space needs two files minimum:

**`app.py`** — your Gradio application code

**`requirements.txt`** — libraries your app needs
```
scikit-learn
gradio
numpy
pandas
```

Upload these via the **Files** tab in your Space, or use Git:

```bash
# Clone your space locally
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Add your files
cp app.py YOUR_SPACE_NAME/
cp requirements.txt YOUR_SPACE_NAME/

# Push to deploy
cd YOUR_SPACE_NAME
git add .
git commit -m "Initial deployment of capstone project"
git push
```

#### Step 3: Watch It Build
Hugging Face will automatically install your requirements and launch your app. In 2–3 minutes, your app is **live at**:
```
https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

#### Step 4: Share It!
- Add the link to your resume
- Share it in your LinkedIn profile under **Projects**
- Submit the URL for your capstone grade

---

## 📋 Capstone Project Requirements Summary

Your Hugging Face Space must include:

| Requirement | Details |
|---|---|
| **`app.py`** | Gradio-based interactive web application |
| **`requirements.txt`** | All dependencies listed |
| **`README.md`** | Project description, dataset used, model used, how to interact |
| **Trained Model** | Your own trained sklearn / PyTorch / other model |
| **Input/Output UI** | User can input data and see real predictions |
| **Model Card** | Description on the Space's main page (edit via HF interface) |

---

## 🗓️ Suggested Exploration Schedule

| Week | Kaggle Task | Hugging Face Task |
|---|---|---|
| This week | Create account, browse 3 datasets | Create account, test 3 models via Inference API |
| Next week | Run Titanic notebook start to finish | Explore 5 Spaces, look at their source code |
| Week 3 | Download a dataset for your capstone | Build and run a local Gradio app |
| Week 4+ | Use Kaggle data in your capstone | Deploy your capstone to a Space |

---

## 🔗 Quick Reference Links

| Resource | URL |
|---|---|
| Kaggle | [kaggle.com](https://www.kaggle.com) |
| Kaggle Learn (free courses) | [kaggle.com/learn](https://www.kaggle.com/learn) |
| Kaggle Titanic Competition | [kaggle.com/c/titanic](https://www.kaggle.com/c/titanic) |
| Kaggle API Docs | [kaggle.com/docs/api](https://www.kaggle.com/docs/api) |
| Hugging Face | [huggingface.co](https://huggingface.co) |
| HF Model Hub | [huggingface.co/models](https://huggingface.co/models) |
| HF Spaces | [huggingface.co/spaces](https://huggingface.co/spaces) |
| HF Datasets | [huggingface.co/datasets](https://huggingface.co/datasets) |
| Gradio Docs | [gradio.app/docs](https://www.gradio.app/docs) |
| Transformers Docs | [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers) |

---

## ✅ Action Items for This Week

- [ ] Create your **Kaggle account** at kaggle.com
- [ ] Complete your **Kaggle profile** (add university, bio)
- [ ] Browse at least **3 datasets** relevant to your capstone idea
- [ ] Open and **run a notebook** on Kaggle (try the Titanic one)
- [ ] Create your **Hugging Face account** at huggingface.co
- [ ] **Test 3 models** using the live Inference API (no code needed)
- [ ] Browse **5 Spaces** and look at the source code of at least one
- [ ] Run the **Gradio Iris app** code above in Google Colab

---

*Prepared for Python for Machine Learning | Metropolitan State University*  
*[learnandhelp.com](https://www.learnandhelp.com) | [jasthi.com](https://www.jasthi.com)*
