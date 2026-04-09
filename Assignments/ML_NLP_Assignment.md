# Week 25 Assignment — Introduction to NLP (25 Points)

**Course:** Python ML — Learn and Help ([www.learnandhelp.com](http://www.learnandhelp.com))

---

## Task 1: Tokenization and Stopword Exploration (6 Points)

Using the Colab notebook, complete the following:

1. **(2 pts)** Pick any paragraph from a book, song description, or news article (at least 3 sentences). Paste it into the tokenization cell and run it. Copy-paste the list of tokens into your answer file.

2. **(2 pts)** Run the same paragraph through stopword removal. List at least 5 stopwords that were removed and explain in 1–2 sentences why removing them doesn't change the meaning.

3. **(2 pts)** Answer this question: If you were building a spam detector for emails, which of these would be more useful — keeping stopwords or removing them? Explain your reasoning in 2–3 sentences.

---

## Task 2: Build a Sentiment Classifier with scikit-learn (7 Points)

Using the scikit-learn section of the Colab notebook:

1. **(2 pts)** Run the TF-IDF + Logistic Regression sentiment classifier on the IMDB dataset. Take a screenshot of the accuracy score and the classification report.

2. **(2 pts)** Write **3 of your own movie reviews** (one positive, one negative, one tricky/mixed) and test them with the model. Copy-paste the reviews and the model's predictions into your answer file.

3. **(3 pts)** Answer these questions:
   - Did the model correctly predict the sentiment of your 3 reviews? If not, which one did it get wrong and why do you think it struggled?
   - Look at the TF-IDF feature names — what are the top 5 most important words for positive reviews and the top 5 for negative reviews?
   - In 2–3 sentences, explain how TF-IDF helps the model distinguish between positive and negative reviews.

---

## Task 3: Explore the Keras Text Classifier (6 Points)

Using the Keras section of the Colab notebook:

1. **(2 pts)** Run the Keras IMDB sentiment classifier. Take a screenshot showing the training progress (epoch-by-epoch accuracy) and the final test accuracy.

2. **(2 pts)** Try changing **one** of the following and record what happens:

   | What to Change | How to Change It | Your Result |
   |----------------|-----------------|-------------|
   | Embedding size | Change `32` to `64` or `128` | Accuracy: ____% |
   | Hidden layer neurons | Change `64` to `32` or `128` | Accuracy: ____% |
   | Number of epochs | Change `5` to `2` or `10` | Accuracy: ____% |
   | Max review length | Change `max_len=200` to `100` or `300` | Accuracy: ____% |

3. **(2 pts)** In 2–3 sentences, explain: What is a word embedding? How is it different from TF-IDF? (Hint: Think about what information each one captures about a word.)

---

## Task 4: Reflection Questions (6 Points)

Answer each question in 2–4 sentences:

1. **(2 pts)** Name 3 real-world applications of NLP that you personally use or encounter in your daily life. For each one, briefly explain what NLP task it performs (e.g., classification, translation, text generation).

2. **(2 pts)** Compare the scikit-learn approach and the Keras approach from this lesson. Which one was easier to understand? Which one do you think would work better on a much harder task (like translating English to Spanish)? Why?

3. **(2 pts)** Think back to the classic ML algorithms we learned in Weeks 1–22 (KNN, Decision Trees, SVM, etc.). How is NLP *different* from those earlier projects? What's the extra step that NLP requires before you can use any ML algorithm?

---

## Submission

- Add your answers as a markdown file (`week25_nlp_answers.md`) to your GitLab repository
- Include all screenshots in your repository
- Submit the link on Google Classroom

---

| Task | Description | Points |
|------|-------------|--------|
| Task 1 | Tokenization & Stopword Exploration | 6 |
| Task 2 | Build a Sentiment Classifier with scikit-learn | 7 |
| Task 3 | Explore the Keras Text Classifier | 6 |
| Task 4 | Reflection Questions | 6 |
| **Total** | | **25** |
