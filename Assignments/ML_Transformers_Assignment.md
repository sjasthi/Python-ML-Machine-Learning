# Week 26 Assignment — Transformers and Large Language Models (25 Points)

**Course:** Python ML — Learn and Help ([www.learnandhelp.com](http://www.learnandhelp.com))

---

## Task 1: Attention and Architecture Concepts (7 Points)

Answer each question in 2–4 sentences:

1. **(2 pts)** In the sentence "The dog chased the ball because it was bouncy," what does "it" refer to? Explain how the attention mechanism helps a Transformer figure this out. Which word would receive the highest attention score from "it"?

2. **(2 pts)** Explain the difference between how an RNN (older model) and a Transformer process a sentence. Why is the Transformer's approach faster and better at understanding long sentences?

3. **(3 pts)** Name the three types of Transformer models (encoder-only, encoder-decoder, decoder-only). For each type, give one real-world product or model that uses it and explain what task it performs.

---

## Task 2: Explore the OpenAI Tokenizer (6 Points)

Go to [platform.openai.com/tokenizer](https://platform.openai.com/tokenizer) and try the following experiments:

1. **(2 pts)** Type in the sentence "I love machine learning!" and record how many tokens it produces. Then type in "Supercalifragilisticexpialidocious" and record how many tokens it produces. Take a screenshot of each. Why does the long word produce more tokens?

2. **(2 pts)** Type in your full name and record the tokens. Then type in a common English sentence of at least 10 words. Calculate the ratio of tokens to words for both. Is it always 1 token per word? Why or why not?

3. **(2 pts)** Type in "123 + 456 = 579" and look at how the numbers are tokenized. Now type "How many R's are in strawberry?" and look at how "strawberry" is tokenized. Based on what you see, explain in 2–3 sentences why LLMs sometimes struggle with math and spelling.

---

## Task 3: Use Pretrained Transformers (6 Points)

Using the Colab notebook:

1. **(2 pts)** Run the sentiment analysis pipeline on **5 of your own sentences** (mix of positive, negative, and tricky/mixed). Copy-paste the sentences and results into your answer file. Did the Transformer get any wrong?

2. **(2 pts)** Run the text generation pipeline with **2 different prompts of your choice**. Copy-paste the generated text. In 2–3 sentences, describe: Was the generated text coherent? Did anything surprise you?

3. **(2 pts)** Run either the summarization or question-answering pipeline with your own input text. Copy-paste the input and output. In 1–2 sentences, rate how well the model performed.

---

## Task 4: Reflection — Connecting the Dots (6 Points)

1. **(2 pts)** Think about the entire course journey from Week 1 to Week 26. Pick **one concept** from an earlier week (e.g., train/test split, overfitting, feature extraction, accuracy, classification) and explain how that same concept shows up in Transformers and LLMs.

2. **(2 pts)** ChatGPT and Claude sometimes "hallucinate" — they confidently say things that are wrong. Based on what you learned about how these models generate text (next-token prediction), explain in 3–4 sentences why hallucination happens.

3. **(2 pts)** If you could build an AI application using Transformers, what would it be? Describe the application in 3–4 sentences. What type of Transformer would you use (encoder-only, encoder-decoder, or decoder-only) and why?

---

## Submission

- Add your answers as a markdown file (`week26_transformers_answers.md`) to your GitLab repository
- Include all screenshots in your repository
- Submit the link on Google Classroom

---

| Task | Description | Points |
|------|-------------|--------|
| Task 1 | Attention and Architecture Concepts | 7 |
| Task 2 | Explore the OpenAI Tokenizer | 6 |
| Task 3 | Use Pretrained Transformers | 6 |
| Task 4 | Reflection — Connecting the Dots | 6 |
| **Total** | | **25** |
