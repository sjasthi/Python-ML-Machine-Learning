# Telugu Speech AI — Capstone Assignment

### Learn and Help | Python Machine Learning Course 

**Points:** 100 points total
**Submission:** Google Classroom

---

## Overview

This capstone project combines **everything you've learned** — data pipelines, model comparison,
evaluation metrics, and AI APIs — applied to a real-world problem:
**making AI understand and speak Telugu**.

Reference the [Project Introduction](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Projects/ML_Telugu_Speech_AI_Introduction.md) and the
[Colab Notebook](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Projects/ML_Telugu_ASR_and_TTS.ipynb) as you work through the tasks.

---

## Project Goals

This project is organized around six core goals. Each goal maps to one or more tasks below.

**Goal 1 — Automatic Speech Recognition (ASR)**
Convert spoken audio into written text using four different models — two originating from US-based providers and two from Indian providers. Evaluate how accurately each model transcribes speech by comparing its output against a known correct transcript, with a particular focus on which model handles Telugu the best.
*(Covered in Task 2)*

**Goal 2 — Text-to-Speech (TTS)**
Take written text and convert it into spoken audio using four models (again, two US-based and two India-based, though not necessarily the same ones from Goal 1). Each model can use whatever default voice it ships with. The key question here is: how do you objectively measure whether the audio sounds good and faithfully represents the original text?
*(Covered in Task 3)*

**Goal 3 — Voice Cloning**
Take Goal 2 a step further by replacing the default model voice with a custom, real-world voice — for example, a well-known public figure or a specific person — and generate speech that mimics that person's vocal style.
*(Covered in Task 3, Option A)*

**Goal 4 — Free vs. Freemium vs. Paid Model Comparison**
Across all three tasks above (ASR, TTS, and voice cloning), compare how free, freemium, and paid models stack up against each other. What do you gain — or give up — by paying more?
*(Covered in Task 5)*

**Goal 5 — A Unified UI for ASR and TTS**
Build a simple two-tab web interface: one tab where users upload an audio file to trigger all the ASR models simultaneously, and another where users paste text to trigger all the TTS models. Both tabs display each model's output side by side, and users can rate the results.
*(Covered in Task 6)*

**Goal 6 — Practical & Cost Considerations**
Address the infrastructure and economics questions: Can these workloads realistically run on a personal laptop or desktop, or do they require a cloud GPU environment like Google Colab? What are the pricing structures and usage limits for each model explored in the project?
*(Covered in Task 4 and Task 5)*

---

## Task 1 — Data Collection and Setup (10 points)

**Goal:** Collect or record Telugu audio and prepare it for experiments.

### What to do:

1. **Collect at least 5 Telugu audio clips** (each 5–15 seconds long). You can:
   * Record yourself speaking simple Telugu sentences
   * Download samples from [OpenSLR Telugu Dataset](https://openslr.org/66/)
   * Use sentences suggested in the notebook
2. **Write the ground truth transcription** (the exact Telugu text) for each clip
3. **Preprocess all clips** to 16kHz, mono WAV format using the notebook code
4. **Create a data table** listing: filename, duration (seconds), sentence, source (recorded/downloaded)

### Deliverable:

* Code cell in notebook showing preprocessing
* A table with your 5+ audio clips and their metadata
* Screenshot or output showing the audio waveform of at least one clip

---

## Task 2 — Speech-to-Text Model Comparison (20 points)

**Goal:** Run your Telugu audio through all 4 models and compare their accuracy.
*(Maps to Project Goal 1)*

### What to do:

1. **Run all 4 models** below on your audio clips. You must use at least **2 US-based** and **2 India-based** models. The defaults are listed below, but you may substitute any model in a category if you find a more current or accessible alternative — just document your substitution and explain why you chose it.

   **Default models (use these unless substituting):**
   * OpenAI Whisper (USA) — `whisper` package
   * Meta MMS (USA) — `facebook/mms-300m` on Hugging Face
   * AI4Bharat IndicWhisper (India) — `ai4bharat/indicwav2vec` on Hugging Face
   * Vakyansh (India) — `Vakyansh/wav2vec2-telugu-tem-100` on Hugging Face

   **Possible substitutions (examples):**
   * USA: AssemblyAI API, Google Cloud Speech-to-Text, Amazon Transcribe, Azure Speech
   * India: Sarvam AI (`sarvam-2b`), Bhashini ASR API, any other AI4Bharat model variant

   > 💡 If you substitute a model, add a one-sentence note in your notebook explaining what it is and why you chose it.

2. **Record the transcription output** from each model for each audio clip
3. **Compute WER and CER** for each model using the `jiwer` library
4. **Create a comparison table** with columns: Model, Origin (USA/India), Tier (Free/Freemium/Paid), WER%, CER%, Speed (RTF)

### Deliverable:

* All 4 model inference code cells in the notebook (with visible outputs)
* A summary comparison table covering all 4 models
* 2–3 sentences answering: *Which model performed best on Telugu? Were India-origin models better? Did any substitution surprise you?*

### Grading:

* 6 pts — All 4 models run with output (2 USA + 2 India)
* 6 pts — WER/CER correctly computed for all 4 models
* 5 pts — Clear comparison table with all required columns
* 3 pts — Written analysis with insight about Telugu performance

---

## Task 3 — Text-to-Speech and Voice Cloning (20 points)

**Goal:** Synthesize Telugu speech — ideally using your own cloned voice.
*(Maps to Project Goals 2 and 3)*

### What to do:

**Option A (Recommended): Voice Cloning with Coqui XTTS-v2**

1. Record yourself speaking one Telugu sentence (at least 10 seconds)
2. Load `tts_models/multilingual/multi-dataset/xtts_v2` using the `TTS` library
3. Pass a new Telugu sentence as text and your recording as `speaker_wav`
4. Save the output audio file and listen to it — does it sound like you?

**Option B: Standard TTS Comparison**

1. Use AI4Bharat IndicTTS API to generate Telugu audio from text
2. Use Google TTS (`gtts`) to generate the same text
3. Compare quality by ear and report a simple MOS score (rate 1–5 yourself)

### For Either Option:

* Test with at least 3 different Telugu sentences
* Listen to the output and note what sounds good / wrong

### Deliverable:

* Code cells with TTS/cloning code and audio output saved as `.wav` files
* Your MOS ratings (1–5 scale) for each output
* 2–3 sentences: *How realistic did the cloned or synthesized voice sound? What could be improved?*

### Grading:

* 6 pts — TTS code runs and produces audio
* 7 pts — Voice cloning attempted (Option A) or two TTS tools compared (Option B)
* 7 pts — MOS scoring and written reflection

---

## Task 4 — Final Report and Reflection (15 points)

**Goal:** Summarize your findings in a short report.
*(Maps to Project Goal 6 — includes infrastructure and runtime considerations)*

### Write a short report (1–2 pages or equivalent notebook cells) that includes:

1. **Introduction** — What is the problem? Why is Telugu ASR/TTS important?
2. **Methods** — Which models did you use and why?
3. **Results** — Show your comparison table (Task 2) and TTS examples (Task 3)
4. **Key Finding** — In your opinion, which model is best for Telugu and why?
5. **Infrastructure Notes** — Can these models run on a personal laptop/desktop, or is a cloud GPU (e.g., Google Colab with GPU runtime) required? What did you observe?
6. **Connection to Course** — How does this project connect to models you learned earlier
   (e.g., how is a Transformer used inside Whisper or XTTS-v2)?
7. **What's Next** — If you had more time, what would you try next?

### Deliverable:

* Report written in a markdown cell at the top or bottom of the notebook, OR
* A separate Google Doc or PDF submitted to Classroom

### Grading:

* 5 pts — Introduction + Methods explained clearly
* 6 pts — Results and key finding with evidence
* 4 pts — Infrastructure notes + Connection to course + "What's Next" reflection

---

## Task 5 — Pricing Tier Reflection (15 points)

**Goal:** Think critically about free vs. freemium vs. paid tools, and make a recommendation.
*(Maps to Project Goals 4 and 6)*

### What to do:

**Part A — Build a Pricing Comparison Table (5 pts)**

Fill in the table below for the models you used. Research the actual pricing pages
(linked in the Introduction) and note the real numbers.

| Tool | Task | Tier | Free Limit | Paid Price | Privacy | Best For |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI Whisper (self-hosted) | ASR | Free | Unlimited | $0 | Full (local) | Students, researchers |
| Meta MMS | ASR | Free | Unlimited | $0 | Full (local) | Low-resource languages |
| AI4Bharat Vakyansh | ASR | Free | Unlimited | $0 | Full (local) | Indian languages |
| Coqui XTTS-v2 | TTS + Cloning | Free | Unlimited | $0 | Full (local) | Voice cloning on a budget |
| ElevenLabs | TTS + Cloning | Freemium | 10,000 chars/mo | $22/mo (Pro) | Partial | High-quality TTS demos |
| Google Cloud STT | ASR | Paid | 60 min/mo free | $0.016/min | Low | Production apps |
| AssemblyAI | ASR | Freemium | 5 hrs audio free | $0.37/hr | Partial | Startups |

Add any additional tools you tried, and note which goal (ASR, TTS, or Voice Cloning) each tool serves.

**Part B — Written Reflection (5 pts)**

Write 3–5 sentences answering **all three** of these questions:

1. **Cost vs. Quality:** Did the paid or freemium tools produce noticeably better Telugu
   output than the free open-source models? Was the difference worth it for a student project?
2. **Privacy Tradeoff:** When you use a paid API (like Google Cloud), your audio is sent to
   and processed on their servers. Why might this matter for a voice assistant used in someone's home?
   Can you think of a situation where using a free, local model is *better* — even if it's less accurate?
3. **Your Recommendation:** Imagine you are advising a small nonprofit in Andhra Pradesh that
   wants to build a Telugu voice assistant for farmers who can't read. They have a tiny budget
   ($0–$50/month) and handle sensitive personal conversations. Which tier/model combination
   would you recommend, and why?

### Grading:

* 6 pts — Pricing table filled in accurately with real numbers
* 5 pts — Written reflection covers all three questions
* 4 pts — Recommendation is specific and well-reasoned

---

## Task 6 — Unified UI for ASR and TTS (20 points)

**Goal:** Build a simple interactive app that brings all your ASR and TTS models together in one place.
*(Maps to Project Goal 5)*

### What to do:

Build a **Gradio app** with two tabs:

**Tab 1 — ASR (Speech to Text)**
* User uploads a Telugu audio file
* All 4 ASR models from Task 2 run automatically on the file
* Each model's transcription is displayed side-by-side
* User can rate each transcription (thumbs up / thumbs down, or a 1–5 star rating)

**Tab 2 — TTS (Text to Speech)**
* User pastes or types a Telugu sentence
* All TTS models from Task 3 generate audio from that text
* Each model's audio output is playable in the UI
* User can rate each audio output (same 1–5 scale)

### Tips:

* Use `gradio` in Google Colab — it generates a public share link automatically
* You do not need to display all models in real time; you can pre-run them and display results
* Keep the UI simple — labels, audio players, and a rating widget are enough

### Deliverable:

* Working Gradio app code cell in the notebook
* Screenshot of the running app showing both tabs
* 2–3 sentences: *What was the most challenging part of building the UI? What would you add if you had more time?*

### Grading:

* 8 pts — Both tabs functional (ASR upload + TTS text input)
* 6 pts — All models wired up and producing visible/playable output in the UI
* 4 pts — Rating mechanism present in both tabs
* 2 pts — Written reflection

---

## Bonus Challenge (+5 points)

Pick **one** of the following:

* **Bonus B:** Fine-tune Whisper for 1 epoch on 20 of your own Telugu recordings and compare the before/after WER
* **Bonus C:** Try voice cloning in a second Indian language (e.g., Hindi or Tamil) and compare the quality to Telugu

---

## Submission Checklist

* Colab notebook with all 6 tasks completed and outputs visible
* At least 5 Telugu audio files used for testing
* Comparison table (Task 2) with WER/CER for all 4 models
* TTS/voice cloning audio output files (.wav) uploaded or linked
* Final report / reflection (Task 4)
* Screenshot of running Gradio app (Task 6)
* Notebook shared as "anyone with link can view" and link submitted to Google Classroom

---

## Grading Summary

| Task | Points |
| --- | --- |
| Task 1: Data Collection & Setup | 10 |
| Task 2: ASR Model Comparison *(Goal 1)* | 20 |
| Task 3: TTS + Voice Cloning *(Goals 2 & 3)* | 20 |
| Task 4: Final Report *(Goal 6)* | 15 |
| Task 5: Pricing Tier Reflection *(Goals 4 & 6)* | 15 |
| Task 6: Unified ASR + TTS UI *(Goal 5)* | 20 |
| **Total** | **100** |
| Bonus (B or C) | +5 |

---

*Learn and Help — learnandhelp.com*
*"Every voice matters — including yours, in Telugu!"*
