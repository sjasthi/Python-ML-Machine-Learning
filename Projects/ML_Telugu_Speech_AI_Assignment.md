# Telugu Speech AI — Capstone Assignment
### Learn and Help | Python Machine Learning Course | Week 27

**Student:** Ahala
**Points:** 50 points total
**Submission:** Google Classroom

---

## Overview

This capstone project combines **everything you've learned** — data pipelines, model comparison,
evaluation metrics, and AI APIs — applied to a real-world problem:
**making AI understand and speak Telugu**.

Reference the [Project Introduction](ML_Telugu_Speech_AI_Introduction.md) and the
[Colab Notebook](../Colab_Notebooks/ML_Telugu_Speech_AI_Project.ipynb) as you work through the tasks.

---

## Task 1 — Data Collection and Setup (10 points)

**Goal:** Collect or record Telugu audio and prepare it for experiments.

### What to do:
1. **Collect at least 5 Telugu audio clips** (each 5–15 seconds long). You can:
   - Record yourself speaking simple Telugu sentences
   - Download samples from [OpenSLR Telugu Dataset](https://openslr.org/66/)
   - Use sentences suggested in the notebook
2. **Write the ground truth transcription** (the exact Telugu text) for each clip
3. **Preprocess all clips** to 16kHz, mono WAV format using the notebook code
4. **Create a data table** listing: filename, duration (seconds), sentence, source (recorded/downloaded)

### Deliverable:
- Code cell in notebook showing preprocessing
- A table with your 5+ audio clips and their metadata
- Screenshot or output showing the audio waveform of at least one clip

---

## Task 2 — Speech-to-Text Model Comparison (15 points)

**Goal:** Run your Telugu audio through at least 3 models and compare their accuracy.

### What to do:
1. **Run at least 3 of the 4 models** below on your audio clips:
   - OpenAI Whisper (USA) — `whisper` package
   - Meta MMS (USA) — `facebook/mms-300m` on Hugging Face
   - AI4Bharat IndicWhisper (India) — `ai4bharat/indicwav2vec` on Hugging Face
   - Vakyansh (India) — `Vakyansh/wav2vec2-telugu-tem-100` on Hugging Face
2. **Record the transcription output** from each model for each clip
3. **Compute WER and CER** for each model using the `jiwer` library
4. **Create a comparison table** with columns: Model, Origin, WER%, CER%, Speed (RTF)

### Deliverable:
- All model inference code cells in the notebook (with outputs)
- A summary table comparing all models
- 2–3 sentences answering: *Which model performed best on Telugu? Were India-origin models better?*

### Grading:
- 5 pts — At least 3 models run with output
- 5 pts — WER/CER correctly computed for all models
- 5 pts — Clear comparison table + written analysis

---

## Task 3 — Text-to-Speech and Voice Cloning (15 points)

**Goal:** Synthesize Telugu speech — ideally using your own cloned voice.

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
- Test with at least 3 different Telugu sentences
- Listen to the output and note what sounds good / wrong

### Deliverable:
- Code cells with TTS/cloning code and audio output saved as `.wav` files
- Your MOS ratings (1–5 scale) for each output
- 2–3 sentences: *How realistic did the cloned or synthesized voice sound? What could be improved?*

### Grading:
- 5 pts — TTS code runs and produces audio
- 5 pts — Voice cloning attempted (Option A) or two TTS tools compared (Option B)
- 5 pts — MOS scoring and written reflection

---

## Task 4 — Final Report and Reflection (10 points)

**Goal:** Summarize your findings in a short report.

### Write a short report (1–2 pages or equivalent notebook cells) that includes:

1. **Introduction** — What is the problem? Why is Telugu ASR/TTS important?
2. **Methods** — Which models did you use and why?
3. **Results** — Show your comparison table (Task 2) and TTS examples (Task 3)
4. **Key Finding** — In your opinion, which model is best for Telugu and why?
5. **Connection to Course** — How does this project connect to models you learned earlier
   (e.g., how is a Transformer used inside Whisper or XTTS-v2)?
6. **What's Next** — If you had more time, what would you try next?

### Deliverable:
- Report written in a markdown cell at the top or bottom of the notebook, OR
- A separate Google Doc or PDF submitted to Classroom

### Grading:
- 3 pts — Introduction + Methods explained clearly
- 4 pts — Results and key finding with evidence
- 3 pts — Connection to course + "What's Next" reflection

---

## Task 5 — Pricing Tier Reflection (10 points)

**Goal:** Think critically about free vs. freemium vs. paid tools, and make a recommendation.

### What to do:

**Part A — Build a Pricing Comparison Table (5 pts)**

Fill in the table below for the models you used. Research the actual pricing pages
(linked in the Introduction) and note the real numbers.

| Tool | Tier | Free Limit | Paid Price | Privacy | Best For |
|---|---|---|---|---|---|
| OpenAI Whisper (self-hosted) | Free | Unlimited | $0 | Full (local) | Students, researchers |
| Meta MMS | Free | Unlimited | $0 | Full (local) | Low-resource languages |
| AI4Bharat Vakyansh | Free | Unlimited | $0 | Full (local) | Indian languages |
| Coqui XTTS-v2 | Free | Unlimited | $0 | Full (local) | Voice cloning on a budget |
| ElevenLabs | Freemium | 10,000 chars/mo | $22/mo (Pro) | Partial | High-quality TTS demos |
| Google Cloud STT | Paid | 60 min/mo free | $0.016/min | Low | Production apps |
| AssemblyAI | Freemium | 5 hrs audio free | $0.37/hr | Partial | Startups |

Add any additional tools you tried.

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
- 5 pts — Pricing table filled in accurately with real numbers
- 3 pts — Written reflection covers all three questions
- 2 pts — Recommendation is specific and well-reasoned

---

## Bonus Challenge (+5 points)

Pick **one** of the following:

- **Bonus A:** Build a simple Gradio app where a user can upload a Telugu audio file and
  see the transcription from all 3 models side-by-side
- **Bonus B:** Fine-tune Whisper for 1 epoch on 20 of your own Telugu recordings and
  compare the before/after WER
- **Bonus C:** Try voice cloning in a second Indian language (e.g., Hindi or Tamil)
  and compare the quality to Telugu

---

## Submission Checklist

- [ ] Colab notebook with all 4 tasks completed and outputs visible
- [ ] At least 5 Telugu audio files used for testing
- [ ] Comparison table (Task 2) with WER/CER for 3+ models
- [ ] TTS/voice cloning audio output files (.wav) uploaded or linked
- [ ] Final report / reflection (Task 4)
- [ ] Notebook shared as "anyone with link can view" and link submitted to Google Classroom

---

## Grading Summary

| Task | Points |
|---|---|
| Task 1: Data Collection & Setup | 10 |
| Task 2: ASR Model Comparison | 15 |
| Task 3: TTS + Voice Cloning | 15 |
| Task 4: Final Report | 10 |
| Task 5: Pricing Tier Reflection | 10 |
| **Total** | **60** |
| Bonus | +5 |

---

*Learn and Help — learnandhelp.com*
*"Every voice matters — including yours, in Telugu!"*
