# Telugu Speech AI — Capstone Final Project

### Learn and Help | Python for Machine Learning
### www.learnandhelp.com

---

## What is This Project?

You've spent this year building machine learning models that can **classify images, understand text,
and generate language**. Now it's time to combine those ideas into something personal and meaningful.

In this capstone project, you'll build a **Telugu Speech AI system** that can:

1. **Listen** to spoken Telugu and convert it to text (Speech-to-Text / ASR)
2. **Compare** how well models trained in India vs. the USA understand Telugu
3. **Speak** by synthesizing a new voice — even one that sounds like *you* (Text-to-Speech + Voice Cloning)
4. **Evaluate** whether free, freemium, or paid models are worth their tradeoffs
5. **Build** a unified UI that lets anyone interact with all your models in one place

---

## Project Goals

This project is organized around six goals. Each one connects directly to a task in the
[Assignment File](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Projects/ML_Telugu_Speech_AI_Assignment.md).

| Goal | What You'll Do | Assignment Task |
| --- | --- | --- |
| Goal 1 — ASR | Run 4 speech-to-text models and measure Telugu accuracy | Task 2 |
| Goal 2 — TTS | Generate Telugu audio with 4 text-to-speech models | Task 3 |
| Goal 3 — Voice Cloning | Clone a real voice and synthesize speech with it | Task 3 (Option A) |
| Goal 4 — Model Comparison | Compare free, freemium, and paid models across all tasks | Task 5 |
| Goal 5 — Unified UI | Build a two-tab Gradio app for ASR and TTS | Task 6 |
| Goal 6 — Practical Costs | Evaluate GPU needs, pricing, and token limits | Tasks 4 & 5 |

---

## Project Overview

[![Project Overview](https://github.com/sjasthi/Python-ML-Machine-Learning/raw/main/Projects/images_telugu/telugu_overview.png)](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Projects/images_telugu/telugu_overview.png)

Telugu is one of the oldest classical languages in the world, spoken by 90+ million people.
Yet most voice AI is built for English. This project explores the **gap** — and helps close it.

---

## Part 1: Automatic Speech Recognition (ASR) — Background

### What is ASR?

**Automatic Speech Recognition (ASR)** is the technology that converts spoken audio into written text.
You interact with it every time you say "Hey Siri," dictate a message, or use live captions on YouTube.
At its core, ASR answers one question: *given this sequence of sounds, what words were spoken?*

Think of it like a very fast translator — except it translates from sound waves into letters.

---

### A Brief History of ASR

Understanding where ASR came from helps you appreciate why modern models are so powerful.

**1950s–1970s: Rule-Based Systems**
The earliest ASR systems worked by hard-coding rules: match this sound pattern to this phoneme,
then combine phonemes into words. They could recognize only a handful of isolated words spoken
by a single speaker in a quiet room. They were brittle and almost completely useless for real speech.

**1980s–1990s: Hidden Markov Models (HMMs)**
The big breakthrough was treating speech as a statistical problem. A **Hidden Markov Model** models
speech as a sequence of hidden states (phonemes) that generate observable outputs (audio features).
HMMs allowed systems to handle continuous speech for the first time. Combined with n-gram language
models, they became the backbone of commercial ASR for nearly 20 years — used in early phone
systems, dictation software like Dragon NaturallySpeaking, and voice dialing.

**2010s: Deep Neural Networks (DNNs)**
Researchers discovered that replacing parts of the HMM pipeline with neural networks dramatically
improved accuracy. The acoustic model (which matches sounds to phonemes) was replaced by a DNN,
leading to a 20–30% error rate reduction almost overnight. This was the era of systems like
Google Voice Search and IBM Watson.

**2014–2018: End-to-End Learning (CTC + Attention)**
Instead of separate acoustic, pronunciation, and language models, **end-to-end** architectures
learned to go directly from raw audio to text in a single model. Two key approaches emerged:
CTC (Connectionist Temporal Classification) and attention-based encoder-decoder models.
Models like Baidu's Deep Speech showed that with enough data, a single neural network could
outperform complex pipeline systems.

**2019–Present: Transformers Take Over**
The Transformer architecture — the same one powering GPT and BERT — revolutionized ASR.
Models like OpenAI Whisper and Meta MMS use Transformer-based encoder-decoders trained on
hundreds of thousands of hours of audio. They are multilingual, robust to accents and noise,
and can be run locally on a laptop CPU (slowly) or a GPU (fast).

---

### How Modern ASR Works — The Pipeline

Even though the latest models are "end-to-end," it helps to understand the internal stages:

```
Raw Audio (.wav file)
        ↓
[1. Preprocessing]
  • Resample to 16,000 Hz (16kHz)
  • Convert to mono (single channel)
  • Normalize volume
        ↓
[2. Feature Extraction]
  • Convert audio to a mel spectrogram
  • A spectrogram shows frequency vs. time (like a heat map of sound)
  • Mel scale matches how the human ear perceives pitch
        ↓
[3. Encoder (Transformer)]
  • Reads the mel spectrogram as a sequence
  • Produces a rich representation of the audio's meaning
  • Similar to how BERT reads a sentence of text
        ↓
[4. Decoder (Transformer)]
  • Generates text one token at a time
  • Uses attention to "look back" at the audio encoding
  • For multilingual models, a language tag (e.g., <|te|> for Telugu) is passed here
        ↓
Output: Telugu Text
```

**What is a Mel Spectrogram?**
A spectrogram is a 2D image of sound — the horizontal axis is time, the vertical axis is
frequency, and the brightness of each pixel shows how loud that frequency is at that moment.
The "mel" scale compresses higher frequencies (where human hearing is less sensitive) to
match how our ears actually work. This is what the model "sees" when it listens to audio.

---

### Why Telugu ASR is Hard

Telugu is a classical Dravidian language with properties that make it challenging for ASR:

**Script complexity:** Telugu has 56 characters in its base alphabet, plus hundreds of
conjunct characters formed by combining consonants. A single "syllable" can span multiple
Unicode code points. This makes character-level error rate (CER) especially important.

**Agglutinative morphology:** Telugu stacks suffixes onto root words to express tense,
number, and case. The word "వెళ్తున్నారు" (they are going) is a single word. A model
that misses one suffix completely changes the meaning.

**Low-resource problem:** English ASR models are trained on millions of hours of data.
Telugu datasets — even the best ones like OpenSLR's Telugu corpus — are a fraction of
that size. Models trained on scarce data make more errors, especially on rare words.

**Speaker variation:** Telugu is spoken differently in Andhra Pradesh vs. Telangana,
with dialect differences in vocabulary and pronunciation. A model trained on one dialect
may struggle with another.

---

### Key ASR Metrics

**WER — Word Error Rate**

> WER = (Substitutions + Deletions + Insertions) / Total Reference Words × 100%

WER counts how many words the model got wrong, expressed as a percentage.
A WER of 0% is perfect. A WER of 100% means every word was wrong.
For context: state-of-the-art English ASR achieves around 2–5% WER. Telugu ASR typically
ranges from 15–40% depending on the model and audio quality.

Example:
```
Reference (correct):  నేను    తెలుగు  నేర్చుకుంటున్నాను
Hypothesis (model):   నేను    తెలుగు  నేర్చుకుంటున్నాడు

Substitution: 1 word wrong ("నాను" vs "నేను")
WER = 1/3 × 100% = 33.3%
```

**CER — Character Error Rate**
Same formula but at the character level. For languages like Telugu with complex script,
CER is often more informative than WER — a single wrong suffix character shouldn't
count as an entirely wrong word.

**RTF — Real-Time Factor**
RTF = (time to transcribe audio) / (duration of audio)

An RTF of 0.5 means the model transcribes 1 minute of audio in 30 seconds — twice as fast
as real-time. An RTF > 1.0 means it's slower than real-time and not usable for live applications.

---

## Part 2: India-Origin vs. USA-Origin ASR Models

This is the **core comparison** of the project.

[![India vs USA Models](https://github.com/sjasthi/Python-ML-Machine-Learning/raw/main/Projects/images_telugu/india_vs_usa.png)](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Projects/images_telugu/india_vs_usa.png)

### USA-Origin Models

| Model | Creator | Architecture | Training Data |
| --- | --- | --- | --- |
| **OpenAI Whisper** | OpenAI (San Francisco) | Transformer encoder-decoder | 680,000 hours of multilingual audio from the internet |
| **Meta MMS** | Meta AI (New York/California) | Wav2Vec 2.0 backbone | 1,100+ languages via religious text recordings (MMS dataset) |

### India-Origin Models

| Model | Creator | Architecture | Training Data |
| --- | --- | --- | --- |
| **AI4Bharat IndicWhisper** | IIT Madras + AI4Bharat | Whisper fine-tuned | IndicSUPERB dataset — curated Indian language audio |
| **Vakyansh** | EkStep Foundation / AI4Bharat | Wav2Vec 2.0 | 10+ Indian languages; Telugu-specific checkpoint available |

### The Big Question

> Do models built specifically for Indian languages perform better on Telugu
> than models trained on massive multilingual datasets from the USA?

This is not obvious. Whisper was trained on far more data overall — but how much of that
was Telugu? IndicWhisper was specifically fine-tuned on Indian languages — but does
specialization beat scale?

Your experiments in Task 2 will give you a real answer.

### Sample Results (you'll compute your own!)

[![WER Comparison](https://github.com/sjasthi/Python-ML-Machine-Learning/raw/main/Projects/images_telugu/wer_comparison.png)](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Projects/images_telugu/wer_comparison.png)

> Note: The values above are illustrative examples. Your actual results will vary based on
> your test audio clips and the specific model versions you use.

---

## Part 3: Text-to-Speech (TTS) — Background

### What is TTS?

**Text-to-Speech (TTS)** is the reverse of ASR: it converts written text into spoken audio.
TTS is how your phone reads navigation directions aloud, how audiobooks are narrated by AI,
and how accessibility tools help visually impaired users hear web content.

A good TTS system produces audio that sounds natural, correctly pronounces the language,
and conveys the right rhythm and intonation (called **prosody**).

---

### A Brief History of TTS

**1960s–1980s: Formant Synthesis**
Early TTS systems generated speech by mathematically modeling the human vocal tract — using
equations to simulate the resonant frequencies (formants) produced by different mouth shapes.
The result was robotic and instantly recognizable as not human. Think of Stephen Hawking's
speech synthesizer. These systems required no recorded audio at all — everything was computed
from rules.

**1990s–2000s: Concatenative Synthesis**
Instead of equations, concatenative systems recorded a human speaker saying thousands of
short audio segments (phones, diphones, triphones), then stitched them together at runtime.
This produced much more natural-sounding speech — but only for the one voice that was recorded.
Changing the voice required recording a new speaker. AT&T's Natural Voices and Festival TTS
were popular examples.

**2010s: Statistical Parametric Synthesis (HMM-TTS)**
Similar to ASR's HMM era, TTS systems switched to generating speech parameters statistically
rather than stitching recordings. This allowed more flexible control over prosody and made
it easier to build voices for new languages — but the output still had a characteristic
"buzzy" quality.

**2017: WaveNet Changes Everything**
DeepMind's **WaveNet** (2016) showed that a deep neural network could generate raw audio
waveforms sample-by-sample and produce speech indistinguishable from humans in listening tests.
This was the first time a neural network generated audio directly rather than predicting
parameters for a traditional vocoder.

**2018–2022: Neural TTS Becomes Practical**
WaveNet was too slow for real-time use. A wave of faster architectures followed:
**Tacotron 2** (Google, 2017) and **FastSpeech 2** (Microsoft, 2020) introduced the
two-stage pipeline that most modern TTS uses today:
1. A **text-to-mel** model converts text into a mel spectrogram
2. A **vocoder** (neural signal processor) converts the mel spectrogram into audio waveforms

**2023–Present: End-to-End + Voice Cloning at Scale**
Modern models like **VITS**, **Coqui XTTS-v2**, and **ElevenLabs** combine both stages
into a single end-to-end model and add voice cloning — the ability to adapt the model to
any speaker from just a short audio sample.

---

### How Neural TTS Works — The Pipeline

```
Input Text ("నేను తెలుగు మాట్లాడతాను")
        ↓
[1. Text Analysis]
  • Tokenize text into characters or subword units
  • For Telugu: handle complex conjuncts and matras (vowel diacritics)
  • Convert to phoneme sequence if using a pronunciation dictionary
        ↓
[2. Text Encoder (Transformer)]
  • Converts text tokens into a sequence of embeddings
  • Captures meaning, context, and linguistic structure
        ↓
[3. Duration Model / Alignment]
  • Decides how long each phoneme should be spoken
  • Controls speaking rate and rhythm (prosody)
        ↓
[4. Acoustic Decoder]
  • Predicts a mel spectrogram from the encoded text
  • This is the "blueprint" of what the audio should sound like
        ↓
[5. Vocoder (HiFi-GAN or similar)]
  • Converts the mel spectrogram into an actual audio waveform
  • The output is a .wav file you can play
        ↓
Output: Telugu Audio (.wav)
```

**Why two stages?**
The mel spectrogram is a compact, structured representation that is much easier to learn
from text than raw audio waveforms. Vocoders, trained separately, are very good at
converting spectrograms to realistic audio. Splitting the problem makes training more stable
and the results much higher quality.

---

### TTS Models in This Project

You will use at least two USA-based and two India-based TTS models. The defaults are:

| Model | Origin | Type | Voice Cloning? |
| --- | --- | --- | --- |
| **Coqui XTTS-v2** | USA (open source) | End-to-end neural TTS | ✅ Yes — 10–30 sec sample |
| **ElevenLabs** | USA | Commercial cloud API | ✅ Yes — instant clone |
| **AI4Bharat IndicTTS** | India (IIT Madras) | Neural TTS for Indian languages | ❌ Default voices only |
| **Google TTS (`gtts`)** | USA / India (multilingual) | Cloud-based TTS | ❌ Default voices only |

> 💡 You may substitute models here just as in Task 2 — for example, Sarvam AI's TTS API
> (India) or Microsoft Azure Neural TTS (USA). Document any substitution with a brief note.

---

### How to Measure TTS Quality

Unlike ASR, there is no single "ground truth" for TTS — you can't compute a WER because
you're generating audio, not transcribing it. Instead, TTS quality is measured using a
combination of human judgment and automated metrics.

**MOS — Mean Opinion Score**
The gold standard. Human listeners rate the audio on a scale of 1 to 5:

| Score | Meaning |
| --- | --- |
| 5 | Excellent — completely natural, indistinguishable from a human |
| 4 | Good — mostly natural, slight artificiality |
| 3 | Fair — clearly synthetic but understandable |
| 2 | Poor — noticeable errors, unpleasant |
| 1 | Bad — nearly unintelligible |

For this project, you will be the listener — score each model yourself for each sentence.

**Intelligibility**
Can you understand every word? Intelligibility is a binary check before you even rate quality.
If you can't understand it, it scores 1.

**Naturalness**
Does the speech flow with the right rhythm and intonation? Does the stress fall on the right syllables?
Telugu has complex intonation patterns — models trained specifically on Telugu tend to score
higher here than general-purpose models.

**Telugu Pronunciation Accuracy**
Does the model correctly pronounce conjunct consonants and vowel diacritics?
Try sentences with challenging combinations and note any mispronunciations.

---

## Part 4: Voice Cloning — Background

### What is Voice Cloning?

Voice cloning takes TTS one step further. Instead of speaking in a generic default voice,
the model learns the specific characteristics of *your* voice from a short recording —
then generates new speech that sounds like you saying anything you type.

Real-world uses include: audiobook narration (author reads their own book without hours of
recording), accessibility tools (users with ALS preserve their voice before losing it),
and entertainment (dubbing actors into new languages while keeping their vocal style).

---

### How Voice Cloning Works — Speaker Embeddings

The key concept is a **speaker embedding** — a mathematical "fingerprint" of a voice.

```
Your Voice Recording (10–30 seconds)
        ↓
[Speaker Encoder]
  • A neural network trained to distinguish between thousands of speakers
  • Extracts features: pitch range, speaking rate, timbre, resonance
  • Compresses all of this into a single vector (e.g., 512 numbers)
        ↓
Speaker Embedding (a unique vector for your voice)
        ↓
[TTS Model receives BOTH text AND your embedding]
  • Normal TTS would speak in a default voice
  • With your embedding, it "conditions" the output on your vocal style
        ↓
Audio that sounds like YOU saying the new text
```

The speaker encoder is typically trained separately on a large dataset of many speakers
(called speaker verification data). The model learns to map any voice to a point in a
high-dimensional space where similar-sounding voices are close together.

**Zero-shot voice cloning** means the model can clone a voice it has *never heard before*
during training — it generalizes from its understanding of speaker differences to any new voice.
This is what makes Coqui XTTS-v2 and ElevenLabs impressive: you provide a clip, and within
seconds they generate audio in that voice.

---

### Ethics of Voice Cloning

Voice cloning is powerful — and raises real ethical questions that you should think about
as you build this project.

**Consent:** Was the original speaker's voice recorded with their permission?
Cloning a voice without consent — especially of a public figure — can be done but is ethically
problematic and in some jurisdictions illegal.

**Deepfakes:** A cloned voice can be used to make someone appear to say things they never said.
This is already being used for fraud (fake CEO voice calls authorizing wire transfers) and
political disinformation.

**Ownership:** If you clone a famous singer's voice and use it to generate a new song,
who owns the output? This is an active legal debate worldwide.

**Guardrails in responsible tools:** ElevenLabs requires voice owners to consent before their
voice can be cloned on their platform. Coqui (open source) places no such restrictions —
which is both its power and its responsibility on the user.

> 🤔 **Discussion Question for Task 3:** You'll clone a voice in this project. Whose voice
> are you using, and do you have their permission? What would you do differently if you were
> building a commercial product?

---

## Part 5: How We Measure Success

[![Evaluation Metrics](https://github.com/sjasthi/Python-ML-Machine-Learning/raw/main/Projects/images_telugu/evaluation_metrics.png)](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Projects/images_telugu/evaluation_metrics.png)

### For Speech-to-Text (ASR):

| Metric | What It Measures | Better = |
| --- | --- | --- |
| WER | % of words transcribed incorrectly | Lower |
| CER | % of characters transcribed incorrectly | Lower |
| RTF | Transcription time ÷ audio duration | Lower (< 1.0 = faster than real-time) |

### For Text-to-Speech (TTS):

| Metric | What It Measures | Better = |
| --- | --- | --- |
| MOS | Human-rated naturalness (1–5) | Higher |
| Telugu Pronunciation | Accuracy of conjunct/matra sounds | Higher |
| Intelligibility | Can every word be understood? | Higher |

### For Voice Cloning:

| Metric | What It Measures | Better = |
| --- | --- | --- |
| Speaker Similarity | How close does output sound to the original speaker? | Higher |
| MOS | Overall quality of the cloned speech | Higher |

---

## Analogy: The Sports Team Analogy

Imagine you're picking a cricket team to play a match in Hyderabad:

* **USA-origin models** are like *all-rounders* trained on international pitches — they've
  played in 100 countries and can handle most conditions
* **India-origin models** are like *specialists* who grew up playing on Indian pitches,
  in Indian conditions, against Indian bowling

Who performs better on home turf? That's exactly what this project tests!

---

## Code Activities

### Activity 1: Run OpenAI Whisper on Telugu Audio

```python
import whisper

model = whisper.load_model("medium")
result = model.transcribe("telugu_audio.wav", language="te")
print(result["text"])
```

### Activity 2: Run Meta MMS

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition",
               model="facebook/mms-300m",
               generate_kwargs={"language": "tel"})
result = asr("telugu_audio.wav")
print(result["text"])
```

### Activity 3: Run AI4Bharat IndicWhisper

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition",
               model="ai4bharat/indicwav2vec-hindi",   # swap with Telugu variant
               device=0)  # GPU
result = asr("telugu_audio.wav")
print(result["text"])
```

### Activity 4: Compute WER and CER

```python
from jiwer import wer, cer

ground_truth = "నేను తెలుగు నేర్చుకుంటున్నాను"
hypothesis   = result["text"]

print(f"WER: {wer(ground_truth, hypothesis)*100:.1f}%")
print(f"CER: {cer(ground_truth, hypothesis)*100:.1f}%")
```

### Activity 5: Basic TTS with Google TTS

```python
from gtts import gTTS
import IPython.display as ipd

tts = gTTS(text="నేను తెలుగు మాట్లాడతాను", lang="te")
tts.save("gtts_output.wav")
ipd.Audio("gtts_output.wav")
```

### Activity 6: TTS with AI4Bharat IndicTTS API

```python
import requests

response = requests.post(
    "https://api.ai4bharat.org/tts",
    json={"text": "నేను తెలుగు మాట్లాడతాను", "lang": "te", "gender": "female"}
)
with open("indicttts_output.wav", "wb") as f:
    f.write(response.content)
```

### Activity 7: Voice Cloning with Coqui XTTS-v2

```python
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(
    text="నేను ఆహల. నేను తెలుగు మాట్లాడతాను.",   # "I am Ahala. I speak Telugu."
    speaker_wav="my_voice_sample.wav",              # your 10–30 sec recording
    language="te",
    file_path="cloned_output.wav"
)
```

### Activity 8: Build the Gradio UI (Task 6 Preview)

```python
import gradio as gr

def run_asr(audio_file):
    # Run all 4 ASR models and return their transcriptions
    whisper_out  = run_whisper(audio_file)
    mms_out      = run_mms(audio_file)
    indic_out    = run_indic_whisper(audio_file)
    vakyansh_out = run_vakyansh(audio_file)
    return whisper_out, mms_out, indic_out, vakyansh_out

def run_tts(text):
    # Run all TTS models and return audio file paths
    gtts_out  = run_gtts(text)
    indic_out = run_indictts(text)
    xtts_out  = run_xtts(text)
    eleven_out = run_elevenlabs(text)
    return gtts_out, indic_out, xtts_out, eleven_out

with gr.Blocks() as demo:
    with gr.Tab("ASR — Speech to Text"):
        audio_input  = gr.Audio(type="filepath", label="Upload Telugu Audio")
        asr_button   = gr.Button("Transcribe with All Models")
        whisper_out  = gr.Textbox(label="Whisper")
        mms_out      = gr.Textbox(label="Meta MMS")
        indic_out    = gr.Textbox(label="IndicWhisper")
        vakyansh_out = gr.Textbox(label="Vakyansh")
        asr_button.click(run_asr, inputs=audio_input,
                         outputs=[whisper_out, mms_out, indic_out, vakyansh_out])

    with gr.Tab("TTS — Text to Speech"):
        text_input = gr.Textbox(label="Enter Telugu Text")
        tts_button = gr.Button("Speak with All Models")
        gtts_out   = gr.Audio(label="Google TTS")
        indic_out  = gr.Audio(label="IndicTTS")
        xtts_out   = gr.Audio(label="Coqui XTTS-v2")
        eleven_out = gr.Audio(label="ElevenLabs")
        tts_button.click(run_tts, inputs=text_input,
                         outputs=[gtts_out, indic_out, xtts_out, eleven_out])

demo.launch(share=True)  # share=True gives you a public URL in Colab
```

---

## Part 6: Free vs. Freemium vs. Paid — What's the Difference?

Every tool in this project falls into one of three pricing tiers.
Understanding this helps you make smart decisions as a developer — not just for Telugu AI,
but for any real-world project.

[![Pricing Tiers](https://github.com/sjasthi/Python-ML-Machine-Learning/raw/main/Projects/images_telugu/pricing_tiers.png)](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Projects/images_telugu/pricing_tiers.png)

### The Three Tiers Explained

**Free / Open Source**
The full model is publicly released — you download and run it yourself.
There are no usage limits, no monthly bill, and your audio data never leaves your computer.
The tradeoff: you need a GPU (Google Colab provides one for free), and setup takes more effort.

Examples in this project: OpenAI Whisper, Meta MMS, AI4Bharat Vakyansh, Coqui XTTS-v2.

**Freemium**
A company hosts the model for you and gives you a free quota each month.
When you exceed the quota, you either wait or pay. Great for experimenting — but watch the limits.

Examples in this project: ElevenLabs (10,000 characters/month free), Hugging Face Inference API
(free but rate-limited at peak hours), AssemblyAI (5 free hours of audio).

**Paid / Commercial API**
You send audio to the company's servers and pay per minute or per character.
In return, you get the fastest speeds, the best accuracy, and enterprise-grade support.
The tradeoff: your audio is processed on someone else's server (privacy consideration), and
costs can add up quickly for large projects.

Examples in this project: Google Cloud Speech-to-Text ($0.016/min), Azure Speech ($1.00/hr),
ElevenLabs Pro ($22/month).

---

### Tradeoff Radar: Visualizing the Differences

[![Tradeoff Radar](https://github.com/sjasthi/Python-ML-Machine-Learning/raw/main/Projects/images_telugu/tradeoff_radar.png)](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Projects/images_telugu/tradeoff_radar.png)

| Dimension | Free (Whisper/MMS) | Freemium (ElevenLabs) | Paid (Google Cloud) |
| --- | --- | --- | --- |
| Accuracy | Good | Very Good | Best |
| Speed | Depends on your GPU | Fast | Fastest |
| Privacy | Full — data stays local | Partial — sent to server | Lowest — stored by provider |
| Cost Efficiency | Best (free forever) | Good up to quota | Expensive at scale |
| Ease of Use | Moderate (setup needed) | Easy (web UI + API) | Easy (but billing setup) |
| Customization | Highest (fine-tune freely) | Limited | Some (custom models extra $) |

**Key Insight:** For a student project, free models are almost always the right choice.
For a startup building a product with millions of users, paid APIs may be worth the cost.

---

### The Analogy: Restaurant vs. Home Cooking vs. Meal Kit

Think of it this way:

* **Free/Open Source** = cooking at home from scratch. Total control, costs almost nothing,
  but you need to know what you're doing and buy the ingredients.
* **Freemium** = a meal kit service (HelloFresh, etc.). Someone preps the ingredients and
  gives you the recipe. A few meals are free; after that you pay per box.
* **Paid API** = ordering from a restaurant. Fastest and easiest — just place your order
  and it arrives ready. But you pay for every dish, and you can't see the kitchen.

---

### Can This Run on a Laptop? (Goal 6)

This is a real question, not just a theoretical one. Speech AI is compute-intensive.

| Model | Can Run on Laptop CPU? | Recommended Environment |
| --- | --- | --- |
| Whisper (tiny/base) | ✅ Yes, slowly | Laptop CPU is OK for testing |
| Whisper (medium/large) | ⚠️ Very slow | Colab GPU strongly recommended |
| Meta MMS | ⚠️ Possible but slow | Colab GPU recommended |
| IndicWhisper | ❌ Needs GPU | Google Colab with T4 GPU |
| Vakyansh | ❌ Needs GPU | Google Colab with T4 GPU |
| Coqui XTTS-v2 | ⚠️ Very slow on CPU | Colab GPU strongly recommended |
| ElevenLabs API | ✅ Yes (cloud-based) | Any machine with internet |
| Google TTS (`gtts`) | ✅ Yes (cloud-based) | Any machine with internet |

**Bottom line:** For full experiments, use Google Colab with GPU runtime enabled
(Runtime → Change runtime type → T4 GPU). It's free and fast enough for this project.
Cloud-based APIs (ElevenLabs, Google TTS) run on any machine since the heavy computation
happens on their servers.

---

### Which Model Should Ahala Use?

[![Decision Flowchart](https://github.com/sjasthi/Python-ML-Machine-Learning/raw/main/Projects/images_telugu/pricing_decision.png)](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Projects/images_telugu/pricing_decision.png)

For this school project: **start with free models** (Whisper + Vakyansh + Coqui XTTS-v2)
run on Google Colab. They are more than powerful enough, and you'll learn the most by
running the models yourself rather than just calling an API.

Use a freemium tool (ElevenLabs free tier) to *compare* quality — it's a great benchmark
for how much better a commercial tool sounds, and it's free within the limit.

---

## Resources

### Playgrounds & Tools

* [AI4Bharat Demo](https://ai4bharat.iitm.ac.in/) — Test IndicASR and IndicTTS directly in browser
* [OpenAI Whisper on Hugging Face](https://huggingface.co/openai/whisper-medium)
* [Meta MMS on Hugging Face](https://huggingface.co/facebook/mms-300m)
* [Coqui TTS GitHub](https://github.com/coqui-ai/TTS)
* [ElevenLabs](https://elevenlabs.io/) — Free tier available for voice cloning
* [Sarvam AI](https://www.sarvam.ai/) — Indian startup with Telugu ASR and TTS APIs

### Telugu Audio Datasets

* [OpenSLR Telugu](https://openslr.org/66/) — Free Telugu speech corpus
* [IndicSUPERB](https://github.com/AI4Bharat/IndicSUPERB) — Benchmark dataset for Indian languages
* Record your own sentences using Audacity or your phone!

### Further Reading

* [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356) — "Robust Speech Recognition via Large-Scale Weak Supervision"
* [Meta MMS Paper](https://arxiv.org/abs/2305.13516) — "Scaling Speech Technology to 1,000+ Languages"
* [AI4Bharat IndicSUPERB](https://arxiv.org/abs/2208.11761) — Indian language speech benchmarks
* [XTTS-v2 on Hugging Face](https://huggingface.co/coqui/XTTS-v2) — Voice cloning model card

### YouTube Videos

* Search: "AI4Bharat IndicASR Telugu demo"
* Search: "Coqui XTTS voice cloning tutorial"
* Search: "OpenAI Whisper multilingual"
* Search: "How does WaveNet work explained"

---

## Project Timeline

| Week | Task |
| --- | --- |
| Week 1 | Set up Colab, install libraries, record Telugu audio samples (Task 1) |
| Week 2 | Run all 4 ASR models, compute WER/CER, build comparison table (Task 2) |
| Week 3 | Experiment with TTS and voice cloning, rate outputs with MOS (Task 3) |
| Week 4 | Write final report and pricing reflection (Tasks 4 & 5) |
| Week 5 | Build Gradio UI with both ASR and TTS tabs (Task 6) |

---

## What You'll Submit

See the [Assignment File](https://github.com/sjasthi/Python-ML-Machine-Learning/blob/main/Projects/ML_Telugu_Speech_AI_Assignment.md) for the full task list, point breakdown, and submission checklist.

---

*Learn and Help — learnandhelp.com*
*"From Voice to Text, From Text to Voice — in Telugu!"*
