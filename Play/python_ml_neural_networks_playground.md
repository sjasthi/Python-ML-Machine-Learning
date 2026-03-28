# Neural Networks Playground – Student Guide

Welcome! In this activity, you'll explore how neural networks learn by experimenting with an interactive tool from Google.

**Open the playground here:** [https://playground.tensorflow.org](https://playground.tensorflow.org)

---

## What You'll See on Screen

When you open the playground, you'll see three main areas:

```
┌──────────────┬───────────────────────────┬──────────────┐
│  LEFT PANEL  │      MIDDLE AREA          │  RIGHT SIDE  │
│              │                           │              │
│  - Dataset   │  Input → Hidden → Output  │  Output      │
│  - Settings  │  (The Neural Network)     │  Visualization│
│  - Features  │                           │  (Blue/Orange)│
│              │                           │              │
└──────────────┴───────────────────────────┴──────────────┘
```

- **Left panel:** Choose your data, features, and settings (learning rate, activation, etc.)
- **Middle area:** The neural network — you can add/remove layers and neurons here
- **Right side:** Shows how well the model is separating blue from orange dots
- **Top bar:** Play/Pause button, Epoch counter, and Loss values

---

## Part 1: Glossary (Key Terms with Simple Analogies)

### 1. Dataset
- **What it means:** The data you use to train the model.
- **Analogy:** Practice questions before an exam — the more you practice, the better you get.

```
 ● ● ● ● (Blue class)
 ○ ○ ○ ○ (Orange class)
```

In the playground, you can choose from four datasets: Circle, XOR, Gaussian, and Spiral. Each one is a different "shape" the model needs to learn.

---

### 2. Features (X1, X2, etc.)
- **What it means:** Inputs to the model — the information it uses to make predictions.
- **Analogy:** Ingredients in a recipe — different ingredients change the final dish.

```
X1 → Horizontal position
X2 → Vertical position
```

The playground also offers transformed features like X1², X2², X1·X2, sin(X1), and sin(X2). These help the model detect curved patterns.

---

### 3. Label (Output)
- **What it means:** The answer the model is trying to predict (blue or orange).
- **Analogy:** The final grade on your test — it's what everything builds toward.

```
Input Features → [Model] → Predicted Label (Blue or Orange)
```

---

### 4. Epoch
- **What it means:** One full pass through the entire dataset during training.
- **Analogy:** Reading a textbook once from start to finish. More passes = more learning.

```
Epoch 1 → Epoch 2 → Epoch 3 → ... → Epoch 500
        (model gets better each time)
```

Watch the epoch counter at the top of the playground as the model trains!

---

### 5. Learning Rate
- **What it means:** How much the model adjusts itself after each mistake. Controls the "step size."
- **Analogy:** Walking toward a target — big steps get you close fast but you might overshoot. Small steps are slow but precise.

```
Too big  (0.1+)  → Overshoots, never settles
Just right (0.01) → Learns smoothly
Too small (0.001) → Takes forever
```

---

### 6. Activation Function
- **What it means:** A math function that helps the model learn curved and complex patterns (adds "non-linearity").
- **Analogy:** A decision filter — it transforms the signal passing through each neuron.

Options you'll see in the dropdown:

| Activation | What it does | When to use |
|------------|-------------|-------------|
| ReLU | Keeps positive values, turns negatives to 0 | Good default choice |
| Tanh | Squishes values between -1 and 1 | Works well for most problems |
| Sigmoid | Squishes values between 0 and 1 | Good for yes/no decisions |
| Linear | No change at all | Rarely useful alone |

---

### 7. Hidden Layers
- **What it means:** Layers between the input and output where the actual learning happens.
- **Analogy:** Brain layers — each layer detects more complex patterns than the last.

```
Input → [Hidden Layer 1] → [Hidden Layer 2] → Output
         (simple edges)     (complex shapes)
```

You can add up to 6 hidden layers using the + and - buttons in the playground.

---

### 8. Neurons
- **What it means:** Individual processing units inside each layer. Each one learns a small piece of the pattern.
- **Analogy:** Students in a group project — each one solves a different part, and together they solve the whole problem.

```
Layer with 4 neurons:   (●) (●) (●) (●)
Layer with 2 neurons:   (●) (●)
```

More neurons = the layer can detect more patterns, but too many can cause problems.

---

### 9. Weights
- **What it means:** Numbers that control how much each input matters. The model adjusts these during training.
- **Analogy:** Trust level — a high weight means "I trust this input a lot."

```
X1 ──[0.9]──→  (high importance)
X2 ──[0.2]──→  (low importance)
```

In the playground, the lines connecting neurons show the weights. Thicker/darker lines = bigger weights.

---

### 10. Bias
- **What it means:** An extra number added to shift the output. Helps the model fit data better.
- **Analogy:** A head start — even before looking at the inputs, the neuron starts with a small boost or penalty.

```
Output = (Input × Weight) + Bias
```

---

### 11. Training Loss
- **What it means:** How many mistakes the model makes on the data it's training on. Lower = better.
- **Analogy:** Errors on your practice homework.

```
High loss (0.5) → Model is confused
Low loss (0.01) → Model learned the pattern
```

---

### 12. Test Loss
- **What it means:** How many mistakes the model makes on new data it hasn't seen before. This is the real test.
- **Analogy:** Errors on the actual exam (not the practice).

A good model has low training loss AND low test loss.

---

### 13. Overfitting
- **What it means:** The model memorized the training data instead of learning the real pattern. Works great on practice, fails on the test.
- **Analogy:** Memorizing every answer to the study guide word-for-word, then failing because the real test asks the same ideas in different ways.

```
Training Loss: 0.001  ← Looks great!
Test Loss:     0.450  ← Fails on new data!
```

**How to spot it:** Big gap between training loss and test loss.

---

### 14. Underfitting
- **What it means:** The model is too simple to learn the pattern. Fails on both training and test data.
- **Analogy:** Not studying enough for the exam.

```
Training Loss: 0.400  ← Still bad
Test Loss:     0.420  ← Also bad
```

**How to spot it:** Both losses stay high no matter how long you train.

---

### 15. Regularization
- **What it means:** A technique to prevent overfitting by keeping the model from getting too complex.
- **Analogy:** A teacher saying "Keep your answer simple and clear — don't overthink it."

Options you'll see in the dropdown:

| Option | What it does |
|--------|-------------|
| None | No restriction |
| L1 | Pushes unimportant weights to exactly 0 (simpler model) |
| L2 | Keeps all weights small and smooth |

---

### 16. Regularization Rate
- **What it means:** How strongly regularization is applied. Higher = stricter.
- **Analogy:** How strict the teacher is — a little guidance helps, but too much and you can't learn freely.

---

### 17. Noise
- **What it means:** Random messiness added to the data points. Makes the problem harder.
- **Analogy:** Distractions during a test — some dots are in the "wrong" spot on purpose.

```
No noise:  Clean groups      →  Easy to separate
High noise: Mixed-up dots    →  Hard to separate
```

---

### 18. Batch Size
- **What it means:** How many data points the model looks at before updating its weights.
- **Analogy:** Studying in chunks — do you review 1 flashcard at a time or 30 at once?

| Batch Size | Effect |
|-----------|--------|
| Small (1-10) | Updates often, can be jumpy |
| Large (20-30) | Smoother updates, slower learning |

---

### 19. Classification
- **What it means:** A task where the model predicts a category (like blue vs. orange).
- **Analogy:** Sorting mail into bins — this letter goes HERE, that one goes THERE.

This is the default problem type in the playground.

---

### 20. Regression
- **What it means:** A task where the model predicts a number (like temperature or price).
- **Analogy:** Guessing "how much" instead of "which one."

You can switch to regression using the dropdown at the top of the playground.

---

### 21. Problem Type
- **What it means:** Whether you're doing classification (categories) or regression (numbers).
- **Analogy:** Multiple choice (classification) vs. fill-in-the-blank with a number (regression).

---

## Part 2: How to Play (Step-by-Step)

Follow these steps in order. Take your time with each one!

**Step 1 — Pick a simple dataset.**
Select the first dataset (two clusters/Gaussian). This is the easiest pattern to learn. You'll see blue and orange dots in two separate groups.

**Step 2 — Press Play and watch.**
Click the ▶ Play button in the top-left corner. Watch the background colors change as the model trains. The epoch counter will go up. Try to get the loss below 0.01.

**Step 3 — Change the learning rate.**
Pause the model (click ⏸). Reset it (click 🔄). Try different learning rates: 0.001, 0.01, 0.1, 1, and 10. For each one, press Play and watch. Which learning rate works best? Which ones break?

**Step 4 — Add hidden layers.**
Reset the model. Click the "+" button above the hidden layers to add a new one. Try 1 layer, then 2, then 4. Does more always mean better?

**Step 5 — Try different activation functions.**
Change the activation dropdown to ReLU, Tanh, Sigmoid, and Linear. Reset and retrain for each one. Notice how the decision boundary shape changes.

**Step 6 — Add noise.**
Move the noise slider to 25, then 50. How does this affect the model's ability to learn? Does it need more epochs?

**Step 7 — Add extra features.**
Check the boxes for X1², X2², and X1·X2. These give the model more information to work with. Do they help?

**Step 8 — Watch the loss values.**
Pay attention to both the Training Loss and Test Loss numbers. Are they both going down? If one goes down but the other goes up, that's overfitting!

**Step 9 — Switch to a harder dataset.**
Now try the Circle or Spiral dataset. Can you find settings that solve it? You'll probably need more layers, more neurons, and the right activation function.

---

## Part 3: Experiments (Mini-Challenges)

### Experiment 1: Can a Flat Brain Solve a Circle?
- **Goal:** Try to classify the Circle dataset with zero hidden layers.
- **Steps:** Select the Circle dataset. Remove all hidden layers. Use only X1 and X2 as features. Press Play and let it run for 500+ epochs.
- **What happened?** The model can't solve it! A straight line can't draw a circle.
- **Now try:** Add the features X1² and X2². Reset and retrain. What changed?
- **Think about:** Why did the extra features help? (Hint: circles need curved boundaries.)

### Experiment 2: What Happens with a Crazy Learning Rate?
- **Goal:** See what happens when the learning rate is way too high.
- **Steps:** Pick the Gaussian dataset (two clusters). Set learning rate to 10. Press Play.
- **What happened?** The loss probably jumps around wildly or explodes instead of going down.
- **Now try:** Set it to 0.01 and retrain. Much smoother, right?
- **Think about:** Why do big steps cause problems?

### Experiment 3: The Minimum Neuron Challenge
- **Goal:** Find the fewest neurons needed to solve the XOR dataset.
- **Steps:** Select the XOR dataset. Start with 1 hidden layer and 1 neuron. Train it. If it doesn't work, add 1 more neuron. Keep going until you find the minimum.
- **Think about:** Why can't 1 neuron solve XOR? How many did you need?

### Experiment 4: Break the Model on Purpose
- **Goal:** Create the worst possible model — highest loss you can get!
- **Steps:** Pick the Spiral dataset. Use linear activation. Use only 1 hidden layer with 1 neuron. Set noise to 50. Set learning rate to 10.
- **What happened?** Total chaos! The model can't learn anything.
- **Now try:** Fix one setting at a time. Which fix helped the most?
- **Think about:** What makes a neural network work well vs. fail?

### Experiment 5: Overfitting Detective
- **Goal:** Create a model that overfits (memorizes instead of learning).
- **Steps:** Pick the Gaussian dataset. Add noise (50). Add 4+ hidden layers with 8 neurons each. Set regularization to None. Train for 1000+ epochs.
- **What to watch:** Does the training loss go down while the test loss stays high or goes up? That's overfitting!
- **Now try:** Turn on L2 regularization and retrain. Did it help?

---

## Part 4: Reflection Questions

Answer these after completing the experiments above.

1. What happened when you tried to classify the Circle dataset with no hidden layers? Why couldn't the model solve it?

2. What was the best learning rate you found for the Gaussian dataset? What went wrong when the learning rate was too high or too low?

3. What is the minimum number of neurons and layers you needed to solve XOR? Why do you think simpler networks couldn't solve it?

4. In Experiment 4, which single fix made the biggest difference? Why?

5. How can you tell the difference between overfitting and underfitting just by looking at the training loss and test loss?

6. Pick one glossary term that confused you at first but makes sense now after the experiments. Explain it in your own words.

7. If you were building a neural network to tell cats from dogs in photos, what settings would you start with and why? (Think about layers, neurons, learning rate, and activation function.)

---

## Key Takeaways

After today's experiments, remember these big ideas:

- **Neural networks learn patterns** by adjusting weights over many epochs — not by being told the rules.
- **Simple problems need simple networks.** Two clusters? One layer is enough. Spiral? You need more power.
- **Balance is everything.** Too few neurons → underfitting. Too many → overfitting. Learning rate too high → chaos. Too low → takes forever.
- **Features matter.** Giving the model better inputs (like X1² or sin(X1)) can be more powerful than adding more layers.
- **Always watch both losses.** If training loss is low but test loss is high, your model memorized instead of learned.

---

*Neural Networks Playground – Student Guide for Python ML Class*
*Playground link: [https://playground.tensorflow.org](https://playground.tensorflow.org)*
