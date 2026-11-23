# jokesGPT — a Transformer joke chatbot

A runnable companion to **Part IV (Transformers)** of the [Frontier Models guide](https://frontiermodels.guide). Train a small **decoder-only Transformer from scratch** to answer requests with complete, topic-relevant jokes.

The goal is a **working educational chatbot**, not a production-grade humor model. Inference uses only learned model weights, a tokenizer, and configuration: **no pretrained language model, joke retrieval, RAG, or external API**.

## Two ways to learn the Transformer: sampleGPT and jokesGPT

Both [sampleGPT](https://github.com/phtuananh-pro/sampleGPT) and this repository illustrate the **same Part IV Transformer principles**: tokenization, embeddings, masked self-attention, feed-forward blocks, next-token prediction, backpropagation, and autoregressive generation. They demonstrate different applications.

| | sampleGPT | jokesGPT |
|---|---|---|
| Teaching goal | Learn how a decoder-only Transformer predicts text | Learn how the same Transformer becomes a task-specific chatbot |
| Dataset | Continuous text | Curated, labeled jokes in JSONL |
| Training samples | Arbitrary token windows | One complete request/answer conversation at a time |
| Objective | Predict every next text token | Predict assistant completion only; weight joke identity |
| Conditioning | Continue a text prefix | Condition on requested joke style or subject |
| Generation | Open-ended text continuation | Complete joke, stopping at end-of-answer token |
| Evaluation | Text prediction loss | Loss plus generated-joke completeness and topic checks |

**The Transformer architecture stays the same.** Data representation, training targets, and decoding constraints produce the chatbot behavior.

## Repository layout

```text
jokesGPT/
├── data/
│   ├── jokes_clean.jsonl  # pre-curated jokes, one JSON object per line
│   └── topics.json        # subject names and keyword patterns
├── src/
│   ├── model.py           # decoder-only Transformer, attention and feed-forward blocks
│   ├── tokenizer.py       # byte-level BPE and special joke-ID tokens
│   ├── train.py           # conversation examples, masked loss and training loop
│   ├── chat.py            # topic-conditioned autoregressive joke generation
│   ├── topics.py          # keyword routing to training-compatible prompts
│   └── evaluate.py        # fixed prompts and simple generation diagnostics
├── tests/                 # focused regression tests
├── requirements.txt
└── README.md
```

Training creates a local `models/` folder, ignored by Git.

## Install

Python **3.11+** is recommended.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`--device auto` uses a supported NVIDIA CUDA GPU when available and otherwise uses CPU. CUDA requires a compatible driver; a separate CUDA Toolkit installation is not normally required by the PyTorch wheel.

## Dataset

`data/jokes_clean.jsonl` contains **1,440 pre-curated jokes** derived from [Humor ShareGPT 5K](https://huggingface.co/datasets/briancconnelly/humor-sharegpt-5k) (upstream dataset published under MIT terms).

Each JSONL record contains:

```json
{"prompt":"One liner","joke":"I tried to be a radiologist, but the job made me glow.","style":"one_liner","topics":["jobs_careers"]}
```

- `prompt`: original source prompt (provenance; not the user's live request).
- `joke`: complete expected answer.
- `style`: humor format, such as pun, one-liner or dad joke.
- `topics`: zero or more proposed **subject** labels, such as programming or food.

The dataset is already deduplicated and automatically screened. **It has not received exhaustive human review; topic labels and safety screening may be imperfect.** There is **no cleaning script to run**.

## Train

```powershell
python -m src.train --device auto
```

The script trains the BPE tokenizer, constructs individual joke conversations, trains the Transformer, and saves the best checkpoint. It asks before replacing an existing output folder. Use `--overwrite` to replace it without prompting.

For a quick **code smoke test** (not a useful trained chatbot):

```powershell
python -m src.train --device cpu --max_iters 3 --eval_interval 1 --eval_iters 1 --n_layer 2 --n_embd 128 --batch_size 2 --out_dir models/smoke
```

### Model output folder

```text
models/jokes/
├── model.pt        # learned Transformer weights
├── tokenizer.json  # byte-level BPE merges and special tokens
└── config.json     # architecture, training, dataset, topic-routing and checkpoint metadata
```

Only these three files are required for chat. The curated jokes file is **not loaded during inference**.

### Estimation training time

**Illustrative ranges, not measured benchmarks**, using the current device-specific defaults. CPU runs use a smaller model (3 layers, width 192, batch 8, 3,000 updates); CUDA runs use a larger model (5 layers, width 320, batch 24, 12,000 updates). Both use a 256-token context, so the workloads are **not directly comparable**.

| Hardware | Example | Rough duration |
|---|---|---|
| Older CPU | Intel i7-5820K | 4–18 hours |
| High-end desktop CPU | Intel i7-12700K / Ryzen 9 5900X | 1–6 hours |
| Entry-level NVIDIA GPU | GTX 1660 / RTX 3050 | 1–5 hours |
| Midrange NVIDIA GPU | RTX 3060 / RTX 4060 | 20 minutes–2 hours |
| High-end NVIDIA GPU | RTX 4090 / RTX 5090 | 10–60 minutes |

Use the training console's elapsed-time and ETA measurements for your own machine; hardware, PyTorch build, thermals and settings affect performance.

## Chat

```powershell
python -m src.chat --device auto
```

Illustrative examples (actual answers vary):

```text
You: tell me a joke
Joke bot: What's a hawk's favorite subject? Fawk-abulary!

You: jokes on teacher
Joke bot: Why did the music teacher go to the police? Because of the treble!

You: joke on programming
Joke bot: Why are spiders so good at computer programming? Because they are great at web design.

You: joke on food
Joke bot: What do you call a confused mushroom? A mixed-up fungi!
```

Type `exit`, `quit`, or `q` to finish.

## Evaluate and test

```powershell
python -m unittest discover -s tests -v
python -m src.evaluate --device auto
```

`src.evaluate` generates jokes for 20 fixed prompts and writes `models/jokes/evaluation.json`. It reports general, topic, fallback, and incomplete answers. **These are lightweight diagnostics, not a human evaluation of humor quality.**

## How training works

### 1. Create a conversation from each joke

The trainer reads the curated JSONL and creates different **general, style-specific, and supported topic-specific requests** for each complete joke. Every request teaches the *same* joke answer, rather than mixing together unrelated fragments:

```text
<|bos|><|user|>Tell me a joke about animals and pets.<|assistant|><|j0123|>A complete joke...<|eos|>
```

`<|j0123|>` represents a **dedicated learned joke-identity token**; it is not a key used to retrieve a record from storage. The tokenizer and neural weights learn the association between the prompt, that identity, and the full joke text.

All variants of a joke remain on the same side of the train/validation split. Examples too long for the context window are skipped, not truncated; shorter ones are padded.

### 2. Training objective: conditional next-token prediction

The decoder-only Transformer still predicts the **next token using only preceding tokens**. The difference from sampleGPT is the loss mask: the user prompt provides context, but contributes no target loss.

| Segment | Used as context | Contributes to training loss |
|---|---|---|
| User request | Yes | No |
| Joke-identity token | Yes | Yes, extra weight |
| Joke body | Yes | Yes |
| `<|eos|>` | Yes | Yes |
| Padding | No useful content | No |

The weighted next-token cross-entropy is:

\[
\mathcal{L}=\frac{\sum_{t\in A}w_t[-\log P_{\theta}(x_t\mid x_{<t})]}{\sum_{t\in A}w_t}
\]

Here `A` contains assistant-target positions. By default, the joke-identity token receives five times the loss weight of an ordinary response token (`--id_loss_weight 5`). This encourages the model to learn **which complete joke fits a request**.

### 3. Train and select a checkpoint

The loop samples padded conversation batches, computes the masked loss, backpropagates through the Transformer, and updates parameters with AdamW. It measures loss periodically and saves the checkpoint with the **lowest training completion loss**, favoring recall of learned whole jokes. Held-out validation loss is reported separately, measuring generalization rather than punchline quality.

### 4. Generate one complete answer

Inference maps a request to a training-compatible prompt, samples a learned joke-identity token, then generates text token by token until `<|eos|>` or a safety limit. Incomplete or malformed candidates are rejected. No joke texts, JSONL records, or external index are consulted.

## Topic matching

`src/topics.py` maps keywords from the user's request to one of the supported subject labels from `data/topics.json`. The same canonical prompt is used during training and inference; topics with too few labeled examples are not enabled as dedicated prompts.

A request such as `joke on programmer` can route to `programming`. Requests for a `pun` or `dad joke` use style prompts. The generator prefers completed answers with topical keyword evidence; if none matches well, it may explicitly offer a general-joke fallback.

This is deliberately **lexical routing**, not semantic understanding or retrieval. Unknown synonyms, ambiguous words, limited coverage and imperfect labels can cause off-topic answers.

## Limits

This is a small model trained from scratch on a small corpus. It can memorize, mix punchlines, produce imperfect or off-topic jokes, and occasionally fail to finish. EOS and lexical topic checks cannot guarantee good humor or safe content. The educational value is understanding how a basic Transformer can be trained as a **useful but imperfect chatbot**.

## Acknowledgements

The teaching approach draws on Andrew Karpathy's [minGPT](https://github.com/karpathy/minGPT), [nanoGPT](https://github.com/karpathy/nanoGPT), and [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), as well as [Part IV of the Frontier Models guide](https://frontiermodels.guide).

## Training arguments

`python -m src.train --help` lists all options.

| Argument | Default | Purpose |
|---|---|---|
| `--data` | `data/jokes_clean.jsonl` | Curated JSONL input |
| `--topics` | `data/topics.json` | Topic keyword taxonomy |
| `--out_dir` | `models/jokes` | Output checkpoint directory |
| `--device` | `auto` | Prefer CUDA, otherwise CPU |
| `--max_iters` | CPU 3,000; CUDA 12,000 | Optimization updates |
| `--batch_size` | CPU 8; CUDA 24 | Conversations per batch |
| `--block_size` | 256 | Maximum sequence length |
| `--n_layer` | CPU 3; CUDA 5 | Transformer blocks |
| `--n_head` | 4 | Attention heads |
| `--n_embd` | CPU 192; CUDA 320 | Hidden width |
| `--dropout` | 0.1 | Dropout probability |
| `--bpe_vocab_size` | 4,500 | Target BPE vocabulary |
| `--learning_rate` | 0.0003 | AdamW learning rate |
| `--weight_decay` | 0.05 | AdamW weight decay |
| `--id_loss_weight` | 5.0 | Extra joke-ID target weight |
| `--min_topic_examples` | 10 | Minimum examples to enable a topic |
| `--val_fraction` | 0.05 | Fraction of held-out jokes |
| `--eval_interval` | 200 | Steps between loss evaluations |
| `--eval_iters` | 8 | Batches per evaluation |
| `--seed` | 1337 | Random seed |
| `--overwrite` | off | Replace an existing model without confirmation |

## Chat arguments

`python -m src.chat --help` lists all options.

| Argument | Default | Purpose |
|---|---|---|
| `--model_dir` | `models/jokes` | Folder containing checkpoint, tokenizer and configuration |
| `--device` | `auto` | Prefer CUDA, otherwise CPU |
| `--attempts` | 6 | Maximum generated candidates per user request |
| `--max_new_tokens` | 240 | Per-candidate maximum generated body tokens |
| `--id_temperature` | 0.75 | Sampling randomness when choosing a joke identity |
| `--temperature` | 0.0 | Body sampling; 0 uses greedy decoding |
| `--top_k` | 20 | Limit token sampling to highest-probability options |
| `--verbose` | off | Print detected topic, status and response latency |
