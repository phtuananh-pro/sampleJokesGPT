# jokesGPT — a Transformer joke chatbot

A runnable companion to **Part IV (Transformers)** of the [Frontier Models guide](https://frontiermodels.guide). Train a small **decoder-only Transformer from scratch** to answer requests with complete, topic-relevant jokes.

The goal is a **working educational chatbot**, not a production-grade humor model. Supports **English (`en`), French (`fr`), and Vietnamese (`vi`)** as independently trained models. Inference uses only learned model weights, a tokenizer, and configuration: **no pretrained language model, joke retrieval, RAG, or external API**.

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
│   ├── jokes_clean.jsonl        # 1,440 curated English jokes
│   ├── jokes_clean_fr.jsonl     # 69 draft French adaptations (pilot)
│   ├── jokes_clean_vi.jsonl     # 69 draft Vietnamese adaptations (pilot)
│   ├── topics.json              # English topic patterns
│   ├── topics_fr.json           # French labels and keyword patterns
│   └── topics_vi.json           # Vietnamese labels and keyword patterns
├── src/
│   ├── model.py                 # decoder-only Transformer (unchanged for all languages)
│   ├── tokenizer.py             # language-specific byte-level BPE training
│   ├── i18n.py                  # localized prompts and default file locations
│   ├── topics.py                # per-language topic and humor-style routing
│   ├── train.py                 # next-token training with masked completion loss
│   ├── chat.py                  # joke generation with no retrieval
│   └── evaluate.py              # generation diagnostics
├── runnable_models/
│   └── jokes_en/                # English checkpoint metadata/tokenizer
├── tests/
├── requirements.txt
└── README.md
```

Training creates separate local folders `models/jokes_en`, `models/jokes_fr`, and `models/jokes_vi`. These outputs are ignored by Git.

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

## Run an already trained model

```powershell
python -m src.chat -l en --model_dir runnable_models/jokes_en
```

For a fresh clone, the folder must contain **`model.pt`**, `tokenizer.json`, and `config.json`. **The English `model.pt` is not currently tracked in the repository**, although its configuration and tokenizer are; copy your trained weights into `runnable_models/jokes_en/model.pt` and commit them to make the example genuinely runnable without training. The updated `.gitignore` permits `runnable_models/*/model.pt`.

A French or Vietnamese model **cannot reuse English weights**. Train the chosen language separately, then run `python -m src.chat -l fr` or `-l vi`.

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

The English dataset is already deduplicated and automatically screened. **It has not received exhaustive human review; topic labels and safety screening may be imperfect.** There is **no cleaning script to run**.

### French and Vietnamese datasets

- `data/jokes_clean_fr.jsonl`: **69 AI-drafted French adaptations**, retaining `source_index` for traceability.
- `data/jokes_clean_vi.jsonl`: **69 AI-drafted Vietnamese adaptations**, retaining the same source indices.
- `data/topics_fr.json` and `topics_vi.json`: all **49 stable topic IDs**, with localized labels and keyword patterns.

**Translation status: PILOT — not yet native-speaker validated.** These are draft adapted subsets, **not full translations of the 1,440 English jokes**. Many English puns cannot be translated literally while keeping the punchline. Have native speakers review the localized jokes and expand these corpora before claiming comparable chatbot quality. The program does not silently fall back to English jokes.

## Train

Choose a language with `-l` / `--language` (default: `en`). Each language uses its own data, topic taxonomy, trained tokenizer, and checkpoint.

```powershell
python -m src.train -l en --device auto
python -m src.train -l fr --device auto
python -m src.train -l vi --device auto
```

Training reads the selected JSONL and taxonomy, builds complete conversations, learns a byte-level BPE tokenizer, and saves the best checkpoint under `models/jokes_<language>/`. An existing folder requires confirmation before overwrite. Use `--overwrite` to replace it without prompting. **French/Vietnamese are small teaching pilots**; their generated humor will be less varied than English.

For a short code **smoke test** (not a useful trained chatbot):

```powershell
python -m src.train -l fr --device cpu --max_iters 3 --eval_interval 1 --eval_iters 1 --n_layer 2 --n_embd 128 --batch_size 2 --out_dir models/smoke_fr
```

### Model output folder

```text
models/jokes_fr/
├── model.pt        # French model weights (independent of English/Vietnamese)
├── tokenizer.json  # French-trained BPE and joke-ID tokens
└── config.json     # architecture, language, topic routing, checkpoint metadata
```

These three files are sufficient for inference. The source JSONL is not loaded during chat.

### Estimation training time

**Illustrative, unbenchmarked ranges** for the current presets; hardware and corpus size affect elapsed time. CPU uses 3 layers, width 192, batch 8, 3,000 steps; CUDA uses 5 layers, width 320, batch 24, 12,000 steps. The workloads are not equivalent.

| Hardware | Example | Rough duration |
|---|---|---|
| Older CPU | Intel i7-5820K | 4–18 hours |
| High-end desktop CPU | Intel i7-12700K / Ryzen 9 5900X | 1–6 hours |
| Entry-level NVIDIA GPU | GTX 1660 / RTX 3050 | 1–5 hours |
| Midrange NVIDIA GPU | RTX 3060 / RTX 4060 | 20 minutes–2 hours |
| High-end NVIDIA GPU | RTX 4090 / RTX 5090 | 10–60 minutes |

Use your console's measured ETA instead of treating these estimates as guarantees.

## Chat

```powershell
python -m src.chat -l en --device auto
python -m src.chat -l fr --device auto
python -m src.chat -l vi --device auto
```

Chat selects the corresponding `models/jokes_<language>` checkpoint if trained locally, or a complete `runnable_models/jokes_<language>` checkpoint when available. Use `--model_dir` to select a specific model folder. The checkpoint language must match `-l`.

Sample English interaction (generated jokes vary):

```text
You: tell me a joke
Joke bot: What's a hawk's favorite subject? Fawk-abulary!

You: jokes on teacher
Joke bot: Why did the music teacher go to the police? Because of the treble!

You: joke on programming
Joke bot: Why are spiders so good at computer programming? Because they are great at web design.
```

French and Vietnamese use localized prompts and speaker labels. Type `exit`, `quit`, or `q` to leave.

## Evaluate and test

```powershell
python -m unittest discover -s tests -v
python -m src.evaluate -l en --device auto
python -m src.evaluate -l fr --device auto
python -m src.evaluate -l vi --device auto
```

Evaluation writes an `evaluation.json` in the selected model directory. It reports general, topic, fallback and incomplete responses. These are **lexical/format diagnostics, not humor-quality judgments**.

## How training works

### 1. Create a conversation from each joke

The trainer reads the selected language's curated JSONL and creates different **localized general, style-specific, and supported topic-specific requests** for each complete joke. Every request teaches the *same* joke answer, rather than mixing together unrelated fragments:

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

Inference routes the request in the **checkpoint's language** to a training-compatible prompt, samples a learned joke-identity token, then generates text token by token until `<|eos|>` or a safety limit. Incomplete or malformed candidates are rejected. No joke texts, JSONL records, or external index are consulted.

## Topic matching

`src/topics.py` maps keywords in the selected language to stable topic IDs from `data/topics.json`, `topics_fr.json`, or `topics_vi.json`. `src/i18n.py` supplies the localized prompt grammar. **Training and inference use identical prompts for that language**; topics with too few examples do not get dedicated prompts.

A request such as `joke on programmer` can route to `programming`. Requests for a `pun` or `dad joke` use style prompts. The generator prefers completed answers with topical keyword evidence; if none matches well, it may explicitly offer a general-joke fallback.

This is deliberately **lexical routing**, not semantic understanding or retrieval. Unknown synonyms, ambiguous words, limited coverage and imperfect labels can cause off-topic answers.

## Limits

This is a small model trained from scratch on a small corpus. It can memorize, mix punchlines, produce imperfect or off-topic jokes, and occasionally fail to finish. EOS and lexical topic checks cannot guarantee good humor or safe content. The educational value is understanding how a basic Transformer can be trained as a **useful but imperfect chatbot**.

## Acknowledgements

The teaching approach draws on Andrew Karpathy's [minGPT](https://github.com/karpathy/minGPT), [nanoGPT](https://github.com/karpathy/nanoGPT), and [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), as well as [Part IV of the Frontier Models guide](https://frontiermodels.guide).

## Training arguments

`python -m src.train --help` lists all options. Language-dependent defaults resolve automatically.

| Argument | Default | Purpose |
|---|---|---|
| `-l`, `--language` | `en` | Language: `en`, `fr`, `vi` (separate data and model) |
| `--data` | language-dependent | Curated JSONL input |
| `--topics` | language-dependent | Topic keyword taxonomy |
| `--out_dir` | `models/jokes_<language>` | Output checkpoint directory |
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
| `--min_topic_examples` | EN 10; FR/VI 3 | Minimum examples to enable a topic |
| `--val_fraction` | 0.05 | Fraction of held-out jokes |
| `--eval_interval` | 200 | Steps between loss evaluations |
| `--eval_iters` | 8 | Batches per evaluation |
| `--seed` | 1337 | Random seed |
| `--overwrite` | off | Replace an existing model without confirmation |

## Chat arguments

`python -m src.chat --help` lists all options.

| Argument | Default | Purpose |
|---|---|---|
| `-l`, `--language` | `en` | Language: `en`, `fr`, `vi` |
| `--model_dir` | autodetected for language | Folder containing checkpoint, tokenizer and configuration |
| `--device` | `auto` | Prefer CUDA, otherwise CPU |
| `--attempts` | 6 | Maximum generated candidates per user request |
| `--max_new_tokens` | 240 | Per-candidate maximum generated body tokens |
| `--id_temperature` | 0.75 | Sampling randomness when choosing a joke identity |
| `--temperature` | 0.0 | Body sampling; 0 uses greedy decoding |
| `--top_k` | 20 | Limit token sampling to highest-probability options |
| `--verbose` | off | Print detected topic, status and response latency |
