# jokesGPT — a decoder-only Transformer joke chatbot (Part IV)

**jokesGPT** is a small, runnable companion to [Part IV of the Frontier Models guide](https://frontiermodels.guide). It trains a **decoder-only Transformer from scratch** to generate complete jokes in response to requests such as "Tell me a joke about programmers." This is an educational model, not a pretrained language model or a production joke service.

The two behavior goals are **(1) a complete joke rather than an arbitrary text fragment**, and **(2) a joke relevant to a topic mentioned in the user's request**. Neither is guaranteed by training on a small dataset; the evaluation program reports failures rather than claiming perfect output. **No retrieval, RAG, external joke catalog, or pretrained model is used at inference.**

## What this tutorial demonstrates

| Part IV mechanism | Where to see it |
|---|---|
| Byte-level BPE, control tokens, and learned joke IDs | `src/tokenizer.py` |
| Topic-to-prompt conditioning from user keywords | `src/topics.py`, `src/chat.py` |
| Complete conversations, shifted next-token targets, masked prompt/padding loss | `src/train.py` |
| Token and position embeddings | `src/model.py:GPT` |
| Q/K/V projections, scaling, causal masking, multi-head attention | `src/model.py:CausalSelfAttention` |
| Residual connections, layer normalization, feed-forward network | `src/model.py:Block`, `src/model.py:MLP` |
| Cross-entropy, backpropagation, optimizer steps | `src/train.py:completion_loss`, `src/train.py:main` |
| Autoregressive decoding and EOS termination | `src/chat.py:generate_candidate` |
| Generation checks and their limitations | `src/chat.py:generate_best`, `src/evaluate.py` |

The Transformer layers are written explicitly in PyTorch. PyTorch provides tensors, automatic differentiation, and AdamW. The tokenizer is trained with Hugging Face `tokenizers`.

## Install

Python 3.11 or later is recommended. From the repository root:

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

The PyTorch package configuration in `requirements.txt` supports CUDA on compatible Windows/Linux NVIDIA systems. `--device auto` chooses CUDA when available, otherwise CPU; `--device cpu` and `--device cuda` are explicit overrides. A compatible NVIDIA driver is required for CUDA.

## Repository layout

```text
jokesGPT/
├── data/
│   ├── jokes_clean.jsonl     # 1,440 pre-curated jokes; one JSON object per line
│   └── topics.json           # topic names and lexical keyword patterns
├── src/
│   ├── model.py              # decoder-only Transformer (Part IV architecture)
│   ├── tokenizer.py          # BPE and special tokens
│   ├── topics.py             # user request -> canonical topic prompt
│   ├── train.py              # JSONL -> training sequences -> optimization
│   ├── chat.py               # neural generation without retrieving joke records
│   └── evaluate.py           # sample prompts and outcome report
├── tests/                    # focused unit tests
├── models/                   # locally generated checkpoints
├── requirements.txt
└── README.md
```

## Dataset and training objective

`data/jokes_clean.jsonl` contains **1,440** curated records from [Humor ShareGPT 5K](https://huggingface.co/datasets/briancconnelly/humor-sharegpt-5k), distributed under the upstream dataset's stated MIT terms. Each row has a source `prompt`, the complete `joke`, a humor `style`, and zero or more candidate `topics`. The dataset was deduplicated and automatically screened, but **has not undergone a complete human safety or joke-quality review**. Topic annotations are weak labels, not verified semantic ground truth.

Example schema (illustrative, not a promise of exact output):

```json
{"prompt":"One liner","joke":"I told my computer I needed a break, and it froze.","style":"one_liner","topics":["computers_it"]}
```

The JSONL is *already the cleaned input*: **there is no data-cleaning step to run**. The trainer reads each record, creates generic, style-specific, and supported subject-specific requests, and turns one complete joke into one bounded training example. Topic prompts are enabled only when a subject has at least 10 records by default (`--min_topic_examples`); sparse subjects remain in the dataset and still contribute to general-joke training.

An example of the internal conversation format is:

```text
<|bos|><|user|>Tell me a joke about programming.<|assistant|><|j0123|>A complete joke...<|eos|>
```

The `<|j0123|>` token is a learned joke identity **inside the model's vocabulary**, not a database key. The Transformer first learns to assign likelihood to an identity given the prompt, then learns to continue from that identity through a full joke and `<|eos|>`. The identity token can help it reproduce coherent learned jokes, but does not guarantee verbatim reproduction or relevance. Inference does not load the JSONL file or use the identity to look up a joke.

Training remains **autoregressive next-token prediction**. Prompt and padding targets are masked (`-100`), so cross-entropy is computed only on the assistant's identity, joke text and EOS. The first answer token receives additional weight (`--id_loss_weight`) to emphasize the association between topic and joke identity. Train/validation examples are separated by original joke; validation loss measures transfer to held-out jokes. This v2 implementation selects checkpoints by **training completion loss**, favoring recall of learned complete jokes over held-out loss; that trade-off is deliberate and should not be mistaken for generalization.

## Train

```powershell
python -m src.train --data data/jokes_clean.jsonl --topics data/topics.json --out_dir models/jokes_v2 --device auto
```

A short **smoke test** (tests execution only; it cannot produce a useful chatbot):

```powershell
python -m src.train --device cpu --max_iters 3 --eval_interval 1 --eval_iters 1 --n_layer 2 --n_embd 128 --batch_size 2 --out_dir models/jokes_smoke
```

Training creates `model.pt` (weights), `tokenizer.json` (BPE vocabulary and special tokens), and `config.json` (architecture, dataset hash, runtime settings, topic routing metadata, available learned identities, and checkpoint losses). **Only these three files are needed for inference.** Training asks before overwriting an existing model folder; `--overwrite` skips confirmation.

CPU runs are supported with smaller defaults; full training is much better suited to CUDA. Duration depends on hardware and settings; no measured training-time benchmark is claimed here.

### Main training arguments

| Argument | Default | Meaning |
|---|---:|---|
| `--data` | `data/jokes_clean.jsonl` | Curated input JSONL |
| `--topics` | `data/topics.json` | Topic keyword taxonomy |
| `--out_dir` | `models/jokes_v2` | Checkpoint destination |
| `--device` | `auto` | CUDA if available, else CPU |
| `--block_size` | `256` | Maximum causal context length in tokens |
| `--batch_size` | CPU `8`, CUDA `24` | Complete conversations per update |
| `--max_iters` | CPU `3000`, CUDA `12000` | Optimization updates |
| `--n_layer` | CPU `3`, CUDA `5` | Transformer blocks |
| `--n_head` | `4` | Attention heads per block |
| `--n_embd` | CPU `192`, CUDA `320` | Embedding and residual width |
| `--dropout` | `0.1` | Training dropout |
| `--bpe_vocab_size` | `4500` | Target BPE vocabulary size, including ID tokens |
| `--learning_rate` | `0.0003` | AdamW learning rate |
| `--weight_decay` | `0.05` | AdamW decay |
| `--id_loss_weight` | `5.0` | Relative weight of first answer (joke-ID) token |
| `--min_topic_examples` | `10` | Minimum labeled jokes for a topic prompt |
| `--val_fraction` | `0.05` | Held-out record fraction |
| `--eval_interval` | `200` | Steps between loss checks/checkpoints |
| `--eval_iters` | `8` | Sampled batches per loss estimate |
| `--seed` | `1337` | Random seed |
| `--overwrite` | off | Replace existing output without prompting |

## Chat

```powershell
python -m src.chat --model_dir models/jokes_v2 --device auto
```

Examples of requests the bot attempts to handle:

```text
You: Tell me a joke.
You: Tell me a pun.
You: Tell me a joke about programmers.
You: Tell me a joke about teachers.
You: Tell me a joke about cats.
You: Tell me a joke about food.
```

The `You:` and `Joke bot:` labels have distinct colors; the actual request/answer keeps the normal terminal color. Type `exit`, `quit`, or `q` to end the session.

For an identifiable subject, `src/topics.py` matches keywords in the request against the saved taxonomy and chooses a canonical training prompt. The model samples one **learned joke-ID token**, generates the body token by token (greedy by default), and stops at EOS. It tries up to six candidates by default; incomplete or malformed answers are rejected, and topic/keyword matches are preferred. A topic mismatch is identified as an **explicit fallback**, not silently passed off as on-topic. This router is lexical: polysemy, synonyms, and weak source labels cause mistakes. It is not retrieval of jokes.

Sampling options include `--attempts` (6), `--max_new_tokens` (240), `--id_temperature` (0.75), `--temperature` (0 for greedy joke body), and `--top_k` (20). Increasing randomness may produce novel variations but also more broken or incomplete jokes.

## Evaluate and test

Unit tests check dataset shape, topic routing, and generation mechanics:

```powershell
python -m unittest discover -s tests -v
```

After actually training a checkpoint, run:

```powershell
python -m src.evaluate --model_dir models/jokes_v2 --device auto
```

This generates responses to 20 fixed requests and writes `models/jokes_v2/evaluation.json` with `general`, `topic`, `fallback`, and `incomplete` outcomes. Check the **full joke text** manually as well: EOS and lexical keyword matches are necessary engineering diagnostics, not evidence of humor quality or truthfully interpreted topics. A trained v2 model has not yet been demonstrated or quantitatively validated in this repository.

## Limits

- **No retrieval:** the inference program loads only model parameters, tokenizer and config; jokes are generated from learned weights.
- **Small corpus:** the model can memorize, blend, corrupt, or truncate punchlines; generalization beyond curated prompts is limited.
- **Topic routing:** many distinct subjects are supported, but exact coverage depends on label counts and keyword patterns. No topic request is guaranteed to match.
- **Safety:** the corpus was machine-screened, not exhaustively reviewed; generation is not guaranteed safe.
- **Educational priority:** the design exposes the Transformer learning and decoding mechanisms, rather than building a dependable content product.

## Acknowledgements

The teaching style draws on Andrew Karpathy's [minGPT](https://github.com/karpathy/minGPT), [nanoGPT](https://github.com/karpathy/nanoGPT), and [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html). This implementation follows the conceptual mechanisms developed in [Part IV of the Frontier Models guide](https://frontiermodels.guide).
