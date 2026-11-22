# sampleJokesGPT: a Part IV Transformer chatbot

## Purpose

This repository is a runnable, joke-focused companion to Part IV, “The Transformer Revolution,” of the [Frontier Models guide](https://frontiermodels.guide). It takes a text file of jokes, prepares short chat examples, and trains a decoder-only Transformer from scratch. The result is an interactive chatbot that generates jokes from patterns learned in that dataset.

The model is trained with self-supervised next-token prediction: the serialized conversation already contains the next token used as the target, so no separately labeled answer is required. Its implementation makes the Part IV path inspectable: token and position embeddings; Q/K/V projections; scaled causal attention; multiple heads; residual connections and normalization; a position-wise feed-forward network; shifted-token cross-entropy; parameter updates; and autoregressive generation. The chat program adds intent routing and ranks several generated candidates with simple quality heuristics.

This is a small teaching chatbot, not a general-purpose or pretrained language model. Its answers depend on the jokes used for training. The Transformer is written explicitly in `src/model.py`; PyTorch supplies tensor operations, automatic differentiation, and optimization. The tokenizer uses byte-level BPE from Hugging Face `tokenizers`.

## Install

Python 3.11 or later is recommended. From the repository root, create a virtual environment and install dependencies.

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:
```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Windows and Linux, `requirements.txt` installs the CUDA-enabled PyTorch build. macOS installs the standard PyTorch build. The CUDA PyTorch package includes the required CUDA runtime; a compatible NVIDIA driver is needed, but the separate CUDA Toolkit is not. On a machine without an available NVIDIA CUDA device, `--device auto` selects CPU.

## Repository layout

```text
sampleJokesGPT/
├── data/
│   ├── raw_data.txt            # source jokes, one per line
│   ├── clean_data.txt          # cleaned serialized chat samples
├── models/                     # one folder per training run
├── src/
│   ├── clean_data.py           # normalize, filter, and serialize jokes
│   ├── tokenizer.py            # byte-level BPE tokenizer teaching example
│   ├── model.py                # decoder-only Transformer and generation
│   ├── train.py                # self-supervised training and checkpointing
│   ├── chat.py                 # interactive joke generation and ranking
│   └── topics.py               # shared topic prompts and keyword rules
├── requirements.txt
└── README.md
```

## Prepare the dataset (optional)

`data/raw_data.txt` contains one joke per line. The cleaner normalizes and filters the jokes, removes duplicates, then writes one general training example per joke. It also adds topic-specific examples when whole-word keywords match programming, teachers, or animals. These keyword labels are automatic estimates; they can miss jokes that imply a topic without naming it, or tag a joke that only mentions the topic.

Regenerate the cleaned data before training to use topic-specific requests. Review the reported per-topic counts and inspect examples if you change the corpus or keyword lists. The cleaner writes multiple samples for a joke when it matches more than one topic.

macOS/Linux:
```bash
python -m src.clean_data --input data/raw_data.txt --output data/clean_data.txt
```

Windows PowerShell:
```pwsh
python -m src.clean_data --input .\data\raw_data.txt --output .\data\clean_data.txt
```

A prepared example has explicit boundary markers:

```text
<|bos|>
<|user|>Tell me a joke.
<|assistant|>
Why did the chicken cross the road? To get to the other side.
<|eos|>
```

For example, a keyword-matched teacher joke also gets a sample with this prompt:

```text
<|user|>Tell me a joke about teachers.
```

These markers are ordinary vocabulary items for this model. During training, the model predicts the next token throughout the entire serialized text, including both the request and the joke.

## Train

Train with the CPU or CUDA defaults selected automatically. If `models/jokes_demo` already exists, the program asks before replacing it.

macOS/Linux:
```bash
python -m src.train --data data/clean_data.txt --out_dir models/jokes_demo --device auto
```

Windows PowerShell:
```pwsh
python -m src.train --data .\data\clean_data.txt --out_dir .\models\jokes_demo --device auto
```

Choose a device explicitly when needed:

macOS/Linux:
```bash
python -m src.train --data data/clean_data.txt --out_dir models/jokes_cuda --device cuda
```

Windows PowerShell:
```pwsh
python -m src.train --data .\data\clean_data.txt --out_dir .\models\jokes_cuda --device cuda
```

`--device cuda` reports an error if CUDA is unavailable. Use `--device cpu` to force CPU execution. The automatically selected device also controls the default model size: CPU uses a smaller configuration; CUDA uses a wider, deeper configuration. Pass individual training options to override those presets.

The training loop reports train and validation loss and saves the best validation checkpoint. Training time varies with dataset size, model settings, processor, GPU, PyTorch build, and system load. For a quicker demonstration, lower `--max_iters`, `--eval_iters`, and `--n_embd`.

### Model output folder

Each run creates a folder under `models/` containing:

```text
models/<run-name>/
├── model.pt         # learned model parameters
├── tokenizer.json   # the vocabulary and BPE merges used by the model
├── config.json      # architecture, training, data, and runtime metadata
└── data_clean.txt   # copy of the serialized training text
```

Keep `model.pt`, `tokenizer.json`, and `config.json` together when moving a trained model. The copied dataset is useful for provenance but is not required to chat.

## Chat

Start an interactive session with the trained model:

macOS/Linux:
```bash
python -m src.chat --model_dir models/jokes_demo --device auto
```

Windows PowerShell:
```pwsh
python -m src.chat --model_dir .\models\jokes_demo --device auto
```

The chatbot routes requests mentioning programming, teachers, or animals to the corresponding prompt learned during training. Other requests use the general joke prompt. It generates multiple candidates, filters malformed or repetitive text, then displays the highest-scoring candidate. Type `exit`, `quit`, or `q` to leave. The candidate scoring is a small heuristic, not a measure of whether a joke is actually funny. The `You:` and `Joke bot:` labels are colored; the query and answer use the terminal's normal text color.

Illustrative interaction (sample wording; actual generated answers vary):

```text
You: Tell me a joke about programmers.
Joke bot: Why do programmers prefer dark mode? Because light attracts bugs.

You: Tell me a joke about teachers.
Joke bot: Why did the teacher wear sunglasses? Because her students were so bright.

You: Tell me a joke about animals.
Joke bot: What do you call a bear with no teeth? A gummy bear.
```

To get these topic prompts in the model, regenerate the corpus and train a new checkpoint:

```pwsh
python -m src.clean_data --input .\data\raw_data.txt --output .\data\clean_data.txt
python -m src.train --data .\data\clean_data.txt --out_dir .\models\jokes_topics --device auto
python -m src.chat --model_dir .\models\jokes_topics --device auto
```

The existing checkpoint does not gain topic conditioning from changing the chat code alone; the new prompt examples must be included in training.

Sampling controls can be changed at launch:

macOS/Linux:
```bash
python -m src.chat --model_dir models/jokes_demo --attempts 20 --max_new_tokens 100 --temperature 0.9 --top_k 50
```

Windows PowerShell:
```pwsh
python -m src.chat --model_dir .\models\jokes_demo --attempts 20 --max_new_tokens 100 --temperature 0.9 --top_k 50
```

## What this tutorial demonstrates

| Part IV idea | Where to see it |
|---|---|
| Chat text becomes token IDs | `src/tokenizer.py` |
| Shifted input and target windows | `src/train.py:get_batch` |
| Token and position embeddings | `src/model.py:GPT` |
| Q/K/V projections and scaled dot products | `src/model.py:CausalSelfAttention` |
| Causal masking before softmax | `src/model.py:CausalSelfAttention.forward` |
| Attention heads and output projection | `src/model.py:CausalSelfAttention.forward` |
| Position-wise feed-forward network | `src/model.py:MLP` |
| Residual connections and normalization | `src/model.py:Block` |
| Stacked Transformer blocks and logits | `src/model.py:GPT.forward` |
| Next-token cross-entropy | `src/model.py:GPT.forward` |
| Backpropagation and parameter updates | `src/train.py:main` |
| Autoregressive generation | `src/model.py:GPT.generate` |
| Chat formatting, candidate filtering, and ranking | `src/chat.py` |

The complete Transformer computation is visible in the Python source. The legacy `src/tokenizer_bpe_ownlogic.py` is not used by the chatbot; it remains as a separate example of how byte-pair merges can be implemented directly.

## Training arguments

Pass these options to `python -m src.train`. Defaults shown as “device preset” depend on whether `--device` resolves to CPU or CUDA.

| Argument | Default | Meaning |
|---|---:|---|
| `--data` | `data/clean_data.txt` | Serialized joke training text. |
| `--out_dir` | `models/jokes_demo` | Output folder; existing folders require overwrite confirmation. |
| `--device` | `auto` | CUDA if available, otherwise CPU; also accepts `cpu` or `cuda`. |
| `--bpe_vocab_size` | `2000` | Target maximum vocabulary size for the byte-level BPE tokenizer. |
| `--bpe_min_pair_freq` | `2` | Minimum occurrence count for a tokenizer merge. |
| `--bpe_train_chars` | `3000000` | Maximum initial text characters used to train the tokenizer; `0` uses the full file. |
| `--block_size` | `128` | Maximum preceding-token context used for each prediction. |
| `--batch_size` | CPU `32`; CUDA `64` | Number of token windows in one optimizer update. |
| `--max_iters` | CPU `5000`; CUDA `10000` | Number of training updates. |
| `--eval_interval` | CPU `100`; CUDA `500` | Updates between loss evaluations and best-checkpoint saves. |
| `--eval_iters` | `100` | Batches used to estimate each split's loss. |
| `--learning_rate` | `0.0003` | AdamW optimizer step size. |
| `--weight_decay` | `0.1` | AdamW weight-decay coefficient. |
| `--grad_clip` | `1.0` | Maximum gradient norm; set to `0` to disable clipping. |
| `--n_layer` | CPU `4`; CUDA `6` | Number of Transformer blocks. |
| `--n_head` | CPU `4`; CUDA `6` | Attention heads per block; `n_embd` must be divisible by this. |
| `--n_embd` | CPU `256`; CUDA `384` | Width of token and residual representations. |
| `--dropout` | `0.2` | Dropout probability during training. |
| `--seed` | `1337` | Random seed for repeatable sampling and initialization. |

For example, train a smaller model with fewer updates:

macOS/Linux:
```bash
python -m src.train --data data/clean_data.txt --out_dir models/jokes_small --device cpu --max_iters 1000 --eval_iters 20 --n_layer 2 --n_head 4 --n_embd 128
```

Windows PowerShell:
```pwsh
python -m src.train --data .\data\clean_data.txt --out_dir .\models\jokes_small --device cpu --max_iters 1000 --eval_iters 20 --n_layer 2 --n_head 4 --n_embd 128
```

Use the `--help` option with the training or chat command to list all available options.

## Acknowledgements

The repository's teaching style is inspired by Andrew Karpathy's educational GPT implementations, especially [minGPT](https://github.com/karpathy/minGPT), [nanoGPT](https://github.com/karpathy/nanoGPT), and [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html). This implementation is organized around the concepts and chapter sequence of Part IV of the [Frontier Models guide](https://frontiermodels.guide).
