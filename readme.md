# DERMARK: A Dynamic, Efficient and Robust Multi-bit Watermark for Large Language Models

This repository provides a robust multi-bit watermarking system for language models using green-list based token manipulation. It supports watermark embedding, detection, and robustness evaluation via dynamic programming.

## 🔧 Features

- **Bit Partitioning**: Dynamically distribute bits across token positions.
- **Watermark Embedding**: Embed bits into token sequences during generation (`generate` function).
- **Robustness Detection**: Extract watermark bits even under post-generation edits (`detect_robustness` function).
- **Segment Recovery**: Use dynamic programming to identify watermark-carrying segments.

## 📁 File Structure

```bash
.
├── watermarking.py          # Main logic for watermark embedding and detection
├── bit_distributing.py      # Partitioning scheme for embedding bits
├── robustness.py            # Robust bit extraction via dynamic programming
└── README.md                # This file
```

------

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- NumPy

Install dependencies:

```bash
pip install torch numpy
```

## 📍 Model Path Configuration

Before using the toolkit, make sure to specify the correct path to your deployed language model:

```python
model_path = '../Llama-2-7b'
```

This variable should point to the directory where your model (e.g., LLaMA-2 or other Hugging Face-compatible model) is located.

- If the model is already downloaded and stored locally, set `model_path` to the corresponding local directory.
- If you are loading the model from Hugging Face's `transformers` library directly, you can pass the model name instead (e.g., `"meta-llama/Llama-2-7b-hf"`).

**Example:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "../Llama-2-7b"  # Change this path to your local model directory

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
```

> 🔁 *If you're using a different model, make sure to update `model_path` accordingly and ensure compatibility with causal language modeling.*

## 🚀Usage

### 1. Initialization

```python
from watermarking import generate, detect_robustness

model_path = '../Llama-2-7b'
alpha = 0.90            
max_length = 100
delta = 1
prompt = "I am going to tell a story:"
```

### 2. Text Generation with Watermark

```python
generated_tokens, N, bit_count, _ = generate(
    model, 
    vocab_size,
    input_prompt,
    mark_info,
    alpha,
    max_length=100,
    delta=delta
)
```

### 3. Watermark Detection

```python
detection_info = detect_robustness(
    model,
    input_prompt,
    generated_tokens,
    vocab_size,
    alpha,
    delta,
    N,
    bit_count
)
```

## Parameters

|  Parameter   |                         Description                          |
| :----------: | :----------------------------------------------------------: |
|   `alpha`    |      Statistical significance level (Type I error rate)      |
|   `delta`    | Watermark strength parameter (higher = more robust, lower = better quality) |
| `mark_info`  |       Secret bit pattern defining watermark signature        |
| `gls_ratio`  | Proportion of vocabulary considered "green list" (default: 0.5) |
| `max_length` |            Maximum sequence length for generation            |

## Algorithm Overview

1. **Partition-based Watermarking** (`fitpartition` class):
   - Dynamically partitions vocabulary into green/red lists
   - Maintains statistical guarantees through adaptive thresholding
   - Uses Monte Carlo estimation for robustness
2. **Robust Detection**:
   - Dynamic programming segmentation for optimal watermark recovery
   - Color-aware cost minimization for tamper resistance
   - Adaptive weighting between statistical validity and content preservation

## Example Output

```
mark info [0, 0, 1, 1, 1, 0, 1, 1, 0, 0]
( 0,297,),( 1,372,),...,( 98,29887,),|||
detection result: [[0, 0, 1, 1, 0, 0, 1]]
```

In each tuple like `(0, 297)`, the first number (e.g., `0`) indicates the token's position in the sequence, and the second number (e.g., `297`) represents its vocabulary index. The color of the second number corresponds to the embedded watermark bit: **green for bit 1**, **red for bit 0**.