# Learning Notes: Transformers, LLMs, RAG, and Vector Retrieval

## 1. AI Models & Large Language Models (LLMs)

### Model Definition & Statistical Foundation
At a mathematical level, an **AI Model** is a parameterized function $f_\theta: \mathcal{X} \rightarrow \mathcal{Y}$ where $\theta \in \mathbb{R}^d$ represents internal trainable weights (parameters).

A **Large Language Model (LLM)** is a probabilistic autoregressive model that operates over a discrete vocabulary $\mathcal{V}$. Given a sequence of $t$ tokens $(w_1, w_2, \dots, w_t)$, the model computes the probability distribution of the next token $w_{t+1}$:
$$P(w_{t+1} \mid w_1, w_2, \dots, w_t) = \text{softmax}(z_{t+1})$$
where $z_{t+1}$ represents the logit vector output from the final linear projection layer.

### Training vs. Inference
* **Training (Weight Optimization):**
  - Iterative parameter updates using gradient descent: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$.
  - Loss function is typically Cross-Entropy over millions or billions of tokens.
  - Highly compute-intensive; requires distributed GPU/TPU clusters, FP16/BF16 mixed precision, and backpropagation.
* **Inference (Forward Pass):**
  - Weights $\theta$ are **frozen** ($\nabla_\theta = 0$).
  - Given an input prompt, the model performs forward passes token-by-token (decoding).
  - Primary constraints are memory bandwidth (KV-cache) and latency.

### The Decision Matrix: Prompt Engineering vs. Fine-Tuning vs. RAG

| Dimension | Prompt Engineering | Fine-Tuning | RAG (Retrieval-Augmented Generation) |
| :--- | :--- | :--- | :--- |
| **Model Weights** | Unchanged | Updated ($\Delta \theta \ne 0$) | Unchanged |
| **Knowledge Dynamic** | Static (parametric) | Fixed at fine-tune date | **Dynamic real-time** (external index) |
| **Source Attribution** | None / Hallucinated | None / Hallucinated | **Verifiable citations & chunk IDs** |
| **Hallucination Risk** | High for unlearned facts | Medium | **Low** (grounded in retrieved context) |
| **Compute Cost** | Lowest (prompt tokens) | High (GPU training cycles) | Moderate (retrieval + inference tokens) |
| **Update Latency** | Instant (change prompt) | Hours to days (re-training) | **Seconds** (re-index document) |

---

## 2. Transformer Architecture Deep Dive

Reference Papers & Visual Guides:
* *Vaswani et al. (2017)*, ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
* *Jay Alammar*, ["The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/)

```
                    +---------------------------+
                    |        Output Text        |
                    +---------------------------+
                                  ^
                                  |
                    +---------------------------+
                    |    Linear + Softmax       |
                    +---------------------------+
                                  ^
                                  |
                    +---------------------------+
                    |      Decoder Blocks       | <----+ Cross-Attention
                    +---------------------------+      |
                                  ^                    |
                                  |                    |
+---------------------+     +-----------+              |
|     Input Text      |     |  Target   |              |
+---------------------+     +-----------+              |
           |                      |                    |
           v                      v                    |
+---------------------+     +-----------+              |
| Input Embedding +   |     | Output    |              |
| Positional Encoding |     | Embedding |              |
+---------------------+     +-----------+              |
           |                                           |
           v                                           |
+---------------------+                                |
|    Encoder Blocks   | -------------------------------+
+---------------------+
```

### Encoder vs. Decoder
* **Encoder (Bidirectional):** Consists of stacked layers of Self-Attention and Feed-Forward Networks. Every token can attend to all other tokens (left and right). Used in models like BERT for embeddings and feature extraction.
* **Decoder (Autoregressive / Causal):** Uses **Masked Self-Attention** so token $i$ can only attend to tokens $j \le i$. This enforces causality: predicting the future without "looking ahead". Used in modern generative models (GPT-4, Claude, Gemini, LLaMA).
* **Cross-Attention:** In Encoder-Decoder models (like original Vaswani Transformer or T5), the Decoder takes Queries ($Q$) from its own generated tokens, but takes Keys ($K$) and Values ($V$) from the Encoder's final hidden states.

### Self-Attention: Query, Key, and Value ($Q, K, V$)
Think of Self-Attention like a database retrieval operation:
1. **Query ($Q$):** What information the current token is seeking: $Q = X W_Q$
2. **Key ($K$):** What information each token possesses: $K = X W_K$
3. **Value ($V$):** The actual substantive representation of each token: $V = X W_V$

The **Scaled Dot-Product Attention** formula:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

* Why dot product $Q K^T$? It measures the geometric similarity (projection) between what token $i$ is asking for and what token $j$ provides.
* Why scale by $\sqrt{d_k}$? For high dimensional vectors ($d_k$ e.g., 64 or 128), dot products grow large in magnitude, pushing the softmax function into regions with near-zero gradients (saturation). Scaling by $\frac{1}{\sqrt{d_k}}$ preserves unit variance.

### Multi-Head Attention
Instead of performing a single attention function, Multi-Head Attention projects $Q, K, V$ into $h$ different subspaces with learned projection matrices:
$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$
This allows the model to simultaneously attend to information from different representation subspaces (e.g., one head tracks syntactic subject-verb agreement, another tracks semantic coreference).

### Positional Encoding
Transformers lack recurrence and convolution; the attention operation is permutation-invariant:
$$\text{Attention}(P X) = P \text{Attention}(X)$$
To inject sequence order, a positional vector is added to the token embedding:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
This geometric wave encoding allows the model to learn relative positions easily because for any fixed offset $k$, $PE_{pos+k}$ is a linear function of $PE_{pos}$. Modern models also use learned or Rotary Positional Embeddings (RoPE).

---

## 3. RAG Architecture & Vector Database Mechanics

```
 [User PDF/Docs]
        |
        v
 [Text Extraction] (PyMuPDF)
        |
        v
 [Chunking & Overlap] (e.g. 500 chars, 100 overlap)
        |
        v
 [Embedding Model] (e.g. all-MiniLM-L6-v2) ---> Vector Database (ChromaDB)
                                                        |
 [User Query] ---> [Query Embedding]                    | (Cosine Similarity)
        |                     |                         |
        |                     v                         v
        |              [Top-k Nearest Neighbor Retrieval]
        |                               |
        +-------------------------------+
                        |
                        v
         [Grounded Prompt Construction]
                        |
                        v
                 [LLM Generation]
                        |
                        v
         [Answer + Source References]
```

### Chunking Mechanisms & Overlap
* **Why Chunk?** LLMs have maximum context windows, and embedding models (e.g. BERT/MiniLM) typically support at most 256 or 512 tokens. Furthermore, granular chunks isolate relevant information from irrelevant noise.
* **Chunk Size:** Too small ($\le 50$ words) loses semantic context; too large ($\ge 2000$ words) dilutes the embedding signal.
* **Overlap (e.g., 10-20%):** Essential to prevent semantic cutoff where a critical fact or sentence spans the boundary between two adjacent chunks.

### Dense vs. Sparse vs. Hybrid Retrieval
* **Sparse Retrieval (e.g., BM25, TF-IDF):**
  - Matches exact keywords, acronyms, and product IDs based on term frequency and inverse document frequency.
  - Fails on synonyms (e.g., searching "automobile" will not match "car").
* **Dense Retrieval (Vector Embeddings):**
  - Maps text into a continuous vector space $\mathbb{R}^d$ using deep neural networks (e.g., Sentence-Transformers).
  - Matches conceptual semantics and synonyms regardless of phrasing.
  - Can struggle with exact serial numbers, specific code tokens, or rare proper nouns.
* **Hybrid Search:** Combines Dense + Sparse scores (e.g., Reciprocal Rank Fusion: $RRF(d) = \sum \frac{1}{60 + r_i(d)}$) to capture both semantic meaning and precise keyword matches.

### Vector Databases & Cosine Similarity
Vector databases (like ChromaDB, Pinecone, Milvus) index high-dimensional vectors for fast Approximate Nearest Neighbor (ANN) search (e.g., using HNSW - Hierarchical Navigable Small World graphs).
The similarity between a query vector $\vec{q}$ and document vector $\vec{d}$ is computed using **Cosine Similarity**:
$$\text{Cosine}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|} = \frac{\sum_{i=1}^n q_i d_i}{\sqrt{\sum_{i=1}^n q_i^2} \sqrt{\sum_{i=1}^n d_i^2}}$$

### Reranking
Initial retrieval fetches a candidate pool (e.g., top 25 chunks via bi-encoder). A **Cross-Encoder Reranker** then evaluates each `(query, document_chunk)` pair jointly through full self-attention, reordering them to put the most relevant 3-5 chunks directly into the prompt.
