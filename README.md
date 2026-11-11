# CircuitMind
*Engineering decisions, grounded in technical evidence.*

**Organization:** iCog Labs, Ethiopia  
**Role:** 5th-year Electrical & Computer Engineering Student & AI Engineering Intern  
**Core Technologies:** Python, Streamlit, PyMuPDF, ChromaDB, Docker Redis, Google Gemini (`gemini-flash-lite-latest`), and Offline Fallback Engine  
**Technical Reading Deliverable:** [`CACHING_SUMMARY.md`](CACHING_SUMMARY.md) (or [`docs/caching_reading_assignment.md`](docs/caching_reading_assignment.md))

---

## 1. Product Identity & Purpose

**CircuitMind** is an engineering design intelligence workspace crafted specifically for Electrical and Computer Engineering (ECE) workflows. Whether analyzing microcontroller datasheets, investigating communication protocol timings, selecting power regulation stages, or reviewing hardware specifications, CircuitMind grounds every technical claim in verifiable engineering evidence.

### Supported Technical Domains
* **Embedded Systems & Microcontrollers:** ESP32, STM32, ARM Cortex, peripheral registers, pinouts, clock configurations.
* **Communication Protocols:** UART, SPI, I2C, CAN bus, Bluetooth Low Energy (BLE), Wi-Fi.
* **Electronics & Hardware Design:** Power regulation (LDO vs switching buck/boost), passives, signal conditioning, sensor interfacing.
* **Control Systems & Robotics:** Feedback loops, PID tuning, motor drivers, sensor fusion.
* **Telecommunications & Signal Processing:** Modulation, filtering, FFT, wireless channel models.
* **Thesis & Research Engineering:** BLE offline-payment architecture, cryptographic hardware engines, embedded IoT security.

---

## 2. System Architecture & Caching Strategy

CircuitMind employs a **Unified Single-Page Architecture** with a dual-pane layout: a comprehensive control sidebar on the left, and a chat conversation feed on the right.

```text
+--------------------------------------------------------------------------------------------------+
|                                           CIRCUITMIND                                            |
|                  ✦ EMBEDDED HARDWARE INTELLIGENCE · v2.0                                         |
|  [● Gemini Flash Lite]      [● ChromaDB: Ready]      [● Redis: Port 6379]      [● Knowledge Base]|
+--------------------------------------------------------------------------------------------------+
|                                    |                                                             |
|  LEFT SIDEBAR                      |  MAIN CONVERSATION FEED                                     |
|                                    |                                                             |
|  1. Drop or Upload File            |  [ What engineering problem are you solving?              ] |
|     • Active Corpus Status         |  [ High-contrast native query input + Ask CircuitMind →   ] |
|     • File drag & drop uploader    |                                                             |
|     • Passage / chunk counter      |                            User Question (Right-Aligned)    |
|                                    |                            +------------------------------+ |
|  2. Session Inquiries              |                            | You                          | |
|     • Past inquiry pills           |                            | "What is UART?"              | |
|     • Latency tags (0.4ms / 1.5s)  |                            +------------------------------+ |
|     • Cache HIT (⚡) vs MISS (🔍)   |                                                             |
|                                    |  Assistant Answer (Left-Aligned, Cyan Accent Border)        |
|  3. System Insights & Controls     |  +--------------------------------------------------------+ |
|     • Cache Hit Rate & Counters    |  | ⚡ CIRCUITMIND GROUNDED ANALYSIS                        | |
|     • RAG & Vector Engine Metrics  |  | Grounded engineering answer body with [Source X]       | |
|     • Query execution breakdown    |  | in-text citations.                                     | |
|     • TTL & Top-K sliders          |  |                                                        | |
|     • Flush Redis Cache button     |  | ▼ Retrieved Context Passages (3 Chunks Retrieved)      | |
|                                    |  |   Source 1: esp32_hardware_reference.md (Page 1)       | |
|                                    |  |   Source 2: esp32_hardware_reference.md (Page 2)       | |
|                                    |  +--------------------------------------------------------+ |
+--------------------------------------------------------------------------------------------------+
```

### RAG & Cache Flow (Cache-Aside + Write-Around):
1. **User Submits Question**: Query is normalized (lowercased, trimmed, punctuation stripped) and hashed into:
   `rag:cache:{doc_version_hash[:12]}:{model}:{prompt_version}:{query_hash[:24]}`
2. **Check Redis Before Retrieval**:
   - **Cache HIT**: Retrieves answer + source references atomically in **$< 1\text{ ms}$** (`0.4 ms` typical). ChromaDB search and Gemini LLM generation are completely bypassed.
   - **Cache MISS**: Retrieves Top-$K$ passages from ChromaDB vector store, synthesizes grounded answer via Gemini Flash Lite, and writes result to Redis via `SETEX` with TTL.
3. **Graceful Degradation (Design A — Bypass Mode)**:
   - If Redis is unavailable (`docker stop redis-rag`), the top status ribbon honestly reports:
     `Redis: Unavailable — Cache bypassed` (amber indicator).
   - Queries seamlessly route directly through ChromaDB and Gemini without crashing or faking an in-memory store.
   - When Redis restarts (`docker start redis-rag`), CircuitMind auto-reconnects and resumes caching instantly.

---

## 3. Key Technical Decisions & Invariants

1. **Pre-Retrieval Redis Caching (Cache-Aside):**  
   The Redis lookup happens **before** vector database querying and **before** calling the LLM. Cache hits bypass retrieval and inference entirely, eliminating network latency and API costs.
2. **Atomic Answer + Source Preservation:**  
   Every cached entry stores the generated answer *together* with its verifiable source passages (filename, page number, chunk ID, and excerpt). A cache hit produces the exact same citations as the original generation.
3. **Cryptographic Document-Version Invalidation:**  
   The cache key incorporates a cryptographic SHA-256 hash of all currently indexed files (`doc_version`). Re-indexing or modifying documents changes this hash instantly, preventing the serving of stale engineering data.
4. **Resilient Failure Mode (Design A — Bypass Mode):**  
   If the Redis container stops or becomes unreachable, the client catches connection errors, logs a diagnostic warning, and routes requests directly to ChromaDB and Gemini without crashing.
5. **Secure Gemini Integration:**  
   The Google Gemini API communicates with `gemini-flash-lite-latest` using secure header authentication (`x-goog-api-key`). API keys are never embedded in URL query strings and all error messages are sanitized against token leakage.
6. **Built-in Offline Extractive Fallback:**  
   When working in isolated or offline environments without an API key, CircuitMind features a deterministic keyword-overlap extractive generator that extracts and cites authentic document sentences.

---

## 4. Directory Structure

```text
icog_rag_redis/
├── README.md              # Project documentation, architecture & defense guide
├── LEARNING_NOTES.md      # Comprehensive notes on Transformers, LLMs, and RAG
├── PROGRESS.md            # Daily milestone tracker across internship sessions
├── ARCHITECTURE.md        # Architectural flow, contracts, and failure modes
├── CACHING_SUMMARY.md     # 1-page deliverable on caching strategies & eviction
├── requirements.txt       # Project dependencies
├── data/
│   ├── chroma_db/                  # Persistent ChromaDB vector database storage
│   ├── esp32_hardware_reference.md # Official ECE Pinout, Register & Protocol Reference
│   └── sample_transformers.txt     # Benchmark text corpus
├── tests/
│   ├── test_hardware_extractor.py  # Pinout & register map extraction tests
│   ├── test_comparison.py          # Component & protocol comparison matrix tests
│   ├── test_pipeline.py            # 9-point verification suite (Cache-Aside, TTL, Invalidation)
│   └── test_synthesis.py           # Synthesis and grounded citation tests
├── scripts/
│   └── test_gemini.py     # Isolated Gemini Flash API diagnostic script
└── src/
    ├── __init__.py
    ├── app.py             # CircuitMind unified single-page Streamlit application
    ├── config.py          # Centralized configuration & environment loader
    ├── styles.py          # Minimal academic engineering CSS (Newsreader + DM Sans)
    ├── extractor.py       # PyMuPDF PDF, TXT, and Markdown extractor
    ├── chunker.py         # Recursive character text chunking with overlap
    ├── vector_store.py    # ChromaDB persistent vector store & similarity search
    ├── cache.py           # Redis Cache-Aside client with fallback engine
    └── llm.py             # Grounded prompt builder & Gemini/Offline generator
```

---

## 5. Technology Stack

| Layer | Component | Specification |
| :--- | :--- | :--- |
| **Web UI Framework** | Streamlit (v1.30+) | Responsive dark navy blueprint interface with chat feed |
| **Caching Engine** | Redis (Docker / Local) | Sub-millisecond Cache-Aside layer on port `6379` |
| **Vector Database** | ChromaDB (Persistent) | Cosine similarity semantic search over document chunks |
| **Embeddings** | `all-MiniLM-L6-v2` | Sentence-transformers dense embeddings |
| **LLM Inference** | Google Gemini API | `gemini-flash-lite-latest` (1,500 free requests/day) |
| **Document Processing** | PyMuPDF (`fitz`) | Text and PDF parser with recursive character chunker |
| **Testing Suite** | Python `unittest` | 20 unit tests covering cache, hashing, and retrieval |

---

## 6. Quickstart & Setup Instructions

Follow these steps to clone, configure, and run CircuitMind locally:

### Prerequisites
* **Python 3.10+**
* **Docker Desktop** (or a local Redis service)
* **Google Gemini API Key** (Free tier from [Google AI Studio](https://aistudio.google.com/))

### Step 1: Clone Repository & Create Virtual Environment
```bash
# Clone the repository
git clone <your-github-repo-url>
cd icog_rag_redis

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Step 2: Start Redis via Docker
```bash
# Start Redis container on port 6379
docker run -d --name redis-rag -p 6379:6379 redis:alpine

# Verify Redis is responding
docker exec -it redis-rag redis-cli ping
# Expected output: PONG
```

### Step 3: Configure Environment Variables
Copy the template configuration and insert your Gemini API key:
```bash
# Windows:
copy .env.example .env
# macOS / Linux:
cp .env.example .env
```
Edit `.env` and set your key:
```env
REDIS_URL=redis://localhost:6379/0
GEMINI_API_KEY=your_actual_gemini_api_key_here
LLM_PROVIDER=gemini
LLM_MODEL=gemini-flash-lite-latest
RAG_CACHE_TTL=3600
RAG_TOP_K=3
```

### Step 4: Run the Automated Verification Suite
Verify that all 20 tests pass:
```bash
python -m unittest discover tests
```
*Expected output:* `Ran 20 tests in ~2.8s ... OK`

### Step 5: Launch CircuitMind
```bash
streamlit run src/app.py
```
Open your browser at `http://localhost:8501`.

---

## 7. License & Attribution
Developed at **iCog Labs, Ethiopia** for the AI Engineering Internship Program.  
Open-source under the MIT License.
