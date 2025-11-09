# =========================
# SVAMP (eval) + GSM8K (Example-RAG corpus) with Mistral API (retry-safe)
# =========================
!pip -q install datasets pandas sentence-transformers scikit-learn rank-bm25 requests

# -------------------------
# Imports
# -------------------------
import os, re, json, time, numpy as np, pandas as pd, requests
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
from google.colab import files

# -------------------------
# Config: Mistral API (OpenAI-compatible)
# -------------------------
os.environ["OPENAI_BASE_URL"] = "https://api.mistral.ai/v1"
os.environ["OPENAI_API_KEY"]  = "hYnJKM53YmlSWDwjdiXGdTWos4VRrgHd"   # <-- put your key

MODEL_ID = "mistral-small"   # or "mistral-medium" / "mistral-large-latest"

# Few-shot / retrieval configs
TOP_N  = 20      # candidate pool before MMR
K_EX   = 3       # examples inserted into the prompt
ALPHA  = 0.6     # hybrid weight: alpha*dense + (1-alpha)*bm25
LAMB   = 0.5     # MMR relevance vs diversity
N_EVAL = 100       # how many SVAMP items to evaluate (start small)
SLEEP_BETWEEN_CALLS = 3  # seconds to sleep between API calls to avoid 429

# -------------------------
# Helpers
# -------------------------
def call_mistral_chat(model: str, prompt: str, temperature=0.2, max_retries=6):
    """
    OpenAI-compatible /v1/chat/completions with retry & exponential backoff for 429.
    """
    base = os.environ.get("OPENAI_BASE_URL", "")
    key  = os.environ.get("OPENAI_API_KEY", "")
    if not base or not key:
        return "[ERR] Missing OPENAI_BASE_URL or OPENAI_API_KEY"

    url = f"{base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "You are a careful math assistant. Provide concise reasoning and a final numeric answer."},
            {"role": "user",   "content": prompt},
        ],
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=120)
            # Handle rate limit explicitly
            if r.status_code == 429:
                wait = min(15, 2 + attempt * 3)  # capped backoff
                print(f"⚠️ Rate limit (429). Retrying in {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            # For other 5xx transient errors, also retry
            code = getattr(e.response, "status_code", None)
            if code and 500 <= code < 600 and attempt < max_retries - 1:
                wait = min(15, 2 + attempt * 3)
                print(f"⚠️ Server error {code}. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            # Last attempt or non-retryable error
            return f"[ERR HTTP] {e} :: {getattr(e.response, 'text', '')[:200]}"
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(10, 2 + attempt * 2)
                print(f"⚠️ Error '{e}'. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            return f"[ERR CALL] {e}"

def clean_one_word_number(text: str) -> str:
    """Extract a single numeric answer (int or decimal) from model output."""
    if not text or not isinstance(text, str):
        return "NaN"
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if nums:
        return nums[-1]
    toks = re.findall(r"[A-Za-z0-9\-\+\.]+", text.strip())
    return toks[-1] if toks else "NaN"

def prompt_example_rag_gsm8k(q: str, examples):
    """Few-shot prompt using examples retrieved from GSM8K."""
    shots = "\n\n".join([f"Example:\nQ: {e['question']}\nAnswer: {e['answer']}" for e in examples])
    return f"""You solve math word problems. Use the structure from these solved examples from GSM8K.

{shots}

Now solve this problem:
Q: {q}

Show short reasoning and end with:
Final Answer: <number>"""

# -------------------------
# Load datasets
# -------------------------
# Evaluation set (SVAMP)
svamp = load_dataset("ChilleD/SVAMP", split="train")
svamp_df = pd.DataFrame(svamp)

# Example-RAG corpus (GSM8K)
gsm8k = load_dataset("openai/gsm8k", "main", split="train")
corpus_df = pd.DataFrame(gsm8k)[["question","answer"]].dropna().reset_index(drop=True)

print(f"SVAMP size: {len(svamp_df)} | GSM8K (corpus) size: {len(corpus_df)}")

# -------------------------
# Build Hybrid retriever on GSM8K (BM25 + Dense) + MMR
# -------------------------
corpus_q = corpus_df["question"].astype(str).tolist()
corpus_a = corpus_df["answer"].astype(str).tolist()

# BM25 (keyword)
tokenized = [re.findall(r"\w+", q.lower()) for q in corpus_q]
bm25 = BM25Okapi(tokenized)

# Dense (semantic)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb = embedder.encode(corpus_q, show_progress_bar=True, normalize_embeddings=True)
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

def retrieve_examples_from_gsm8k(query: str, top_n=TOP_N, k=K_EX, alpha=ALPHA, lamb=LAMB):
    # Dense top
    qv = embedder.encode([query], normalize_embeddings=True)[0]
    dense_sims = emb @ qv
    d_top = np.argsort(-dense_sims)[:top_n]

    # BM25 top
    q_toks = re.findall(r"\w+", query.lower())
    bm_scores = bm25.get_scores(q_toks)
    b_top = np.argsort(-bm_scores)[:top_n]

    # Hybrid combine
    pool = list(set(d_top.tolist() + b_top.tolist()))
    d_map = {i: dense_sims[i] for i in d_top}
    b_map = {i: bm_scores[i]    for i in b_top}
    d_arr = np.array([d_map.get(i,0.0) for i in pool]).reshape(-1,1)
    b_arr = np.array([b_map.get(i,0.0) for i in pool]).reshape(-1,1)
    sc = MinMaxScaler()
    d_arr = sc.fit_transform(d_arr)
    b_arr = sc.fit_transform(b_arr)
    hybrid = alpha*d_arr + (1-alpha)*b_arr
    cand = [pool[i] for i in np.argsort(-hybrid.ravel())]

    # MMR diversification
    selected = []
    rel = dense_sims[cand]
    remaining = cand.copy()
    while remaining and len(selected) < k:
        if not selected:
            selected.append(remaining[0]); remaining.pop(0)
        else:
            sel_vecs = emb[selected]
            rem_vecs = emb[remaining]
            sim_to_sel = sel_vecs @ rem_vecs.T
            max_sim = sim_to_sel.max(axis=0) if sim_to_sel.size else np.zeros(len(remaining))
            mmr = lamb * rel[:len(remaining)] - (1 - lamb) * max_sim
            j = int(np.argmax(mmr))
            selected.append(remaining[j]); remaining.pop(j)

    return [{"question": corpus_q[i], "answer": corpus_a[i]} for i in selected]

# -------------------------
# Evaluate first N_EVAL SVAMP items with GSM8K Example-RAG (rate-limit safe)
# -------------------------
rows = []
for idx in range(min(N_EVAL, len(svamp_df))):
    row = svamp_df.iloc[idx]
    question_text = (str(row.get("Body","")) + " " + str(row.get("Question",""))).strip()
    gold = str(row.get("Answer",""))

    # Retrieve examples from GSM8K
    examples = retrieve_examples_from_gsm8k(question_text, k=K_EX)
    # Build prompt
    prompt = prompt_example_rag_gsm8k(question_text, examples)
    # Call Mistral (with retry + backoff)
    raw = call_mistral_chat(MODEL_ID, prompt)
    # Extract final numeric answer
    pred = clean_one_word_number(raw)

    rows.append({
        "Dataset": "SVAMP",
        "Idx": idx,
        "RAG_Type": "Example-RAG (GSM8K)",
        "Model": MODEL_ID,
        "Gold": gold,
        "Predicted": pred,
        "Raw": raw  # uncomment if you want to keep raw model text
    })

    # Sleep between calls to reduce 429s when N_EVAL > 1
    time.sleep(SLEEP_BETWEEN_CALLS)

results = pd.DataFrame(rows)
print(f"Done. Evaluated N={len(results)} SVAMP items with GSM8K as Example-RAG.")
display(results)

results.to_csv("svamp_results.csv", index=False)
print("✅ Results saved as svamp_results.csv")


files.download("svamp_example_rag_results.csv")

