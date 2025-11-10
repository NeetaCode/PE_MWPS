# ================================================
# ARAP for SVAMP (Dual-RAG + Context Optimization + Light Decomposition)
# Resume-safe logging to Google Drive (Colab) or local folder
# Mistral (OpenAI-compatible) API with retry + backoff
# ================================================

# ---- installs (Colab) ----
!pip -q install datasets pandas sentence-transformers scikit-learn rank-bm25 requests

# ---- imports ----
import os, re, json, time, pathlib, sys
from datetime import datetime
import numpy as np
import pandas as pd
import requests

from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

# ================================================
# Drive mounting (optional in Colab) + output dirs
# ================================================
try:
    import google.colab  # noqa: F401
    from google.colab import drive
    drive.mount('/content/drive')
    DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive"
except Exception:
    DEFAULT_DRIVE_ROOT = os.getcwd()

OUTPUT_DIR = os.path.join(DEFAULT_DRIVE_ROOT, "llm_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================
# API config (Mistral via OpenAI-compatible endpoint)
# Prefer: in Colab run ->  %env OPENAI_API_KEY=sk-...
# ================================================
os.environ.setdefault("OPENAI_BASE_URL", "https://api.mistral.ai/v1")
os.environ.setdefault("OPENAI_API_KEY", "hYnJKM53YmlSWDwjdiXGdTWos4VRrgHd")  # set via %env for safety
if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️ OPENAI_API_KEY is not set. In Colab, run: %env OPENAI_API_KEY=sk-...")

MODEL_ID = "mistral-small"

# ================================================
# Run naming + resume settings
# ================================================
RUN_TS       = datetime.now().strftime("%Y%m%d_%H%M%S")
BASENAME     = f"svamp_arap_{MODEL_ID}"
LATEST_FILE  = os.path.join(OUTPUT_DIR, f"{BASENAME}_latest.csv")
ARCHIVE_FILE = os.path.join(OUTPUT_DIR, f"{BASENAME}_{RUN_TS}.csv")
RESUME       = True

# ================================================
# Evaluation and retrieval hyperparams
# ================================================
TOP_N  = 20      # hybrid pool size before MMR
K_EX   = 3       # retrieved examples into the prompt (we will compress to 2)
ALPHA  = 0.6     # hybrid weight: alpha*dense + (1-alpha)*bm25
LAMB   = 0.5     # MMR relevance vs diversity
N_EVAL = 100     # number of SVAMP items to evaluate this run
SLEEP_BETWEEN_CALLS = 3

# ARAP-specific knobs
ARAP_K_FORMULAS   = 2   # top formulas to include
ARAP_K_EXAMPLES   = 2   # top examples to include
TEMPERATURE       = 0.2
SELF_CONSISTENCY  = 1   # set to 3 for a small vote/median improvement

# ================================================
# HTTP call with retry
# ================================================
def call_mistral_chat(model: str, prompt: str, temperature=0.2, max_retries=6):
    base = os.environ.get("OPENAI_BASE_URL", "")
    key  = os.environ.get("OPENAI_API_KEY", "")
    if not base or not key:
        return "[ERR] Missing OPENAI_BASE_URL or OPENAI_API_KEY"

    url = f"{base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": float(temperature),
        "messages": [
            {"role": "system",
             "content": "You are a careful math assistant. Provide concise reasoning and a final numeric answer."},
            {"role": "user", "content": prompt},
        ],
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=120)
            if r.status_code == 429:
                wait = min(15, 2 + attempt * 3)
                print(f"⚠️ Rate limit (429). Retrying in {wait}s...")
                time.sleep(wait)
                continue
            if 500 <= r.status_code < 600:
                wait = min(15, 2 + attempt * 3)
                print(f"⚠️ Server error {r.status_code}. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            payload = r.json()
            return payload["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            txt = getattr(e.response, "text", "")[:200]
            return f"[ERR HTTP] {e} :: {txt}"
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(10, 2 + attempt * 2)
                print(f"⚠️ Error '{e}'. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            return f"[ERR CALL] {e}"

# ================================================
# Utilities
# ================================================
def clean_one_word_number(text: str) -> str:
    if not text or not isinstance(text, str) or text.startswith("[ERR"):
        return "NaN"
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if nums:
        return nums[-1]
    toks = re.findall(r"[A-Za-z0-9\-\+\.]+", text.strip())
    return toks[-1] if toks else "NaN"

def append_row_safely(csv_path: str, row_dict: dict):
    file_exists = os.path.exists(csv_path)
    df = pd.DataFrame([row_dict])
    df.to_csv(csv_path, mode="a", header=not file_exists, index=False)

# ================================================
# Load datasets
# ================================================
svamp = load_dataset("ChilleD/SVAMP", split="train")
svamp_df = pd.DataFrame(svamp)

gsm8k = load_dataset("openai/gsm8k", "main", split="train")
corpus_df = pd.DataFrame(gsm8k)[["question", "answer"]].dropna().reset_index(drop=True)

print(f"SVAMP size: {len(svamp_df)} | GSM8K (corpus) size: {len(corpus_df)}")

# ================================================
# Mini formula knowledge base + BM25 index
# ================================================
KB = [
    {"name":"Addition", "triggers":"sum together in all total combined altogether add plus",
     "formula":"total = a + b (+ ...)", "tip":"Look for combining quantities."},
    {"name":"Subtraction", "triggers":"left remain fewer less minus decrease difference after gave away",
     "formula":"difference = a - b", "tip":"Look for leftover or reduction."},
    {"name":"Multiplication", "triggers":"times each groups of repeated addition product per box",
     "formula":"product = a * b", "tip":"Equal groups or repeated addition."},
    {"name":"Division", "triggers":"per each evenly split share quotient distribute",
     "formula":"quotient = a / b", "tip":"Equal sharing or per-unit value."},
    {"name":"Average (mean)", "triggers":"average mean on average",
     "formula":"avg = (sum of items) / (count of items)", "tip":"Sum then divide by count."},
    {"name":"Percent (of)", "triggers":"percent of %",
     "formula":"x% of N = (x/100)*N", "tip":"Convert % to decimal then multiply."},
    {"name":"Percent increase", "triggers":"increase by percent more than",
     "formula":"new = base * (1 + x/100)", "tip":"Add the percent."},
    {"name":"Percent decrease", "triggers":"decrease by percent less than discount off",
     "formula":"new = base * (1 - x/100)", "tip":"Subtract the percent."},
    {"name":"Ratio/Proportion", "triggers":"ratio proportion per scale similar",
     "formula":"a/b = c/d ⇒ a*d = b*c", "tip":"Use cross-multiplication."},
    {"name":"Rate: speed-distance-time", "triggers":"speed rate mph kph distance time hour minute",
     "formula":"d = r * t; r = d / t; t = d / r", "tip":"Match units first."},
    {"name":"Work rates", "triggers":"together working per hour fill drain pipe",
     "formula":"combined rate = r1 + r2; time = total / (r1 + r2)", "tip":"Add rates for joint work."},
    {"name":"Unit convert (time)", "triggers":"minutes hours seconds",
     "formula":"1 hr = 60 min; 1 min = 60 s", "tip":"Convert to same unit."},
    {"name":"Unit convert (length)", "triggers":"cm m km",
     "formula":"100 cm = 1 m; 1000 m = 1 km", "tip":"Keep units consistent."},
    {"name":"Unit convert (mass)", "triggers":"g kg",
     "formula":"1000 g = 1 kg", "tip":"Convert to same mass unit."},
    {"name":"Money", "triggers":"dollars cents price cost change",
     "formula":"$1 = 100 cents; total = price * quantity", "tip":"Watch change/remaining."},
    {"name":"Rectangle perimeter", "triggers":"rectangle perimeter fence border",
     "formula":"P = 2*(L + W)", "tip":"Perimeter adds all sides."},
    {"name":"Rectangle area", "triggers":"rectangle area floor cover",
     "formula":"A = L * W", "tip":"Multiply length by width."},
    {"name":"Triangle area", "triggers":"triangle area",
     "formula":"A = (base * height) / 2", "tip":"Use base × height / 2."},
    {"name":"Circle circumference", "triggers":"circle circumference around",
     "formula":"C = 2πr", "tip":"Use radius r."},
    {"name":"Circle area", "triggers":"circle area",
     "formula":"A = πr^2", "tip":"Square the radius."},
    {"name":"LCM (equal groups)", "triggers":"every each schedule together again least common multiple",
     "formula":"use LCM for repeating cycles", "tip":"Find first common multiple."},
    {"name":"GCD (sharing equally)", "triggers":"greatest common divisor share equally split exact",
     "formula":"use GCD for largest equal group", "tip":"Use GCD to maximize equal groups."},
    {"name":"Remainder", "triggers":"left over remainder groups of",
     "formula":"a = b*q + r (0 ≤ r < b)", "tip":"Division with leftover r."},
    {"name":"Weighted average", "triggers":"average of groups combined average",
     "formula":"avg = (Σ w_i * x_i) / (Σ w_i)", "tip":"Weights are counts or proportions."},
]
kb_docs = [(k["name"] + " " + k["triggers"] + " " + k["formula"] + " " + k["tip"]).lower() for k in KB]
bm25_kb = BM25Okapi([re.findall(r"\w+", d) for d in kb_docs])

# ================================================
# GSM8K hybrid retriever + MMR (from your code)
# ================================================
corpus_q = corpus_df["question"].astype(str).tolist()
corpus_a = corpus_df["answer"].astype(str).tolist()

tokenized = [re.findall(r"\w+", q.lower()) for q in corpus_q]
bm25 = BM25Okapi(tokenized)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb = embedder.encode(corpus_q, show_progress_bar=True, normalize_embeddings=True)
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

def retrieve_examples_from_gsm8k(query: str, top_n=TOP_N, k=K_EX, alpha=ALPHA, lamb=LAMB):
    qv = embedder.encode([query], normalize_embeddings=True)[0]
    dense_sims = emb @ qv
    d_top = np.argsort(-dense_sims)[:top_n]

    q_toks = re.findall(r"\w+", query.lower())
    bm_scores = bm25.get_scores(q_toks)
    b_top = np.argsort(-bm_scores)[:top_n]

    pool = list(set(d_top.tolist() + b_top.tolist()))
    d_map = {i: dense_sims[i] for i in d_top}
    b_map = {i: bm_scores[i]    for i in b_top}

    d_arr = np.array([d_map.get(i, 0.0) for i in pool]).reshape(-1, 1)
    b_arr = np.array([b_map.get(i, 0.0) for i in pool]).reshape(-1, 1)
    sc = MinMaxScaler()
    d_arr = sc.fit_transform(d_arr)
    b_arr = sc.fit_transform(b_arr)
    hybrid = alpha * d_arr + (1 - alpha) * b_arr
    cand = [pool[i] for i in np.argsort(-hybrid.ravel())]

    selected = []
    remaining = cand.copy()
    while remaining and len(selected) < k:
        if not selected:
            selected.append(remaining.pop(0))
            continue
        rel_rem = np.array([dense_sims[i] for i in remaining])
        sel_vecs = emb[selected]
        rem_vecs = emb[remaining]
        if sel_vecs.ndim == 1:
            sel_vecs = sel_vecs.reshape(1, -1)
        sim_to_sel = sel_vecs @ rem_vecs.T
        max_sim = sim_to_sel.max(axis=0) if sim_to_sel.size else np.zeros(len(remaining))
        mmr = lamb * rel_rem - (1 - lamb) * max_sim
        j = int(np.argmax(mmr))
        selected.append(remaining.pop(j))

    return [{"question": corpus_q[i], "answer": corpus_a[i]} for i in selected]

# ================================================
# ARAP helpers (Dual-RAG + Context Optimization + Light Decomposition)
# ================================================
UNIT_TOKS = set("""
cm m km mm inch inches feet foot ft yard yd mile mi kg g mg lb lbs
hour hours hr hrs minute minutes min second seconds s
dollar dollars $ percent % degree degrees °
""".split())

def _sentences(text):
    return [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', str(text)) if s.strip()]

def _extract_numbers_with_units(text):
    pairs = re.findall(r'(\d+(?:\.\d+)?)\s*([A-Za-z%°$]+)?', text)
    out = []
    for num, unit in pairs:
        unit = unit or ""
        if unit and unit.lower() not in UNIT_TOKS and not re.match(r'[%°$]', unit):
            unit = ""
        out.append((num, unit))
    return out

def decompose_problem(q: str) -> str:
    sents = _sentences(q)
    question_sent = sents[-1] if sents else q
    nums = _extract_numbers_with_units(q)
    uniq, seen = [], set()
    for num, unit in nums:
        key = (num, unit)
        if key not in seen:
            seen.add(key)
            uniq.append(f"{num}{unit if unit else ''}")
    knowns = ", ".join(uniq[:6]) if uniq else "None obvious"
    return f"Knowns: {knowns}\nUnknown: not explicitly given\nGoal: {question_sent}"

def rank_formulas_for_query(query_text: str, k_top: int = ARAP_K_FORMULAS):
    q = re.findall(r"\w+", query_text.lower())
    scores = bm25_kb.get_scores(q)
    top = np.argsort(-scores)[:k_top]
    picks = []
    for i in top:
        k = KB[i]
        picks.append(f"{k['name']}: {k['formula']}. {k['tip']}")
    return picks

def _shorten_answer(ans: str, max_chars=220):
    s = re.sub(r'\s+', ' ', str(ans)).strip()
    return s[:max_chars] + ("…" if len(s) > max_chars else "")

def compress_examples(examples, k_keep=ARAP_K_EXAMPLES):
    picked = examples[:k_keep]
    hints = []
    for e in picked:
        q = re.sub(r'\s+', ' ', e.get('question','')).strip()
        a = _shorten_answer(e.get('answer',''))
        hints.append(f"Q: {q}\nA (pattern): {a}")
    return hints

def build_arap_prompt(problem_text: str, formula_hints, example_hints, decomposition_text):
    f_block = "\n".join([f"- {h}" for h in formula_hints]) if formula_hints else "- (none)"
    e_block = "\n\n".join([f"* {h}" for h in example_hints]) if example_hints else "* (none)"
    return f"""You solve math word problems carefully and concisely.

Decomposition
{decomposition_text}

Principle Hints
{f_block}

Example Patterns
{e_block}

Problem
{problem_text}

Instructions
1) Think step by step.
2) Write any needed equations.
3) Compute carefully.
4) Give the final numeric answer only in the exact format: Final Answer: <number>"""

def final_number_or_retry(raw_text, model_id, prompt, temperature=0.2, sc=SELF_CONSISTENCY):
    def one_clean_call():
        raw_local = call_mistral_chat(model_id, prompt, temperature=temperature)
        return clean_one_word_number(raw_local), raw_local

    pred = clean_one_word_number(raw_text)
    raws = [raw_text]

    if pred == "NaN":
        fix_prompt = prompt + "\n\nYour previous answer was not a single number. Recheck and output only: Final Answer: <number>"
        pred2, raw2 = one_clean_call()
        pred, raws = pred2, raws + [raw2]

    if sc > 1:
        votes = []
        for _ in range(sc - 1):
            p, r = one_clean_call()
            votes.append((p, r))
        # keep only numeric votes
        nums = []
        for p, r in votes + [(pred, raws[-1])]:
            try:
                nums.append(float(p))
            except Exception:
                pass
        if nums:
            # majority/median heuristic: choose median for robustness
            nums_sorted = sorted(nums)
            chosen = nums_sorted[len(nums_sorted)//2]
            pred = str(int(chosen)) if abs(chosen - int(chosen)) < 1e-9 else str(chosen)

    return pred, raws[-1]

# ================================================
# Resume handling
# ================================================
done_idxs = set()
if RESUME and os.path.exists(LATEST_FILE):
    try:
        prev = pd.read_csv(LATEST_FILE, usecols=["Idx"])
        done_idxs = set(prev["Idx"].astype(int).tolist())
        print(f"↩️ Resume mode: found {len(done_idxs)} completed rows in latest CSV, will skip them.")
    except Exception as e:
        print(f"⚠️ Could not read LATEST_FILE to resume: {e}")

# ================================================
# ARAP evaluation loop on SVAMP
# ================================================
rows_for_display = []
count = 0

for idx in range(min(N_EVAL, len(svamp_df))):
    if idx in done_idxs:
        continue

    row = svamp_df.iloc[idx]
    question_text = (str(row.get("Body","")) + " " + str(row.get("Question",""))).strip()
    gold = str(row.get("Answer",""))

    # Dual retrieval
    formula_hints = rank_formulas_for_query(question_text, k_top=ARAP_K_FORMULAS)
    examples      = retrieve_examples_from_gsm8k(question_text, k=ARAP_K_EXAMPLES)
    example_hints = compress_examples(examples, k_keep=ARAP_K_EXAMPLES)

    # Decomposition
    decomp = decompose_problem(question_text)

    # Prompt
    prompt = build_arap_prompt(question_text, formula_hints, example_hints, decomp)

    # Call model (+ tiny retry + optional self-consistency)
    raw = call_mistral_chat(MODEL_ID, prompt, temperature=TEMPERATURE)
    pred, raw_used = final_number_or_retry(raw, MODEL_ID, prompt, temperature=TEMPERATURE, sc=SELF_CONSISTENCY)

    row_out = {
        "Dataset": "SVAMP",
        "Idx": idx,
        "RAG_Type": "ARAP (Dual-RAG + Decomposition + ContextOpt)",
        "Model": MODEL_ID,
        "Gold": gold,
        "Predicted": pred,
        "Raw": raw_used
    }

    append_row_safely(LATEST_FILE, row_out)
    rows_for_display.append(row_out)
    count += 1

    print(f"✅ Done: Row {idx} (Gold={gold}, Pred={pred})")
    time.sleep(SLEEP_BETWEEN_CALLS)

print(f"Done. Evaluated {count} new SVAMP items this session (N_EVAL cap={N_EVAL}).")
print(f"📄 Live results (resumable): {LATEST_FILE}")

# Archive snapshot at end
try:
    latest_df = pd.read_csv(LATEST_FILE)
    latest_df.to_csv(ARCHIVE_FILE, index=False)
    print(f"🗂️ Archived snapshot for this run: {ARCHIVE_FILE}")
except Exception as e:
    print(f"⚠️ Could not create archive snapshot: {e}")

# Quick display of just-processed rows (not the entire historical CSV)
try:
    disp_df = pd.DataFrame(rows_for_display)
    from IPython.display import display
    display(disp_df)
except Exception:
    pass
