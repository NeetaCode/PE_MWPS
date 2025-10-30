# =========================
# Colab Setup: Deps
# =========================
!pip -q install datasets pandas numpy requests rank-bm25

# =========================
# Imports & Config
# =========================
import os, re, json, time, random, math, statistics
import numpy as np, pandas as pd, requests
from datetime import datetime
from datasets import load_dataset
from rank_bm25 import BM25Okapi

# ---- Repro ----
random.seed(42); np.random.seed(42)

# ---- Mistral (OpenAI-compatible) ----
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
MISTRAL_API_KEY  = "hYnJKM53YmlSWDwjdiXGdTWos4VRrgHd"   # <-- replace if needed
MISTRAL_MODEL    = "mistral-small"

os.environ["OPENAI_BASE_URL"] = MISTRAL_BASE_URL
os.environ["OPENAI_API_KEY"]  = MISTRAL_API_KEY
MODEL_ID = MISTRAL_MODEL

# ---- Eval knobs ----
N_EVAL      = 100         # evaluate first N rows
TEMP_COT    = 0.0         # deterministic CoT
TEMP_RAG    = 0.0         # deterministic RAG
TOP_P       = 1.0
MAX_TOKENS  = 512
RT_MAX_RETRY= 6
PAUSE_SEC   = 0.8         # light delay to avoid 429s

# =========================
# Parsing / Normalization helpers
# =========================
FINAL_PATTERNS = [
    r"<ans>\s*([-+]?\d*\.?\d+)\s*</ans>",
    r"FINAL_ANSWER\s*[:：]?\s*([-+]?\d*\.?\d+)",
    r"\banswer\s*is\s*([-+]?\d*\.?\d+)\b",
    r"\bboxed\{\s*([-+]?\d*\.?\d+)\s*\}",
    r"=\s*([-+]?\d*\.?\d+)\s*$",
]
NUM_FALLBACK = r"[-+]?\d*\.?\d+"

def normalize_numeric_string(s):
    if s is None: return ""
    s = str(s).replace(",", " ").strip()
    m = re.findall(NUM_FALLBACK, s)
    return m[-1] if m else s

def extract_one_number(text):
    if not text: return "NaN"
    for pat in FINAL_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m: return m.group(1)
    nums = re.findall(NUM_FALLBACK, text)
    return nums[-1] if nums else "NaN"

def to_float_safe(x):
    try: 
        f = float(str(x))
        if math.isfinite(f): return f
        return None
    except Exception:
        return None

def nearly_equal(a, b, tol=1e-6):
    if a is None or b is None: return False
    return abs(a - b) <= tol

# =========================
# API caller (retry-safe)
# =========================
def call_chat(model, system_prompt, user_prompt, temperature=0.0, top_p=1.0,
              max_tokens=512, max_retries=RT_MAX_RETRY, timeout_s=120):
    base, key = os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_API_KEY"]
    url = f"{base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=timeout_s)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(10, 1.5 + attempt * 1.5) + random.uniform(0, 0.4)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(8, 1.0 + attempt * 1.25) + random.uniform(0, 0.4)
                time.sleep(wait)
            else:
                return f"[ERROR] {e}"
    return "[ERROR] Max retries exceeded"

# =========================
# Prompts (CoT + MiniKB-RAG)
# =========================
def build_cot_prompt(problem):
    return (
        "Solve the following grade-school math word problem.\n"
        "Think step by step in a few concise lines, avoid unnecessary text.\n"
        "If the answer is an integer, print it as an integer (no decimals).\n"
        "If the answer is a decimal, round to two decimal places. Do not include units.\n\n"
        f"Problem:\n{problem}\n\n"
        "Answer:"
    )

SYSTEM_ENFORCE_TAGS = (
    "You are a careful math assistant. Solve step by step with concise logic.\n"
    "On the LAST two lines, output exactly both forms:\n"
    "<ans>NUMBER</ans>\n"
    "FINAL_ANSWER: NUMBER"
)

# --- Mini formula knowledge base ---
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
     "formula":"a/b = c/d ⇒ cross-multiply: a*d = b*c", "tip":"Use cross-multiplication."},
    {"name":"Rate: speed-distance-time", "triggers":"speed rate mph kph distance time hour minute",
     "formula":"d = r * t;  r = d / t;  t = d / r", "tip":"Match units first (e.g., minutes→hours)."},
    {"name":"Work rates", "triggers":"together working per hour fill drain pipe",
     "formula":"combined rate = r1 + r2; time = total / combined rate", "tip":"Add rates for joint work."},
    {"name":"Unit convert (time)", "triggers":"minutes hours seconds",
     "formula":"1 hr = 60 min; 1 min = 60 s", "tip":"Convert to same unit before computing."},
    {"name":"Unit convert (length)", "triggers":"cm m km",
     "formula":"100 cm = 1 m; 1000 m = 1 km", "tip":"Keep units consistent."},
    {"name":"Unit convert (mass)", "triggers":"g kg",
     "formula":"1000 g = 1 kg", "tip":"Convert to same mass unit."},
    {"name":"Money", "triggers":"dollars cents price cost change",
     "formula":"$1 = 100 cents; total = price * quantity", "tip":"Watch for change/remaining."},
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
      "formula":"LCM for repeating cycles", "tip":"Find first common multiple."},
    {"name":"GCD (sharing equally)", "triggers":"greatest common divisor share equally split exact",
      "formula":"GCD for largest equal group size", "tip":"Use GCD to maximize equal groups."},
    {"name":"Remainder", "triggers":"left over remainder groups of",
      "formula":"a = b*q + r (0 ≤ r < b)", "tip":"Division with leftover r."},
    {"name":"Weighted average", "triggers":"average of groups combined average",
      "formula":"avg = (Σ w_i * x_i) / (Σ w_i)", "tip":"Use weights."},
]

kb_docs = [(k["name"] + " " + k["triggers"] + " " + k["formula"] + " " + k["tip"]).lower() for k in KB]
bm25_kb = BM25Okapi([re.findall(r"\w+", d) for d in kb_docs])

def kb_lookup(query_text, k=5):
    q = re.findall(r"\w+", str(query_text).lower())
    scores = bm25_kb.get_scores(q)
    top = np.argsort(-scores)[:k]
    return [KB[i] for i in top], float(np.max(scores)) if len(scores) else 0.0

def prompt_formula_kb_rag(problem, kb_entries):
    kb_lines = [f"- {k['name']}: {k['formula']} ({k['tip']})" for k in kb_entries]
    kb_text = "\n".join(kb_lines)
    return f"""Use the following math knowledge base to solve the problem.

Knowledge Base:
{kb_text}

Problem:
{problem}

Explain briefly and end with:
<ans>NUMBER</ans>
FINAL_ANSWER: NUMBER
"""

# =========================
# Load SVAMP
# =========================
svamp = load_dataset("ChilleD/SVAMP", split="train")
df = pd.DataFrame(svamp)

# Robust column mapping
cols = {c.lower(): c for c in df.columns}
body_col     = cols.get("body")
question_col = cols.get("question")
answer_col   = cols.get("answer") or cols.get("ans")

# =========================
# Evaluate (CoT + RAG; no self-consistency)
# =========================
rows = []
for idx, row in df.head(N_EVAL).iterrows():
    problem_text = f"{str(row.get(body_col,''))} {str(row.get(question_col,''))}".strip()
    gold_raw = str(row.get(answer_col, ""))
    gold_norm = normalize_numeric_string(gold_raw)
    gold_f = to_float_safe(gold_norm)

    # --- CoT ---
    cot_user = build_cot_prompt(problem_text)
    cot_out  = call_chat(MODEL_ID, SYSTEM_ENFORCE_TAGS, cot_user,
                         temperature=TEMP_COT, top_p=TOP_P, max_tokens=MAX_TOKENS)
    cot_pred = normalize_numeric_string(extract_one_number(cot_out))
    cot_f    = to_float_safe(cot_pred)

    time.sleep(PAUSE_SEC)

    # --- Formula/KB-RAG ---
    kb_hits, kb_topscore = kb_lookup(problem_text, k=5)
    rag_user = prompt_formula_kb_rag(problem_text, kb_hits)
    rag_out  = call_chat(MODEL_ID, SYSTEM_ENFORCE_TAGS, rag_user,
                         temperature=TEMP_RAG, top_p=TOP_P, max_tokens=MAX_TOKENS)
    rag_pred = normalize_numeric_string(extract_one_number(rag_out))
    rag_f    = to_float_safe(rag_pred)

    # --- Final pick (no self-consistency) ---
    # 1) If both numeric and agree → use that
    # 2) Else prefer RAG (aligns with your request)
    # 3) If RAG missing → use CoT; if CoT missing → use RAG; else NaN
    method_used = ""
    if cot_f is not None and rag_f is not None and nearly_equal(cot_f, rag_f):
        final_pred, method_used = rag_pred, "Agree(COT=RAG)"
    elif rag_f is not None:
        final_pred, method_used = rag_pred, "Prefer RAG"
    elif cot_f is not None:
        final_pred, method_used = cot_pred, "Fallback CoT"
    else:
        final_pred, method_used = "NaN", "No numeric"

    final_f = to_float_safe(final_pred)

    is_correct_final = (gold_f is not None and final_f is not None and nearly_equal(gold_f, final_f))
    is_correct_cot   = (gold_f is not None and cot_f   is not None and nearly_equal(gold_f, cot_f))
    is_correct_rag   = (gold_f is not None and rag_f   is not None and nearly_equal(gold_f, rag_f))

    rows.append({
        "index": idx,
        "problem": problem_text,
        "gold_answer": gold_raw,
        "cot_pred": cot_pred,
        "rag_pred": rag_pred,
        "final_pred": final_pred,
        "method": method_used,
        "kb_topscore": kb_topscore,
        "is_correct_cot": bool(is_correct_cot),
        "is_correct_rag": bool(is_correct_rag),
        "is_correct_final": bool(is_correct_final),
        # Optional raw text (commented to keep CSV small)
        # "cot_out": cot_out,
        # "rag_out": rag_out,
    })

    print(f"✅ Row {idx} | Gold={gold_norm} | CoT={cot_pred} | RAG={rag_pred} | Final={final_pred} ({method_used}) | Correct={is_correct_final}")

# =========================
# Summary + Save
# =========================
res = pd.DataFrame(rows)

def pct(x): 
    return 100.0 * x.sum() / max(1, len(x))

acc_cot   = pct(res["is_correct_cot"])
acc_rag   = pct(res["is_correct_rag"])
acc_final = pct(res["is_correct_final"])

print("\n📊 CoT + Formula-RAG (No Self-Consistency)")
print("--------------------------------------------")
print(f"Model:           {MODEL_ID}")
print(f"N evaluated:     {len(res)}")
print(f"Accuracy (CoT):  {acc_cot:.2f}%")
print(f"Accuracy (RAG):  {acc_rag:.2f}%")
print(f"Accuracy (Final):{acc_final:.2f}%")
print("--------------------------------------------")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"svamp_cot_plus_formula_rag_noSC_{MODEL_ID}_N{N_EVAL}_{ts}.csv"
res.to_csv(csv_path, index=False)
print(f"📁 Saved results to: {csv_path}")
