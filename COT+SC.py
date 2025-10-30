# =========================
# Colab Setup: Deps
# =========================
!pip -q install datasets pandas requests numpy sympy

# =========================
# Imports
# =========================
from datasets import load_dataset
import pandas as pd, numpy as np, requests, os, re, time, random, statistics
from collections import Counter
from datetime import datetime
import sympy as sp

# =========================
# Config: Mistral (OpenAI-compatible)
# =========================
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
MISTRAL_API_KEY  = "hYnJKM53YmlSWDwjdiXGdTWos4VRrgHd"   # <-- 🔑 do NOT hardcode in real runs
MISTRAL_MODEL    = "mistral-small"       # or "mistral-medium"

os.environ["OPENAI_BASE_URL"] = MISTRAL_BASE_URL
os.environ["OPENAI_API_KEY"]  = MISTRAL_API_KEY
MODEL_ID = MISTRAL_MODEL

# Inference knobs (self-consistency)
TEMP_SC, TOP_P_SC = 0.5, 0.95
MAX_TOKENS = 512
K_SAMPLES = 12      # try 12–20 for better stability
random.seed(42); np.random.seed(42)

# =========================
# Parsing helpers
# =========================
FINAL_PATTERNS = [
    r"<ans>\s*([-+]?\d*\.?\d+)\s*</ans>",
    r"FINAL_ANSWER\s*[:：]\s*([-+]?\d*\.?\d+)",
]
NUM_FALLBACK = r"[-+]?\d*\.?\d+"

def normalize_numeric_string(s):
    if s is None: return ""
    s = str(s).replace(",", " ").strip()
    m = re.findall(NUM_FALLBACK, s)
    return m[-1] if m else s

def clean_one_word_number(text):
    if not text: return "NaN"
    t = text.strip()
    if t.startswith("[ERROR]") or "Unauthorized" in t: return "NaN"
    for pat in FINAL_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE | re.MULTILINE)
        if m: return m.group(1)
    nums = re.findall(NUM_FALLBACK, t)
    return nums[-1] if nums else "NaN"

def to_float_safe(x):
    try: return float(str(x))
    except Exception: return None

def nearly_equal(a, b, tol=1e-6):
    return (a is not None and b is not None and abs(a - b) <= tol)

# Prefer “math result” over prose by evaluating <eq>...</eq>
EQ_PATTERN = re.compile(r"<eq>(.*?)</eq>", flags=re.IGNORECASE | re.DOTALL)

def try_eval_eq(text):
    m = EQ_PATTERN.search(text or "")
    if not m: return None
    expr = m.group(1).strip()
    try:
        expr = expr.replace("^", "**")
        val = sp.N(sp.sympify(expr))
        return float(val)
    except Exception:
        return None

def two_decimals_if_needed(x):
    """Format: integers as int, decimals rounded to 2 dp."""
    if x is None: return None
    if abs(x - round(x)) < 1e-9:
        return float(int(round(x)))
    return float(round(x, 2))

def compare_gold_pred(gold_raw, pred_float):
    """SVAMP is mostly integers; be tolerant:
       - If gold looks integer, compare to rounded(pred).
       - Else compare to 2-decimal rounding."""
    gold_norm = normalize_numeric_string(gold_raw)
    gold_f = to_float_safe(gold_norm)
    if gold_f is None or pred_float is None:
        return False, gold_f
    # integer-like gold?
    if abs(gold_f - round(gold_f)) < 1e-9:
        return nearly_equal(round(gold_f), round(pred_float)), gold_f
    # decimal-like gold: compare to 2dp
    return nearly_equal(round(gold_f, 2), round(pred_float, 2)), gold_f

# =========================
# Prompts
# =========================
SYSTEM_PROMPT = (
    "You are a careful math assistant.\n"
    "Solve step by step with concise logic (3–8 short lines).\n"
    "Include exactly ONE arithmetic/algebra line that evaluates to the final number, wrapped as:\n"
    "<eq> ...a single Pythonic expression using only numbers and + - * / ( ) ... </eq>\n"
    "On the LAST two lines, output exactly:\n"
    "<ans>NUMBER</ans>\n"
    "FINAL_ANSWER: NUMBER\n"
    "If the answer is an integer, print it as an integer (no decimals).\n"
    "If the answer is a decimal, round it to two decimal places.\n"
    "No units anywhere."
)

# =========================
# Chain-of-Thought prompt (2-decimals for non-integers)
# =========================
def build_cot_prompt(problem):
    return (
        "Solve the following grade-school math word problem.\n"
        "Reason step by step in 3–8 short lines. "
        "If the answer is an integer, print it as an integer (no decimals). "
        "If the answer is a decimal, round it to two decimal places. "
        "Do not include units.\n\n"
        f"Problem:\n{problem}\n\n"
        "Answer:"
    )

# =========================
# Chat call (single sample)
# =========================
def call_chat_once(model, prompt, temperature=0.5, top_p=0.95,
                   max_tokens=512, max_retries=4, timeout_s=60):
    base, key = os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_API_KEY"]
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model, "temperature": temperature, "top_p": top_p,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout_s)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep((2**attempt) + random.uniform(0, 0.5)); continue
            return f"[ERROR] API error {r.status_code}: {r.text}"
        except requests.exceptions.RequestException:
            time.sleep((2**attempt) + random.uniform(0, 0.5))
    return "[ERROR] Max retries exceeded"

# =========================
# Self-consistency aggregator
# =========================
def aggregate_numbers(nums):
    nums = [x for x in nums if x is not None]
    if not nums: return None
    rounded = [round(x, 6) for x in nums]
    counts = Counter(rounded).most_common()
    top_freq = counts[0][1]
    candidates = [v for v, c in counts if c == top_freq]
    if len(candidates) == 1:
        return candidates[0]
    filtered = [x for x in nums if round(x, 6) in candidates]
    try:
        return float(statistics.median(filtered))
    except Exception:
        return filtered[0]

# =========================
# Run self-consistency for one problem
# =========================
def run_self_consistency(problem, k=12, temp=0.5, top_p=0.95):
    prompt = build_cot_prompt(problem)
    votes_float, outs = [], []
    for _ in range(k):
        out = call_chat_once(MODEL_ID, prompt, temperature=temp, top_p=top_p, max_tokens=MAX_TOKENS)
        outs.append(out)
        # 1) prefer <ans> tag
        num = normalize_numeric_string(clean_one_word_number(out))
        fnum = to_float_safe(num)
        # 2) cross-check with <eq>
        eq_val = try_eval_eq(out)
        # if <ans> missing or NaN, fallback to eq; else prefer <ans>
        final = fnum if fnum is not None else eq_val
        # normalize formatting (int vs 2dp)
        final = two_decimals_if_needed(final)
        votes_float.append(final)
    agg = aggregate_numbers(votes_float)
    return agg, votes_float, outs

# =========================
# Optional verifier pass (can correct)
# =========================
def verify_with_model(problem, candidate):
    if candidate is None: return None
    base, key = os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_API_KEY"]
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    verify_user = (
        "Check the proposed numeric answer. Reply on ONE line with:\n"
        "CORRECT <ans>NUMBER</ans>\n"
        "or\n"
        "INCORRECT <ans>CORRECT_NUMBER</ans>\n\n"
        f"Problem:\n{problem}\n"
        f"Proposed: {candidate}\n"
        "Verdict:"
    )
    body = {
        "model": MODEL_ID, "temperature": 0.0, "top_p": 1.0, "max_tokens": 64,
        "messages": [
            {"role":"system","content":"Be strict and concise. Use only the specified format."},
            {"role":"user","content": verify_user}
        ]
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=60)
        if r.status_code != 200: return candidate
        txt = r.json()["choices"][0]["message"]["content"]
        verdict_correct = txt.strip().upper().startswith("CORRECT")
        num = to_float_safe(normalize_numeric_string(clean_one_word_number(txt)))
        if verdict_correct:
            return candidate
        # normalize corrected value formatting
        num = two_decimals_if_needed(num)
        return num if num is not None else candidate
    except Exception:
        return candidate

# =========================
# Load SVAMP & Evaluate first N rows with Self-Consistency
# =========================
svamp = load_dataset("ChilleD/SVAMP")
train_df = pd.DataFrame(svamp["train"])
cols = {c.lower(): c for c in train_df.columns}
body_col, question_col, answer_col = cols.get("body"), cols.get("question"), cols.get("answer") or cols.get("ans")

results = []
N = 100  # <-- set to 1 for debugging; raise to 100 when happy

print(f"Running CoT + Self-Consistency (k={K_SAMPLES}) on first {N} SVAMP items...")
for idx, row in train_df.head(N).iterrows():
    problem_text = f"{row.get(body_col,'')} {row[question_col]}".strip()
    gold_raw = row[answer_col]

    agg_pred, votes, outs = run_self_consistency(problem_text, k=K_SAMPLES, temp=TEMP_SC, top_p=TOP_P_SC)
    # optional verifier to correct aggregates
    final_pred = verify_with_model(problem_text, agg_pred)
    is_correct, gold_f = compare_gold_pred(gold_raw, final_pred)

    # Quick visibility for the single-row case
    print("\n--- DEBUG ---")
    print("Problem:", problem_text)
    print("Gold:", gold_raw, "(parsed:", gold_f, ")")
    print("Votes:", votes)
    print("Aggregated:", agg_pred, "-> Verified:", final_pred)
    print("Correct?:", is_correct)
    print("-----------\n")

    results.append({
        "index": idx,
        "problem": problem_text,
        "gold_answer": gold_raw,
        "predicted_agg": agg_pred,
        "predicted_final": final_pred,
        "correct": is_correct,
        "k": K_SAMPLES,
        "votes_float": votes,
        "sample_output_1": outs[0] if outs else ""
    })

# =========================
# Summary and Save Results
# =========================
results_df = pd.DataFrame(results)
accuracy = (results_df["correct"].sum() / len(results_df)) * 100

print("\n📊 CoT + Self-Consistency + Verifier Summary")
print("-------------------------------------------------------")
print(f"Model:            {MODEL_ID}")
print(f"Total Questions:  {len(results_df)}")
print(f"K (samples/item): {K_SAMPLES}")
print(f"Correct Count:    {results_df['correct'].sum()}")
print(f"Accuracy:         {accuracy:.2f}%")
print("-------------------------------------------------------")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_name = f"svamp_cot_sc_verify_{MODEL_ID}_N{len(results_df)}_k{K_SAMPLES}_{timestamp}.csv"
results_df.to_csv(csv_name, index=False)
print(f"✅ Saved results to: {csv_name}")
