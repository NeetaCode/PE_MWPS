# =========================
# Colab Setup: Deps
# =========================
!pip -q install datasets pandas requests numpy

# =========================
# Imports
# =========================
from datasets import load_dataset
import pandas as pd, numpy as np, requests, os, re, time, random
from datetime import datetime

# =========================
# Config: Mistral (OpenAI-compatible)
# =========================
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
MISTRAL_API_KEY  = "hYnJKM53YmlSWDwjdiXGdTWos4VRrgHd"      
MISTRAL_MODEL    = "mistral-small"

os.environ["OPENAI_BASE_URL"] = MISTRAL_BASE_URL
os.environ["OPENAI_API_KEY"]  = MISTRAL_API_KEY
MODEL_ID = MISTRAL_MODEL

TEMP, TOP_P, MAX_TOKENS = 0.0, 1.0, 512
random.seed(42); np.random.seed(42)

# =========================
# Parsing helpers
# =========================
FINAL_PATTERNS = [
    r"<ans>\s*([-+]?\d*\.?\d+)\s*</ans>",
    r"FINAL_ANSWER\s*[:：]\s*([-+]?\d*\.?\d+)",
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

def nearly_equal(a,b,tol=1e-6): return abs(a-b)<=tol

# =========================
# Chain-of-Thought prompt
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
# Chat call
# =========================
def call_chat(model, prompt, temperature=0.0, top_p=1.0,
              max_tokens=512, max_retries=4, timeout_s=60):
    base, key = os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_API_KEY"]
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model, "temperature": temperature, "top_p": top_p,
        "max_tokens": max_tokens,
        "messages": [
            {"role":"system",
             "content":("You are a careful math assistant. Solve step by step with concise logic. "
                        "On the LAST two lines, output exactly both forms:\n"
                        "<ans>NUMBER</ans>\nFINAL_ANSWER: NUMBER")},
            {"role":"user","content":prompt}
        ]
    }
    for attempt in range(max_retries):
        try:
            r=requests.post(url,headers=headers,json=body,timeout=timeout_s)
            if r.status_code==200:
                return r.json()["choices"][0]["message"]["content"]
            if r.status_code in (429,500,502,503,504):
                time.sleep((2**attempt)+random.uniform(0,0.5)); continue
            return f"[ERROR] API error {r.status_code}: {r.text}"
        except requests.exceptions.RequestException:
            time.sleep((2**attempt)+random.uniform(0,0.5))
    return "[ERROR] Max retries exceeded"

# =========================
# Load SVAMP & Evaluate first 100 rows
# =========================
svamp = load_dataset("ChilleD/SVAMP")
train_df = pd.DataFrame(svamp["train"])
cols = {c.lower():c for c in train_df.columns}
body_col, question_col, answer_col = cols.get("body"), cols.get("question"), cols.get("answer") or cols.get("ans")

results = []

for idx, row in train_df.head(100).iterrows():
    problem_text = f"{row.get(body_col,'')} {row[question_col]}".strip()
    gold_raw = row[answer_col]
    gold_norm = normalize_numeric_string(gold_raw)
    gold_f = to_float_safe(gold_norm)

    prompt = build_cot_prompt(problem_text)
    out = call_chat(MODEL_ID, prompt, temperature=TEMP, top_p=TOP_P, max_tokens=MAX_TOKENS)

    pred_num = normalize_numeric_string(clean_one_word_number(out))
    pred_f = to_float_safe(pred_num)
    is_correct = (gold_f is not None and pred_f is not None and nearly_equal(gold_f,pred_f)) or \
                 (gold_f is None and gold_norm.strip()==pred_num.strip())

    results.append({
        "index": idx,
        "problem": problem_text,
        "gold_answer": gold_raw,
        "predicted": pred_num,
        "is_correct": is_correct,
        "model_output": out
    })

# =========================
# Summary and Save Results
# =========================
results_df = pd.DataFrame(results)
accuracy = (results_df["is_correct"].sum() / len(results_df)) * 100

print("\n📊 CoT Evaluation Summary (First 100 Questions)")
print("------------------------------------------------")
print(f"Model:          {MODEL_ID}")
print(f"Total Questions:{len(results_df)}")
print(f"Correct Count:  {results_df['is_correct'].sum()}")
print(f"Accuracy:       {accuracy:.2f}%")
print("------------------------------------------------")

# Optionally save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_name = f"svamp_cot_100rows_{MODEL_ID}_{timestamp}.csv"
results_df.to_csv(csv_name, index=False)
print(f"✅ Saved results to: {csv_name}")
