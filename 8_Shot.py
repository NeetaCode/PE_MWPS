# =========================
# Colab Setup: Deps
# =========================
!pip -q install datasets pandas requests

# =========================
# Imports
# =========================
from datasets import load_dataset
import pandas as pd
import requests, os, re, time, random
from datetime import datetime

# =========================
# (Optional) Google Drive mount + output dir
# =========================
OUTPUT_DIR = None
try:
    import google.colab  # noqa: F401
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_DIR = "/content/drive/MyDrive/llm_results"
except Exception:
    OUTPUT_DIR = os.getcwd()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Config: Mistral API (OpenAI-compatible)
# =========================
os.environ.setdefault("OPENAI_BASE_URL", "https://api.mistral.ai/v1")
os.environ.setdefault("OPENAI_API_KEY", "YOUR_API_KEY_HERE")  # <-- replace or set via env

MODEL_ID = "mistral-small"   # or "mistral-medium" / "mistral-large-latest"
N_SHOTS = 8

# =========================
# Utilities
# =========================
def clean_one_word_number(text: str) -> str:
    """Extract a single numeric answer from model output.
       Prefers 'FINAL_ANSWER:'; else falls back to the last number found."""
    if not text:
        return "NaN"
    m = re.search(r"FINAL_ANSWER\s*:\s*([-+]?\d*\.?\d+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    return nums[-1] if nums else "NaN"

def call_mistral_chat(model: str, prompt: str, temperature=0.0, max_retries=5, timeout_s=60):
    """Calls Mistral /v1/chat/completions (OpenAI-compatible) with simple retry/backoff."""
    base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    key  = os.environ.get("OPENAI_API_KEY", "")
    if not base or not key:
        raise RuntimeError("Missing OPENAI_BASE_URL or OPENAI_API_KEY")

    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "You are a careful math assistant. Provide concise reasoning and a final numeric answer."},
            {"role": "user", "content": prompt},
        ],
    }

    for attempt in range(max_retries):
        resp = requests.post(url, headers=headers, json=body, timeout=timeout_s)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 * (attempt + 1))
            continue
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    raise RuntimeError("Max retries exceeded calling Mistral API")

def to_float_safe(x: str):
    try:
        return float(x)
    except Exception:
        return None

def build_8shot_prompt(target_problem: str, exemplars: list) -> str:
    """Build an 8-shot prompt: 8 QA exemplars (problem + gold numeric answer), then the target problem."""
    header = (
        "You solve grade-school math word problems.\n"
        "For each problem, show brief reasoning (1–3 short lines) and end with:\n"
        "FINAL_ANSWER: <number>\n\n"
        "Here are examples:\n"
    )
    ex_txt = []
    for k, ex in enumerate(exemplars, start=1):
        ex_txt.append(
            f"Example {k}:\n"
            f"Problem:\n{ex['problem']}\n"
            f"Answer:\nFINAL_ANSWER: {ex['answer']}\n"
            f"---\n"
        )
    footer = (
        "Now solve the next problem in the same format.\n\n"
        f"Problem:\n{target_problem}\n\n"
        "Answer with your short reasoning and then:\n"
        "FINAL_ANSWER: <number>"
    )
    return header + "".join(ex_txt) + footer

# =========================
# Load SVAMP
# =========================
svamp = load_dataset("ChilleD/SVAMP")
train_df = pd.DataFrame(svamp["train"])
assert len(train_df) > 0, "SVAMP train split is empty."

# Normalize column names
cols = {c.lower(): c for c in train_df.columns}
body_col     = cols.get("body", None)
question_col = cols.get("question", None)
answer_col   = cols.get("answer", None)
if question_col is None or answer_col is None:
    raise RuntimeError(f"Expected columns not found. Got: {list(train_df.columns)}")

# =========================
# Define evaluation set (first 100) and exemplar pool (next rows)
# =========================
eval_df = train_df.head(100).copy()

# Exemplar pool: rows 100..(100+N_SHOTS-1). If dataset is small, fallback to random sample excluding first 100.
if len(train_df) >= 100 + N_SHOTS:
    ex_pool = train_df.iloc[100:100+N_SHOTS]
else:
    rest = train_df.iloc[100:]
    ex_pool = rest.sample(n=min(N_SHOTS, len(rest)), random_state=42)

# Build exemplar dicts (problem text + gold answer)
exemplars = []
for _, r in ex_pool.iterrows():
    b = str(r[body_col]) if body_col in r and pd.notna(r[body_col]) else ""
    q = str(r[question_col])
    g = str(r[answer_col]).strip()
    problem_text = (b + " " + q).strip()
    exemplars.append({"problem": problem_text, "answer": g})

# Safety: if we somehow have < N_SHOTS exemplars, proceed with what we have.
print(f"Using {len(exemplars)} exemplars for few-shot prompting.")

# =========================
# Run 8-shot prompting on first 100 items
# =========================
rows = []
correct_count = 0

print("⚙️ Running 8-shot evaluation on first 100 SVAMP items...\n")

for idx, row in eval_df.iterrows():
    body = str(row.get(body_col, "")) if body_col else ""
    question = str(row[question_col])
    gold = str(row[answer_col]).strip()
    problem_text = (body + " " + question).strip()

    prompt = build_8shot_prompt(problem_text, exemplars)

    try:
        pred_raw = call_mistral_chat(MODEL_ID, prompt, temperature=0.0)
    except Exception as e:
        pred_raw = f"[ERROR] {e}"

    pred_num = clean_one_word_number(pred_raw)
    gold_f, pred_f = to_float_safe(gold), to_float_safe(pred_num)
    is_correct = abs(gold_f - pred_f) < 1e-6 if (gold_f is not None and pred_f is not None) else (gold.strip() == pred_num.strip())

    correct_count += int(is_correct)
    rows.append({
        "idx": idx,
        "Problem": problem_text,
        "Gold": gold,
        "Pred_raw": pred_raw,
        "Pred_num": pred_num,
        "Correct": is_correct
    })

    if (idx + 1) % 10 == 0:
        print(f"✅ Processed {idx + 1} items...")

results = pd.DataFrame(rows)
acc = (correct_count / len(results)) * 100 if len(results) else 0.0

# =========================
# Save CSV with timestamp (Drive if mounted, else local)
# =========================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
safe_model = MODEL_ID.replace("/", "_")
out_path = os.path.join(OUTPUT_DIR, f"svamp_{N_SHOTS}shot_{safe_model}_{ts}.csv")
results.to_csv(out_path, index=False)

# =========================
# Simple Clean Summary (no *70 lines)
# =========================
print("\n📊 8-Shot SVAMP Evaluation Summary")
print("----------------------------------------")
print(f"Model:         {MODEL_ID}")
print(f"Shots:         {N_SHOTS}")
print(f"Total items:   {len(results)}")
print(f"Correct:       {correct_count}")
print(f"Accuracy:      {acc:.2f}%")
#print(f"Saved to CSV:  {out_path}\n")

# Preview
#results.head(5)
