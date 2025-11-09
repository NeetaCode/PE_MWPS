# =========================
# Formula/KB-RAG for SVAMP (MiniMathKB, retry-safe)
# =========================
!pip -q install datasets pandas rank-bm25 requests

import os, re, json, time, numpy as np, pandas as pd, requests
from datasets import load_dataset
from rank_bm25 import BM25Okapi

# --- Mistral API setup ---
os.environ.setdefault("OPENAI_BASE_URL", "https://api.mistral.ai/v1")
os.environ.setdefault("OPENAI_API_KEY",  "hYnJKM53YmlSWDwjdiXGdTWos4VRrgHd")  # <-- put key
MODEL_ID = "mistral-small"

# ---- 429-safe chat call ----
def call_mistral_chat(model, prompt, temperature=0.2, max_retries=6):
    base = os.environ["OPENAI_BASE_URL"]; key = os.environ["OPENAI_API_KEY"]
    url = f"{base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": model, "temperature": temperature,
            "messages": [
                {"role": "system", "content": "You are a careful math assistant. Provide concise reasoning and a final numeric answer."},
                {"role": "user", "content": prompt},
            ]}
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=120)
            if r.status_code == 429:
                wait = min(15, 2 + attempt * 3)
                print(f"⚠️ Rate limit (429). Retrying in {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(10, 2 + attempt * 2)
                print(f"⚠️ Error {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return f"[ERR] {e}"

def clean_one_word_number(text):
    if not text or not isinstance(text, str): return "NaN"
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if nums: return nums[-1]
    toks = re.findall(r"[A-Za-z0-9\-\+\.]+", text.strip())
    return toks[-1] if toks else "NaN"

# --- Load SVAMP ---
svamp = load_dataset("ChilleD/SVAMP", split="train")
svamp_df = pd.DataFrame(svamp)

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
      "formula":"avg = (Σ w_i * x_i) / (Σ w_i)", "tip":"Use weights (counts or proportions)."},
]

# Build BM25 index
kb_docs = [ (k["name"] + " " + k["triggers"] + " " + k["formula"] + " " + k["tip"]).lower() for k in KB ]
bm25_kb = BM25Okapi([re.findall(r"\w+", d) for d in kb_docs])

def kb_lookup(query_text, k=5):
    q = re.findall(r"\w+", query_text.lower())
    scores = bm25_kb.get_scores(q)
    top = np.argsort(-scores)[:k]
    return [KB[i] for i in top]

def prompt_formula_kb_rag(q, kb_entries):
    kb_lines = [f"- {k['name']}: {k['formula']} ({k['tip']})" for k in kb_entries]
    kb_text = "\n".join(kb_lines)
    return f"""Use the following math knowledge base to solve the problem.

Knowledge Base:
{kb_text}

Problem:
{q}

Explain briefly and end with:
Final Answer: <number>"""

# --- Run on first 10 SVAMP questions ---
results = []
N_EVAL = 100   # <--- change this number to evaluate more rows later

for idx in range(min(N_EVAL, len(svamp_df))):
    row = svamp_df.iloc[idx]
    question_text = (str(row.get("Body","")) + " " + str(row.get("Question",""))).strip()
    gold = str(row.get("Answer",""))

    kb_hits = kb_lookup(question_text, k=5)
    prompt = prompt_formula_kb_rag(question_text, kb_hits)
    raw = call_mistral_chat(MODEL_ID, prompt)
    pred = clean_one_word_number(raw)

    results.append({
        "Dataset": "SVAMP",
        "Idx": idx,
        "RAG_Type": "Formula/KB-RAG (MiniMathKB)",
        "Model": MODEL_ID,
        "Gold": gold,
        "Predicted": pred,
        # "Raw": raw
    })

    print(f"✅ Done: Row {idx} (Gold={gold}, Pred={pred})")
    time.sleep(3)   # small delay to avoid 429s

# --- Convert to DataFrame and save ---
df_out = pd.DataFrame(results)
print(f"\n✅ Finished {len(df_out)} SVAMP rows.")
display(df_out)

# Save to CSV (optional)
df_out.to_csv("svamp_formula_rag_results.csv", index=False)
print("📁 Saved results to svamp_formula_rag_results.csv")
