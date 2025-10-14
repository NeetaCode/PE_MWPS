import os
import re
import time
import json
import argparse
import pandas as pd
import requests

# =============================
# Config (defaults; override via CLI)
# =============================
DEFAULT_INPUT = "hf://datasets/hiyouga/geometry3k/data/train-00000-of-00001.parquet"
DEFAULT_BATCH = 100

# =============================
# Final-answer parser (shared)
# =============================
BOXED_RE       = re.compile(r"\\boxed\{\s*([^}]*)\s*\}")
INLINE_MATH_RE = re.compile(r"\$([^$]+)\$")
NUMERICY_RE    = re.compile(
    r"(?:-?\d+(?:\.\d+)?)"                  # 12, 12.3, -4.5
    r"|(?:-?\d+\s*/\s*\d+)"                 # 3/5
    r"|(?:\\frac\{[^{}]+\}\{[^{}]+\})"      # \frac{a}{b}
    r"|(?:-?\d*\s*\\sqrt\{\s*[^{}]+\s*\})"  # 2\sqrt{221}
    r"|(?:\\sqrt\{\s*[^{}]+\s*\})"          # \sqrt{221}
)

def _clean(s: str) -> str:
    s = s.strip().strip('"\'')

    # strip labels
    s = re.sub(r"^(final\s+answer\s+is[:\-\s]*)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(answer\s*[:\-\s]*)", "", s, flags=re.IGNORECASE)

    # trailing noise
    s = re.sub(r"[ \t\n\r]+$", "", s)
    s = re.sub(r"[.,;:\s]+$", "", s)
    return s

def extract_final_only(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    # 1) \boxed{...}
    m = BOXED_RE.search(text)
    if m:
        return _clean(m.group(1))
    # 2) last inline math
    maths = INLINE_MATH_RE.findall(text)
    if maths:
        return _clean(maths[-1])
    # 3) last mathy token
    tokens = list(NUMERICY_RE.finditer(text))
    if tokens:
        return _clean(tokens[-1].group(0))
    # 4) else first line, cleaned
    return _clean(text.splitlines()[0])

SYSTEM_INSTRUCTION = (
    "You are a meticulous math solver. "
    "Given a geometry problem, output the FINAL ANSWER ONLY—no steps, no prose. "
    "If units are explicit, include them; otherwise omit. "
    "If insufficient information, output exactly: INSUFFICIENT"
)

def build_user_prompt(q: str) -> str:
    q_clean = q.replace("<image>", "").strip()
    return f"{q_clean}\n\nFinal answer only:"

# =============================
# Provider: OLLAMA (local)
# =============================
def ask_ollama(model: str, prompt: str, url: str = "http://127.0.0.1:11434/api/generate", retries: int = 3):
    payload = {"model": model, "prompt": prompt, "stream": False}
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            # Ollama returns {"response": "..."}
            return (data.get("response") or "").strip()
        except Exception as e:
            if attempt == retries - 1:
                return f"ERROR: {e}"
            time.sleep(2 ** attempt)

# =============================
# Provider: DEEPSEEK (cloud)
#   - API: Chat Completions compatible (OpenAI-style)
#   - Endpoint: https://api.deepseek.com/chat/completions
#   - Model: deepseek-chat  (or deepseek-reasoner)
# =============================
def ask_deepseek(model: str, prompt: str, retries: int = 3):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "ERROR: Missing DEEPSEEK_API_KEY"
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        # keep defaults simple; temperature optional
        "stream": False
    }
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
            data = r.json()
            # OpenAI-style: choices[0].message.content
            return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        except Exception as e:
            if attempt == retries - 1:
                return f"ERROR: {e}"
            time.sleep(2 ** attempt)

# =============================
# Provider: GEMINI (cloud)
#   - Library: google-generativeai
#   - Model: gemini-1.5-pro / gemini-1.5-flash
# =============================
def ask_gemini(model: str, prompt: str, retries: int = 3):
    try:
        import google.generativeai as genai
    except Exception as e:
        return "ERROR: google-generativeai not installed (pip install google-generativeai)"
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "ERROR: Missing GOOGLE_API_KEY"
    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model)
    for attempt in range(retries):
        try:
            resp = gmodel.generate_content(
                contents=[{"role": "user", "parts": [SYSTEM_INSTRUCTION + "\n" + prompt]}],
                safety_settings=None,
                generation_config={"candidate_count": 1},
            )
            # Gemini returns resp.text
            return (getattr(resp, "text", "") or "").strip()
        except Exception as e:
            if attempt == retries - 1:
                return f"ERROR: {e}"
            time.sleep(2 ** attempt)

# =============================
# Main runner
# =============================
def main():
    parser = argparse.ArgumentParser(description="Geometry3K final-answer-only runner for Ollama / DeepSeek / Gemini")
    parser.add_argument("--provider", required=True, choices=["ollama", "deepseek", "gemini"])
    parser.add_argument("--model", required=True, help="Provider model name (e.g., llama3.1 | deepseek-chat | gemini-1.5-pro)")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Parquet path (HF path ok)")
    parser.add_argument("--output", default=None, help="Output CSV (default auto-named)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    args = parser.parse_args()

    # Output file name
    if args.output:
        output_csv = args.output
    else:
        output_csv = f"geometry3k_{args.provider}_final_only.csv"

    # Load data
    df = pd.read_parquet(args.input)
    questions = df["problem"].tolist()
    ground_truths = df["answer"].tolist() if "answer" in df.columns else [""] * len(questions)

    # Resume logic
    start_index = 0
    if os.path.exists(output_csv):
        saved = pd.read_csv(output_csv)
        start_index = len(saved)
        print(f"Resuming from question index {start_index}")
    end_index = min(start_index + args.batch_size, len(questions))
    print(f"Processing {start_index}..{end_index-1} of {len(questions)} with {args.provider}:{args.model}")

    # Provider selector
    if args.provider == "ollama":
        ask_fn = lambda q: ask_ollama(args.model, build_user_prompt(q))
        col_name = "ollama_final_answer"
    elif args.provider == "deepseek":
        ask_fn = lambda q: ask_deepseek(args.model, build_user_prompt(q))
        col_name = "deepseek_final_answer"
    else:  # gemini
        ask_fn = lambda q: ask_gemini(args.model, build_user_prompt(q))
        col_name = "gemini_final_answer"

    # Loop & collect
    new_rows = []
    for i in range(start_index, end_index):
        q = questions[i]
        preview = q[:70].replace("\n", " ")
        print(f"Asking {i+1}/{len(questions)}: {preview}...")
        raw = ask_fn(q)
        final_only = extract_final_only(raw)
        new_rows.append({
            "problem": q,
            "ground_truth": ground_truths[i],
            col_name: final_only
        })

    # Append or create; save only the three columns
    new_df = pd.DataFrame(new_rows)
    if os.path.exists(output_csv):
        old_df = pd.read_csv(output_csv)
        out_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out_df = new_df

    out_df[["problem", "ground_truth", col_name]].to_csv(output_csv, index=False)
    print(f"✅ Saved questions {start_index+1} to {end_index} to {output_csv}")

if __name__ == "__main__":
    main()
