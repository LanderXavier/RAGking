"""
core/llm.py — LLM-as-a-Judge scoring and embedding cosine similarity.
"""

import os
import json
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ui.helpers import warn, _warn_once, _looks_like_placeholder_secret
from core.config import state

# ══════════════════════════════════════════════════════════════════════════════
#  JUDGE PROMPT
# ══════════════════════════════════════════════════════════════════════════════
JUDGE_PROMPT = """You are an expert factual evaluator assessing responses in a strict legal context. 
Your primary directives are:
- MISTAKEN FACT: Providing incorrect or hallucinated information is the WORST possible outcome.
- IGNORANCE IS SAFER: Safely admitting lack of information is better than lying.
- CONCISENESS: Providing excessive, unasked-for information (verbosity/hallucination of scope) must be penalized.

Score from 1 to 5 using this strict rubric:
1 → FATAL ERROR: Contains major factual errors, hallucinates laws, or contradicts the 'Correct Answer'.
2 → NO INFO / REFUSAL: The model safely admits it lacks information, says "I don't know", or provides no relevant answer without lying.
3 → PARTIALLY CORRECT or BLOATED: The core answer is partially correct but misses key details, OR the answer is correct but contains significant extra, unrequested information.
4 → MOSTLY CORRECT: Factually aligned with the 'Correct Answer', but slightly verbose or missing a very minor detail.
5 → PERFECT: Fully correct, precise, and highly concise. Answers exactly what was asked without unnecessary fluff.

Return ONLY JSON:
{{"score": <1-5>}}

Question: {question}
Correct Answer: {correct}
Generated Answer: {generated}
"""

# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════
def get_embedding(text: str, provider: str, model: str, base_url: str = None):
    """Return embedding vector for a text using the configured provider."""
    if not text or not isinstance(text, str):
        return None
    try:
        if provider == "openai":
            if _looks_like_placeholder_secret(os.getenv("OPENAI_API_KEY")):
                _warn_once(("openai", "embedding_key"), "Embedding disabled: set OPENAI_API_KEY in .env")
                return None
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "deepseek":
            if _looks_like_placeholder_secret(os.getenv("DEEPSEEK_API_KEY")):
                _warn_once(("deepseek", "embedding_key"), "Embedding disabled: set DEEPSEEK_API_KEY in .env")
                return None
            client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
        elif provider == "local":
            client = OpenAI(base_url=base_url or state["llm"]["local_base_url"], api_key="not-needed")
        else:
            return None

        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding

    except Exception as e:
        warn(f"Embedding error: {e}")
        return None


def compute_similarity(emb1, emb2) -> float:
    """Cosine similarity between two embedding vectors."""
    if emb1 is None or emb2 is None:
        return 0.0
    emb1 = np.array(emb1).reshape(1, -1)
    emb2 = np.array(emb2).reshape(1, -1)
    return float(cosine_similarity(emb1, emb2)[0][0])


# ══════════════════════════════════════════════════════════════════════════════
#  JUDGE
# ══════════════════════════════════════════════════════════════════════════════
def call_judge(question: str, correct_answer: str, generated_answer: str,
               provider: str, model: str, base_url: str = None) -> int:
    """Call LLM judge and return integer score 1–5."""
    if not generated_answer or not isinstance(generated_answer, str) or generated_answer.strip() == "":
        return 1

    prompt = JUDGE_PROMPT.format(
        question=question,
        correct=correct_answer,
        generated=generated_answer,
    )
    judge_temp = float(state.get("llm", {}).get("judge_temperature", 0.0))

    try:
        if provider == "openai":
            if _looks_like_placeholder_secret(os.getenv("OPENAI_API_KEY")):
                _warn_once(("openai", "judge_key"), "Judge disabled: set OPENAI_API_KEY in .env")
                return 1
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=judge_temp,
                response_format={"type": "json_object"},
            )
        elif provider == "deepseek":
            if _looks_like_placeholder_secret(os.getenv("DEEPSEEK_API_KEY")):
                _warn_once(("deepseek", "judge_key"), "Judge disabled: set DEEPSEEK_API_KEY in .env")
                return 1
            client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=judge_temp,
            )
        elif provider == "local":
            client = OpenAI(base_url=base_url or state["llm"]["local_base_url"], api_key="not-needed")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=judge_temp,
            )
        else:
            return 1

        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        score = data.get("score", 1)
        return int(score) if isinstance(score, (int, float)) and 1 <= score <= 5 else 1

    except Exception as e:
        warn(f"Judge error: {e}")
        return 1


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def compute_correctness_and_similarity(df, llm_cfg: dict, progress: bool = True):
    """Add 'Correctness' and 'String Similarity' columns to a DataFrame copy."""
    import pandas as pd
    df = df.copy()
    correctness, similarity = [], []

    embed_provider = llm_cfg["embedding_provider"]
    embed_model    = llm_cfg["embedding_model"]
    judge_provider = llm_cfg["judge_provider"]
    judge_model    = llm_cfg["judge_model"]
    local_url      = llm_cfg["local_base_url"]

    iterator = (
        tqdm(df.iterrows(), total=len(df), desc="Computing metrics")
        if progress else df.iterrows()
    )

    for _, row in iterator:
        question  = str(row["Question"])
        correct   = str(row["Correct Answer"])
        generated = str(row["Generated Answer"])

        score = call_judge(question, correct, generated, judge_provider, judge_model, local_url)
        emb_c = get_embedding(correct,   embed_provider, embed_model, local_url)
        emb_g = get_embedding(generated, embed_provider, embed_model, local_url)
        sim   = compute_similarity(emb_c, emb_g)

        correctness.append(score)
        similarity.append(sim)

    df["Correctness"]      = correctness
    df["String Similarity"] = similarity
    return df
