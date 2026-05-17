import json
import os
import re
from collections import Counter

from openai import OpenAI

from src.lexicon_sentiment import (
    LABELS,
    classify_review,
    is_valid_review,
    review_score,
    rounded_distribution,
    token_polarity,
    tokenize,
)

DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MIXED_MARKERS = {"но", "однако", "зато", "but", "however", "although", "though"}
SARCASM_MARKERS = {"ага", "конечно", "ну да", "as if", "yeah right"}
CONFIDENT_SCORE = 1.2
LONG_REVIEW_WORDS = 80
MIN_LLM_CONFIDENCE = 0.6


def create_groq_client(api_key=None):
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY before running Groq LLM")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def has_mixed_markers(text):
    tokens = set(tokenize(text))
    lowered = str(text).lower()
    return bool(tokens & MIXED_MARKERS) or any(marker in lowered for marker in MIXED_MARKERS)


def has_sarcasm_markers(text):
    lowered = str(text).lower()
    quoted_praise = bool(re.search(r"[\"'«“](отлич|хорош|great|good|perfect)", lowered))
    return quoted_praise or any(marker in lowered for marker in SARCASM_MARKERS)


def has_both_sentiment_sides(text):
    positives = 0
    negatives = 0
    for token in tokenize(text):
        polarity = token_polarity(token)
        if polarity > 0:
            positives += 1
        elif polarity < 0:
            negatives += 1
    return positives > 0 and negatives > 0


def is_english_review(text):
    tokens = tokenize(text)
    if not tokens:
        return False
    latin = sum(1 for token in tokens if re.search(r"[a-z]", token))
    return latin / len(tokens) >= 0.5


def needs_llm_review(text, score=None):
    if not is_valid_review(text):
        return False
    if score is None:
        score = review_score(text)
    tokens = tokenize(text)
    if abs(score) < CONFIDENT_SCORE:
        return True
    if has_mixed_markers(text):
        return True
    if has_both_sentiment_sides(text):
        return True
    if len(tokens) > LONG_REVIEW_WORDS:
        return True
    if is_english_review(text):
        return True
    if has_sarcasm_markers(text):
        return True
    return False


def build_ensemble_llm_messages(review, lexicon_label, score):
    system_prompt = (
        "You are a careful sentiment reviewer for student feedback about courses and teachers. "
        "You receive one review plus a simple lexicon prediction. Your job is to check cases "
        "where the lexicon may fail: sarcasm, mixed pros and cons, weak sentiment, English text, "
        "or long nuanced reviews. Return only JSON with keys label, confidence, and flags. "
        "Allowed labels are positive, negative, neutral. Use neutral for mixed or balanced reviews. "
        "If uncertain, choose neutral."
    )
    examples = [
        (
            "Отличный курс, всё понятно",
            "positive",
            1.5,
            {"label": "positive", "confidence": 0.95, "flags": []},
        ),
        (
            "Курс ужасный, непонятный и плохо организованный",
            "negative",
            -1.7,
            {"label": "negative", "confidence": 0.95, "flags": []},
        ),
        (
            "Лекции интересные, но домашка слишком тяжёлая",
            "neutral",
            0.0,
            {"label": "neutral", "confidence": 0.85, "flags": ["mixed"]},
        ),
        (
            "Ну да, конечно, 'отличная' организация: дедлайны менялись каждый день",
            "positive",
            1.0,
            {"label": "negative", "confidence": 0.8, "flags": ["sarcasm"]},
        ),
        (
            "Not bad, the lectures were useful",
            "positive",
            1.0,
            {"label": "positive", "confidence": 0.9, "flags": ["english"]},
        ),
        (
            "The lectures were clear and useful, but the workload was too heavy and deadlines were confusing.",
            "neutral",
            0.0,
            {"label": "neutral", "confidence": 0.8, "flags": ["mixed", "english"]},
        ),
    ]
    messages = [{"role": "system", "content": system_prompt}]
    for text, label, example_score, answer in examples:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Review:\n{text}\n\n"
                    f"Lexicon label: {label}\n"
                    f"Lexicon score: {example_score:.3f}"
                ),
            }
        )
        messages.append({"role": "assistant", "content": json.dumps(answer)})
    messages.append(
        {
            "role": "user",
            "content": (
                f"Review:\n{review}\n\n"
                f"Lexicon label: {lexicon_label}\n"
                f"Lexicon score: {score:.3f}"
            ),
        }
    )
    return messages


def parse_ensemble_llm_result(text):
    try:
        data = json.loads(str(text))
    except json.JSONDecodeError:
        return None
    label = str(data.get("label", "")).lower().strip()
    if label not in LABELS:
        return None
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    flags = data.get("flags", [])
    if not isinstance(flags, list):
        flags = []
    return {"label": label, "confidence": confidence, "flags": flags}


def classify_review_ensemble(review, client=None, model=DEFAULT_LLM_MODEL):
    if not is_valid_review(review):
        return None, {"source": "ignored", "llm_used": False, "fallback": False}
    score = review_score(review)
    lexicon_label = classify_review(review)
    should_use_llm = needs_llm_review(review, score)
    if not should_use_llm:
        return lexicon_label, {"source": "lexicon", "llm_used": False, "fallback": False}
    if client is None:
        return lexicon_label, {"source": "lexicon_fallback", "llm_used": False, "fallback": True}
    try:
        response = client.chat.completions.create(
            model=model,
            messages=build_ensemble_llm_messages(review, lexicon_label, score),
            temperature=0,
            top_p=1,
            max_tokens=60,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        parsed = parse_ensemble_llm_result(content)
        if parsed is not None:
            if parsed["confidence"] >= MIN_LLM_CONFIDENCE:
                return parsed["label"], {"source": "llm", "llm_used": True, "fallback": False}
            return "neutral", {"source": "llm_low_confidence", "llm_used": True, "fallback": False}
    except Exception:
        pass
    return lexicon_label, {"source": "lexicon_fallback", "llm_used": False, "fallback": True}


def ensemble_distribution(reviews, client=None, model=DEFAULT_LLM_MODEL):
    counts = Counter()
    llm_used = 0
    fallback_count = 0
    for review in reviews:
        label, meta = classify_review_ensemble(review, client=client, model=model)
        if label in LABELS:
            counts[label] += 1
        if meta.get("llm_used"):
            llm_used += 1
        if meta.get("fallback"):
            fallback_count += 1
    return {
        "method": "lexicon first, LLM for uncertain reviews",
        "distribution": rounded_distribution(counts),
        "label_counts": {label: counts.get(label, 0) for label in LABELS},
        "llm_used": llm_used,
        "fallback_count": fallback_count,
    }


def analyze_reviews_ensemble(reviews, client=None, model=DEFAULT_LLM_MODEL):
    return ensemble_distribution(list(reviews), client=client, model=model)
