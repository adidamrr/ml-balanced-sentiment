import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline_sentiment import baseline_distribution
from src.ensemble_sentiment import analyze_reviews_ensemble, create_groq_client
from src.lexicon_sentiment import improved_distribution


def make_scenarios():
    positive_short = "Отличный курс, все понятно и полезно"
    negative_long = (
        "Курс ужасный непонятный сложный тяжелый перегруженный неудобный "
        "скучный плохой проблемный "
    ) * 8
    neutral = "Обычный курс без сильных плюсов и минусов"
    positive_equal = "Отличный понятный полезный интересный удобный курс"
    negative_equal = "Плохой непонятный тяжелый скучный неудобный курс"
    mixed_long = (
        "Лекции интересные и преподаватель объясняет понятно, но домашние задания "
        "тяжелые, дедлайны неудобные, а организация местами непонятная. "
    ) * 6
    return {
        "A: 45 positive / 45 negative / 10 neutral, negative is longer": (
            [positive_short] * 45 + [negative_long] * 45 + [neutral] * 10
        ),
        "B: 60 positive / 25 negative / 15 neutral, equal length": (
            [positive_equal] * 60 + [negative_equal] * 25 + [neutral] * 15
        ),
        "C: long mixed + neutral": [mixed_long] * 20 + [neutral] * 20,
    }


def make_ensemble_client():
    if not os.environ.get("GROQ_API_KEY"):
        return None
    return create_groq_client()


def compare_reviews(reviews, scenario="custom", client=None):
    baseline = baseline_distribution(reviews)
    lexicon_distribution, lexicon_counts = improved_distribution(reviews)
    ensemble = analyze_reviews_ensemble(reviews, client=client)
    rows = [
        {
            "scenario": scenario,
            "approach": "baseline",
            "positive": baseline["positive"],
            "negative": baseline["negative"],
            "neutral": baseline["neutral"],
            "positive_count": None,
            "negative_count": None,
            "neutral_count": None,
            "llm_used": 0,
            "fallback_count": 0,
        },
        {
            "scenario": scenario,
            "approach": "lexicon",
            "positive": lexicon_distribution["positive"],
            "negative": lexicon_distribution["negative"],
            "neutral": lexicon_distribution["neutral"],
            "positive_count": lexicon_counts["positive"],
            "negative_count": lexicon_counts["negative"],
            "neutral_count": lexicon_counts["neutral"],
            "llm_used": 0,
            "fallback_count": 0,
        },
        {
            "scenario": scenario,
            "approach": "ensemble",
            "positive": ensemble["distribution"]["positive"],
            "negative": ensemble["distribution"]["negative"],
            "neutral": ensemble["distribution"]["neutral"],
            "positive_count": ensemble["label_counts"]["positive"],
            "negative_count": ensemble["label_counts"]["negative"],
            "neutral_count": ensemble["label_counts"]["neutral"],
            "llm_used": ensemble["llm_used"],
            "fallback_count": ensemble["fallback_count"],
        },
    ]
    return pd.DataFrame(rows)


def compare_scenarios(client=None):
    frames = []
    for scenario, reviews in make_scenarios().items():
        frames.append(compare_reviews(reviews, scenario=scenario, client=client))
    return pd.concat(frames, ignore_index=True)


def make_sensitivity_table(client=None):
    positive = "Отличный курс, все понятно и полезно"
    neutral = "Обычный курс без сильных плюсов и минусов"
    rows = []
    for multiplier in [1, 2, 4, 8, 12]:
        negative = (
            "Курс ужасный непонятный сложный тяжелый перегруженный неудобный "
            "скучный плохой проблемный "
        ) * multiplier
        reviews = [positive] * 45 + [negative] * 45 + [neutral] * 10
        table = compare_reviews(reviews, scenario=f"negative length x{multiplier}", client=client)
        table.insert(1, "negative_length_multiplier", multiplier)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def main():
    client = make_ensemble_client()
    print("Comparison")
    print(compare_scenarios(client=client).to_string(index=False))
    print()
    print("Sensitivity")
    print(make_sensitivity_table(client=client).to_string(index=False))


if __name__ == "__main__":
    main()
