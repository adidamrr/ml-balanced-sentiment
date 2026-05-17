from collections import Counter

from src.lexicon_sentiment import LABELS, NEGATIONS, is_valid_review, rounded_distribution, token_polarity, tokenize


def baseline_distribution(reviews, threshold=0.05):
    counts = Counter()
    for review in reviews:
        if not is_valid_review(review):
            continue
        tokens = tokenize(review)
        for i, token in enumerate(tokens):
            polarity = token_polarity(token)
            if polarity == 0:
                continue
            if i > 0 and tokens[i - 1] in NEGATIONS:
                polarity *= -1
            if polarity > 0:
                counts["positive"] += 1
            elif polarity < 0:
                counts["negative"] += 1
        if not any(token_polarity(token) for token in tokens):
            counts["neutral"] += 1
    total = sum(counts.values())
    if total == 0:
        return {label: 0.0 for label in LABELS}
    return rounded_distribution(counts)


def analyze_reviews_baseline(reviews):
    reviews = list(reviews)
    valid_count = sum(1 for review in reviews if is_valid_review(review))
    return {
        "review_count": valid_count,
        "distribution": baseline_distribution(reviews),
    }
