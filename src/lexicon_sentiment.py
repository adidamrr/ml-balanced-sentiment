import math
import re
from collections import Counter

LABELS = ("positive", "negative", "neutral")

POSITIVE_PATTERNS = (
    "отлич",
    "хорош",
    "понят",
    "интерес",
    "полез",
    "крут",
    "кайф",
    "спасиб",
    "респект",
    "нрав",
    "удоб",
    "структур",
    "легк",
    "ясн",
    "люблю",
    "помог",
    "адекват",
    "зачет",
    "плюс",
    "good",
    "great",
    "excellent",
    "useful",
    "clear",
    "interesting",
    "like",
    "liked",
    "love",
    "perfect",
    "helpful",
    "thanks",
)

NEGATIVE_PATTERNS = (
    "плох",
    "ужас",
    "непонят",
    "сложн",
    "тяжел",
    "перегруж",
    "опазд",
    "неудоб",
    "бесполез",
    "бред",
    "спам",
    "флуд",
    "странн",
    "проблем",
    "минус",
    "скуч",
    "хаос",
    "ошиб",
    "душн",
    "болезн",
    "убива",
    "bad",
    "awful",
    "terrible",
    "useless",
    "hard",
    "confusing",
    "boring",
    "problem",
    "difficult",
    "unclear",
    "overloaded",
    "late",
)

NEGATIONS = {"не", "ни", "нет", "not", "no", "never", "pas", "nunca"}
NEUTRAL_PATTERNS = {"обычный", "обычно", "норм", "нейтрально", "средне", "без", "комментариев", "ok", "okay"}
POSITIVE_EMOJIS = {"🙂", "😊", "😀", "😁", "😍", "👍", "❤", "❤️", ":)", ":-)"}
NEGATIVE_EMOJIS = {"😞", "😢", "😭", "👎", "💩", ":(", ":-("}


def tokenize(text):
    return re.findall(r"[a-zа-яё]+", str(text).lower())


def rating_score(text):
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*/\s*(\d+(?:[.,]\d+)?)", str(text))
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    maximum = float(match.group(2).replace(",", "."))
    if maximum <= 0:
        return None
    ratio = value / maximum
    if ratio >= 0.75:
        return 1.0
    if ratio <= 0.4:
        return -1.0
    return 0.0


def emoji_score(text):
    value = str(text).strip()
    if value in POSITIVE_EMOJIS:
        return 1.0
    if value in NEGATIVE_EMOJIS:
        return -1.0
    return None


def is_valid_review(text):
    if rating_score(text) is not None:
        return True
    if emoji_score(text) is not None:
        return True
    tokens = tokenize(text)
    if not tokens:
        return False
    if len(tokens) == 1 and len(tokens[0]) <= 1:
        return False
    return True


def token_polarity(token):
    for pattern in NEGATIVE_PATTERNS:
        if pattern in token:
            return -1
    for pattern in POSITIVE_PATTERNS:
        if pattern in token:
            return 1
    return 0


def review_score(text):
    rating = rating_score(text)
    if rating is not None:
        return rating
    emoji = emoji_score(text)
    if emoji is not None:
        return emoji
    tokens = tokenize(text)
    score = 0
    hits = 0
    for i, token in enumerate(tokens):
        polarity = token_polarity(token)
        if polarity == 0:
            continue
        if i > 0 and tokens[i - 1] in NEGATIONS:
            polarity *= -1
        score += polarity
        hits += 1
    if hits == 0:
        return 0.0
    return score / math.sqrt(hits)


def classify_review(text, threshold=0.75):
    if not is_valid_review(text):
        return None
    score = review_score(text)
    if abs(score) < threshold:
        return "neutral"
    if score > 0:
        return "positive"
    return "negative"


def rounded_distribution(counts):
    total = sum(counts.values())
    if total == 0:
        return {label: 0.0 for label in LABELS}
    raw = {label: counts.get(label, 0) * 100 / total for label in LABELS}
    rounded = {label: round(value, 1) for label, value in raw.items()}
    diff = round(100.0 - sum(rounded.values()), 1)
    if diff:
        biggest = max(LABELS, key=lambda x: rounded[x])
        rounded[biggest] = round(rounded[biggest] + diff, 1)
    return rounded


def improved_distribution(reviews):
    labels = [classify_review(review) for review in reviews]
    labels = [label for label in labels if label is not None]
    counts = Counter(labels)
    return rounded_distribution(counts), {label: counts.get(label, 0) for label in LABELS}


def analyze_reviews(reviews):
    reviews = list(reviews)
    distribution, label_counts = improved_distribution(reviews)
    valid_count = sum(1 for review in reviews if is_valid_review(review))
    return {
        "review_count": valid_count,
        "distribution": distribution,
        "label_counts": label_counts,
    }
