from src.lexicon_sentiment import analyze_reviews, classify_review, improved_distribution


def test_empty_list_returns_zero_distribution():
    result = analyze_reviews([])

    assert result["review_count"] == 0
    assert result["distribution"] == {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    assert result["label_counts"] == {"positive": 0, "negative": 0, "neutral": 0}


def test_garbage_reviews_are_ignored():
    result = analyze_reviews(["", ".", "a", "🤖"])

    assert result["review_count"] == 0
    assert result["distribution"] == {"positive": 0.0, "negative": 0.0, "neutral": 0.0}


def test_short_positive_review():
    assert classify_review("Хороший преподаватель, всё понятно") == "positive"


def test_short_negative_review():
    assert classify_review("Плохой непонятный курс") == "negative"


def test_neutral_review():
    assert classify_review("Обычный курс без сильных плюсов и минусов") == "neutral"


def test_mixed_review_is_neutral():
    assert classify_review("Лекции интересные, но домашка убивает") == "neutral"


def test_english_reviews_are_classified():
    assert classify_review("Great useful and clear course") == "positive"
    assert classify_review("The course was boring and confusing") == "negative"


def test_negations_are_handled():
    assert classify_review("не плохо") == "positive"
    assert classify_review("не понравилось") == "negative"
    assert classify_review("not bad") == "positive"
    assert classify_review("not useful") == "negative"


def test_numeric_ratings_and_emojis_are_classified():
    assert classify_review("5/5") == "positive"
    assert classify_review("1/5") == "negative"
    assert classify_review("🙂") == "positive"
    assert classify_review("😢") == "negative"


def test_long_negative_reviews_do_not_break_review_level_balance():
    positive = "Отличный курс, все понятно и полезно"
    negative = (
        "Курс ужасный непонятный сложный тяжелый перегруженный неудобный "
        "скучный плохой проблемный "
    ) * 8
    neutral = "Обычный курс без сильных плюсов и минусов"
    reviews = [positive] * 45 + [negative] * 45 + [neutral] * 10

    distribution, counts = improved_distribution(reviews)

    assert distribution == {"positive": 45.0, "negative": 45.0, "neutral": 10.0}
    assert counts == {"positive": 45, "negative": 45, "neutral": 10}
