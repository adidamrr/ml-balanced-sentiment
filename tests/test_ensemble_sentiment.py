from src.ensemble_sentiment import analyze_reviews_ensemble, classify_review_ensemble, ensemble_distribution


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return FakeResponse(self.content)


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, content):
        self.completions = FakeCompletions(content)
        self.chat = FakeChat(self.completions)


def test_empty_list_returns_zero_distribution():
    result = analyze_reviews_ensemble([])

    assert result["distribution"] == {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    assert result["label_counts"] == {"positive": 0, "negative": 0, "neutral": 0}


def test_garbage_reviews_are_ignored():
    result = analyze_reviews_ensemble(["", ".", "a", "🤖"])

    assert result["distribution"] == {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    assert result["label_counts"] == {"positive": 0, "negative": 0, "neutral": 0}


def test_short_positive_review():
    label, meta = classify_review_ensemble("Хороший преподаватель, всё понятно")

    assert label == "positive"
    assert meta["llm_used"] is False


def test_short_negative_review():
    label, meta = classify_review_ensemble("Плохой непонятный курс")

    assert label == "negative"
    assert meta["llm_used"] is False


def test_neutral_review():
    label, meta = classify_review_ensemble("Обычный курс без сильных плюсов и минусов")

    assert label == "neutral"
    assert meta["fallback"] is True


def test_mixed_review_is_neutral_and_sarcasm_can_use_llm():
    mixed_label, mixed_meta = classify_review_ensemble("Лекции интересные, но домашка убивает")
    client = FakeClient('{"label": "negative", "confidence": 0.9, "flags": ["sarcasm"]}')
    sarcasm_label, sarcasm_meta = classify_review_ensemble(
        "Ну да, конечно, 'отличная' организация курса",
        client=client,
    )

    assert mixed_label == "neutral"
    assert mixed_meta["fallback"] is True
    assert sarcasm_label == "negative"
    assert sarcasm_meta["llm_used"] is True
    assert client.completions.calls == 1


def test_english_reviews_are_classified():
    positive_label, positive_meta = classify_review_ensemble("Great useful and clear course")
    negative_label, negative_meta = classify_review_ensemble("The course was boring and confusing")

    assert positive_label == "positive"
    assert negative_label == "negative"
    assert positive_meta["fallback"] is True
    assert negative_meta["fallback"] is True


def test_negations_are_handled():
    assert classify_review_ensemble("не плохо")[0] == "positive"
    assert classify_review_ensemble("не понравилось")[0] == "negative"
    assert classify_review_ensemble("not bad")[0] == "positive"
    assert classify_review_ensemble("not useful")[0] == "negative"


def test_numeric_ratings_and_emojis_are_classified():
    assert classify_review_ensemble("5/5")[0] == "positive"
    assert classify_review_ensemble("1/5")[0] == "negative"
    assert classify_review_ensemble("🙂")[0] == "positive"
    assert classify_review_ensemble("😢")[0] == "negative"


def test_long_negative_reviews_do_not_break_review_level_balance():
    positive = "Отличный курс, все понятно и полезно"
    negative = (
        "Курс ужасный непонятный сложный тяжелый перегруженный неудобный "
        "скучный плохой проблемный "
    ) * 8
    neutral = "Обычный курс без сильных плюсов и минусов"
    reviews = [positive] * 45 + [negative] * 45 + [neutral] * 10

    result = ensemble_distribution(reviews)

    assert result["distribution"] == {"positive": 45.0, "negative": 45.0, "neutral": 10.0}
    assert result["label_counts"] == {"positive": 45, "negative": 45, "neutral": 10}
