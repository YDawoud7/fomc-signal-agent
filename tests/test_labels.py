from fomc_signal.baselines import TfidfClassifier
from fomc_signal.labels import evaluate


def test_evaluate_metrics():
    out = evaluate(["hawkish", "dovish", "neutral", "neutral"], ["hawkish", "neutral", "neutral", "neutral"])
    assert out["accuracy"] == 0.75
    assert out["confusion"][0] == [0, 1, 0]  # dovish row: predicted neutral


def test_tfidf_classifier_learns_simple_split():
    X = ["raise rates inflation high", "inflation pressures rise", "cut rates weak growth", "downside risks weak",
         "the committee met", "meeting held today"] * 5
    y = ["hawkish", "hawkish", "dovish", "dovish", "neutral", "neutral"] * 5
    clf = TfidfClassifier().fit(X, y)
    assert clf.predict(["inflation high raise rates"]) == ["hawkish"]
