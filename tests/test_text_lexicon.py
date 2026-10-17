from fomc_signal import lexicon
from fomc_signal.text import policy_sentences, split_sentences


def test_split_keeps_abbreviations():
    s = split_sentences("The U.S. economy grew. Inflation eased in Jan. 2024 slightly? Yes.")
    assert s[0] == "The U.S. economy grew."
    assert len(s) == 3


def test_policy_sentences_drop_votes():
    text = "The Committee decided to raise the target range.\n\nVoting for the action were Jerome H. Powell, Chair."
    assert policy_sentences(text) == ["The Committee decided to raise the target range."]


def test_lexicon_directions():
    assert lexicon.classify("The Committee decided to raise the target range for the federal funds rate.") == "hawkish"
    assert lexicon.classify("The Committee decided to lower the target range given downside risks.") == "dovish"
    assert lexicon.classify("The Committee will meet again in March.") == "neutral"


def test_lexicon_negation_flips():
    assert lexicon.sentence_score("Policy is no longer accommodative.") > 0


def test_document_score_range():
    sents = ["Inflation remains elevated.", "Growth has weakened.", "The meeting ended."]
    assert lexicon.document_score(sents) == 0.0
    assert -1 <= lexicon.document_score(["Growth has weakened."]) <= 1
