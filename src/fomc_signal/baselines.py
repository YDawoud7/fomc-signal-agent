"""Non-LLM baselines for sentence classification.

* ``TfidfClassifier``: word and bigram TF-IDF with multinomial logistic
  regression, trained on the Trillion Dollar Words training split.
* ``RobertaClassifier``: the FOMC-RoBERTa model released with that dataset
  (``gtfintechlab/FOMC-RoBERTa``), a fine-tuned transformer. Needs the
  optional ``roberta`` extra (transformers + torch).
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


class TfidfClassifier:
    def __init__(self, C: float = 4.0):
        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
        )

    def fit(self, sentences, labels) -> "TfidfClassifier":
        self.model.fit(list(sentences), list(labels))
        return self

    def predict(self, sentences) -> list[str]:
        return list(self.model.predict(list(sentences)))


class RobertaClassifier:
    MODEL = "gtfintechlab/FOMC-RoBERTa"
    _MAP = {"LABEL_0": "dovish", "LABEL_1": "hawkish", "LABEL_2": "neutral"}

    def __init__(self, batch_size: int = 16):
        try:
            from transformers import pipeline
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("install the 'roberta' extra: pip install -e '.[roberta]'") from e
        self.pipe = pipeline("text-classification", model=self.MODEL, tokenizer=self.MODEL,
                             truncation=True, max_length=256, batch_size=batch_size)

    def predict(self, sentences) -> list[str]:
        return [self._MAP[o["label"]] for o in self.pipe(list(sentences))]
