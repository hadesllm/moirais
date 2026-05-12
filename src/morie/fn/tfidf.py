"""The man who moves a mountain begins by carrying away small stones. -- Confucius"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ._containers import DescriptiveResult


def tfidf(
    documents: list[str],
    max_features: int = 1000,
) -> DescriptiveResult:
    """Compute Term Frequency-Inverse Document Frequency matrix.

    Pure Python/NumPy implementation of TF-IDF vectorization for
    document-term analysis without external NLP dependencies.

    Parameters
    ----------
    documents : list[str]
        Collection of text documents.
    max_features : int
        Maximum number of terms to retain, ranked by corpus frequency.

    Returns
    -------
    DescriptiveResult
        name='TF-IDF', value=ndarray of shape (n_docs, n_features),
        extra has 'vocabulary' (term->index mapping), 'idf' (array),
        'feature_names' (list of terms).

    References
    ----------
    Salton, G. & Buckley, C. (1988). Term-weighting approaches in
    automatic text retrieval. *Information Processing & Management*,
    24(5), 513-523. doi:10.1016/0306-4573(88)90021-0
    """
    n_docs = len(documents)
    if n_docs == 0:
        return DescriptiveResult(
            name="TF-IDF",
            value=np.empty((0, 0)),
            extra={"vocabulary": {}, "idf": np.array([]), "feature_names": []},
        )

    tokenized = [doc.lower().split() for doc in documents]

    corpus_freq: Counter = Counter()
    for tokens in tokenized:
        corpus_freq.update(set(tokens))

    top_terms = [t for t, _ in corpus_freq.most_common(max_features)]
    vocab = {t: i for i, t in enumerate(top_terms)}
    n_features = len(vocab)

    idf = np.zeros(n_features)
    for term, idx in vocab.items():
        df = corpus_freq[term]
        idf[idx] = math.log((1 + n_docs) / (1 + df)) + 1

    tfidf_matrix = np.zeros((n_docs, n_features))
    for d, tokens in enumerate(tokenized):
        tf_counts = Counter(tokens)
        n_tokens = len(tokens) if tokens else 1
        for term, count in tf_counts.items():
            if term in vocab:
                tf = count / n_tokens
                tfidf_matrix[d, vocab[term]] = tf * idf[vocab[term]]

    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf_matrix /= norms

    return DescriptiveResult(
        name="TF-IDF",
        value=tfidf_matrix,
        extra={
            "vocabulary": vocab,
            "idf": idf,
            "feature_names": top_terms,
            "n_docs": n_docs,
            "n_features": n_features,
        },
    )


def cheatsheet() -> str:
    return "tfidf({}) -> TF-IDF vectorization. 'In my experience there is no such thi"
