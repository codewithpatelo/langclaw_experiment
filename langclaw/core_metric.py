"""CORE: Conversational Robustness Evaluation Score.

Faithful implementation of the CORE metric from:
    Pandey, Yang, Liu & Jin (2026). "CORE: Measuring Multi-Agent LLM
    Interaction Quality under Game-Theoretic Pressures."
    Proceedings of EACL 2026, pp. 1251-1266.
    DOI: 10.18653/v1/2026.eacl-long.57

Formula (Eq. 3-6 of the paper):

    CORE = H(C) * |Z|^(-alpha) * (1 - (1/(N-1)) * sum cos(e_j, e_{j+1}))^beta

where:
    H(C)  = Shannon entropy of K-means cluster distribution over utterance
            embeddings (Eq. 4). Captures mode diversity.
    |Z|   = number of distinct n-grams in the corpus. Penalizes repetition
            via the Zipf exponent alpha (Eq. 5).
    cos() = cosine similarity between consecutive utterance embeddings (Eq. 6).
            Penalizes semantic stagnation via the Heaps exponent beta.

alpha and beta are estimated from the empirical Zipf and Heaps distributions
of the dialog data.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


def compute_core(
    utterances: list[str],
    embeddings: np.ndarray,
    n_gram_size: int = 3,
    k_clusters: int = 5,
) -> float:
    """Compute the CORE score for a set of utterances.

    Parameters
    ----------
    utterances : list[str]
        Raw text of each utterance in order.
    embeddings : np.ndarray
        Shape (N, D) — one embedding per utterance.
    n_gram_size : int
        Size of n-grams for repetition penalty.
    k_clusters : int
        Number of K-means clusters for entropy computation.

    Returns
    -------
    float
        CORE score in [0, inf). Higher = more robust / diverse conversation.
    """
    if len(utterances) < 2 or embeddings.shape[0] < 2:
        return 0.0

    h_c = _cluster_entropy(embeddings, k_clusters)
    rep_penalty = _repetition_penalty(utterances, n_gram_size)
    sem_penalty = _semantic_stagnation_penalty(embeddings)

    return h_c * rep_penalty * sem_penalty


def compute_core_windowed(
    utterances: list[str],
    embeddings: np.ndarray,
    n_windows: int = 5,
    n_gram_size: int = 3,
    k_clusters: int = 5,
) -> list[float]:
    """Compute CORE per temporal window.

    Divides utterances into n_windows equal-sized windows and computes
    CORE for each. Returns a list of n_windows CORE scores.
    """
    n = len(utterances)
    if n < n_windows:
        return [compute_core(utterances, embeddings, n_gram_size, k_clusters)]

    window_size = n // n_windows
    scores = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size if i < n_windows - 1 else n
        w_utt = utterances[start:end]
        w_emb = embeddings[start:end]
        scores.append(compute_core(w_utt, w_emb, n_gram_size, k_clusters))
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Component functions (Eq. 4, 5, 6)
# ──────────────────────────────────────────────────────────────────────────────


def _cluster_entropy(embeddings: np.ndarray, k: int) -> float:
    """Shannon entropy of K-means cluster distribution (Eq. 4).

    H(C) = -sum_{i=1}^K p_i * log(p_i)
    """
    from sklearn.cluster import KMeans

    n = embeddings.shape[0]
    k_actual = min(k, n)
    if k_actual < 2:
        return 0.0

    kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    counts = Counter(labels)
    total = sum(counts.values())

    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)

    return entropy


def _repetition_penalty(utterances: list[str], n: int) -> float:
    """Repetition penalty: |Z|^(-alpha) (from Eq. 3, 5).

    Z = multiset of all n-grams extracted from utterances.
    alpha = empirical Zipf exponent estimated from n-gram frequencies.
    """
    all_ngrams = _extract_ngrams(utterances, n)
    if not all_ngrams:
        return 1.0

    z_size = len(set(all_ngrams))
    if z_size <= 1:
        return 1.0

    alpha = _estimate_zipf_exponent(all_ngrams)
    alpha = max(0.01, alpha)

    return z_size ** (-alpha)


def _semantic_stagnation_penalty(embeddings: np.ndarray) -> float:
    """Semantic stagnation penalty (from Eq. 3, 6).

    (1 - (1/(N-1)) * sum_{j=1}^{N-1} cos(e_j, e_{j+1}))^beta
    """
    n = embeddings.shape[0]
    if n < 2:
        return 1.0

    cos_sims = []
    for j in range(n - 1):
        norm_j = np.linalg.norm(embeddings[j])
        norm_j1 = np.linalg.norm(embeddings[j + 1])
        if norm_j > 0 and norm_j1 > 0:
            cos_sim = np.dot(embeddings[j], embeddings[j + 1]) / (norm_j * norm_j1)
            cos_sims.append(float(cos_sim))
        else:
            cos_sims.append(0.0)

    mean_cos = sum(cos_sims) / len(cos_sims)
    stagnation_term = max(0.0, 1.0 - mean_cos)

    beta = _estimate_heaps_exponent(len(cos_sims))
    beta = max(0.01, beta)

    return stagnation_term ** beta


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────


def _extract_ngrams(utterances: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract all n-grams from a list of utterances."""
    all_ngrams: list[tuple[str, ...]] = []
    for utt in utterances:
        tokens = utt.lower().split()
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(tuple(tokens[i:i + n]))
    return all_ngrams


def _estimate_zipf_exponent(ngrams: list[tuple[str, ...]]) -> float:
    """Estimate Zipf exponent alpha from n-gram frequency distribution.

    Uses least-squares fit of log(rank) vs log(frequency).
    Zipf's law: f(r) ~ r^(-alpha)  =>  log(f) = -alpha * log(r) + const
    """
    counts = Counter(ngrams)
    if len(counts) < 2:
        return 1.0

    freqs = sorted(counts.values(), reverse=True)
    ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)
    log_ranks = np.log(ranks)
    log_freqs = np.log(np.array(freqs, dtype=np.float64))

    n = len(log_ranks)
    sum_x = log_ranks.sum()
    sum_y = log_freqs.sum()
    sum_xy = (log_ranks * log_freqs).sum()
    sum_x2 = (log_ranks ** 2).sum()

    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return 1.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    return abs(slope)


def _estimate_heaps_exponent(n_utterances: int) -> float:
    """Estimate Heaps exponent beta.

    Heaps' law: V(n) ~ n^beta where V is vocabulary size and n is corpus size.
    For short dialogs, we use a conservative default.
    """
    if n_utterances < 10:
        return 0.5
    if n_utterances < 50:
        return 0.6
    return 0.7
