"""Baselines for RT2 exploit-vulnerability ranking.

Classical / statistical:
  tfidf           : TF-IDF cosine similarity
  lsa             : Latent Semantic Analysis (TruncatedSVD on TF-IDF)
  bm25            : Okapi BM25 (inline)
  bm25_plus       : BM25+ — lower-bounds term frequency with δ=1
  bm25_l          : BM25L — re-normalises document length c(q,d)
  query_likelihood: Dirichlet-smoothed unigram language model
  jaccard         : token-set Jaccard coefficient
  char_ngram      : character n-gram (n=3,4) Jaccard
  logistic        : logistic regression on TF-IDF features (pairwise score)
  svm             : linear SVM decision scores on TF-IDF features
  random_forest   : random forest probability on TF-IDF features

Neural (no large pretrained model):
  w2v_mean        : word2vec average-pooled embeddings (gensim, trained on EV corpus)
  fasttext_mean   : fasttext average-pooled embeddings (gensim, trained on EV corpus)
  doc2vec         : PV-DM paragraph vectors (gensim)
  siamese_bilstm  : Siamese BiLSTM with mean-pool + cosine, triplet-trained
  cnn_encoder     : CTE ablation — 1D-CNN bank only (no BiLSTM, no attention)
  bilstm_cnn      : CTE ablation — BiLSTM + CNN, mean-pool (no local-guided attn)

CTE ablations (same architecture, subset of pretraining):
  cte_mlm_only    : MLM pretraining only (λ_contrastive=0), then fine-tune
  cte_cont_only   : contrastive pretraining only (λ_mlm=0), then fine-tune
  cte_ft_only     : skip pretraining entirely, supervised fine-tune from scratch

Pretrained (optional, gated):
  simcse          : princeton-nlp/sup-simcse-bert-base-uncased
  contriever      : facebook/contriever
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections import Counter
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..config import Config
from ..data.schemas import EVPair

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")


def _tok(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# ---------------------------------------------------------------------------
# Shared: TF-IDF vectoriser builder
# ---------------------------------------------------------------------------

def build_tfidf_ranker(pairs: list[EVPair]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = list({p.exploit_text for p in pairs} | {p.vulnerability_text for p in pairs})
    vec = TfidfVectorizer(tokenizer=_tok, lowercase=False, token_pattern=None)
    vec.fit(corpus)
    return vec


# ---------------------------------------------------------------------------
# TF-IDF cosine
# ---------------------------------------------------------------------------

def tfidf_score(vec, exploit_texts: list[str], candidate_texts: list[str]) -> np.ndarray:
    A = vec.transform(exploit_texts)
    B = vec.transform(candidate_texts)
    nA = np.sqrt(A.multiply(A).sum(axis=1))
    nB = np.sqrt(B.multiply(B).sum(axis=1))
    A_n = A.multiply(1.0 / (nA + 1e-12))
    B_n = B.multiply(1.0 / (nB + 1e-12))
    return np.asarray(A_n @ B_n.T.tocsr().toarray())


# ---------------------------------------------------------------------------
# LSA
# ---------------------------------------------------------------------------

def lsa_score(
    vec, exploit_texts: list[str], candidate_texts: list[str], dim: int = 64
) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD

    corpus = list(dict.fromkeys([*exploit_texts, *candidate_texts]))
    M = vec.transform(corpus)
    dim = max(2, min(dim, min(M.shape) - 1))
    svd = TruncatedSVD(n_components=dim, random_state=0)
    svd.fit(M)
    A = svd.transform(vec.transform(exploit_texts))
    B = svd.transform(vec.transform(candidate_texts))
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return A @ B.T


# ---------------------------------------------------------------------------
# BM25 family (inline)
# ---------------------------------------------------------------------------

class BM25:
    """Okapi BM25."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(1, self.N)
        self.df: Counter = Counter()
        self.dl: list[int] = []
        self.tf: list[Counter] = []
        for doc in corpus:
            self.dl.append(len(doc))
            tf = Counter(doc)
            self.tf.append(tf)
            for t in tf:
                self.df[t] += 1
        self.idf = {
            t: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for t, df in self.df.items()
        }

    def score(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        for qt in query:
            if qt not in self.idf:
                continue
            idf = self.idf[qt]
            for j, tf in enumerate(self.tf):
                if qt not in tf:
                    continue
                f = tf[qt]
                K = self.k1 * (1 - self.b + self.b * self.dl[j] / self.avgdl)
                scores[j] += idf * f * (self.k1 + 1) / (f + K + 1e-12)
        return scores


class BM25Plus(BM25):
    """BM25+ — adds δ=1 lower-bound on term frequency contribution."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        super().__init__(corpus, k1, b)
        self.delta = delta

    def score(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        for qt in query:
            if qt not in self.idf:
                continue
            idf = self.idf[qt]
            for j, tf in enumerate(self.tf):
                if qt not in tf:
                    continue
                f = tf[qt]
                K = self.k1 * (1 - self.b + self.b * self.dl[j] / self.avgdl)
                scores[j] += idf * (f * (self.k1 + 1) / (f + K + 1e-12) + self.delta)
        return scores


class BM25L(BM25):
    """BM25L — re-normalised term frequency to reduce length-bias."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, delta: float = 0.5):
        super().__init__(corpus, k1, b)
        self.delta = delta

    def score(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        for qt in query:
            if qt not in self.idf:
                continue
            idf = self.idf[qt]
            for j, tf in enumerate(self.tf):
                if qt not in tf:
                    continue
                f = tf[qt]
                c = f / (1 - self.b + self.b * self.dl[j] / self.avgdl)
                scores[j] += idf * (self.k1 + 1) * (c + self.delta) / (self.k1 + c + self.delta)
        return scores


def _bm25_matrix(
    exploit_texts: list[str], candidate_texts: list[str], cls=BM25
) -> np.ndarray:
    bm = cls([_tok(c) for c in candidate_texts])
    return np.stack([bm.score(_tok(q)) for q in exploit_texts], axis=0)


def bm25_score_matrix(exploit_texts: list[str], candidate_texts: list[str]) -> np.ndarray:
    return _bm25_matrix(exploit_texts, candidate_texts, BM25)


def bm25_plus_score_matrix(exploit_texts: list[str], candidate_texts: list[str]) -> np.ndarray:
    return _bm25_matrix(exploit_texts, candidate_texts, BM25Plus)


def bm25_l_score_matrix(exploit_texts: list[str], candidate_texts: list[str]) -> np.ndarray:
    return _bm25_matrix(exploit_texts, candidate_texts, BM25L)


# ---------------------------------------------------------------------------
# Query Likelihood — Dirichlet smoothing
# ---------------------------------------------------------------------------

def query_likelihood_score(
    exploit_texts: list[str],
    candidate_texts: list[str],
    mu: float = 2000.0,
) -> np.ndarray:
    """Dirichlet-smoothed unigram language model P(q | d)."""
    cand_tokens = [_tok(t) for t in candidate_texts]
    # Corpus-level unigram distribution
    corpus_cf: Counter = Counter(t for doc in cand_tokens for t in doc)
    corpus_total = sum(corpus_cf.values()) + 1e-12
    scores = np.zeros((len(exploit_texts), len(candidate_texts)), dtype=np.float32)
    for i, q_text in enumerate(exploit_texts):
        query = _tok(q_text)
        if not query:
            continue
        for j, (doc, d_tokens) in enumerate(zip(candidate_texts, cand_tokens)):
            dl = len(d_tokens)
            tf = Counter(d_tokens)
            log_prob = 0.0
            for qt in query:
                tf_qd = tf.get(qt, 0)
                cf_q = corpus_cf.get(qt, 0) / corpus_total
                smoothed = (tf_qd + mu * cf_q) / (dl + mu)
                log_prob += math.log(smoothed + 1e-30)
            scores[i, j] = log_prob
    return scores


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------

def jaccard_score_matrix(exploit_texts: list[str], candidate_texts: list[str]) -> np.ndarray:
    """Token-level Jaccard: |A∩B| / |A∪B|."""
    scores = np.zeros((len(exploit_texts), len(candidate_texts)), dtype=np.float32)
    cand_sets = [set(_tok(t)) for t in candidate_texts]
    for i, e_text in enumerate(exploit_texts):
        e_set = set(_tok(e_text))
        for j, c_set in enumerate(cand_sets):
            inter = len(e_set & c_set)
            union = len(e_set | c_set)
            scores[i, j] = inter / (union + 1e-12)
    return scores


# ---------------------------------------------------------------------------
# Character n-gram Jaccard
# ---------------------------------------------------------------------------

def char_ngram_score_matrix(
    exploit_texts: list[str],
    candidate_texts: list[str],
    ns: tuple[int, ...] = (3, 4),
) -> np.ndarray:
    """Character n-gram Jaccard over the union of n-gram sets for n ∈ ns."""

    def _ngrams(text: str) -> set[str]:
        t = text.lower()
        result: set[str] = set()
        for n in ns:
            result.update(t[k: k + n] for k in range(len(t) - n + 1))
        return result

    scores = np.zeros((len(exploit_texts), len(candidate_texts)), dtype=np.float32)
    cand_ng = [_ngrams(t) for t in candidate_texts]
    for i, e_text in enumerate(exploit_texts):
        e_ng = _ngrams(e_text)
        for j, c_ng in enumerate(cand_ng):
            inter = len(e_ng & c_ng)
            union = len(e_ng | c_ng)
            scores[i, j] = inter / (union + 1e-12)
    return scores


# ---------------------------------------------------------------------------
# Supervised sklearn baselines (Logistic Regression / SVM / Random Forest)
# ---------------------------------------------------------------------------

def _pairwise_features(
    vec,
    exploit_texts: list[str],
    candidate_texts: list[str],
    positive_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build concatenated [e; v] TF-IDF features + labels for training."""
    X_e = np.asarray(vec.transform(exploit_texts).todense())
    X_v = np.asarray(vec.transform(candidate_texts).todense())
    rows, cols, labels = [], [], []
    for i in range(len(exploit_texts)):
        pos_j = int(positive_indices[i])
        neg_j = (pos_j + 7) % len(candidate_texts)  # deterministic negative
        rows.append(np.concatenate([X_e[i], X_v[pos_j]]))
        labels.append(1)
        rows.append(np.concatenate([X_e[i], X_v[neg_j]]))
        labels.append(0)
    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int32)


def _sklearn_score_matrix(
    clf,
    vec,
    exploit_texts: list[str],
    candidate_texts: list[str],
    positive_indices: np.ndarray,
    score_method: str = "predict_proba",
) -> np.ndarray:
    X_train, y_train = _pairwise_features(vec, exploit_texts, candidate_texts, positive_indices)
    clf.fit(X_train, y_train)
    X_e = np.asarray(vec.transform(exploit_texts).todense())
    X_v = np.asarray(vec.transform(candidate_texts).todense())
    scores = np.zeros((len(exploit_texts), len(candidate_texts)), dtype=np.float32)
    for i in range(len(exploit_texts)):
        pairs = np.hstack([
            np.tile(X_e[i], (len(candidate_texts), 1)),
            X_v,
        ]).astype(np.float32)
        if score_method == "predict_proba":
            s = clf.predict_proba(pairs)[:, 1]
        else:
            s = clf.decision_function(pairs)
        scores[i] = s
    return scores


def logistic_score_matrix(
    vec,
    exploit_texts: list[str],
    candidate_texts: list[str],
    positive_indices: np.ndarray,
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=200, C=1.0, random_state=0)
    return _sklearn_score_matrix(clf, vec, exploit_texts, candidate_texts, positive_indices)


def svm_score_matrix(
    vec,
    exploit_texts: list[str],
    candidate_texts: list[str],
    positive_indices: np.ndarray,
) -> np.ndarray:
    from sklearn.svm import LinearSVC

    clf = LinearSVC(max_iter=500, C=1.0, random_state=0)
    return _sklearn_score_matrix(
        clf, vec, exploit_texts, candidate_texts, positive_indices, score_method="decision_function"
    )


def random_forest_score_matrix(
    vec,
    exploit_texts: list[str],
    candidate_texts: list[str],
    positive_indices: np.ndarray,
) -> np.ndarray:
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0, n_jobs=-1)
    return _sklearn_score_matrix(clf, vec, exploit_texts, candidate_texts, positive_indices)


# ---------------------------------------------------------------------------
# Word2Vec mean-pool
# ---------------------------------------------------------------------------

def _build_w2v_model(pairs: list[EVPair], config: Config):
    from gensim.models import Word2Vec as GW2V

    sentences = [_tok(p.exploit_text) + _tok(p.vulnerability_text) for p in pairs]
    return GW2V(
        sentences=sentences, vector_size=config.cte_emb_dim,
        window=5, min_count=1, sg=1, epochs=3, seed=config.seed, workers=1,
    )


def _mean_pool_w2v(model, texts: list[str]) -> np.ndarray:
    dim = model.vector_size
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        toks = [t for t in _tok(text) if t in model.wv]
        if toks:
            out[i] = np.mean([model.wv[t] for t in toks], axis=0)
    return out


def w2v_mean_score_matrix(
    pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
) -> np.ndarray:
    model = _build_w2v_model(pairs, config)
    A = _mean_pool_w2v(model, exploit_texts)
    B = _mean_pool_w2v(model, candidate_texts)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return A @ B.T


# ---------------------------------------------------------------------------
# FastText mean-pool
# ---------------------------------------------------------------------------

def fasttext_mean_score_matrix(
    pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
) -> np.ndarray:
    from gensim.models import FastText as GFT

    sentences = [_tok(p.exploit_text) + _tok(p.vulnerability_text) for p in pairs]
    model = GFT(
        sentences=sentences, vector_size=config.cte_emb_dim,
        window=5, min_count=1, sg=1, epochs=3, seed=config.seed, workers=1,
    )

    def _pool(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), model.vector_size), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = _tok(text)
            if toks:
                out[i] = np.mean([model.wv[t] for t in toks], axis=0)
        return out

    A = _pool(exploit_texts)
    B = _pool(candidate_texts)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return A @ B.T


# ---------------------------------------------------------------------------
# Doc2Vec
# ---------------------------------------------------------------------------

def doc2vec_score_matrix(
    pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
) -> np.ndarray:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    corpus: list[TaggedDocument] = []
    for i, p in enumerate(pairs):
        corpus.append(TaggedDocument(_tok(p.exploit_text), [f"e_{i}"]))
        corpus.append(TaggedDocument(_tok(p.vulnerability_text), [f"v_{i}"]))
    model = Doc2Vec(
        documents=corpus, vector_size=config.cte_emb_dim,
        window=5, min_count=1, epochs=5, seed=config.seed, workers=1,
    )

    def _infer(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), model.vector_size), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i] = model.infer_vector(_tok(text), epochs=10)
        return out

    A = _infer(exploit_texts)
    B = _infer(candidate_texts)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return A @ B.T


# ---------------------------------------------------------------------------
# Tokenisation shared by neural baselines
# ---------------------------------------------------------------------------

def _tokenize_batch(texts: list[str], config: Config) -> Tensor:
    from ..rt2.trigram_hasher import tokenize_cte

    return torch.tensor(
        [tokenize_cte(t, config.trigram_buckets, config.cte_max_len) for t in texts],
        dtype=torch.long,
    )


# ---------------------------------------------------------------------------
# Siamese BiLSTM
# ---------------------------------------------------------------------------

class SiameseBiLSTM(nn.Module):
    """Lightweight Siamese BiLSTM: embed → BiLSTM → mean-pool → cosine."""

    def __init__(self, config: Config):
        super().__init__()
        from ..rt2.trigram_hasher import PAD_ID

        self.pad = PAD_ID
        d = max(32, config.cte_emb_dim // 2)
        self.emb = nn.Embedding(config.trigram_buckets, d, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            d, d, batch_first=True, bidirectional=True, dropout=0.0
        )
        self.proj = nn.Linear(2 * d, config.cte_emb_dim)

    def encode(self, ids: Tensor) -> Tensor:
        mask = (ids != self.pad).float()  # [B, L]
        x = self.emb(ids)
        out, _ = self.lstm(x)  # [B, L, 2d]
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / lengths  # [B, 2d]
        return self.proj(pooled)


def _triplet_train(
    model: nn.Module,
    encode_fn: Callable[[Tensor], Tensor],
    pairs: list[EVPair],
    config: Config,
    device: torch.device,
    epochs: int = 3,
    margin: float = 0.3,
) -> None:
    """Generic triplet-loss fine-tuning for encoder baselines."""
    from ..rt2.trigram_hasher import tokenize_cte

    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=config.cte_lr)
    positives = [p for p in pairs if p.label == 1]
    negatives = [p for p in pairs if p.label == 0]
    if not positives or not negatives:
        return
    bs = min(config.cte_batch_size, len(positives))
    for _ in range(epochs):
        np.random.shuffle(positives)  # type: ignore[arg-type]
        for start in range(0, len(positives), bs):
            batch_pos = positives[start: start + bs]
            batch_neg = negatives[: len(batch_pos)]
            e_ids = torch.tensor(
                [tokenize_cte(p.exploit_text, config.trigram_buckets, config.cte_max_len)
                 for p in batch_pos], device=device,
            )
            v_pos = torch.tensor(
                [tokenize_cte(p.vulnerability_text, config.trigram_buckets, config.cte_max_len)
                 for p in batch_pos], device=device,
            )
            v_neg = torch.tensor(
                [tokenize_cte(p.vulnerability_text, config.trigram_buckets, config.cte_max_len)
                 for p in batch_neg], device=device,
            )
            z_e = encode_fn(e_ids)
            z_p = encode_fn(v_pos)
            z_n = encode_fn(v_neg)
            z_e = nn.functional.normalize(z_e, dim=-1)
            z_p = nn.functional.normalize(z_p, dim=-1)
            z_n = nn.functional.normalize(z_n, dim=-1)
            sim_pos = (z_e * z_p).sum(-1)
            sim_neg = (z_e * z_n).sum(-1)
            loss = torch.relu(margin - sim_pos + sim_neg).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()


def _encode_all(
    model: nn.Module,
    encode_fn: Callable[[Tensor], Tensor],
    texts: list[str],
    config: Config,
    device: torch.device,
    bs: int = 64,
) -> np.ndarray:
    model.eval()
    out_parts: list[np.ndarray] = []
    ids_all = _tokenize_batch(texts, config)
    with torch.no_grad():
        for start in range(0, len(texts), bs):
            batch = ids_all[start: start + bs].to(device)
            z = encode_fn(batch)
            out_parts.append(nn.functional.normalize(z, dim=-1).cpu().numpy())
    return np.concatenate(out_parts, axis=0)


def siamese_bilstm_score_matrix(
    pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
    epochs: int = 3,
) -> np.ndarray:
    model = SiameseBiLSTM(config)
    _triplet_train(model, model.encode, pairs, config, device, epochs=epochs)
    A = _encode_all(model, model.encode, exploit_texts, config, device)
    B = _encode_all(model, model.encode, candidate_texts, config, device)
    return A @ B.T


# ---------------------------------------------------------------------------
# CNN-only encoder (CTE ablation: no BiLSTM, no attention)
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """1D-CNN bank over raw embeddings → max-pool → projection.

    CTE ablation: replaces BiLSTM and local-guided attention with max-pool.
    """

    def __init__(self, config: Config):
        super().__init__()
        from ..rt2.trigram_hasher import PAD_ID

        self.pad = PAD_ID
        self.emb = nn.Embedding(config.trigram_buckets, config.cte_emb_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList([
            nn.Conv1d(config.cte_emb_dim, config.cte_cnn_channels, k, padding=k // 2)
            for k in config.cte_cnn_kernels
        ])
        d_local = config.cte_cnn_channels * len(config.cte_cnn_kernels)
        self.proj = nn.Linear(d_local, config.cte_emb_dim)
        self.drop = nn.Dropout(config.cte_dropout)

    def encode(self, ids: Tensor) -> Tensor:
        x = self.emb(ids).transpose(1, 2)  # [B, E, L]
        parts = []
        for conv in self.convs:
            h = torch.relu(conv(x))  # [B, C, L']
            parts.append(h.max(dim=-1).values)
        h_cat = torch.cat(parts, dim=-1)
        return self.proj(self.drop(h_cat))


def cnn_encoder_score_matrix(
    pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
    epochs: int = 3,
) -> np.ndarray:
    model = CNNEncoder(config)
    _triplet_train(model, model.encode, pairs, config, device, epochs=epochs)
    A = _encode_all(model, model.encode, exploit_texts, config, device)
    B = _encode_all(model, model.encode, candidate_texts, config, device)
    return A @ B.T


# ---------------------------------------------------------------------------
# BiLSTM + CNN encoder (CTE ablation: no local-guided attention)
# ---------------------------------------------------------------------------

class BiLSTMCNNEncoder(nn.Module):
    """BiLSTM → 1D-CNN → mean-pool.  CTE without the LG-GCA attention."""

    def __init__(self, config: Config):
        super().__init__()
        from ..rt2.trigram_hasher import PAD_ID

        self.pad = PAD_ID
        self.emb = nn.Embedding(config.trigram_buckets, config.cte_emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            config.cte_emb_dim, config.cte_lstm_hidden,
            batch_first=True, bidirectional=True, dropout=0.0,
        )
        lstm_out = 2 * config.cte_lstm_hidden
        self.convs = nn.ModuleList([
            nn.Conv1d(lstm_out, config.cte_cnn_channels, k, padding=k // 2)
            for k in config.cte_cnn_kernels
        ])
        d_local = config.cte_cnn_channels * len(config.cte_cnn_kernels)
        self.proj = nn.Linear(d_local, config.cte_emb_dim)
        self.drop = nn.Dropout(config.cte_dropout)

    def encode(self, ids: Tensor) -> Tensor:
        mask = (ids != self.pad).float().unsqueeze(-1)  # [B, L, 1]
        x = self.emb(ids)
        lstm_out, _ = self.lstm(x)         # [B, L, 2H]
        conv_in = lstm_out.transpose(1, 2)  # [B, 2H, L]
        parts = []
        for conv in self.convs:
            h = torch.relu(conv(conv_in))
            if h.shape[-1] > conv_in.shape[-1]:
                h = h[..., : conv_in.shape[-1]]
            elif h.shape[-1] < conv_in.shape[-1]:
                h = torch.nn.functional.pad(h, (0, conv_in.shape[-1] - h.shape[-1]))
            parts.append(h)
        local = torch.cat(parts, dim=1).transpose(1, 2)  # [B, L, d_local]
        local = self.proj(self.drop(local))
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = (local * mask).sum(dim=1) / lengths
        return pooled


def bilstm_cnn_score_matrix(
    pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
    epochs: int = 3,
) -> np.ndarray:
    model = BiLSTMCNNEncoder(config)
    _triplet_train(model, model.encode, pairs, config, device, epochs=epochs)
    A = _encode_all(model, model.encode, exploit_texts, config, device)
    B = _encode_all(model, model.encode, candidate_texts, config, device)
    return A @ B.T


# ---------------------------------------------------------------------------
# CTE ablations
# ---------------------------------------------------------------------------

def _run_cte_ablation(
    pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
    mlm_weight: float,
    cont_weight: float,
    do_pretrain: bool = True,
) -> np.ndarray:
    from ..rt2.cte_model import CTE
    from ..rt2.pretrain_cte import pretrain_cte
    from ..rt2.finetune_cte import finetune_cte
    from ..rt2.trigram_hasher import tokenize_cte

    ablation_cfg = dataclasses.replace(
        config,
        cte_mlm_weight=mlm_weight,
        cte_contrastive_weight=cont_weight,
    )
    model = CTE(ablation_cfg).to(device)
    if do_pretrain:
        pretrain_cte(model, pairs, ablation_cfg, device)
    finetune_cte(model, pairs, ablation_cfg, device)

    def _encode(texts: list[str]) -> np.ndarray:
        model.eval()
        parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), config.cte_batch_size):
                batch = torch.tensor(
                    [tokenize_cte(t, config.trigram_buckets, config.cte_max_len)
                     for t in texts[start: start + config.cte_batch_size]],
                    device=device,
                )
                z = nn.functional.normalize(model.encode_sentence(batch), dim=-1)
                parts.append(z.cpu().numpy())
        return np.concatenate(parts, axis=0)

    A = _encode(exploit_texts)
    B = _encode(candidate_texts)
    return A @ B.T


def cte_mlm_only_score_matrix(
    pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
) -> np.ndarray:
    """CTE pretrained with MLM loss only (λ_contrastive=0), then fine-tuned."""
    return _run_cte_ablation(pairs, exploit_texts, candidate_texts, config, device,
                             mlm_weight=1.0, cont_weight=0.0)


def cte_contrastive_only_score_matrix(
    pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
) -> np.ndarray:
    """CTE pretrained with contrastive loss only (λ_mlm=0), then fine-tuned."""
    return _run_cte_ablation(pairs, exploit_texts, candidate_texts, config, device,
                             mlm_weight=0.0, cont_weight=1.0)


def cte_finetune_only_score_matrix(
    pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
) -> np.ndarray:
    """CTE with no pretraining — supervised fine-tune from random init."""
    return _run_cte_ablation(pairs, exploit_texts, candidate_texts, config, device,
                             mlm_weight=1.0, cont_weight=1.0, do_pretrain=False)


# ---------------------------------------------------------------------------
# Optional pretrained sentence-transformer baselines
# ---------------------------------------------------------------------------

def _try_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception:
        return None


def simcse_score(exploit_texts: list[str], candidate_texts: list[str]) -> np.ndarray | None:
    ST = _try_sentence_transformers()
    if ST is None:
        return None
    try:
        model = ST("princeton-nlp/sup-simcse-bert-base-uncased")
    except Exception:
        return None
    A = model.encode(exploit_texts, convert_to_numpy=True, normalize_embeddings=True)
    B = model.encode(candidate_texts, convert_to_numpy=True, normalize_embeddings=True)
    return A @ B.T


def contriever_score(exploit_texts: list[str], candidate_texts: list[str]) -> np.ndarray | None:
    ST = _try_sentence_transformers()
    if ST is None:
        return None
    try:
        model = ST("facebook/contriever")
    except Exception:
        return None
    A = model.encode(exploit_texts, convert_to_numpy=True, normalize_embeddings=True)
    B = model.encode(candidate_texts, convert_to_numpy=True, normalize_embeddings=True)
    return A @ B.T


# ===========================================================================
# NEW BASELINES — appended below
# ===========================================================================

# ---------------------------------------------------------------------------
# Lexical overlap: ROUGE-L
# ---------------------------------------------------------------------------

def _lcs_length(a: list[str], b: list[str]) -> int:
    """Dynamic-programming LCS length between two token lists."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    # Use two-row rolling DP to save memory.
    prev = np.zeros(m + 1, dtype=np.int32)
    curr = np.zeros(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, prev
        curr[:] = 0
    return int(prev[m])


def rouge_l_score_matrix(
    exploit_texts: list[str],
    candidate_texts: list[str],
    max_tokens: int = 100,
) -> np.ndarray:
    """ROUGE-L F1 between exploits and candidates.

    Score = LCS-based F1 = 2 * P * R / (P + R), where
        P = lcs_len / len(candidate_tokens)
        R = lcs_len / len(exploit_tokens)
    Texts are truncated to *max_tokens* tokens before scoring.
    """
    scores = np.zeros((len(exploit_texts), len(candidate_texts)), dtype=np.float32)
    e_tok = [_tok(t)[:max_tokens] for t in exploit_texts]
    c_tok = [_tok(t)[:max_tokens] for t in candidate_texts]
    for i, a in enumerate(e_tok):
        la = len(a)
        if la == 0:
            continue
        for j, b in enumerate(c_tok):
            lb = len(b)
            if lb == 0:
                continue
            lcs = _lcs_length(a, b)
            p = lcs / lb
            r = lcs / la
            scores[i, j] = 2.0 * p * r / (p + r + 1e-12)
    return scores


# ---------------------------------------------------------------------------
# Lexical overlap: normalised edit distance (word-level Levenshtein)
# ---------------------------------------------------------------------------

def _word_edit_distance(a: list[str], b: list[str]) -> int:
    """Wagner-Fischer DP on token lists."""
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[m]


def edit_distance_score_matrix(
    exploit_texts: list[str],
    candidate_texts: list[str],
    max_tokens: int = 50,
) -> np.ndarray:
    """Normalised word-level Levenshtein similarity.

    similarity = 1 - edit_distance(a, b) / max(len(a), len(b))
    Texts are truncated to *max_tokens* tokens.
    """
    scores = np.zeros((len(exploit_texts), len(candidate_texts)), dtype=np.float32)
    e_tok = [_tok(t)[:max_tokens] for t in exploit_texts]
    c_tok = [_tok(t)[:max_tokens] for t in candidate_texts]
    for i, a in enumerate(e_tok):
        for j, b in enumerate(c_tok):
            maxlen = max(len(a), len(b))
            if maxlen == 0:
                scores[i, j] = 1.0
                continue
            ed = _word_edit_distance(a, b)
            scores[i, j] = 1.0 - ed / maxlen
    return scores


# ---------------------------------------------------------------------------
# Lexical overlap: Word Mover's Distance approximation
# ---------------------------------------------------------------------------

def wmd_score_matrix(
    ev_pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    max_tokens: int = 20,
) -> np.ndarray:
    """Approximate WMD using exact EMD (scipy linear_sum_assignment).

    Trains a small word2vec model on ev_pairs, then computes pairwise
    transportation distances.  Score = -wmd_distance (higher = more similar).
    Texts are truncated to *max_tokens* tokens for speed.
    """
    from gensim.models import Word2Vec as _GW2V
    from scipy.optimize import linear_sum_assignment

    sentences = [_tok(p.exploit_text) + _tok(p.vulnerability_text) for p in ev_pairs]
    w2v = _GW2V(
        sentences=sentences,
        vector_size=config.cte_emb_dim,
        window=5,
        min_count=1,
        sg=1,
        epochs=5,
        seed=config.seed,
        workers=1,
    )

    def _get_vecs(tokens: list[str]) -> np.ndarray:
        toks = [t for t in tokens if t in w2v.wv]
        if not toks:
            return np.zeros((1, config.cte_emb_dim), dtype=np.float32)
        return np.array([w2v.wv[t] for t in toks], dtype=np.float32)

    def _wmd_approx(va: np.ndarray, vb: np.ndarray) -> float:
        # Compute pairwise L2 distance matrix.
        diff = va[:, None, :] - vb[None, :, :]          # [n, m, d]
        D = np.sqrt((diff ** 2).sum(axis=-1))            # [n, m]
        n, m = D.shape
        # Uniform weights.
        w_a = np.ones(n, dtype=np.float64) / n
        w_b = np.ones(m, dtype=np.float64) / m
        # Exact EMD via assignment (valid when n == m; pad otherwise).
        size = max(n, m)
        D_sq = np.zeros((size, size), dtype=np.float64)
        D_sq[:n, :m] = D
        row_ind, col_ind = linear_sum_assignment(D_sq)
        cost = D_sq[row_ind, col_ind].sum() / size
        return float(cost)

    scores = np.zeros((len(exploit_texts), len(candidate_texts)), dtype=np.float32)
    e_vecs = [_get_vecs(_tok(t)[:max_tokens]) for t in exploit_texts]
    c_vecs = [_get_vecs(_tok(t)[:max_tokens]) for t in candidate_texts]
    for i, va in enumerate(e_vecs):
        for j, vb in enumerate(c_vecs):
            scores[i, j] = -_wmd_approx(va, vb)
    return scores


# ---------------------------------------------------------------------------
# Interaction-based: ESIM-Lite
# ---------------------------------------------------------------------------

class ESIMLite(nn.Module):
    """Simplified ESIM: embed → BiLSTM → soft-align → enhance → compose → MLP."""

    def __init__(self, config: Config):
        super().__init__()
        from ..rt2.trigram_hasher import PAD_ID

        self.pad = PAD_ID
        d = config.cte_emb_dim
        h = config.cte_lstm_hidden
        self.emb = nn.Embedding(config.trigram_buckets, d, padding_idx=PAD_ID)
        self.encoder = nn.LSTM(d, h, batch_first=True, bidirectional=True)
        self.compose = nn.LSTM(4 * 2 * h, h, batch_first=True, bidirectional=True)
        # MLP scorer: mean+max over both sides → scalar
        self.mlp = nn.Linear(4 * 2 * h, 1)

    def _encode(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (lstm_out [B, L, 2H], mask [B, L])."""
        mask = (ids != self.pad)
        x = self.emb(ids)
        out, _ = self.encoder(x)
        return out, mask

    def _soft_align(
        self, e: Tensor, f: Tensor, mask_e: Tensor, mask_f: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Cross-attention: A_ij = e_i · f_j."""
        A = torch.bmm(e, f.transpose(1, 2))  # [B, Le, Lf]
        # Mask padding before softmax.
        inf_mask_f = (~mask_f).unsqueeze(1).float() * -1e9
        inf_mask_e = (~mask_e).unsqueeze(2).float() * -1e9
        alpha = torch.softmax(A + inf_mask_f, dim=2)   # [B, Le, Lf]
        beta  = torch.softmax(A + inf_mask_e, dim=1).transpose(1, 2)  # [B, Lf, Le]
        e_tilde = torch.bmm(alpha, f)  # [B, Le, 2H]
        f_tilde = torch.bmm(beta,  e)  # [B, Lf, 2H]
        return e_tilde, f_tilde

    def _enhance(self, x: Tensor, x_tilde: Tensor) -> Tensor:
        """[x; x_tilde; x-x_tilde; x*x_tilde]."""
        return torch.cat([x, x_tilde, x - x_tilde, x * x_tilde], dim=-1)

    def _pool(self, h: Tensor, mask: Tensor) -> Tensor:
        """Mean-pool + max-pool → concat."""
        mask_f = mask.float().unsqueeze(-1)
        mean = (h * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        # replace padding with large negative before max-pool
        h_masked = h.masked_fill(~mask.unsqueeze(-1), -1e9)
        mx = h_masked.max(1).values
        return torch.cat([mean, mx], dim=-1)

    def score(self, e_ids: Tensor, v_ids: Tensor) -> Tensor:
        """Returns scalar scores [B]."""
        e_out, e_mask = self._encode(e_ids)
        f_out, f_mask = self._encode(v_ids)
        e_tilde, f_tilde = self._soft_align(e_out, f_out, e_mask, f_mask)
        v_e = self._enhance(e_out, e_tilde)
        v_f = self._enhance(f_out, f_tilde)
        # Composition BiLSTM.
        c_e, _ = self.compose(v_e)
        c_f, _ = self.compose(v_f)
        pe = self._pool(c_e, e_mask)
        pf = self._pool(c_f, f_mask)
        combined = torch.cat([pe, pf], dim=-1)
        return self.mlp(combined).squeeze(-1)


def _esim_triplet_train(
    model: ESIMLite,
    pairs: list[EVPair],
    config: Config,
    device: torch.device,
    margin: float = 0.3,
) -> None:
    from ..rt2.trigram_hasher import tokenize_cte

    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=config.cte_lr)
    positives = [p for p in pairs if p.label == 1]
    negatives = [p for p in pairs if p.label == 0]
    if not positives or not negatives:
        return
    bs = min(config.cte_batch_size, len(positives))

    def _tok_ids(texts: list[str]) -> Tensor:
        return torch.tensor(
            [tokenize_cte(t, config.trigram_buckets, config.cte_max_len) for t in texts],
            device=device,
        )

    for _ in range(config.cte_finetune_epochs):
        np.random.shuffle(positives)  # type: ignore[arg-type]
        for start in range(0, len(positives), bs):
            batch_pos = positives[start: start + bs]
            batch_neg = negatives[: len(batch_pos)]
            e_ids  = _tok_ids([p.exploit_text         for p in batch_pos])
            vp_ids = _tok_ids([p.vulnerability_text   for p in batch_pos])
            vn_ids = _tok_ids([p.vulnerability_text   for p in batch_neg])
            s_pos = model.score(e_ids, vp_ids)
            s_neg = model.score(e_ids, vn_ids)
            loss = torch.relu(margin - s_pos + s_neg).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()


def esim_score_matrix(
    ev_pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
) -> np.ndarray:
    from ..rt2.trigram_hasher import tokenize_cte

    model = ESIMLite(config)
    _esim_triplet_train(model, ev_pairs, config, device)
    model.eval()

    scores = np.zeros((len(exploit_texts), len(candidate_texts)), dtype=np.float32)
    bs = config.cte_batch_size
    with torch.no_grad():
        e_ids_all = torch.tensor(
            [tokenize_cte(t, config.trigram_buckets, config.cte_max_len) for t in exploit_texts],
            device=device,
        )
        c_ids_all = torch.tensor(
            [tokenize_cte(t, config.trigram_buckets, config.cte_max_len) for t in candidate_texts],
            device=device,
        )
        for i in range(len(exploit_texts)):
            e_rep = e_ids_all[i: i + 1].expand(len(candidate_texts), -1)
            row_scores = []
            for start in range(0, len(candidate_texts), bs):
                e_batch = e_rep[start: start + bs]
                c_batch = c_ids_all[start: start + bs]
                row_scores.append(model.score(e_batch, c_batch).cpu().numpy())
            scores[i] = np.concatenate(row_scores)
    return scores


# ---------------------------------------------------------------------------
# Interaction-based: MatchPyramid
# ---------------------------------------------------------------------------

class MatchPyramid(nn.Module):
    """Pairwise cosine similarity matrix → 2D-CNN → score."""

    def __init__(self, config: Config):
        super().__init__()
        from ..rt2.trigram_hasher import PAD_ID

        self.pad = PAD_ID
        self.max_len = min(config.cte_max_len, 32)
        d = config.cte_emb_dim
        self.emb = nn.Embedding(config.trigram_buckets, d, padding_idx=PAD_ID)
        self.conv = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.pool = nn.AdaptiveMaxPool2d((4, 4))
        self.fc = nn.Linear(16 * 4 * 4, 1)

    def _match_matrix(self, e_ids: Tensor, v_ids: Tensor) -> Tensor:
        """Build cosine similarity matrix M [B, 1, L, L]."""
        xe = self.emb(e_ids)  # [B, L, d]
        xv = self.emb(v_ids)
        xe = nn.functional.normalize(xe, dim=-1)
        xv = nn.functional.normalize(xv, dim=-1)
        M = torch.bmm(xe, xv.transpose(1, 2))  # [B, L, L]
        return M.unsqueeze(1)                   # [B, 1, L, L]

    def score(self, e_ids: Tensor, v_ids: Tensor) -> Tensor:
        M = self._match_matrix(e_ids, v_ids)
        h = torch.relu(self.conv(M))   # [B, 16, L, L]
        h = self.pool(h)               # [B, 16, 4, 4]
        h = h.flatten(1)               # [B, 256]
        return self.fc(h).squeeze(-1)  # [B]


def _mp_triplet_train(
    model: MatchPyramid,
    pairs: list[EVPair],
    config: Config,
    device: torch.device,
    margin: float = 0.3,
) -> None:
    from ..rt2.trigram_hasher import tokenize_cte

    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=config.cte_lr)
    positives = [p for p in pairs if p.label == 1]
    negatives = [p for p in pairs if p.label == 0]
    if not positives or not negatives:
        return
    bs = min(config.cte_batch_size, len(positives))
    ml = model.max_len

    def _tok_ids(texts: list[str]) -> Tensor:
        return torch.tensor(
            [tokenize_cte(t, config.trigram_buckets, ml) for t in texts],
            device=device,
        )

    for _ in range(config.cte_finetune_epochs):
        np.random.shuffle(positives)  # type: ignore[arg-type]
        for start in range(0, len(positives), bs):
            batch_pos = positives[start: start + bs]
            batch_neg = negatives[: len(batch_pos)]
            e_ids  = _tok_ids([p.exploit_text       for p in batch_pos])
            vp_ids = _tok_ids([p.vulnerability_text for p in batch_pos])
            vn_ids = _tok_ids([p.vulnerability_text for p in batch_neg])
            s_pos = model.score(e_ids, vp_ids)
            s_neg = model.score(e_ids, vn_ids)
            loss = torch.relu(margin - s_pos + s_neg).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()


def match_pyramid_score_matrix(
    ev_pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
) -> np.ndarray:
    from ..rt2.trigram_hasher import tokenize_cte

    model = MatchPyramid(config)
    _mp_triplet_train(model, ev_pairs, config, device)
    model.eval()
    ml = model.max_len
    bs = config.cte_batch_size

    scores = np.zeros((len(exploit_texts), len(candidate_texts)), dtype=np.float32)
    with torch.no_grad():
        e_ids_all = torch.tensor(
            [tokenize_cte(t, config.trigram_buckets, ml) for t in exploit_texts],
            device=device,
        )
        c_ids_all = torch.tensor(
            [tokenize_cte(t, config.trigram_buckets, ml) for t in candidate_texts],
            device=device,
        )
        for i in range(len(exploit_texts)):
            row_scores = []
            e_rep = e_ids_all[i: i + 1].expand(len(candidate_texts), -1)
            for start in range(0, len(candidate_texts), bs):
                e_batch = e_rep[start: start + bs]
                c_batch = c_ids_all[start: start + bs]
                row_scores.append(model.score(e_batch, c_batch).cpu().numpy())
            scores[i] = np.concatenate(row_scores)
    return scores


# ---------------------------------------------------------------------------
# Interaction-based: DSSM
# ---------------------------------------------------------------------------

class DSSM(nn.Module):
    """Dual-encoder MLP (DSSM) with trigram-hash input."""

    def __init__(self, config: Config):
        super().__init__()
        B = config.trigram_buckets
        self._tower = nn.Sequential(
            nn.Linear(B, 300), nn.Tanh(),
            nn.Linear(300, 300), nn.Tanh(),
            nn.Linear(300, 128),
        )
        # Separate towers share architecture but have independent weights.
        self.exploit_tower = nn.Sequential(
            nn.Linear(B, 300), nn.Tanh(),
            nn.Linear(300, 300), nn.Tanh(),
            nn.Linear(300, 128),
        )
        self.vuln_tower = nn.Sequential(
            nn.Linear(B, 300), nn.Tanh(),
            nn.Linear(300, 300), nn.Tanh(),
            nn.Linear(300, 128),
        )
        self._B = B

    def _text_to_bow(self, ids: Tensor) -> Tensor:
        """Convert token id sequences to a multi-hot bag-of-buckets vector."""
        B_size = self._B
        bow = torch.zeros(ids.shape[0], B_size, device=ids.device, dtype=torch.float32)
        bow.scatter_add_(1, ids.clamp(min=0), torch.ones_like(ids, dtype=torch.float32))
        return bow

    def encode_exploit(self, ids: Tensor) -> Tensor:
        return nn.functional.normalize(self.exploit_tower(self._text_to_bow(ids)), dim=-1)

    def encode_vuln(self, ids: Tensor) -> Tensor:
        return nn.functional.normalize(self.vuln_tower(self._text_to_bow(ids)), dim=-1)


def dssm_score_matrix(
    ev_pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
) -> np.ndarray:
    """DSSM with in-batch NCE training."""
    from ..rt2.trigram_hasher import tokenize_cte

    model = DSSM(config).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=config.cte_lr)

    positives = [p for p in ev_pairs if p.label == 1]
    if not positives:
        positives = ev_pairs
    bs = min(config.cte_batch_size, max(2, len(positives)))

    def _tok_ids(texts: list[str]) -> Tensor:
        return torch.tensor(
            [tokenize_cte(t, config.trigram_buckets, config.cte_max_len) for t in texts],
            device=device,
        )

    for _ in range(config.cte_pretrain_epochs):
        np.random.shuffle(positives)  # type: ignore[arg-type]
        for start in range(0, len(positives), bs):
            batch = positives[start: start + bs]
            if len(batch) < 2:
                continue
            e_ids = _tok_ids([p.exploit_text       for p in batch])
            v_ids = _tok_ids([p.vulnerability_text for p in batch])
            z_e = model.encode_exploit(e_ids)   # [B, 128]
            z_v = model.encode_vuln(v_ids)       # [B, 128]
            # In-batch NCE: cosine sim matrix scaled by temperature.
            sim = z_e @ z_v.T / config.cte_contrastive_temp  # [B, B]
            labels = torch.arange(len(batch), device=device)
            loss = nn.functional.cross_entropy(sim, labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

    model.eval()
    scores = np.zeros((len(exploit_texts), len(candidate_texts)), dtype=np.float32)
    encode_bs = config.cte_batch_size
    with torch.no_grad():
        e_ids_all = _tok_ids(exploit_texts)
        c_ids_all = _tok_ids(candidate_texts)
        A_parts, B_parts = [], []
        for start in range(0, len(exploit_texts), encode_bs):
            A_parts.append(model.encode_exploit(e_ids_all[start: start + encode_bs]).cpu().numpy())
        for start in range(0, len(candidate_texts), encode_bs):
            B_parts.append(model.encode_vuln(c_ids_all[start: start + encode_bs]).cpu().numpy())
    A = np.concatenate(A_parts, axis=0)
    B = np.concatenate(B_parts, axis=0)
    return A @ B.T


# ---------------------------------------------------------------------------
# Stronger supervised ML: XGBoost-style (GradientBoosting on SVD features)
# ---------------------------------------------------------------------------

def xgboost_score_matrix(
    vec,
    exploit_texts: list[str],
    candidate_texts: list[str],
    positive_indices: np.ndarray,
) -> np.ndarray:
    """GradientBoostingClassifier on SVD-reduced element-wise product features."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import GradientBoostingClassifier

    # SVD-reduce the TF-IDF matrix.
    all_texts = list(dict.fromkeys([*exploit_texts, *candidate_texts]))
    M_all = vec.transform(all_texts)
    n_components = min(50, min(M_all.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    svd.fit(M_all)

    X_e = svd.transform(vec.transform(exploit_texts))  # [n_e, 50]
    X_v = svd.transform(vec.transform(candidate_texts))  # [n_c, 50]

    # Build training set.
    rng = np.random.RandomState(42)
    rows, labels = [], []
    n_c = len(candidate_texts)
    for i in range(len(exploit_texts)):
        pos_j = int(positive_indices[i])
        rows.append(X_e[i] * X_v[pos_j])
        labels.append(1)
        neg_j = rng.randint(0, n_c)
        while neg_j == pos_j:
            neg_j = rng.randint(0, n_c)
        rows.append(X_e[i] * X_v[neg_j])
        labels.append(0)

    X_train = np.array(rows, dtype=np.float32)
    y_train = np.array(labels, dtype=np.int32)

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=0)
    clf.fit(X_train, y_train)

    scores = np.zeros((len(exploit_texts), len(candidate_texts)), dtype=np.float32)
    for i in range(len(exploit_texts)):
        feats = X_e[i][None, :] * X_v  # [n_c, 50]
        scores[i] = clf.predict_proba(feats)[:, 1]
    return scores


# ---------------------------------------------------------------------------
# Stronger supervised ML: hand-crafted feature GradientBoosting
# ---------------------------------------------------------------------------

def _jaccard_tok(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / (union + 1e-12)


def _char3gram_jaccard(a: str, b: str) -> float:
    def _g(t: str) -> set[str]:
        return {t[k: k + 3] for k in range(len(t) - 2)}
    ga, gb = _g(a.lower()), _g(b.lower())
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / (union + 1e-12)


def _cosine_raw_tf(a: list[str], b: list[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    vocab = set(ca) | set(cb)
    if not vocab:
        return 0.0
    va = np.array([ca.get(w, 0) for w in vocab], dtype=np.float32)
    vb = np.array([cb.get(w, 0) for w in vocab], dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((va @ vb) / (na * nb))


def _hand_features(ta: list[str], tb: list[str], raw_a: str, raw_b: str) -> list[float]:
    la, lb = len(ta), len(tb)
    lmax = max(la, lb, 1)
    lcs = _lcs_length(ta, tb)
    feat = [
        _jaccard_tok(ta, tb),
        _char3gram_jaccard(raw_a, raw_b),
        min(la, lb) / lmax,
        len(set(ta) & set(tb)) / lmax,
        _cosine_raw_tf(ta, tb),
        lcs / lmax,
    ]
    return feat


def hand_feature_gb_score_matrix(
    exploit_texts: list[str],
    candidate_texts: list[str],
    positive_indices: np.ndarray,
    ev_pairs: list[EVPair] | None = None,
) -> np.ndarray:
    """GradientBoosting on 6-dim hand-crafted features."""
    from sklearn.ensemble import GradientBoostingClassifier

    e_tok = [_tok(t) for t in exploit_texts]
    c_tok = [_tok(t) for t in candidate_texts]

    rng = np.random.RandomState(42)
    rows, labels = [], []
    n_c = len(candidate_texts)
    for i in range(len(exploit_texts)):
        pos_j = int(positive_indices[i])
        rows.append(_hand_features(e_tok[i], c_tok[pos_j], exploit_texts[i], candidate_texts[pos_j]))
        labels.append(1)
        neg_j = rng.randint(0, n_c)
        while neg_j == pos_j:
            neg_j = rng.randint(0, n_c)
        rows.append(_hand_features(e_tok[i], c_tok[neg_j], exploit_texts[i], candidate_texts[neg_j]))
        labels.append(0)

    X_train = np.array(rows, dtype=np.float32)
    y_train = np.array(labels, dtype=np.int32)

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=0)
    clf.fit(X_train, y_train)

    scores = np.zeros((len(exploit_texts), len(candidate_texts)), dtype=np.float32)
    for i in range(len(exploit_texts)):
        feats = np.array(
            [_hand_features(e_tok[i], c_tok[j], exploit_texts[i], candidate_texts[j])
             for j in range(n_c)],
            dtype=np.float32,
        )
        scores[i] = clf.predict_proba(feats)[:, 1]
    return scores


# ---------------------------------------------------------------------------
# Neural encoder: GloVe-style (co-occurrence + SVD)
# ---------------------------------------------------------------------------

def glove_mean_score_matrix(
    ev_pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
) -> np.ndarray:
    """GloVe-style embeddings via co-occurrence matrix + TruncatedSVD.

    Steps:
      1. Build global co-occurrence matrix (bigrams) over all ev_pairs texts.
      2. Apply f(x) = min(1, (x/100)^0.75) weighting.
      3. TruncatedSVD to get word embeddings.
      4. Mean-pool for each text.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import TruncatedSVD

    all_texts = list({p.exploit_text for p in ev_pairs} | {p.vulnerability_text for p in ev_pairs})

    # Unigram count matrix for co-occurrence proxy.
    cv = CountVectorizer(tokenizer=_tok, lowercase=False, token_pattern=None, min_df=1)
    X = cv.fit_transform(all_texts)  # [n_docs, V]
    vocab = cv.get_feature_names_out()
    V = len(vocab)

    # Co-occurrence: X^T X (word × word co-occurrence across documents).
    cooc = (X.T @ X).toarray().astype(np.float32)
    np.fill_diagonal(cooc, 0.0)

    # GloVe weighting: f(x) = min(1, (x/x_max)^0.75).
    x_max = 100.0
    weights = np.minimum(1.0, (cooc / x_max) ** 0.75)
    weighted = weights * np.log1p(cooc)

    n_components = min(config.cte_emb_dim, V - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    word_vecs = svd.fit_transform(weighted)  # [V, d]

    word2idx = {w: i for i, w in enumerate(vocab)}

    def _mean_pool(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), n_components), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = [t for t in _tok(text) if t in word2idx]
            if toks:
                out[i] = word_vecs[[word2idx[t] for t in toks]].mean(axis=0)
        return out

    A = _mean_pool(exploit_texts)
    B = _mean_pool(candidate_texts)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return A @ B.T


# ---------------------------------------------------------------------------
# Neural encoder: Text Autoencoder
# ---------------------------------------------------------------------------

class TextAutoencoder(nn.Module):
    """Embed → mean-pool → MLP encoder → latent → MLP decoder → reconstruct."""

    def __init__(self, config: Config):
        super().__init__()
        from ..rt2.trigram_hasher import PAD_ID

        self.pad = PAD_ID
        d = config.cte_emb_dim
        h = config.cte_emb_dim
        latent = config.cte_emb_dim // 2

        self.emb = nn.Embedding(config.trigram_buckets, d, padding_idx=PAD_ID)
        self.encoder_net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(),
            nn.Linear(h, latent),
        )
        self.decoder_net = nn.Sequential(
            nn.Linear(latent, h), nn.ReLU(),
            nn.Linear(h, d),
        )
        self.latent_dim = latent

    def _mean_pool(self, ids: Tensor) -> Tensor:
        mask = (ids != self.pad).float().unsqueeze(-1)  # [B, L, 1]
        x = self.emb(ids)                               # [B, L, d]
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled

    def encode(self, ids: Tensor) -> Tensor:
        return self.encoder_net(self._mean_pool(ids))

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (latent [B, latent], reconstructed [B, d])."""
        z = self.encode(ids)
        recon = self.decoder_net(z)
        target = self._mean_pool(ids).detach()
        return z, recon, target


def autoencoder_score_matrix(
    ev_pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
) -> np.ndarray:
    from ..rt2.trigram_hasher import tokenize_cte

    model = TextAutoencoder(config).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=config.cte_lr)

    all_texts = list({p.exploit_text for p in ev_pairs} | {p.vulnerability_text for p in ev_pairs})
    if not all_texts:
        all_texts = exploit_texts + candidate_texts

    bs = config.cte_batch_size
    ids_all = torch.tensor(
        [tokenize_cte(t, config.trigram_buckets, config.cte_max_len) for t in all_texts],
    )

    for _ in range(config.cte_pretrain_epochs):
        perm = np.random.permutation(len(all_texts))
        for start in range(0, len(all_texts), bs):
            idx = perm[start: start + bs]
            batch = ids_all[idx].to(device)
            _, recon, target = model(batch)
            loss = nn.functional.mse_loss(recon, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

    A = _encode_all(model, model.encode, exploit_texts, config, device)
    B = _encode_all(model, model.encode, candidate_texts, config, device)
    return A @ B.T


# ---------------------------------------------------------------------------
# Neural encoder: Text VAE
# ---------------------------------------------------------------------------

class TextVAE(nn.Module):
    """Variational autoencoder over mean-pooled token embeddings."""

    def __init__(self, config: Config):
        super().__init__()
        from ..rt2.trigram_hasher import PAD_ID

        self.pad = PAD_ID
        d = config.cte_emb_dim
        h = config.cte_emb_dim
        latent = config.cte_emb_dim // 2

        self.emb = nn.Embedding(config.trigram_buckets, d, padding_idx=PAD_ID)
        self.enc_shared = nn.Sequential(nn.Linear(d, h), nn.ReLU())
        self.mu_head     = nn.Linear(h, latent)
        self.logvar_head = nn.Linear(h, latent)
        self.decoder_net = nn.Sequential(
            nn.Linear(latent, h), nn.ReLU(),
            nn.Linear(h, d),
        )
        self.latent_dim = latent

    def _mean_pool(self, ids: Tensor) -> Tensor:
        mask = (ids != self.pad).float().unsqueeze(-1)
        x = self.emb(ids)
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1)

    def encode(self, ids: Tensor) -> Tensor:
        """Returns μ (deterministic at inference)."""
        h = self.enc_shared(self._mean_pool(ids))
        return self.mu_head(h)

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns (mu, logvar, recon, target) for loss computation."""
        pooled = self._mean_pool(ids)
        h = self.enc_shared(pooled)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        # Reparameterization.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        recon = self.decoder_net(z)
        target = pooled.detach()
        return mu, logvar, recon, target


def vae_score_matrix(
    ev_pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
    beta: float = 0.1,
) -> np.ndarray:
    from ..rt2.trigram_hasher import tokenize_cte

    model = TextVAE(config).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=config.cte_lr)

    all_texts = list({p.exploit_text for p in ev_pairs} | {p.vulnerability_text for p in ev_pairs})
    if not all_texts:
        all_texts = exploit_texts + candidate_texts

    bs = config.cte_batch_size
    ids_all = torch.tensor(
        [tokenize_cte(t, config.trigram_buckets, config.cte_max_len) for t in all_texts],
    )

    for _ in range(config.cte_pretrain_epochs):
        perm = np.random.permutation(len(all_texts))
        for start in range(0, len(all_texts), bs):
            idx = perm[start: start + bs]
            batch = ids_all[idx].to(device)
            mu, logvar, recon, target = model(batch)
            recon_loss = nn.functional.mse_loss(recon, target)
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss = recon_loss + beta * kl_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

    A = _encode_all(model, model.encode, exploit_texts, config, device)
    B = _encode_all(model, model.encode, candidate_texts, config, device)
    return A @ B.T


# ---------------------------------------------------------------------------
# Neural encoder: SiameseBiLSTM with hard negative mining
# ---------------------------------------------------------------------------

def siamese_bilstm_hardneg_score_matrix(
    ev_pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
) -> np.ndarray:
    """Same SiameseBiLSTM architecture but trained with in-batch hard negatives."""
    from ..rt2.trigram_hasher import tokenize_cte

    model = SiameseBiLSTM(config).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=config.cte_lr)
    margin = 0.3

    positives = [p for p in ev_pairs if p.label == 1]
    if not positives:
        positives = ev_pairs
    bs = min(config.cte_batch_size, max(2, len(positives)))

    def _tok_ids(texts: list[str]) -> Tensor:
        return torch.tensor(
            [tokenize_cte(t, config.trigram_buckets, config.cte_max_len) for t in texts],
            device=device,
        )

    for _ in range(config.cte_finetune_epochs):
        np.random.shuffle(positives)  # type: ignore[arg-type]
        for start in range(0, len(positives), bs):
            batch = positives[start: start + bs]
            if len(batch) < 2:
                continue
            e_ids = _tok_ids([p.exploit_text       for p in batch])
            v_ids = _tok_ids([p.vulnerability_text for p in batch])

            z_e = nn.functional.normalize(model.encode(e_ids), dim=-1)  # [B, d]
            z_v = nn.functional.normalize(model.encode(v_ids), dim=-1)  # [B, d]

            # Pairwise cosine sim between all (e_i, v_j) in batch.
            sim_matrix = z_e @ z_v.T  # [B, B]
            B_sz = sim_matrix.shape[0]

            sim_pos = sim_matrix.diag()  # [B] — positive pairs
            # For each anchor, mask out the diagonal and pick hardest negative.
            mask_diag = torch.eye(B_sz, device=device, dtype=torch.bool)
            sim_negs = sim_matrix.masked_fill(mask_diag, -1e9)
            sim_neg_hard = sim_negs.max(dim=1).values  # [B]

            loss = torch.relu(margin - sim_pos + sim_neg_hard).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

    A = _encode_all(model, model.encode, exploit_texts, config, device)
    B = _encode_all(model, model.encode, candidate_texts, config, device)
    return A @ B.T


# ---------------------------------------------------------------------------
# Neural ranking: KNRM
# ---------------------------------------------------------------------------

class KNRM(nn.Module):
    """Kernel-based Neural Ranking Model."""

    def __init__(self, config: Config, n_kernels: int = 11, sigma: float = 0.1):
        super().__init__()
        from ..rt2.trigram_hasher import PAD_ID

        self.pad = PAD_ID
        self.n_kernels = n_kernels
        self.sigma = sigma
        d = config.cte_emb_dim
        self.emb = nn.Embedding(config.trigram_buckets, d, padding_idx=PAD_ID)
        # Fixed kernel means μ_k evenly spaced in [-1, 1].
        mus = torch.linspace(-1.0, 1.0, n_kernels)
        self.register_buffer("mus", mus)
        # Learnable final linear layer.
        self.w = nn.Linear(n_kernels, 1, bias=False)

    def _kernel_score(self, M: Tensor, mask_q: Tensor, mask_d: Tensor) -> Tensor:
        """
        M      : [B, Lq, Ld] cosine similarity matrix
        mask_q : [B, Lq]
        mask_d : [B, Ld]
        Returns [B, K] kernel features.
        """
        mus = self.mus.view(1, 1, 1, self.n_kernels)     # [1,1,1,K]
        M_exp = M.unsqueeze(-1)                            # [B, Lq, Ld, 1]
        K = torch.exp(-(M_exp - mus) ** 2 / (2 * self.sigma ** 2))  # [B, Lq, Ld, K]
        # Mask document padding.
        K = K * mask_d.unsqueeze(1).unsqueeze(-1).float()
        # Log-sum over document tokens per query token.
        soft_match = torch.log1p(K.sum(dim=2))  # [B, Lq, K]
        # Mask query padding then sum over query tokens.
        soft_match = soft_match * mask_q.unsqueeze(-1).float()
        return soft_match.sum(dim=1)  # [B, K]

    def score(self, q_ids: Tensor, d_ids: Tensor) -> Tensor:
        mask_q = (q_ids != self.pad)
        mask_d = (d_ids != self.pad)
        xq = nn.functional.normalize(self.emb(q_ids), dim=-1)  # [B, Lq, d]
        xd = nn.functional.normalize(self.emb(d_ids), dim=-1)  # [B, Ld, d]
        M = torch.bmm(xq, xd.transpose(1, 2))                  # [B, Lq, Ld]
        feats = self._kernel_score(M, mask_q, mask_d)           # [B, K]
        return self.w(feats).squeeze(-1)                        # [B]


def knrm_score_matrix(
    ev_pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
) -> np.ndarray:
    from ..rt2.trigram_hasher import tokenize_cte

    model = KNRM(config).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=config.cte_lr)
    margin = 0.3

    positives = [p for p in ev_pairs if p.label == 1]
    negatives = [p for p in ev_pairs if p.label == 0]
    if not positives or not negatives:
        return np.zeros((len(exploit_texts), len(candidate_texts)), dtype=np.float32)

    bs = min(config.cte_batch_size, len(positives))

    def _tok_ids(texts: list[str]) -> Tensor:
        return torch.tensor(
            [tokenize_cte(t, config.trigram_buckets, config.cte_max_len) for t in texts],
            device=device,
        )

    for _ in range(config.cte_finetune_epochs):
        np.random.shuffle(positives)  # type: ignore[arg-type]
        for start in range(0, len(positives), bs):
            batch_pos = positives[start: start + bs]
            batch_neg = negatives[: len(batch_pos)]
            e_ids  = _tok_ids([p.exploit_text       for p in batch_pos])
            vp_ids = _tok_ids([p.vulnerability_text for p in batch_pos])
            vn_ids = _tok_ids([p.vulnerability_text for p in batch_neg])
            s_pos = model.score(e_ids, vp_ids)
            s_neg = model.score(e_ids, vn_ids)
            loss = torch.relu(margin - s_pos + s_neg).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

    model.eval()
    scores = np.zeros((len(exploit_texts), len(candidate_texts)), dtype=np.float32)
    encode_bs = config.cte_batch_size
    with torch.no_grad():
        e_ids_all = _tok_ids(exploit_texts)
        c_ids_all = _tok_ids(candidate_texts)
        for i in range(len(exploit_texts)):
            row = []
            e_rep = e_ids_all[i: i + 1].expand(len(candidate_texts), -1)
            for start in range(0, len(candidate_texts), encode_bs):
                e_b = e_rep[start: start + encode_bs]
                c_b = c_ids_all[start: start + encode_bs]
                row.append(model.score(e_b, c_b).cpu().numpy())
            scores[i] = np.concatenate(row)
    return scores


# ---------------------------------------------------------------------------
# CTE ablation: no temporal (random negatives in fine-tuning)
# ---------------------------------------------------------------------------

def cte_no_temporal_score_matrix(
    ev_pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
) -> np.ndarray:
    """Full CTE (pretrain + finetune) but with random (not time-aware) negatives."""
    from ..rt2.cte_model import CTE
    from ..rt2.finetune_cte import finetune_cte
    from ..rt2.pretrain_cte import pretrain_cte
    from ..rt2.trigram_hasher import tokenize_cte

    model = CTE(config).to(device)
    pretrain_cte(model, ev_pairs, config, device)
    finetune_cte(model, ev_pairs, config, device, use_time_aware_negatives=False)

    def _encode(texts: list[str]) -> np.ndarray:
        model.eval()
        parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), config.cte_batch_size):
                batch = torch.tensor(
                    [tokenize_cte(t, config.trigram_buckets, config.cte_max_len)
                     for t in texts[start: start + config.cte_batch_size]],
                    device=device,
                )
                z = nn.functional.normalize(model.encode_sentence(batch), dim=-1)
                parts.append(z.cpu().numpy())
        return np.concatenate(parts, axis=0)

    A = _encode(exploit_texts)
    B = _encode(candidate_texts)
    return A @ B.T


# ---------------------------------------------------------------------------
# CTE ablation: separate exploit and vulnerability encoders (bi-encoder)
# ---------------------------------------------------------------------------

def cte_shared_encoder_score_matrix(
    ev_pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
) -> np.ndarray:
    """Bi-encoder with SEPARATE weights for exploit and vulnerability towers.

    Both encoders share the CTE architecture but are initialised independently
    and trained with InfoNCE (in-batch contrastive), giving an asymmetric
    dual-encoder as opposed to CTE's single shared encoder.
    """
    from ..rt2.cte_model import CTE
    from ..rt2.trigram_hasher import tokenize_cte

    exploit_enc = CTE(config).to(device)
    vuln_enc    = CTE(config).to(device)

    params = list(exploit_enc.parameters()) + list(vuln_enc.parameters())
    opt = torch.optim.Adam(params, lr=config.cte_lr)

    positives = [p for p in ev_pairs if p.label == 1]
    if not positives:
        positives = ev_pairs
    bs = min(config.cte_batch_size, max(2, len(positives)))

    def _tok_ids(texts: list[str]) -> Tensor:
        return torch.tensor(
            [tokenize_cte(t, config.trigram_buckets, config.cte_max_len) for t in texts],
            device=device,
        )

    exploit_enc.train(); vuln_enc.train()
    total_epochs = config.cte_pretrain_epochs + config.cte_finetune_epochs
    for _ in range(total_epochs):
        np.random.shuffle(positives)  # type: ignore[arg-type]
        for start in range(0, len(positives), bs):
            batch = positives[start: start + bs]
            if len(batch) < 2:
                continue
            e_ids = _tok_ids([p.exploit_text       for p in batch])
            v_ids = _tok_ids([p.vulnerability_text for p in batch])
            z_e = nn.functional.normalize(exploit_enc.encode_sentence(e_ids), dim=-1)
            z_v = nn.functional.normalize(vuln_enc.encode_sentence(v_ids),   dim=-1)
            sim = z_e @ z_v.T / config.cte_contrastive_temp
            labels = torch.arange(len(batch), device=device)
            loss = nn.functional.cross_entropy(sim, labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

    def _encode(enc: CTE, texts: list[str]) -> np.ndarray:
        enc.eval()
        parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), config.cte_batch_size):
                batch = _tok_ids(texts[start: start + config.cte_batch_size])
                z = nn.functional.normalize(enc.encode_sentence(batch), dim=-1)
                parts.append(z.cpu().numpy())
        return np.concatenate(parts, axis=0)

    A = _encode(exploit_enc, exploit_texts)
    B = _encode(vuln_enc,    candidate_texts)
    return A @ B.T


# ---------------------------------------------------------------------------
# CTE ablation: temperature sensitivity
# ---------------------------------------------------------------------------

def cte_temperature_score_matrix(
    ev_pairs: list[EVPair],
    exploit_texts: list[str],
    candidate_texts: list[str],
    config: Config,
    device: torch.device,
    tau: float = 0.07,
) -> np.ndarray:
    """Full CTE (pretrain + finetune) with a specified InfoNCE temperature τ.

    Call with tau ∈ {0.01, 0.07, 0.2} to probe temperature sensitivity.
    Uses dataclasses.replace to override cte_contrastive_temp.
    """
    from ..rt2.cte_model import CTE
    from ..rt2.pretrain_cte import pretrain_cte
    from ..rt2.finetune_cte import finetune_cte
    from ..rt2.trigram_hasher import tokenize_cte

    tau_config = dataclasses.replace(config, cte_contrastive_temp=tau)
    model = CTE(tau_config).to(device)
    pretrain_cte(model, ev_pairs, tau_config, device)
    finetune_cte(model, ev_pairs, tau_config, device)

    def _encode(texts: list[str]) -> np.ndarray:
        model.eval()
        parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), tau_config.cte_batch_size):
                batch = torch.tensor(
                    [tokenize_cte(t, tau_config.trigram_buckets, tau_config.cte_max_len)
                     for t in texts[start: start + tau_config.cte_batch_size]],
                    device=device,
                )
                z = nn.functional.normalize(model.encode_sentence(batch), dim=-1)
                parts.append(z.cpu().numpy())
        return np.concatenate(parts, axis=0)

    A = _encode(exploit_texts)
    B = _encode(candidate_texts)
    return A @ B.T
