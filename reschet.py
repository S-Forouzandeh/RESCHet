"""
RESCHet: Recommendation Based on Embedding Spectral Clustering
in Heterogeneous Networks

Reference:
    Forouzandeh, S., Berahmand, K., Sheikhpour, R., & Li, Y. (2023).
    A new method for recommendation based on embedding spectral clustering
    in heterogeneous networks (RESCHet).
    Expert Systems with Applications, 231, 120699.
    https://doi.org/10.1016/j.eswa.2023.120699

Dataset / original repo: https://github.com/S-Forouzandeh/RESCHet
"""

import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Heterogeneous Information Network  (Definition 1 & 2)
# ─────────────────────────────────────────────────────────────────────────────

class HIN:
    """
    Heterogeneous Information Network  G = (V, E).

    Nodes carry a *type* label (e.g. 'user', 'movie', 'director').
    Edges are stored as typed adjacency lists keyed by
    (src_type, relation, dst_type) triples.  Reverse edges are added
    automatically so random walks can traverse in both directions.
    """

    def __init__(self):
        self.node_type: dict[str, str] = {}
        # adj[(src_type, rel, dst_type)][src_id] = [dst_id, ...]
        self.adj: dict[tuple, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

    # ── building ──────────────────────────────────────────────────────────────

    def add_node(self, node_id: str, ntype: str) -> None:
        self.node_type[node_id] = ntype

    def add_edge(self, src: str, dst: str, rel: str) -> None:
        """Add a directed edge and its symmetric reverse."""
        src_t = self.node_type[src]
        dst_t = self.node_type[dst]
        self.adj[(src_t, rel, dst_t)][src].append(dst)
        self.adj[(dst_t, rel + "_inv", src_t)][dst].append(src)

    # ── querying ──────────────────────────────────────────────────────────────

    def neighbors(self, node: str, rel_triple: tuple) -> list:
        """All neighbours reachable from *node* via *rel_triple*."""
        return self.adj[rel_triple].get(node, [])

    def nodes_of_type(self, ntype: str) -> list:
        return [n for n, t in self.node_type.items() if t == ntype]

    def info(self) -> None:
        type_counts: dict[str, int] = defaultdict(int)
        for t in self.node_type.values():
            type_counts[t] += 1
        edge_counts = {
            k: sum(len(v) for v in adj.values())
            for k, adj in self.adj.items()
        }
        print("  Node types :", dict(type_counts))
        print("  Edge triples:", {str(k): v for k, v in edge_counts.items()})


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Metapath2Vec  (Definition 5 — Dong et al. 2017)
# ─────────────────────────────────────────────────────────────────────────────

class MetaPath2Vec:
    """
    Meta-path guided random walks on a HIN followed by a Word2Vec
    skip-gram model to produce node embeddings.

    A *meta_path* is a list of relation triples that defines the walk
    pattern, e.g.::

        [('user','rates','movie'), ('movie','rates_inv','user')]

    The walk repeats the pattern cyclically until *walk_length* is reached.
    """

    def __init__(
        self,
        hin: HIN,
        embedding_dim: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        window: int = 5,
        workers: int = 4,
        epochs: int = 10,
        seed: int = 42,
    ):
        self.hin = hin
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window = window
        self.workers = workers
        self.epochs = epochs
        self.seed = seed
        self.model: Word2Vec | None = None

    # ── random walk ──────────────────────────────────────────────────────────

    def _walk(self, start: str, meta_path: list) -> list:
        walk = [start]
        node = start
        path_len = len(meta_path)
        step = 0
        while len(walk) < self.walk_length:
            triple = meta_path[step % path_len]
            nbrs = self.hin.neighbors(node, triple)
            if not nbrs:
                break
            node = random.choice(nbrs)
            walk.append(node)
            step += 1
        return walk

    def _generate_walks(self, start_type: str, meta_path: list) -> list:
        seeds = self.hin.nodes_of_type(start_type)
        walks = []
        for _ in range(self.num_walks):
            random.shuffle(seeds)
            for s in seeds:
                w = self._walk(s, meta_path)
                if len(w) > 1:
                    walks.append(w)
        return walks

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, start_type: str, meta_paths: list) -> "MetaPath2Vec":
        """
        Train on walks drawn from every meta-path in *meta_paths*.
        All walks share a single Word2Vec model (heterogeneous skip-gram).
        """
        random.seed(self.seed)
        all_walks: list[list[str]] = []
        for mp in meta_paths:
            all_walks.extend(self._generate_walks(start_type, mp))
        random.shuffle(all_walks)

        self.model = Word2Vec(
            sentences=all_walks,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=1,
            sg=1,           # skip-gram as in the paper
            workers=self.workers,
            epochs=self.epochs,
            seed=self.seed,
        )
        return self

    # ── retrieval ─────────────────────────────────────────────────────────────

    def embeddings(self, nodes: list) -> dict[str, np.ndarray]:
        """Return {node_id: vector} for each node present in the vocabulary."""
        if self.model is None:
            raise RuntimeError("Call .train() first.")
        return {n: self.model.wv[n] for n in nodes if n in self.model.wv}

    @staticmethod
    def as_matrix(
        emb_dict: dict[str, np.ndarray],
        node_list: list,
        dim: int,
    ) -> np.ndarray:
        """Stack embeddings into (n, dim); fill zeros for absent nodes."""
        mat = np.zeros((len(node_list), dim))
        for i, n in enumerate(node_list):
            if n in emb_dict:
                mat[i] = emb_dict[n]
        return mat


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Non-linear Fusion  (Eq. 7)
# ─────────────────────────────────────────────────────────────────────────────

class NonLinearFusion:
    """
    Fuses L meta-path embedding matrices with the function::

        g({e^(l)}) = σ( Σ_l  w^l · σ(M^l · e^(l) + b^l) )

    Transformation matrices M^l and biases b^l are drawn once and held
    fixed (unsupervised setting).  Replace with a learnable nn.Module
    for end-to-end training.
    """

    def __init__(self, embedding_dim: int, num_paths: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.Ms = [
            rng.standard_normal((embedding_dim, embedding_dim)) * 0.01
            for _ in range(num_paths)
        ]
        self.bs = [np.zeros(embedding_dim) for _ in range(num_paths)]
        self.D = embedding_dim
        self.L = num_paths

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def fuse(self, path_mats: list[np.ndarray]) -> np.ndarray:
        """
        path_mats : list of L arrays, each (n_nodes, D).
        Returns fused matrix of shape (n_nodes, D).
        """
        transformed = [
            self._sigmoid(mat @ self.Ms[l].T + self.bs[l])
            for l, mat in enumerate(path_mats)
        ]
        # Uniform attention weights (could be learned with supervision)
        stacked = np.stack(transformed, axis=1)          # (n, L, D)
        weights = np.ones(self.L, dtype=float) / self.L  # (L,)
        weighted = np.einsum("l,nld->nd", weights, stacked)  # (n, D)
        return self._sigmoid(weighted)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Embedding Spectral Clustering  (Section 4.2)
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingSpectralClustering:
    """
    Clusters nodes via the pipeline described in Section 4.2:

    1. Build cosine similarity matrix S from fused embeddings.
    2. Compute graph Laplacian  L = D − S  (Eqs. 4–6).
    3. Decompose L; use eigengap heuristic to choose k.
    4. Apply K-means on the k leading eigenvectors.
    """

    def __init__(self, max_k: int = 20, random_state: int = 42):
        self.max_k = max_k
        self.random_state = random_state
        self.labels_: np.ndarray | None = None
        self.k_: int | None = None

    @staticmethod
    def _eigengap(eigenvalues: np.ndarray) -> int:
        """Return the k that maximises the spectral gap (eigengap heuristic)."""
        sv = np.sort(np.abs(eigenvalues))
        gaps = np.diff(sv)
        k = int(np.argmax(gaps)) + 1
        return max(2, k)

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        embeddings : (n_nodes, D)
        Returns integer cluster label array of shape (n_nodes,).
        """
        # ── Step 1: cosine similarity matrix (non-negative) ─────────────────
        S = cosine_similarity(embeddings)
        np.fill_diagonal(S, 0.0)
        S = np.clip(S, 0.0, 1.0)

        # ── Step 2: Laplacian  L = D − S ────────────────────────────────────
        d = S.sum(axis=1)
        L = np.diag(d) - S

        # ── Step 3: k smallest eigenpairs ────────────────────────────────────
        n = embeddings.shape[0]
        k_req = min(self.max_k + 1, n - 1)
        try:
            eigenvalues, eigenvectors = eigsh(
                csr_matrix(L), k=k_req, which="SM", tol=1e-6
            )
        except Exception:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            eigenvalues = eigenvalues[:k_req]
            eigenvectors = eigenvectors[:, :k_req]

        # ── Step 4: eigengap → k ─────────────────────────────────────────────
        self.k_ = min(self._eigengap(eigenvalues), self.max_k)

        # ── Step 5: K-means on k-dimensional spectral space ──────────────────
        sorted_idx = np.argsort(eigenvalues)
        feature_mat = eigenvectors[:, sorted_idx[: self.k_]]
        kmeans = KMeans(
            n_clusters=self.k_,
            random_state=self.random_state,
            n_init=10,
        )
        self.labels_ = kmeans.fit_predict(feature_mat)
        return self.labels_


# ─────────────────────────────────────────────────────────────────────────────
# 5.  RESCHet — main model  (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

class RESCHet:
    """
    RESCHet: Recommendation Based on Embedding Spectral Clustering
    in Heterogeneous Networks.

    Algorithm overview (Algorithm 1 in the paper):
    ──────────────────────────────────────────────
    For each meta-path in G_U, learn user & item embeddings via Metapath2vec.
    Fuse multi-path embeddings with a non-linear function.
    Perform Embedding Spectral Clustering on fused user vectors.
    For each cluster, compute Hadamard-product vectors from 3 submeta-path
    pairs (Eqs. 8–10) and generate recommendations via cosine similarity.
    """

    # Meta-paths used for the primary HIN embedding stage.
    # Defined for a U–M–D (user / movie / director) network schema.
    PRIMARY_META_PATHS = [
        # UMU : users sharing movies
        [("user", "rates", "movie"), ("movie", "rates_inv", "user")],
        # UMDU : users sharing directors via movies
        [
            ("user", "rates", "movie"),
            ("movie", "has_director", "director"),
            ("director", "has_director_inv", "movie"),
            ("movie", "rates_inv", "user"),
        ],
    ]

    # Submeta-path pairs  (Section 4.3, Eqs. 8–10).
    # Each entry is (path_A, path_B); E = emb(A) ⊙ emb(B).
    SUBMETA_PAIRS = [
        # Pair 1 — UMU / MUM
        (
            [("user", "rates", "movie"), ("movie", "rates_inv", "user")],
            [("user", "rates", "movie"), ("movie", "rates_inv", "user")],
        ),
        # Pair 2 — UDU / MDM  (user→director→user  /  movie→director→movie)
        (
            [
                ("user", "rates", "movie"),
                ("movie", "has_director", "director"),
                ("director", "has_director_inv", "movie"),
                ("movie", "rates_inv", "user"),
            ],
            [
                ("user", "rates", "movie"),
                ("movie", "has_director", "director"),
                ("director", "has_director_inv", "movie"),
                ("movie", "rates_inv", "user"),
            ],
        ),
        # Pair 3 — UMDU / MDUM
        (
            [
                ("user", "rates", "movie"),
                ("movie", "has_director", "director"),
                ("director", "has_director_inv", "movie"),
                ("movie", "rates_inv", "user"),
            ],
            [("user", "rates", "movie"), ("movie", "rates_inv", "user")],
        ),
    ]

    def __init__(
        self,
        embedding_dim: int = 128,
        walk_length: int = 40,
        num_walks: int = 5,
        window: int = 5,
        max_k: int = 15,
        top_n: int = 10,
        workers: int = 4,
        epochs: int = 10,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window = window
        self.max_k = max_k
        self.top_n = top_n
        self.workers = workers
        self.epochs = epochs
        self.seed = seed

        # Set after fit()
        self.hin: HIN | None = None
        self.user_ids: list[str] = []
        self.item_ids: list[str] = []
        self.fused_embs: np.ndarray | None = None   # (n_users, D)
        self.cluster_labels: np.ndarray | None = None
        self.k_: int | None = None
        # cluster_id → (cluster_user_ids, fused_emb_matrix)
        self.cluster_data: dict[int, tuple] = {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _m2v(self, meta_paths: list) -> MetaPath2Vec:
        """Build and train a MetaPath2Vec model on *meta_paths*."""
        m = MetaPath2Vec(
            self.hin,
            embedding_dim=self.embedding_dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            window=self.window,
            workers=self.workers,
            epochs=self.epochs,
            seed=self.seed,
        )
        m.train("user", meta_paths)
        return m

    def _emb_mat(self, m2v: MetaPath2Vec, nodes: list) -> np.ndarray:
        return MetaPath2Vec.as_matrix(
            m2v.embeddings(nodes), nodes, self.embedding_dim
        )

    # ── Stage 1 & 2: heterogeneous embedding + non-linear fusion ─────────────

    def _build_fused_embeddings(self) -> np.ndarray:
        """
        Train one Metapath2vec per meta-path, collect user embeddings,
        then fuse with the non-linear function (Eq. 7).
        Returns fused matrix of shape (n_users, D).
        """
        per_path: list[np.ndarray] = []
        for mp in self.PRIMARY_META_PATHS:
            m2v = self._m2v([mp])
            per_path.append(self._emb_mat(m2v, self.user_ids))

        fusion = NonLinearFusion(self.embedding_dim, len(per_path), self.seed)
        return fusion.fuse(per_path)

    # ── Stage 3: spectral clustering ─────────────────────────────────────────

    def _cluster_users(self, fused: np.ndarray) -> np.ndarray:
        esc = EmbeddingSpectralClustering(
            max_k=self.max_k, random_state=self.seed
        )
        labels = esc.fit(fused)
        self.k_ = esc.k_
        return labels

    # ── Stage 4: per-cluster Hadamard embeddings ──────────────────────────────

    def _build_cluster_embeddings(self) -> dict[int, tuple]:
        """
        For each cluster and each submeta-path pair, compute
        the Hadamard product embedding (Eqs. 8–10).
        Accumulates 3 Hadamard matrices per cluster, averages them.

        Returns {cluster_id: (user_ids_in_cluster, emb_matrix)}
        """
        cluster_data: dict[int, tuple] = {}
        for c in np.unique(self.cluster_labels):
            mask = self.cluster_labels == c
            c_users = [u for u, m in zip(self.user_ids, mask) if m]
            if len(c_users) < 2:
                continue

            hadamard_accum: list[np.ndarray] = []
            for path_a, path_b in self.SUBMETA_PAIRS:
                mat_a = self._emb_mat(self._m2v([path_a]), c_users)
                mat_b = self._emb_mat(self._m2v([path_b]), c_users)
                hadamard_accum.append(mat_a * mat_b)  # element-wise ⊙

            # Average the 3 Hadamard vectors (Eqs. 8, 9, 10 combined)
            final_mat = np.mean(hadamard_accum, axis=0)  # (n_c, D)
            cluster_data[c] = (c_users, final_mat)

        return cluster_data

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, hin: HIN) -> "RESCHet":
        """
        Fit all stages of RESCHet on *hin*.
        """
        self.hin = hin
        self.user_ids = hin.nodes_of_type("user")
        self.item_ids = hin.nodes_of_type("movie")
        random.seed(self.seed)
        np.random.seed(self.seed)

        print(f"  Users: {len(self.user_ids)},  Items: {len(self.item_ids)}")

        print("  [1/3] HIN Embedding + Non-linear Fusion ...")
        self.fused_embs = self._build_fused_embeddings()

        print("  [2/3] Embedding Spectral Clustering ...")
        self.cluster_labels = self._cluster_users(self.fused_embs)
        print(f"        → {self.k_} clusters found.")

        print("  [3/3] Submeta-path Hadamard Embeddings ...")
        self.cluster_data = self._build_cluster_embeddings()
        print("  ✓ Training complete.\n")
        return self

    # ── recommend ────────────────────────────────────────────────────────────

    def recommend(
        self,
        target_user: str,
        user_item_ratings: dict[str, dict],
        top_n: int | None = None,
    ) -> list[str]:
        """
        Return a ranked list of up to *top_n* item IDs for *target_user*.

        Parameters
        ----------
        user_item_ratings : {user_id: {item_id: rating}}
            Historical ratings (any positive value signals interaction).
        """
        top_n = top_n or self.top_n
        if target_user not in self.user_ids:
            return []

        u_idx = self.user_ids.index(target_user)
        u_cluster = int(self.cluster_labels[u_idx])

        if u_cluster not in self.cluster_data:
            return []

        c_users, c_mat = self.cluster_data[u_cluster]
        if target_user not in c_users:
            return []

        u_local = c_users.index(target_user)
        u_vec = c_mat[u_local : u_local + 1]  # (1, D)

        # Cosine similarity to all other cluster members  (Eq. 3)
        sims = cosine_similarity(u_vec, c_mat)[0]
        sims[u_local] = -1.0  # exclude self

        # Collect items from the 5 most similar users
        k_nbrs = min(5, len(c_users) - 1)
        top_idx = np.argsort(sims)[::-1][:k_nbrs]
        similar_users = [c_users[i] for i in top_idx]

        already_seen = set(user_item_ratings.get(target_user, {}).keys())
        candidates: dict[str, float] = defaultdict(float)
        for sim_user in similar_users:
            for item, rating in user_item_ratings.get(sim_user, {}).items():
                if item not in already_seen:
                    weight = sims[c_users.index(sim_user)]
                    candidates[item] += weight * rating

        ranked = sorted(candidates, key=candidates.__getitem__, reverse=True)
        return ranked[:top_n]

    # ── evaluate ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        ratings_df: pd.DataFrame,
        train_ratios: list[float] | None = None,
        user_col: str = "userId",
        item_col: str = "movieId",
        rating_col: str = "rating",
    ) -> pd.DataFrame:
        """
        Evaluate rating-prediction accuracy (MAE / RMSE) across several
        train/test splits, matching Tables 1–3 in the paper.

        Prediction strategy: weighted average of training ratings,
        weighted by cosine similarity of fused user embeddings.
        """
        if train_ratios is None:
            train_ratios = [0.2, 0.4, 0.6, 0.8]

        # Build integer-indexed rating matrix
        users = list(ratings_df[user_col].unique())
        items = list(ratings_df[item_col].unique())
        u2i = {u: i for i, u in enumerate(users)}
        i2j = {m: j for j, m in enumerate(items)}
        n_u, n_i = len(users), len(items)

        R = np.zeros((n_u, n_i), dtype=np.float32)
        for row in ratings_df.itertuples(index=False):
            R[u2i[getattr(row, user_col)], i2j[getattr(row, item_col)]] = (
                getattr(row, rating_col)
            )

        # Map fitted user embeddings to the rating-matrix order
        id_to_emb: dict[str, np.ndarray] = {}
        if self.fused_embs is not None:
            for uid, emb in zip(self.user_ids, self.fused_embs):
                id_to_emb[uid] = emb

        results = []
        prev_mae = prev_rmse = None

        for ratio in train_ratios:
            idx = np.arange(n_u)
            train_idx, test_idx = train_test_split(
                idx, train_size=ratio, random_state=self.seed
            )
            R_train = R[train_idx]
            R_test = R[test_idx]

            # Similarity between test and train users
            train_user_ids = [users[i] for i in train_idx]
            test_user_ids = [users[i] for i in test_idx]

            if id_to_emb:
                train_mat = np.array([
                    id_to_emb.get(f"u_{u}", np.zeros(self.embedding_dim))
                    for u in train_user_ids
                ])
                test_mat = np.array([
                    id_to_emb.get(f"u_{u}", np.zeros(self.embedding_dim))
                    for u in test_user_ids
                ])
                sim = cosine_similarity(test_mat, train_mat)  # (n_test, n_train)
            else:
                sim = cosine_similarity(R_test, R_train)

            # Predict each observed rating in the test split
            y_true, y_pred = [], []
            for ti in range(R_test.shape[0]):
                rated_items = np.where(R_test[ti] > 0)[0]
                if rated_items.size == 0:
                    continue
                w = np.maximum(sim[ti], 0)  # non-negative weights
                denom = w.sum()
                if denom == 0:
                    continue
                preds = (w @ R_train) / denom  # (n_items,)
                y_true.extend(R_test[ti, rated_items].tolist())
                y_pred.extend(preds[rated_items].tolist())

            if not y_true:
                continue

            mae = mean_absolute_error(y_true, y_pred)
            rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)

            mae_imp = (
                (prev_mae - mae) / prev_mae * 100
                if prev_mae is not None
                else float("nan")
            )
            rmse_imp = (
                (prev_rmse - rmse) / prev_rmse * 100
                if prev_rmse is not None
                else float("nan")
            )

            results.append(
                {
                    "Train Ratio": f"{int(ratio * 100)}%",
                    "MAE": round(mae, 4),
                    "RMSE": round(rmse, 4),
                    "MAE Improve": f"{mae_imp:+.2f}%" if not np.isnan(mae_imp) else "—",
                    "RMSE Improve": f"{rmse_imp:+.2f}%" if not np.isnan(rmse_imp) else "—",
                }
            )
            prev_mae, prev_rmse = mae, rmse

        return pd.DataFrame(results)
