#!/usr/bin/env python3


# https://graphsinspace.net
# https://tigraphs.pmf.uns.ac.rs


from __future__ import annotations

import argparse
import itertools
import random
import sys
import types
from pathlib import Path

import typing as _typing
import typing_extensions as _typing_extensions

if not hasattr(_typing_extensions, "TypeIs"):
    if hasattr(_typing, "TypeIs"):
        _typing_extensions.TypeIs = _typing.TypeIs  # type: ignore[attr-defined]
    else:
        _typing_extensions.TypeIs = _typing.TypeGuard  # type: ignore[attr-defined]

import numpy as np
import pandas as pd

if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined, assignment]

REPO_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "graspe" / "src" / "graspe"
if not REPO_ROOT.is_dir():
    raise SystemExit(
        f"Nedostaje klon graspe: {REPO_ROOT}\n"
        "git clone https://github.com/graphsinspace/graspe.git third_party/graspe"
    )

if "dgl" not in sys.modules:
    _dgl = types.ModuleType("dgl")

    class _DGLGraph:
        pass

    _dgl.DGLGraph = _DGLGraph
    _dgl.from_networkx = lambda *a, **k: None
    sys.modules["dgl"] = _dgl

sys.path.insert(0, str(REPO_ROOT))

import networkx as nx  # noqa: E402
from common.graph import Graph as GraspeGraph  # noqa: E402
from embeddings.embedding_randw import UnbiasedWalk  # noqa: E402

_DATA = Path(__file__).resolve().parents[1] / "data"
DEFAULT_CSV = _DATA / "loto7hh_4586_k24.csv"
DEFAULT_COMBOS = _DATA / "kombinacijeH_39C7.csv"
SEED = 39

TUNED_DECAY = 0.999
TUNED_DIM = 32
TUNED_PATH_NUMBER = 55
TUNED_PATH_LENGTH = 14
TUNED_NUM_WALKS_UNBIASED = 56
TUNED_WALK_LENGTH_UNBIASED = 14
TUNED_W2V_EPOCHS = 18

NODE_ORDER = list(range(1, 40))


def load_draws(csv_path: Path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    cols = [f"Num{i}" for i in range(1, 8)]
    if all(c in df.columns for c in cols):
        use = cols
    else:
        use = list(df.columns[:7])
    draws = []
    for _, row in df.iterrows():
        draws.append(sorted(int(row[c]) for c in use))
    return draws


def dynamic_pair_weights(draws: list[list[int]], decay: float) -> dict[tuple[int, int], float]:
    T = len(draws)
    acc: dict[tuple[int, int], float] = {}
    for t, nums in enumerate(draws):
        w = float(decay) ** (T - 1 - t)
        for u, v in itertools.combinations(nums, 2):
            a, b = (u, v) if u < v else (v, u)
            acc[(a, b)] = acc.get((a, b), 0.0) + w
    return acc


def marginal_node_weights(draws: list[list[int]], decay: float) -> dict[int, float]:
    T = len(draws)
    acc = {n: 0.0 for n in range(1, 40)}
    for t, nums in enumerate(draws):
        w = float(decay) ** (T - 1 - t)
        for n in nums:
            acc[n] += w
    return acc


def build_graspe_graph(pair_w: dict[tuple[int, int], float]) -> GraspeGraph:
    G = GraspeGraph()
    for i in range(1, 40):
        G.add_node(i)
    for (u, v), w in pair_w.items():
        if w <= 0:
            continue
        G.add_edge(u, v, weight=w)
        G.add_edge(v, u, weight=w)
    return G


def _to_undirected_nx(G: GraspeGraph) -> nx.Graph:
    return G.to_networkx().to_undirected()


def deepwalk_corpus_nx(
    G_undir: nx.Graph,
    path_number: int,
    path_length: int,
    rng: random.Random,
) -> list[list[str]]:
    nodes = list(G_undir.nodes())
    walks: list[list[str]] = []
    for _ in range(path_number):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            for _ in range(path_length - 1):
                nbrs = list(G_undir.neighbors(walk[-1]))
                if not nbrs:
                    break
                walk.append(rng.choice(nbrs))
            walks.append([str(x) for x in walk])
    return walks


def _word2vec_from_walks(walks: list[list[str]], dim: int, seed: int) -> dict[int, np.ndarray]:
    import gensim
    from gensim.models import Word2Vec

    major = int(gensim.__version__.split(".")[0])
    workers = 1
    if major >= 4:
        model = Word2Vec(
            walks,
            vector_size=dim,
            window=5,
            min_count=0,
            sg=1,
            workers=workers,
            seed=seed,
            epochs=TUNED_W2V_EPOCHS,
        )
    else:
        model = Word2Vec(
            walks,
            size=dim,
            window=5,
            min_count=0,
            sg=1,
            workers=workers,
            seed=seed,
        )
    wv = model.wv
    out: dict[int, np.ndarray] = {}
    for n in range(1, 40):
        s = str(n)
        out[n] = np.asarray(wv[s], dtype=np.float64) if s in wv else np.zeros(dim, dtype=np.float64)
    return out


def run_deepwalk_nx(
    G: GraspeGraph, dim: int, path_number: int, path_length: int, seed: int
) -> dict[int, np.ndarray]:
    rng = random.Random(seed)
    walks = deepwalk_corpus_nx(_to_undirected_nx(G), path_number, path_length, rng)
    return _word2vec_from_walks(walks, dim, seed)


def _patch_randw_word2vec_epochs() -> None:
    import gensim
    from gensim.models import Word2Vec

    import embeddings.embedding_randw as rw

    def embed_patched(self) -> None:
        rw.Embedding.embed(self)
        walks = self.simulate_walks()
        walks = [list(map(str, w)) for w in walks]
        major = int(gensim.__version__.split(".")[0])
        workers = max(1, int(self._workers))
        if major >= 4:
            model = Word2Vec(
                sentences=walks,
                vector_size=self._d,
                min_count=0,
                sg=1,
                workers=workers,
                seed=int(self._seed),
                epochs=TUNED_W2V_EPOCHS,
            )
        else:
            model = Word2Vec(
                sentences=walks,
                size=self._d,
                min_count=0,
                sg=1,
                workers=workers,
                seed=int(self._seed),
            )
        self._embedding = {}
        for node in self._g.nodes():
            self._embedding[node[0]] = np.asarray(
                model.wv[str(node[0])], dtype=np.float64
            )

    rw.RWEmbBase.embed = embed_patched


_patch_randw_word2vec_epochs()


def run_unbiased_walk(
    G: GraspeGraph, dim: int, num_walks: int, walk_length: int, seed: int
) -> dict[int, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    emb = UnbiasedWalk(
        G,
        d=dim,
        num_walks=num_walks,
        walk_length=walk_length,
        workers=1,
        seed=seed,
    )
    emb.embed()
    return {n: emb[n].copy() for n in range(1, 40)}


def l2_normalize_rows(vecs: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    out = {}
    for k, v in vecs.items():
        n = np.linalg.norm(v)
        out[k] = (v / n).astype(np.float64) if n > 1e-12 else v.copy()
    return out


def ensemble_vectors(
    a: dict[int, np.ndarray], b: dict[int, np.ndarray], dim: int
) -> dict[int, np.ndarray]:
    a_n = l2_normalize_rows(a)
    b_n = l2_normalize_rows(b)
    return {n: (a_n[n] + b_n[n]) / 2.0 for n in range(1, 40)}


def vectors_to_matrix(vectors: dict[int, np.ndarray]) -> np.ndarray:
    return np.stack([vectors[n].astype(np.float64, copy=False) for n in NODE_ORDER])


def fit_cluster_labels(
    X: np.ndarray,
    clusterer: str,
    k: int,
    seed: int,
    *,
    scale: bool,
) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
    from sklearn.preprocessing import StandardScaler

    if k < 2:
        raise ValueError("--k mora biti >= 2")
    if X.shape[0] < k:
        raise ValueError(f"premalo čvorova ({X.shape[0]}) za k={k}")

    Xf = StandardScaler().fit_transform(X) if scale else X.copy()
    rng = int(seed)
    c = clusterer.lower().strip()
    if c == "kmeans":
        model = KMeans(n_clusters=k, random_state=rng, n_init="auto")
    elif c == "agglomerative":
        model = AgglomerativeClustering(n_clusters=k)
    elif c == "spectral":
        model = SpectralClustering(
            n_clusters=k, random_state=rng, affinity="nearest_neighbors"
        )
    else:
        raise ValueError(clusterer)
    return model.fit_predict(Xf)


def labels_to_clusters(labels: np.ndarray, nodes: list[int]) -> dict[int, list[int]]:
    m: dict[int, list[int]] = {}
    for n, lab in zip(nodes, labels):
        m.setdefault(int(lab), []).append(int(n))
    for v in m.values():
        v.sort()
    return m


def diversified_seven(
    clusters: dict[int, list[int]],
    marginal: dict[int, float],
    *,
    want_distinct_clusters: int = 7,
) -> tuple[int, ...]:
    """
    Pokušaj sedmorke sa čim više različitih klastera; unutar klastera max marginal.
    Ako ima < 7 klastera sa čvorovima, dopuni globalno po marginalu.
    """
    nonempty = {c: ns[:] for c, ns in clusters.items() if ns}
    used: set[int] = set()
    picked: list[int] = []

    def best_in_cluster(cid: int) -> int | None:
        pool = [n for n in nonempty[cid] if n not in used]
        if not pool:
            return None
        return max(pool, key=lambda n: (marginal.get(n, 0.0), -n))

    if len(nonempty) >= want_distinct_clusters:
        ranked_c = sorted(
            nonempty.keys(),
            key=lambda c: (-sum(marginal.get(n, 0.0) for n in nonempty[c]), c),
        )
        for cid in ranked_c[:want_distinct_clusters]:
            b = best_in_cluster(cid)
            if b is not None:
                picked.append(b)
                used.add(b)
    else:
        for cid in sorted(
            nonempty.keys(),
            key=lambda c: (-max(marginal.get(n, 0.0) for n in nonempty[c]), c),
        ):
            b = best_in_cluster(cid)
            if b is not None:
                picked.append(b)
                used.add(b)

    rest = [n for n in NODE_ORDER if n not in used]
    rest.sort(key=lambda n: (-marginal.get(n, 0.0), n))
    for n in rest:
        if len(picked) >= 7:
            break
        picked.append(n)
        used.add(n)

    if len(picked) < 7:
        raise RuntimeError("nije moguće skupiti 7 čvorova (neočekivano)")

    return tuple(sorted(picked[:7]))


def main():
    ap = argparse.ArgumentParser(
        description="GRASP loto grupa 5: klasterovanje embeddinga + diverzifikovana sedmorka"
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--combos", type=Path, default=DEFAULT_COMBOS)
    ap.add_argument("--decay", type=float, default=TUNED_DECAY)
    ap.add_argument("--dim", type=int, default=TUNED_DIM)
    ap.add_argument(
        "--embed",
        choices=("unbiased", "deepwalk", "ensemble"),
        default="ensemble",
        help="Izvor vektora pre klasterovanja (ensemble = L2-prosek DW+UB kao graspe2)",
    )
    ap.add_argument("--path-number", type=int, default=TUNED_PATH_NUMBER)
    ap.add_argument("--path-length", type=int, default=TUNED_PATH_LENGTH)
    ap.add_argument("--num-walks", type=int, default=TUNED_NUM_WALKS_UNBIASED)
    ap.add_argument("--walk-length", type=int, default=TUNED_WALK_LENGTH_UNBIASED)
    ap.add_argument(
        "--clusterer",
        choices=("kmeans", "agglomerative", "spectral"),
        default="kmeans",
        help="Isti skup kao u graspe evaluation/clustering.py (agglomerative|kmeans|spectral)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=7,
        help="Broj klastera (k=7 omogućava jednu sedmorku iz 7 različitih klastera ako su svi neprazni)",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--scale",
        action="store_true",
        help="StandardScaler pre klasterovanja",
    )
    ap.add_argument(
        "--show-clusters",
        action="store_true",
        help="Ispiši mapiranje čvor → klaster",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    draws = load_draws(args.csv)
    pair_w = dynamic_pair_weights(draws, args.decay)
    marginal = marginal_node_weights(draws, args.decay)
    G = build_graspe_graph(pair_w)

    if args.embed == "deepwalk":
        vectors = run_deepwalk_nx(
            G, args.dim, args.path_number, args.path_length, args.seed
        )
        emb_label = "DeepWalk (NX+W2V)"
    elif args.embed == "unbiased":
        vectors = run_unbiased_walk(
            G, args.dim, args.num_walks, args.walk_length, args.seed
        )
        emb_label = "UnbiasedWalk"
    else:
        v_dw = run_deepwalk_nx(
            G, args.dim, args.path_number, args.path_length, args.seed
        )
        v_ub = run_unbiased_walk(
            G, args.dim, args.num_walks, args.walk_length, args.seed
        )
        vectors = ensemble_vectors(v_dw, v_ub, args.dim)
        emb_label = "Ensemble (DW+UB)"

    X = vectors_to_matrix(vectors)
    labels = fit_cluster_labels(
        X, args.clusterer, args.k, args.seed, scale=bool(args.scale)
    )
    clusters = labels_to_clusters(labels, NODE_ORDER)

    try:
        from sklearn.metrics import silhouette_score

        sil = float(silhouette_score(X, labels)) if len(set(labels)) >= 2 else float("nan")
    except Exception:
        sil = float("nan")

    combo = diversified_seven(clusters, marginal, want_distinct_clusters=7)

    print(f"CSV izvučenih: {args.csv.resolve()}")
    print(f"CSV svih komb.: {args.combos.resolve()}  (postoji: {args.combos.is_file()})")
    print(f"Izvlačenja: {len(draws)} | parova: {len(pair_w)} | decay={args.decay}")
    print(f"graspe: {REPO_ROOT}")
    sil_s = f"{sil:.4f}" if np.isfinite(sil) else "n/a"
    print(
        f"Embedding: {emb_label} | klasterovanje: {args.clusterer} | k={args.k} | "
        f"silhouette≈{sil_s}"
    )

    if args.show_clusters:
        print("\nČvor → klaster:")
        for n in NODE_ORDER:
            print(f"  {n:2d} → {int(labels[n - 1])}")
        print("\nKlaster → čvorovi:")
        for c in sorted(clusters.keys()):
            print(f"  {c}: {clusters[c]}")

    print()
    print("Predikcija (diverzifikacija po klasterima + marginal unutar klastera):")
    print(list(combo))
    print()


if __name__ == "__main__":
    main()




"""
python3 graspe5_loto_clustering.py

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Embedding: Ensemble (DW+UB) | klasterovanje: kmeans | k=7 | silhouette≈0.0702

Predikcija (diverzifikacija po klasterima + marginal unutar klastera):
[8, 22, x, y, z, 37, 39]
"""




"""
python3 graspe5_loto_clustering.py --clusterer agglomerative --k 10 --show-clusters

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Embedding: Ensemble (DW+UB) | klasterovanje: agglomerative | k=10 | silhouette≈0.0825

Čvor → klaster:
   1 → 2
   2 → 7
   3 → 3
   4 → 5
   5 → 0
   6 → 3
   7 → 3
   8 → 3
   9 → 5
  10 → 6
  11 → 3
  12 → 3
  13 → 6
  14 → 0
  15 → 4
  16 → 2
  17 → 1
  18 → 2
  19 → 3
  20 → 8
  21 → 4
  22 → 5
  23 → 6
  24 → 7
  25 → 2
  26 → 7
  27 → 6
  28 → x
  29 → 1
  30 → 5
  31 → 7
  32 → 0
  33 → 2
  34 → 1
  35 → 4
  36 → 3
  37 → 7
  38 → 0
  39 → 0

Klaster → čvorovi:
  0: [5, 14, x, 38, 39]
  1: [17, 29, 34]
  2: [1, 16, x, 25, 33]
  3: [3, 6, 7, x, y, z, 19, 36]
  4: [15, 21, 35]
  5: [4, 9, 22, 30]
  6: [10, 13, 23, 27]
  7: [2, 24, x, 31, 37]
  8: [20]
  9: [28]

Predikcija (diverzifikacija po klasterima + marginal unutar klastera):
[8, 22, x, y, z, 34, 37]
"""




"""
python3 graspe5_loto_clustering.py --embed deepwalk --scale

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Embedding: DeepWalk (NX+W2V) | klasterovanje: kmeans | k=7 | silhouette≈0.0676

Predikcija (diverzifikacija po klasterima + marginal unutar klastera):
[8, 23, x, y, z, 37, 39]
"""




"""
python3 graspe5_loto_clustering.py --embed deepwalk --show-clusters

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Embedding: DeepWalk (NX+W2V) | klasterovanje: kmeans | k=7 | silhouette≈0.0892

Čvor → klaster:
   1 → 0
   2 → 3
   3 → 2
   4 → 0
   5 → 2
   6 → 1
   7 → 1
   8 → 1
   9 → 3
  10 → 5
  11 → 4
  12 → 2
  13 → 4
  14 → 3
  15 → 1
  16 → 4
  17 → x
  18 → 0
  19 → 1
  20 → 2
  21 → 1
  22 → 0
  23 → 6
  24 → 2
  25 → 0
  26 → 3
  27 → 4
  28 → 1
  29 → 4
  30 → 3
  31 → 1
  32 → 5
  33 → 4
  34 → 1
  35 → 1
  36 → 4
  37 → 3
  38 → 2
  39 → 2

Klaster → čvorovi:
  0: [1, 4, x, 22, 25]
  1: [6, 7, 8, x, y, z, 28, 31, 34, 35]
  2: [3, 5, x, y, z, 38, 39]
  3: [2, 9, 14, 26, 30, 37]
  4: [11, 13, x, y, z, 33, 36]
  5: [10, 17, 32]
  6: [23]

Predikcija (diverzifikacija po klasterima + marginal unutar klastera):
[10, 23, x, y, z, 37, 39]
"""




"""
python3 graspe5_loto_clustering.py --clusterer spectral --k 9 --scale

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
Embedding: Ensemble (DW+UB) | klasterovanje: spectral | k=9 | silhouette≈0.0316

Predikcija (diverzifikacija po klasterima + marginal unutar klastera):
[8, 11, x, y, z, 34, 37]
"""




################################################################################




"""
K-means (i opciono agglomerative/spectral) na embeddingu čvorova 1-39, 
zatim pravilo sedmorke — po jedan broj iz različitih klastera gde je moguće, 
sa marginom iz istorije za izbor unutar klastera, kao u graspe2/graspe3.

Embedding (pre klasterovanja): --embed unbiased | deepwalk | ensemble (podrazumevano ensemble = L2-prosek DW+UB kao u graspe2.
Klasterovanje (isti skup kao u evaluation/clustering.py): kmeans, agglomerative, spectral; --k broj klastera (podrazumevano 7 da može sedmorka iz 7 različitih klastera).
Opcije: --scale (StandardScaler), --seed, --show-clusters (čvor→klaster i klaster→čvorovi).
Sedmorka: što više različitih klastera; u svakom klasteru čvor sa najvećom marginalnom težinom iz istorije; ako ima < 7 klastera sa čvorovima, dopuna po globalnom marginalu.
Silhouette se ispisuje kad ga sklearn može izračunati.


Klasterovanje sedmorke — --cluster on
   KMeans na vektorima prvog seed-a; 
   best_combo_from_scores_cluster_cap 
   (max po klasteru + opciono min različitih).


Klasterovanje (evaluation/clustering.py, clustering_eval.py)
K-means (i slično) na vektorima čvorova.
→ Ne „predviđa“ loto, ali može pravilo za sedmorku: 
npr. biraj brojeve iz različitih klastera 
da kombinacija nije sve iz jednog oblaka u embeddingu 
— diverzifikacija tiketa. 
Unutar klastera bira se čvor sa većom marginalnom težinom iz istorije.


GRASP loto — **grupa 5 (klasterovanje)**, po tački 2 u `graspe3_loto_fusion_kom.py`.

K-means / Agglomerative / Spectral na vektorima čvorova (1-39), 
u duhu `third_party/graspe/src/graspe/evaluation/clustering.py` 
i `clustering_eval.py` (sklearn). 


Zahtevi: numpy, pandas, networkx, gensim, sklearn; torch samo ako se ne koristi čist DeepWalk/Unbiased 
preko graspe RandW (isti put kao graspe2).
"""
