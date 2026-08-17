import numpy as np
import pennylane as qml
import random
import networkx as nx
import os


def decisions_generator(
    n_qubits: int,
    x_num: tuple[int, int],
    param_num_range: tuple[int, int],
    ent_num_range: tuple[int, int] = (0, 0),
    allowed_gates=None,
    coupling_map=None,
    num_circuits: int = 1,
    seed=None,
    shuffle=True,
    qubit_probs=None,
):
    if seed is not None:
        random.seed(seed)

    if allowed_gates is None:
        allowed_gates = ["RX", "RY", "RZ", "CNOT", "CZ"]


    if coupling_map is None:

        edges = [(i, j) for i in range(n_qubits) for j in range(n_qubits) if i != j]
    elif isinstance(coupling_map, nx.DiGraph):

        edges = list(coupling_map.edges())
    elif isinstance(coupling_map, nx.Graph):

        edges = list(coupling_map.to_directed().edges())
    else:


        edges = [tuple(edge) for edge in coupling_map]


    gate_aliases = {"CX": "CNOT", "ID": "I"}
    normalized_allowed = [gate_aliases.get(g.upper(), g.upper()) for g in allowed_gates]

    KNOWN_ENT_GATES = {"CNOT", "CZ", "CY", "SWAP", "ISWAP", "ECR", "MS", "RXX", "RZZ", "CRX", "CRY", "CRZ"}
    native_ent_gates = list(set(normalized_allowed).intersection(KNOWN_ENT_GATES))

    native_rotations = list(set(normalized_allowed).intersection({"RX", "RY", "RZ"}))
    if len(native_rotations) >= 2:
        virtual_param_single_gates = native_rotations
    else:
        virtual_param_single_gates = ["RX", "RY", "RZ"]

    KNOWN_NON_PARAM_ENT = {"CNOT", "CZ", "CY", "SWAP", "ISWAP", "ECR", "MS"}
    ent_candidates = [g for g in native_ent_gates if g in KNOWN_NON_PARAM_ENT]
    native_param_ent_gates = [g for g in native_ent_gates if g not in KNOWN_NON_PARAM_ENT]
    param_candidates = virtual_param_single_gates + native_param_ent_gates
    single_qubit_param_gates = set(virtual_param_single_gates)

    if ent_num_range[1] > 0 and not ent_candidates:
        raise ValueError(f"Error: requested a pure entangling gate, but no supported physical entangling gate was found in the native gate set {allowed_gates}!")
    if (x_num[1] > 0 or param_num_range[1] > 0) and not param_candidates:
        raise ValueError("Error: no usable parameterized gate set; please check the HC-VGS logic.")
    has_two_qubit_gate = len(ent_candidates) > 0 or len(native_param_ent_gates) > 0
    if has_two_qubit_gate and len(edges) == 0:
        raise ValueError("Error: the native gate set contains two-qubit entangling instructions, but the coupling_map is empty!")

    if qubit_probs is None:
        encode_weights = param_weights = [1.0 / n_qubits] * n_qubits
    elif qubit_probs == "random":
        encode_raw_weights = [random.random() for _ in range(n_qubits)]
        param_raw_weights = [random.random() for _ in range(n_qubits)]
        encode_weights = [w / sum(encode_raw_weights) for w in encode_raw_weights]
        param_weights = [w / sum(param_raw_weights) for w in param_raw_weights]
    else:
        encode_weights = param_weights = qubit_probs

    qubit_list = list(range(n_qubits))

    def sample_single_qubit(flag):
        weights = encode_weights if flag else param_weights
        return random.choices(qubit_list, weights=weights, k=1)[0]

    def sample_hardware_edge(flag):
        weights = encode_weights if flag else param_weights
        edge_weights = []
        for u, v in edges:

            edge_weights.append(weights[u] * weights[v])

        total_edge_weight = sum(edge_weights)
        if total_edge_weight > 1e-9:
            normalized_edge_weights = [ew / total_edge_weight for ew in edge_weights]
            chosen_edge = random.choices(edges, weights=normalized_edge_weights, k=1)[0]
        else:
            chosen_edge = random.choice(edges)

        c, t = chosen_edge
        return [c, t]

    print("Virtual gate set: ", param_candidates + ent_candidates)
    for _ in range(num_circuits):
        decisions = []
        n_x = random.randint(*x_num)
        n_p = random.randint(*param_num_range)
        n_ent = random.randint(*ent_num_range)

        for xi in range(n_x):
            gate = random.choice(param_candidates)
            if gate in single_qubit_param_gates:
                wires = [sample_single_qubit(True)]
            else:
                wires = sample_hardware_edge(True)
            decisions.append({"gate": gate, "wires": wires, "x_index": xi})

        for pi in range(n_p):
            gate = random.choice(param_candidates)
            if gate in single_qubit_param_gates:
                wires = [sample_single_qubit(False)]
            else:
                wires = sample_hardware_edge(False)
            decisions.append({"gate": gate, "wires": wires, "param_idx": pi})

        for _ in range(n_ent):
            gate = random.choice(ent_candidates)
            wires = sample_hardware_edge(False)
            decisions.append({"gate": gate, "wires": wires})

        if shuffle:
            random.shuffle(decisions)
        yield decisions


def make_shifted_circles(n_samples=2000, radii=[0.2, 0.6, 0.9], noise=0.01, offset=0.05, random_state=42):
    rng = np.random.RandomState(random_state)


    centers = rng.uniform(-offset, offset, size=(len(radii), 2))


    X = rng.uniform(-1, 1, (n_samples, 2))


    rs = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    radii = np.sort(radii)

    y = np.zeros(n_samples, dtype=int)
    for i, radius in enumerate(reversed(radii)):
        mask = rs[:, i] < radius
        y[mask] = i + 1


    if noise > 0:
        X = X + rng.normal(scale=noise, size=X.shape)

    return X, y


def get_cluster_hamiltonian(n_qubits: int):
    r"""
    H = - \sum_{i} Z_{i-1} X_i Z_{i+1}
    """
    obs = []
    coeffs = []


    for i in range(n_qubits):
        left = (i - 1) % n_qubits
        right = (i + 1) % n_qubits
        obs.append(qml.PauliZ(left) @ qml.PauliX(i) @ qml.PauliZ(right))
        coeffs.append(-1.0)

    return qml.Hamiltonian(coeffs, obs)


import pandas as pd


from sklearn.decomposition import PCA
import math
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, Sequence, List
from sklearn.base import BaseEstimator, TransformerMixin, clone

from sklearn.utils.validation import check_is_fitted


class NonlinearFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, scaler: Optional[MinMaxScaler] = None, copy: bool = True):
        self.scaler = scaler
        self.copy = copy

    def _apply_nonlinear(self, df: pd.DataFrame) -> pd.DataFrame:
        """ if """
        if "expressibility" in df.columns:
            df["expressibility"] = -np.log1p(-df["expressibility"] * 1e5)
        if "expressibility_embed" in df.columns:
            df["expressibility_embed"] = -np.log1p(-df["expressibility_embed"] * 1e5)
        if "width" in df.columns:
            df["width"] = df["width"].apply(math.log1p)
        if "encoding_entropy" in df.columns:
            df["encoding_entropy"] = np.log1p(df["encoding_entropy"] * 1e2)
        if "param_entropy" in df.columns:
            df["param_entropy"] = np.log1p(df["param_entropy"] * 1e2)
        if "global_entropy" in df.columns:
            df["global_entropy"] = np.log1p(df["global_entropy"] * 1e2)
        if "ent_entropy_universal" in df.columns:
            df["ent_entropy_universal"] = np.log1p(df["ent_entropy_universal"] * 1e1)
        if "ent_entropy_hardware" in df.columns:
            df["ent_entropy_hardware"] = np.log1p(df["ent_entropy_hardware"] * 1e1)
        if "ep_js_divergence" in df.columns:
            df["ep_js_divergence"] = np.log1p(df["ep_js_divergence"] * 1e1)
        if "snip" in df.columns:
            df["snip"] = np.log1p(df["snip"] * 1e2)
        if "lifespan_imbalance" in df.columns:
            df["lifespan_imbalance"] = np.log1p(df["lifespan_imbalance"])

        return df

    def _apply_inverse_nonlinear(self, df: pd.DataFrame) -> pd.DataFrame:
        """"""
        if "expressibility" in df.columns:
            df["expressibility"] = -np.expm1(-df["expressibility"]) / 1e5
        if "expressibility_embed" in df.columns:
            df["expressibility_embed"] = -np.expm1(-df["expressibility_embed"]) / 1e5
        if "width" in df.columns:
            df["width"] = np.expm1(df["width"])
        if "encoding_entropy" in df.columns:
            df["encoding_entropy"] = np.expm1(df["encoding_entropy"]) / 1e2
        if "param_entropy" in df.columns:
            df["param_entropy"] = np.expm1(df["param_entropy"]) / 1e2
        if "global_entropy" in df.columns:
            df["global_entropy"] = np.expm1(df["global_entropy"]) / 1e2
        if "ent_entropy_universal" in df.columns:
            df["ent_entropy_universal"] = np.expm1(df["ent_entropy_universal"]) / 1e1
        if "ent_entropy_hardware" in df.columns:
            df["ent_entropy_hardware"] = np.expm1(df["ent_entropy_hardware"]) / 1e1
        if "ep_js_divergence" in df.columns:
            df["ep_js_divergence"] = np.expm1(df["ep_js_divergence"]) / 1e1
        if "snip" in df.columns:
            df["snip"] = np.expm1(df["snip"]) / 1e2
        if "lifespan_imbalance" in df.columns:
            df["lifespan_imbalance"] = np.expm1(df["lifespan_imbalance"])

        return df

    def fit(self, X: pd.DataFrame, y=None):
        X_proc = X.copy()
        X_proc = self._apply_nonlinear(X_proc)


        self.cols_to_drop_ = []


        if "ent_entropy_hardware" in X_proc.columns and "ent_entropy_universal" in X_proc.columns:
            if np.allclose(X_proc["ent_entropy_hardware"], X_proc["ent_entropy_universal"], atol=1e-8):
                self.cols_to_drop_.append("ent_entropy_hardware")


        for col in X_proc.columns:
            if X_proc[col].nunique(dropna=False) <= 1:
                if col not in self.cols_to_drop_:
                    self.cols_to_drop_.append(col)


        self.feature_names_out_ = [c for c in X_proc.columns if c not in self.cols_to_drop_]


        self.scaler_ = clone(self.scaler) if self.scaler is not None else MinMaxScaler()
        self.scaler_.fit(X_proc[self.feature_names_out_])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "scaler_")
        X_proc = X.copy() if self.copy else X


        X_proc = self._apply_nonlinear(X_proc)


        X_proc.drop(columns=[c for c in self.cols_to_drop_ if c in X_proc.columns], inplace=True, errors="ignore")


        arr = self.scaler_.transform(X_proc[self.feature_names_out_])
        return pd.DataFrame(arr, index=X.index, columns=self.feature_names_out_)

    def inverse_transform(self, X_scaled: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "scaler_")


        arr_inv = self.scaler_.inverse_transform(X_scaled[self.feature_names_out_])
        df_inv = pd.DataFrame(arr_inv, index=X_scaled.index, columns=self.feature_names_out_)


        df_inv = self._apply_inverse_nonlinear(df_inv)

        return df_inv


def perform_pca_and_plot(df: pd.DataFrame, seed=42):


    pca = PCA(random_state=seed)
    transformed_data = pca.fit_transform(df)


    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)


    print("\n--- PCA Report ---")
    for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
        print(f"PC{i+1}: explained variance {var*100:.2f}%  |  cumulative {cum_var*100:.2f}%")


    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(10, 6), dpi=150)


    x_axis = range(1, len(cumulative_variance) + 1)
    plt.plot(
        x_axis,
        cumulative_variance,
        marker="o",
        markersize=6,
        linestyle="-",
        linewidth=2,
        color="#0F766E",
        label="Cumulative Variance",
    )


    plt.bar(x_axis, explained_variance_ratio, alpha=0.3, color="#14B8A6", label="Individual Variance")


    plt.axhline(y=0.90, color="#EF4444", linestyle="--", linewidth=1.5, alpha=0.8, label="90% Threshold")
    plt.axhline(y=0.95, color="#F59E0B", linestyle="--", linewidth=1.5, alpha=0.8, label="95% Threshold")


    plt.title("Cumulative Explained Variance by PCA Components", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Principal Components", fontsize=12)
    plt.ylabel("Explained Variance Ratio", fontsize=12)
    plt.xticks(x_axis)
    plt.ylim(0, 1.05)

    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.tight_layout()


    os.makedirs("./proxy_evaluation", exist_ok=True)
    plt.savefig("./proxy_evaluation/pca_variance_plot.svg", bbox_inches="tight")
    print("\nFigure saved as './proxy_evaluation/pca_variance_plot.svg'")
    plt.show()

    return pca, transformed_data


import random


def plot_pca_loadings_heatmap(pca: PCA, feature_names: list, top_k: int = 5):
    print(f"\nPlotting the loadings heatmap for the top {top_k} principal components...")


    loadings = pca.components_[:top_k, :]


    sns.set_theme(style="white", context="paper")
    plt.figure(figsize=(20, 8), dpi=150)


    ax = sns.heatmap(
        loadings,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        center=0,
        xticklabels=feature_names,
        yticklabels=[f"PC{i+1}" for i in range(top_k)],
        cbar_kws={"label": "Feature Weight (Loading)"},
    )

    plt.title(
        f"PCA Feature Loadings Heatmap (Top {top_k} Principal Components)", fontsize=16, fontweight="bold", pad=20
    )
    plt.xlabel("Original Proxy Features", fontsize=14, labelpad=10)
    plt.ylabel("Principal Components", fontsize=14, labelpad=10)


    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()
    os.makedirs("./proxy_evaluation", exist_ok=True)
    plt.savefig("./proxy_evaluation/pca_loadings_heatmap.svg", bbox_inches="tight")
    print(f"Feature loadings heatmap saved as 'pca_loadings_heatmap.svg'")
    plt.show()


from typing import Union, List


def compute_latent_boundaries(
    pca_data: np.ndarray, num_components: int, quantiles: List[float] = [0.33, 0.66]
) -> List[np.ndarray]:
    """


    """
    X_latent = pca_data[:, :num_components]
    boundaries = []

    for dim in range(num_components):
        bounds = np.quantile(X_latent[:, dim], quantiles)
        boundaries.append(bounds)

    return boundaries


def sample_by_boundaries(
    pca_data: np.ndarray,
    num_components: int,
    boundaries: List[np.ndarray],
    total_samples: Union[int, List[int]],
    seed: int = 42,
) -> Union[List[int], List[List[int]]]:
    random.seed(seed)

    is_single_round = isinstance(total_samples, int)
    sample_targets = [total_samples] if is_single_round else total_samples
    bin_indices = get_bin_indices(pca_data, num_components, boundaries)

    stratified_bins = defaultdict(list)
    for original_idx, b_tuple in enumerate(bin_indices):
        stratified_bins[tuple(b_tuple)].append(original_idx)

    num_bins_per_dim = len(boundaries[0]) + 1
    return sample_from_bins(bin_indices, sample_targets, seed, num_bins_per_dim)


def get_bin_indices(pca_data: np.ndarray, num_components: int, boundaries: List[np.ndarray]) -> np.ndarray:
    """

    """

    X_latent = pca_data[:, :num_components]


    bin_indices = []
    for dim in range(num_components):
        b_idx = np.digitize(X_latent[:, dim], boundaries[dim])
        bin_indices.append(b_idx)


    return np.array(bin_indices).T


def sample_from_bins(
    bin_indices: np.ndarray,
    total_samples: Union[int, List[int]],
    seed: int = 42,
    num_bins_per_dim: Optional[int] = None,
    incremental: bool = False,
    quiet: bool = False,
) -> Union[List[int], List[List[int]]]:
    """

    """
    random.seed(seed)

    is_single_round = isinstance(total_samples, int)
    sample_targets = [total_samples] if is_single_round else total_samples

    if incremental and not is_single_round:
        for i in range(1, len(sample_targets)):
            if sample_targets[i] <= sample_targets[i - 1]:
                raise ValueError(f"In incremental mode, total_samples must be strictly increasing; got: {total_samples}")
        delta_targets = [sample_targets[0]]
        for i in range(1, len(sample_targets)):
            delta_targets.append(sample_targets[i] - sample_targets[i - 1])
    else:
        delta_targets = sample_targets

    total_population = bin_indices.shape[0]
    num_components = bin_indices.shape[1]

    stratified_bins = defaultdict(list)
    for original_idx, b_tuple in enumerate(bin_indices):
        stratified_bins[tuple(b_tuple)].append(original_idx)

    if not quiet:
        if num_bins_per_dim is not None:
            total_theoretical_bins = num_bins_per_dim**num_components
            print(f"\nStratified sampling init: {num_components}D latent space partitioned into {total_theoretical_bins} joint quadrants.")
        else:
            print(f"\nStratified sampling init: {num_components}D latent space contains {len(stratified_bins)} non-empty joint quadrants.")

        mode_str = "incremental mode" if incremental else "non-overlapping partition mode"
        print(f"   Mode: {mode_str} | candidate pool size: {total_population} samples.")

    all_sampled_rounds = []
    cumulative_samples = []

    for round_idx, target_count in enumerate(delta_targets):
        current_remaining_total = sum(len(v) for v in stratified_bins.values())
        actual_target = min(target_count, current_remaining_total)

        sampled_indices = []

        if actual_target > 0:
            active_bins = [k for k, v in stratified_bins.items() if len(v) > 0]
            base_target_per_bin = actual_target // len(active_bins)
            remainder = actual_target % len(active_bins)

            unfulfilled_quota = 0

            for i, b_key in enumerate(active_bins):
                available = len(stratified_bins[b_key])
                target = base_target_per_bin + (1 if i < remainder else 0)

                if available >= target:
                    sampled = random.sample(stratified_bins[b_key], target)
                    sampled_indices.extend(sampled)
                    stratified_bins[b_key] = list(set(stratified_bins[b_key]) - set(sampled))
                else:
                    sampled_indices.extend(stratified_bins[b_key])
                    unfulfilled_quota += target - available
                    stratified_bins[b_key] = []

            if unfulfilled_quota > 0:
                remaining_population = []
                for b_key in active_bins:
                    remaining_population.extend(stratified_bins[b_key])

                if len(remaining_population) >= unfulfilled_quota:
                    extra_sampled = random.sample(remaining_population, unfulfilled_quota)
                else:
                    extra_sampled = remaining_population

                sampled_indices.extend(extra_sampled)

                extra_sampled_set = set(extra_sampled)
                for b_key in active_bins:
                    if not extra_sampled_set:
                        break
                    intersection = extra_sampled_set.intersection(stratified_bins[b_key])
                    if intersection:
                        stratified_bins[b_key] = list(set(stratified_bins[b_key]) - intersection)
                        extra_sampled_set -= intersection
        else:
            if not quiet:
                print(f"   Round {round_idx+1}: target increment {target_count} but the candidate pool is exhausted.")

        if incremental:
            cumulative_samples.extend(sampled_indices)
            all_sampled_rounds.append(list(cumulative_samples))
            if not quiet:
                print(
                    f"Round {round_idx+1} done: cumulative target {sample_targets[round_idx]} -> "
                    f"accumulated {len(cumulative_samples)} samples (this round adds {len(sampled_indices)})."
                )
        else:
            all_sampled_rounds.append(sampled_indices)
            if not quiet:
                print(
                    f"Round {round_idx+1} done: target {target_count} -> obtained {len(sampled_indices)} non-overlapping samples."
                )

    if is_single_round:
        return all_sampled_rounds[0]
    return all_sampled_rounds


def latent_space_stratified_sampling(
    pca_data: np.ndarray,
    num_components: int,
    total_samples: Union[int, List[int]],
    quantiles: List[float] = [0.33, 0.66],
    seed: int = 42,
) -> Union[List[int], List[List[int]]]:
    boundaries = compute_latent_boundaries(pca_data, num_components, quantiles)
    return sample_by_boundaries(pca_data, num_components, boundaries, total_samples, seed)
