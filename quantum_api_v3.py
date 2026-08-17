import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import networkx as nx
import torch.nn.functional as F
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict
import shelve


def compute_topological_center_diff(graph: nx.DiGraph) -> float:
    """

             ΔL = L_T - L_E
    """
    flags = nx.get_node_attributes(graph, "flag")

    try:

        generations = list(nx.topological_generations(graph))
    except nx.NetworkXUnfeasible:

        return 0.0

    total_generations = len(generations)
    if total_generations == 0:
        return 0.0


    node_to_gen = {}
    for gen_idx, nodes_in_gen in enumerate(generations):
        for node in nodes_in_gen:
            node_to_gen[node] = gen_idx


    e_gens = [node_to_gen[n] for n in graph.nodes if flags.get(n) == 1]
    t_gens = [node_to_gen[n] for n in graph.nodes if flags.get(n) == 0]


    mean_e_gen = np.mean(e_gens) if e_gens else 0.0
    mean_t_gen = np.mean(t_gens) if t_gens else 0.0


    gen_diff_normalized = (mean_t_gen - mean_e_gen) / total_generations

    return float(gen_diff_normalized)


def decisions_to_graph(decisions, n_qubits):
    graph = nx.DiGraph()


    symmetric_gates = {"XX", "YY", "ZZ", "RXX", "RYY", "RZZ", "CZ", "SWAP"}


    graph.add_node("start", label="start", gate="START", wires=[], flag=None)


    for i, act in enumerate(decisions):
        if "x_index" in act:
            flag = 1
        elif "param_idx" in act:
            flag = 0
        else:
            flag = None

        gate_name = act["gate"].upper()

        graph.add_node(
            i,
            label=(gate_name, flag),
            gate=gate_name,
            wires=act["wires"],
            flag=flag,
            depth=act.get("depth", None),
            x_index=act.get("x_index", None),
            param_idx=act.get("param_idx", None),
        )

    graph.add_node("end", label="end", gate="END", wires=[], flag=None)


    last = ["start" for _ in range(n_qubits)]


    for i, act in enumerate(decisions):
        wires = act["wires"]
        gate_name = act["gate"].upper()

        if len(wires) == 1:

            q = wires[0]
            graph.add_edge(last[q], i, role="single")
            last[q] = i  # type: ignore

        elif len(wires) == 2:

            c, t = wires
            if gate_name in symmetric_gates:

                graph.add_edge(last[c], i, role="symmetric")
                graph.add_edge(last[t], i, role="symmetric")
            else:

                graph.add_edge(last[c], i, role="control")
                graph.add_edge(last[t], i, role="target")

            last[c] = i  # type: ignore
            last[t] = i  # type: ignore

        else:
            raise NotImplementedError()


            # for idx, w in enumerate(wires):
            #     role = "target" if idx == len(wires) - 1 else f"control_{idx}"
            #     graph.add_edge(last[w], i, role=role)
            #     last[w] = i


    for q in range(n_qubits):
        if not graph.has_edge(last[q], "end"):
            graph.add_edge(last[q], "end", role="terminal")


    return graph, None


import math
from collections import Counter


def compute_spatiotemporal_mutual_information(graph: nx.DiGraph, topo_order) -> float:
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The circuit graph must be a DAG")

    node_layers = {}
    if not topo_order:
        topo_order = nx.topological_sort(graph)
    for node in topo_order:
        predecessors = list(graph.predecessors(node))
        node_layers[node] = 0 if not predecessors else max(node_layers[p] for p in predecessors) + 1

    ql_pairs = []
    for node, attr in graph.nodes(data=True):
        wires = attr.get("wires")
        if wires:
            l = node_layers[node]
            ql_pairs.extend((q, l) for q in wires)

    if not ql_pairs:
        return 0.0

    total_ops = len(ql_pairs)
    q_counts = Counter(q for q, l in ql_pairs)
    l_counts = Counter(l for q, l in ql_pairs)
    ql_counts = Counter(ql_pairs)

    def _entropy(counts_dict):
        return sum(-(c / total_ops) * math.log2(c / total_ops) for c in counts_dict.values())

    h_q, h_l, h_ql = _entropy(q_counts), _entropy(l_counts), _entropy(ql_counts)
    mutual_info = max(0.0, h_q + h_l - h_ql)
    max_mi = min(h_q, h_l)
    return float(mutual_info / max_mi) if max_mi > 0 else 0.0


def compute_qubit_distribution_entropies(graph: nx.DiGraph, n_qubits: int, coupling_map=None):

    encoding_wires = []
    param_wires = []
    global_wires = []
    entanglement_edges = []

    for n, attr in graph.nodes(data=True):
        wires = attr.get("wires")
        if not wires:
            continue
        global_wires.extend(wires)
        if len(wires) > 1:
            entanglement_edges.append(tuple(sorted(wires[:2])))
        flag = attr.get("flag")
        if flag == 1:
            encoding_wires.extend(wires)
        elif flag == 0:
            param_wires.extend(wires)


    def _js_divergence_from_counts(p_counts, q_counts, keys):
        p_total = sum(p_counts.values())
        q_total = sum(q_counts.values())
        if p_total == 0 and q_total == 0:
            return 0.0
        if p_total == 0 or q_total == 0:
            return 1.0
        js = 0.0
        for k in keys:
            p = p_counts.get(k, 0) / p_total
            q = q_counts.get(k, 0) / q_total
            m = 0.5 * (p + q)
            if p > 0:
                js += 0.5 * p * math.log2(p / m)
            if q > 0:
                js += 0.5 * q * math.log2(q / m)

        return float(js)


    # vs uniform distribution
    def _calculate_node_js_entropy(wires_list):

        keys = list(range(n_qubits))
        p_counts = Counter(wires_list)

        q_counts = {k: 1 for k in keys}

        return _js_divergence_from_counts(p_counts, q_counts, keys)

    # edge distribution JS entropy
    def _calculate_edge_js_entropy(edges_list, ideal_edges):

        if len(ideal_edges) == 0:
            return 0.0
        p_counts = Counter(edges_list)

        q_counts = {e: 1 for e in ideal_edges}
        return _js_divergence_from_counts(p_counts, q_counts, ideal_edges)

    # Universal edge space
    universal_edges = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]

    # Hardware edge space
    if coupling_map is not None:

        hardware_edges = list(set(tuple(sorted(edge[:2])) for edge in coupling_map))

    else:

        hardware_edges = universal_edges


    encoding_entropy = _calculate_node_js_entropy(encoding_wires)
    param_entropy = _calculate_node_js_entropy(param_wires)
    global_entropy = _calculate_node_js_entropy(global_wires)
    ent_entropy_universal = _calculate_edge_js_entropy(entanglement_edges, universal_edges)
    ent_entropy_hardware = _calculate_edge_js_entropy(entanglement_edges, hardware_edges)

    # encoding vs parameter JS
    ep_js_divergence = _js_divergence_from_counts(Counter(encoding_wires), Counter(param_wires), list(range(n_qubits)))

    return (
        encoding_entropy,
        param_entropy,
        global_entropy,
        ent_entropy_universal,
        ent_entropy_hardware,
        ep_js_divergence,
    )


def compute_causal_path_alternation_intrinsic(graph: nx.DiGraph, topo_order, consider_redundancy: bool = True) -> float:
    """

    """

    flags = nx.get_node_attributes(graph, "flag")


    redundancy_marks = nx.get_node_attributes(graph, "is_redundant")

    try:
        if topo_order is None:
            topo_order = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        raise ValueError("The graph is not a DAG; topological sort is impossible!")


    DP = {u: defaultdict(lambda: [0, 0]) for u in graph.nodes()}


    start_node = "start"
    if start_node not in graph:
        return 0.0

    f_start = flags.get(start_node)

    is_start_redundant = redundancy_marks.get(start_node, False)


    if f_start in [0, 1] and not (consider_redundancy and is_start_redundant):
        DP[start_node][(f_start, 1)] = [1, 0]
    else:
        DP[start_node][(None, 0)] = [1, 0]


    for u in topo_order:
        for v in graph.successors(u):
            f_v = flags.get(v)
            is_v_redundant = redundancy_marks.get(v, False)

            for (last_flag, seq_len), (p_count, f_count) in DP[u].items():


                is_v_redundant_effective = is_v_redundant if consider_redundancy else False

                if is_v_redundant_effective or f_v not in [0, 1]:


                    new_last = last_flag
                    new_len = seq_len
                    new_f_count = f_count

                else:


                    new_last = f_v
                    new_len = seq_len + 1
                    is_flip = 1 if (last_flag is not None and last_flag != f_v) else 0
                    new_f_count = f_count + p_count * is_flip


                state = DP[v][(new_last, new_len)]
                state[0] += p_count
                state[1] += new_f_count


    end_node = "end"
    if end_node not in DP:
        return 0.0

    total_rate = 0.0
    total_paths = 0

    for (last_flag, seq_len), (p_count, f_count) in DP[end_node].items():
        total_paths += p_count
        if seq_len > 1:

            total_rate += f_count / (seq_len - 1)

    if total_paths == 0:
        return 0.0

    return float(total_rate / total_paths)


def compute_basic_and_redundancy_proxies(graph: nx.DiGraph, n_qubits: int, topo_order):
    n_encoding = 0
    n_trainable = 0
    redundancy_count = 0
    total_gate_count = 0
    n_two_qubit_gates = 0


    last_single_axis = {q: None for q in range(n_qubits)}


    single_qubit_streak = {q: 0 for q in range(n_qubits)}


    last_two_qubit_gate = {q: None for q in range(n_qubits)}


    symmetric_gates = {"XX", "YY", "ZZ", "RXX", "RYY", "RZZ", "CZ", "SWAP", "MS", "ISWAP"}


    self_inverse_gates = {"CNOT", "CX", "CZ", "SWAP", "CY", "XX", "YY", "ZZ"}


    parameterized_two_qubit_gates = {"RXX", "RYY", "RZZ", "CRX", "CRY", "CRZ", "CPHASE"}

    for node in topo_order:
        attr = graph.nodes[node]
        gate_type = attr.get("gate", "").upper().strip()
        wires = attr.get("wires", [])

        if not gate_type or gate_type in ["START", "END"]:
            continue

        total_gate_count += 1

        flag = attr.get("flag")
        if flag == 1:
            n_encoding += 1
        elif flag == 0:
            n_trainable += 1
        if len(wires) == 2:
            n_two_qubit_gates += 1

        if gate_type in symmetric_gates:
            norm_wires = tuple(sorted(wires))
        else:
            norm_wires = tuple(wires)
        current_sig = (gate_type, norm_wires)

        is_redundant = False

        if wires:
            if len(wires) == 1:
                q = wires[0]


                if last_single_axis[q] == gate_type:
                    is_redundant = True


                elif single_qubit_streak[q] >= 3:
                    is_redundant = True

                else:

                    last_single_axis[q] = gate_type
                    single_qubit_streak[q] += 1


                last_two_qubit_gate[q] = None

            elif len(wires) == 2:

                is_match = True
                for q in wires:
                    if last_two_qubit_gate[q] != current_sig:
                        is_match = False
                        break

                if is_match:

                    if gate_type in self_inverse_gates:

                        is_redundant = True
                        for q in wires:
                            last_two_qubit_gate[q] = None

                    elif gate_type in parameterized_two_qubit_gates:


                        is_redundant = True
                        for q in wires:

                            last_two_qubit_gate[q] = current_sig
                else:

                    for q in wires:
                        last_two_qubit_gate[q] = current_sig

                        last_single_axis[q] = None
                        single_qubit_streak[q] = 0


        graph.nodes[node]["is_redundant"] = is_redundant

        if is_redundant:
            redundancy_count += 1

    total_data_param = n_encoding + n_trainable
    encoding_param_ratio = (n_encoding / total_data_param) if total_data_param > 0 else 0.5
    redundancy_ratio = (redundancy_count / total_gate_count) if total_gate_count > 0 else 0.0

    return n_encoding, n_trainable, encoding_param_ratio, redundancy_ratio, n_two_qubit_gates


def compute_centrality_proxies(graph: nx.DiGraph):
    try:
        centrality = nx.betweenness_centrality(graph)
    except Exception:
        return {"encode_cent": 0.0, "trainable_cent": 0.0, "data_dominance": 0.0}

    encode_cents = [centrality[n] for n, attr in graph.nodes(data=True) if attr.get("flag") == 1]
    trainable_cents = [centrality[n] for n, attr in graph.nodes(data=True) if attr.get("flag") == 0]

    mean_encode_cent = sum(encode_cents) / len(encode_cents) if encode_cents else 0.0
    mean_trainable_cent = sum(trainable_cents) / len(trainable_cents) if trainable_cents else 0.0

    data_dominance = mean_encode_cent / (mean_encode_cent + mean_trainable_cent + 1e-5)

    return mean_encode_cent, mean_trainable_cent, data_dominance


def compute_trainability_and_snip(
    decisions: list[dict],
    n_qubits: int,
    num_classes: int = 4,
    sel_samples_per_class: int = 16,
    num_random_initial: int = 32,
    seed: int = 0,
    noise_param: dict = None,  # type: ignore
    in_features: int = 2,
    consider_linear: bool = True,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)


    num_embed = 1 + max([act["x_index"] for act in decisions if act.get("x_index", None) is not None], default=-1)
    num_params = 1 + max([act["param_idx"] for act in decisions if act.get("param_idx", None) is not None], default=-1)


    if num_params == 0:
        return 0.0, 0.0


    is_state_prep_task = num_embed == 0


    ansatz = DecisionsAnsatz()

    if is_state_prep_task:
        measurement = StateMeasurement()
    else:
        measurement = PauliZMeasurement(n_qubits)

    circuit = make_noisy_qnode(ansatz, measurement, n_qubits, noise_param=noise_param)  # type: ignore


    batch_size = num_classes * sel_samples_per_class

    if is_state_prep_task:
        dim = 2**n_qubits
        target_vec = torch.randn(dim, dtype=torch.complex128) + 1j * torch.randn(dim, dtype=torch.complex128)
        target_vec = target_vec / torch.norm(target_vec)
        target_rho = torch.outer(target_vec, target_vec.conj())
    else:

        x_raw_np = np.random.uniform(-1, 1, (batch_size, in_features))
        x_raw_tensor = torch.tensor(x_raw_np, dtype=torch.float64)

    trainabilities, snips = [], []


    for i in range(num_random_initial):

        temp_param_np = np.random.uniform(0, 1, num_params) * np.pi * 2
        temp_param = torch.tensor(temp_param_np, dtype=torch.float64, requires_grad=True)

        if is_state_prep_task:

            output_state = circuit(decisions, params=temp_param, x=None)
            if output_state.dim() == 1:
                fidelity = torch.abs(torch.dot(output_state, target_vec.conj())) ** 2
            else:
                fidelity = torch.real(torch.trace(output_state @ target_rho))
            loss = -fidelity


            all_params = [temp_param]

        else:


            linear_layer = torch.nn.Linear(in_features, num_embed, dtype=torch.float64)


            x_embed_tensor = linear_layer(x_raw_tensor)


            ypreds = circuit(decisions, params=temp_param, x=x_embed_tensor)

            if isinstance(ypreds, (list, tuple)):
                ypreds = torch.stack(ypreds, dim=-1)


            loss = torch.mean(torch.sum(torch.abs(ypreds), dim=-1))

            if consider_linear:

                all_params = [temp_param, linear_layer.weight, linear_layer.bias]
            else:

                all_params = [temp_param]


        loss.backward()


        total_grad_norm_sq = 0.0
        snip_elements = []

        for p in all_params:
            if p.grad is not None:
                grad_data = p.grad.detach().cpu().numpy()
                param_data = p.detach().cpu().numpy()


                total_grad_norm_sq += np.sum(grad_data**2)


                snip_elements.extend(np.abs(grad_data * param_data).flatten())


        trainabilities.append(np.sqrt(total_grad_norm_sq))

        if len(snip_elements) > 0:
            snips.append(np.mean(snip_elements))
        else:
            snips.append(0.0)

    return float(np.mean(trainabilities)), float(np.mean(snips))


def calculate_circuit_depth_and_width(Graph, topo_order):

    try:
        if topo_order is None:
            topo_order = list(nx.topological_sort(Graph))
    except nx.NetworkXUnfeasible:
        raise ValueError("The graph is not a DAG; topological sort is impossible!")


    N = {node: 0 for node in Graph.nodes()}
    S = {node: 0 for node in Graph.nodes()}


    if "end" in N:
        N["end"] = 1
        S["end"] = 0


    for u in reversed(topo_order):
        for v in Graph.successors(u):
            N[u] += N[v]
            S[u] += S[v] + N[v]

    width = N.get("start", 0)

    if width == 0:
        return 0.0, 0

    mean_length = S["start"] / width
    return mean_length - 1, width


def calculate_haar_dist(n_qubits, points):
    N = n_qubits
    space = 1 / points
    x = [space * (i + 1) for i in range(-1, points)]
    haar_points = []
    for i in range(1, len(x)):
        temp1 = -1 * np.power((1 - x[i]), np.power(2, N) - 1)
        temp0 = -1 * np.power((1 - x[i - 1]), np.power(2, N) - 1)
        haar_points.append(temp1 - temp0)
    haar_points = np.array(haar_points)
    return haar_points


def calculate_expressibility_single(points, num_random_initial, decisions, qnode, haar_points, device="cuda"):

    par = torch.rand((num_random_initial, len(decisions)), device=device, dtype=torch.float64) * (2 * np.pi)
    with torch.no_grad():
        output_states = qnode(decisions, None, par)

    half = num_random_initial // 2
    output_states1 = output_states[:half]
    output_states2 = output_states[half : 2 * half]

    # fidelity = |<psi1|psi2>|^2
    fidelity = torch.abs((output_states1 * output_states2.conj()).sum(-1)) ** 2


    bin_index = torch.floor(fidelity * points).long()
    bin_index = torch.clamp(bin_index, 0, points - 1)


    counts = F.one_hot(bin_index, num_classes=points).sum(dim=0).float()
    prob = counts / counts.sum()


    haar_probs = torch.tensor(haar_points, device=device, dtype=torch.float64)
    m = 0.5 * (prob + haar_probs)
    kl_p_m = torch.sum(prob * torch.log((prob + 1e-12) / (m + 1e-12)))
    kl_h_m = torch.sum(haar_probs * torch.log((haar_probs + 1e-12) / (m + 1e-12)))
    js_div = 0.5 * kl_p_m + 0.5 * kl_h_m

    if torch.isnan(js_div):
        print(js_div)
        print("NaN detected in expressibility calculation.")
    return -js_div.item()


def calculate_expressibility_embed_single(qnode, num_random_initial, decisions, points, haar_points, device="cuda"):
    swapped_decisions = []
    num_embed = 0
    num_params = 0
    for act in decisions:
        new_act = act.copy()
        if act.get("x_index", None) is not None:
            num_embed = max(num_embed, act["x_index"] + 1)
            new_act["param_idx"] = new_act.pop("x_index")
        elif act.get("param_idx", None) is not None:
            num_params = max(num_params, act["param_idx"] + 1)
            new_act["x_index"] = new_act.pop("param_idx")
        swapped_decisions.append(new_act)

    if num_embed == 0:
        return 0.0


    par = torch.rand((num_random_initial, max(num_params, 1)), device=device, dtype=torch.float64) * (2 * np.pi)
    embed = torch.rand(num_embed, device=device, dtype=torch.float64) * (2 * np.pi)

    with torch.no_grad():
        output_states = qnode(swapped_decisions, embed, par)

    half = num_random_initial // 2
    output_states1 = output_states[:half]
    output_states2 = output_states[half : 2 * half]

    fidelity = torch.abs((output_states1 * output_states2.conj()).sum(-1)) ** 2

    bin_index = torch.clamp(torch.floor(fidelity * points).long(), 0, points - 1)
    counts = F.one_hot(bin_index, num_classes=points).sum(dim=0).float()
    prob = counts / counts.sum()


    haar_probs = torch.tensor(haar_points, device=device, dtype=torch.float64)
    m = 0.5 * (prob + haar_probs)
    kl_p_m = torch.sum(prob * torch.log((prob + 1e-12) / (m + 1e-12)))
    kl_h_m = torch.sum(haar_probs * torch.log((haar_probs + 1e-12) / (m + 1e-12)))
    js_div = 0.5 * kl_p_m + 0.5 * kl_h_m

    if torch.isnan(js_div):
        print("NaN detected in expressibility calculation.")
    return -js_div.item()


def compute_noise_aware_proxies(Graph: nx.DiGraph, n_qubits: int):
    """


        (error_lightcone_vol, lifespan_imbalance, data_reinjection_index)
    """

    gate_nodes = [n for n in Graph.nodes() if isinstance(n, int)]

    if not gate_nodes:
        return 0.0, 0.0, 0.0


    total_infected_qubits = 0
    for node in gate_nodes:

        descendants = nx.descendants(Graph, node)
        infected_wires = set()
        for d in descendants:
            if isinstance(d, int):
                wires = Graph.nodes[d].get("wires", [])
                infected_wires.update(wires)
        total_infected_qubits += len(infected_wires)


    error_lightcone_vol = float(total_infected_qubits / len(gate_nodes))


    gate_counts_per_qubit = np.zeros(n_qubits)
    for node in gate_nodes:
        wires = Graph.nodes[node].get("wires", [])
        for w in wires:
            gate_counts_per_qubit[w] += 1

    lifespan_imbalance = float(np.std(gate_counts_per_qubit) / np.mean(gate_counts_per_qubit))


    encoding_nodes = [n for n in gate_nodes if Graph.nodes[n].get("flag") == 1]

    if not encoding_nodes:
        data_reinjection_index = 0.0
    else:

        try:
            dag_depth = nx.dag_longest_path_length(Graph)
        except Exception:
            dag_depth = len(gate_nodes)

        closeness_scores = []
        for enc_node in encoding_nodes:
            try:

                dist_to_end = nx.shortest_path_length(Graph, source=enc_node, target="end")

                closeness = (dag_depth - dist_to_end) / max(dag_depth, 1)
                closeness_scores.append(max(0.0, closeness))
            except nx.NetworkXNoPath:
                continue

        if closeness_scores:
            data_reinjection_index = float(np.mean(closeness_scores))
        else:
            data_reinjection_index = 0.0

    return error_lightcone_vol, lifespan_imbalance, data_reinjection_index


class DecisionsAnsatzSimple:
    """

    decisions: Iterable[dict]
          - gate: "RX","RY","RZ","CRX","CRY","CRZ","H","S","T","CNOT","CZ","I","ID",
                  "SX","ECR","ISWAP","SWAP","MS","GPI","GPI2","RXX","RZZ","ISINGXX","ISINGZZ"
    """

    def __init__(self):

        self.compiled_plans = {}

    def _hash_decisions(self, decisions):
        """"""
        key = []
        for act in decisions:
            items = []
            for k, v in sorted(act.items()):
                items.append((k, tuple(v) if isinstance(v, list) else v))
            key.append(tuple(items))
        return hash(tuple(key))

    def _compile(self, decisions):

        plan = []
        param_counter = 0

        for act in decisions:
            gate = act["gate"].upper()


            wires = act["wires"]
            if not isinstance(wires, (list, tuple)):
                wires = [wires]
            wires = [int(w) for w in wires]


            if gate in ("RX", "RY", "RZ", "GPI", "GPI2"):
                op_type = 1
                aux = param_counter
                param_counter += 1

                if gate in ("GPI", "GPI2") and not hasattr(qml, gate):
                    raise ValueError(f"Pennylane environment missing native support for IonQ {gate}")

                plan.append((op_type, getattr(qml, gate), wires[0], aux))


            elif gate in ("CRX", "CRY", "CRZ"):
                op_type = 1
                aux = param_counter
                param_counter += 1

                control, target = wires[0], wires[1]
                plan.append((op_type, getattr(qml, gate), [control, target], aux))


            elif gate in ("H", "S", "T", "SX", "I", "ID"):
                op_type = 0
                q_func = getattr(qml, "Identity") if gate in ("I", "ID") else getattr(qml, gate)
                plan.append((op_type, q_func, wires[0], None))


            elif gate in ("CNOT", "CZ", "SWAP", "ISWAP", "ECR"):
                op_type = 0
                q_func = getattr(qml, gate)
                control, target = wires[0], wires[1]
                plan.append((op_type, q_func, [control, target], None))


            elif gate == "MS":
                control, target = wires[0], wires[1]
                if hasattr(qml, "IsingXX"):
                    plan.append((3, getattr(qml, "IsingXX"), [control, target], np.pi / 2))
                else:
                    plan.append((0, getattr(qml, "H"), target, None))
                    plan.append((0, getattr(qml, "CNOT"), [control, target], None))
                    plan.append((0, getattr(qml, "H"), target, None))


            elif gate in ("RXX", "RZZ", "ISINGXX", "ISINGZZ"):
                op_type = 1
                aux = param_counter
                param_counter += 1

                control, target = wires[0], wires[1]

                if gate in ("RXX", "ISINGXX"):
                    if hasattr(qml, "IsingXX"):
                        plan.append((op_type, getattr(qml, "IsingXX"), [control, target], aux))
                    else:
                        plan.append((0, getattr(qml, "CNOT"), [control, target], None))
                        plan.append((op_type, getattr(qml, "RX"), target, aux))
                        plan.append((0, getattr(qml, "CNOT"), [control, target], None))

                elif gate in ("RZZ", "ISINGZZ"):
                    if hasattr(qml, "IsingZZ"):
                        plan.append((op_type, getattr(qml, "IsingZZ"), [control, target], aux))
                    else:
                        plan.append((0, getattr(qml, "CNOT"), [control, target], None))
                        plan.append((op_type, getattr(qml, "RZ"), target, aux))
                        plan.append((0, getattr(qml, "CNOT"), [control, target], None))
            else:
                raise ValueError(f"Unsupported gate in decisions: {gate}")
        return plan

    def __call__(self, decisions, params=None, x=None, noise_injector=None, coupling_map=None, dynamic_routing=False):
        d_id = self._hash_decisions(decisions)


        if d_id not in self.compiled_plans:
            self.compiled_plans[d_id] = self._compile(decisions)


        for op_type, func, wires, aux in self.compiled_plans[d_id]:
            if op_type == 0:
                func(wires=wires)
            elif op_type == 1:

                func(x[..., aux], wires=wires)
            elif op_type == 2:

                if params is not None:
                    func(params[aux], wires=wires)
            elif op_type == 3:

                func(aux, wires=wires)


class PauliZMeasurement:

    def __init__(self, n_qubits: int):
        self.n_qubits = int(n_qubits)
        self.output_dim = self.n_qubits

    def __call__(self):
        obs = []
        for i in range(self.n_qubits):
            obs.append(qml.expval(qml.PauliZ(i)))
        return obs


class StateMeasurement:

    def __call__(self):
        return qml.state()


class HamiltonianMeasurement:

    def __init__(self, hamiltonian):
        self.hamiltonian = hamiltonian

        self.output_dim = 1

    def __call__(self):

        return [qml.expval(self.hamiltonian)]


class ClockScheduledNoiseInjector:
    def __init__(
        self,
        mode="ALAP",  # supported "ALAP" or "ASAP"
        T1=100e-6,
        T2=50e-6,
        t_1q=20e-9,
        t_2q=200e-9,
        p_depol_1q=0.001,
        p_depol_2q=0.01,
        p_measure=0.02,
    ):
        self.mode = mode.upper()
        self.T1, self.T2 = T1, T2
        self.t_1q, self.t_2q = t_1q, t_2q
        self.p_depol_1q, self.p_depol_2q = p_depol_1q, p_depol_2q
        self.p_measure = p_measure

        self._cached_2q_kraus = None
        self.p_amp_1q = 1.0 - np.exp(-self.t_1q / self.T1)
        self.p_phase_1q = 1.0 - np.exp(-self.t_1q / self.T2)
        self.p_amp_2q = 1.0 - np.exp(-self.t_2q / self.T1)
        self.p_phase_2q = 1.0 - np.exp(-self.t_2q / self.T2)

    def _calc_idle_damping(self, idle_time: float):
        if idle_time <= 0:
            return 0.0, 0.0
        return 1.0 - np.exp(-idle_time / self.T1), 1.0 - np.exp(-idle_time / self.T2)

    def _get_2q_global_depolarizing_kraus(self):
        if self._cached_2q_kraus is not None:
            return self._cached_2q_kraus
        I = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        paulis = [I, X, Y, Z]

        p = self.p_depol_2q
        kraus_ops = []
        for i, p1 in enumerate(paulis):
            for j, p2 in enumerate(paulis):
                op = np.kron(p1, p2)
                kraus_ops.append(np.sqrt(1 - p) * op if i == 0 and j == 0 else np.sqrt(p / 15) * op)
        self._cached_2q_kraus = [torch.tensor(k, dtype=torch.complex128) for k in kraus_ops]
        return self._cached_2q_kraus

    def _get_raw_gate_noise(self, gate, wires):
        instructions = []
        if len(wires) == 1:
            w = wires[0]
            if self.p_depol_1q > 0:
                instructions.append((3, qml.DepolarizingChannel, [w], self.p_depol_1q))
            if self.p_amp_1q > 0:
                instructions.append((3, qml.AmplitudeDamping, [w], self.p_amp_1q))
            if self.p_phase_1q > 0:
                instructions.append((3, qml.PhaseDamping, [w], self.p_phase_1q))
        else:
            if self.p_depol_2q > 0:
                instructions.append((3, qml.QubitChannel, wires[:2], self._get_2q_global_depolarizing_kraus()))
            for w in wires[:2]:
                if self.p_amp_2q > 0:
                    instructions.append((3, qml.AmplitudeDamping, [w], self.p_amp_2q))
                if self.p_phase_2q > 0:
                    instructions.append((3, qml.PhaseDamping, [w], self.p_phase_2q))
        return instructions

    def compute_noise_map(self, timing_trace, all_wires):
        """"""
        forward_starts = []

        if self.mode == "ALAP":

            rev_clocks = defaultdict(float)
            alap_starts = []
            for op in reversed(timing_trace):
                start_rev = max([rev_clocks[w] for w in op["wires"]], default=0.0)
                end_rev = start_rev + op["duration"]
                alap_starts.append(start_rev)
                for w in op["wires"]:
                    rev_clocks[w] = end_rev
            alap_starts.reverse()
            t_total = max(rev_clocks.values(), default=0.0)


            for i, op in enumerate(timing_trace):
                fwd_start = t_total - (alap_starts[i] + op["duration"])
                forward_starts.append(fwd_start)
        else:

            fwd_clocks_asap = defaultdict(float)
            for op in timing_trace:
                start_fwd = max([fwd_clocks_asap[w] for w in op["wires"]], default=0.0)
                forward_starts.append(start_fwd)
                for w in op["wires"]:
                    fwd_clocks_asap[w] = start_fwd + op["duration"]
            t_total = max(fwd_clocks_asap.values(), default=0.0)


        fwd_clocks = defaultdict(float)
        noise_map = {}

        for i, op in enumerate(timing_trace):
            wires = op["wires"]
            start_time = forward_starts[i]
            insts = []

            for w in wires:
                idle = start_time - fwd_clocks[w]
                if idle > 1e-12:


                    if fwd_clocks[w] > 0:
                        p_a, p_p = self._calc_idle_damping(idle)
                        if p_a > 0:
                            insts.append((3, qml.AmplitudeDamping, [w], p_a))
                        if p_p > 0:
                            insts.append((3, qml.PhaseDamping, [w], p_p))
                fwd_clocks[w] = start_time + op["duration"]

            if op["type"] == "gate":
                insts.extend(self._get_raw_gate_noise(op["gate"], wires))
            noise_map[i] = insts


        final_sync = []
        for w in all_wires:
            idle = t_total - fwd_clocks[w]
            if idle > 1e-12:

                if fwd_clocks[w] > 0:
                    p_a, p_p = self._calc_idle_damping(idle)
                    if p_a > 0:
                        final_sync.append((3, qml.AmplitudeDamping, [w], p_a))
                    if p_p > 0:
                        final_sync.append((3, qml.PhaseDamping, [w], p_p))
            fwd_clocks[w] = t_total

        return noise_map, final_sync

    def get_readout_noise_instructions(self, w: int):
        return [(3, qml.BitFlip, [w], self.p_measure)] if self.p_measure > 0 else []


import matplotlib.patches as mpatches


class DecisionsAnsatz:
    def __init__(self):
        self.compiled_plans = {}

    def _hash_decisions(self, decisions):
        key = []
        for act in decisions:
            items = []
            for k, v in sorted(act.items()):
                items.append((k, tuple(v) if isinstance(v, list) else v))
            key.append(tuple(items))
        return hash(tuple(key))

    def _compile(self, decisions, noise_injector):
        t_1q = noise_injector.t_1q if noise_injector is not None else 20e-9
        t_2q = noise_injector.t_2q if noise_injector is not None else 200e-9

        plan = []
        timing_trace = []
        all_logical_wires = set()

        for act in decisions:
            gate = act["gate"].upper()
            wires = tuple(act["wires"])
            all_logical_wires.update(wires)

            x_idx, p_idx = act.get("x_index", None), act.get("param_idx", None)
            op_type = 1 if x_idx is not None else (2 if p_idx is not None else 0)
            aux = x_idx if x_idx is not None else p_idx


            if gate in ("RX", "RY", "RZ"):
                plan.append((op_type, getattr(qml, gate), wires[0], aux))
            elif gate in ("CNOT", "CZ"):
                plan.append((0, getattr(qml, gate), list(wires), None))
            elif gate == "RXX":
                plan.append((op_type, qml.IsingXX, list(wires), aux))
            elif gate == "RZZ":
                plan.append((op_type, qml.IsingZZ, list(wires), aux))
            else:
                plan.append((0, getattr(qml, gate), wires[0] if len(wires) == 1 else list(wires), aux))

            timing_trace.append(
                {"type": "gate", "gate": gate, "wires": list(wires), "duration": t_1q if len(wires) == 1 else t_2q}
            )
            plan.append(("MARKER", len(timing_trace) - 1, None, None))


        if noise_injector is not None:
            noise_map, final_sync = noise_injector.compute_noise_map(timing_trace, all_logical_wires)
            final_plan = []
            for item in plan:
                if item[0] == "MARKER":
                    final_plan.extend(noise_map[item[1]])
                else:
                    final_plan.append(item)
            final_plan.extend(final_sync)
            return final_plan

        return [item for item in plan if not isinstance(item[0], str)]

    def __call__(self, decisions, params=None, x=None, noise_injector=None):

        d_id = self._hash_decisions(decisions)

        if d_id not in self.compiled_plans:
            self.compiled_plans[d_id] = self._compile(decisions, noise_injector)

        for op_type, func, wires, aux in self.compiled_plans[d_id]:
            if op_type == 0:
                func(wires=wires)
            elif op_type == 1:
                func(x[..., aux], wires=wires)
            elif op_type == 2:
                func(params[aux], wires=wires)
            elif op_type == 3:
                func(aux, wires=wires)


def make_noisy_qnode(ansatz, measurement, n_qubits, dev_name="default.qubit", noise_param=None):
    noise_type = noise_param.get("type", "none") if noise_param else "none"

    noise_injector = None
    if noise_type == "heuristic" or noise_type == "advanced":
        noise_injector = ClockScheduledNoiseInjector(
            mode=noise_param.get("mode", "ALAP"),
            T1=noise_param.get("T1", 100e-6),
            T2=noise_param.get("T2", 50e-6),
            t_1q=noise_param.get("t_1q", 20e-9),
            t_2q=noise_param.get("t_2q", 200e-9),
            p_depol_1q=noise_param.get("p_single", 0.001),
            p_depol_2q=noise_param.get("p_double", 0.01),
            p_measure=noise_param.get("p_measure", 0.02),
        )
    is_mixed_required = dev_name == "default.mixed" or noise_injector is not None

    if is_mixed_required:
        dev = qml.device("default.mixed", wires=n_qubits)
        diff_method = "backprop"

        precompiled_readouts = []
        if noise_injector is not None:
            for w in range(n_qubits):
                precompiled_readouts.extend(noise_injector.get_readout_noise_instructions(w))

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(decisions, params=None, x=None):

            ansatz(decisions, params=params, x=x, noise_injector=noise_injector)

            for op_type, func, wires, aux in precompiled_readouts:
                if op_type == 3:
                    func(aux, wires=wires)

            return measurement()

        return circuit

    else:
        dev = qml.device(dev_name, wires=n_qubits)
        diff_method = "best"

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(decisions, params=None, x=None):

            ansatz(decisions, params=params, x=x, noise_injector=None)
            return measurement()

        return circuit


import torch
import torch.nn as nn
import pennylane as qml


class VQCFeatureExtractor(nn.Module):
    def __init__(
        self, decisions, n_qubits, measurement, dev_name="default.qubit", trainable=True, seed=42, noise_param=None
    ):
        super().__init__()
        self.n_qubits = int(n_qubits)

        self.decisions = list(decisions)
        self.measurement = measurement

        self.dev_name = dev_name
        self.noise_param = noise_param

        self.qnode = make_noisy_qnode(
            DecisionsAnsatz(),
            self.measurement,
            self.n_qubits,
            dev_name=self.dev_name,
            noise_param=self.noise_param,
        )

        self.num_params = self.get_max_param_idx(self.decisions)
        self.num_embed_x = self.get_max_x_idx(self.decisions)

        # small perturbation initialization
        init_params = torch.empty(self.num_params, dtype=torch.float64).normal_(
            mean=0.0, std=1e-2, generator=torch.Generator().manual_seed(seed)
        )

        if trainable:
            self.params = nn.Parameter(init_params)
        else:
            self.register_buffer("params", init_params)

    def get_max_param_idx(self, decisions):
        return 1 + max([act["param_idx"] for act in decisions if act.get("param_idx", None) is not None], default=-1)

    def get_max_x_idx(self, decisions):
        return 1 + max([act["x_index"] for act in decisions if act.get("x_index", None) is not None], default=-1)

    def sync_pruning_mask(self) -> int:
        """

        Returns:
        """

        if not hasattr(self, "params_mask"):
            print("PyTorch params_mask not detected; pruning has not been performed, so no sync is possible.")
            return 0

        with torch.no_grad():

            mask = getattr(self, "params_mask").bool()

            new_decisions = []
            pruned_count = 0

            for act in self.decisions:
                if "param_idx" in act and act["param_idx"] is not None:
                    pidx = act["param_idx"]

                    if not mask[pidx].item():
                        pruned_count += 1
                        continue


                new_decisions.append(act)


            self.decisions = new_decisions

        return pruned_count


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        feats = self.qnode(self.decisions, self.params, x)

        if isinstance(feats, (list, tuple)):
            return torch.stack(feats, dim=-1)

        return feats

    @torch.no_grad()
    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def draw_circuit(self):
        dummy_size = max(2**self.n_qubits, self.num_embed_x)
        dummy_x = torch.zeros(dummy_size)
        dummy_x[0] = 1.0
        drawer = qml.draw(self.qnode)
        return drawer(self.decisions, self.params, dummy_x)

    def draw_circuit_mpl(self, **kwargs):
        dummy_size = max(2**self.n_qubits, self.num_embed_x)
        dummy_x = torch.zeros(dummy_size)
        dummy_x[0] = 1.0
        fig, ax = qml.draw_mpl(self.qnode, **kwargs)(self.decisions, self.params, dummy_x)
        return fig, ax


def _make_worker_qnodes(n_qubits, device="cuda"):
    """ worker  qnode/ansatz """
    ansatz_simple = DecisionsAnsatzSimple()
    ansatz = DecisionsAnsatz()
    measurement = StateMeasurement()
    qnode = make_noisy_qnode(ansatz_simple, measurement, n_qubits, dev_name="default.qubit")
    qnode_embed = make_noisy_qnode(ansatz, measurement, n_qubits, dev_name="default.qubit")
    return qnode, qnode_embed


import ray
import os


@ray.remote
def _compute_one_remote(
    decisions,
    n_qubits,
    coupling_map,
    seed,
    trainability_kwargs,
    haar_points,
    points,
    num_random_initial_expr,
):
    np.random.seed(seed)
    torch.manual_seed(seed)


    qnode, qnode_embed = _make_worker_qnodes(n_qubits)

    Graph, _ = decisions_to_graph(decisions, n_qubits)
    fingerprint = nx.weisfeiler_lehman_graph_hash(Graph, node_attr="label", edge_attr="role")


    expressibility_embed = calculate_expressibility_embed_single(
        qnode_embed, num_random_initial_expr, decisions, points, haar_points
    )
    expressibility = calculate_expressibility_single(points, num_random_initial_expr, decisions, qnode, haar_points)

    topo_order = list(nx.topological_sort(Graph))
    n_encoding, n_trainable, encoding_param_ratio, redundancy_ratio, n_two_qubit_gates = (
        compute_basic_and_redundancy_proxies(Graph, n_qubits, topo_order)
    )
    mean_length, width = calculate_circuit_depth_and_width(Graph, topo_order)
    path_alternation_rates = compute_causal_path_alternation_intrinsic(Graph, topo_order)
    mean_encode_cent, mean_trainable_cent, data_dominance = compute_centrality_proxies(Graph)

    encoding_entropy, param_entropy, global_entropy, ent_entropy_universal, ent_entropy_hardware, ep_js_divergence = (
        compute_qubit_distribution_entropies(Graph, n_qubits, coupling_map)
    )
    st_mutual_info = compute_spatiotemporal_mutual_information(Graph, topo_order)
    topological_center_diff = compute_topological_center_diff(Graph)

    trainability, snip = compute_trainability_and_snip(
        decisions=decisions, n_qubits=n_qubits, seed=seed, **trainability_kwargs
    )
    error_lightcone_vol, lifespan_imbalance, data_reinjection_index = compute_noise_aware_proxies(Graph, n_qubits)

    proxies_metric = {
        "graph": Graph,
        "expressibility": expressibility,
        "expressibility_embed": expressibility_embed,
        "mean_length": mean_length,
        "width": width,
        "mean_encode_cent": mean_encode_cent,
        "mean_trainable_cent": mean_trainable_cent,
        "data_dominance": data_dominance,
        "n_encoding": n_encoding,
        "n_trainable": n_trainable,
        "n_two_qubit_gates": n_two_qubit_gates,
        "encoding_param_ratio": encoding_param_ratio,
        "redundancy_ratio": redundancy_ratio,
        "encoding_entropy": encoding_entropy,
        "param_entropy": param_entropy,
        "global_entropy": global_entropy,
        "ent_entropy_universal": ent_entropy_universal,
        "ent_entropy_hardware": ent_entropy_hardware,
        "ep_js_divergence": ep_js_divergence,
        "st_mutual_info": st_mutual_info,
        "topological_center_diff": topological_center_diff,
        "path_alternation_rates": path_alternation_rates,
        "trainability": trainability,
        "snip": snip,
        "error_lightcone_vol": error_lightcone_vol,
        "lifespan_imbalance": lifespan_imbalance,
        "data_reinjection_index": data_reinjection_index,
    }

    return fingerprint, proxies_metric


def calculate_graph_proxies_ray(
    decisions_iterable,
    n_qubits: int,
    coupling_map: list[list[int]] | None,
    db_name: str,
    disable_tqdm: bool = False,
    seed: int = 0,
    num_cpus: float = 1,
    num_gpus: float = 0.1,
    trainability_kwargs: dict = None,
    total_circuits: int = None,
    shutdown_ray: bool = True,
):
    if trainability_kwargs is None:
        trainability_kwargs = {"num_random_initial": 32, "num_classes": 4}

    np.random.seed(seed)
    torch.manual_seed(seed)

    points = 100
    haar_points = calculate_haar_dist(n_qubits, points)
    num_random_initial_expr = 8000


    haar_points_ref = ray.put(haar_points)
    train_kwargs_ref = ray.put(trainability_kwargs)


    ordered_fps = []
    in_flight_fps = set()
    fp_to_count = defaultdict(int)

    pending_refs = []

    MAX_IN_FLIGHT = 2000
    batch_size = 64

    iterator = iter(decisions_iterable)
    generator_exhausted = False

    print("[Main] Starting the lazy streaming pipeline...")
    pbar = tqdm(total=total_circuits, desc="Ray Proxy Computation", disable=disable_tqdm)

    try:

        with shelve.open(db_name) as db:
            while True:

                while len(pending_refs) < MAX_IN_FLIGHT and not generator_exhausted:
                    try:
                        decisions = next(iterator)
                    except StopIteration:
                        generator_exhausted = True
                        break


                    Graph, _ = decisions_to_graph(decisions, n_qubits)
                    fp = nx.weisfeiler_lehman_graph_hash(Graph, node_attr="label", edge_attr="role")

                    ordered_fps.append(fp)


                    if fp in db:
                        pbar.update(1)
                    elif fp in in_flight_fps:
                        fp_to_count[fp] += 1
                    else:
                        in_flight_fps.add(fp)
                        fp_to_count[fp] = 1


                        ref = _compute_one_remote.options(num_cpus=num_cpus, num_gpus=num_gpus).remote(
                            decisions,
                            n_qubits,
                            coupling_map,
                            seed,
                            train_kwargs_ref,
                            haar_points_ref,
                            points,
                            num_random_initial_expr,
                        )
                        pending_refs.append(ref)


                if generator_exhausted and not pending_refs:
                    break


                done, pending_refs = ray.wait(pending_refs, num_returns=min(batch_size, len(pending_refs)), timeout=1.0)

                if done:
                    results_batch = ray.get(done)
                    for fp_done, metric in results_batch:

                        db[fp_done] = metric


                        in_flight_fps.remove(fp_done)


                        count = fp_to_count.pop(fp_done, 0)
                        if count > 0:
                            pbar.update(count)


                    db.sync()


                    ray.internal.free(done, local_only=False)

    finally:

        pbar.close()
        ray.internal.free([haar_points_ref, train_kwargs_ref], local_only=False)


        if shutdown_ray:
            print("[Main] shutdown_ray=True; terminating the Ray cluster safely...")
            ray.shutdown()


    return ordered_fps


from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeCasablancaV2


from qiskit.primitives import BackendEstimatorV2, StatevectorEstimator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient


def apply_gate_to_circuit(qc: QuantumCircuit, gate: str, wires: list, angle=None):
    gate = gate.upper()


    if gate in ("RX", "RY", "RZ"):
        w = wires[0]
        if gate == "RX":
            qc.rx(angle, w)
        elif gate == "RY":
            qc.ry(angle, w)
        elif gate == "RZ":
            qc.rz(angle, w)


    elif gate in ("CRX", "CRY", "CRZ"):
        c, t = wires[0], wires[1]
        if gate == "CRX":
            qc.crx(angle, c, t)
        elif gate == "CRY":
            qc.cry(angle, c, t)
        elif gate == "CRZ":
            qc.crz(angle, c, t)


    elif gate == "H":
        qc.h(wires[0])
    elif gate == "S":
        qc.s(wires[0])
    elif gate == "T":
        qc.t(wires[0])
    elif gate in ("I", "ID"):
        qc.id(wires[0])
    elif gate == "SX":
        qc.sx(wires[0])


    elif gate == "GPI":

        w = wires[0]
        qc.rz(-angle, w)
        qc.rx(np.pi, w)
        qc.rz(angle, w)
    elif gate == "GPI2":

        w = wires[0]
        qc.rz(-angle, w)
        qc.rx(np.pi / 2, w)
        qc.rz(angle, w)


    elif gate in ("CNOT", "CX"):
        qc.cx(wires[0], wires[1])
    elif gate == "CZ":
        qc.cz(wires[0], wires[1])
    elif gate == "SWAP":
        qc.swap(wires[0], wires[1])
    elif gate == "ISWAP":
        qc.iswap(wires[0], wires[1])
    elif gate == "ECR":
        qc.ecr(wires[0], wires[1])


    elif gate == "MS":
        theta = angle if angle is not None else (np.pi / 2)
        qc.ms(theta, wires)


    elif gate in ("RXX", "ISINGXX"):
        qc.rxx(angle, wires[0], wires[1])
    elif gate in ("RZZ", "ISINGZZ"):
        qc.rzz(angle, wires[0], wires[1])

    else:
        raise ValueError(f"[Qiskit Converter] unsupported gate type: {gate}")


def build_qiskit_circuit(decisions, n_qubits, num_x, num_theta):
    qc = QuantumCircuit(n_qubits)

    x_params = ParameterVector("x", num_x) if num_x > 0 else []
    theta_params = ParameterVector("θ", num_theta) if num_theta > 0 else []

    for act in decisions:
        gate = act["gate"]
        wires = act["wires"]

        angle = None
        if "x_index" in act:
            angle = x_params[act["x_index"]]
        elif "param_idx" in act:
            angle = theta_params[act["param_idx"]]


        apply_gate_to_circuit(qc, gate, wires, angle)

    return qc, x_params, theta_params


from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


class QiskitVQCFeatureExtractor(nn.Module):
    def __init__(self, decisions, n_qubits, task="VQE", noise_param=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.task = task


        self.num_embed_x = 1 + max([act["x_index"] for act in decisions if "x_index" in act], default=-1)
        self.num_params = 1 + max([act["param_idx"] for act in decisions if "param_idx" in act], default=-1)


        self.qc, self.x_params, self.theta_params = build_qiskit_circuit(
            decisions, n_qubits, self.num_embed_x, self.num_params
        )


        if task == "Classification":
            observables = []
            for i in range(n_qubits):
                pauli = ["I"] * n_qubits
                pauli[i] = "Z"

                observables.append(SparsePauliOp("".join(pauli)[::-1]))

        elif task == "VQE":

            pauli_list = []
            for i in range(1, n_qubits - 1):
                pauli = ["I"] * n_qubits
                pauli[i - 1] = "Z"
                pauli[i] = "X"
                pauli[i + 1] = "Z"

                pauli_str = "".join(pauli)[::-1]
                pauli_list.append((pauli_str, -1.0))


            observables = [SparsePauliOp.from_list(pauli_list)]

        else:
            raise ValueError(f"Unsupported task: {task}")


        backend = noise_param.get("backend", ()) if noise_param else FakeCasablancaV2()
        self.estimator = BackendEstimatorV2(backend=backend)

        grad = ParamShiftEstimatorGradient(self.estimator)

        pm = generate_preset_pass_manager(optimization_level=1, target=backend.target)


        self.isa_qc = pm.run(self.qc)


        if self.isa_qc.layout is not None:
            isa_observables = [obs.apply_layout(self.isa_qc.layout) for obs in observables]
        else:
            isa_observables = observables


        self.qnn = EstimatorQNN(
            circuit=self.isa_qc,
            observables=isa_observables,
            input_params=self.x_params if len(self.x_params) > 0 else None,
            weight_params=self.theta_params if len(self.theta_params) > 0 else None,
            estimator=self.estimator,
            gradient=grad,
            input_gradients=True,
        )


        self.vqc_layer = TorchConnector(self.qnn)

    def forward(self, x=None):

        if x is None:
            x = torch.empty((1, 0))

        q_out = self.vqc_layer(x)

        if q_out.dim() == 1:
            q_out = q_out.unsqueeze(0)

        return q_out
