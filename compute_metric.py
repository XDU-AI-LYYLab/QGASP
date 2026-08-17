import torch
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

from quantum_api_v3 import VQCFeatureExtractor, PauliZMeasurement, QiskitVQCFeatureExtractor
import gc
import ray
import shelve
from quantum_api_v3 import decisions_to_graph, nx
import copy


def ensure_dataloader(ds_or_loader, batch_size=32, shuffle=True, num_workers=0, pin_memory=False, seed=None):
    if isinstance(ds_or_loader, DataLoader):
        return ds_or_loader
    if isinstance(ds_or_loader, Dataset):
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        return DataLoader(
            ds_or_loader,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=gen,
        )
    raise TypeError("train/val/test must be torch.utils.data.Dataset (DataLoader not accepted).")


@ray.remote
def _train_decision_remote(
    decisions,
    train_dataset,
    val_dataset,
    test_dataset,
    n_qubits,
    input_dim,
    dev_name,
    noise_param,
    epochs,
    patience,
    log_dir,
    device_str,
    batch_size,
    num_workers,
    pin_memory,
    seed: int,
    shuffle_seed=None,
    min_epochs=0,
):

    device = torch.device(device_str)

    train_ds = train_dataset
    val_ds = val_dataset
    test_ds = test_dataset  #


    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(shuffle_seed if shuffle_seed is not None else seed)
        if (shuffle_seed is not None or seed is not None)
        else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


    model = VQCClassifier(
        decisions,
        n_qubits,
        input_dim,
        dev_name,
        seed=seed,
        noise_param=noise_param,
    )
    model = model.to(device)


    if getattr(model, "embed_dim", 0) == 0:
        try:
            del model
        except Exception:
            pass
        del train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 0.0


    try:
        with torch.enable_grad():
            acc = train_and_evaluate(
                model,
                train_loader,
                val_loader,
                test_loader,
                epochs=epochs,
                patience=patience,
                log_dir=log_dir,
                device=device,
                min_epochs=min_epochs,
            )
            acc_test = float(acc)
    finally:

        try:
            del model
        except Exception:
            pass
        try:
            del train_loader, val_loader, test_loader
        except Exception:
            pass
        try:
            del train_ds, val_ds, test_ds
        except Exception:
            pass


        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return acc_test


class VQCPipeline:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        n_qubits=4,
        input_dim=16,
        epochs=200,
        patience=25,
        log_dir=None,
        dev_name="default.qubit",
        noise_param=None,
        device=None,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        seed=42,
        shuffle_seed=None,
        min_epochs=15,
        db_name="target_results.db",
    ):
        if (
            not isinstance(train_dataset, Dataset)
            or not isinstance(val_dataset, Dataset)
            or not isinstance(test_dataset, Dataset)
        ):
            raise TypeError("train_dataset, val_dataset, and test_dataset must be torch.utils.data.Dataset instances.")

        self._train_input = train_dataset
        self._val_input = val_dataset
        self._test_input = test_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.shuffle_seed = shuffle_seed
        self.min_epochs = min_epochs

        self.n_qubits = n_qubits
        self.input_dim = input_dim
        self.epochs = epochs
        self.patience = patience
        self.log_dir = log_dir
        self.noise_param = noise_param
        self.dev_name = dev_name
        self.db_name = db_name

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _make_dataloaders(self):
        train_loader = ensure_dataloader(
            self._train_input,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            seed=self.shuffle_seed if self.shuffle_seed is not None else self.seed,
        )
        val_loader = ensure_dataloader(
            self._val_input,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            seed=None,
        )
        test_loader = ensure_dataloader(
            self._test_input,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            seed=None,
        )
        return train_loader, val_loader, test_loader

    def __call__(self, obs_list, mode="ray", num_cpus=2, num_gpus=0.05, disable_tqdm=True):
        decisions_list = [list(iter_obs_decisions(obs)) for obs in obs_list]
        return self.by_decisions(
            decisions_list, mode=mode, num_cpus=num_cpus, num_gpus=num_gpus, disable_tqdm=disable_tqdm
        )

    def by_decisions(
        self, decisions_list, mode="ray", num_cpus=2, num_gpus=0.05, disable_tqdm=False, shutdown_ray=True, update=False
    ):
        if mode not in ("loop", "ray"):
            raise ValueError("mode must be one of 'loop' or 'ray'")


        if update not in (False, True, "greater", "less"):
            raise ValueError("update parameter must be one of: False, True, 'greater', 'less'")

        results = [None] * len(decisions_list)
        tasks_to_run = []


        cache_key = self.seed if self.shuffle_seed is None else f"{self.seed}|{self.shuffle_seed}"


        with shelve.open(self.db_name) as db:
            if not update:
                print(f"Scanning architecture cache DB (read-only, seed={self.seed})...")
            else:
                print(f"Forced computation mode (update={update}, seed={self.seed}): computing all required architectures and updating the cache.")


            for idx, decisions in enumerate(decisions_list):
                Graph, _ = decisions_to_graph(decisions, self.n_qubits)
                fingerprint = nx.weisfeiler_lehman_graph_hash(Graph, node_attr="label", edge_attr="role")

                if fingerprint in db:
                    db_val = db[fingerprint]


                    if not isinstance(db_val, dict):
                        db_dict = {self.seed: db_val}
                    else:
                        db_dict = db_val


                    old_val = db_dict.get(cache_key, None)

                    if old_val is not None and not update:

                        results[idx] = old_val
                    else:

                        tasks_to_run.append((idx, decisions, fingerprint, old_val))
                else:

                    tasks_to_run.append((idx, decisions, fingerprint, None))

            if not update:
                print(
                    f"Cache hit for {len(decisions_list) - len(tasks_to_run)} architectures; {len(tasks_to_run)} remain to compute."
                )


            if not tasks_to_run:
                return results


            if mode == "loop":
                for idx, decisions, fingerprint, old_val in tqdm(
                    tasks_to_run, desc="Evaluating architectures serially", disable=disable_tqdm
                ):
                    model = VQCClassifier(
                        decisions,
                        self.n_qubits,
                        self.input_dim,
                        self.dev_name,
                        seed=self.seed,
                        noise_param=self.noise_param,
                    )
                    model = model.to(self.device)

                    train_loader, val_loader, test_loader = self._make_dataloaders()

                    if getattr(model, "embed_dim", 0) == 0:
                        acc = 0.0
                    else:
                        with torch.enable_grad():
                            acc = train_and_evaluate(
                                model,
                                train_loader,
                                val_loader,
                                test_loader,
                                epochs=self.epochs,
                                patience=self.patience,
                                log_dir=self.log_dir if self.log_dir is None else self.log_dir + f"_{idx}",
                                device=self.device,
                                min_epochs=self.min_epochs,
                            )

                    res = float(acc)


                    final_res = res
                    if old_val is not None:
                        if update == "greater" and res <= old_val:
                            final_res = old_val
                        elif update == "less" and res >= old_val:
                            final_res = old_val

                    results[idx] = final_res


                    if old_val is None or final_res != old_val:

                        db_dict = db.get(fingerprint, {})
                        if not isinstance(db_dict, dict):
                            db_dict = {self.seed: db_dict}

                        db_dict[cache_key] = final_res
                        db[fingerprint] = db_dict
                        db.sync()


            elif mode == "ray":
                train_ref, val_ref, test_ref = None, None, None
                in_flight, ref_to_info = None, None

                try:
                    train_ref = ray.put(self._train_input)
                    val_ref = ray.put(self._val_input)
                    test_ref = ray.put(self._test_input)

                    K = 100
                    ray_wait_batch_size = 32

                    in_flight = []
                    ref_to_info = {}

                    task_iter = iter(tasks_to_run)
                    total_tasks = len(tasks_to_run)

                    def submit_tasks(num_to_submit):
                        for _ in range(num_to_submit):
                            try:

                                idx, decisions, fingerprint, old_val = next(task_iter)
                                log_dir_curr = self.log_dir if self.log_dir is None else (self.log_dir + f"_{idx}")

                                fut = _train_decision_remote.options(num_cpus=num_cpus, num_gpus=num_gpus).remote(  # type: ignore
                                    decisions,
                                    train_ref,
                                    val_ref,
                                    test_ref,
                                    self.n_qubits,
                                    self.input_dim,
                                    self.dev_name,
                                    self.noise_param,
                                    self.epochs,
                                    self.patience,
                                    log_dir_curr,
                                    str(self.device),
                                    self.batch_size,
                                    self.num_workers,
                                    self.pin_memory,
                                    self.seed,
                                    shuffle_seed=self.shuffle_seed,
                                    min_epochs=self.min_epochs,
                                )
                                in_flight.append(fut)

                                ref_to_info[fut] = (idx, fingerprint, old_val)
                            except StopIteration:
                                break

                    print(f"Submitting tasks to the Ray cluster (max concurrency K={K})...")

                    submit_tasks(K)

                    with tqdm(total=total_tasks, desc="Ray async evaluation progress", disable=disable_tqdm) as pbar:
                        while in_flight:
                            wait_limit = min(ray_wait_batch_size, len(in_flight))
                            done, in_flight = ray.wait(in_flight, num_returns=wait_limit, timeout=1.0)

                            if done:
                                db_updated_flag = False
                                for ref in done:
                                    res = ray.get(ref)

                                    idx, fingerprint, old_val = ref_to_info.pop(ref)


                                    final_res = res
                                    if old_val is not None:
                                        if update == "greater" and res <= old_val:
                                            final_res = old_val
                                        elif update == "less" and res >= old_val:
                                            final_res = old_val

                                    results[idx] = final_res


                                    if old_val is None or final_res != old_val:
                                        db_dict = db.get(fingerprint, {})
                                        if not isinstance(db_dict, dict):
                                            db_dict = {self.seed: db_dict}

                                        db_dict[cache_key] = final_res
                                        db[fingerprint] = db_dict
                                        db_updated_flag = True

                                    if hasattr(ray.internal, "free"):
                                        ray.internal.free(ref, local_only=False)

                                    pbar.update(1)


                                if db_updated_flag:
                                    db.sync()

                            slots_available = K - len(in_flight)
                            if slots_available > 0:
                                submit_tasks(slots_available)

                finally:
                    refs_to_free = [r for r in (train_ref, val_ref, test_ref) if r is not None]
                    if refs_to_free and hasattr(ray.internal, "free"):
                        ray.internal.free(refs_to_free, local_only=False)

                    try:
                        del in_flight, train_ref, val_ref, test_ref, ref_to_info
                    except NameError:
                        pass

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if shutdown_ray:
                        print("[Main] shutdown_ray=True (or an exception occurred); terminating the Ray cluster safely...")
                        ray.shutdown()

        return results


class VQCClassifier(nn.Module):
    def __init__(self, decisions, n_qubits, input_dim, dev_name, seed=42, noise_param=None):
        super().__init__()
        self.vqc = VQCFeatureExtractor(
            decisions,
            n_qubits,
            PauliZMeasurement(n_qubits),
            dev_name=dev_name,
            trainable=True,
            seed=seed,
            noise_param=noise_param,
        )

        self.embed_dim = self.vqc.num_embed_x
        self.param_dim = self.vqc.num_params
        if self.embed_dim == 0:
            return
        self.fc = nn.Linear(input_dim, self.embed_dim)
        nn.init.kaiming_normal_(self.fc.weight, generator=torch.Generator().manual_seed(seed))
        nn.init.zeros_(self.fc.bias)

    def draw(self):
        self.vqc.draw_circuit_mpl()

    def forward(self, x):
        x = self.fc(x)
        return self.vqc(x)


def train_and_evaluate(
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    epochs=300,
    patience=25,
    log_dir=None,
    device=torch.device("cpu"),
    min_epochs=0,
):
    try:
        model.to(device)
    except Exception:
        device = torch.device("cpu")
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_counter = 0
    best_state_dict = None

    writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None

    for epoch in range(epochs):

        total_loss = 0.0
        preds_list, y_list = [], []
        model.train()

        for x_batch, y_batch in train_dataloader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            preds_list.append(outputs.argmax(dim=-1).detach())
            y_list.append(y_batch.detach())

        avg_train_loss = total_loss / max(1, len(train_dataloader))
        preds_all = torch.cat(preds_list, dim=0)
        y_all = torch.cat(y_list, dim=0)
        train_acc = (preds_all == y_all).float().mean().item()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x_val, y_val in val_dataloader:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                outputs = model(x_val)
                val_preds.append(outputs.argmax(dim=-1).detach())
                val_labels.append(y_val.detach())

        val_preds = torch.cat(val_preds, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        val_acc = (val_preds == val_labels).float().mean().item()
        if writer:
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch)


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            if epoch >= min_epochs:
                patience_counter += 1
                if patience_counter >= patience:
                    break


    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for x_test, y_test in test_dataloader:
            x_test = x_test.to(device, non_blocking=True)
            y_test = y_test.to(device, non_blocking=True)
            outputs = model(x_test)
            test_preds.append(outputs.argmax(dim=-1).detach())
            test_labels.append(y_test.detach())


    if len(test_preds) > 0:
        test_preds = torch.cat(test_preds, dim=0)
        test_labels = torch.cat(test_labels, dim=0)
        test_acc = (test_preds == test_labels).float().mean().item()
    else:
        test_acc = 0.0

    if writer:
        writer.add_scalar("Accuracy/Test_Final", test_acc, epoch)
        writer.close()


    return test_acc


from quantum_api_v3 import HamiltonianMeasurement


class VQEModel(nn.Module):

    def __init__(self, decisions, n_qubits, dev_name, hamiltonian, seed=42, noise_param=None):
        super().__init__()
        self.vqc = VQCFeatureExtractor(
            decisions,
            n_qubits,
            HamiltonianMeasurement(hamiltonian),
            dev_name=dev_name,
            trainable=True,
            seed=seed,
            noise_param=noise_param,
        )
        self.param_dim = self.vqc.num_params
        self.embed_dim = self.vqc.num_embed_x

    def forward(self, x=None):
        return self.vqc(x)


def train_vqe_model(
    model,
    steps=1200,
    patience=3,
    lr=0.01,
    log_dir=None,
    device=torch.device("cpu"),
):
    try:
        model.to(device)
    except Exception:
        device = torch.device("cpu")
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_energy = float("inf")
    patience_counter = 0

    writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None

    for step in range(steps):
        model.train()
        optimizer.zero_grad()


        energy = model().mean()


        energy.backward()
        optimizer.step()

        current_energy = energy.item()
        if writer:
            writer.add_scalar("Energy", current_energy, step)

        if step % 50 == 0:

            if current_energy < best_energy - 1e-4:
                best_energy = current_energy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    if writer:
        writer.close()

    return best_energy


@ray.remote
def _train_decision_remote_vqe(
    decisions,
    n_qubits,
    hamiltonian,
    dev_name,
    noise_param,
    steps,
    patience,
    lr,
    log_dir,
    seed,
    device_str,
):
    device = torch.device(device_str)


    model = VQEModel(decisions, n_qubits, dev_name, hamiltonian, seed=seed, noise_param=noise_param)
    model = model.to(device)


    if getattr(model, "param_dim", 0) == 0:
        with torch.no_grad():
            energy_val = model().mean().item()
        try:
            del model
            gc.collect()
        except Exception:
            pass
        return float(energy_val)


    try:
        with torch.enable_grad():
            energy_val = train_vqe_model(
                model,
                steps=steps,
                patience=patience,
                lr=lr,
                log_dir=log_dir,
                device=device,
            )
    finally:
        try:
            del model
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return float(energy_val)


class VQEPipeline:

    def __init__(
        self,
        hamiltonian,
        n_qubits=4,
        steps=1500,
        patience=3,
        lr=0.01,
        log_dir=None,
        dev_name="default.qubit",
        noise_param=None,
        device=None,
        seed=42,
        db_name="vqe_target_energy.db",
    ):
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits
        self.steps = steps
        self.patience = patience
        self.lr = lr
        self.log_dir = log_dir
        self.noise_param = noise_param
        self.dev_name = (dev_name,)
        self.seed = seed
        self.db_name = db_name

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def __call__(self, obs_list, mode="ray", num_cpus=2, num_gpus=0.05, disable_tqdm=True):
        decisions_list = [list(iter_obs_decisions(obs)) for obs in obs_list]
        return self.by_decisions(
            decisions_list, mode=mode, num_cpus=num_cpus, num_gpus=num_gpus, disable_tqdm=disable_tqdm
        )

    def by_decisions(
        self, decisions_list, mode="ray", num_cpus=2, num_gpus=0.05, disable_tqdm=False, shutdown_ray=True, update=False
    ):
        if mode not in ("loop", "ray"):
            raise ValueError("mode must be one of 'loop' or 'ray'")


        if update not in (False, True, "greater", "less"):
            raise ValueError("update parameter must be one of: False, True, 'greater', 'less'")

        results = [None] * len(decisions_list)
        tasks_to_run = []

        with shelve.open(self.db_name) as db:
            if not update:
                print(f"Scanning VQE architecture cache DB (read-only, seed={self.seed})...")
            else:
                print(f"Forced computation mode (update={update}, seed={self.seed}): computing all required architectures and updating the cache.")


            for idx, decisions in enumerate(decisions_list):
                Graph, _ = decisions_to_graph(decisions, self.n_qubits)
                fingerprint = nx.weisfeiler_lehman_graph_hash(Graph, node_attr="label", edge_attr="role")

                if fingerprint in db:
                    db_val = db[fingerprint]


                    if not isinstance(db_val, dict):
                        db_dict = {self.seed: db_val}
                    else:
                        db_dict = db_val


                    old_val = db_dict.get(self.seed, None)

                    if old_val is not None and not update:

                        results[idx] = old_val
                    else:

                        tasks_to_run.append((idx, decisions, fingerprint, old_val))
                else:

                    tasks_to_run.append((idx, decisions, fingerprint, None))

            if not update:
                print(
                    f"Cache hit for {len(decisions_list) - len(tasks_to_run)} architectures; {len(tasks_to_run)} remain to compute."
                )

            if not tasks_to_run:
                return results


            if mode == "loop":
                for idx, decisions, fingerprint, old_val in tqdm(
                    tasks_to_run, desc="Evaluating ground-state energies serially", disable=disable_tqdm
                ):
                    model = VQEModel(
                        decisions,
                        self.n_qubits,
                        self.dev_name,
                        self.hamiltonian,
                        seed=self.seed,
                        noise_param=self.noise_param,
                    )
                    model = model.to(self.device)

                    if getattr(model, "param_dim", 0) == 0:
                        with torch.no_grad():
                            energy = float(model().mean().item())
                    else:
                        with torch.enable_grad():
                            energy = train_vqe_model(
                                model,
                                steps=self.steps,
                                patience=self.patience,
                                lr=self.lr,
                                log_dir=self.log_dir if self.log_dir is None else self.log_dir + f"_{idx}",
                                device=self.device,
                            )

                    res = float(energy)


                    final_res = res
                    if old_val is not None:
                        if update == "greater" and res <= old_val:
                            final_res = old_val
                        elif update == "less" and res >= old_val:
                            final_res = old_val

                    results[idx] = final_res


                    if old_val is None or final_res != old_val:

                        db_dict = db.get(fingerprint, {})
                        if not isinstance(db_dict, dict):
                            db_dict = {self.seed: db_dict}

                        db_dict[self.seed] = final_res
                        db[fingerprint] = db_dict
                        db.sync()


            elif mode == "ray":
                in_flight = None
                ref_to_info = None

                try:


                    hamiltonian_ref = ray.put(self.hamiltonian)


                    K = 100
                    batch_size = 32

                    ref_to_info = {}
                    in_flight = []


                    task_iter = iter(tasks_to_run)
                    total_tasks = len(tasks_to_run)

                    def submit_tasks(num_to_submit):
                        """"""
                        for _ in range(num_to_submit):
                            try:

                                idx, decisions, fingerprint, old_val = next(task_iter)
                                log_dir_curr = self.log_dir if self.log_dir is None else (self.log_dir + f"_{idx}")

                                fut = _train_decision_remote_vqe.options(num_cpus=num_cpus, num_gpus=num_gpus).remote(  # type: ignore
                                    decisions,
                                    self.n_qubits,
                                    hamiltonian_ref,
                                    self.dev_name,
                                    self.noise_param,
                                    self.steps,
                                    self.patience,
                                    self.lr,
                                    log_dir_curr,
                                    self.seed,
                                    str(self.device),
                                )
                                in_flight.append(fut)

                                ref_to_info[fut] = (idx, fingerprint, old_val)
                            except StopIteration:
                                break

                    print(f"Submitting VQE tasks to the Ray cluster (max concurrency K={K})...")


                    submit_tasks(K)

                    with tqdm(total=total_tasks, desc="Ray async ground-state progress", disable=disable_tqdm) as pbar:
                        while in_flight:

                            wait_limit = min(batch_size, len(in_flight))
                            done, in_flight = ray.wait(in_flight, num_returns=wait_limit, timeout=1.0)

                            if done:
                                db_updated_flag = False
                                for ref in done:
                                    res = ray.get(ref)


                                    idx, fingerprint, old_val = ref_to_info.pop(ref)


                                    final_res = res
                                    if old_val is not None:
                                        if update == "greater" and res <= old_val:
                                            final_res = old_val
                                        elif update == "less" and res >= old_val:
                                            final_res = old_val

                                    results[idx] = final_res


                                    if old_val is None or final_res != old_val:
                                        db_dict = db.get(fingerprint, {})
                                        if not isinstance(db_dict, dict):
                                            db_dict = {self.seed: db_dict}

                                        db_dict[self.seed] = final_res
                                        db[fingerprint] = db_dict
                                        db_updated_flag = True

                                    if hasattr(ray.internal, "free"):
                                        ray.internal.free(ref, local_only=False)

                                    pbar.update(1)


                                if db_updated_flag:
                                    db.sync()


                            slots_available = K - len(in_flight)
                            if slots_available > 0:
                                submit_tasks(slots_available)

                finally:
                    if "hamiltonian_ref" in locals() and hasattr(ray.internal, "free"):
                        ray.internal.free([hamiltonian_ref], local_only=False)

                    try:
                        del in_flight, ref_to_info
                    except NameError:
                        pass

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if shutdown_ray:
                        print("[Main] Terminating the Ray cluster safely...")
                        ray.shutdown()

        return results


import numpy as np
import matplotlib.pyplot as plt


def compute_relative_topk_gain_auc(score_pred, score_target, exp_decay_rate=5.0, plot=False):
    """

    """
    N = len(score_target)
    if N <= 1:
        return 0.0, 0.0


    x = np.arange(1, N + 1) / N


    ideal_sort_idx = np.argsort(score_target)[::-1]
    target_sorted_by_ideal = score_target[ideal_sort_idx]


    worst_sort_idx = np.argsort(score_target)
    target_sorted_by_worst = score_target[worst_sort_idx]


    pred_sort_idx = np.argsort(score_pred)[::-1]
    target_sorted_by_pred = score_target[pred_sort_idx]


    actual_means = np.cumsum(target_sorted_by_pred) / np.arange(1, N + 1)
    ideal_means = np.cumsum(target_sorted_by_ideal) / np.arange(1, N + 1)
    worst_means = np.cumsum(target_sorted_by_worst) / np.arange(1, N + 1)
    global_mean = target_sorted_by_ideal.mean()

    actual_maxs = np.maximum.accumulate(target_sorted_by_pred)
    ideal_maxs = np.maximum.accumulate(target_sorted_by_ideal)
    worst_maxs = np.maximum.accumulate(target_sorted_by_worst)


    shift_val = np.min(score_target)

    actual_means_shift = actual_means - shift_val
    ideal_means_shift = ideal_means - shift_val
    worst_means_shift = worst_means - shift_val

    actual_maxs_shift = actual_maxs - shift_val
    ideal_maxs_shift = ideal_maxs - shift_val
    worst_maxs_shift = worst_maxs - shift_val


    weights = ideal_means_shift - np.min(ideal_means_shift) + 1e-6


    auc_actual_mean = np.trapezoid(actual_means_shift * weights, x)
    auc_ideal_mean = np.trapezoid(ideal_means_shift * weights, x)
    auc_worst_mean = np.trapezoid(worst_means_shift * weights, x)

    auc_actual_max = np.trapezoid(actual_maxs_shift, x)
    auc_ideal_max = np.trapezoid(ideal_maxs_shift, x)
    auc_worst_max = np.trapezoid(worst_maxs_shift, x)


    def normalize_auc_to_01(actual, ideal, worst):
        denom = ideal - worst
        if denom < 1e-9:
            return 0.0

        return float(np.clip((actual - worst) / denom, 0.0, 1.0))

    adjusted_mean_auc = normalize_auc_to_01(auc_actual_mean, auc_ideal_mean, auc_worst_mean)
    max_auc = normalize_auc_to_01(auc_actual_max, auc_ideal_max, auc_worst_max)


    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))


        axes[0].plot(
            x, ideal_means, label="Ideal (God Mode)", color="#2ca02c", linestyle="--", linewidth=2.5, alpha=0.8
        )
        axes[0].plot(x, actual_means, label="Actual Mean", color="#1f77b4", linewidth=2.5)
        axes[0].axhline(global_mean, color="gray", linestyle="-.", label="Random Pick (Global Mean)", linewidth=1.5)
        axes[0].plot(
            x, worst_means, label="Worst (Reverse Sort)", color="#d62728", linestyle=":", linewidth=2.0, alpha=0.7
        )

        axes[0].text(
            0.50,
            0.75,
            f"Norm Mean AUC: {adjusted_mean_auc:.4f}",
            transform=axes[0].transAxes,
            fontsize=14,
            fontweight="bold",
            color="#d62728",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="#d62728", boxstyle="round,pad=0.5"),
        )

        axes[0].set_xlabel("Top-K Ratio (Percentile)", fontweight="bold")
        axes[0].set_ylabel("Mean True Target Performance", fontweight="bold")
        axes[0].set_title("Relative Top-K Gain Curve (Mean Score)", fontweight="bold")
        axes[0].set_xlim(left=x[0], right=1.0)
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.2)


        axes[1].plot(
            x, ideal_maxs, label="Ideal Max (God Mode)", color="#2ca02c", linestyle="--", linewidth=2.5, alpha=0.8
        )
        axes[1].plot(x, actual_maxs, label="Actual Max", color="#1f77b4", linewidth=2.5)
        axes[1].plot(
            x, worst_maxs, label="Worst Max (Reverse Sort)", color="#d62728", linestyle=":", linewidth=2.0, alpha=0.7
        )

        axes[1].text(
            0.50,
            0.75,
            f"Norm Max AUC: {max_auc:.4f}",
            transform=axes[1].transAxes,
            fontsize=14,
            fontweight="bold",
            color="#d62728",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="#d62728", boxstyle="round,pad=0.5"),
        )

        axes[1].set_xlabel("Top-K Ratio (Percentile)", fontweight="bold")
        axes[1].set_ylabel("Cumulative Max True Performance", fontweight="bold")
        axes[1].set_title("Top-K Maximum Discovery Curve (Max Score)", fontweight="bold")
        axes[1].set_xlim(left=x[0], right=1.0)
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.2)

        plt.tight_layout()
        if isinstance(plot, str):
            plt.savefig(plot, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    return float(adjusted_mean_auc), float(max_auc)
