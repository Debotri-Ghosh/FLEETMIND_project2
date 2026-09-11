"""Flower client + strategy wiring (Phase 4).

`DepotClient(flwr.client.NumPyClient)`: thin wrapper that plugs the
client-side helpers from `client.py` into Flower's NumPyClient protocol so a
real `flwr.server` (FedAvg / FedProx strategy) can orchestrate rounds over
gRPC. `run.py` uses an in-process loop with the same semantics for the M.Tech
demo (no open ports required), so this module is the seam where the demo
graduates to a true distributed deployment: swap `run_federated()`'s loop for
`fl.server.start_server(strategy=...)` + one `DepotClient` process per depot.
"""

from __future__ import annotations

import flwr as fl

from fleetmind.federated.client import (
    evaluate_model,
    get_parameters,
    local_train,
    set_parameters,
)
from fleetmind.ml.model import LABEL_MAP, BearingCNN


class DepotClient(fl.client.NumPyClient):
    def __init__(
        self,
        depot_id: int,
        train_rows: list[tuple[list[float], int]],
        test_rows: list[tuple[list[float], int]],
        *,
        local_epochs: int = 2,
        batch_size: int = 64,
        lr: float = 3e-3,
        mu: float = 0.0,
        seed: int = 42,
    ):
        self.depot_id = depot_id
        self.train_rows = train_rows
        self.test_rows = test_rows
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.mu = mu
        self.seed = seed
        self.model = BearingCNN(num_classes=len(LABEL_MAP))

    def get_parameters(self, config):  # noqa: ANN001, ANN201 - Flower protocol
        import numpy as np

        return [np.asarray(p, dtype="float32") for p in get_parameters(self.model)]

    def fit(self, parameters, config):  # noqa: ANN001, ANN201 - Flower protocol
        import numpy as np

        set_parameters(self.model, [np.asarray(p).tolist() for p in parameters])
        stats = local_train(
            self.model,
            self.train_rows,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            mu=self.mu,
            seed=self.seed,
        )
        import numpy as np

        return (
            [np.asarray(p, dtype="float32") for p in get_parameters(self.model)],
            len(self.train_rows),
            {"loss": float(stats["loss"])},
        )

    def evaluate(self, parameters, config):  # noqa: ANN001, ANN201 - Flower protocol
        import numpy as np

        set_parameters(self.model, [np.asarray(p).tolist() for p in parameters])
        res = evaluate_model(self.model, self.test_rows, batch_size=self.batch_size)
        return float(1.0 - res["accuracy"]), len(self.test_rows), res


def make_strategy(name: str, num_rounds: int):  # noqa: ANN201 - flwr types vary
    """Return the Flower server strategy for 'fedavg' or 'fedprox'."""
    name = name.lower()
    if name == "fedprox":
        from flwr.server.strategy import FedProx

        return FedProx(
            proximal_mu=0.1,
            min_available_clients=4,
            min_fit_clients=4,
            min_evaluate_clients=4,
        )
    from flwr.server.strategy import FedAvg

    return FedAvg(
        min_available_clients=4,
        min_fit_clients=4,
        min_evaluate_clients=4,
    )
