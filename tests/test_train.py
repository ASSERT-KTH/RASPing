"""Trainer behaviour that the golden numbers don't make explicit."""

from rasping import data, raspbugs
from rasping.losses import cross_entropy_loss
from rasping.train import Trainer

JOB = "2a722d5d5449441fad93f98c73c633a2"  # hist, order 1 (small and fast)


def _history(seed: int) -> dict:
    model = raspbugs.get(JOB).compile()
    X, Y = data.encode(data.load("hist", "train")[:128], model)
    return Trainer(model, X, Y, cross_entropy_loss, n_epochs=2, batch_size=32, lr=1e-3,
                   progress=False, seed=seed).train().history()


def test_batch_order_is_seeded():
    """Same seed -> identical run; different seed -> different batch order (F36)."""
    a, b, c = _history(0), _history(0), _history(1)
    assert a == b
    assert a["train_losses"] != c["train_losses"]
