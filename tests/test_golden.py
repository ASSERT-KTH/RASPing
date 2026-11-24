"""Regression test: the refactored code reproduces the v1 (TMLR) code exactly.

`tests/golden/golden_v1.json` was recorded with the code at tag `tmlr-submission-1`
(src/ + experiments/): for one mutant per base program it stores digests of the
compiled weights, accuracies, a short training run with early stopping, and a Xavier
re-initialisation; and a tiny GP and BFS search for two mutants.

Any intentional behaviour change (a FINDINGS fix) that breaks this test must
re-record the golden file in the same commit and say so in docs/FINDINGS.md.

Training is only bit-reproducible on one machine with one thread count. XLA's CPU backend
splits reductions across threads, so the summation order differs between machines. The
compiled models have very sharp logits, and a few optimizer steps amplify those last-bit
differences: by epoch 4 a loss can be off by a few percent. The equivalence with the v1
code was established with `--golden-exact` on the recording machine. By default this test
checks what holds everywhere:

- compiled and re-initialised weights, encodings and accuracies, exactly;
- the first epoch of each training run to `FIRST_EPOCH_RTOL` (hardware noise there is
  ~2e-6 relative; a 1% learning-rate change moves it by >=2.5e-4 for every program);
- later epochs' losses, loosely (`TAIL_RTOL`), as a sanity check.

Pass `--golden-exact` to compare whole runs to 1e-7, including the trained weights.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from rasping import data, raspbugs
from rasping.losses import cross_entropy_loss
from rasping.train import Trainer

GOLDEN = json.loads((Path(__file__).parent / "golden" / "golden_v1.json").read_text())
PROGRAMS = list(GOLDEN["programs"])
RTOL = 1e-9
EXACT_EPOCHS = 1  # epochs compared to FIRST_EPOCH_RTOL
FIRST_EPOCH_RTOL = 2e-5  # see module docstring
TAIL_RTOL = 0.1   # later epochs diverge chaotically (observed: up to 2.1%)


@pytest.fixture
def exact(request) -> bool:
    return request.config.getoption("--golden-exact")


def digest(params) -> dict:
    flat = np.concatenate(
        [np.asarray(params[k][s], dtype=np.float64).ravel()
         for k in sorted(params) for s in sorted(params[k])]
    )
    return {"n": int(flat.size), "sum": float(flat.sum()), "abs_sum": float(np.abs(flat).sum()),
            "sq_sum": float((flat ** 2).sum()), "keys": sorted(params)}


def assert_digest(got: dict, exp: dict):
    assert got["n"] == exp["n"] and got["keys"] == exp["keys"]
    for k in ("sum", "abs_sum", "sq_sum"):
        np.testing.assert_allclose(got[k], exp[k], rtol=RTOL, atol=1e-9, err_msg=k)


def assert_history(got: dict, exp: dict, exact: bool):
    for k, v in exp.items():
        if exact:
            np.testing.assert_allclose(got[k], v, rtol=1e-7, atol=1e-9, err_msg=k)
            continue
        # Losses and train accuracies are per epoch; val_accs are every `val_every` epochs
        # (from epoch 0). Compare the entries from the first EXACT_EPOCHS epochs tightly.
        n_exact = max(1, len(v) * EXACT_EPOCHS // len(exp["train_losses"]))
        np.testing.assert_allclose(got[k][:n_exact], v[:n_exact], rtol=FIRST_EPOCH_RTOL, atol=1e-9, err_msg=k)
        if k.endswith("losses"):  # accuracies are discrete and can flip once runs diverge
            n = min(len(got[k]), len(v))
            np.testing.assert_allclose(got[k][:n], v[:n], rtol=TAIL_RTOL, err_msg=k)


@pytest.mark.parametrize("program", PROGRAMS)
def test_golden_training(program, exact):
    g = GOLDEN["programs"][program]
    mutant = raspbugs.get(g["job_id"])
    model = mutant.compile()
    assert_digest(digest(model.params), g["compiled"])

    test = data.load(program, "test")[:200]
    assert float(np.mean(model.evaluate(test[:30]))) == pytest.approx(g["evaluate_model_acc_30"])
    Xt, Yt = data.encode(test, model)
    assert [int(np.asarray(Xt).sum()), int(np.asarray(Yt).sum())] == g["encoded_test_digest"]
    assert float(model.accuracy(Xt, Yt)) == pytest.approx(g["fast_eval_acc_200"])

    X, Y = data.encode(data.load(program, "train")[:256], model)
    Xv, Yv = data.encode(data.load(program, "val")[:64], model)
    tr = Trainer(model, X, Y, cross_entropy_loss, n_epochs=4, batch_size=64, lr=1e-4,
                 X_val=Xv, Y_val=Yv, early_stopping_patience=2, early_stopping_min_delta=1e-4,
                 val_every=2, progress=False)
    res = tr.train()
    assert_history(res.history(), g["train"], exact)
    if exact:
        assert_digest(digest(tr.state.params), g["trained"])

    m2 = mutant.compile()
    m2.set_prng_key(3)
    m2.set_random_weights("xavier")
    assert_digest(digest(m2.params), g["xavier_seed3"])
    res2 = Trainer(m2, X, Y, cross_entropy_loss, n_epochs=2, batch_size=64, lr=1e-4,
                   X_val=Xv, Y_val=Yv, val_every=1, progress=False).train()
    assert_history(res2.history(), g["xavier_train"], exact)
