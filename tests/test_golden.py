"""Regression test: the refactored code reproduces the v1 (TMLR) code exactly.

`tests/golden/golden_v1.json` was recorded with the code at tag `tmlr-submission-1`
(src/ + experiments/): for one mutant per base program it stores digests of the
compiled weights, accuracies, a short training run with early stopping, and a Xavier
re-initialisation; and a tiny GP and BFS search for two mutants.

Any intentional behaviour change (a FINDINGS fix) that breaks this test must
re-record the golden file in the same commit and say so in docs/FINDINGS.md.
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


def assert_history(got: dict, exp: dict):
    for k, v in exp.items():
        np.testing.assert_allclose(got[k], v, rtol=1e-7, atol=1e-9, err_msg=k)


@pytest.mark.parametrize("program", PROGRAMS)
def test_golden_training(program):
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
    assert_history(res.history(), g["train"])
    assert_digest(digest(tr.state.params), g["trained"])

    m2 = mutant.compile()
    m2.set_prng_key(3)
    m2.set_random_weights("xavier")
    assert_digest(digest(m2.params), g["xavier_seed3"])
    res2 = Trainer(m2, X, Y, cross_entropy_loss, n_epochs=2, batch_size=64, lr=1e-4,
                   X_val=Xv, Y_val=Yv, val_every=1, progress=False).train()
    assert_history(res2.history(), g["xavier_train"])
