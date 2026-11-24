"""Re-record entries of golden_v1.json with the *current* code.

Only for intentional behaviour changes (see CLAUDE.md): re-record exactly the entries the
change affects, in the same commit as the change and its docs/FINDINGS.md entry.

    python tests/golden/record.py training shuffle_dyck2   # one program's training entry
    python tests/golden/record.py search                   # the GP/BFS search entries
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from rasping import data, raspbugs
from rasping.losses import cross_entropy_loss
from rasping.train import Trainer

GOLDEN = Path(__file__).with_name("golden_v1.json")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.test_golden import digest  # noqa: E402


def floats(xs):
    return [float(x) for x in xs]


def record_training(program: str, job_id: str) -> dict:
    out = {"job_id": job_id}
    m = raspbugs.get(job_id)
    model = m.compile()
    out["compiled"] = digest(model.params)
    test = data.load(program, "test")[:200]
    out["evaluate_model_acc_30"] = float(np.mean(model.evaluate(test[:30])))
    Xt, Yt = data.encode(test, model)
    out["encoded_test_digest"] = [int(np.asarray(Xt).sum()), int(np.asarray(Yt).sum())]
    out["fast_eval_acc_200"] = float(model.accuracy(Xt, Yt))
    X, Y = data.encode(data.load(program, "train")[:256], model)
    Xv, Yv = data.encode(data.load(program, "val")[:64], model)
    tr = Trainer(model, X, Y, cross_entropy_loss, n_epochs=4, batch_size=64, lr=1e-4, X_val=Xv, Y_val=Yv,
                 early_stopping_patience=2, early_stopping_min_delta=1e-4, val_every=2, progress=False)
    out["train"] = tr.train().history()
    out["trained"] = digest(tr.state.params)
    m2 = m.compile()
    m2.set_prng_key(3)
    m2.set_random_weights("xavier")
    out["xavier_seed3"] = digest(m2.params)
    out["xavier_train"] = Trainer(m2, X, Y, cross_entropy_loss, n_epochs=2, batch_size=64, lr=1e-4,
                                  X_val=Xv, Y_val=Yv, val_every=1, progress=False).train().history()
    return out


def record_search(program: str, job_id: str) -> dict:
    import hashlib

    from rasping.baselines.bfs import run_bfs_for_mutant
    from rasping.baselines.gp import run_gp_for_mutant

    m = raspbugs.get(job_id)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for split, n in (("train", 60), ("val", 20), ("test", 40)):
            src = data.dataset_path(program, split)
            (tmp / src.name).write_text("\n".join(src.read_text().splitlines()[:n]) + "\n")
        gp = run_gp_for_mutant(m, data_dir=tmp, budget=12, population_size=4, tournament_k=2,
                               seed=7, log_every=100, eval_timeout=60)
        bfs = run_bfs_for_mutant(m, data_dir=tmp, max_depth=1, budget=12, log_every=100, eval_timeout=60)

    def strip(r):
        r = dict(r)
        r.pop("duration_s", None)
        r["best_source_sha"] = hashlib.sha256(r.pop("best_source").encode()).hexdigest()
        return json.loads(json.dumps(r, default=float))
    return {"job_id": job_id, "gp": strip(gp), "bfs": strip(bfs)}


def main():
    golden = json.loads(GOLDEN.read_text())
    kind, *names = sys.argv[1:]
    if kind == "training":
        for p in names or list(golden["programs"]):
            golden["programs"][p] = record_training(p, golden["programs"][p]["job_id"])
            print("re-recorded training", p)
    elif kind == "search":
        for p in names or list(golden["search"]):
            golden["search"][p] = record_search(p, golden["search"][p]["job_id"])
            print("re-recorded search", p)
    else:
        sys.exit(__doc__)
    GOLDEN.write_text(json.dumps(golden, indent=1))


if __name__ == "__main__":
    main()
