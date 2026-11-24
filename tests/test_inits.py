"""Starting points for repair (`rasping.repair.INITS`) and the model methods behind them."""

import jax
import numpy as np
import pytest

from rasping import raspbugs
from rasping.model import MLP_EXACTNESS, SOFTMAX_COLDNESS
from rasping.repair import check_init, repair

JOB = "2a722d5d5449441fad93f98c73c633a2"  # hist, order 1


def leaves(params):
    return {f"{k}/{s}": np.asarray(params[k][s]) for k in params for s in params[k]}


@pytest.fixture(scope="module")
def compiled():
    return raspbugs.get(JOB).compile()


def test_default_sharpness_is_exact(compiled):
    soft = raspbugs.get(JOB).compile(softmax_coldness=SOFTMAX_COLDNESS, mlp_exactness=MLP_EXACTNESS)
    a, b = leaves(compiled.params), leaves(soft.params)
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def test_softer_compilation_keeps_shapes_and_shrinks_weights(compiled):
    soft = raspbugs.get(JOB).compile(softmax_coldness=30, mlp_exactness=36)
    a, b = leaves(compiled.params), leaves(soft.params)
    assert {k: v.shape for k, v in a.items()} == {k: v.shape for k, v in b.items()}
    assert max(np.abs(v).max() for v in b.values()) < max(np.abs(v).max() for v in a.values())


def test_shuffled_permutes_within_tensors():
    m = raspbugs.get(JOB).compile()
    before = leaves(m.params)
    m.set_prng_key(0)
    m.set_shuffled_weights()
    after = leaves(m.params)
    for k in before:
        assert np.array_equal(np.sort(before[k], axis=None), np.sort(after[k], axis=None)), k
    assert any(not np.array_equal(before[k], after[k]) for k in before)


def test_rescaled_keeps_structure_at_xavier_scale():
    m = raspbugs.get(JOB).compile()
    before = leaves(m.params)
    m.set_rescaled_weights()
    for k, v in leaves(m.params).items():
        if v.ndim < 2:
            assert not v.any(), k  # biases zeroed, as in "xavier"
        elif before[k].std() > 0:
            fan_in, fan_out = int(np.prod(v.shape[:-1])), v.shape[-1]
            assert np.isclose(v.std(), np.sqrt(2 / (fan_in + fan_out))), k
            assert np.array_equal(np.sign(v), np.sign(before[k])), k  # same structure


@pytest.mark.parametrize("ok", ["compiled", "xavier", "shuffled", "rescaled", "soft:30", "soft:2.5",
                                "from:" + "a" * 32])
def test_check_init_accepts(ok):
    assert check_init(ok) == ok


@pytest.mark.parametrize("bad", ["bogus", "soft:", "soft:x", "soft:0", "from:123", "xavier:1"])
def test_check_init_rejects(bad):
    with pytest.raises(ValueError):
        check_init(bad)


def test_from_rejects_itself():
    with pytest.raises(ValueError, match="another mutant"):
        repair(JOB, init=f"from:{JOB}", n_train=64, progress=False)


def test_from_rejects_different_shapes():
    t = raspbugs.architectures()
    sig = t.loc[t.job_id == JOB, "signature"].item()
    other = t[(t.program == "hist") & (t.signature != sig)].job_id.iloc[0]
    with pytest.raises(AssertionError, match="mismatch"):
        repair(JOB, init=f"from:{other}", n_train=64, progress=False)


def test_repair_records_start_metrics():
    from dataclasses import replace

    from rasping.config import TrainConfig

    cfg = replace(TrainConfig(), n_epochs=1)
    rec = repair(JOB, config=cfg, init="soft:100", n_train=64, n_val=32, progress=False)
    assert rec.start_agreement == 1.0
    assert rec.start_test_accuracy == rec.initial_test_accuracy
    rec = repair(JOB, config=cfg, init="xavier", n_train=64, n_val=32, progress=False)
    assert rec.start_agreement < 0.5
