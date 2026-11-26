"""Cluster submission: large launches are split into SLURM arrays of at most MAX_ARRAY_SIZE."""

import contextlib
import sys
import types

from rasping import cluster


def test_submit_splits_into_arrays(monkeypatch, tmp_path):
    batches = []

    class FakeExecutor:
        def __init__(self, folder):
            pass

        def update_parameters(self, **kw):
            self.params = kw

        @contextlib.contextmanager
        def batch(self):
            batches.append(0)
            yield

        def submit(self, fn, cmd):
            batches[-1] += 1
            return cmd

    monkeypatch.setitem(sys.modules, "submitit", types.SimpleNamespace(AutoExecutor=FakeExecutor))
    monkeypatch.setattr(cluster, "LOG_DIR", tmp_path)
    jobs = cluster.submit([["echo", str(i)] for i in range(2 * cluster.MAX_ARRAY_SIZE + 5)], "t",
                          container=False)
    assert batches == [cluster.MAX_ARRAY_SIZE, cluster.MAX_ARRAY_SIZE, 5]
    assert len(jobs) == 2 * cluster.MAX_ARRAY_SIZE + 5 and jobs[0] == ["echo", "0"]
