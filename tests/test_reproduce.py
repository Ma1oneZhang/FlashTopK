# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""The reproduction must report failed calls and retry them without retiming completed calls."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

REPRODUCE = Path(__file__).resolve().parents[1] / "reproduce"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(REPRODUCE))
    return tuple(importlib.import_module(name) for name in ("bench", "report", "records"))


def record(method="flashtopk", **changes):
    return dict(dict(ops="inputs/full_fp8/rand_128k/call0.pt", method=method, median_ms=1.0,
                     checked_rows=8192, wrong=0, short_rows=2047, short_wrong=0, digest="matching"), **changes)


def write_record(root, r, mode="w"):
    path = root / "h20_fp8" / "random" / f"{r['method']}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode) as f:
        f.write(json.dumps(r) + "\n")
    return path


@pytest.mark.parametrize("failure", [{"error": "check failed"}, {"wrong": 1}, {"short_wrong": 1},
                                     {"checked_rows": 0}, {"median_ms": float("nan")}])
def test_report_excludes_failed_calls_and_accepts_retry(modules, tmp_path, failure):
    _, report, _ = modules
    write_record(tmp_path, record(**failure))
    failures = []
    assert not report.load(tmp_path, failures)
    assert len(failures) == 1
    write_record(tmp_path, record(), mode="a")
    failures = []
    data = report.load(tmp_path, failures)
    assert not failures
    assert data[("h20", "fp8", "random")][("128k", "call0")]["flashtopk"]["wrong"] == 0


@pytest.mark.parametrize("change", [{"median_ms": 10.0}, {"digest": "different"}, {"wrong": 1}])
def test_report_returns_failure(modules, tmp_path, monkeypatch, change):
    _, report, _ = modules
    results, reference = tmp_path / "results", tmp_path / "reference"
    for root in (results, reference):
        write_record(root, record())
        write_record(root, record("dense", median_ms=2.0))
    write_record(results, record(**change))
    monkeypatch.setattr(report, "figures", lambda data, out: None)
    monkeypatch.setattr(sys, "argv", ["report.py", "--results", str(results), "--reference", str(reference)])
    assert report.main() == 1


@pytest.mark.parametrize("compare", [False, True])
def test_report_passes_matching_results(modules, tmp_path, monkeypatch, capsys, compare):
    _, report, _ = modules
    write_record(tmp_path, record())
    write_record(tmp_path, record("dense", median_ms=2.0))
    monkeypatch.setattr(report, "figures", lambda data, out: None)
    args = ["report.py", "--results", str(tmp_path)]
    if compare:
        args += ["--reference", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", args)
    assert report.main() == 0
    assert ("against the reference" in capsys.readouterr().out) == compare


def test_report_checks_digest_without_dense_baseline(modules, tmp_path, monkeypatch):
    _, report, _ = modules
    results, reference = tmp_path / "results", tmp_path / "reference"
    write_record(reference, record())
    write_record(results, record(digest="different"))
    monkeypatch.setattr(report, "figures", lambda data, out: None)
    monkeypatch.setattr(sys, "argv", ["report.py", "--results", str(results), "--reference", str(reference)])
    assert report.main() == 1


def test_benchmark_retries_only_failed_calls(modules, tmp_path, monkeypatch):
    bench, _, records = modules
    calls = []
    broken = {"bad.pt"}

    class Method:
        def __init__(self, args):
            pass

        def __call__(self, op):
            calls.append(op["path"])
            if op["path"] in broken:
                raise RuntimeError("injected kernel failure")
            return torch.zeros(2, 2, dtype=torch.int32)

        def stats(self, op):
            return {}

    monkeypatch.setitem(bench.methods.METHODS, "flashtopk", Method)
    monkeypatch.setattr(bench, "load_operands", lambda path: (
        dict(q_bf16=torch.empty(2, 32, 128), kv_bf16=torch.empty(4096, 128), path=path), {}))
    monkeypatch.setattr(bench, "cuda_times", lambda fn, warmup, iters: (fn(), [1.0])[1])
    monkeypatch.setattr(bench, "check", lambda op, selected: dict(checked_rows=2, wrong=0, short_rows=1, short_wrong=0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA H20")
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    output = tmp_path / "calls.jsonl"
    monkeypatch.setattr(sys, "argv", ["bench.py", "--method", "flashtopk", "--out", str(output), "bad.pt", "good.pt"])
    assert bench.main() == 1
    saved = records.latest(output)
    assert "error" in saved["bad.pt"] and records.completed(saved["good.pt"])
    broken.clear()
    calls.clear()
    assert bench.main() == 0
    assert calls and set(calls) == {"bad.pt"}
    assert all(records.completed(r) for r in records.latest(output).values())
    calls.clear()
    assert bench.main() == 0
    assert not calls


def test_runner_rejects_empty_sweep(tmp_path):
    script = tmp_path / "run.sh"
    script.write_text((REPRODUCE / "run.sh").read_text())
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\ncase "$2" in *get_device_capability*) echo 9.0 ;; *) echo h20 ;; esac\n')
    python.chmod(0o755)
    proc = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                          env=dict(os.environ, SETS="random", TYPES="fp8", LENGTHS="128k", METHODS="flashtopk"))
    assert proc.returncode == 1
    assert "no inputs found" in proc.stdout and "RUN_DONE" not in proc.stdout
