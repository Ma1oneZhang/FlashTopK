# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Completion and retry rules shared by the benchmark and its report."""
import json
import math


def completed(record):
    """A timed call whose correctness checks finished without incorrect rows."""
    ms = record.get("median_ms")
    return ("error" not in record and isinstance(ms, (int, float)) and math.isfinite(ms) and ms > 0
            and record.get("checked_rows", 0) > 0 and record.get("short_rows", 0) > 0
            and record.get("wrong") == 0 and record.get("short_wrong") == 0)


def latest(path):
    """The last attempt for each input; a successful retry supersedes an earlier error."""
    records = {}
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            records[record["ops"]] = record
    return records
