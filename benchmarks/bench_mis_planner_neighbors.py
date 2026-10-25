# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Paired original-vs-automatic MIS planning on identical inputs and kernels.

Run from this checkout: python benchmarks/bench_mis_planner_neighbors.py --output out.json
With --baseline-prefill, the baseline uses that unchanged public plan method;
otherwise only the baseline selector is bypassed. Both launches use the same
naturally compiled kernel. The automatic candidate receives no split override.
"""

import argparse
import ast
import contextlib
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch

import flashinfer
import flashinfer.prefill as prefill
from bench_multi_item_scoring import make_inputs


def cases():
    rows = [
        ("short256", 1, 4096, 256, 3, 1, "NHD", "mis", 64),
        ("short512", 1, 8192, 512, 7, 1, "NHD", "mis", 64),
        ("long_prefix256", 1, 8192, 256, 15, 4, "NHD", "mis", 64),
        ("batch4_256", 4, 2048, 256, 15, 1, "NHD", "mis", 64),
    ]
    rows += [
        (f"prefix{prefix}", 1, prefix, 512, 7, 1, "NHD", "mis", 64)
        for prefix in [4096, 6144, 7168, 8191, 8193, 9216, 10240, 12288]
    ]
    rows += [
        (f"items{items}", 1, 8192, items, 7, 1, "NHD", "mis", 64)
        for items in [384, 432, 448, 480, 496, 511, 513, 528, 544, 576, 640]
    ]
    rows += [
        (f"content{length}", 1, 8192, 4096 // (length + 1), length, 1, "NHD", "mis", 64)
        for length in [3, 15, 31]
    ]
    rows += [
        ("batch2", 2, 8192, 512, 7, 1, "NHD", "mis", 64),
        ("gqa2", 1, 8192, 512, 7, 2, "NHD", "mis", 64),
        ("hnd", 1, 8192, 512, 7, 1, "HND", "mis", 64),
        ("small_workspace", 1, 8192, 512, 7, 1, "NHD", "mis", 16),
        ("causal_control", 1, 8192, 512, 7, 1, "NHD", "causal", 64),
        ("none_control", 1, 8192, 512, 7, 1, "NHD", "none", 64),
    ]
    return rows


def dense_reference(q, kv, metadata, causal):
    qlen, heads, dim = q.shape
    klen = int(metadata["paged_kv_indptr"][-1]) * 16
    k = kv[:, 0].reshape(-1, heads, dim)[:klen].float()
    v = kv[:, 1].reshape(-1, heads, dim)[:klen].float()
    outputs, lses = [], []
    for start in range(0, qlen, 128):
        end = min(start + 128, qlen)
        scores = torch.einsum("qhd,khd->hqk", q[start:end].float(), k) / math.sqrt(dim)
        if causal:
            keys = torch.arange(klen, device=q.device)
            queries = torch.arange(start, end, device=q.device) + klen - qlen
            scores.masked_fill_(
                keys[None, None, :] > queries[None, :, None], -torch.inf
            )
        outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).to(q.dtype))
        lses.append(scores.logsumexp(-1).transpose(0, 1) / math.log(2))
    return torch.cat(outputs), torch.cat(lses)


def load_original_plan(path):
    """Bind the unchanged checkout's public Python method to a baseline wrapper.

    Its decorator and implementation are retained. Kernel dispatch comes from
    the current package, so this isolates the policy with identical kernels.
    """
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == "BatchPrefillWithPagedKVCacheWrapper"
    )
    plan = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "plan"
    )
    namespace = dict(vars(prefill))
    # Retain the baseline policy helpers too when comparing against a prior
    # planner revision; binding its plan to today's selector is not a v6 baseline.
    helpers = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_mis_default_split_chunks", "_select_mis_default_split_size")
    ]
    exec(
        compile(ast.Module(body=helpers, type_ignores=[]), str(path), "exec"), namespace
    )
    exec(
        compile(ast.Module(body=[plan], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["plan"]


class Run:
    def __init__(
        self, q, kv, metadata, layout, workspace_mib, original, original_plan=None
    ):
        self.q, self.kv, self.metadata = q, kv, metadata
        self.original = original
        self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(workspace_mib * 1024 * 1024, dtype=torch.uint8, device="cuda"),
            layout,
            backend="auto",
        )
        self.bypass = original and original_plan is None
        self.original_plan = original_plan if original else None
        self.plan()
        self.out, self.lse = self.wrapper.run(q, kv, return_lse=True)
        for _ in range(5):
            self.call()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for _ in range(20):
                self.call()
        self.graph.replay()
        torch.cuda.synchronize()

    def plan_context(self):
        return (
            patch.object(
                prefill,
                "_select_mis_default_split_size",
                lambda wrapper, **kw: kw["fixed_split_size"],
            )
            if self.bypass
            else contextlib.nullcontext()
        )

    def plan_call(self):
        if self.original_plan is not None:
            self.original_plan(self.wrapper, **self.metadata)
        else:
            self.wrapper.plan(**self.metadata)

    def plan(self):
        with self.plan_context():
            self.plan_call()

    def call(self):
        self.wrapper.run(self.q, self.kv, out=self.out, lse=self.lse, return_lse=True)

    def check(self, reference, lse):
        torch.testing.assert_close(self.out.cpu(), reference, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(self.lse.cpu(), lse, rtol=1e-2, atol=1e-2)


def measure(a, b, samples, wall=False):
    values = {"original": [], "automatic": []}
    for _ in range(20):
        a.graph.replay()
        b.graph.replay()
    torch.cuda.synchronize()
    for repeat in range(samples):
        order = [("original", a), ("automatic", b)]
        for name, run in order if repeat % 2 == 0 else order[::-1]:
            if wall:
                # Set up the baseline ablation outside the timed interval.
                with run.plan_context():
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    run.plan_call()
                    run.call()
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter_ns() - start) / 1000
            else:
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                run.graph.replay()
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end) * 1000 / 20
            values[name].append(elapsed)
    a_us, b_us = (statistics.median(values[name]) for name in values)
    return dict(
        original_us=a_us, automatic_us=b_us, speedup=a_us / b_us, samples_us=values
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--samples", type=int, default=35)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--wall", action="store_true")
    parser.add_argument("--baseline-prefill", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    source = Path(flashinfer.__file__).resolve().parents[1]
    original_plan = (
        load_original_plan(args.baseline_prefill) if args.baseline_prefill else None
    )
    info = dict(
        gpu=torch.cuda.get_device_name(),
        compute_capability=torch.cuda.get_device_capability(),
        sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
        source=str(source),
        prefill_sha256=hashlib.sha256(
            (source / "flashinfer/prefill.py").read_bytes()
        ).hexdigest(),
        header_sha256=hashlib.sha256(
            (source / "include/flashinfer/attention/prefill.cuh").read_bytes()
        ).hexdigest(),
        method="alternating paired CUDA events, warm-L2 graphs of 20 calls",
        baseline=(
            dict(
                method="unchanged public plan AST, identical current kernel dispatch",
                path=str(args.baseline_prefill),
                sha256=hashlib.sha256(args.baseline_prefill.read_bytes()).hexdigest(),
            )
            if args.baseline_prefill
            else dict(method="new selector bypass ablation")
        ),
        rows=[],
    )
    for dtype in (torch.float16, torch.bfloat16):
        for case in cases():
            name, batch, prefix, items, length, group, layout, mask, mib = case
            if args.case and name not in args.case:
                continue
            q, kv, metadata, reference, lse = make_inputs(
                batch, prefix, items, length, group, dtype
            )
            if mask != "mis":
                for key in (
                    "prefix_len_ptr",
                    "token_pos_in_items_ptr",
                    "token_pos_in_items_len",
                    "max_item_len_ptr",
                ):
                    metadata.pop(key)
                metadata["causal"] = mask == "causal"
                reference, lse = dense_reference(q, kv, metadata, mask == "causal")
            if layout == "HND":
                kv = kv.permute(0, 1, 3, 2, 4).contiguous()
            # Freeze independent truth before opaque native launches and graph
            # setup; device reference buffers are never trusted after execution.
            reference, lse = reference.cpu(), lse.cpu()
            assert torch.isfinite(reference).all() and torch.isfinite(lse).all()
            a, b = (
                Run(q, kv, metadata, layout, mib, original, original_plan)
                for original in (True, False)
            )
            a.check(reference, lse)
            b.check(reference, lse)
            plans_equal = list(a.wrapper._plan_info) == list(b.wrapper._plan_info)
            if plans_equal:
                torch.testing.assert_close(a.out, b.out, rtol=0, atol=0)
                torch.testing.assert_close(a.lse, b.lse, rtol=0, atol=0)
            for repeat in range(args.repeats):
                row = dict(
                    case=name,
                    shape=case,
                    dtype=str(dtype),
                    repeat=repeat,
                    original_plan=list(a.wrapper._plan_info),
                    automatic_plan=list(b.wrapper._plan_info),
                    independent_reference_passed=True,
                    reference_snapshot="CPU copy before native execution",
                    reference_output_sha256=hashlib.sha256(
                        reference.contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                    reference_lse_sha256=hashlib.sha256(
                        lse.contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                    plans_equal=plans_equal,
                    unchanged_plan_bitwise_equal=plans_equal,
                    max_output_abs_diff=float(
                        (a.out.float() - b.out.float()).abs().max()
                    ),
                    max_lse_abs_diff=float((a.lse - b.lse).abs().max()),
                    gpu=measure(a, b, args.samples),
                )
                if args.wall:
                    row["wall"] = measure(a, b, args.samples, wall=True)
                # Validate executed graphs and completed plan+run calls too.
                a.check(reference, lse)
                b.check(reference, lse)
                info["rows"].append(row)
                mapped = sorted(
                    {
                        line.split()[-1]
                        for line in Path("/proc/self/maps").read_text().splitlines()
                        if "batch_prefill_with_kv_cache" in line and ".so" in line
                    }
                )
                info["mapped_libraries"] = [
                    dict(
                        path=p, sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()
                    )
                    for p in mapped
                ]
                args.output.write_text(json.dumps(info, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in row.items()
                            if k
                            not in (
                                "original_plan",
                                "automatic_plan",
                                "gpu",
                                "wall",
                                "cold",
                            )
                        },
                        default=str,
                    ),
                    row["gpu"]["speedup"],
                    flush=True,
                )
            del a, b, q, kv, metadata, reference, lse
            gc.collect()
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
