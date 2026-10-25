# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""MIS planner boundaries and public-plan regressions."""

import functools
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import flashinfer
import flashinfer.prefill as prefill


@functools.cache
def _reference_case_maker():
    # Support pytest's importlib mode and running this file on its own.
    spec = importlib.util.spec_from_file_location(
        "_mis_planner_reference",
        Path(__file__).with_name("test_multi_item_scoring_tile_skip.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._make_case


def _policy_inputs(monkeypatch, dtype=torch.float16):
    monkeypatch.setattr(prefill, "get_compute_capability", lambda _: (8, 0))
    monkeypatch.setattr(prefill, "get_device_sm_count", lambda _: 108)
    wrapper = SimpleNamespace(
        is_cuda_graph_enabled=False,
        _jit_module=None,
        _kv_layout="NHD",
        _float_workspace_buffer=torch.empty(64 * 1024 * 1024, dtype=torch.uint8),
        _int_workspace_buffer=torch.empty(8 * 1024 * 1024, dtype=torch.uint8),
        device=torch.device("cpu"),
    )
    options = dict(
        fixed_split_size=-1,
        disable_split_kv=False,
        prefix_len_ptr=torch.tensor([8192], dtype=torch.int64),
        qo_indptr_host=torch.tensor([0, 4096], dtype=torch.int32),
        kv_lens_arr_host=torch.tensor([12288], dtype=torch.int32),
        index_dtype=torch.int32,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=16,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
        pos_encoding_mode="NONE",
        use_fp16_qk_reduction=False,
        window_left=-1,
        logits_soft_cap=0.0,
    )
    return wrapper, options


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mis_planner_policy_candidate(monkeypatch, dtype):
    wrapper, options = _policy_inputs(monkeypatch, dtype)
    assert prefill._select_mis_default_split_size(wrapper, **options) == 128
    holder = object.__new__(flashinfer.BatchPrefillWithPagedKVCacheWrapper)
    holder.device = torch.device("cpu")
    assert holder._mis_default_split_supported


@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mis_planner_fp8_kv_keeps_generic_policy(monkeypatch, q_dtype, kv_dtype):
    wrapper, options = _policy_inputs(monkeypatch, q_dtype)

    def unexpected():
        raise AssertionError("ineligible FP8 KV request read MIS device metadata")

    options.update(
        kv_data_type=kv_dtype, prefix_len_ptr=SimpleNamespace(item=unexpected)
    )
    assert prefill._select_mis_default_split_size(wrapper, **options) == -1


@pytest.mark.parametrize(
    "change",
    [
        {"prefix_len_ptr": None},
        {"fixed_split_size": 256},
        {"disable_split_kv": True},
        {"num_qo_heads": 16},
        {"num_kv_heads": 8},
        {"head_dim_qk": 64},
        {"head_dim_vo": 256},
        {"page_size": 32},
        {"index_dtype": torch.int64},
        {"q_data_type": torch.float32},
        {"kv_data_type": torch.bfloat16},
        {"o_data_type": torch.bfloat16},
        {"pos_encoding_mode": "ALIBI"},
        {"use_fp16_qk_reduction": True},
        {"window_left": 0},
        {"logits_soft_cap": 8.0},
        {"qo_indptr_host": torch.tensor([0, 4096, 8192], dtype=torch.int32)},
        {"kv_lens_arr_host": torch.tensor([12288, 12288])},
        {"qo_indptr_host": torch.tensor([0, 4096], dtype=torch.int64)},
        {"qo_indptr_host": torch.tensor([0, 1024], dtype=torch.int32)},
        {"qo_indptr_host": torch.tensor([0, 6912], dtype=torch.int32)},
        {"kv_lens_arr_host": torch.tensor([8192])},
        {"kv_lens_arr_host": torch.tensor([12287])},
        {"prefix_len_ptr": torch.tensor([8191])},
        {"kv_lens_arr_host": torch.tensor([16384])},
    ],
)
def test_mis_planner_policy_fallback(monkeypatch, change):
    wrapper, options = _policy_inputs(monkeypatch)
    options.update(change)
    assert (
        prefill._select_mis_default_split_size(wrapper, **options)
        == options["fixed_split_size"]
    )


@pytest.mark.parametrize(
    "change",
    [
        {"is_cuda_graph_enabled": True},
        {"_jit_module": object()},
        {"_kv_layout": "HND"},
    ],
)
def test_mis_planner_early_fallback_has_no_device_query(monkeypatch, change):
    wrapper, options = _policy_inputs(monkeypatch)
    for name, value in change.items():
        setattr(wrapper, name, value)

    def unexpected(_):
        raise AssertionError("fallback queried the device")

    monkeypatch.setattr(prefill, "get_compute_capability", unexpected)
    assert prefill._select_mis_default_split_size(wrapper, **options) == -1


@pytest.mark.parametrize("cc,sm", [((9, 0), 132), ((11, 0), 20), ((8, 0), 80)])
def test_mis_planner_other_devices_keep_default(monkeypatch, cc, sm):
    wrapper, options = _policy_inputs(monkeypatch)
    monkeypatch.setattr(prefill, "get_compute_capability", lambda _: cc)
    monkeypatch.setattr(prefill, "get_device_sm_count", lambda _: sm)
    assert prefill._select_mis_default_split_size(wrapper, **options) == -1
    queries = []
    monkeypatch.setattr(
        prefill, "get_compute_capability", lambda _: queries.append(cc) or cc
    )
    holder = object.__new__(flashinfer.BatchPrefillWithPagedKVCacheWrapper)
    holder.device = torch.device("cpu")
    assert not holder._mis_default_split_supported
    assert not holder._mis_default_split_supported
    assert queries == [cc]


@pytest.mark.parametrize("q_len,int_bytes", [(4095, 18704), (4096, 18708)])
def test_mis_planner_checks_prefix_and_workspace(monkeypatch, q_len, int_bytes):
    wrapper, options = _policy_inputs(monkeypatch)
    options["qo_indptr_host"][1] = q_len
    options["kv_lens_arr_host"][0] = 8192 + q_len
    options["prefix_len_ptr"] = torch.tensor([4096])
    assert prefill._select_mis_default_split_size(wrapper, **options) == -1
    options["prefix_len_ptr"].fill_(8192)
    needed = 4096 * 6 * 4 * (128 * 4 + 4)
    wrapper._float_workspace_buffer = torch.empty(needed - 1, dtype=torch.uint8)
    assert prefill._select_mis_default_split_size(wrapper, **options) == -1
    # Capacity is in bytes even when the caller supplies a typed workspace.
    wrapper._float_workspace_buffer = torch.empty(needed // 4, dtype=torch.float32)
    assert prefill._select_mis_default_split_size(wrapper, **options) == 128
    wrapper._int_workspace_buffer = torch.empty(int_bytes - 1, dtype=torch.uint8)
    assert prefill._select_mis_default_split_size(wrapper, **options) == -1
    wrapper._int_workspace_buffer = torch.empty(int_bytes, dtype=torch.uint8)
    assert prefill._select_mis_default_split_size(wrapper, **options) == 128
    wrapper._int_workspace_buffer = torch.empty(1024, dtype=torch.uint8)
    assert (
        prefill._select_mis_default_split_size(
            wrapper, **options, check_workspace=False
        )
        == 128
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "prefix,items,query_tokens",
    [
        (8192, 448, None),
        (8192, 512, None),
        (8192, 511, None),
        (8192, 512, 3457),
        (8192, 512, 4095),
    ],
)
def test_mis_default_planner_public_candidate(dtype, prefix, items, query_tokens):
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (8, 0)
        or torch.cuda.get_device_properties(0).multi_processor_count != 108
    ):
        pytest.skip("policy performance region requires a 108-SM SM80 device")
    make_case = _reference_case_maker()

    q, kv, metadata, reference, ref_lse = make_case(
        [(prefix, [7] * items)], dtype=dtype, query_tokens=query_tokens
    )
    reference, ref_lse = reference.cpu(), ref_lse.cpu()
    wrappers = []
    for options in ({}, {"fixed_split_size": 128}):
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
            "NHD",
            backend="fa2",
        )
        wrapper.plan(**metadata, **options)
        out, lse = wrapper.run(q, kv, return_lse=True)
        torch.testing.assert_close(out.cpu(), reference, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse.cpu(), ref_lse, rtol=1e-2, atol=1e-2)
        wrappers.append(wrapper)
    assert wrappers[0]._plan_info == wrappers[1]._plan_info


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "options",
    [
        {"fixed_split_size": 256},
        {"disable_split_kv": True},
        {"seq_lens": torch.tensor([12288], dtype=torch.int32)},
    ],
)
def test_mis_public_planner_preserves_explicit_options(dtype, options, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    make_case = _reference_case_maker()

    q, kv, metadata, reference, ref_lse = make_case([(8192, [7] * 512)], dtype=dtype)
    reference, ref_lse = reference.cpu(), ref_lse.cpu()
    plans = []
    for original in (False, True):
        with monkeypatch.context() as patch:
            if original:
                patch.setattr(
                    prefill,
                    "_select_mis_default_split_size",
                    lambda wrapper, **kw: kw["fixed_split_size"],
                )
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
                "NHD",
                backend="fa2",
            )
            wrapper.plan(**metadata, **options)
        out, lse = wrapper.run(q, kv, return_lse=True)
        torch.testing.assert_close(out.cpu(), reference, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse.cpu(), ref_lse, rtol=1e-2, atol=1e-2)
        plans.append(wrapper._plan_info)
    assert plans[0] == plans[1]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mis_public_workspace_query_and_exact_capacity(dtype):
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (8, 0)
        or torch.cuda.get_device_properties(0).multi_processor_count != 108
    ):
        pytest.skip("requires a 108-SM SM80 device")
    make_case = _reference_case_maker()

    q, kv, metadata, reference, ref_lse = make_case([(8192, [7] * 512)], dtype=dtype)
    reference, ref_lse = reference.cpu(), ref_lse.cpu()
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa2",
    )
    # The query selects the preferred plan independently of current capacity,
    # and consumes its own prefix argument before any plan has been called.
    float_bytes, int_bytes = wrapper.workspace_size(**metadata)
    assert float_bytes == 4096 * 6 * 4 * (128 * 4 + 4)
    wrapper.reset_workspace_buffer(
        torch.empty(float_bytes, dtype=torch.uint8, device="cuda"),
        torch.empty(int_bytes, dtype=torch.uint8, device="cuda"),
    )
    wrapper.plan(**metadata)
    automatic = wrapper._plan_info
    out, lse = wrapper.run(q, kv, return_lse=True)
    torch.testing.assert_close(out.cpu(), reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse.cpu(), ref_lse, rtol=1e-2, atol=1e-2)
    wrapper.plan(**metadata, fixed_split_size=128)
    assert automatic == wrapper._plan_info
    # Reusing a previously MIS wrapper for a non-MIS workspace query must
    # use the new arguments rather than the prefix stored by the last plan.
    dense = {
        name: value
        for name, value in metadata.items()
        if name
        not in (
            "prefix_len_ptr",
            "token_pos_in_items_ptr",
            "token_pos_in_items_len",
            "max_item_len_ptr",
        )
    }
    assert wrapper.workspace_size(**dense)[0] == 0
    # An explicit sequence-length override retains the generic scheduler:
    # its value need not describe the number of reserved physical pages.
    overridden = dict(metadata, seq_lens=torch.tensor([12288], dtype=torch.int32))
    assert wrapper.workspace_size(**overridden)[0] == 0
    wrapper.plan(**overridden)
    assert not wrapper._plan_info[14]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("location", ["cuda", "cpu", "mixed"])
def test_mis_host_metadata_fresh_values_and_stream(dtype, location):
    """Re-plan from mutated metadata on a non-default stream, including fallback."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa2",
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    reusable = {}
    with torch.cuda.stream(stream):
        for prefix, query_tokens in ((8192, None), (4096, 4095)):
            q, kv, metadata, ref, ref_lse = _reference_case_maker()(
                [(prefix, [7] * 512)], dtype=dtype, query_tokens=query_tokens
            )
            ref, ref_lse = ref.cpu(), ref_lse.cpu()
            for key in ("qo_indptr", "paged_kv_indptr", "paged_kv_last_page_len"):
                value = metadata[key]
                if location == "cpu" or (location == "mixed" and key == "qo_indptr"):
                    value = value.cpu()
                if key not in reusable:
                    reusable[key] = value.clone()
                else:
                    reusable[key].copy_(value)
                metadata[key] = reusable[key]
            # Explicit CPU staging must also survive a CUDA default device.
            with torch.device("cuda"):
                wrapper.plan(**metadata)
            assert wrapper._max_q_len == q.shape[0]
            assert wrapper._max_kv_len == prefix + 4096
            torch.testing.assert_close(
                wrapper._kv_lens_buffer[:1].cpu(),
                torch.tensor([prefix + 4096], dtype=torch.int32),
            )
            out, lse = wrapper.run(q, kv, return_lse=True)
            torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(lse.cpu(), ref_lse, rtol=1e-2, atol=1e-2)
    torch.cuda.current_stream().wait_stream(stream)


def test_mis_host_metadata_cpu_does_not_query_cuda(monkeypatch):
    holder = SimpleNamespace(device=torch.device("cpu"))
    qo = torch.tensor([0, 7], dtype=torch.int32)
    kv = torch.tensor([0, 2], dtype=torch.int32)
    tail = torch.tensor([3], dtype=torch.int32)

    def unexpected(*args, **kwargs):
        raise AssertionError("CPU metadata queried a CUDA stream")

    monkeypatch.setattr(torch.cuda, "current_stream", unexpected)
    host = prefill._single_mis_metadata_to_host(holder, qo, kv, tail)
    assert all(a is b for a, b in zip(host, (qo, kv, tail), strict=True))
    assert not hasattr(holder, "_mis_host_metadata")
