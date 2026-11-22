import json
from datetime import datetime, timezone

import fp8_tensor_check_summary as s


def test_rise_per_overlap_is_the_median_difference_over_three():
    pairs = [(0, 40.0), (0, 42.0), (0, 44.0), (3, 70.0), (3, 72.0), (1, 99.0)]
    assert s.rise_per_overlap(pairs) == (71.0 - 42.0) / 3


def test_rise_per_overlap_needs_both_groups():
    assert s.rise_per_overlap([(0, 40.0), (2, 60.0)]) is None


def test_prompt_differences_match_requests_by_index():
    rows = [{"index": 0, "prompt_tokens": 2000}, {"index": 1, "prompt_tokens": 2105}, {"index": 2, "prompt_tokens": None}]
    reference = [{"index": 0, "prompt_tokens": 2000}, {"index": 1, "prompt_tokens": 2005}, {"index": 2, "prompt_tokens": 1}]
    assert s.prompt_differences(rows, reference) == (1, 100)


def test_find_key_reaches_a_nested_config():
    info = {"vllm_config": {"model_config": {"quantization": "modelopt"}}, "other": [{"quantization": "x"}]}
    assert s.find_key(info, "quantization") == "modelopt"
    assert s.find_key({"model_config": {"quantization": None}}, "quantization") is None


def test_mean_clock_takes_the_rows_inside_the_cell(tmp_path):
    gpu = tmp_path / "gpu.csv"
    gpu.write_text("timestamp, memory.used [MiB], clocks.current.sm [MHz]\n"
                   "2026/11/22 10:00:00.000, 1 MiB, 2040 MHz\n"
                   "2026/11/22 10:00:01.000, 1 MiB, 1000 MHz\n"
                   "2026/11/22 10:00:02.000, 1 MiB, 1200 MHz\n"
                   "2026/11/22 10:00:03.000, 1 MiB, 2040 MHz\n")
    start = datetime(2026, 11, 22, 10, 0, 1, tzinfo=timezone.utc)
    end = datetime(2026, 11, 22, 10, 0, 2, tzinfo=timezone.utc)
    assert s.mean_clock(gpu, start, end) == 1100.0
    assert s.mean_clock(tmp_path / "absent.csv", start, end) is None


def test_server_facts_read_the_kernel_the_warning_and_the_quantization(tmp_path):
    (tmp_path / "server.log").write_text("Selected CutlassFP8ScaledMMLinearKernel for ModelOptFp8LinearMethod\n"
                                         "In FP8 linear, the static input scale differs across parallel layers\n")
    (tmp_path / "engine-server_info-before.json").write_text(json.dumps({"model_config": {"quantization": "modelopt"}}))
    assert s.server_facts(tmp_path) == {"kernels": ["CutlassFP8ScaledMMLinearKernel"], "fused_scale_warning": True,
                                        "quantization": "modelopt"}
