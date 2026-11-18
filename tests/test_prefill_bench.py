"""scripts/prefill_bench.py: the medians and the decode step it reports
(docs/prefill-kernel-check.md). vLLM itself runs only on the GPU instance."""
import prefill_bench


def test_summarize_takes_medians_and_derives_the_decode_step():
    rows = ([{"kind": "prefill", "length": 16, "ms": ms} for ms in (10, 30, 20)]
            + [{"kind": "prefill", "length": 2048, "ms": ms} for ms in (700, 720, 710)]
            + [{"kind": "decode", "length": 2048, "ms": ms} for ms in (710 + 64 * 35, 710 + 64 * 36, 710 + 64 * 34)])
    s = prefill_bench.summarize(rows)
    assert s["prefill_ms"] == {16: 20, 2048: 710}
    assert s["decode_step_ms"] == 35


def test_summarize_without_decode_rows():
    s = prefill_bench.summarize([{"kind": "prefill", "length": 64, "ms": 5}])
    assert s == {"prefill_ms": {64: 5}, "decode_step_ms": None}
