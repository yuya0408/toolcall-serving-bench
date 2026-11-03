"""The CI regression gate (docs/adr/0010-ci-golden-gate.md): replay the saved
8B answers through the current harness. Any mismatch fails. After an intended
G2/G4 change, run scripts/update_golden.py and list the changed rows in the PR;
a G1/G3 mismatch contradicts the run record, so fix the code instead."""
import random

import pytest

from eval_bench import golden

ENGINES = list(golden.CELLS)


@pytest.fixture(scope="module")
def inputs():
    return golden.harness_inputs()


@pytest.fixture(scope="module", params=ENGINES)
def engine_golden(request):
    cell, answers, expected = golden.load_golden(request.param)
    return request.param, cell, answers, expected


def _fail_message(mismatches, what):
    case_id, trial, want, got = mismatches[0]
    return f"{len(mismatches)} rows: {what} changed, first {case_id} trial {trial}: {want!r} -> {got!r}"


def test_golden_rows_line_up(engine_golden):
    _, _, answers, expected = engine_golden
    assert len(answers) == len(expected) == 1000
    assert [(a["case_id"], a["trial"]) for a in answers] == [(e["case_id"], e["trial"]) for e in expected]


def test_g1_tool_order(engine_golden, inputs):
    _, cell, answers, _ = engine_golden
    bad = golden.check_g1_tool_order(cell, answers, *inputs)
    assert not bad, _fail_message(bad, "tool order")


def test_g2_request(engine_golden, inputs):
    engine, cell, answers, expected = engine_golden
    bad = golden.check_g2_request(engine, cell, answers, expected, *inputs)
    assert not bad, _fail_message(bad, "request")


def test_g3_decoding(engine_golden):
    engine, _, answers, _ = engine_golden
    bad = golden.check_g3_decoding(engine, answers)
    assert not bad, _fail_message(bad, "decoded call")


def test_g4_scoring(engine_golden, inputs):
    _, _, answers, expected = engine_golden
    bad = golden.check_g4_scoring(answers, expected, inputs[0])
    assert not bad, _fail_message(bad, "score")


def test_selection_mean_matches_the_run_manifest(engine_golden):
    _, cell, _, expected = engine_golden
    assert round(golden.selection_mean(expected), 3) == cell["selection_mean"]


# --- each check catches a deliberately broken harness -------------------------

@pytest.fixture(scope="module")
def vllm_golden():
    return golden.load_golden("vllm")


def test_g1_catches_a_changed_shuffle(vllm_golden, inputs):
    cell, answers, _ = vllm_golden

    def broken(case, tools, cell, trial):
        order = golden.tool_order(case, tools, cell, trial)
        random.Random(f"other:{case['id']}:{trial}").shuffle(order)
        return order
    assert golden.check_g1_tool_order(cell, answers, *inputs, order_fn=broken)


def test_g2_catches_a_dropped_parameter(vllm_golden, inputs):
    cell, answers, expected = vllm_golden

    def broken(*args):
        req = golden.request(*args)
        req.pop("min_p")
        return req
    assert golden.check_g2_request("vllm", cell, answers, expected, *inputs, request_fn=broken)


def test_g3_catches_a_changed_argument_type(vllm_golden):
    _, answers, _ = vllm_golden
    real = golden.decoder_for("vllm")

    class Broken:
        def decode(self, raw, schema):
            p = real.decode(raw, schema)
            p.args = {k: str(v) for k, v in p.args.items()}
            return p
    assert golden.check_g3_decoding("vllm", answers, decoder=Broken())


def test_g4_catches_a_changed_comparison(vllm_golden, inputs):
    _, answers, expected = vllm_golden

    def broken(case, decoded):
        s = golden.score(case, decoded)
        s["arg_exact"] = s["selection"]  # args no longer compared
        return s
    assert golden.check_g4_scoring(answers, expected, inputs[0], score_fn=broken)
