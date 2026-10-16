"""Smoke test: schema validation, separated scoring, provenance split."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval_bench.dataset.loader import load_cases, by_provenance
from eval_bench.decoders.base import IdentityDecoder
from eval_bench.scorers.separated import score_case, Prediction

def _judge(exp, pred):  # trivial semantic judge for the smoke test
    return exp.strip() in pred or pred.strip() in exp

def test_load_and_score():
    cases = load_cases("data")
    assert len(cases) >= 4
    prov = by_provenance(cases)
    assert set(prov) == {"seed_manual", "grounded", "adversarial_manual", "generated"}

    dec = IdentityDecoder()
    # Simulate a correct call on the grounded case.
    g = next(c for c in cases if c["id"] == "case_0101")
    raw = '{"tool":"search_report","args":{"report_type":"yuho","fiscal_year":2023,"metric":"employee_count","query_text":"従業員数"}}'
    pred = dec.decode(raw, {})
    s = score_case(g, pred, _judge)
    assert s.selection is True
    assert s.arg_exact is True
    assert s.arg_semantic is None  # tools.json 2.1.0: no semantic-scored fields
    assert s.e2e is True

    # Negative case: model correctly declines.
    neg = next(c for c in cases if c["negative"])
    s2 = score_case(neg, Prediction(tool=None, args={}), _judge)
    assert s2.selection is True
    assert s2.arg_exact is None  # N/A, not a forced failure

    # Selection miss => arg metrics are N/A, not False.
    s3 = score_case(g, Prediction(tool="search_master", args={}), _judge)
    assert s3.selection is False
    assert s3.arg_exact is None
    print("smoke ok")

def test_exact_match_normalizes_width_but_not_synonyms():
    """Free-text search terms are scored exact after NFKC + strip: a
    half-width/padded rendering of the right term passes, a synonym fails.
    This is what keeps the scoring mechanical (no judge model) without
    penalizing models for presentation-only variance."""
    case = next(c for c in load_cases("data") if c["id"] == "case_0003")
    base = {"entity": "item", "active_only": True}

    halfwidth = Prediction(tool="list_master_records", args={**base, "keyword": " ｽﾃﾝﾚｽ "})
    assert score_case(case, halfwidth, _judge).arg_exact is True

    synonym = Prediction(tool="list_master_records", args={**base, "keyword": "SUS"})
    assert score_case(case, synonym, _judge).arg_exact is False

if __name__ == "__main__":
    test_load_and_score()
    test_exact_match_normalizes_width_but_not_synonyms()
