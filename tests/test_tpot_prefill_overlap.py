import json

import tpot_prefill_overlap as tpo


def _row(sent, ttft_ms, done, phase="measured", tpot=50.0, tokens=40, status="ok"):
    return {"sent_at_s": sent, "ttft_ms": ttft_ms, "done_at_s": done, "phase": phase,
            "tpot_ms": tpot, "output_tokens": tokens, "status": status}


def test_counts_other_prefills_inside_the_generation():
    rows = [
        _row(0.0, 1000, 5.0, tpot=40.0),            # generates 1.0..5.0
        _row(2.0, 1000, 8.0, tpot=60.0),            # prefill 2.0..3.0 overlaps the first; generates 3.0..8.0
        _row(6.0, 500, 9.0, phase="cooldown"),      # prefill 6.0..6.5 overlaps the second only
    ]
    assert tpo.overlaps(rows) == [(1, 40.0), (1, 60.0)]


def test_prefill_ending_exactly_at_the_generation_start_does_not_overlap():
    rows = [_row(0.0, 1000, 5.0), _row(0.5, 500, 4.0)]   # second prefill 0.5..1.0, first generation from 1.0
    assert tpo.overlaps(rows)[0][0] == 0


def test_only_measured_requests_with_enough_tokens_are_grouped():
    rows = [_row(0.0, 100, 1.0, phase="warmup"), _row(0.0, 100, 1.0, tokens=4), _row(0.0, 100, 1.0)]
    assert len(tpo.overlaps(rows)) == 1


def test_summary_counts_requests_over_the_limit():
    pairs = [(0, 40.0), (3, 90.0), (3, 120.0), (3, 130.0)]
    assert tpo.summarize(pairs, 100) == [
        {"overlap": 0, "requests": 1, "tpot_median_ms": 40.0, "over_limit": 0},
        {"overlap": 3, "requests": 3, "tpot_median_ms": 120.0, "over_limit": 2},
    ]


def test_main_skips_failed_requests_and_prints_each_cell(tmp_path, capsys):
    cell = tmp_path / "lat-x-c4"
    cell.mkdir()
    rows = [_row(0.0, 1000, 5.0), _row(2.0, 1000, 8.0, status="error")]
    (cell / "run.raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert tpo.main([str(tmp_path), "lat-x-c4"]) == 0
    out = capsys.readouterr().out
    assert "measured=1" in out and "overlap=0: requests=1" in out
