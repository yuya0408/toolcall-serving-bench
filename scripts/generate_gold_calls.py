"""Sample the gold calls for the case-set expansion (docs/adr/0004-expand-case-set.md).

    python scripts/generate_gold_calls.py            # writes data_build/expand/gold_calls.jsonl

Step 1 of 4: the answer comes first, from the tool catalog alone; an LLM
writes the query from it afterwards (step 2), so no model ever decides a
gold value. Every positive gold call here is distinct from every other and
from the existing cases -- a paraphrase of the same call would not be an
independent case.

Rules:
  - allocation: every tool gets at least 4 positive cases counting the
    existing ones; the rest go to the confusable families (search / aggregate
    / trend across modules). Negatives are written in step 2 (no gold call).
  - required arguments always appear; optional ones with probability 1/2.
  - enum values are drawn uniformly, except sentinels that mean "no
    constraint" ("none", "any"): stating them in a query is unnatural, so the
    argument is left out instead.
  - cross-argument constraints: period_a != period_b, year_from < year_to,
    and the business-sense rules in plausible() (which fields a master
    entity has, what a module can be grouped by, financial metrics only in
    the annual securities report); violating draws are redrawn.
  - distractors: up to 3, from the same module and from the same family in
    other modules (what makes the selection genuinely confusable).
"""
from __future__ import annotations
import argparse
import json
import pathlib
import random
import re
import sys
from collections import Counter
from typing import Any

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from eval_bench.dataset.loader import load_cases  # noqa: E402
from eval_bench.dataset.tools import load_tools, validate_case_against_tools  # noqa: E402

SEED = 20261015
TOTAL = 200
NEGATIVES = 30
MIN_PER_TOOL = 4
FIRST_NEW_ID = 1001
SENTINELS = {"none", "any"}
OPTIONAL_P = 0.5
FISCAL_YEARS = range(2018, 2026)

FAMILIES = {
    "search_record": re.compile(r"^search_(sales|purchase|inventory|accounting|hr)_record$"),
    "get_detail": re.compile(r"^get_(sales|purchase|inventory|accounting|hr)_detail$"),
    "aggregate": re.compile(r"^aggregate_(sales|purchase|inventory|accounting|hr)$"),
    "trend": re.compile(r"^trend_(sales|purchase|inventory|hr)$"),
    "list_open": re.compile(r"^list_open_(sales|purchase|inventory|hr)_items$"),
    "list_pending": re.compile(r"^list_pending_(sales|purchase|hr)_approvals$"),
}
CONFUSABLE_FAMILIES = ("search_record", "aggregate", "trend")

# Free-text values. Fictitious names; search terms an ERP user would type.
KEYWORDS = {
    "sales": ["ステンレス配管", "保守契約", "東日本支店", "返品", "スポット案件", "定期便"],
    "purchase": ["梱包資材", "電子部品", "外注加工", "消耗品", "原材料A", "輸入品"],
    "inventory": ["第2倉庫", "棚卸差異", "滞留在庫", "ボルトM8", "検品待ち", "移動中"],
    "accounting": ["減価償却", "仮払金", "交際費", "前受金", "本社経費", "為替差損"],
    "hr": ["中途採用", "時間外労働", "有給休暇", "営業部", "契約社員", "評価面談"],
    "master": ["精密", "物流", "東京", "樹脂", "サービス", "北海道"],
}
REPORT_KEYWORDS = {   # module "report": search terms differ by document type
    "yuho": ["人的資本", "セグメント", "気候変動", "株主還元", "研究開発", "内部統制"],
    "ordinance": ["個人情報保護", "情報公開", "手数料", "補助金交付", "公共施設"],
    "law": ["下請代金", "労働時間", "電子帳簿保存", "インボイス", "個人情報"],
    "internal_manual": ["経費精算", "稟議", "情報セキュリティ", "出張旅費", "購買手順"],
}
NAMES = {
    "supplier": ["東都精工株式会社", "三葉化成株式会社", "北辰ロジスティクス株式会社"],
    "customer": ["株式会社みなと建設", "西海フーズ株式会社", "株式会社ひかり医療"],
    "item": ["ステンレスボルトM8", "梱包用段ボール大", "樹脂ペレットPP"],
    "department": ["第一営業部", "購買課", "品質保証部"],
    "employee": ["佐藤 健一", "鈴木 美咲", "高橋 誠"],
}
CODE_PREFIX = {"supplier": "S", "customer": "C", "item": "ITM", "department": "D", "employee": "E"}
DOC_PREFIX = {   # by document type, so a receipt never carries a purchase-order number
    "sales_order": "SO", "invoice": "INV", "shipment": "SHP",
    "purchase_order": "PO", "receipt": "RC", "supplier_invoice": "SI",
    "stock": "ST", "warehouse": "WH", "movement": "MV",
    "journal": "JE", "account_balance": "AB", "cost_center": "CC",
    "attendance": "AT", "payroll": "PY", "headcount": "HC",
}


# Combinations the tool schemas allow but no user would ask for (an HR
# headcount grouped by supplier, an employee's credit limit). Independent enum
# draws produce them; a query written for one would be unnatural, so they are
# redrawn. Each entry: which values of `key` make sense given another argument.
FIELDS_BY_ENTITY = {
    "supplier": {"registered_address", "shipping_address", "phone_number", "email", "contact_person", "payment_terms"},
    "customer": {"registered_address", "shipping_address", "phone_number", "email", "contact_person",
                 "payment_terms", "credit_limit"},
    "item": {"unit_price", "unit_of_measure", "item_category"},
    "department": {"parent_department", "manager", "phone_number"},
    "employee": {"hire_date", "position", "email", "phone_number"},
}
GROUP_BY_BY_MODULE = {
    "sales": {"department", "customer", "item", "month"},
    "purchase": {"department", "supplier", "item", "month"},
    "inventory": {"item", "month"},
    "accounting": {"department", "month"},
    "hr": {"department", "month"},
}
GROUP_BY_BY_METRIC = {   # aggregate_metric
    "sales": {"department", "item", "month"},
    "purchase_amount": {"department", "supplier", "item", "month"},
    "headcount": {"department", "month"},
    "inventory_qty": {"item", "month"},
    "work_hours": {"department", "month"},
}
YUHO_ONLY = {"metric", "section"}   # financial metrics and report sections exist only in the annual securities report
INTERVALS_BY_PERIOD = {   # a trend's interval must be finer than its period
    "last_month": {"day", "week"}, "this_month": {"day", "week"},
    "last_quarter": {"day", "week", "month"},
    "this_fiscal_year": {"week", "month", "quarter"}, "last_fiscal_year": {"week", "month", "quarter"},
}
AGGREGATIONS_BY_ENTITY = {   # aggregate_<module>: what can sensibly be summed / averaged / counted
    "sales_order": {"sum", "avg", "max", "count"}, "invoice": {"sum", "avg", "max", "count"},
    "shipment": {"sum", "count"},
    "purchase_order": {"sum", "avg", "max", "count"}, "receipt": {"sum", "count"},
    "supplier_invoice": {"sum", "avg", "max", "count"},
    "stock": {"sum", "avg"}, "warehouse": {"sum"}, "movement": {"sum", "count"},
    "journal": {"sum", "count"}, "account_balance": {"sum", "avg"}, "cost_center": {"sum"},
    "attendance": {"sum", "avg"}, "payroll": {"sum", "avg", "max", "min"}, "headcount": {"count", "avg"},
}
GROUP_BY_BY_ENTITY = {"warehouse": {"month"}, "headcount": {"department", "month"}}   # narrower than the module rule
SINGLE_MONTH = {"last_month", "this_month"}   # grouping or trending these by month says nothing
HR_TREND_METRICS = {"headcount": {"count"}, "payroll": {"amount", "count"}, "attendance": {"quantity", "count"}}
COMPARABLE_PERIODS = [{"last_month", "this_month"}, {"last_fiscal_year", "this_fiscal_year"}]


def plausible(tool: dict[str, Any], args: dict[str, Any]) -> bool:
    name, module = tool["name"], tool["module"]
    if "interval" in args and "period" in args and args["interval"] not in INTERVALS_BY_PERIOD[args["period"]]:
        return False
    if name == "trend_hr" and args["metric"] not in HR_TREND_METRICS[args["entity"]]:
        return False
    if name == "search_report" and {"metric", "query_text"} <= args.keys():
        return False   # look up a metric, or search by text -- not both at once
    if name == "compare_period_metric":
        return {args["period_a"], args["period_b"]} in COMPARABLE_PERIODS
    if name == "search_master" and "field" in args:
        return args["field"] in FIELDS_BY_ENTITY[args["entity"]]
    if args.get("group_by") == "month" and args.get("period") in SINGLE_MONTH:
        return False
    if name.startswith("aggregate_") and name != "aggregate_metric":
        if args["aggregation"] not in AGGREGATIONS_BY_ENTITY[args["entity"]]:
            return False
        if "group_by" in args and args["group_by"] not in GROUP_BY_BY_ENTITY.get(args["entity"], {args["group_by"]}):
            return False
    if name == "aggregate_metric" and args["aggregation"] == "count":
        return False   # every metric here is a quantity; counting it is not a question anyone asks
    if name == "search_report" and not {"fiscal_year", "metric", "query_text"} & args.keys():
        return False   # with nothing to look up it is just "open the report"
    if name == "search_report_section" and "section" not in args:
        return False   # without a section it is search_report
    if "group_by" in args:
        if name == "aggregate_metric":
            return args["group_by"] in GROUP_BY_BY_METRIC[args["metric"]]
        return args["group_by"] in GROUP_BY_BY_MODULE[module]
    if module == "report" and YUHO_ONLY & args.keys():
        return args.get("report_type") == "yuho"
    return True


def family_of(name: str) -> str | None:
    for fam, rx in FAMILIES.items():
        if rx.match(name):
            return fam
    return None


def _value(rng: random.Random, tool: dict[str, Any], key: str, spec: dict[str, Any],
           args: dict[str, Any]) -> Any:
    module = tool["module"]
    if "enum" in spec:
        choices = [v for v in spec["enum"] if v not in SENTINELS]
        if key == "period_b":
            choices = [v for v in choices if v != args.get("period_a")]
        return rng.choice(choices)
    if spec.get("type") == "boolean":
        return rng.choice([True, False])
    if spec.get("type") == "integer":
        if key == "fiscal_year":
            return rng.choice(FISCAL_YEARS)
        if key == "year_from":
            return rng.choice(FISCAL_YEARS[:-1])
        if key == "year_to":
            lo = args.get("year_from", FISCAL_YEARS[0] - 1)
            return rng.choice([y for y in FISCAL_YEARS if y > lo])
        if key == "depth":
            return rng.randint(1, 3)
    if key in ("code", "root_code"):
        prefix = CODE_PREFIX.get(args.get("entity"), "D")
        return f"{prefix}-{rng.randint(100, 9999)}" if prefix != "ITM" else f"ITM-{rng.randint(1000, 99999)}"
    if key == "document_no":
        return f"{DOC_PREFIX[args['entity']]}-{rng.choice(FISCAL_YEARS)}-{rng.randint(1, 99999):05d}"
    if key == "name":
        return rng.choice(NAMES.get(args.get("entity"), NAMES["supplier"]))
    if key in ("keyword", "query_text"):
        if module == "report":
            return rng.choice(REPORT_KEYWORDS[args["report_type"]])
        return rng.choice(KEYWORDS[module])
    raise ValueError(f"no value rule for {tool['name']}.{key}: {spec}")


def sample_args(rng: random.Random, tool: dict[str, Any]) -> dict[str, Any]:
    params = tool["parameters"]
    required = set(params.get("required", []))
    args: dict[str, Any] = {}
    # Dict order puts entity / year_from / period_a first, which the constraints rely on.
    for key, spec in params["properties"].items():
        if key not in required and rng.random() >= OPTIONAL_P:
            continue
        if key == "year_to" and "year_from" not in args and key not in required:
            continue
        args[key] = _value(rng, tool, key, spec, args)
    # A required year_to with an optional year_from left out still needs a start for the range.
    if "year_to" in args and "year_from" in required and args["year_from"] >= args["year_to"]:
        raise AssertionError("year range drawn backwards")
    return args


def distractors(rng: random.Random, name: str, tools: dict[str, dict[str, Any]]) -> list[str]:
    module = tools[name]["module"]
    fam = family_of(name)
    same_module = [n for n, t in tools.items() if t["module"] == module and n != name]
    same_family = [n for n in tools if fam and n != name and family_of(n) == fam]
    rng.shuffle(same_module)
    rng.shuffle(same_family)
    picked = same_family[:1] + same_module[:2]
    for n in same_family[1:] + same_module[2:]:
        if len(picked) >= 3:
            break
        if n not in picked:
            picked.append(n)
    return picked[:3]


def allocation(rng: random.Random, tools: dict[str, dict[str, Any]],
               existing: list[dict[str, Any]]) -> Counter:
    have = Counter(c["expected_tool"] for c in existing if not c["negative"])
    positives_needed = TOTAL - NEGATIVES - sum(have.values())
    plan = Counter({n: max(0, MIN_PER_TOOL - have[n]) for n in tools})
    extra = positives_needed - sum(plan.values())
    if extra < 0:
        raise ValueError(f"{MIN_PER_TOOL} per tool does not fit in {TOTAL - NEGATIVES} positives")
    confusable = sorted(n for n in tools if family_of(n) in CONFUSABLE_FAMILIES)
    rng.shuffle(confusable)
    for i in range(extra):
        plan[confusable[i % len(confusable)]] += 1
    return plan


def _key(tool: str, args: dict[str, Any]) -> str:
    return json.dumps([tool, args], sort_keys=True, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=pathlib.Path, default=_ROOT / "data_build" / "expand" / "gold_calls.jsonl")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    tools = load_tools(_ROOT / "schema" / "tools.json")
    # The hand-made cases only: once data/generated/ exists, it must not feed back into its own draw.
    existing = [c for c in load_cases(_ROOT / "data") if c["provenance"] != "generated"]
    seen = {_key(c["expected_tool"], c["expected_args"]["exact"]) for c in existing if not c["negative"]}

    rows = []
    next_id = FIRST_NEW_ID
    for name, count in sorted(allocation(rng, tools, existing).items()):
        for _ in range(count):
            for _attempt in range(200):
                call_args = sample_args(rng, tools[name])
                if plausible(tools[name], call_args) and _key(name, call_args) not in seen:
                    break
            else:
                raise RuntimeError(f"could not draw a distinct call for {name}")
            seen.add(_key(name, call_args))
            row = {
                "id": f"case_{next_id:04d}",
                "expected_tool": name,
                "expected_args": {"exact": call_args, "semantic": {}},
                "distractor_tools": distractors(rng, name, tools),
                "negative": False,
            }
            validate_case_against_tools(row, tools)
            rows.append(row)
            next_id += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    per_module = Counter(tools[r["expected_tool"]]["module"] for r in rows)
    print(f"wrote {len(rows)} gold calls to {args.out} (seed {args.seed}); by module: {dict(per_module)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
