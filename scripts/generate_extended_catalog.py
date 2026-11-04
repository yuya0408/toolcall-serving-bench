"""Generates schema/tools_extended.json: the 100-tool catalog for the
engine-performance layer's tool_stage=50/100 (the design notes kept outside this repo).

Design constraints this generator encodes (see the plan doc for the full
rationale, decided 2026-08-29):
  - The original 40 tools (schema/tools.json) are carried over byte-for-byte
    (loaded, not re-typed) so stage=10 results stay comparable to the original 40-tool results.
  - 60 new tools, ERP domain throughout. Allocation is weighted by which
    modules existing cases in data/ actually anchor on (report/sales/master
    carry 10 of 12 positive cases; accounting/purchase 1 each; inventory/hr
    0) -- build_tool_set pads same-module-first, so deepening a module no
    case references only ever shows up as generic other-module padding,
    same failure mode as inventing a disconnected new module. Two new
    modules (production, project) are added per the plan's "necessary
    additional ERP module" allowance, kept to real record types
    (work_order/bom/routing, project/wbs_task/timesheet) rather than
    invented ones.
  - Deepening tools reuse the original 40's per-module template shapes
    (search_X_record / get_X_detail / aggregate_X / trend_X /
    list_open_X_items / list_pending_X_approvals) where that closes a real
    structural gap (e.g. accounting/inventory never got list_open/
    list_pending in the original 40); bespoke tools are genuinely distinct
    business functions, not shape clones.

Run: python scripts/generate_extended_catalog.py
"""
from __future__ import annotations
import json, pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ORIGINAL_PATH = _ROOT / "schema" / "tools.json"
_OUTPUT_PATH = _ROOT / "schema" / "tools_extended.json"

PERIOD_ENUM = ["last_month", "this_month", "last_quarter", "this_fiscal_year", "last_fiscal_year"]
STATUS_ENUM = ["draft", "confirmed", "closed", "cancelled", "any"]
AGGREGATION_ENUM = ["sum", "avg", "max", "min", "count"]
TREND_METRIC_ENUM = ["amount", "quantity", "count"]
CODE_PATTERN = "^[0-9A-Z-]{1,16}$"
DOCNO_PATTERN = "^[0-9A-Z-]{1,20}$"


def prop(type_, description, x_scoring="exact", **extra):
    d = {"type": type_, "description": description, "x-scoring": x_scoring}
    d.update(extra)
    return d


def enum_prop(values, description, x_scoring="exact"):
    return prop("string", description, x_scoring, enum=values)


def code_prop(description, pattern=CODE_PATTERN, x_scoring="exact"):
    return prop("string", description, x_scoring, pattern=pattern)


def free_text_prop(description):
    # Scored exact after NFKC normalization (eval_bench/scorers/separated.py),
    # matching schema/tools.json 2.1.0: no field in either catalog is semantic.
    return prop("string", description)


def tool(name, module, description, required, properties):
    return {
        "name": name,
        "module": module,
        "description": description,
        "parameters": {
            "type": "object",
            "required": required,
            "additionalProperties": False,
            "properties": properties,
        },
    }


def standard_module_tools(module, jp, entity_enum, group_by_enum, approval_stage_enum, approval_note):
    """The five templated shapes shared by every original module, retargeted
    at a new module. list_pending_X_approvals is deliberately NOT templated
    here -- its description must name what it is being contrasted against
    (list_open_X_items's specific failure mode for this module), which the
    original catalog also writes by hand per module.
    """
    entity_desc = "対象レコード種別"
    return [
        tool(
            f"search_{module}_record", module,
            f"{jp}モジュールの伝票・レコードを条件で検索し、該当する一覧を返す。集計値ではなく個々のレコードが必要な場合に使う。",
            ["entity"],
            {
                "entity": enum_prop(entity_enum, entity_desc),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
                "status": enum_prop(STATUS_ENUM, "レコードの状態"),
                "keyword": free_text_prop("検索条件の自由記述"),
            },
        ),
        tool(
            f"get_{module}_detail", module,
            f"{jp}モジュールの単一レコードを伝票番号で特定し、明細行を含む詳細を返す。一覧検索ではなく特定レコードの中身が必要な場合に使う。",
            ["entity", "document_no"],
            {
                "entity": enum_prop(entity_enum, entity_desc),
                "document_no": code_prop("伝票番号", DOCNO_PATTERN),
                "include_lines": prop("boolean", "明細行を含めるか"),
            },
        ),
        tool(
            f"aggregate_{module}", module,
            f"{jp}モジュールの数量を期間・区分で集約する。個々のレコードではなく合計や平均が求められている場合に使う。",
            ["entity", "aggregation"],
            {
                "entity": enum_prop(entity_enum, "集計対象種別"),
                "aggregation": enum_prop(AGGREGATION_ENUM, "集約方法"),
                "group_by": enum_prop(group_by_enum, "区分軸"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
            },
        ),
        tool(
            f"trend_{module}", module,
            f"{jp}モジュールの数量の時系列推移を返す。単一期間の集計ではなく、期間をまたぐ推移が求められている場合に使う。",
            ["entity", "metric"],
            {
                "entity": enum_prop(entity_enum, "対象種別"),
                "metric": enum_prop(TREND_METRIC_ENUM, "推移させる数量"),
                "interval": enum_prop(["day", "week", "month", "quarter"], "時系列の粒度"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
            },
        ),
        tool(
            f"list_open_{module}_items", module,
            f"{jp}モジュールの未処理・未消込レコードを一覧する。全レコードの検索ではなく、滞留しているものだけが必要な場合に使う。",
            ["entity"],
            {
                "entity": enum_prop(entity_enum, "対象種別"),
                "overdue_only": prop("boolean", "期限超過分のみに限定するか"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
            },
        ),
        tool(
            f"list_pending_{module}_approvals", module,
            f"承認待ちで停止している{jp}系レコードを一覧する。list_open_{module}_itemsが指す未処理状態とは異なり、{approval_note}にあるものだけを対象とする。",
            ["entity"],
            {
                "entity": enum_prop(entity_enum, "対象種別"),
                "approval_stage": enum_prop(approval_stage_enum, "承認ステージ"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
            },
        ),
    ]


def deepening_tools():
    out = []

    # -- report (9): 4 of 12 positive cases anchor here, the largest share.
    out += [
        tool(
            "get_report_disclosure_schedule", "report",
            "文書の開示・提出スケジュール（提出期限、提出日）を返す。記載内容そのものではなく、いつ開示されるかが必要な場合に使う。",
            ["report_type"],
            {
                "report_type": enum_prop(["yuho", "ordinance", "law", "internal_manual"], "対象文書の種別"),
                "fiscal_year": prop("integer", "対象年度", minimum=1900, maximum=2100),
                "disclosure_type": enum_prop(["scheduled", "filed", "amended", "any"], "開示区分"),
            },
        ),
        tool(
            "get_report_audit_status", "report",
            "文書に対する監査・レビューの状況（未監査／レビュー中／監査済み／限定付き）を返す。記載値そのものではなく、信頼性の状況が必要な場合に使う。",
            ["report_type", "fiscal_year"],
            {
                "report_type": enum_prop(["yuho", "ordinance", "law", "internal_manual"], "対象文書の種別"),
                "fiscal_year": prop("integer", "対象年度", minimum=1900, maximum=2100),
            },
        ),
        tool(
            "list_report_amendments", "report",
            "文書の訂正・追補の履歴を一覧する。原本の記載値ではなく、後から変更された箇所の履歴が必要な場合に使う。",
            ["report_type"],
            {
                "report_type": enum_prop(["yuho", "ordinance", "law", "internal_manual"], "対象文書の種別"),
                "fiscal_year": prop("integer", "対象年度", minimum=1900, maximum=2100),
                "keyword": free_text_prop("訂正箇所の絞り込み条件の自由記述"),
            },
        ),
        tool(
            "get_report_filing_status", "report",
            "文書が実際に提出済みかどうかと提出日を返す。開示スケジュール（予定）ではなく、実際の提出実績が必要な場合に使う。",
            ["report_type", "fiscal_year"],
            {
                "report_type": enum_prop(["yuho", "ordinance", "law", "internal_manual"], "対象文書の種別"),
                "fiscal_year": prop("integer", "対象年度", minimum=1900, maximum=2100),
            },
        ),
        tool(
            "search_report_reference", "report",
            "文書間の参照・引用関係を検索する。単一文書内の記載ではなく、他の文書や規程を参照している箇所を横断的に探す場合に使う。",
            ["report_type"],
            {
                "report_type": enum_prop(["yuho", "ordinance", "law", "internal_manual"], "対象文書の種別"),
                "ref_report_type": enum_prop(["yuho", "ordinance", "law", "internal_manual"], "参照先文書の種別"),
                "keyword": free_text_prop("参照内容の絞り込み条件の自由記述"),
            },
        ),
        tool(
            "get_report_revision_history", "report",
            "文書自体の版数・改定履歴を返す。記載値の変遷ではなく、文書というオブジェクトそのものがいつ改版されたかが必要な場合に使う。",
            ["report_type"],
            {
                "report_type": enum_prop(["yuho", "ordinance", "law", "internal_manual"], "対象文書の種別"),
                "fiscal_year": prop("integer", "対象年度", minimum=1900, maximum=2100),
            },
        ),
        tool(
            "list_report_distribution_targets", "report",
            "文書の配布・開示先（社内限定／官公庁提出／一般開示）の一覧を返す。記載内容ではなく、誰に向けた文書かが必要な場合に使う。",
            ["report_type"],
            {
                "report_type": enum_prop(["yuho", "ordinance", "law", "internal_manual"], "対象文書の種別"),
                "fiscal_year": prop("integer", "対象年度", minimum=1900, maximum=2100),
            },
        ),
        tool(
            "compare_report_scope", "report",
            "文書の対象範囲（連結／単体、対象子会社数など）を2年度間で比較する。指標値の比較（compare_report_metric）ではなく、報告対象そのものの範囲変更が必要な場合に使う。",
            ["report_type", "year_from", "year_to"],
            {
                "report_type": enum_prop(["yuho", "ordinance", "law", "internal_manual"], "対象文書の種別"),
                "year_from": prop("integer", "比較開始年度", minimum=1900, maximum=2100),
                "year_to": prop("integer", "比較終了年度", minimum=1900, maximum=2100),
            },
        ),
        tool(
            "get_report_disclosure_category", "report",
            "文書がどの開示区分（適時開示／法定開示／任意開示）に属するかを返す。記載内容ではなく、開示の法的位置づけが必要な場合に使う。",
            ["report_type"],
            {
                "report_type": enum_prop(["yuho", "ordinance", "law", "internal_manual"], "対象文書の種別"),
                "fiscal_year": prop("integer", "対象年度", minimum=1900, maximum=2100),
            },
        ),
    ]

    # -- sales (8): 3 of 12 positive cases anchor here.
    sales_entity = ["sales_order", "invoice", "shipment"]
    out += [
        tool(
            "get_customer_credit_status", "sales",
            "得意先の与信状況（与信限度額・利用額・与信超過有無）を返す。個別の販売レコードではなく、取引先単位の与信情報が必要な場合に使う。",
            ["customer_code"],
            {
                "customer_code": code_prop("得意先コード"),
                "as_of": enum_prop(PERIOD_ENUM, "基準時点"),
            },
        ),
        tool(
            "list_sales_returns", "sales",
            "販売モジュールの返品レコードを一覧する。通常の出荷・請求レコードではなく、返品として処理されたものだけが必要な場合に使う。",
            ["return_type"],
            {
                "return_type": enum_prop(["customer_return", "credit_note", "any"], "返品区分"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
                "reason": free_text_prop("返品理由の自由記述"),
            },
        ),
        tool(
            "get_sales_commission", "sales",
            "営業担当者ごとのインセンティブ・歩合計算結果を返す。伝票の内容そのものではなく、担当者への支給額計算が必要な場合に使う。",
            ["employee_code", "period"],
            {
                "employee_code": code_prop("営業担当者の従業員コード"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
            },
        ),
        tool(
            "compare_sales_channel", "sales",
            "販売チャネル（直販／代理店／EC）間で売上を比較する。期間比較（compare_period_metric）ではなく、チャネルという区分軸での比較が必要な場合に使う。",
            ["metric", "channel_a", "channel_b"],
            {
                "metric": enum_prop(TREND_METRIC_ENUM, "比較する数量"),
                "channel_a": enum_prop(["direct", "agency", "ec"], "比較元チャネル"),
                "channel_b": enum_prop(["direct", "agency", "ec"], "比較先チャネル"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
            },
        ),
        tool(
            "get_sales_price_history", "sales",
            "特定品目・得意先の販売価格の改定履歴を返す。個別伝票の金額ではなく、価格そのものがいつ・どう変わったかが必要な場合に使う。",
            ["item_code"],
            {
                "item_code": code_prop("品目コード"),
                "customer_code": code_prop("得意先コード（限定する場合）"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
            },
        ),
        tool(
            "list_sales_backorders", "sales",
            "在庫不足等により出荷保留になっている受注を一覧する。list_open_sales_itemsが指す未処理全般とは異なり、欠品起因で止まっているものだけを対象とする。",
            ["entity"],
            {
                "entity": enum_prop(sales_entity, "対象種別"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
            },
        ),
        tool(
            "get_sales_document_history", "sales",
            "伝票のステータス変更履歴（起票→承認→出荷→請求などの遷移）を返す。現在のステータスではなく、いつ何が変わったかが必要な場合に使う。",
            ["entity", "document_no"],
            {
                "entity": enum_prop(sales_entity, "対象種別"),
                "document_no": code_prop("伝票番号", DOCNO_PATTERN),
            },
        ),
        tool(
            "aggregate_sales_margin", "sales",
            "販売モジュールの粗利・利益率を期間・区分で集約する。売上高そのもの（aggregate_sales）ではなく、原価差引後の利益指標が必要な場合に使う。",
            ["aggregation"],
            {
                "aggregation": enum_prop(AGGREGATION_ENUM, "集約方法"),
                "group_by": enum_prop(["department", "supplier", "customer", "item", "month", "none"], "区分軸"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
            },
        ),
    ]

    # -- master (7): 3 of 12 positive cases anchor here.
    master_entity = ["supplier", "customer", "item", "department", "employee"]
    out += [
        tool(
            "get_master_contact", "master",
            "取引先・部門などマスタレコードに紐づく担当者連絡先を返す。登録属性そのもの（search_master）ではなく、連絡先情報だけが必要な場合に使う。",
            ["entity", "code"],
            {
                "entity": enum_prop(master_entity, "対象マスタ種別"),
                "code": code_prop("レコードを特定するコード"),
            },
        ),
        tool(
            "get_master_credit_limit", "master",
            "取引先マスタの与信限度額設定を返す。get_customer_credit_status（利用実績込みの与信状況）とは異なり、マスタに登録された限度額そのものだけを対象とする。",
            ["entity", "code"],
            {
                "entity": enum_prop(["supplier", "customer"], "対象マスタ種別"),
                "code": code_prop("レコードを特定するコード"),
            },
        ),
        tool(
            "list_master_duplicate_candidates", "master",
            "名称やコードが類似するマスタレコードの重複候補を一覧する。単一レコードの検索ではなく、名寄せ・データクレンジング目的で使う。",
            ["entity"],
            {
                "entity": enum_prop(master_entity, "対象マスタ種別"),
                "keyword": free_text_prop("絞り込み条件の自由記述"),
            },
        ),
        tool(
            "get_master_relationship", "master",
            "マスタレコード間の紐付け（品目と仕入先、部門と従業員など）を返す。単一エンティティの属性ではなく、エンティティ間の関連が必要な場合に使う。",
            ["entity", "code", "related_entity"],
            {
                "entity": enum_prop(master_entity, "起点となるマスタ種別"),
                "code": code_prop("起点レコードのコード"),
                "related_entity": enum_prop(master_entity, "紐付け先のマスタ種別"),
            },
        ),
        tool(
            "list_master_certifications", "master",
            "取引先・品目に登録された認証・資格情報（ISO認証、有資格者情報など）を一覧する。登録属性の取得（search_master）ではなく、証明書類の状況が必要な場合に使う。",
            ["entity", "code"],
            {
                "entity": enum_prop(master_entity, "対象マスタ種別"),
                "code": code_prop("レコードを特定するコード"),
            },
        ),
        tool(
            "search_master_by_attribute", "master",
            "コードや名称ではなく、属性条件（地域・業種・等級など）でマスタレコードを横断検索する。resolve_codeやsearch_masterのような単一キー特定ではなく、条件一致による複数件検索に使う。",
            ["entity"],
            {
                "entity": enum_prop(master_entity, "対象マスタ種別"),
                "attribute": free_text_prop("絞り込みたい属性条件の自由記述"),
                "active_only": prop("boolean", "有効レコードのみに限定するか"),
            },
        ),
        tool(
            "get_master_approval_status", "master",
            "マスタ新規登録・変更の承認状況を返す。get_master_change_history（確定済み変更の履歴）とは異なり、現在承認待ちかどうかの状態を対象とする。",
            ["entity", "code"],
            {
                "entity": enum_prop(master_entity, "対象マスタ種別"),
                "code": code_prop("レコードを特定するコード"),
            },
        ),
    ]

    # -- accounting (5): 1 of 12 positive cases anchors here. Also closes a
    # structural gap -- the original 40 never gave accounting the
    # list_open/list_pending pair that sales/purchase/hr have.
    accounting_entity = ["journal", "account_balance", "cost_center"]
    out += [
        tool(
            "list_open_accounting_items", "accounting",
            "会計モジュールの未消込・未処理のレコードを一覧する。search_accounting_recordの全件検索とは異なり、消込・処理が完了していないものだけを対象とする。",
            ["entity"],
            {
                "entity": enum_prop(accounting_entity, "対象種別"),
                "overdue_only": prop("boolean", "期限超過分のみに限定するか"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
            },
        ),
        tool(
            "list_pending_accounting_approvals", "accounting",
            "承認待ちで停止している会計系レコードを一覧する。list_open_accounting_itemsが指す未消込状態とは異なり、承認ワークフロー上の停止（経理承認・部門長承認など）にあるものだけを対象とする。",
            ["entity"],
            {
                "entity": enum_prop(accounting_entity, "対象種別"),
                "approval_stage": enum_prop(["dept_approval", "finance_approval", "cfo_signoff", "any"], "承認ステージ"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
            },
        ),
        tool(
            "get_budget_variance", "accounting",
            "予算と実績の差異を部門・勘定科目単位で返す。aggregate_metricの単純集計ではなく、計画値との比較が必要な場合に使う。",
            ["metric", "period"],
            {
                "metric": enum_prop(["sales", "purchase_amount", "headcount", "inventory_qty", "work_hours"], "対象数量"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
                "group_by": enum_prop(["department", "supplier", "customer", "item", "month", "none"], "区分軸"),
            },
        ),
        tool(
            "get_tax_summary", "accounting",
            "消費税等の税額集計を期間で返す。個々の仕訳明細ではなく、申告・納付のための税額サマリが必要な場合に使う。",
            ["period"],
            {
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
                "tax_type": enum_prop(["consumption_tax", "withholding_tax", "corporate_tax", "any"], "税目"),
            },
        ),
        tool(
            "get_cost_center_hierarchy", "accounting",
            "コストセンターの階層構造を返す。get_master_hierarchyが対象とする部門・品目階層とは異なり、会計上のコストセンター区分の親子関係を対象とする。",
            ["root_code"],
            {
                "root_code": code_prop("起点ノードのコード"),
                "depth": prop("integer", "取得する階層の深さ", minimum=1, maximum=10),
            },
        ),
    ]

    # -- purchase (4): 1 of 12 positive cases anchors here.
    out += [
        tool(
            "get_supplier_performance", "purchase",
            "仕入先の納期遵守率・品質不良率などの評価指標を返す。個別の購買レコードではなく、仕入先単位のパフォーマンス評価が必要な場合に使う。",
            ["supplier_code", "period"],
            {
                "supplier_code": code_prop("仕入先コード"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
                "metric": enum_prop(["on_time_delivery_rate", "defect_rate", "lead_time_days"], "評価指標"),
            },
        ),
        tool(
            "list_purchase_returns", "purchase",
            "購買モジュールの返品・返送レコードを一覧する。通常の入荷・請求レコードではなく、仕入先への返品として処理されたものだけが必要な場合に使う。",
            ["return_type"],
            {
                "return_type": enum_prop(["supplier_return", "debit_note", "any"], "返品区分"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
                "reason": free_text_prop("返品理由の自由記述"),
            },
        ),
        tool(
            "get_contract_terms", "purchase",
            "仕入先との取引条件・契約内容（支払条件、単価契約の有効期間など）を返す。個別の伝票内容ではなく、契約そのものの条件が必要な場合に使う。",
            ["supplier_code"],
            {
                "supplier_code": code_prop("仕入先コード"),
                "contract_type": enum_prop(["price_agreement", "payment_terms", "framework_contract", "any"], "契約種別"),
            },
        ),
        tool(
            "compare_purchase_price", "purchase",
            "同一品目の仕入価格を2期間または2仕入先間で比較する。個々の購買レコードではなく、単価の増減や仕入先間の差が必要な場合に使う。",
            ["item_code", "period_a", "period_b"],
            {
                "item_code": code_prop("品目コード"),
                "period_a": enum_prop(PERIOD_ENUM, "比較元期間"),
                "period_b": enum_prop(PERIOD_ENUM, "比較先期間"),
                "supplier_code": code_prop("仕入先コード（限定する場合）"),
            },
        ),
    ]

    # -- inventory (2): 0 positive cases anchor here today. Minimal --
    # closes the same list_pending structural gap accounting had, plus one
    # genuinely distinct valuation query.
    inventory_entity = ["stock", "warehouse", "movement"]
    out += [
        tool(
            "list_pending_inventory_approvals", "inventory",
            "承認待ちで停止している在庫系レコードを一覧する。list_open_inventory_itemsが指す未処理状態とは異なり、承認ワークフロー上の停止（棚卸差異承認・廃棄承認など）にあるものだけを対象とする。",
            ["entity"],
            {
                "entity": enum_prop(inventory_entity, "対象種別"),
                "approval_stage": enum_prop(["adjustment_approval", "disposal_approval", "manager_signoff", "any"], "承認ステージ"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
            },
        ),
        tool(
            "get_inventory_valuation", "inventory",
            "在庫の評価額（原価・簿価）を倉庫・品目単位で返す。aggregate_inventoryの数量集計とは異なり、金額ベースの評価が必要な場合に使う。",
            ["period"],
            {
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
                "group_by": enum_prop(["warehouse", "item", "none"], "区分軸"),
            },
        ),
    ]

    # -- hr (1): 0 positive cases anchor here today. Minimal.
    out += [
        tool(
            "get_employee_leave_balance", "hr",
            "従業員の有給休暇等の残日数を返す。attendance（勤怠実績）レコードとは異なり、休暇の残数という状態値そのものを対象とする。",
            ["employee_code"],
            {
                "employee_code": code_prop("従業員コード"),
                "leave_type": enum_prop(["paid_leave", "sick_leave", "special_leave", "any"], "休暇種別"),
            },
        ),
    ]

    assert len(out) == 36, f"expected 36 deepening tools, got {len(out)}"
    return out


def new_module_tools():
    out = []

    # -- production (12): standard 6 + 6 bespoke. Real PP-module record
    # types (work_order/production_plan/material_requirement), not
    # invented ones.
    out += standard_module_tools(
        "production", "生産",
        entity_enum=["work_order", "production_plan", "material_requirement"],
        group_by_enum=["work_center", "department", "item", "month", "none"],
        approval_stage_enum=["engineering_change_approval", "capacity_release", "manager_signoff", "any"],
        approval_note="承認ワークフロー上の停止（設計変更承認・能力確保承認など）",
    )
    out += [
        tool(
            "get_bill_of_materials", "production",
            "品目の部品表（構成品目と数量）を返す。get_production_detailが個別の指図伝票を対象とするのに対し、本ツールは品目そのものの構成定義を対象とする。",
            ["item_code"],
            {
                "item_code": code_prop("対象品目コード"),
                "revision": prop("string", "部品表の版数（未指定なら最新版）"),
                "include_components": prop("boolean", "構成品目の内訳を含めるか"),
            },
        ),
        tool(
            "get_production_routing", "production",
            "品目の製造工程順序（ルーティング）を返す。部品表（構成品目）ではなく、どの工程をどの順で通るかが必要な場合に使う。",
            ["item_code"],
            {
                "item_code": code_prop("対象品目コード"),
                "process_type": enum_prop(["standard", "rework", "subcontract"], "工程区分"),
            },
        ),
        tool(
            "get_capacity_utilization", "production",
            "工程・設備単位の稼働率を返す。aggregate_productionの数量集計とは異なり、設備の稼働状況そのものが必要な場合に使う。",
            ["work_center"],
            {
                "work_center": code_prop("対象工程・設備コード"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
                "metric": enum_prop(["utilization_rate", "available_hours", "planned_hours"], "稼働指標"),
            },
        ),
        tool(
            "get_yield_variance", "production",
            "製造指図の歩留・不良率の計画差異を返す。get_production_detailの伝票内容そのものではなく、良品率の差異という評価値が必要な場合に使う。",
            ["document_no"],
            {
                "document_no": code_prop("対象製造指図番号", DOCNO_PATTERN),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
            },
        ),
        tool(
            "get_engineering_change_history", "production",
            "品目の設計変更（ECN）履歴を返す。部品表の現在の構成ではなく、いつ・どう変更されたかの履歴が必要な場合に使う。",
            ["item_code"],
            {
                "item_code": code_prop("対象品目コード"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
            },
        ),
        tool(
            "compare_production_plan_actual", "production",
            "生産計画と実績の差異を2期間または計画対実績で比較する。trend_productionの推移そのものではなく、計画値との乖離が求められている場合に使う。",
            ["metric", "period_a", "period_b"],
            {
                "metric": enum_prop(["planned_qty", "actual_qty", "planned_hours", "actual_hours"], "比較する数量"),
                "period_a": enum_prop(PERIOD_ENUM, "比較元期間"),
                "period_b": enum_prop(PERIOD_ENUM, "比較先期間"),
            },
        ),
    ]

    # -- project (12): standard 6 + 6 bespoke. Real PS-module record types
    # (project/wbs_task/timesheet).
    out += standard_module_tools(
        "project", "プロジェクト",
        entity_enum=["project", "wbs_task", "timesheet"],
        group_by_enum=["project", "wbs_task", "department", "month", "none"],
        approval_stage_enum=["budget_approval", "milestone_signoff", "closure_approval", "any"],
        approval_note="承認ワークフロー上の停止（予算承認・マイルストン承認・完了承認など）",
    )
    out += [
        tool(
            "get_project_budget_variance", "project",
            "プロジェクトの予算と実績の差異を返す。aggregate_projectの単純集計ではなく、計画値との比較が必要な場合に使う。",
            ["project_code", "period"],
            {
                "project_code": code_prop("対象プロジェクトコード"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
                "metric": enum_prop(["budget_amount", "actual_cost", "committed_cost"], "比較する数量"),
            },
        ),
        tool(
            "get_project_milestone_status", "project",
            "プロジェクトのマイルストン達成状況を返す。get_project_detailの伝票内容そのものではなく、マイルストン単位の進捗が必要な場合に使う。",
            ["project_code"],
            {
                "project_code": code_prop("対象プロジェクトコード"),
                "milestone_name": free_text_prop("絞り込みたいマイルストン名の自由記述"),
            },
        ),
        tool(
            "get_project_resource_allocation", "project",
            "プロジェクトへの要員アサイン状況を返す。get_timesheet_summaryの実績工数とは異なり、計画上の割り当てそのものが必要な場合に使う。",
            ["project_code"],
            {
                "project_code": code_prop("対象プロジェクトコード"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
                "role": free_text_prop("絞り込みたい役割・職種の自由記述"),
            },
        ),
        tool(
            "get_timesheet_summary", "project",
            "プロジェクトの工数実績を集計して返す。get_project_resource_allocationの計画アサインとは異なり、実際に計上された工数実績を対象とする。",
            ["project_code", "period"],
            {
                "project_code": code_prop("対象プロジェクトコード"),
                "period": enum_prop(PERIOD_ENUM, "対象期間"),
                "group_by": enum_prop(["employee", "wbs_task", "none"], "区分軸"),
            },
        ),
        tool(
            "list_project_change_requests", "project",
            "プロジェクトのスコープ・期間変更要求を一覧する。list_open_project_itemsの未処理レコード全般とは異なり、変更要求として提起されたものだけを対象とする。",
            ["project_code"],
            {
                "project_code": code_prop("対象プロジェクトコード"),
                "status": enum_prop(["submitted", "approved", "rejected", "any"], "変更要求の状態"),
            },
        ),
        tool(
            "compare_project_plan_actual", "project",
            "プロジェクトの計画と実績のスケジュール・コスト差異を比較する。get_project_budget_varianceの予算差異とは異なり、スケジュールやスコープ変更件数も含めた比較を対象とする。",
            ["metric", "period_a", "period_b"],
            {
                "metric": enum_prop(["schedule_variance_days", "cost_variance", "scope_change_count"], "比較する指標"),
                "period_a": enum_prop(PERIOD_ENUM, "比較元期間"),
                "period_b": enum_prop(PERIOD_ENUM, "比較先期間"),
            },
        ),
    ]

    assert len(out) == 24, f"expected 24 new-module tools, got {len(out)}"
    return out


def build_catalog():
    original = json.loads(_ORIGINAL_PATH.read_text(encoding="utf-8"))
    original_tools = original["tools"]
    assert len(original_tools) == 40, f"expected 40 original tools, got {len(original_tools)}"

    new_tools = deepening_tools() + new_module_tools()
    assert len(new_tools) == 60, f"expected 60 new tools, got {len(new_tools)}"

    all_tools = original_tools + new_tools
    names = [t["name"] for t in all_tools]
    assert len(names) == len(set(names)), "duplicate tool name across original+new catalog"
    assert len(all_tools) == 100, f"expected 100 total tools, got {len(all_tools)}"

    return {
        "$comment": (
            "Extended catalog for the engine-performance layer's tool_stage=50/100 "
            "(the design notes kept outside this repo). The first 40 entries are byte-identical "
            "to schema/tools.json (the original 40-tool catalog) so tool_stage=10 must be run "
            "against schema/tools.json directly, never against this file -- passing this "
            "100-tool dict at stage=10 would change the other-module padding candidate "
            "pool and break the stage-10 comparability property. Generated by "
            "scripts/generate_extended_catalog.py; do not hand-edit."
        ),
        "version": "ext100-1.1.0",
        "scoring_convention": original["scoring_convention"],
        "tool_set_stages": {
            "$comment": (
                "Stage 10 is NOT listed here -- it uses the original 40-tool catalog "
                "(schema/tools.json), not this file. This file only ever backs stage 50 and 100."
            ),
            "stages": [50, 100],
        },
        "tools": all_tools,
    }


def main():
    catalog = build_catalog()
    _OUTPUT_PATH.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(catalog['tools'])} tools to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
