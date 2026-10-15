# ケース追加の確認シート（ADR-0004 手順4）

各行の「判断」に keep（そのまま）/ fix（クエリだけ直す。直した文を書く）/ drop（捨てる）を記入してください。
正解の呼び出しは直しません。正解がおかしいものは drop にします。

- 印の付いた件：17件（Claudeの判断案つき）
- 無作為に選んだ件：50件

## 1. 印の付いた件

| ID | クエリ | 正解の呼び出し | 印の理由 | 判断案 | 判断 |
|---|---|---|---|---|---|
| case_1002 | 前四半期の原価センタの合計を見せて | `aggregate_accounting` {"entity": "cost_center", "aggregation": "sum", "period": "last_quarter"} | writer note: 原価センタのsumは何の合計か曖昧 | **drop**：原価センタの「合計」は何の合計か決まらない。正解の呼び出し自体が曖昧 | |
| case_1007 | 人事の人員レコードが何件あるか数えて | `aggregate_hr` {"entity": "headcount", "aggregation": "count"} | writer note: 人員(headcount)のcountは不自然 | **drop**：人員レコードの件数を数える依頼は業務として不自然 | |
| case_1010 | 合計や件数ではなく、人事の人員の平均を知りたい | `aggregate_hr` {"entity": "headcount", "aggregation": "avg"} | writer note: 人員(headcount)のavgは不自然 | **drop**：人員の平均は意味が通らない | |
| case_1012 | 前年度の在庫の平均を品目別に見せて | `aggregate_inventory` {"entity": "stock", "aggregation": "avg", "group_by": "item", "period": "last_fiscal_year"} | blind note: aggregate_metric(metric=inventory_qty)も候補 | **keep**：aggregate_metric(在庫数量)とも読めるが、品目別の在庫レコードの平均としては正解が妥当。カタログ上の重なりで、選択の難しさとして残す価値がある | |
| case_1013 | 前四半期の在庫の平均を教えて | `aggregate_inventory` {"entity": "stock", "aggregation": "avg", "period": "last_quarter"} | blind note: aggregate_metric(metric=inventory_qty)も候補 | **keep**：同上 | |
| case_1014 | 今月の在庫の平均を教えて | `aggregate_inventory` {"entity": "stock", "aggregation": "avg", "period": "this_month"} | blind note: aggregate_metric(metric=inventory_qty)も候補 | **keep**：同上 | |
| case_1015 | 個別の倉庫じゃなくて、前年度の倉庫の合計で | `aggregate_inventory` {"entity": "warehouse", "aggregation": "sum", "period": "last_fiscal_year"} | writer note: 倉庫のsumは何の合計か曖昧 | **drop**：倉庫の「合計」は何の合計か決まらない | |
| case_1018 | 仕入額の最小値はいくら？ | `aggregate_metric` {"metric": "purchase_amount", "aggregation": "min"} | blind note: aggregate_purchase(entityはsupplier_invoice/purchase_order等で不定)も候補 | **keep**：aggregate_purchaseは伝票の種類が決まらないので、仕入額ならaggregate_metricが自然 | |
| case_1019 | 前年度の従業員数の最小値を部門別に出して | `aggregate_metric` {"metric": "headcount", "aggregation": "min", "group_by": "department", "period": "last_fiscal_year"} | blind note: aggregate_hr(entity=headcount)も候補 | **keep**：従業員数はaggregate_metricの指標にあり、正解が自然 | |
| case_1028 | 仕入額を今月と先月で比較して | `compare_period_metric` {"metric": "purchase_amount", "period_a": "this_month", "period_b": "last_month"} | blind note: period_a/period_bの順序は文中の語順に従った(逆順も可) | **keep**：期間の順序は語順どおりで、検査も同じ答え | |
| case_1116 | 今年度の在庫で「移動中」の記録を出して | `search_inventory_record` {"entity": "stock", "period": "this_fiscal_year", "keyword": "移動中"} | blind note: 「在庫」でstockとしたが、移動中の記録ならmovementの可能性も | **keep**：「在庫」と明記されているのでstockで妥当。検査も同じ答え | |
| case_1119 | 部門コードD-1635、購買課の登録情報を出して | `search_master` {"entity": "department", "code": "D-1635", "name": "購買課"} | blind note: コードと名称の両方が書かれている。codeのみ指定も可 | **keep**：コードと名称の両方が書かれていて、正解も両方を持つ | |
| case_1121 | 顧客の出荷先住所を教えて | `search_master` {"entity": "customer", "field": "shipping_address"} | writer note: コードも名称も指定がなく、どの顧客か特定できない<br>blind note: どの顧客か(コード/名称)が書かれていない。list_master_recordsやnullも考えられる | **drop**：どの顧客か特定できない。正解の呼び出しに識別子が無い | |
| case_1128 | 2019年度の法令の記載を調べて | `search_report` {"report_type": "law", "fiscal_year": 2019} | writer note: 年度と文書種別だけで何を調べるか曖昧<br>blind note: 何を調べるか不明。list_report_documents/search_report_sectionも候補 | **drop**：何を調べるかが無く、検索として成り立たない | |
| case_1129 | 2025年度の有価証券報告書の記載内容を調べて | `search_report` {"report_type": "yuho", "fiscal_year": 2025} | writer note: 年度と文書種別だけで何を調べるか曖昧<br>blind note: 何を調べるか(指標/セクション)不明。search_report_sectionも候補 | **drop**：同上 | |
| case_1146 | 先月の倉庫の件数の推移 | `trend_inventory` {"entity": "warehouse", "metric": "count", "period": "last_month"} | writer note: 倉庫の件数推移は意味が曖昧 | **drop**：倉庫の件数の推移は意味が曖昧 | |
| case_1153 | 有報の平均年齢、2023年度から2024年度まで各年度の値を並べて | `trend_report_metric` {"report_type": "yuho", "metric": "average_age", "year_from": 2023, "year_to": 2024} | blind note: 隣接2年度なのでcompare_report_metricとも区別しにくい。「各年度の値を並べて」で系列と判断 | **keep**：「各年度の値を並べて」で推移と読める。検査も同じ答え | |

## 2. 無作為に選んだ件（検査では問題なし）

クエリを読んで、正解の呼び出しが1つに決まるか、正解に無い条件が書かれていないかを見てください。

| ID | クエリ | 正解の呼び出し | 判断 |
|---|---|---|---|
| case_1001 | 前年度の勘定残高の合計を出して | `aggregate_accounting` {"entity": "account_balance", "aggregation": "sum", "period": "last_fiscal_year"} | |
| case_1004 | 前四半期の勘定残高の合計を部門別に出して | `aggregate_accounting` {"entity": "account_balance", "aggregation": "sum", "group_by": "department", "period": "last_quarter"} | |
| case_1005 | 金額じゃなくて、今月の仕訳が何件あるかだけ知りたい | `aggregate_accounting` {"entity": "journal", "aggregation": "count", "period": "this_month"} | |
| case_1006 | 前年度の給与の最小値を出して | `aggregate_hr` {"entity": "payroll", "aggregation": "min", "period": "last_fiscal_year"} | |
| case_1009 | 今年度の給与の平均を出してください | `aggregate_hr` {"entity": "payroll", "aggregation": "avg", "period": "this_fiscal_year"} | |
| case_1011 | 前年度の在庫移動の合計を出して | `aggregate_inventory` {"entity": "movement", "aggregation": "sum", "period": "last_fiscal_year"} | |
| case_1020 | 入荷の件数を部門別に数えて | `aggregate_purchase` {"entity": "receipt", "aggregation": "count", "group_by": "department"} | |
| case_1021 | 仕入先からの請求書、一覧じゃなくて最大値を知りたい | `aggregate_purchase` {"entity": "supplier_invoice", "aggregation": "max"} | |
| case_1024 | 請求書の平均を部門別に見せて | `aggregate_sales` {"entity": "invoice", "aggregation": "avg", "group_by": "department"} | |
| case_1025 | 個別の請求書じゃなくて平均を出して | `aggregate_sales` {"entity": "invoice", "aggregation": "avg"} | |
| case_1027 | 請求書の件数を顧客別に出して | `aggregate_sales` {"entity": "invoice", "aggregation": "count", "group_by": "customer"} | |
| case_1037 | テーブル全体をCSVにして | `export_dataset` {"format": "csv", "scope": "full_table"} | |
| case_1040 | 一覧ではなく、原価センタCC-2025-19474のこの1件を詳しく | `get_accounting_detail` {"entity": "cost_center", "document_no": "CC-2025-19474"} | |
| case_1053 | 倉庫WH-2022-90189の詳細を明細行まで見せて | `get_inventory_detail` {"entity": "warehouse", "document_no": "WH-2022-90189", "include_lines": true} | |
| case_1056 | 従業員E-4560の変更履歴を教えて | `get_master_change_history` {"entity": "employee", "code": "E-4560"} | |
| case_1057 | 従業員E-8900の今年度の変更履歴を見せて | `get_master_change_history` {"entity": "employee", "code": "E-8900", "period": "this_fiscal_year"} | |
| case_1059 | 品目分類の階層構造を見せて | `get_master_hierarchy` {"entity": "item"} | |
| case_1060 | 品目の一覧じゃなくてツリーで。ITM-21403を起点に2階層分 | `get_master_hierarchy` {"entity": "item", "root_code": "ITM-21403", "depth": 2} | |
| case_1061 | 仕入先からの請求書SI-2023-69763を明細も含めて見せて | `get_purchase_detail` {"entity": "supplier_invoice", "document_no": "SI-2023-69763", "include_lines": true} | |
| case_1065 | 請求書INV-2020-69620について、一覧ではなくこの1件を。明細行は要らない | `get_sales_detail` {"entity": "invoice", "document_no": "INV-2020-69620", "include_lines": false} | |
| case_1067 | 出荷SHP-2019-21150の詳細をください | `get_sales_detail` {"entity": "shipment", "document_no": "SHP-2019-21150"} | |
| case_1070 | 無効なものは除いて、「樹脂」に関する部門を一覧にして | `list_master_records` {"entity": "department", "active_only": true, "keyword": "樹脂"} | |
| case_1071 | 人員の未処理分を、期限切れかどうかは問わず全部出して | `list_open_hr_items` {"entity": "headcount", "overdue_only": false} | |
| case_1079 | 今年度の未処理の入荷を、期限超過に限らず全部見せて | `list_open_purchase_items` {"entity": "receipt", "overdue_only": false, "period": "this_fiscal_year"} | |
| case_1080 | 前年度の発注で滞留しているもの。期限切れだけに絞らなくていい | `list_open_purchase_items` {"entity": "purchase_order", "overdue_only": false, "period": "last_fiscal_year"} | |
| case_1082 | 今月の受注で未処理のものを出して | `list_open_sales_items` {"entity": "sales_order", "period": "this_month"} | |
| case_1100 | 文書の中身じゃなくて、例規で「情報公開」に関する文書がどれかを知りたい | `list_report_documents` {"report_type": "ordinance", "keyword": "情報公開"} | |
| case_1108 | 前四半期の仕訳で「為替差損」に関する記録を出して | `search_accounting_record` {"entity": "journal", "period": "last_quarter", "keyword": "為替差損"} | |
| case_1111 | 給与のレコードから「契約社員」を探して | `search_hr_record` {"entity": "payroll", "keyword": "契約社員"} | |
| case_1112 | 今年度の勤怠で「契約社員」に関するものを一覧にして | `search_hr_record` {"entity": "attendance", "period": "this_fiscal_year", "keyword": "契約社員"} | |
| case_1122 | 入荷の記録で「原材料A」に関するものを探して | `search_purchase_record` {"entity": "receipt", "keyword": "原材料A"} | |
| case_1126 | 先月の発注で下書きのものを一覧にして | `search_purchase_record` {"entity": "purchase_order", "period": "last_month", "status": "draft"} | |
| case_1130 | 指標の数値じゃなくて、有報のコーポレート・ガバナンスの記述そのものを出して | `search_report_section` {"report_type": "yuho", "section": "corporate_governance"} | |
| case_1138 | 今月の給与の件数推移を日別で見せて | `trend_hr` {"entity": "payroll", "metric": "count", "interval": "day", "period": "this_month"} | |
| case_1139 | 給与の件数の推移を四半期ごとに出して | `trend_hr` {"entity": "payroll", "metric": "count", "interval": "quarter"} | |
| case_1143 | 今月の在庫の数量の推移を見せて | `trend_inventory` {"entity": "stock", "metric": "quantity", "period": "this_month"} | |
| case_1151 | 前年度の仕入先からの請求書の件数の推移 | `trend_purchase` {"entity": "supplier_invoice", "metric": "count", "period": "last_fiscal_year"} | |
| case_1154 | 有価証券報告書の平均年齢の推移を、2021年度から2025年度まで | `trend_report_metric` {"report_type": "yuho", "metric": "average_age", "year_from": 2021, "year_to": 2025} | |
| case_1155 | 2点の差分じゃなくて系列で。有報の平均年齢、2018年度から2020年度 | `trend_report_metric` {"report_type": "yuho", "metric": "average_age", "year_from": 2018, "year_to": 2020} | |
| case_1159 | この発注を確定して | （どのツールも呼ばない） | |
| case_1164 | 仕入先S-4623をマスタから削除して | （どのツールも呼ばない） | |
| case_1165 | 提出済みの勤怠申請を差し戻して | （どのツールも呼ばない） | |
| case_1170 | 最近寒くなってきましたね | （どのツールも呼ばない） | |
| case_1171 | お昼なに食べようかな | （どのツールも呼ばない） | |
| case_1172 | よろしくお願いします | （どのツールも呼ばない） | |
| case_1173 | 減価償却って何？ | （どのツールも呼ばない） | |
| case_1174 | インボイス制度の概要を教えて | （どのツールも呼ばない） | |
| case_1176 | 在庫回転率の計算式を教えてください | （どのツールも呼ばない） | |
| case_1179 | 来月の売上を予測して | （どのツールも呼ばない） | |
| case_1184 | 取引先の外部信用格付けを一覧にして | （どのツールも呼ばない） | |
