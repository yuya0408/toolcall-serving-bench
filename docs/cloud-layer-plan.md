# クラウド層の設計方針（計画中）

**現状：ここに書かれているものは設計方針であり、実装はまだ着手していない。**
実装が進むごとに本ドキュメントと README を更新する。

検討過程の生メモはこのリポジトリの外（個人の作業ノート）で管理しており、
雇用関係・社内IPに関する記述を含むためここには置いていない。
本ドキュメントは、そこから技術方針だけを抽出・整理したもの。

## なぜクラウド層を足すのか

`toolcall-serving-bench` は評価ハーネスとしての設計（分離採点・N/A規則・Adapter/Decoderの
責務境界・データ透明性ルール）は完成度が高いが、実行はローカル完結している。
IaC・CI/CD・マネージドLLM統合を同じリポジトリの上に重ねることで、
「評価設計」だけでなく「クラウド上で運用可能な形にする実装」までを一つの成果物として残す。

既存の設計判断（分離採点・x-scoring導出・Decoderの穴など）は変更しない。
この上に薄い層を足すだけ、という制約を維持する。

## スコープ（固定）

やる：Terraform / GitHub Actions / S3 / CloudWatch / Bedrock adapter / 二段ゲートCI
やらない：Kubernetes、VPC設計、dev/prod分離。完走を優先してスコープを固定する。

## 1. インフラ（Terraform）

- インフラ一式をコード化し、`destroy → apply` で再現できることを確認する
- 評価結果JSONを **S3** に蓄積し、時系列で追えるようにする
- **CloudWatch** でログとコスト監視、**AWS Budgets のアラートを初日に設定**

## 2. Bedrock を ModelAdapter の1つとして結線

既存の `ModelAdapter` 抽象（`eval_bench/adapters/base.py`）に Bedrock backend を
1つ追加するだけで、設計そのものは変更しない。

これにより「クラウドのマネージドLLM vs ローカル小型モデル」を同じ採点基準・同じ
ツールセットで比較する軸が加わる。ツール数の劣化曲線（`schema/tools.json` の
`tool_set_stages`：5 / 10 / 20 / 40）に対して、上限側の基準点ができる。

## 3. 二段ゲートCI（GitHub Actions）

PRとNightlyでゲートの内容を分ける。PR実行はスモーク15ケース×n=1（§5）で
consistencyは定義できないため、consistencyの絶対下限はNightlyでのみ評価する。

```
PR時（selection.floor + regression）
  1. 絶対下限   selection >= 0.85
  2. 回帰許容   Δselection >= -0.05   （main の baseline 比）

Nightly（上記に加えてconsistency）
  3. 絶対下限   consistency >= 0.90
```

- 許容幅を広めに取るのは、LLM の非決定性によるノイズを回帰と誤判定させないため。
  しきい値の根拠は [ADR-0001](adr/0001-ci-regression-threshold.md) に記録する。
- `forbidden_tools` の呼び出しは減点でなく **即fail**（安全性に関わるため許容幅の対象外、PR/Nightly両方）
- semantic judge を含む LLM-as-Judge は CI に入れない（既存方針と整合）。
  埋め込み judge は決定的なので CI 可
- baseline は `eval/baselines/main.json` に置く

## 4. GTのバージョニングと抜け道封じ

`manifest.json` に GT のバージョンとハッシュを持たせ、変わったら回帰ゲートを
自動スキップして警告を出す。GTが変われば baseline 比較は無意味なので、
明示的な baseline 更新を要求する。

これがないと「数値が落ちたら GT をいじって通す」という抜け道が開く。

## 5. 実行の分割

全件を PR で回すとコストと待ち時間で破綻する。

| トリガー | 内容 |
|---|---|
| PR時 | スモーク15ケース × n=1（数分） |
| Nightly | 全ケース × 反復あり（consistency はこちらで測る） |

反復・シャッフルによるconsistency算出そのもの（`eval_bench/repetition.py`）は実装済み。
未実装なのはこれをPR/Nightlyのトリガーに載せて自動実行する部分（GitHub Actions側）。

## 6. 実験1本 = PR 1本

`configs/base.yaml` を基準とし、差分だけを `experiments/` に置く One-at-a-Time。
GitHub の PR 履歴がそのまま実験ログになることを狙う。何を試し、何が落ち、
なぜ棄却したかが URL 付きで残る。

## ロードマップ

| 時期 | 内容 |
|---|---|
| 2026-08 | WSL2 + Docker + AWS CLI 環境構築、IAM、Budgets、コンテナ1個を手動デプロイして疎通 |
| 2026-09 | Terraform化（手で作ったものを一度消してコードから再構築する）、GitHub Actionsでハーネス実行 |
| 2026-10 | Bedrock adapter結線、二段ゲート実装、S3蓄積 |
| 2026-11〜12 | GT本番セットの拡充、段階セット（5/10/20/40）での劣化曲線の実測 |

## 変更しないもの

分離採点・N/A規則・semantic judgeの校正方針・Adapter/Decoderの責務境界・
x-scoring導出・データ透明性ルール・GT3層構成は、この計画によって一切変更しない。
