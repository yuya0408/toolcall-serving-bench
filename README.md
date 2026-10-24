# toolcall-serving-bench

同じモデル（Qwen3-8B）を3つの推論エンジン（transformers・llama.cpp・vLLM）に載せ、**ツール呼び出し（function calling）の精度と
レイテンシがエンジンと並行数でどう変わるか**を、同じGPUインスタンス上で測る評価ハーネス。問いは2つある。

1. この並行数で、本番のレイテンシ基準を満たせるのはどのエンジンか
2. 並行数16で提供できる構成のうち、負荷の下でも精度を保つのはどれか

評価セットは、汎用ERPのツール群に対する日本語の200件を自前で作った（後述）。

**進み具合（2026-10-24）**：疎通とパイロット（本番の件数などを決めるための予備の実行）を済ませ、本番のセル一覧を固定した（[ADR-0007](docs/adr/0007-main-run-layout.md)）。
次は本番で、**結果はまだ無い**。

## 比べるもの

| 層 | エンジン | チェックポイント | 量子化 |
| --- | --- | --- | --- |
| 1：エンジンの比較 | `transformers_hf` | `Qwen/Qwen3-8B` | bf16 |
| 1：エンジンの比較 | `llama_cpp` | `Qwen/Qwen3-8B`から自前で変換したGGUF | bf16（GGUFへの詰め替えのみ） |
| 1：エンジンの比較 | `vllm` | `Qwen/Qwen3-8B` | bf16 |
| 2：量子化 | `vllm` | `Qwen/Qwen3-8B-AWQ`（公式） | AWQ 4bit |

- **層を2つに分ける理由**：GGUFのQ4_K_MとAWQは別のアルゴリズムで、4bit同士で比べると、差がエンジンのせいか量子化のせいか区別できない。
  そこでエンジンの比較はbf16にそろえ、量子化はvLLMの中だけで比べる。
- **ハードウェア**：全エンジンを同じ`g6.xlarge`（NVIDIA L4、ap-northeast-1）で動かす。エンジンごとにハードウェアが変わると、差の原因を言えなくなるため。
  インスタンスはTerraformで立て、測り終えたら`destroy`する（常時起動はしない）。
- **エンジン以外の条件をそろえる仕組み**
  - どのセルも、エンジンのサーバーを起動し直してから回す（llama.cppにはプロンプトのキャッシュを全部消すAPIが無いため）
  - 並行数が2以上のセルは、回す前にKVキャッシュに全リクエストが入るかを全エンジン共通の式で確かめ、回した後に中断や打ち切りが無かったかを確かめる
    （[ADR-0005](docs/adr/0005-capacity-check.md)）。入らない構成は「対応していない」と記録し、設定を上げて載せることはしない
  - 実行ごとにrun manifest（エンジンのバージョン、起動コマンド、ハードウェア、サンプリング設定、容量の判定）を残し、
    schema（`schema/run_manifest.schema.json`）で検証する。生出力はmanifestより先に書くので、インスタンスを消した後でも採点し直せる

## 判定のしかた

| パス | 何を見るか | 判定 |
| --- | --- | --- |
| 精度（accuracy） | 並行数1で、各エンジンの分離採点（下記）。提示順を変えて5試行 | 判定はしない。構成ごとの値を、ケース単位のブートストラップCIつきで報告する（問い2の比較の土台。構成の差はエンジンと量子化が混ざるため、優劣は結論にしない） |
| 負荷の下の精度（第3パス） | 並行数1と16で、greedyの答えが変わるか。並行数16に対応できる構成すべてで行う | 差のCIが±2.6ptに収まれば「等価」（TOST型）。2.6ptはQwen3-8Bと4BのBFCLの差、つまりモデル1段階分（[ADR-0002](docs/adr/0002-equivalence-margin.md)） |
| レイテンシ（latency） | TTFTとTPOT。並行数1/2/4/8から倍々に、満たさなくなるか対応できなくなるまで（上限32） | MLPerf InferenceのLlama 3.1 8B向け基準を、超過率1%以下（p99）で満たすかの3値（満たす／満たさない／判定不能）。Clopper-Pearsonの片側95%限界で判定する（[ADR-0003](docs/adr/0003-latency-pass-verdict.md)） |

精度のパスとレイテンシのパスは別々に実行する。精度を測る負荷がレイテンシを歪めないようにするためである。

## 評価セットと採点

### 分離採点

ツール呼び出しの能力を1つの成功率にまとめず、質の違う3つを別々に測る。

- **selection** — 正しいツールを選べるか（呼ぶべきでない場面で呼ばない、も含む）
- **arg_exact** — enum・数値・日付など、完全一致で判定すべき引数を埋められるか
- **arg_semantic** — 自由文の引数を、表記揺れを許して意味的に埋められるか

`e2e`（全部正解）は派生指標として足すだけで、一次指標にはしない。ツールの選択を外した事例では引数の指標をN/Aにし、
選択の失敗で引数を取り出す能力まで巻き添えにしない。

### ツールの数と紛らわしさ

ツール選択の正確さは、提示するツールの数と紛らわしさに強く左右される。3個から選ぶときの「選択95%」と、
似た機能のツールが並ぶ40個から選ぶときの「選択95%」は別物である。そこで次のように作っている。

- ツールは汎用ERPの標準モジュール（マスタ・販売・購買・在庫・会計・人事・帳票）の40個。各モジュールに、機能の近いツール
  （検索・明細取得・集計・推移・未処理一覧）を意図的に置く。拡張カタログ（`schema/tools_extended.json`）は100個まで。
- 提示するツールの数（`tool_stage`）は実験の変数にできる。正解のツール、事例が宣言した紛らわしいツール、同じモジュールの近いツール、
  ほかのモジュールのツールの順で埋め、固定シードで並べ替える。宣言した紛らわしいツールが入りきらない場合は、黙って落とさずエラーにする。
  エンジンの比較は`tool_stage=10`で行う。
- **想定する構成**：カタログを全部渡さず、クエリごとに10個に絞って渡す構成（ツールを検索して絞り込む、いわゆるTool RAG）を想定する。
  40個を全部渡すとプロンプトは約8,000トークン（10個の約3.5倍）になり、紛らわしいツールが並ぶほど選択の精度も落ちるためである。
  理想化している点が3つある：正解のツールが必ず入っている（検索が当たった後の選び分けを測る）、並び順はランダム（5試行で平均する）、10個に固定。
  本物の検索は走らせていない。この構成では、ツール定義の部分がリクエストごとに変わるので、速さの測定でprefix cacheはほぼ効かない（実測2〜9%）。
  ツール一式が固定でキャッシュが効く構成は範囲外（[ADR-0003](docs/adr/0003-latency-pass-verdict.md)の追記2026-10-24）。
- 同じ事例で、提示するツールの構成は固定したまま並び順だけを変えて5試行する（`eval_bench/repetition.py`）。並び順の運を平均で消し、
  5回とも同じ答えだったかを**consistency**として数える。

### 200件の作り方とデータの透明性

日本語のツール呼び出しの評価セットには、Nejumi LLMリーダーボードのBFCL日本語版があるが、英語の汎用APIを翻訳したもので、
単発のカテゴリは各30件しかない。日本語の業務領域で、提示するツールの数を変えて測れる評価セットは見当たらなかったので自作した。

1. 事例ごとに`provenance`（`seed_manual`／`grounded`／`adversarial_manual`／`generated`）を持ち、どの作り方の事例を除いても数値を計算し直せる。
2. `generated`（186件）は、**正解をスクリプトが規則と固定シードで作り**、LLMはクエリ文だけを書く。正解を見せない別のモデルの検査と、
   全件の人の確認を通している。人が直した件は`generation.human_edit`に残す（[ADR-0004](docs/adr/0004-expand-case-set.md)）。
3. `grounded`（4件）の引数の正解は、有価証券報告書の公開データから取る（`source`必須）。LLMはクエリ文の言い換えだけを担う。
4. クエリの言い換えに使ったモデルは、評価の対象から外す。

内訳は200件（negative 30件）で、`generated` 186件・`seed_manual` 5件・`adversarial_manual` 5件・`grounded` 4件。
件数は、第3パスで±2.6ptの等価性を言うのに要る約140件を超えるように決めた。

## 使い方

手元（GPU不要）：

```bash
pip install -r requirements.txt pytest
python -m pytest -q                                          # PRごとにCIでも回る
python scripts/count_prompt_tokens.py --tool-stage 10 --write-budget   # ケースかカタログを変えたら数え直す
```

1セルの実行（エンジンのサーバーが立っている前提）：

```bash
python scripts/run_cell.py --pass-type accuracy --engine vllm \
    --base-url http://localhost:8000 --model Qwen/Qwen3-8B --quantization bf16-none \
    --engine-version 0.x.y --tool-stage 10
```

GPUインスタンスでの実行（apply → 準備 → セルの一覧を回す → 回収 → destroy）は、[docs/gpu-runbook.md](docs/gpu-runbook.md)の手順で行う。
手で打つのは`scripts/gpu/local.sh`のコマンドだけで、セルの一覧は`scripts/gpu/cells/*.json`にある。

## リポジトリの構成

| 場所 | 中身 |
| --- | --- |
| `data/` | 評価セット（provenanceごとのディレクトリ） |
| `schema/` | 事例・run manifestのschema、ツールカタログ、プロンプトの最大長（`prompt_budget.json`） |
| `eval_bench/dataset/` | 事例の読み込みとschemaの検証、ツールの段階セットの組み立て |
| `eval_bench/adapters/` | エンジンごとの呼び出しの違いを吸収する（OpenAI互換API・transformers） |
| `eval_bench/decoders/` | 生出力 →（ツール, 引数）。制約デコーディングを足すときの差し替え点（今は恒等） |
| `eval_bench/scorers/` | 分離採点 |
| `eval_bench/` 直下 | 試行の繰り返しとCI（`repetition.py`）、レイテンシ（`latency.py`）、容量の判定（`capacity.py`）、セルの実行（`runner.py`） |
| `scripts/` | セルの実行（`run_cell.py`）、評価セットの生成と検査、プロンプトの長さの計測 |
| `scripts/gpu/` | GPUインスタンスでの準備とセルの一覧の実行 |
| `infra/` | Terraform（`infra/gpu/`がGPUインスタンス） |
| `data_build/` | 評価セットを作ったときの中間データと検査の記録 |

## 判断の記録

数字と方針の根拠は、却下した案とともにADRに残している。

| ADR | 内容 |
| --- | --- |
| [0001](docs/adr/0001-ci-regression-threshold.md) | CIの回帰ゲートのしきい値 |
| [0002](docs/adr/0002-equivalence-margin.md) | 第3パスの等価性マージンを±2.6pt（モデル1段階分の差）に固定する |
| [0003](docs/adr/0003-latency-pass-verdict.md) | latencyパスは、MLPerfの8B向け基準を満たすかの3値判定にする |
| [0004](docs/adr/0004-expand-case-set.md) | 評価セットを200件に増やす（正解は機械で作り、クエリはLLMが書く） |
| [0005](docs/adr/0005-capacity-check.md) | 並行数のセルが成り立つかを、全エンジン共通の容量の判定で確かめる（Proposed：GPUで動作を確かめてからAccepted） |

計画の全体は[docs/engine-performance-plan.md](docs/engine-performance-plan.md)、1回目のapplyの記録は
[docs/gpu-smoke-2026-10-18.md](docs/gpu-smoke-2026-10-18.md)にある。
