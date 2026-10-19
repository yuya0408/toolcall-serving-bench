# エンジン性能観測層の設計方針（計画中）

**現状：ここに書かれているものは設計方針であり、実装はまだ着手していない。**
実装が進むごとに本ドキュメントと README を更新する。

このドキュメントは [`cloud-layer-plan.md`](cloud-layer-plan.md) とは別軸の計画。
クラウド層がハーネスの実行環境（IaC/CI/S3）を作る計画なのに対し、こちらは
ハーネスが比較する**実験変数そのもの**を増やす計画で、両者は独立に進められる。

## なぜこの層を足すのか

`toolcall-serving-bench` は元々「モデル横断 × 素の few-shot」（記事1のスコープ）に閉じていたが、
既存の設計（`schema/tools.json` のツール数段階セット、`repetition.py` の
seed固定シャッフル×n=5×ブートストラップCI、差し替えコストの非対称性という層構成）は、
ツール数以外の実験変数にもそのまま転用できる骨格になっている。

`eval_bench/adapters/base.py` のdocstringも当初から
"Add one subclass per backend (Ollama, PrismML, cloud)" とバックエンドの差し替えを
想定していた（`PrismML` は過去セッションのプレースホルダー名で、実装対象として
確定しているものではない）。

今回追加するのは **engine（推論エンジン/配信スタック）を制御変数にする軸**。
副次的な目的として、LLMOpsで実務に活かせるスキル（複数の推論エンジンの構成・
比較・コスト管理を規律立てて行う経験）を成果物として残すことも兼ねる。

## スコープ（固定）

やる：
- engine を実験変数にした精度スイープ（`selection`/`arg_exact`/`arg_semantic`、
  既存の分離採点・N/A規則をそのまま使う）
- レイテンシ/スループットの計測（精度スイープとは別のシリアライズされたパス）
- **全エンジンを同一クラウドGPUインスタンス上で比較する**（後述の通り、ハードウェアが
  エンジンごとに変わると差分が帰属不能になるため）
- 比較可能性を守るための run manifest（後述）

やらない：
- 精度スイープとレイテンシ計測を同一実行で兼ねること（負荷生成が計測を歪めるため）
- フルグリッド（model × engine × tool段階 × n の総当たり）をデフォルトの実行単位にすること
- 常時起動のGPUインフラ（`apply→run→destroy` の使い捨て運用のみ）

## セル定義：(engine, model) を制御変数にする

比較の単位を「(model)ごとに、たまたま動くbackendで測る」ではなく、
**同じ重み（同一quantization）を複数engineに同一構成で載せて比較する**、
に決めた。「同じモデルでも配信スタックが変わると精度・速度が変わるか」という
問いに答えるのが目的で、既存のモデル横断比較（[[article1_outline|記事1]]）の数字はそのまま維持しつつ、
別テーブルとして加法的に積む。

### ハードウェアは全エンジン共通のクラウドGPUインスタンス1台に揃える（2026-08-29決定）

作業機のGPUはAMD Radeon 610M（統合GPU、2GB、CUDA/ROCm対象外）で、当初は
「vLLMだけクラウドGPU、他はローカルCPU」という案を検討した。しかしこれだと
**ハードウェアの違いがエンジンの違いと交絡し、精度差・速度差のどちらも
「エンジンのせいかハードウェアのせいか」言えなくなる**ため却下した。

全engineを**同一のクラウドGPUインスタンス上**で動かし、比較可能性の規律
（後述）を効かせる。ローカルハードウェアは今回の(engine, model)比較には使わない。

### エンジンロースター確定（2026-08-29）

**コア3本**（この3本で「素の実装 → 単一リクエスト最適化 → マルチユーザー
継続バッチング」という一貫したストーリーが作れる）：

1. **`transformers_hf`** — HF transformersの素の推論。特別なサービング最適化
   なしのナイーブベースライン。vLLM/llama.cppが何を買っているかの基準点
2. **`llama_cpp`** — llama.cpp server（GGUF量子化、C++コア）。軽量・単一
   リクエスト志向の実装
3. **`vllm`** — continuous batching型のプロダクション級サービングエンジン。
   この計画全体の主目的（マルチユーザー同時アクセス前提の実務スキル証明）

**Ollamaはコアから除外、任意の追加枠に格下げ**：llama.cppを内部エンジンとして
ラップしているため、単独で比較しても「エンジンの違い」というより「配信レイヤー
（APIの形・チャットテンプレート適用・モデル常駐管理）の違い」にしかならない
可能性が高い。予算に余裕があれば4本目として追加し、その場合はこの限界を
結果に明記する（`llama_cpp`とのペアはnear-nullになりうる前提で報告する）。

コアを3本に絞ったことで、当初の「4engine想定」より必要工数がやや減り、
コスト見積り（後述）にprefix caching検証のバッファを持たせられる。
[Source: User, エンジンロースター確定の壁打ち, 2026-08-29]

### モデルサイズと具体的なインスタンス（2026-08-29決定）

対象モデルは **8Bクラス** に決めた。ユーザーの実務は現在Qwen/Gemmaの20-30B級が
中心だが、20-30B級はg6e.xlarge（L40S 44GB, $2.699/hr）クラスが必須になり
予算が3倍程度に膨らむ一方、この計画で証明したいのは「特定のモデルサイズを
扱った実績」ではなく**比較可能性の規律・run manifestによる帰属可能性・
vLLMのcontinuous batchingを正しく評価できる設計**という方法論であり、
モデルサイズには依存しない。8Bクラスで安く反復回数を確保する方を優先し、
成果物では「手法自体は20-30B級の本番規模にそのまま適用可能」と明記する。

- **インスタンス**: g6.xlarge（NVIDIA L4, 22GB VRAM）、ap-northeast-1オンデマンド
  **$1.1672/hr**（2026-08-29、WSL2側`aws pricing get-products`で一次情報確認済み。
  SKU `ZEHXGXZKA86RDSNK`、価格改定日2026-08-01。Web検索での$1.167/hrとほぼ一致）
- **VRAM試算（2026-08-30改訂、レイヤー1がbf16基準になったため）**：8Bクラスbf16で重み約16GB、
  L4 22GBだと残り約5-6GBがKVキャッシュ＋バッチング余地。ラフ計算（GQA前提、tool_stage=100の
  長いプロンプトを想定）では同時実行数十程度までは収まりそうだが、実際のトークン数依存なので
  GPUインスタンス起動時に要検証（収まらない場合はモデルクラスを3-4Bに落とすことも検討）。
  レイヤー2（vLLM単体、AWQ 4bit）はこの制約が緩む

## 比較可能性の規律

同じ重みでもエンジンによって量子化フォーマット対応・チャットテンプレート適用・
サンプリング実装が異なりうる。ここがズレると精度差もレイテンシ差も
「何由来か言えない」数字になり、計測そのものが無意味になる。

- quantization形式・サンプリングパラメータをセル単位で明示的に固定する
- 揃えられないengineが出た場合は、こっそり調整せずそのセルを除外する
- **chat/tools APIを使う（2026-09-10改訂。2026-08-30時点では逆に「使わない」だったが転換した）**：
  当初は3engineとも生の補完エンドポイント（vLLM `/v1/completions`、`llama_cpp` `/completion`、
  `transformers_hf`は直接`generate`）にharness側で組み立てた生プロンプト文字列を渡す設計だった。
  理由はテンプレート実装差がprefix cachingのヒット判定（先頭からの完全一致）を汚染すること、
  および`prompt_layout`の`query_first`（クエリを先頭に置く）を実現するためだった。

  しかし2026-09-10のユーザー指摘で再検討した結果、転換した。決め手は**第3パス
  （並行負荷下の精度維持）が測りたい信号の質**：生の補完APIでは、モデルにJSON形式で
  応答させる手段がfew-shotによるパターン継続しかなく、出力形式の崩れ（パース失敗）が
  「選択ミス」と同じ`tool=None`として計上され、両者が区別できない。chat/tools API
  （vLLM/`llama_cpp`はOpenAI互換のchat補完エンドポイント、`transformers_hf`は
  `apply_chat_template(..., tools=...)` + `generate`）を使えば、vLLM/`llama_cpp`は
  構造化された`tool_calls`をそのまま返すため、この形式崩れという交絡ノイズがほぼ消え、
  「並行化が精度を落とすか」という測りたい信号がクリアになる。副次的に、本番で実際に
  使われる呼び出し方（あるモデルをあるエンジンにデプロイしたら実際にどう叩くか）とも
  一致する、という利点もある。

  この転換に伴うトレードオフは2点：
  1. `prompt_layout=query_first`はチャットテンプレートの構造上表現不可能なため
     スコープ外に格下げ（詳細は後述「prompt layout」節参照）
  2. `transformers_hf`は`apply_chat_template`を通してもモデル自身はHermes形式
     （`<tool_call>{...}</tool_call>`というタグ付きテキストブロック）で応答するため、
     vLLM/`llama_cpp`のように構造化`tool_calls`が自動で返らない。この1エンジンだけ
     Decoder側でHermesタグのパースが必要になる（vLLM/`llama_cpp`はサーバー内蔵の
     `--tool-call-parser hermes`相当の処理が構造化レスポンスとして返してくれるため不要）

  既存の「Adapterが出力を整形しない」方針（`eval_bench/adapters/base.py`）は維持される
  （Adapterは各engineのAPIが返した生のtool_calls/テキストをそのまま返し、パースは
  Decoderの責務のまま）。
- **メモリは「要件」を揃え、KVキャッシュの総量は揃えない（2026-10-19追加、[[docs/adr/0005-capacity-check|ADR-0005]]）**：
  全エンジンで揃えるのは、並行数と1リクエストの最大（このワークロードのプロンプトの最大＋`max_tokens`）。各エンジンは要件ぶんだけメモリを載せる
  （vLLMは既定の`gpu_memory_utilization`、llama.cppは`--ctx-size` ＝ 並行数 × 1リクエストの最大）。並行数が2以上のセルは、
  全リクエストが最大の長さまで伸びても入りきるか（保証型）を全エンジン同じ式で事前に確かめ、入らなければ「対応していない」とする
- **`engine_version`はコミットハッシュ単位で固定する（2026-08-30追加）**：特に`llama_cpp`は
  バージョンタグの粒度が粗く、カーネル・量子化まわりの挙動がコミット間で変わりうるため、
  run manifestの`engine_version`にはビルド元のコミットハッシュを記録する
- **結果JSONに run manifest を必ず持たせる**：スキーマを
  [`schema/run_manifest.schema.json`](../schema/run_manifest.schema.json) として確定済み。
  `cell`（engine/engine_version/model/quantization/sampling_params/tool_stage/
  prefix_caching/concurrency）と`hardware`（全セル共通のg6.xlarge固定値）を必須にし、
  `pass_type`（accuracy/latency）でresultsの形を`allOf`で分岐させる。後から条件を
  紐付け直すのは困難なため、S3蓄積フェーズ（[[cloud-layer-plan]]参照）より前に
  この形式を固めた

**実装前提として1点判明した既存ギャップ**：`eval_bench/repetition.py`の
`aggregate_repeated`は現状`selection_mean`/`selection_ci95`/`consistency`しか
集計しておらず、`arg_exact_mean`/`arg_semantic_mean`が未実装（`separated.py`の
`CaseScore`はcase単位で`arg_exact`/`arg_semantic`を持つが、集計側で捨てられている）。
run manifestのaccuracy resultsはこの2つを必須フィールドにしているため、
**この層の実装に着手する前に`aggregate_repeated`を拡張する必要がある**。

## 精度パスとレイテンシパスの分離

n=5 × 4段階の精度スイープ自体が負荷生成源になり、レイテンシ計測を歪める
（ウォームアップ・並行実行・熱の影響）。そのため：

- 精度パスとレイテンシパスは同じセルに対する**別々の実行**として設計する
- **精度パスは全engine`concurrency=1`（直列）固定とする（2026-08-30明記）**：
  `vllm`のcontinuous batchingはバッチ構成次第で数値がわずかに変わりうるため、精度パスで
  並行リクエストを許すと`consistency`（`RepeatedCaseResult.consistent`のunanimity判定）の
  差が「engineの違い」ではなく「バッチ構成というノイズ」由来になってしまう。`concurrency>1`は
  レイテンシパス専用の軸として扱う
- レイテンシパスはウォームアップを除外し、まず`concurrency=1`（直列）で
  帰属可能な素の値を取る。`concurrency>1`（同時リクエスト）は別結果として
  追加測定し、直列の値と混ぜて平均しない
- 使用ハードウェアを結果に明記する（run manifestの`hardware`）

## prompt layout をキャッシュ効果の軸として採用（2026-08-29改訂）

当初「engineのprefix caching機能をON/OFFするトグル」として設計したが、
2026-08-29に実際のデフォルト挙動を確認したところ棄却した：

- **vLLM**：V1エンジンでautomatic prefix cachingは**デフォルトON**
  （切るには`--no-enable-prefix-caching`が必要）
- **llama.cpp server**：`cache_prompt`の既定値は**true**（共通接頭辞は自動再利用）

本番で誰もOFFにしない設定をわざわざOFFにして比較しても実務的な意思決定には
直結しない。代わりに、**キャッシュはengine既定のON前提のまま、プロンプト内の
「安定部分（システムプロンプト＋ツールスキーマ＋few-shot）」と「可変部分
（クエリ）」の並び順**を軸にする（ユーザー提案、2026-08-29）。

prefix cachingはシーケンス先頭からの完全一致でしかヒットしないため、並び順が
ヒット率を直接左右する：

- **`stable_first`**：安定部分 → クエリの順。クエリだけがリクエストごとに
  変わり、それより前の長い共通部分はキャッシュヒットし続ける
- **`query_first`**：クエリ → 安定部分の順。先頭が毎回変わるため、後続の
  ツールスキーマ部分も含めて事実上毎回キャッシュミスになる

ツール数段階との掛け合わせで「ツール数が多いほど`query_first`の罰則が大きくなるか」
という、プロジェクトの核（ツール数を実験変数にする）と直結する問いに答えられる。

**`query_first`は2026-09-10時点でスコープ外**：上記「比較可能性の規律」節の通り
chat/tools APIに転換したため、`query_first`（クエリを安定部分より先に置く）は
実現不可能になった。chat templateは`system`（ツール定義を含む）→`user`（クエリ）
という役割の順序でレンダリングする構造上の契約を持っており、呼び出し側がこの順序を
入れ替える手段を持たない。`stable_first`はこの標準順序と一致するためchat API経由でも
そのまま再現できるが、`query_first`だけを実現するために生の補完APIに戻す（＝2条件を
別々の方法で作る）と、2条件間の差が「並び順の効果」なのか「作り方の違いの効果」なのか
区別できなくなり、この節が解決しようとした交絡と同種の問題を再導入してしまう。そのため
`query_first`のablationは行わず、`prompt_layout`は`stable_first`一択で運用する
（検討の経緯として本節・下記「ON/OFFトグルとの決定的な違い」節の記述は残す）。

### tool_stageの水準とツールカタログの設計（2026-08-29、2回の改訂を経て確定）

`eval_bench/dataset/toolsets.py`の`build_tool_set(case, tools, size, seed)`は
任意の`size`を受け付ける（制約は`size <= len(tools)`と`size >= 正解+宣言済み
distractor数`のみ）。`[5, 10, 20, 40]`は`default_stages()`のdeclared引数の
既定値にすぎず、ハードコードされた制約ではない。「5/10/20/40はArticle1のERP
現実感に基づく選択であり、Article2に最適とは限らない」（ユーザー指摘）は
技術的にも妥当。

**ERP語彙は必要（2026-08-29、一度「不要」と判断した後に再訂正）**：
`build_tool_set`が要求する唯一の妥当性要件（宣言済みdistractorを黙って落とさない）
は既存40ツール内で完結するため、新規60ツールに**近縁機能設計（confusability
エンジニアリング）までは不要**、という判断自体は変わらない。しかし
「難易度設計が不要」と「実在感（もっともらしさ）が不要」は別問題であり、
新規60ツールを意味のない汎用フィラーにすると、モデルが「不自然な名前の
ツールは無視してよい」という別のヒューリスティックを学習しうる。これは
実際のツール数増加が引き起こす難しさとは異なるアーティファクトで、
高段階の数値を歪める。

さらに外部調査（2026-08-29）でも裏付けが取れた：ICLR採択の**τ-bench**
（tool-agent-user interactionの実務ベンチマーク）はretail/airlineの
2ドメイン構成で、各ドメインは"realistic, modular databases and tools"
——1業務領域を深く掘り下げる設計であり、無関係ツールの寄せ集めではない。
BFCL・ToolBenchも同様にドメイン別カテゴリでスケールしている。実務の
大規模AIエージェントも「1つの深い業務領域に特化」が主流（customer service
automation、document analysisなど）。

結論：**新規60ツールもERPのまま**、既存7モジュールを深掘り（近縁機能を
積み増す）し、必要なら実在の大規模ERPにありそうな追加モジュール
（生産管理・プロジェクト管理等）を足す。confusabilityの精緻な設計までは
求めないが、実在感のある業務機能として作る（機械的なプレースホルダーには
しない）。

**既存ケースは新規作成せずそのまま流用する**（ユーザー指摘、2026-08-29）。
`build_tool_set`は不足分を同一モジュール優先→他モジュールで自動パディングする
設計のため、既存40ツール・既存ケース（`data/`配下）を一切変更せず、
カタログを100ツールに拡張しても既存ケースをそのまま流せる。

**accuracy/latency両passを10/50/100に統一する**：

- **段階10**：既存40ツールカタログの範囲内（拡張不要）。Article1の同じ段階と
  完全に同一の構成になるため、**Article1のstage=10の数値と直接重ねられる**
  （唯一残るオーバーレイ比較点）
- **段階50/100**：拡張後の100ツールカタログ（既存40＋新規60、ERP同一ドメインの
  追加業務機能）でpadding。新規データポイントとして、Article1の測定範囲を
  超えた大規模カタログ領域を新たに探索する

Article1の元ファイル・元ケース・5/10/20/40の4段階は一切変更しない
（拡張カタログは別バージョン/別ファイルの純粋な追加）。実装上の注意点として、
段階10の実行では**必ず元の40ツール辞書のみを`build_tool_set`に渡す**こと
（拡張後の100ツール辞書を渡すと、size=10でも他モジュールpaddingの候補プールが
変わり、Article1の元の結果と一致しなくなる）。

新規60ツールの具体的な生成は設計判断ではなくコンテンツ作成作業のため、
実装前の具体タスクとして下記「残る未決定事項」に切り出す。

**完了（2026-08-29、同日中）**：`scripts/generate_extended_catalog.py`で
`schema/tools_extended.json`（100ツール）を生成済み。既存ケース（`data/`配下
14件、report/sales/masterに集中——正例12件中10件、accounting/purchaseは
各1件、inventory/hrは0件）の分布をadvisor壁打ちで確認した上で、深掘り36
（同一モジュールpaddingが実際に効くreport/sales/masterに厚く配分）＋
新規2モジュール（production/project）24の内訳に決定（当初の
「新規4モジュール31／深掘り29」案は、ケースが1件も無いモジュールへの
深掘りも新規モジュール追加も"同一モジュールpadding非該当"という意味で
機能的に同じという指摘を受けて撤回）。`tests/test_extended_catalog.py`で
100件一意性・元40件の完全一致・新規モジュールがERPドメイン内・
全ケースのvalidate_case_against_tools通過・stage 50/100でのbuild_tool_set
非エラーを検証、`pytest tests/`で12件全通過。

**未解消の1点（今回は対応しない、実行系実装時に効かせる）**：
`build_tool_set`に拡張100ツール辞書を渡してstage=10を実行することを
コード上で防ぐガードは無く、現状コメント規律のみに依存している。
実行ランナー実装時にここへガードを入れること。

**ON/OFFトグルとの決定的な違い**：並び順は正解性に影響しうる
（position bias / "lost in the middle" 等の既知の現象）。ON/OFFは精度に
影響しない前提を置けたが、`prompt_layout`はその前提を置けないため、
**latency pass専用ではなくaccuracy passでも（範囲を絞って）検証する**：

- accuracy pass：`stable_first`を基準ラインとして全4段階で通常通り実行し、
  `query_first`は最大段階（40）でのみ追加実行して差を見る
  （one-factor-at-a-timeの規律を維持、フルグリッド化しない）
- latency pass：両layoutを主要段階で比較し、キャッシュヒット率の差を
  レイテンシに反映させる

`transformers_hf`はサービング層のキャッシュ機構を持たないため、両layoutで
レイテンシに有意差が出ないことが期待される想定（「キャッシュ機構がなければ
並び順を工夫しても速くならない」という対照点として機能する）。

**2026-09-10追記**：上記accuracy passの`query_first`ablation（当初は段階40、
schemaでは後に段階100に更新）は、chat/tools APIへの転換に伴いスコープ外になった
（詳細は上記「prompt layout をキャッシュ効果の軸として採用」節末尾の追記参照）。
本節の記述は検討の経緯として残すが、実行はしない。

## モデルチェックポイントと各engineのセットアップ（2026-08-29決定）

### モデルチェックポイント

**`Qwen/Qwen3-8B`**（denseモデル、ネイティブでtool-calling対応）に決定。

当初案のQwen2.5-7B-Instructはユーザー指摘（「もっと新しいモデルが存在すると思う」）を受けて
Web検索で再検討した。2026-08-29時点でQwenはQwen3.5/Qwen3.8系列まで進んでいるが、これらは
不採用：

- **Qwen3.5-9B**：GGUFは公開されているが、**vLLMのGGUFローダーが`qwen35`アーキテクチャに
  未対応**（[unsloth/Qwen3.5-35B-A3B-GGUF discussion #15](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/discussions/15)）。
  この計画の核である「quantization形式をengine間で揃える、揃えられなければそのセルを除外する」
  規律（[schema/run_manifest.schema.json](../schema/run_manifest.schema.json)の`quantization`）
  にいきなり抵触するリスクがある
- **Qwen3.8系列**：最小でも27B級（Qwen3.8-27B、[vLLM Blog 2026-08-12](https://vllm.ai/blog/2026-08-12-qwen3.8)）
  で、8Bクラスに絞った予算判断（上述）と矛盾する

Qwen3-8Bは、Qwen2.5より新しくtool-callingがネイティブ設計されている一方、GGUF/AWQ/safetensors
の3形式ともエコシステムが成熟しており（[Qwen公式llama.cppガイド](https://qwen.readthedocs.io/en/latest/quantization/llama.cpp.html)、
[Qwen公式vLLMガイド](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)）、この計画の
比較可能性の規律と両立する。

**dual-mode thinkingは`enable_thinking=False`で無効化する**（`/no_think`または
`apply_chat_template(..., enable_thinking=False)`）。理由：(1) ツール選択ベンチマークが
測りたいのは選択・引数抽出の精度であって推論過程の質ではない、(2) thinkingトークンが
レイテンシ計測に無関係なノイズを乗せる。non-thinkingモード時の推奨サンプリング値
（temperature=0.7, top_p=0.8, top_k=20, min_p=0）を`sampling_params`のベースラインにする。

### レイヤー構成（2026-08-30改訂）— quantization比較可能性のため2層に分離

当初「`llama_cpp`と`vllm`を4bit同士で比較する」としていたが、GGUF Q4_K_MとAWQ 4bitは
**別アルゴリズムで数値も別物**であり、この2つの精度差・速度差が「engineの違い」によるものか
「量子化アルゴリズムの違い」によるものか区別できない、という指摘（ユーザー+advisor、
2026-08-30）を受けて2層構成に変更する。

- **レイヤー1（engine比較、主軸）**：3engineとも**bf16無量子化**で統一。同一の重みを
  読ませることで、観測される精度差・速度差を「サービングエンジンの実装差」だけに帰属できる
- **レイヤー2（quantization探索、vLLM単体）**：`vllm`のみに絞り、bf16 vs 公式AWQ 4bitを比較。
  比較対象がvLLM単体（engineを跨がない）なので、「quantizationをengine間で揃える」規律の
  対象外になる。この計画の主目的（本番想定の配信スタックが量子化でどう変わるか）はここで示す

### engineごとのチェックポイント・quantization（改訂）

| レイヤー | engine | チェックポイント | quantization | 備考 |
|---|---|---|---|---|
| 1 | `transformers_hf` | `Qwen/Qwen3-8B` | `bf16-none` | |
| 1 | `llama_cpp` | `Qwen/Qwen3-8B`から自前変換（後述） | `gguf-bf16` | 量子化ではなくGGUFへの詰め替えのみ |
| 1 | `vllm` | `Qwen/Qwen3-8B` | `bf16-none` | quantized checkpointではなく素のsafetensorsを直接ロード |
| 2 | `vllm` | `Qwen/Qwen3-8B-AWQ`（公式prequantized） | `awq-4bit` | レイヤー1のvLLM bf16結果との対比。engine間比較ではない |

当初案の`llama_cpp`のQ4_K_Mはレイヤー1から外した。`llama_cpp`単体でbf16 vs Q4_K_Mを比較する
第3のレイヤーとして将来追加する余地はあるが、今回のスコープでは扱わない。

### llama_cppのBF16 GGUF変換手順（自前変換、2026-08-30追加）

第三者配布のGGUF（`unsloth/Qwen3-8B-GGUF`等）は使わず、`transformers_hf`/`vllm`に渡すのと
**同じ`Qwen/Qwen3-8B`のsafetensorsから自前で変換**する。配布元のツール・バージョン・設定が
不明なGGUFだと、重みが本当に同一である保証がなく、帰属可能性が損なわれるため。

```
Qwen/Qwen3-8B (safetensors, bf16)
  │ llama.cppリポジトリ同梱の convert_hf_to_gguf.py --outtype bf16
  │   （量子化ではなく、GGUFフォーマットへの詰め替えのみ。数値劣化なし）
  ▼
Qwen3-8B-bf16.gguf   ← レイヤー1で使用
```

真の量子化（4bit圧縮）が将来必要になった場合は、このBF16 GGUFを起点に`llama-quantize`で
追加変換する（今回のスコープ外）。

### 依存パッケージ・起動フラグ（Web検索で確認、実際のセットアップ時に再確認要）

- **`transformers_hf`**: `transformers>=4.51.0`（Qwen3対応の最小版、[HF Transformers Qwen3 docs](https://huggingface.co/docs/transformers/model_doc/qwen3)）、`torch`（CUDA 12.x系）、`accelerate`
- **`vllm`**: レイヤー1は`Qwen/Qwen3-8B`（bf16）、レイヤー2は`Qwen/Qwen3-8B-AWQ`に切り替えるのみで
  起動フラグは共通。`vllm>=0.19.0`目安（[Qwen公式vLLMガイド](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)がQwen3.6向けに言及した数字を流用しており、Qwen3-8B向けの正確な下限は起動時に再確認する）、CUDA 12.8以上。
  起動フラグに`--enable-auto-tool-choice --tool-call-parser hermes`が必要（Qwen系はHermes形式の
  tool-calling出力のため）。prefix cachingはV1エンジンでデフォルトON（既述）
- **`llama_cpp`**: `llama.cpp`をCUDA対応でビルド（L4向け）。**`--jinja`フラグが必須**——
  これが無いとtools付きリクエストが`tools param requires --jinja flag`で拒否され、
  agentic tool callingが機能しない（[llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)）。
  `cache_prompt`は既定でtrueのためフラグ不要（既述）。加えて`convert_hf_to_gguf.py`実行用に
  `llama.cpp`リポジトリのPython依存（`requirements/requirements-convert_hf_to_gguf.txt`）が必要

**再確認が必要な項目**：vLLMの正確なバージョン下限とtool-call-parser名（`hermes`）は
Qwen3.6向けの情報から類推したもので、Qwen3-8B specificに検証できていない。実際の
GPUインスタンス起動時に`vllm serve --help`と公式ドキュメントで確定させる。

## 実行計画の規律：フルグリッドを避ける

model × engine × tool段階 × n の総当たりは組まない（既存の精度スイープだけで
既に予算いっぱいなため）。ベースラインセルからの one-factor-at-a-time
スイープを基本とし、交互作用を見たい場合のみ縮小サブセットで真の総当たりを行う。

最初の一歩は **1モデル × genuinely異なる2エンジン × 1ツール段階** で
「帰属可能な差」が実際に出せるかを検証してから広げる。

## 方針転換：並行負荷（concurrency）を実験軸に追加（2026-08-30）

**目的の再設定**：ここまでの設計は「クラウドGPUを触った実績を示す」ことが暗黙の目的化
していた。目的を**「ローカルLLMのエンジン・モデル選定/パラメータ設定/プロンプト設計と
いう実務判断の証明」**に立て直した。

### Bedrock adapter・Nightly CI・S3蓄積は一旦棚上げ

理由は2つ：

1. Bedrockのモデルidは日付付きでバージョン固定されており、裏側の無断差し替えは稀
   （AWSは事前告知の上でdeprecateするのが通例）
2. ローカル側も固定重み＋決定的embedding judgeのため、コードが変わらなければ夜間実行しても
   差分は出ない（唯一の現実的なドリフト源は`requirements.txt`の`jsonschema>=4.0`が
   未pinという点だが、それ単体では継続的な検知価値の根拠として弱い）

どちらも「ライブに呼べる比較対象」が無い限り機能しないため、Bedrock結線ができた後
（元のロードマップ通り2026-10目安、[[cloud-layer-plan]]参照）に再検討する。

### 並行負荷を新しい実験軸に追加する理由

実務でvLLMを選ぶ根拠は「マルチユーザー前提の継続バッチング」だが、このプロジェクトは
ユーザー1人しかアクセスしない。素朴には「vLLMの強みを証明できない」というジレンマが
あった。解決策は「マルチユーザーはリクエスト到着パターンの問題であり、人数の問題では
ない」という整理——asyncio等で合成的に並行リクエストを生成すれば、エンジン側からは
複数人からのアクセスと区別がつかない（vLLM自身の`vllm bench serve`もconcurrency/QPSを
パラメータに取る標準的な手法）。既存の「精度パスとレイテンシパスは別実行」という設計は
維持し、latency passに並行数という軸を追加する形で収める（新規インフラは不要）。

**tool_stage(10/50/100)のフルスイープ幅を削り、並行数の深さに振る**方向で合意した
（GPU時間の予算を維持するためのトレードオフ）。tool_stageの幅は「選定判断の証明」と
しては相対的に弱い軸、並行数は強い軸という判断。stage10（記事1と完全重複する唯一の
オーバーレイ比較点）は維持し、残りの段階配分は別途再設計する（未着手）。

### 新しい問い：並行負荷下でのツールコーリング精度維持

> **2026-10-19：問いを書き直した（[[docs/adr/0002-equivalence-margin|ADR-0002]]の追記2026-10-19）。** 下の「並行性のために選んだvLLMが、正しさを犠牲にしていないか」は、
> vLLMを選ぶことを確かめる形で、どれを選ぶかを決める形になっていなかった。今の問いは
> **「並行数16で提供できる構成のうち、負荷の下で精度を保つのはどれか」**で、第3パスは容量の判定（ADR-0005）で並行数16に対応できる構成すべて
> （見込みではvLLM AWQとllama.cpp bf16）で行う。並行数16でのllama.cpp bf16とvLLM AWQの比較は、エンジンと量子化の差が混ざった「構成の比較」として示す。
> 以下は経緯として残す。

continuous batchingは理論上出力に影響しない設計だが、実際はバッチ構成によってカーネル
計算経路・浮動小数点の加算順序が変わり得るため、100ツールのERPカタログでツール選定や
引数抽出が本当にブレないかは未検証。これは「並行性のためにvLLMを選んだ、その選択が
正しさを犠牲にしていないという根拠」という production 判断の裏付けになり、単なる
スループット再検証（既知の事実）より訴求力が強い。

- 精度パスとレイテンシパスの分離は維持しつつ、この検証は**第3のパス**として設計する
  （精度パスの一部を並行数>1で実行し、シリアライズ版ベースラインとスコアを比較）
- 最小構成案：エンジン1つ（`vllm`）・tool_stage1つ・並行数1 vs 高並行数1点。既存の
  `repetition.py`のbootstrap CIで実行ノイズと実質的な精度差を区別できるため安価に
  検証可能。一致すれば「並行化は精度を犠牲にしない」という否定的結果も報告価値があり、
  乖離すれば記事として最も強い発見になる
- **3エンジン比較（accuracy pass・latency pass）自体は廃止しない**。役割が「総合的に
  どのエンジンが勝つか」から「並行性優先で選んだvLLMが単発精度・低並行数性能を
  犠牲にしていないか、`llama_cpp`（単一リクエスト最適化）とのクロスオーバーがどこで
  起きるか」という production 判断の裏付けに変わる

### 等価性マージンの決め方：consistency基準（2026-09-10確定）

> **2026-10-14：この節の決定は [[docs/adr/0002-equivalence-margin|ADR-0002]] で置き換えた。**
> マージンXは「直列ベースラインの`selection_ci95`の半幅」ではなく、**±2.6pt（BFCL v3でのQwen3-8BとQwen3-4Bの差）に固定**する。
> 第3パスはgreedyで実行し、ケース数はパイロットで不一致率を測ってから決める（Qwen3-4Bの自前計測はしない）。旧方式には欠陥が2つあった。
> Xも差分CIも1/√nで縮むのでケースを増やしても検出力が上がらないこと、そして半幅が反映しているのは
> シャッフルのブレではなくケースの難易度の違いであること。以下は検討の経緯として残す。
> [Source: User, 等価性マージンの再検討, 2026-10-14]

第3パスの結論は「並行化しても精度は落ちない」という**証明ではなく、「差は±X pt以内」
という上限のバウンド**として書く。ケース数が n=14 と小さく、「差が有意でなかった」は
検出力不足と区別できないためである。

まず前提として、**このプロジェクトで一時流通していた「n=14 で検出できるのは 30pt
クラスの差だけ」という値は誤り**だった（検証せず伝聞のまま設計に使っていた）。`_cluster_bootstrap_selection_ci` と同じ設計
（ケース単位リサンプル、trial込みでプール）でシミュレーションし直した結果、n=14 での
ブートストラップCI半幅は**約 ±9〜11pt**。ケース難易度（seed/grounded/adversarial）と
trial間相関を入れたモデルでも独立モデルの1.3倍以内に収まり、頑健だった。参考までに
n=30〜40 で ±6〜7pt、n=70〜100 で ±4.5〜5.5pt。

**マージン X は人間が決めた数値（±7pt 等）ではなく、同じセルの通常 accuracy pass で
既に測定されるツール順序シャッフルの trial 間ばらつき（`consistency` / `selection_ci95`
の半幅）を基準値として使う。** 半幅＝`(hi - lo) / 2`（`selection_ci95`が返す
`(lo, hi)`の中心値からの距離）。「差は±X pt以内」という下記の主張の書き方そのものが
Xを「ゼロからの距離」として使うことを前提にしており、半幅と定義して初めて整合する
（全幅=`hi - lo`をそのままXに使うと「±X」の意味が二重になり緩すぎる。2026-09-10、
全幅表記だった旧稿から訂正）。主張は次の形になる：

> 並行化は、ツール順序のシャッフルが既に生んでいるブレより大きなブレを追加しない。

この基準を選ぶ理由は3つ。(1) 基準値がデータ由来で、閾値の恣意性という
`vlm-vs-yolo` で自ら批判した落とし穴を踏まない。(2) 通常の accuracy pass の副産物なので
**追加のGPU実行コストがゼロ**。(3) 「無視できる差」の定義を、この評価系が
そもそも持っている再現性の粒度に接地できる——順序シャッフルで既に動く幅より小さい差を
「精度低下」と呼んでも実務的な意味がない。

**事前宣言の扱い**：基準値そのものは実測されるまで確定しないので、事前宣言するのは
数値ではなく**規則**である。すなわち「X はシリアライズ版ベースライン（並行数1）の
accuracy pass から算出した `selection_ci95` の半幅とする」と並行負荷ランの実行前に宣言し、
並行数>1の結果を見てから基準を選び直さない。これを守らないと基準がデータ由来である
利点が消える。

**実装上の前提（実行系着手前に必要）**：`eval_bench/repetition.py` に、同一ケース集合に
対する**対応ありbootstrap差分CI**を計算する関数を追加する必要がある
（現状の `_cluster_bootstrap_selection_ci` は1条件のプールされた選択率のCIしか返さない）。
これが無いと「2つの独立したCIが重なっているか」で判定することになり、
`vlm-vs-yolo` で批判したのと同種の誤った比較にフォールバックしてしまう。

### 着手前に確認が必要な2点（未確認、2026-08-30時点）

1. **L4の22GB VRAMが、8Bモデルで想定する最大並行数のKVキャッシュを実際に保持できるか**
   （上述「モデルサイズと具体的なインスタンス」のVRAM試算は並行負荷軸を知らない段階の
   ラフ計算であり、この方針転換が想定する並行数の深さでは再検証が必要）
2. **`transformers_hf`（bf16無量子化）は並行数を上げると大幅劣化またはOOMする可能性が
   高い**——サービング層の最適化を持たないナイーブベースラインという定義上、当然の
   帰結ではあるが、結果として報告する対象にするか、高並行数レンジでは対象外構成として
   除外するかを事前に決める必要がある

この2点を解消してから、並行数の具体的な水準・第3パスの実行設計・tool_stage段階配分の
再設計に着手する（未着手、次のセッションで対応）。

### 2点のブロッカーの解消（2026-08-30、実測ベース）

Qwen3-8Bの`config.json`（`hidden_size=4096`, `num_hidden_layers=36`,
`num_key_value_heads=8`, `head_dim=128`, GQA）と実際のトークナイザ（`Qwen/Qwen3-8B`の
`tokenizer.json`）を使い、`schema/tools.json`・`schema/tools_extended.json`を実際に
シリアライズしてトークン数を実測した（`x-scoring`はモデル非公開の採点用アノテーション
なので除去、`$comment`/`module`等の内部用フィールドも除去した「モデル向けビュー」で計測）。

**KVキャッシュ1トークンあたりのバイト数（bf16、全層合計）**：
`2(K,V) × 36層 × 8 kv_heads × 128 head_dim × 2byte = 147,456 byte ≒ 144 KiB/token`

**ツールカタログのトークン数実測**（`x-scoring`除去後、モデル向けビュー）：
- stage=10: 2,176 tokens（約306 MiB相当のKV）
- stage=50: 9,855 tokens（約1.35 GiB相当のKV）
- stage=100: 19,728 tokens（約2.71 GiB相当のKV）

system promptは実測57 tokens、クエリは実測15〜20 tokens/query。few-shot例は実行系コードが
まだ無く未測定（プレースホルダーとして数百tokenを見込む。実装時に本節を実測値で更新すること）。

**論点1「L4 22GBはこの並行数のKVキャッシュを保持できるか」への答え：
保持できるかは`prompt_layout`とengineの実装に依存する**

当初「並行数N倍でKVコストもN倍」という前提で試算すると、bf16重み（8B×2byte≒15.3GiB）を
引いた残り〜4-5GiBの余白では、stage=100で並行数2にすら届かない計算になった。しかしこれは
`prompt_layout=stable_first`前提のvLLMには当てはまらない。vLLMのprefix caching（V1で
デフォルトON）はブロック単位の完全一致で
KVブロックを共有するため、**システムプロンプト＋ツールスキーマ＋few-shotという不変プレフィックス
を共有する並行リクエストは、プレフィックスのKVを1回しか消費しない**。追加リクエストが払う
限界コストはクエリ＋生成分（実測ベースで概算〜100トークン ≒ 14MiB）のみで、bf16の4-5GiB余白
に対して無視できる小ささ。**vLLM + stable_firstに限れば、stage=100でもKVメモリ自体はボトル
ネックにならない**（ボトルネックがあるとすればvLLMの`--max-num-seqs`等のスケジューラ設定で
あり、VRAM容量ではない）。

一方、以下の3構成はこの恩恵を受けない：

- **`prompt_layout=query_first`**はプレフィックス共有が成立しないため、並行リクエストは
  フルコスト（stage=100で1件あたり約2.71GiB+α）を払う。bf16重み後の4-5GiB余白では並行数1
  が事実上の上限
- **`llama_cpp`**は`--parallel N`で`n_ctx`を固定N分割するアーキテクチャで、スロット間で
  KVを共有する機構を持たない。stage=100（1リクエストあたり必要文脈≒約20K token）で
  `--parallel 2`を成立させるには`n_ctx≧40K`が必要になり、Qwen3-8Bの
  `max_position_embeddings=40,960`の上限にほぼ張り付く。実質的にstage=100では並行数1が
  上限になる可能性が高い
- **`transformers_hf`**はマルチリクエストスケジューラもpaged/共有KVも持たないため、
  並行リクエストはそれぞれ独立にフルサイズのKVを確保する（論点2参照）

つまり**「並行数」と「prompt_layout」は独立変数ではなく交絡する**——これは当初の設計に
なかった知見。`query_first`は精度だけでなくKVメモリコストも犠牲にする、という発見は
「並行性を狙った設計判断が引き起こすトレードオフ」の具体例として、単なるスループット
比較より訴求力のある記事の論点になる。

**この知見を反映した設計変更**：

- 「並行負荷下の精度維持」第3パス（vLLM単体）は、bf16（layer1）ではなくAWQ 4bit
  （layer2）で実行する。bf16は重みだけで15.3GiBを消費し余白が薄い一方、AWQ 4bitなら
  重み〜4GiBで残り〜18GiBの余白があり、`query_first`のフルコスト条件でも並行数5〜6は
  現実的に確保できる、より説得力のある「高並行数」を狙える
- レイテンシパスの並行数軸は、bf16 3エンジン比較では**stage=10を主軸**にする
  （並行1リクエストあたり約306MiBなら、query_first・llama_cppの固定スロット方式でも
  並行数10前後は現実的）。stage=50/100での並行スイープは`vllm`×`stable_first`限定の
  追加観測点として扱う
- **並行数の具体的な水準（2026-09-10確定）**：

  > **2026-10-14：latencyパスの水準は [[docs/adr/0003-latency-pass-verdict|ADR-0003]] で置き換えた。**
  > 共通は 1/2/4/8 のまま残し、そこから各エンジンで「満たさない」または起動できなくなるまで、倍々に32まで追加する。
  > 下記の「`llama_cpp`は`--parallel 16`に耐えられない」は検証していない推定だった（16並行で約4.9GiBに対し、余白は4〜5GiBでぎりぎり。
  > 統合KVキャッシュ`--kv-unified`があるので、固定N等分という前提も古い可能性がある）。疎通確認のときに実測する。
  > 判定は「MLPerfの8B向けレイテンシ基準を満たすか」の3値判定に変えた。第3パス（AWQ、並行数1と16）は変更なし。
  > [Source: User, latencyパスの設計の見直し, 2026-10-14]

  - **bf16 layer1・3エンジン比較のレイテンシパス（stage=10主軸）: 並行数 1/2/4/8。**
    stage=10は1リクエストあたり約306MiBなので、プレフィックス共有の恩恵を受けない
    `query_first`・`llama_cpp`の固定スロット方式でも8までは全エンジンで成立する。
    16まで伸ばす案は棄却した——`llama_cpp`が`--parallel 16`の`n_ctx`分割に耐えられず
    測定不能の穴ができ、エンジン間で水準が揃わないと比較可能性の規律（上述）が崩れるため。
    倍々の4点あればクロスオーバー点（低並行では`llama_cpp`が勝ち、高並行で`vllm`が
    逆転する点）の位置は特定できる
  - **AWQ 4bit・`vllm`単体の第3パス（並行負荷下の精度維持）: 並行数 1 vs 16 の2点、
    `tool_stage=10`に固定（2026-09-10訂正・確定）。**
    等価性主張に必要なのは「シリアライズ版ベースライン vs 高並行」の対応あり差分だけで、
    中間点は主張に寄与しない一方でGPU時間と多重比較のコストを増やす。

    **訂正**：旧稿は「`stable_first`のprefix共有が効くため16は余裕を持って成立する」と
    書いていたが誤り。第3パスは`eval_bench/repetition.py`の反復機構を使って精度を測る以上、
    (case×trial)ごとにツール順序を独立にシャッフルするのが前提（「なぜ毎回シャッフルするか」
    は上記「反復とconsistency」の設計そのもので、position biasと真のツール数効果を分離する
    ための必須要件——測定手法として動かせない）。したがって並行して飛ぶ16リクエストは
    それぞれ別々の並びを持ち、prefix共有はシステムプロンプト（約57token）止まりで、
    ツールスキーマ以降は事実上共有されない。共有なしの実コストで見積もると：
    stage=10は1リクエストあたり約306MiBなので16並行で約4.9GiB（AWQの余白〜18GiBに
    余裕で収まる）、stage=50は際どい、stage=100（1リクエストあたり約2.71GiB×16≈43GiB）
    は収まらない。そのため第3パスは`tool_stage=10`に固定する
    （成立の確認は下記の`Maximum concurrency`ガードで行う）

- **要実装（実行系実装時のタスク）**：vLLMは実際にはKV不足時にOOMではなくリクエストを
  キューイングする（`gpu_memory_utilization`から計算した`num_gpu_blocks`を超えるリクエスト
  は待たされる）ため、「並行数Nで実行した」つもりが実際は直列化されたキュー待ち時間を
  測っているだけ、というサイレントな無効データになるリスクがある。対策：vLLM起動時ログの
  `# GPU blocks`・`Maximum concurrency for N tokens per request`をrun manifestに記録し、
  観測された最大並行数が要求した並行数を下回っていたらそのランを失敗扱いにするガードを
  実行系に入れる

**論点2「`transformers_hf`の高並行数扱い」への答え：並行数軸から除外し並行数1のみで
報告する（結論は据え置き、根拠を訂正）**

当初「GILがあるので並行実行できない」という理由を想定していたが誤り。PyTorchはCUDA
カーネル起動中にGILを解放するため、Pythonスレッドは実際にインターリーブしうる。正しい
理由は、`transformers`の素の`generate()`にはマルチリクエスト用のスケジューラも
paged/共有KV機構も無く、並行呼び出しはそれぞれ独立にフルサイズのKVキャッシュを確保して
GPUを奪い合うだけで、バッチ化による恩恵が一切無いこと。この構成上の欠如自体が「サービング
最適化を持たないナイーブベースライン」という`transformers_hf`の役割そのものであり、並行数1
のみで報告し「なぜ並行数を上げられないか」を対照点として明記する（gapではなく発見として
扱う）。

## コスト管理

全エンジンをクラウドGPUに寄せたことで、今回は実弾のドル課金が発生する
（ローカルCPU比較のような無料枠はない）。

- **見積り（2026-08-29再改訂）**：セットアップ（依存関係・重みDL・疎通確認、
  engineあたり1〜3h）＋精度スイープ（10/50/100の3段階、`stable_first`基準＋
  `query_first`のtool_stage=100 ablation）＋レイテンシパス（同じ10/50/100段階
  × `prompt_layout` 2値 × concurrency 1とN、engineあたり1〜2h）をコア3engine分、
  手戻りバッファ込みで**合計35〜55h稼働**（Ollamaを4本目として追加する場合は
  ここに加算）。既存ケース流用により新規ケース作成の手間は消えた一方、
  accuracy passも段階100を含むため精度側のprefillも重くなる（目安2.5倍程度）。
  正味では前回の$30〜55からほぼ横ばい〜微減の見込みだが、保守的に**$30〜55のまま
  据え置き**、実行してみて余裕があればBudgetsの余白として扱う
- **Budgetsアラート**：警告 $30 / ハードストップ想定 $60 の2段階
  （見積りに対して十分なバッファを確保）
- **GPUインスタンスを実際に起動する前に、まずAWS Budgetsアラートを実装する**
  （自動ブレーキが現状存在しないため。
  2026年8月頭のFree Plan期限切れ自動close事故の教訓も踏まえる）
- GPUインスタンスは常時起動せず、ECS smoke-testの時と同じ
  `apply → 動作確認 → 即destroy` の規律を踏襲する（applyの前には実AWS状態を読み取り専用で確認してから進める）
- モデル重みを複数フォーマット（GGUF/safetensors/AWQ等）で保持する分、
  EBSボリュームを既定より拡張する必要がある（数GB〜数十GB分の追加コストは軽微）

## 記事との関係

engine-as-controlled-variable + レイテンシ計測は、記事1の「素の few-shot ×
モデル横断」という主張とは別の問いに答えるものなので、**記事1のスコープは
一切変更せず、別記事（記事2想定）として切り出す**。記事1の
「Adapterが出力を整形しない」「few-shotを全モデル共通に固定する」という
"同じ土俵"の前提はそのまま維持される。

## 変更しないもの

分離採点・N/A規則・semantic judgeの校正方針・Adapter/Decoderの責務境界・
データ透明性ルール・**Article1の既存4段階（5/10/20/40）と既存40ツール
カタログ・既存ケースデータ**は、この計画によって一切変更しない。
Article2の拡張100ツールカタログは、これらに触れない**純粋な追加拡張**として
別ファイル/別バージョンで管理し、既存ケースはそのまま読み込んで使う
（新規ケース作成はしない）。

## 確定した設計（2026-08-29セッションで解消）

- エンジンロースター：コア3本（`transformers_hf`/`llama_cpp`/`vllm`）、
  Ollamaは任意4本目・限界明記の上で採用
- prompt layout：ON/OFFトグルは両engineとも既定ONで実務的でないため棄却。代わりに
  `stable_first`/`query_first`（安定部分とクエリの並び順）をキャッシュ効果の軸に採用
  （`cell.prompt_layout`）。position bias等で正解性にも影響しうるため
  latency専用ではなくaccuracy passにも入れる（`query_first`は最大tool_stage(100)
  のみのablation、スキーマの`allOf`で強制済み）
- run manifestスキーマ：[`schema/run_manifest.schema.json`](../schema/run_manifest.schema.json)
  として確定（`tool_catalog_version`必須化、`tool_stage`はaccuracy/latency共通で
  `[10, 50, 100]`のenum）
- **tool_stageの水準はaccuracy/latency両passで10/50/100に統一**（既存ケースを
  無変更で流用できると判明したため）。段階10は既存40ツールカタログの範囲内で
  構成し、Article1のstage=10と完全に同一の値になる（唯一残るオーバーレイ比較点）。
  段階50/100は既存40＋新規60（ERP同一ドメインの追加業務機能、近縁機能設計までは
  求めないが実在感は担保）の拡張100ツールカタログでpaddingする
- **quantizationはレイヤー1（3engine bf16比較）とレイヤー2（vLLM単体、bf16 vs AWQ 4bit）に
  分離（2026-08-30）**：GGUF Q4_K_MとAWQ 4bitを異なるengine間で直接比較すると、差分が
  「engineの違い」か「量子化アルゴリズムの違い」か区別できないため。`llama_cpp`はレイヤー1で
  無量子化のBF16 GGUFを自前変換して使う（詳細は上述セクション参照）
- **harnessはchat APIを使わない（2026-08-30）**：3engineのchat template実装差がprefix
  caching・`prompt_layout`比較を汚染するため、harness側でプロンプト文字列を組み立て、
  3engineとも生の補完エンドポイントを叩く
- **精度パスは全engine`concurrency=1`固定（2026-08-30）**：vLLMのcontinuous batchingが
  `consistency`指標にノイズを持ち込むのを避けるため

## 残る未決定事項（実装着手前に解消する）

2026-08-29時点で下記4項目は解消済みだった（詳細は各項目内のリンク先セクション参照）。
2026-08-30の方針転換（上述「並行負荷を実験軸に追加」）で新たに乗った2点のブロッカーも、
同日中に実測ベースで解消済み（詳細は上述「2点のブロッカーの解消」節）：

1. ~~L4の22GB VRAMが、方針転換が想定する並行数の深さでKVキャッシュを実際に保持できるか~~
   **解消**。「保持できるかは`prompt_layout`とengineの実装に依存する」が答え。
   `vllm`×`stable_first`はprefix caching共有によりstage=100でもKVメモリはボトルネックに
   ならない。`query_first`・`llama_cpp`（固定スロット方式）・`transformers_hf`はプレフィックス
   共有の恩恵を受けず、bf16ではstage=100で並行数1が事実上の上限
2. ~~`transformers_hf`（bf16無量子化）が高並行数で大幅劣化/OOMした場合、結果として
   報告するか対象外構成として除外するか~~ **解消**。並行数軸から除外し、並行数1のみで
   報告する（マルチリクエストスケジューラもpaged/共有KVも持たないことが理由。GILは無関係
   ——訂正済み）

**次に着手すること（未着手）**：
- 第3パス（並行負荷下の精度検証）の実行設計を`vllm`のAWQ 4bit構成（layer2）に固定して詳細化
- `eval_bench/repetition.py`に対応ありbootstrap差分CI関数を追加（上述「等価性マージンの
  決め方」参照。マージン方式は2026-09-10に確定済みなので、これが第3パスの残る実装ブロッカー）
- tool_stage段階配分の再設計（stage10は維持、50/100の並行スイープ範囲を上記の交絡知見に
  合わせて調整）
- vLLM起動時ログ（`# GPU blocks`・`Maximum concurrency`）をrun manifestへ記録する項目の
  スキーマ追加、および観測並行数が要求並行数を下回った場合にランを失敗扱いにするガードの
  設計（実装は実行系着手時でよいが、`schema/run_manifest.schema.json`への項目追加は
  設計時に済ませておく）

GPUインスタンスのTerraform化・engineごとの起動スクリプト・run manifest出力の実装は、
この設計が固まってから着手する。

1. ~~`eval_bench/repetition.py`の`aggregate_repeated`拡張~~ **完了**。
   `arg_exact_mean`/`arg_semantic_mean`の集計を追加、`tests/test_repetition.py`に
   検証テスト追加（コミット`3aa2689`）
2. ~~拡張ツールカタログの作成（40→100、60ツール追加）~~ **完了**。
   `scripts/generate_extended_catalog.py`→`schema/tools_extended.json`、
   `tests/test_extended_catalog.py`で検証（コミット`ce51969`。配分の根拠は
   上述「tool_stageの水準とツールカタログの設計」参照）
3. ~~g6.xlargeのTokyo単価の一次情報確認~~ **完了**。`wsl.exe -d Ubuntu`
   経由で`aws pricing get-products`を実行し、**$1.1672/hr**（SKU
   `ZEHXGXZKA86RDSNK`、2026-08-01改定）を確認。Web検索値$1.167/hrとほぼ一致
   （コミット`c4f9ae5`）。このWindowsシェル自体にAWS CLIは無いが、
   `wsl.exe -d Ubuntu -- bash -lc "..."`経由でWSL2側のCLIを呼び出せることが判明
   （デフォルトディストロは`docker-desktop`でbash/aws共に無し、`-d Ubuntu`で
   明示指定が必要）
4. ~~各engineの具体的なセットアップ手順・依存パッケージ・8Bモデルの
   チェックポイント選定~~ **完了**。上述「モデルチェックポイントと各engineの
   セットアップ」参照。`Qwen/Qwen3-8B`を軸に、`transformers_hf`=bf16無量子化・
   `llama_cpp`=GGUF Q4_K_M・`vllm`=AWQ 4bit(公式prequantized)で構成。
   vLLMのバージョン下限とtool-call-parser名はQwen3.6向け情報からの類推のため、
   実際のGPUインスタンス起動時に再確認が必要（上述の注記通り）
