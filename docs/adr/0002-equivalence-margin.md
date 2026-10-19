# ADR-0002: 第3パスの等価性マージンを ±2.6pt（モデル1段階分の差）に固定する

- Status: Accepted
- Date: 2026-10-14

## Context

第3パス（`accuracy_under_load`：AWQ 4bit・`vllm`・`tool_stage=10`で、並行数1と16を比べる）の結論は、
「並行化しても精度は落ちない」という証明ではなく、TOST型の等価性判定（`eval_bench/repetition.py`の
`equivalence_verdict`）として書く。差分の信頼区間がまるごと ±X の内側に入れば「等価」とする。このXの決め方が論点である。

2026-09-10の決定（[[docs/engine-performance-plan|エンジン性能観測層の設計方針]]「等価性マージンの決め方」）では、
Xを「直列ベースラインの`selection_ci95`の半幅」とデータから決めていた。2026-10-14に見直したところ、2つの欠陥が見つかった。

1. **ケースを増やしても検出力が上がらない。** Xはおおよそ「ケース間のばらつき/√n」、比較する差分CIの半幅は
   「ケースごとの差のばらつき/√n」で、どちらも1/√nで縮む。判定は2つのばらつきの比だけで決まり、nに依存しない。
   そのため、パイロットで分散を測ってから検出力を計算し、ケース数を決める、という標準的な手順が成り立たない。
2. **計算しているものと主張が合っていない。** 主張は「並行化は、ツール順序のシャッフルが既に生んでいるブレより
   大きなブレを追加しない」だが、`selection_ci95`の半幅が主に反映しているのはケースごとの難易度の違いで、
   シャッフルによるブレではない。

そこで、Xをデータを見る前に数値で固定し、その根拠を外部の情報から取ることにした。[Source: User, 等価性マージンの再検討, 2026-10-14]

## 検討した案

| 案 | X | 採否 | 理由 |
| --- | --- | --- | --- |
| 直列ベースラインのCI半幅（旧決定） | データ次第（14件なら約±9〜11pt） | 却下 | 上記Contextの2つの欠陥 |
| 量子化の業界慣行「元の精度の99%以上」・MLPerfの99%/99.9% | 正解率80%なら約0.8pt | 却下 | 大規模な公開ベンチマーク向けの基準で、点推定で判断している。手作りの日本語ケースでは、CIで示すのに必要な件数が非現実的 |
| Singh (2026) の登録マージン | ±2pt | 却下（参照線として残す） | 対応ありのTOSTでも、不一致率12%ならMMLU規模の約2,000問が必要。根拠も「事前に宣言すること」が中心で、数値そのものの裏付けは薄い |
| 実務の慣行・ADR-0001のCI回帰許容 | ±5pt | 却下（参照線として残す） | LLM回帰テストの実務ガイドに普遍的な数値は無く、5%前後は経験則。ADR-0001の-0.05も「初期値」で根拠は記録されていない。下記のとおり、モデル2段階分の劣化を「等価」と判定してしまう |
| **モデル1段階分の差（採用）** | **±2.6pt** | **採用** | 下記Decision |

## Decision

**X = 2.6pt（固定）。** 根拠は Qwen3 Technical Report（arXiv 2505.09388）Table 18 にある、
BFCL v3（non-thinking）での **Qwen3-8B 60.2 と Qwen3-4B 57.6 の差**。

参考として同じ系列の値を挙げると、Qwen3-14B 61.5（8Bとの差 1.3）、Qwen3-32B 63.0（Table 14・16）、
Qwen2.5-7B-Instruct 55.8（Table 18）。モデルを半分のサイズにしても落ちるのは2.6ptであり、±5ptは緩すぎる。

**筋**：実務では、コスト削減のためにモデルを1段階小さくするかどうかを普通に判断する。並行化による劣化が
「モデルを1段階小さくしたときの劣化」より小さいなら、並行化を選ぶことは精度の面で正当化できる。それより大きいなら
「並行化するより、小さいモデルを並べる方がよい」という判断になる。どちらの結論も、意思決定にそのまま使える。
これは臨床の非劣性試験で、既知の意味のある効果（FDAのM1）を基準にマージンを決めるのと同じ型の考え方である。

あわせて次の3点を決めた。

1. **第3パスはgreedy（temperature=0）で実行する。** temperature 0.7ではサンプリングだけで正誤が入れ替わり、
   ±2.6ptを示すのに数百〜千件規模のケースが必要になる。そもそも第3パスで測りたいのは「バッチ処理が計算を変えるか」
   なので、サンプリングのノイズは取り除くのが正しい実験設計である。greedyでも実行ごとに出力は揺れうる
   （Atil et al. arXiv 2408.04667。原因としてサービング側のバッチ処理が挙げられている）。その揺れこそが、ここで切り出して測る対象になる。
   accuracyパス・latencyパスは、従来どおりQwen推奨値（temperature=0.7など）のままとする。
2. **ケース数はパイロットで決める。** 14件でパイロットを回し、2条件で正誤が入れ替わったケースの割合（不一致率 p_d）を
   実測する。そこから、X=2.6ptを検出力80%で示すのに必要な件数を計算し、その件数までケースを増やしてから本番を実行する。
   パイロットのデータは本番の判定に混ぜない（結果を見ながら件数を足していく形になるのを避けるため）。
   CIは現行実装どおり95%（片側それぞれα=0.025）とする。TOSTの慣行である90%CIより保守的なので、検出力の計算もこれに合わせる。
3. **Qwen3-4Bは自前では計測しない。** 一度は「このベンチマークでも4B→8Bの差が2.6pt程度か確かめる補足計測」として
   追加することにしたが、同日に取り下げた。理由は、この計測で変わる判断が無いこと。Xは固定済みで、4Bの結果を見て動かさない
   （実測値にも信頼区間が付くので、それをXにするとマージンがデータ次第で動き、旧決定と同じ種類の問題に戻る）。
   ケースが少ないうちは差のCIも広く、確認としての力も弱い。さらにモデル同士の比較はArticle1の範囲で、
   Article2の問い（エンジンと並行化）から外れる。本番で使うのは8Bだけである。[Source: User, 4B計測の要否, 2026-10-14]

## Consequences

- **計算のずれ**：BFCL v3は複数ターンを含む英語中心の総合スコアで、このハーネスの日本語ERP・単発のselectionとは尺度が違う。
  この限界は記事に明記する。自前の計測では確かめない（Decisionの3）。
- **パイロットで必要件数が非現実的とわかった場合**：等価性の判定は行わない。代わりに、差分の推定値とCIを
  参照線（±1.3pt＝14B差、±2.6pt＝4B差、±5pt＝ADR-0001）と並べ、「達成できた範囲」として報告する（Singh 2026 の推奨する報告の仕方）。
  Xを後から緩めることはしない。
- 呼んではいけないツールを呼んだケース（`forbidden_tools`・negative事例）は、マージンとは別に、1件でもあれば報告する（ADR-0001と同じ扱い）。
- `eval_bench/repetition.py`の`half_width`・`equivalence_verdict`のdocstringは旧決定（CI半幅）を前提にしている。
  コードの変更なので、段階1の実装PRで直す。`equivalence_verdict`の`margin`は引数なので、ロジックの変更は不要。
- 等価性の主張は「シャッフルのブレより小さい」から「モデル1段階分の差より小さい」に変わる。
  シャッフルによるブレそのものは、直列を2回回すA/A比較で実測し、ノイズの下限として別に報告する。

## 追記（2026-10-15）：検出力シミュレーションの結果

Decisionの2（ケース数はパイロットで決める）に向けて、必要なケース数と、今の解析方法が誤りの率を守れるかをシミュレーションで調べた
（`scripts/power_simulation.py`。Morris, White & Crowther 2019 のADEMPの形で設計）。
[Source: User, 検出力シミュレーション, 2026-10-15]

**データの形**：ケースごとに5回の試行を2つの条件で回し、試行ごとに正誤の組ができる。統計では「クラスターのある対応のある2値データ」
（clustered matched-pair binary data）と呼ばれる形で、ケースがクラスター、試行がその中のペアにあたる。
条件の間で正誤が入れ替わる割合（不一致率 p_d）と、入れ替わりやすさのケース間の偏り（ケース内の相関 ρ）を変えて、偽のデータを作った。

**検出力の水準**：80%に固定せず、80%と90%の両方の件数を見て決める。80%はCohen（1992）が一般用に提案した慣行で、
ICH E9は第2種の過誤を「慣例として10〜20%」としている（検出力80〜90%）。

**結果**（マージン±2.6pt、各点2,000回。シミュレーション自体の誤差は2.5%付近で±0.35%程度）：

| 不一致率 p_d | 検出力80%に必要なケース数 | 90% | 本当の差が±2.6ptのとき「等価」と出る確率（上限2.5%） |
| --- | --- | --- | --- |
| 3% | 100〜120 | 150〜200 | 3.9〜11.5% |
| 5% | 200 | 250 | 2.4〜7.4% |
| 10% | 400 | 400〜500 | 2.2〜4.4% |
| 20% | 700 | 1,000 | 2.1〜4.1% |

（範囲はρ=0・0.3・0.6と方法の違いによる。件数は候補の刻みで探した粗い値。）
必要なケース数はおおよそ 3,100 × p_d になる。差が0のとき、ケースごとの差の分散は p_d/5 で、ρに依存しないためである（導出）。

**今の解析方法は、件数が現実的な範囲で誤りの率を守れない。** パーセンタイル・ブートストラップ（`_paired_cluster_bootstrap_diff_ci`）に加えて、
ケース単位のt区間（SchuirmannのTOST。試行数がそろっているときはObuchowski 1998のクラスター補正した分散と同じ）、
分散を帰無仮説の値の周りで計算する2つの形（Obuchowski 1998・Yang, Sun & Hardin 2010 の差0の検定を±マージンにずらしたもの。
ずらした形は自分で導いたもので、論文では確認していない）を同じデータで比べた。どれも、不一致率が小さいとき（3〜5%）やケース内の相関が強いときに、
誤って「等価」とする確率が2.5%を大きく超えた。不一致が少数のケースに集中すると、実効的なデータが数件分しかなく、正規近似が成り立たないためと考えられる（推定）。
Yang, Sun & Hardin（2012）も、この形の非劣性の検定の性能は、クラスターの数とクラスター内の相関に左右されると報告している。

**そのため、今の方法をそのまま本番の判定に使わない。** 方法は、パイロットで不一致の実際の値を見てから、次のどれかに決める。

| 案 | 内容 | 必要なケース数 |
| --- | --- | --- |
| A. 入れ替わりの割合に上限を付ける | 精度の差は入れ替わったペアからしか生まれないので、差の大きさは不一致率を超えない。さらに不一致率は「1回でも入れ替わったケースの割合」を超えない。後者にClopper-Pearsonの片側97.5%上限を付け、2.6%以下なら等価とする（導出）。正規近似を使わず、ケースを単位に数えるので相関の影響も受けない | 入れ替わったケースが0件なら141件、1件なら212件、2件なら276件 |
| B. 近似の方法の信頼度を、シミュレーションで補正する | 誤りの率が2.5%以下に収まるまで区間を広げる。FDAは解析で示せない場合のシミュレーションによる評価を認めているが、EMAはそれだけに頼るには特別な理由が要るとしている | 数百件以上。パイロットの p_d と ρ に依存する |
| C. 等価性は判定しない | Consequencesの「必要件数が非現実的な場合」のとおり、推定値とCIを参照線と並べて報告する。CIが狭く出る傾向があることも限界として書く | 制約なし |

**パイロットの後の分岐（データを見る前に決めておく）**：入れ替わったケースが0件ならAを候補にし、ケースを約140件まで増やすかを判断する。
入れ替わりが複数あれば、AもBも数百件になるのでCにする。いずれの場合も、マージン（±2.6pt）は動かさない。

**ケース数そのものの問題**：今のケースセットは14件で、どの案でも等価性を言うには少なくとも約140件が要る。
同じ問題は、精度パスのエンジン間の比較にも当てはまる（こちらは判定方法自体がまだ決まっていない）。ケースをどう増やすかは、別に決める。

## 追記（2026-10-19）：第3パスはvLLM AWQだけで行う

1回目のapply（疎通）の前に、セル一覧へllama.cpp（GGUF bf16）の負荷時の精度のセルを足していた。スキーマは第3パスをvLLM AWQに限っていたので、
manifestの検証で拒否された（`docs/gpu-smoke-2026-10-18.md`のD3）。スキーマを広げるか、セルを外すかを検討し、**セルを外し、第3パスはvLLM AWQだけ**とした。

- **第3パスの問いは「並行化のために選んだ構成が、精度を犠牲にしていないか」**（計画書「新しい問い」）。業務で並行稼働させる用途はvLLMが中心なので、
  問いに当てはまるのはvLLMの構成になる。Dockerは「Prototype locally with llama.cpp. Scale to production with vLLM.」と役割を分けている。
  llama.cppを業務で並行稼働させる例もある（Hugging Face Inference Endpointsは、GGUFのモデルにllama.cppのコンテナを選び、同時リクエスト数を設定できる）が、
  文書の設定例は4リクエスト×1,024トークンと小さい。
- **「並行数が高いならvLLM」という前提は、出どころだけに頼らず、自分のデータでも確かめる。** bf16の層で両エンジンが載る並行数（ADR-0003の全エンジン共通の1/2/4/8）の
  latencyパスで、差が出始めるかを見る。差が出始めれば前提の一部を裏付けられ、出なければそれも結果として報告する。
- **検討した案**

  | 案 | 採否 | 理由 |
  | --- | --- | --- |
  | スキーマの限定を広げて、llama.cpp（GGUF bf16）を入れる | 却下 | llama.cppの現実的な使い方（並行数が小さい）は、bf16の層の低い並行数の比較で扱える。パイロットが約1.5〜2時間延びる（推定） |
  | パイロットの後に決める | 却下 | パイロットで出るvLLMの結果は、llama.cppを入れるかの判断材料にほとんどならず、待っても決めやすくならない |
  | llama.cppのGGUF Q4_K_Mを4bitの層に戻し、第3パスを両エンジンで回す | 却下 | 第3パスはエンジンの中で比べるので、量子化の違いによる交絡は入らない。しかし4bitの層でエンジン同士をlatencyで比べることになり、2026-08-30に避けた交絡（エンジンの違いか、Q4_K_MとAWQの違いか）を抱える。並行稼働の用途がvLLM中心である以上、その代償に見合わない |
  | 比べた後に、勝った方だけで第3パスを回す | 却下 | 第3パスは同じ構成の並行数1と16を比べるもので、比較の結果を待つ必要が無い。また並行数16で比べる相手が、bf16の層では載らず、4bitの層にはいない |

- **限界**：第3パスの結論（並行化しても精度が落ちない）は、vLLM AWQについてしか言えない。
[Source: User, 第3パスにllama.cppを入れるかの検討, 2026-10-19]

## 参考

- Qwen3 Technical Report, arXiv 2505.09388, Table 14・16・18 — https://arxiv.org/abs/2505.09388
- Singh, Certifying Compressed Language Models: An Audit and a Statistical Toolkit, arXiv 2608.15046 — https://arxiv.org/abs/2608.15046
- Atil et al., Non-Determinism of "Deterministic" LLM Settings, arXiv 2408.04667 — https://arxiv.org/abs/2408.04667
- Ma et al., (Why) Is My Prompt Getting Worse? Rethinking Regression Testing for Evolving LLM APIs, arXiv 2311.11123 — https://arxiv.org/abs/2311.11123
- Red Hat Developer, Deployment-ready reasoning with quantized DeepSeek-R1 models — https://developers.redhat.com/articles/2025/03/03/deployment-ready-reasoning-quantized-deepseek-r1-models
- （追記）Morris, White & Crowther, Using simulation studies to evaluate statistical methods, Statistics in Medicine 38(11), 2019 — https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164
- （追記）Cohen, A power primer, Psychological Bulletin 112(1), 1992 — https://www.ece.uvic.ca/~bctill/papers/mocap/Cohen_1992.pdf
- （追記）ICH E9, Statistical Principles for Clinical Trials — https://database.ich.org/sites/default/files/E9_Guideline.pdf
- （追記）Schuirmann, A comparison of the Two One-Sided Tests Procedure and the Power Approach for assessing the equivalence of average bioavailability, J Pharmacokinet Biopharm 15, 1987（原典は未確認。TOSTと区間による判定の関係は arXiv 2306.06698 で確認）
- （追記）Obuchowski, On the comparison of correlated proportions for clustered data, Statistics in Medicine 17(13), 1998、Yang, Sun & Hardin, A note on the tests for clustered matched-pair binary data, Biometrical Journal 52(5), 2010（式はRパッケージ clust.bin.pair の実装で確認） — https://cran.r-project.org/web/packages/clust.bin.pair/readme/README.html
- （追記）Yang, Sun & Hardin, Testing non-inferiority for clustered matched-pair binary data in diagnostic medicine, CSDA 56(5), 2012（要旨のみ確認） — https://ideas.repec.org/a/eee/csdana/v56y2012i5p1301-1320.html
- （追記2026-10-19）Docker, Docker Model Runner Integrates vLLM for High-Throughput Inference — https://www.docker.com/blog/docker-model-runner-integrates-vllm/
- （追記2026-10-19）Hugging Face, Inference Endpoints: llama.cpp — https://huggingface.co/docs/inference-endpoints/en/engines/llama_cpp
- （追記）Nam, Power and sample size requirements for non-inferiority in studies comparing two matched proportions where the events are correlated, CSDA 55(10), 2011（要旨のみ確認） — https://ideas.repec.org/a/eee/csdana/v55y2011i10p2880-2887.html
