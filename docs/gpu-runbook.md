# GPUインスタンスでの実行手順書

[[docs/cloud-layer-plan|クラウド層の設計方針]]に基づき、`infra/gpu/`で`terraform apply`してから`destroy`するまでの手順。
インスタンス上の作業（準備・サーバーの起動と再起動・セルの実行）は`scripts/gpu/`のスクリプトがやり、手で打つのは下の6つのコマンドだけにする。
手で繰り返すと、llama.cppの再起動忘れや回収漏れのように、データを黙って壊す間違いが起きるためである。

インスタンスは`terminate`と自動シャットダウン（`var.auto_shutdown_minutes`。既定は6時間、本番のapplyは下の「1. apply」の値を渡す）で、EBSごと確実に消える。
**復元できない情報は、取れたときにすぐ手元へ回収する。** 回収は`local.sh collect`が1分ごとに行う。

コマンドはすべてWSL側で、リポジトリの直下から打つ。

## 0. applyの前に

- SSHを許可するIP（`infra/gpu/variables.tf`の`ssh_allowed_cidr`）が、今の回線のIPと同じか確かめる。違えば`-var`で渡す。
  ```
  curl -4 -s https://checkip.amazonaws.com
  ```
- 回したいコミットをコミットしておく。`local.sh push`はコミット済みの内容だけを送り、未コミットの変更があれば止まる
  （どのコミットとも一致しない実行を作らないため）。送ったコミットは`~/out/repo_commit.txt`に残る。

## 1. apply（手で行う）

```
cd infra/gpu && terraform apply && cd -
IP=$(terraform -chdir=infra/gpu output -raw gpu_public_ip)
```

本番のapplyは、自動シャットダウンを延ばして渡す（ADR-0007）。速さの測定（`main-latency.json`）は18時間、精度と第3パス（`main-accuracy.json`）は10時間。

```
terraform -chdir=infra/gpu apply -var auto_shutdown_minutes=1080   # main-latency.json
terraform -chdir=infra/gpu apply -var auto_shutdown_minutes=600    # main-accuracy.json
```

始める時刻は、自動シャットダウンの前に手元のPCが起きていて、回収とdestroyができるように選ぶ（1回目は約14.4時間かかる見込み）。

`InsufficientInstanceCapacity`で失敗したら、`infra/gpu/instance.tf`のサブネットのindexを変えて、applyし直す（L4の在庫が無いAZ）。

- 2026-10-22は、index 0（ap-northeast-1c）で在庫切れになった。terraformは約10分試し直し続けるので、Ctrl-Cで止めた（インスタンスは作られていない）。
  index 1（ap-northeast-1a）に変えてapplyすると通った。
- indexとAZの対応は、g6.xlargeを出せるAZのサブネットの並び（`infra/gpu/data.tf`）で決まり、順番は保証されない。変える前に
  `terraform -chdir=infra/gpu console`で`data.aws_subnets.default.ids`を出し、WSLの`aws ec2 describe-subnets --subnet-ids ...`でAZを確かめる。
- indexの変更はコミットしない。applyが通ったらすぐ元に戻してよい（インスタンスはstateに残るので、destroyはそのまま通る）。

## 2. 準備

```
scripts/gpu/local.sh push  $IP
scripts/gpu/local.sh setup $IP          # 数秒で戻る（戻らなければ下の「SSHが戻らないとき」）
scripts/gpu/local.sh wait-setup $IP     # SETUP_DONE（成功）かSETUP_FAILED（失敗）が出るまで30秒ごとに見に行く
scripts/gpu/local.sh collect $IP &      # 別のターミナルで回し続けてもよい。DONEが出るまで1分ごとにコピーする
```

準備の実測は37分（2026-10-22、06:52→07:29Z）。準備は毎回やり直すので、applyごとにこの時間がかかる。

終わったかどうかは、ジョブが書く目印のファイル（`SETUP_DONE`・`DONE`など）で判断する。SSHが閉じたかどうかでは判断しない
（2026-10-18の疎通で、SSHが閉じたのをsetupの終わりと読んで、約17分待ち損ねた）。

**SSHが戻らないとき**（`local.sh setup`・`cells`が数秒で戻らない）：別の端末からSSHで入り、詰まっているSSHセッションの子プロセスを見て、
SSHの出力の口（パイプ）を握っているプロセスを特定する。原因はまだ分かっていない（`docs/gpu-smoke-2026-10-18.md`のD5）。

```
pstree -p $(pgrep -f 'sshd: ubuntu@notty' | head -1)    # 詰まっているセッションの子
ls -l /proc/<PID>/fd                                     # 0・1・2が pipe:[...] のままのものが握っている
```

見つけたプロセスと、そのコマンドラインを記録する。特定に手間取る場合は、起動を`systemd-run`（systemdの子として起動し、SSHのセッションから何も引き継がない）に
切り替えることを検討する。ジョブ自体はSSHと切り離して動いているので、戻らないSSHは`Ctrl+C`で止めてよい。

`setup.sh`（`~/out/setup.log`）がやること：

1. **AMIの前提の確認**：`nvidia-smi`と`nvcc --version`。`nvcc`が無ければllama.cppのCUDAビルドができないので、ここで止まる
   （AMI `base-oss-nvidia-driver-gpu-ubuntu-22.04`の前提が崩れたということ）。自動シャットダウンの予定（`shutdown --show`）も記録する。
2. vLLM・transformers（accelerate）の環境を`uv`で作る（`~/venv`）。版は`scripts/gpu/pins.sh`で固定している（ADR-0006）。
3. llama.cppを`pins.sh`のコミットにして、CUDA（L4はsm_89）でビルドする。**コミットハッシュは`~/out/llama_cpp_commit.txt`にすぐ書く**。destroyの後は二度と取れず、
   manifestの`engine_version`がこの形式（`^[0-9a-f]{7,40}$`）を求める。
4. `Qwen/Qwen3-8B`と`Qwen/Qwen3-8B-AWQ`をHugging Faceのキャッシュへダウンロードし、`Qwen/Qwen3-8B`のsafetensorsから
   BF16のGGUFを作る（[[docs/engine-performance-plan|計画書]]の「BF16 GGUF変換手順」。変換は別の環境`~/venv-convert`で行う。
   変換ツールの依存がCPU版のtorchを入れ、vLLMのものを置き換えてしまうため）。
5. 未検証の3点を確かめ、`~/out/checks/`に残す。
   - a. vLLMのバージョン → `~/out/versions.json`
   - b. `--tool-call-parser hermes`があるか → `checks/vllm_tool_call_parser.txt`（ログの最後に「hermes parser: listed」と出る）
   - c. llama.cppが`seed`を守るか → `checks/llama_seed.json`。同じseedで2回送った答えが違えば、llama.cppの実行は1回ごとに再現するとは
     みなさず、そのことを記事・結果に書く。
6. 5のaで`versions.json`を書いた直後に、`pins.sh`と突き合わせ、エンジンの版かモデルのリビジョンが1つでも違えば、準備を失敗させる（`SETUP_FAILED`に「not the pinned versions」）。
7. 終わると`~/out/SETUP_DONE`ができる。

あわせて、次を記録する（次のapplyの見積もりと、測定が壊れていないかの確認に使う）。

- `setup.log`の各段階の見出しに開始時刻。準備は毎回やり直すので、2回目以降が6時間に収まるかの見積もりに使う。
  ただし、準備の間だけ16GBのスワップを使う（下記）。`vmstat.log`の`si`/`so`が0でない時間帯があれば、その段階はスワップで遅くなっているので、見積もりにはそのまま使わない。
- `~/out/gpu.csv`：5秒ごとのGPUのメモリ・使用率・電力・温度・SMクロック（とクロックが下がった理由）。インスタンスが動いている間ずっと記録する。
  長い負荷で熱や電力の上限によりクロックが下がると、後のセルほどlatencyが悪く出るので、セルの間の比較が壊れていないかを後から確かめる。
- `~/out/vmstat.log`：5秒ごとのホストのメモリ。RAMは16GBで、bf16の重みとほぼ同じ大きさなので、落ちたときにメモリ不足かを切り分ける。
- `~/out/checks/disk.txt`：準備の後のディスク使用量。
- `status.jsonl`：セルごとに、サーバーの起動時刻・応答までの秒数（`server_ready_s`）・`run_cell.py`の開始時刻・終了時刻。
  `gpu.csv`と`vmstat.log`のどの時間帯がどのセルかは、この時刻で対応を付ける。
- `status.jsonl`の`kv_capacity_tokens`：vLLMの起動ごとのKVキャッシュの容量。2026-10-18の疎通では、新しいインスタンスで最初の2回の起動だけが
  ちょうど1GiB少なかった（原因不明、ADR-0005）。同じことが起きるかを、この値で確かめる。
  2026-10-22のapplyでも、最初の起動（capacity-smokeのvLLM AWQ）だけが92,240トークンで、その後の起動はすべて99,552だった。
- `~/out/<セル名>/engine-*`：エンジンが自分について報告する状態。`run_cell.py`の前後に`/metrics`（`engine-metrics-before.txt`・`-after.txt`。差がそのセルのスケジューラの数字で、
  llama.cppでは`n_decode_total`・`n_busy_slots_per_decode`・プロンプトと生成の処理時間）。前には実際に決まった設定も保存する：vLLMは開発用モードの
  `/server_info`（`engine-server_info-before.json`。`max_num_batched_tokens`など設定の全体。既定のログには出ない）、llama.cppは`/props`と`/slots`。
  llama-serverは`-lv 4`で起動し、実際に決まった設定（`resolve_fused_ops: Flash Attention enabled`・`n_batch`・`n_ubatch`・`kv_unified`）を`server.log`に残す
  （vLLMはattentionの実装などは既定のログに出るが、`max_num_batched_tokens`は出ない）。2026-10-22のpilot-1では、llama.cppのこれらとvLLMのチャンクの上限が残っておらず、並行数16が遅い理由をエンジンの側から確かめられなかった。
  方針は「エンジンが実際に使った設定と、スケジューラの数字の前後の差を、全エンジンで同じ粒度で残す」。読めなかったものは`status.jsonl`の`engine_state_failed`に出る。

記録はどれも読み取りだけで、結果のファイル（manifest・生出力）や採点には触れない。

**準備の間だけのスワップ**：ホストのRAMは16GBで、bf16の重み（約16.4GB）とほぼ同じ大きさ。llama.cppのCUDAビルドとGGUF変換でRAMが足りなくなると、
プロセスが強制終了され、準備をやり直すことになる。そこで`setup.sh`は、ディスク（200GBのうち使っていない部分）に16GBのスワップを作ってから
この2つを行い、準備の最後に外す。普通は使われない見込みで、保険である。測定中にスワップがあると、データの出し入れでlatencyが歪むので、
`run_cells.py`はスワップが有効なら1つもセルを回さずに止まる（`FAILED`に「swap is active」）。準備が途中で失敗してスワップが残った場合は、
`sudo swapoff -a`で外してから回す。

進み具合は`scripts/gpu/local.sh status $IP`で見る（自動シャットダウンまでの時刻も出る）。

## 3. セルを回す

```
scripts/gpu/local.sh cells $IP scripts/gpu/cells/smoke.json            # 1回目のapply：疎通
scripts/gpu/local.sh cells $IP scripts/gpu/cells/capacity-smoke.json   # 2回目のapply：まず容量の判定の動作確認（約10分）
scripts/gpu/local.sh cells $IP scripts/gpu/cells/pilot-1.json          # 2回目のapply：確認が通ったらパイロットの前半
scripts/gpu/local.sh cells $IP scripts/gpu/cells/pilot-2.json          # 3回目のapply：パイロットの後半（llama.cppの第3パス）
scripts/gpu/local.sh cells $IP scripts/gpu/cells/main-latency.json     # 本番1回目：速さの測定の18セル（約13.8時間）
scripts/gpu/local.sh cells $IP scripts/gpu/cells/main-accuracy.json    # 本番2回目：普通の精度の4セルと第3パスの5セル（約7.1時間）
```

**本番の一覧**（ADR-0007）：`scripts/gpu/make_main_cells.py`の出力で、手で書き換えない。順番は固定のseedでランダムにしてあり、引き直さない。
速さの測定は全部1台に入れ、比べるセルが同じ物理GPUの上になるようにしてある。載らない見込みの構成（vLLM bf16の並行数16、vLLM AWQとllama.cppの並行数32）も
一覧に残し、容量の事前の判定や起動の失敗が「対応していない」と記録する（それぞれ数分）。
回し終えたら、セルごとの電力・温度・クロック・熱による低下の有無（`gpu.csv`の`clocks_event_reasons`の`0x20`・`0x40`）を集計して報告する（ADR-0007のDecision 5）。

**2回目のapplyの順番**：容量の判定（ADR-0005）はGPUの上でまだ一度も動かしていないので、パイロットの前に`capacity-smoke.json`
（`smoke.json`の並行数16の2セル、20件×2試行）だけを回す。`collect`で回収し、次の3点を確かめてからパイロットに進む。
1つでも外れたら、パイロットには進まずdestroyして手元で直す。

1. 2セルとも`ok`。`~/out/<セル名>/capacity_check.json`の`capacity.tokens`が読めている（vLLM AWQは約99,552、llama.cppは53,248）。
   llama.cppは`llama_kv`が`split`、`capacity.per_slot_tokens`が3,328（3,104を256の倍数に切り上げた1スロットの長さ、ADR-0005の追記2026-10-23）。
   `split`で起動できず`unified`に切り替わった場合は、49,664と40,960になり、`server-split-startup.log`が残る（これも結果として記録する）。
   あわせて`checks/llama_seed.json`ができている（2026-10-21のapplyでは不具合で作られなかった）。
2. 2セルのmanifestに`capacity_check`があり、`post_check.ok`が`true`。vLLMは`preemptions`が0、`queue_time.count`が0でない。
3. llama.cppの`server_command.txt`が`--ctx-size 53248 --parallel 16 --no-kv-unified --fit off -lv 4 --metrics`
   （切り替わった場合は`--ctx-size 49664 --parallel 16 --kv-unified ...`）。

**パイロットは2回のapplyに分ける。** 1回に詰めると、6時間の自動シャットダウンまでの余りが約20分以下になる（下の「2回目・3回目：パイロット」）。
見積もりが外れるとセルが途中で切れ、そのセルは丸ごと回し直しになる。そこで、llama.cppの第3パスの2セルだけを`pilot-2.json`に分け、
3回目のapplyで回す。第3パスは同じエンジンの並行数1と16を比べるので、2セルが同じインスタンスにあれば足りる。並行数1のlatencyの3エンジンは
`pilot-1.json`にまとめて残す。あわせて、vLLM AWQの第3パスの結果（ADR-0002の分岐）を見てから3回目に進める。
ビルド済みのllama.cppを使い回して3回目の準備を短くする案は、2回目のapplyで作ったバイナリを回収して初めて使えるので、今回は扱わない。

`run_cells.py`は、一覧のセルごとにエンジンのサーバーを起動し、応答するまで待ち、`scripts/run_cell.py`を呼んで、サーバーを止める。

- **どのエンジンも、セルごとにサーバーを起動し直す。** llama.cppにはプロンプトのキャッシュを全部消すAPIが無いため
  （ADR-0003の追記A）。vLLMも同じにして、セルごとに起動ログを残す。latencyのセルでは、`run_cell.py`がさらにvLLMのprefix cacheをリセットする。
- **並行数が2以上のセルは、回す前に容量を確かめる**（ADR-0005）。サーバーの起動後に、vLLMは`/metrics`、llama.cppは`/props`からKVキャッシュの容量を読み、
  「並行数 × 1リクエストの最大（`schema/prompt_budget.json`のプロンプトの最大＋512）」が入るかを計算する。結果は`~/out/<セル名>/capacity_check.json`。
  vLLMで足りなければ1回だけ起動し直し（最初の起動のログは`server-short-startup.log`に残す）、それでも足りなければ回さずに`unsupported`と記録する。
  回した後は、vLLMは中断の回数（`vllm:num_preemptions`）、llama.cppは長さの上限で止まった応答の数が0かを確かめ、manifestの`capacity_check`に書く。
- `schema/prompt_budget.json`は、ケースとカタログから手元で数えた値（`scripts/count_prompt_tokens.py --tool-stage 10 --write-budget`）。
  ケースかカタログを変えたのに数え直していないと、`run_cells.py`は最初に止まる。
- vLLMは`VLLM_SERVER_DEV_MODE=1`で、`--enable-auto-tool-choice --tool-call-parser hermes --enable-prompt-tokens-details`を付けて起動する。
  `--max-model-len 8192`は、tool_stage 10のプロンプト（約2,200トークン、計画書の1リクエスト約306MiBから）と出力の上限512トークンに余裕を持たせた値。
  既定の40,960トークンだと、bf16の重みを載せた後のKVキャッシュに収まらず、起動できない可能性がある。
- llama.cppは`--jinja --no-kv-unified --fit off -lv 4 --metrics`で起動する。KVキャッシュはスロットごとに分かれる（`--parallel`を指定したときのllama.cppの既定）。
  `--parallel`はセルの並行数、`--ctx-size`は「1リクエストの最大」を256の倍数に切り上げた値 × 並行数で（llama.cppがスロットごとにそう切り上げるため）、
  セル一覧には書かない（ADR-0005：各エンジンは要件ぶんだけメモリを載せる）。並行数2以上でこの大きさで起動できなければ、1回だけ全スロットで共有する
  KVキャッシュ（`--kv-unified`、全体を1回だけ切り上げる）で起動し直す（ADR-0005の追記2026-10-23）。それでも起動できなければ`server_not_ready`で、
  その並行数は「対応していない」。どちらで回ったかは`status.jsonl`と`capacity_check.json`の`llama_kv`に残る。
- 1リクエストの時間の上限は、パスで意味が違う。latencyのパスでは測定の一部で（`--request-timeout`、既定120秒。超えた1件は基準を超えたと数えて続ける）、
  精度のパス（`accuracy`・`accuracy_under_load`）では固まったサーバーを見つけるためだけの上限（600秒、`ACCURACY_REQUEST_TIMEOUT_S`）。
  600秒は、出力の上限512トークン×llama.cppの並行数16の1トークン0.66秒（pilot-2の90パーセンタイル）＋プリフィル約2秒＝約340秒に余裕を持たせた値。
  以前は精度のパスも120秒で、pilot-2の`third-llama-gguf-c16`が178トークンの答え1件で止まった（2026-10-24）。
- transformers_hfはサーバーを使わず、`run_cell.py`の中でモデルを読む。
- 結果は`~/out/<セル名>/`に、manifest・生出力・サーバーのログ・実行したコマンドとともに残る。`~/out/status.jsonl`にはセルごとの結果
  （`ok`／`unsupported`＝ADR-0003の「対応していない」／`capacity_misestimate`＝回したが容量の事後の確かめが通らなかった／`error`／`server_not_ready`）を書く。
  失敗したセルがあっても次のセルへ進む。

### 1回目：疎通（`cells/smoke.json`、すべて`smoke-`・20件）

| セル | 確かめること |
| --- | --- |
| `acc-*`（vLLM bf16・vLLM AWQ・llama.cpp・transformers_hf） | 4つの構成が、それぞれ最後まで回るか |
| `load-vllm-awq-c16` | 容量の事前の判定（`/metrics`）と事後の確かめが動くか |
| `load-llama-gguf-c16` | 要件ぶんの`--ctx-size`で起動でき、`/props`から容量が読めるか |
| `lat-*-c1` | `ttft_ms`と`first_delta_ms`の差。生出力のキャッシュ済みトークン数（vLLMは`usage.prompt_tokens_details.cached_tokens`、llama.cppは`timings.cache_n`）が、起動直後のセルで0になっているか |

### 2回目・3回目：パイロット（`cells/pilot-1.json`・`pilot-2.json`、すべて`pilot-`・200件）

パイロットのデータは本番の判定に混ぜない（ADR-0002）。表の最後の行（`third-llama-gguf-*`）が`pilot-2.json`で、残りが`pilot-1.json`。

| セル | 決まること |
| --- | --- |
| `third-vllm-awq-c1`・`-c16`（greedy、200件×5試行） | 入れ替わったケースの数 → 案A（判定する）か案C（判定しない）か（ADR-0002の「パイロットの後の分岐」）。**結果：0件で案A**（ADR-0002の追記2026-10-22） |
| `lat-*-c1`（3エンジン、100リクエスト） | 1件あたりの所要時間 → 本番の件数n（299件以上）を、6時間に収まる範囲で固定する（ADR-0003）。**結果：n=1,000、ウォームアップ1周**（ADR-0003の追記2026-10-23） |
| `lat-vllm-awq-c16`・`lat-llama-gguf-c16`（300リクエスト） | 安定判定を通るか → 通らなければ窓を広げる（ADR-0003の追記B）。**結果：どちらも最初の窓から安定**。llama.cppは極端に遅く、KVキャッシュを分割に変えた（ADR-0005の追記2026-10-23） |
| `third-llama-gguf-c1`・`-c16`（greedy、200件×5試行） | llama.cpp bf16の入れ替わったケースの数（第3パスは並行数16で対応できる構成すべて、ADR-0002の追記2026-10-19）。**結果：並行数16が694/1,000件で精度のパスの上限（120秒）に止められ、数えられなかった。** 上限を600秒にした。本番の判定の規則は件数によらないので、変わる判断は無い（ADR-0005の追記2026-10-24） |

一覧から外したセル（ADR-0003の追記H）：vLLM bf16の並行数16・32とvLLM AWQの並行数32は事前の判定で「対応していない」、
llama.cppの並行数32は空きメモリに入らない見込み。本番では、事前の判定や起動の失敗がそのまま記録する。

安定判定の窓はケースセット1周分なので、200件ではウォームアップだけで最低600リクエストかかる。パイロット全体の所要時間の見込みは
約4.9〜5.5時間（推定。2つの見積もりがあり、どちらが近いかは2回目のapplyの実測で分かる）。そのうちllama.cppの第3パスは約1.3時間
（推定：並行数1は疎通の1件約3.6秒×1,000リクエストで約1時間、並行数16はそれより短い）なので、`pilot-1.json`は約3.6〜4.2時間、
2回目のapplyは準備と確認を足して約4.4〜5時間になる。
**2回目のapplyの実測（2026-10-22）**：applyから destroyまで約5.8時間（06:51→12:38Z）。Cost Explorerでは$6.81（g6.xlarge 5.68時間$6.63、EBS $0.15、IPv4 $0.03。
2026-10-23時点の見込み値）で、容量の疎通で止めた前日のapplyは$0.95。どちらもクレジットで相殺された。準備37分、capacity-smoke 11分、pilot-1は約4.8時間。
セルごとの所要時間（サーバーの起動を含む）：

| セル | 所要時間 | サーバーの起動 | 結果 |
| --- | --- | --- | --- |
| `third-vllm-awq-c1` | 33分 | 120秒 | ok |
| `third-vllm-awq-c16` | 15分 | 55秒 | ok |
| `lat-vllm-bf16-c1` | 51分 | 220秒 | ok |
| `lat-llama-gguf-c1` | 55分 | 165秒 | ok |
| `lat-hf-bf16-c1` | 55分 | （サーバーなし） | ok |
| `lat-vllm-awq-c16` | 15分 | 115秒 | ok |
| `lat-llama-gguf-c16` | 64分 | 170秒 | `capacity_misestimate`（誤検知。ADR-0005の追記2026-10-23） |

latencyのセルは、どれもウォームアップが最低の600件（3窓）で、所要時間の6割前後を占めた。本番ではウォームアップを1周（200件）にし、計測を1,000件にする
（ADR-0003の追記2026-10-23）。上の見込み（約4.9〜5.5時間）より短く済んだのは、主にllama.cppの第3パスを`pilot-2.json`に分けたためである。

**3回目のapplyの実測（2026-10-24）**：applyからdestroyまで約2.6時間（03:43→06:20Z）。費用は単価からの見積もりで約$3.1〜3.3（Cost Explorerの実額は未確認）。
準備36分、capacity-smoke 11分（3点とも通過。llama.cppの並行数16は分割KVで起動）。

| セル | 所要時間 | サーバーの起動 | 結果 |
| --- | --- | --- | --- |
| `third-llama-gguf-c1` | 76分 | 160秒 | ok（1,000件） |
| `third-llama-gguf-c16` | 21分 | 145秒 | `error`（694件で、1件が178トークンを生成中に120秒の上限に達した） |

並行数16の遅さの原因（プリフィルが律速、電力の上限、統合KV）は、ADR-0005の追記2026-10-24にまとめた。
vLLMの`/server_info`が500を返していた（環境にpipが無く、vLLMの`collect_env`が落ちる）ので、`setup.sh`で環境にpipを入れた。次のapplyの最初のvLLMのセルで、
`engine_state_failed`が空で`engine-server_info-before.json`ができていることを確かめる。

### 本番1回目：速さの測定（`cells/main-latency.json`、18セル・1,000件）

**実測（2026-10-24〜25）**：apply 13:59Z → destroy 02:40Z（インスタンスは約12.7時間）。準備35分。セルは14:35Z→02:36Zの12.0時間で、見込み（約13.8時間）より短かった。
自動シャットダウンは1080分で、約5.3時間の余りがあった。費用は未確認（Cost Explorerで確かめる）。

- 15セルが`ok`。`lat-llama-gguf-c32`は`server_not_ready`（5分）、`lat-vllm-awq-c32`と`lat-vllm-bf16-c16`は事前の判定で`unsupported`（3〜6分）で、どれも見込みどおり。
- vLLMの全セルで`engine_state_failed`が空で、`engine-server_info-before.json`ができた（`setup.sh`でpipを入れた修正が効いた）。
- 速さの測定の上限（1件120秒）を超えたのは、llama.cppの並行数8で5件、並行数16で19件（基準を超えたと数えて続ける決まりどおり）。
- 安定の確かめで3セルが`false`になったが、窓の切り方によるもので、送った順の窓に直して計算し直した（ADR-0003の追記2026-10-25）。
- GPU（`gpu.csv`をセルの計測区間で切ったもの）：電力の中央値は全セルで70〜72W（L4の上限）、クロックが下がった理由の98〜100%が電力の上限（`0x4`）。
  熱による低下（`0x20`・`0x40`）は全セルで0回、温度の最大は83〜85℃。SMクロックの中央値は並行数1で1,440〜1,680MHz、並行数8〜16で915〜1,065MHzで、
  負荷によって変わるが、後のセルほど下がる傾きは無かった（ADR-0007のDecision 5）。

**回収先は毎回指定する。** `local.sh collect`は回収先を省くと`runs/gpu-<今日の日付>`を使う。同じ日に2回applyすると、前のapplyの回収先と重なり、
`gpu.csv`・`vmstat.log`・`setup.log`・`versions.json`・`checks/`などの共通のファイルが上書きされる（2026-10-24に、pilot-2のこれらのファイルを本番1回目の回収で失った。
セルのフォルダと`status.jsonl`は残った）。本番2回目は`scripts/gpu/local.sh collect $IP runs/<名前>`と、新しいフォルダを渡す。

`pilot-1.json`も大事な順（vLLM AWQの第3パス → 並行数1 → 並行数16）に並べてある。それでも6時間に収まらなかったセルは、3回目のapplyで
`pilot-2.json`の後に、名前を指定して回す（`&`が切り離すコマンドだけにかかる形にする。上の「SSHが戻らないとき」）：

```
ssh -n -i ~/.ssh/id_ed25519 ubuntu@$IP "setsid nohup ~/venv/bin/python ~/toolcall-serving-bench/scripts/gpu/run_cells.py ~/toolcall-serving-bench/scripts/gpu/cells/pilot-1.json --only <残ったセル名> > ~/out/cells-rest.log 2>&1 < /dev/null &"
```

## 4. 疎通の後：パイロットに進むか、止めるか

疎通（`smoke.json`）が終わったら、次をすべて満たすときだけ2回目のapplyでパイロットに進む。1つでも外れたら、パイロットには進まず、
destroyしてから原因を手元で調べる（GPUの時間を、原因の分からないまま使わないため）。

1. `versions.json`があり、`checks/vllm_tool_call_parser.txt`に`hermes`がある。
2. `acc-*`の4セルが`ok`。どれかの`selection_mean`が0に近い、または1つだけ他より大きく低い場合は、生出力（`*.raw.jsonl`の`raw_output`）を見て、
   パーサーやチャットテンプレートの不一致でないかを確かめる。モデルの実力ではなく配線の誤りなら、その構成の結果は使えない。
3. `run_cell.log`に、thinkingのガード（`assert_no_thinking`）で止まった記録が無い。
4. `load-vllm-awq-c16`のmanifestに`capacity_check`があり、`post_check.ok`が`true`。
5. `lat-*-c1`の生出力で、起動直後のセルのキャッシュ済みトークン数が0。

`load-llama-gguf-c16`が`server_not_ready`や`unsupported`になるのは、止める理由にしない。llama.cppがその並行数で動かないこと自体が
ADR-0003の結果（「対応していない」）である。その場合は、パイロットのllama.cppの並行数16のセル（`pilot-1.json`のlatencyと`pilot-2.json`の第3パス）を外す（外したことはPRに書く）。

`checks/llama_seed.json`の`same_seed_identical`が`false`でも止めない。llama.cppの結果は1回ごとに再現するとはみなさない、と記録する。

## 5. 止まったとき

| 症状 | 見る場所・対処 |
| --- | --- |
| `collect`が「setup.sh failed」で止まった | `~/out/SETUP_FAILED`（失敗した行とコマンド）と`setup.log`の最後。`setup.sh`は済んだ処理を飛ばすので、直してからもう一度`local.sh setup` |
| `server_not_ready` | `~/out/<セル名>/server.log`（メモリ不足・フラグの誤りなど）と`server_command.txt` |
| `error` | `~/out/<セル名>/run_cell.log`。manifestのガードで止まった場合も、生出力は先に書かれている |
| `timeout` | セルの時間の上限（既定90分、セル一覧の`timeout_min`）を超えた。サーバーが固まったか、見積もりより遅い。`server.log`と`run_cell.log`の最後の時刻を見る。上限は実測した時間ではないので、遅いだけなら`timeout_min`を上げて`--only`で回し直す |
| `unsupported` | 事前の判定で外れた場合は`status.jsonl`の`reason`が「capacity pre-check」で、`~/out/<セル名>/capacity_check.json`に容量と必要量がある。回した後に打ち切った場合は`*.aborted.json`の`reason`。どちらもADR-0003では結果として記録する（やり直さない） |
| `capacity_misestimate` | 回したが、容量の事後の確かめが通らなかった（vLLMの中断が起きた、llama.cppが長さの上限で止まった、実際のプロンプトが数えた値より長かった）。manifestの`capacity_check.post_check.problems`を見る。数字は判定に使わず、原因を調べるまで同じ構成の高い並行数は回さない（ADR-0005） |
| 最初に`prompt_budget.json ... was counted against another catalog or case set`で止まった | ケースかカタログを変えたのに数え直していない。手元で`scripts/count_prompt_tokens.py --tool-stage 10 --write-budget`を回してコミットし、pushし直す |
| `collect`が「the cell list stopped early」で止まった | `~/out/FAILED`。セルの後にGPUのメモリが空かなかった（残りのセルがすべて起動できなくなるので、そこで止めた）か、`run_cells.py`自体が落ちた。`nvidia-smi`で残ったプロセスを確かめて消し、`--only`で残りを回す |
| `collect`が「unreachable for 10 minutes」で止まった | インスタンスが消えた（`local.sh status`が通らず、`terraform -chdir=infra/gpu show`にも無い）か、回線のIPが変わった。下を見る |
| `collect`がほかの理由で止まった | もう一度`local.sh collect $IP`。rsyncなので、続きからコピーする |

**回線のIPが途中で変わったとき**：インスタンス上の処理はそのまま続いている。新しいIPを渡してapplyし直すと、セキュリティグループの許可だけが
置き換わる（インスタンスは作り直されない。planで`aws_instance`が`replace`になっていないことを確かめてからyesと打つ）。その後、`collect`をもう一度打つ。

```
terraform -chdir=infra/gpu apply -var "ssh_allowed_cidr=$(curl -4 -s https://checkip.amazonaws.com)/32"
```

**自動シャットダウンに間に合わないとき**：`local.sh status`の「--- shutdown」の下が自動シャットダウンの予定（`USEC=`の形なら、UNIX時刻のマイクロ秒）。間に合わないセルは、回り終わったものを`collect`で
回収した後に、次のapplyで`--only`を付けて回す。動いているインスタンスのシャットダウンの時刻は延ばさない（課金の歯止めのため）。
延ばすのは、本番のapplyのときに上の値を渡すときだけにする（ADR-0007）。

## 6. destroy（手で行う）

`collect`が「collected into …」を出したのを確かめてから、自動シャットダウンを待たずにdestroyする
（合意済みの運用：apply → 実行 → 確認 → すぐdestroy）。

```
terraform -chdir=infra/gpu destroy
```

destroyの後、動いていた時間（インスタンスの`LaunchTime`から終了まで）×単価で費用を見積もり、翌日以降にCost Explorerの実額で確かめる。
ADR-0003の費用の概算は、この実測で置き換える。

[Source: User, GPUインスタンス起動手順書の整備, 2026-09-12]
[Source: User, GPUでの実行を手動の操作なしでスクリプトにする, 2026-10-17]
