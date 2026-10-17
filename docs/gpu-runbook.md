# GPUインスタンスでの実行手順書

[[docs/cloud-layer-plan|クラウド層の設計方針]]に基づき、`infra/gpu/`で`terraform apply`してから`destroy`するまでの手順。
インスタンス上の作業（準備・サーバーの起動と再起動・セルの実行）は`scripts/gpu/`のスクリプトがやり、手で打つのは下の6つのコマンドだけにする。
手で繰り返すと、llama.cppの再起動忘れや回収漏れのように、データを黙って壊す間違いが起きるためである。

インスタンスは`terminate`と6時間の自動シャットダウン（`var.auto_shutdown_minutes`）で、EBSごと確実に消える。
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

`InsufficientInstanceCapacity`で失敗したら、`infra/gpu/instance.tf`のサブネットのindexを変えて、applyし直す（L4の在庫が無いAZ）。

## 2. 準備

```
scripts/gpu/local.sh push  $IP
scripts/gpu/local.sh setup $IP
scripts/gpu/local.sh collect $IP &      # 別のターミナルで回し続けてもよい。DONEが出るまで1分ごとにコピーする
```

`setup.sh`（`~/out/setup.log`）がやること：

1. **AMIの前提の確認**：`nvidia-smi`と`nvcc --version`。`nvcc`が無ければllama.cppのCUDAビルドができないので、ここで止まる
   （AMI `base-oss-nvidia-driver-gpu-ubuntu-22.04`の前提が崩れたということ）。自動シャットダウンの予定（`shutdown --show`）も記録する。
2. vLLM・transformers（accelerate）の環境を`uv`で作る（`~/venv`）。
3. llama.cppをCUDA（L4はsm_89）でビルドする。**コミットハッシュは`~/out/llama_cpp_commit.txt`にすぐ書く**。destroyの後は二度と取れず、
   manifestの`engine_version`がこの形式（`^[0-9a-f]{7,40}$`）を求める。
4. `Qwen/Qwen3-8B`と`Qwen/Qwen3-8B-AWQ`をHugging Faceのキャッシュへダウンロードし、`Qwen/Qwen3-8B`のsafetensorsから
   BF16のGGUFを作る（[[docs/engine-performance-plan|計画書]]の「BF16 GGUF変換手順」。変換は別の環境`~/venv-convert`で行う。
   変換ツールの依存がCPU版のtorchを入れ、vLLMのものを置き換えてしまうため）。
5. 未検証の3点を確かめ、`~/out/checks/`に残す。
   - a. vLLMのバージョン → `~/out/versions.json`
   - b. `--tool-call-parser hermes`があるか → `checks/vllm_tool_call_parser.txt`（ログの最後に「hermes parser: listed」と出る）
   - c. llama.cppが`seed`を守るか → `checks/llama_seed.json`。同じseedで2回送った答えが違えば、llama.cppの実行は1回ごとに再現するとは
     みなさず、そのことを記事・結果に書く。
6. 終わると`~/out/SETUP_DONE`ができる。

進み具合は`scripts/gpu/local.sh status $IP`で見る（自動シャットダウンまでの時刻も出る）。

## 3. セルを回す

```
scripts/gpu/local.sh cells $IP scripts/gpu/cells/smoke.json     # 1回目のapply：疎通
scripts/gpu/local.sh cells $IP scripts/gpu/cells/pilot.json     # 2回目のapply：パイロット
```

`run_cells.py`は、一覧のセルごとにエンジンのサーバーを起動し、応答するまで待ち、`scripts/run_cell.py`を呼んで、サーバーを止める。

- **どのエンジンも、セルごとにサーバーを起動し直す。** llama.cppにはプロンプトのキャッシュを全部消すAPIが無いため
  （ADR-0003の追記A）。vLLMも同じにして、セルごとに起動ログを残す（並行数が1を超えるセルでは、キューイングのガードがこのログを読む）。
  latencyのセルでは、`run_cell.py`がさらにvLLMのprefix cacheをリセットする。
- vLLMは`VLLM_SERVER_DEV_MODE=1`で、`--enable-auto-tool-choice --tool-call-parser hermes --enable-prompt-tokens-details`を付けて起動する。
  `--max-model-len 8192`は、tool_stage 10のプロンプト（約2,200トークン、計画書の1リクエスト約306MiBから）と出力の上限512トークンに余裕を持たせた値。
  既定の40,960トークンだと、bf16の重みを載せた後のKVキャッシュに収まらず、起動できない可能性がある。
- llama.cppは`--jinja`で起動する。並行数16・32のセルは`--parallel 16/32 --kv-unified --ctx-size 40960`。これでメモリに載るかは疎通で確かめる。
- transformers_hfはサーバーを使わず、`run_cell.py`の中でモデルを読む。
- 結果は`~/out/<セル名>/`に、manifest・生出力・サーバーのログ・実行したコマンドとともに残る。`~/out/status.jsonl`にはセルごとの結果
  （`ok`／`unsupported`＝ADR-0003の「対応していない」／`error`／`server_not_ready`）を書く。失敗したセルがあっても次のセルへ進む。

### 1回目：疎通（`cells/smoke.json`、すべて`smoke-`・20件）

| セル | 確かめること |
| --- | --- |
| `acc-*`（vLLM bf16・vLLM AWQ・llama.cpp・transformers_hf） | 4つの構成が、それぞれ最後まで回るか |
| `load-vllm-awq-c16` | vLLMの起動ログが読めて、キューイングのガードが効くか |
| `load-llama-gguf-c16` | `--parallel 16 --kv-unified`がメモリに載るか |
| `lat-*-c1` | `ttft_ms`と`first_delta_ms`の差。生出力のキャッシュ済みトークン数（vLLMは`usage.prompt_tokens_details.cached_tokens`、llama.cppは`timings.cache_n`）が、起動直後のセルで0になっているか |

### 2回目：パイロット（`cells/pilot.json`、すべて`pilot-`・200件）

パイロットのデータは本番の判定に混ぜない（ADR-0002）。

| セル | 決まること |
| --- | --- |
| `third-vllm-awq-c1`・`-c16`（greedy、200件×5試行） | 入れ替わったケースの数 → 案A（判定する）か案C（判定しない）か（ADR-0002の「パイロットの後の分岐」） |
| `lat-*-c1`（3エンジン、100リクエスト） | 1件あたりの所要時間 → 本番の件数n（299件以上）を、6時間に収まる範囲で固定する（ADR-0003） |
| `lat-*-c16`・`-c32`（vLLM・llama.cpp、300リクエスト） | 安定判定を通るか → 通らなければ窓を広げる（ADR-0003の追記B） |

安定判定の窓はケースセット1周分なので、200件ではウォームアップだけで最低600リクエストかかる。所要時間はパイロットの実測を待つ。
6時間に収まらなかったときに備えて、一覧は大事な順（第3パス → 並行数1 → 16・32）に並べてある。
残ったセルは、次のapplyで準備の後に、名前を指定して回す：

```
ssh -i ~/.ssh/id_ed25519 ubuntu@$IP "cd ~/toolcall-serving-bench && nohup ~/venv/bin/python scripts/gpu/run_cells.py scripts/gpu/cells/pilot.json --only lat-vllm-bf16-c32 lat-llama-gguf-c32 > ~/out/cells-rest.log 2>&1 < /dev/null &"
```

## 4. 疎通の後：パイロットに進むか、止めるか

疎通（`smoke.json`）が終わったら、次をすべて満たすときだけ2回目のapplyでパイロットに進む。1つでも外れたら、パイロットには進まず、
destroyしてから原因を手元で調べる（GPUの時間を、原因の分からないまま使わないため）。

1. `versions.json`があり、`checks/vllm_tool_call_parser.txt`に`hermes`がある。
2. `acc-*`の4セルが`ok`。どれかの`selection_mean`が0に近い、または1つだけ他より大きく低い場合は、生出力（`*.raw.jsonl`の`raw_output`）を見て、
   パーサーやチャットテンプレートの不一致でないかを確かめる。モデルの実力ではなく配線の誤りなら、その構成の結果は使えない。
3. `run_cell.log`に、thinkingのガード（`assert_no_thinking`）で止まった記録が無い。
4. `load-vllm-awq-c16`のmanifestに`vllm_diagnostics`があり、ガードで止まっていない。
5. `lat-*-c1`の生出力で、起動直後のセルのキャッシュ済みトークン数が0。

`load-llama-gguf-c16`が`server_not_ready`や`unsupported`になるのは、止める理由にしない。llama.cppがその並行数で動かないこと自体が
ADR-0003の結果（「対応していない」）である。その場合は、パイロットのllama.cppの並行数16・32のセルを外す（外したことはPRに書く）。

`checks/llama_seed.json`の`same_seed_identical`が`false`でも止めない。llama.cppの結果は1回ごとに再現するとはみなさない、と記録する。

## 5. 止まったとき

| 症状 | 見る場所・対処 |
| --- | --- |
| `collect`が「setup.sh failed」で止まった | `~/out/SETUP_FAILED`（失敗した行とコマンド）と`setup.log`の最後。`setup.sh`は済んだ処理を飛ばすので、直してからもう一度`local.sh setup` |
| `server_not_ready` | `~/out/<セル名>/server.log`（メモリ不足・フラグの誤りなど）と`server_command.txt` |
| `error` | `~/out/<セル名>/run_cell.log`。manifestのガードで止まった場合も、生出力は先に書かれている |
| `timeout` | セルの時間の上限（既定90分、セル一覧の`timeout_min`）を超えた。サーバーが固まったか、見積もりより遅い。`server.log`と`run_cell.log`の最後の時刻を見る。上限は実測した時間ではないので、遅いだけなら`timeout_min`を上げて`--only`で回し直す |
| `unsupported` | `~/out/<セル名>/*.aborted.json`の`reason`。ADR-0003では結果として記録する（やり直さない） |
| `collect`が「the cell list stopped early」で止まった | `~/out/FAILED`。セルの後にGPUのメモリが空かなかった（残りのセルがすべて起動できなくなるので、そこで止めた）か、`run_cells.py`自体が落ちた。`nvidia-smi`で残ったプロセスを確かめて消し、`--only`で残りを回す |
| `collect`が「unreachable for 10 minutes」で止まった | インスタンスが消えた（`local.sh status`が通らず、`terraform -chdir=infra/gpu show`にも無い）か、回線のIPが変わった。下を見る |
| `collect`がほかの理由で止まった | もう一度`local.sh collect $IP`。rsyncなので、続きからコピーする |

**回線のIPが途中で変わったとき**：インスタンス上の処理はそのまま続いている。新しいIPを渡してapplyし直すと、セキュリティグループの許可だけが
置き換わる（インスタンスは作り直されない。planで`aws_instance`が`replace`になっていないことを確かめてからyesと打つ）。その後、`collect`をもう一度打つ。

```
terraform -chdir=infra/gpu apply -var "ssh_allowed_cidr=$(curl -4 -s https://checkip.amazonaws.com)/32"
```

**6時間に間に合わないとき**：`local.sh status`の「--- shutdown」の下が自動シャットダウンの予定（`USEC=`の形なら、UNIX時刻のマイクロ秒）。間に合わないセルは、回り終わったものを`collect`で
回収した後に、次のapplyで`--only`を付けて回す。シャットダウンの時刻は延ばさない（課金の歯止めのため）。

## 6. destroy（手で行う）

`collect`が「collected into …」を出したのを確かめてから、自動シャットダウンを待たずにdestroyする
（合意済みの運用：apply → 実行 → 確認 → すぐdestroy）。

```
terraform -chdir=infra/gpu destroy
```

[Source: User, GPUインスタンス起動手順書の整備, 2026-09-12]
[Source: User, GPUでの実行を手動の操作なしでスクリプトにする, 2026-10-17]
