# GPUインスタンス起動時の手順書

`infra/gpu/`で`terraform apply`した後、SSH接続してから行う手順。`terminate`+6時間の自動
シャットダウン(`var.auto_shutdown_minutes`)で確実にインスタンスとEBSが消えるため、**復元不能な
情報は取得した瞬間にscpでローカルへ回収する**（セル終了ごと・完了を待たない）。

## 0. 接続

```
ssh -i ~/.ssh/id_ed25519 ubuntu@<gpu_public_ip>   # terraform outputのgpu_ssh_command
```

## 1. AMIの前提を最初に確認（ここで問題が出たら他の作業をする前に対処する）

```
nvcc --version
nvidia-smi
```

`nvcc`が無ければllama.cppのCUDAビルドができない。AMI選定(`base-oss-nvidia-driver-gpu-ubuntu-22.04`)
の前提が崩れているということなので、他の作業に進む前にここで気づく。

## 2. 自動シャットダウンタイマーが実際に効いているか確認

```
shutdown --show
```

`user_data`が失敗しているとバックストップが機能しないまま気づかず走り続けることになるので、
早い段階で確認する。6時間はboot時点からのカウント（ビルド・ダウンロード時間も含まれる）。

## 3. リポジトリを取得し、llama.cppのコミットハッシュを即座に記録

```
git clone https://github.com/yuya0408/toolcall-serving-bench.git
cd toolcall-serving-bench

git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git rev-parse HEAD > ~/llama_cpp_commit.txt
cat ~/llama_cpp_commit.txt
```

**このハッシュはdestroy後は二度と取得できない。** `schema/run_manifest.schema.json`が
`engine_version`にコミットハッシュ形式(`^[0-9a-f]{7,40}$`)を要求し、
`eval_bench/runner.py`の`_validate_cell_invariants`が強制する。取得したら即ローカルへ回収:

```
# ローカル(WSL)側で実行
scp ubuntu@<gpu_public_ip>:~/llama_cpp_commit.txt ./
```

## 4. 未検証3点の確認

先に進む前にここで確定させる。結果はdocs/engine-performance-plan.mdに反映すること。

**a. vLLMのバージョン下限**
```
pip index versions vllm   # もしくは pip install vllm==<候補> で試す
vllm serve --help | grep -A3 tool-call-parser
```

**b. `--tool-call-parser hermes`がQwen3-8Bに正しいか**
```
vllm serve --help | grep -i parser
```
上記コマンドで対応パーサー一覧を確認し、`hermes`がQwen3系に妥当か照合する。

**c. llama.cppのOpenAI互換エンドポイントが`seed`を尊重するか**
```
# llama.cppサーバー起動後、同一seedで2回リクエストして出力が一致するか比較
curl -s localhost:8080/v1/chat/completions -d '{"seed": 42, ...}' 
```
尊重しない場合、run manifestにその旨を記録し、決定的な再現性を仮定しないこと。

## 5. vLLM起動ログは生テキストのまま保存・回収

```
vllm serve ... --enable-auto-tool-choice --tool-call-parser hermes 2>&1 | tee ~/vllm-startup.log
```

`eval_bench/runner.py`の`parse_vllm_diagnostics`（`# GPU blocks`・`Maximum concurrency`をパース）
は実機ログ未検証。正規表現が合わなかった時にオフラインで直せるよう、生ログをそのままscpで回収:

```
scp ubuntu@<gpu_public_ip>:~/vllm-startup.log ./
```

## 6. manifestはセルごとに回収（最後にまとめてではない）

```
scp ubuntu@<gpu_public_ip>:~/toolcall-serving-bench/runs/<manifest>.json ./runs/
```

セル完了のたびに実行する。6時間経過でインスタンスがterminateされると、EBS上の未回収データは
すべて失われる。

## 7. 実験終了後

作業が全て終わったら、シャットダウンタイマーを待たずに手動で`terraform destroy`する
（[[project_tool_call_bench_todo]]の合意済み運用方針: apply→run→確認→即destroy）。
