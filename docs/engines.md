# エンジンの違い

このプロジェクトで測るエンジンの仕組みの違いと、測るときの扱いの違いをまとめる。数値の判定には使わない参考の文書で、
扱いの決定はADR（特にADR-0003・0005・0008・0009）にある。版は`scripts/gpu/pins.sh`のもの。出典は各行に書き、ソースを読んで確かめたものは
ファイルの場所を書いた。疎通で確かめていないものは「（疎通で確かめる）」と書いた。

## 系統

| エンジン | 版 | 系統 | 開発 | このプロジェクトでの位置づけ |
| --- | --- | --- | --- | --- |
| transformers | 5.17.0 | 素の実装（`generate()`）。複数のリクエストをまとめる仕組みが無い | Hugging Face | 何も最適化しない基準。並行数1だけ |
| llama.cpp（llama-server） | `848bbda` | C++の実装。スロット（並行数ぶんの席）ごとにKVキャッシュを持つ。連続バッチングもある | ggml-org | 1リクエスト向けの系統の代表（第1段階） |
| vLLM | 0.31.0 | 連続バッチングのサーバー。PagedAttention（KVキャッシュをブロック単位で貸し出す） | vLLMプロジェクト | 連続バッチングの系統（第1・第2段階） |
| SGLang | 0.5.21 | 連続バッチングのサーバー。RadixAttention（KVキャッシュをプレフィックスの木で共有する） | sgl-project | 第2段階（ADR-0009） |
| TensorRT-LLM（trtllm-serve） | 1.2.1 | 連続バッチングのサーバー（インフライトバッチング）。NVIDIAのGPU向けに最適化したカーネル。PyTorchの経路で動かす | NVIDIA | 第2段階（ADR-0009） |

TensorRT-LLMには、モデルをGPUに合わせて事前に「エンジン」へビルドする経路（TensorRTの経路）もある。1.x の既定はビルドの要らないPyTorchの経路で
（`tensorrt_llm/commands/serve.py`の`--backend`、既定`pytorch`）、このプロジェクトもそれを使う。ビルドの経路の方が速い可能性は残る（ADR-0009）。

## リクエストの受け入れ方（スケジューラ）

KVキャッシュは、リクエストごとに「これまでのトークン」のキーと値を覚えておくGPUメモリで、答えを生成するほど使う量が増える。
新しいリクエストを受け入れるとき、この増え方をどう見込むかがエンジンで違う。

| エンジン | 受け入れ方 | 足りなくなったとき | 回した後の確かめ（ADR-0005） |
| --- | --- | --- | --- |
| vLLM | 今の空きで受け入れる（楽観的） | 走っているリクエストを一度外し（preemption）、後で計算し直す | `vllm:num_preemptions`が0か |
| SGLang | 今の空きで受け入れる（楽観的。`schedule_conservativeness`で調整できる） | 走っているリクエストを待ちの列に戻す（retraction） | `sglang:num_retracted_requests_total`が0か（このカウンタは1回目の追い出しまで`/metrics`に出ない。出ていなくて、同じ系統のゲージ`sglang:num_retracted_reqs`が出ていれば0回とみなす。0.5.21の`metrics_collector.py`） |
| TensorRT-LLM | 既定の`GUARANTEED_NO_EVICT`：最大の長さ（プロンプト＋最大の出力）ぶんを先に確保できたときだけ受け入れる（`llmapi/llm_args.py`の`CapacitySchedulerPolicy`） | 起きない。入れないリクエストは列で待つ | 文脈の長さで切れた答えが無いか。加えて、セルの間`/metrics`を2秒ごとに読み、同時に走ったリクエスト数（`numActiveRequests`の最大）が並行数に届いたか（読むたびに統計が消えるので、最後に1回では読めない。ADR-0009の2） |
| llama.cpp | スロットの数（`--parallel`）まで受け入れる。KVは起動時にスロットごとに確保 | 起きない（スロットの文脈を超えると答えが切れる） | 文脈の長さで切れた答えが無いか |

このプロジェクトは、並行数のセルを回す前に「全リクエストが最大の長さまで伸びても入るか」を全エンジン共通の式で確かめ、入らないセルは回さない（ADR-0005）。
そのため、どのエンジンでも追い出しは本来起きない。起きたら、見込みが外れたとしてそのセルの数値を判定に使わない。

## KVの容量の読み方（事前の確かめ、ADR-0005）

| エンジン | どこから読むか | 単位 | 出典 |
| --- | --- | --- | --- |
| vLLM | `/metrics`の`vllm:cache_config_info`（`num_gpu_blocks`×`block_size`） | ブロック（16トークン） | `vllm/v1/metrics/loggers.py`（0.31.0） |
| SGLang | `/metrics`の`sglang:max_total_num_tokens`と`sglang:page_size`。`--enable-metrics`が要る | ページ | `srt/observability/metrics_collector.py`（0.5.21） |
| TensorRT-LLM | `/metrics`（反復ごとの統計のJSON）の`kvCacheStats.maxNumBlocks`×`tokensPerBlock`。`enable_iter_perf_stats`が要り、1件のリクエストを処理するまで空なので、短いリクエストを1件送ってから読む。チャットの書式を通すと先頭の1トークン（`<|im_start|>`）が測定の質問と共有され、空のキャッシュから始まらなくなるので、書式を通さない`/v1/completions`で送る（疎通の1回目で分かった） | ブロック | `cpp/tensorrt_llm/executor/jsonSerialization.cpp`（1.2.1） |
| llama.cpp | 起動時の`--ctx-size`と`/props`の`n_ctx` | トークン | ADR-0005の追記 |

KVに割り当てるメモリの割合は、SGLang以外は既定のまま（vLLMの`--gpu-memory-utilization`は0.92〔本番1回目の`/metrics`〕、
TensorRT-LLMの`free_gpu_memory_fraction` 0.9〔読み込み後の空きに対する割合〕）。そのため、同じGPUでも容量はエンジンで違い、それも測る対象に含まれる。

SGLangだけは`--mem-fraction-static`を0.86に固定する（全チェックポイント共通）。

- **既定で起動しなかった**：自動の値0.684は、プリフィル用に`chunked_prefill_size`×1.5（約6GB）を残す見積もり（`arg_groups/memory_hook.py`、0.5.21）で、
  bf16では重みを載せるとKVの置き場が無く起動しなかった（疎通の1回目、2026-10-28。最低でも0.702が要るとエラーが示した）。
- **値の決め方**：2026-10-18の疎通で決めた「要件を満たすように1エンジンにつき1つの設定を動かしてよい」に従う。vLLMの既定でbf16が載る並行数8
  （8×3,103＋1＝24,825トークン）が載る最小の値を、0.01刻みで実測した。0.84は22,047トークン、0.86は25,224トークン（KV以外の残りは1.40GB）で、
  0.85は差から約23,600トークンと見込み、足りない。
- **AWQ・FP8も同じ値**：チェックポイントごとに値を変えると、同じエンジンの中で条件がそろわないため。既定の0.684で起動していたが、0.86ではKVが増える。
- **残り1.40GBで並行数8のプリフィルに足りるか**は、疎通の流し直しに並行数8のセルを足して確かめる。

## プレフィックスキャッシュ

| エンジン | 仕組み | 既定 | セルの境目での消し方 | 当たった量の読み方 |
| --- | --- | --- | --- | --- |
| vLLM | ブロック単位のハッシュで共有 | 有効 | `POST /reset_prefix_cache`（開発モード）＋サーバーの再起動 | `usage.prompt_tokens_details.cached_tokens`（`--enable-prompt-tokens-details`） |
| SGLang | RadixAttention（プレフィックスの木） | 有効 | `POST /flush_cache`＋サーバーの再起動 | 同じ項目（`--enable-cache-report`）。当たった量が0のときは項目ごと出さない（0.5.21の`usage_processor.py`）ので、無ければ0とする |
| TensorRT-LLM | ブロックの再利用（`enable_block_reuse`） | 有効（`KvCacheConfig`の既定） | サーバーの再起動 | 同じ項目。ストリームで返るかは疎通で確かめる |
| llama.cpp | スロットごとに直前のプロンプトを再利用。ホストのRAMにも退避する | 有効 | サーバーの再起動（全部を消すAPIが無い） | `timings.cache_n` |

このワークロードはリクエストごとにツール定義が変わるので、どのエンジンでもほとんど当たらない（実測2〜9%、ADR-0003の追記2026-10-24）。
SGLangのRadixAttentionの強みは、この条件では出にくい。

## 量子化

| 方式 | vLLM | SGLang | TensorRT-LLM | 中身 |
| --- | --- | --- | --- | --- |
| bf16 | `Qwen/Qwen3-8B` | 同じ | 同じ | 元の重み |
| 4bit（AWQ、W4A16） | `Qwen/Qwen3-8B-AWQ` | 同じ（疎通で確かめる） | **読めない**。1.2.1の読み込み処理に`awq`の分岐が無い（`_torch/model_config.py`の`load_hf_quant_config`）。ModelOptの`hf_ptq.py --qformat int4_awq`で同じbf16から作る | 重みを4bitに、計算は16bit。重要な重みを守るため、調整用のデータで倍率を決める |
| FP8（W8A8） | `Qwen/Qwen3-8B-FP8`（128×128のブロック単位の倍率、動的な活性） | 同じ（疎通で確かめる） | **ブロック単位は対応外**（対応表でQwen-3もAda Lovelaceも`.`、`docs/source/features/quantization.md`）。ModelOptの`--qformat fp8`（テンソル単位）で同じbf16から作る | 重みと活性を8bitの浮動小数点に。L4（Ada）はFP8の計算回路を持つ |

- 量子化の構成は、エンジンの間で方式をそろえず、各エンジンの中でbf16との非劣化を確かめる（ADR-0008・0009）。
- NVIDIAが公開している`nvidia/Qwen3-8B-FP8`は、KVキャッシュもFP8にしてある（`hf_quant_config.json`の`kv_cache_quant_algo`）。
  KVキャッシュの量子化は使わない決まり（ADR-0009の4）なので、使わず、`--kv_cache_qformat none`で自分で作る。
- ModelOptの調整用のデータは、`hf_ptq.py`の既定（cnn_dailymailの512件）。

## ツール呼び出しと思考の扱い

| エンジン | ツールパーサー | 思考を切る | サンプリングの設定 |
| --- | --- | --- | --- |
| vLLM | `--tool-call-parser hermes`（`--enable-auto-tool-choice`） | `chat_template_kwargs: {enable_thinking: false}` | `top_k`・`min_p`・`seed`を受け付ける |
| SGLang | `--tool-call-parser qwen`（`Qwen25Detector`、`srt/function_call/function_call_parser.py`） | 同じ（`ChatCompletionRequest.chat_template_kwargs`） | 同じ（`entrypoints/openai/protocol.py`） |
| TensorRT-LLM | `--tool_parser qwen3`（`serve/tool_parser/tool_parser_factory.py`） | 同じ（`serve/openai_protocol.py`） | 同じ |
| llama.cpp | `--jinja`（チャットテンプレートからツール呼び出しを読む） | 同じ | 同じ。既定は`top_k=40`・`min_p=0.05`で他と違うので、全部を明示して送る |

Qwen3のツール呼び出しは`<tool_call>{...}</tool_call>`の形（Hermes形式）で、どのパーサーもこれを読む。パーサーの実装はエンジンごとに別で、
読み損ねがあると精度に出る。そのため、新しいエンジンのbf16はvLLM bf16に対する非劣化で足切りする（ADR-0009の5）。

### SGLangはツールの定義を書き換えてから書式に入れる

同じリクエストを送っても、SGLangのモデルに届く入力は、vLLM・TensorRT-LLMより1件あたり100トークン長い（疎通の1回目、2026-10-28。
TensorRT-LLMはvLLMと全ケースで同じだった）。

- **原因**：SGLang 0.5.21は、受け取ったツールを自分の型（`Tool`）に読み込み、`model_dump()`で書き戻してからチャットの書式に入れる
  （`serving_chat.py`）。そのとき関数のキーの順番が変わり、`"strict": false`と`"defer_loading": null`が足される（`protocol.py`）。
  ツール1つあたり約10トークンで、段階10（ツール10個）では+100になる。
- **切り替える設定は無い**：既定の経路で必ず起きる。こちらから同じ項目を付けて送る手もあるが、ほかのエンジンと第1段階の入力まで変わるので採らない。
- **扱い**：エンジンの振る舞いとして、そのまま測る（2026-10-28に決めた）。精度や速さの差には、この入力の違いも含まれる。
- **容量の事前の確かめ**：SGLangのセルは、SGLangが作る入力で数えた上限（2,691トークン。全体の上限は2,591）を使う。
  `scripts/count_prompt_tokens.py --engine sglang`が書き換えを再現して数え、`schema/prompt_budget.json`の`by_engine`に記録する。
  再現が正しいことは、インスタンスで段階10の全1,200件のリクエストをSGLang自身の`model_dump()`と突き合わせて確かめた（違いは0件）。

## 環境

| エンジン | Python環境 | torch | transformers | 入れ方 |
| --- | --- | --- | --- | --- |
| vLLM・transformers | `~/venv` | 2.13.0+cu132 | 5.17.0 | `uv pip install vllm==0.31.0` |
| SGLang | `~/venv-sglang` | 2.13.0（SGLangが固定） | 5.12.1（SGLangが固定） | `uv pip install sglang==0.5.21` |
| TensorRT-LLM・ModelOpt | `~/venv-trtllm` | 2.9.1+cu130 | 4.57.3（TensorRT-LLMが固定） | 文書の手順：torchを先に入れ、NVIDIAの索引（pypi.nvidia.com）からwheelを入れる。MPI（`libopenmpi-dev`）が要る |
| llama.cpp | ソースからビルド（CUDA、sm_89） | ― | ― | `cmake -DGGML_CUDA=ON` |

それぞれが固定する版がぶつかるので、環境を分ける。クライアント（`scripts/run_cell.py`）はどのエンジンでも`~/venv`から動かす。

## 出典

- vLLM 0.31.0：`vllm/v1/metrics/loggers.py`、`vllm serve --help`
- SGLang v0.5.21：https://github.com/sgl-project/sglang/tree/v0.5.21 （`python/sglang/srt/observability/metrics_collector.py`、`srt/entrypoints/http_server.py`、
  `srt/entrypoints/openai/protocol.py`、`srt/function_call/function_call_parser.py`、`srt/arg_groups/`）、ツールパーサーの文書 https://docs.sglang.io/advanced_features/tool_parser.html
- TensorRT-LLM v1.2.1：https://github.com/NVIDIA/TensorRT-LLM/tree/v1.2.1 （`tensorrt_llm/commands/serve.py`、`serve/openai_server.py`、`serve/openai_protocol.py`、
  `llmapi/llm_args.py`、`_torch/model_config.py`、`cpp/tensorrt_llm/executor/jsonSerialization.cpp`、`docs/source/features/quantization.md`、
  `docs/source/installation/linux.md`）、対応表 https://nvidia.github.io/TensorRT-LLM/1.2.1/legacy/reference/support-matrix.html
- ModelOpt 0.37.0：https://github.com/NVIDIA/Model-Optimizer/tree/0.37.0 （`examples/llm_ptq/hf_ptq.py`）
- `nvidia/Qwen3-8B-FP8`：https://huggingface.co/nvidia/Qwen3-8B-FP8 （`hf_quant_config.json`）
- PagedAttention：Kwon et al., SOSP 2023。RadixAttention：Zheng et al., SGLang, NeurIPS 2024
