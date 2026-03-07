# DeepSeek-V4.1-Flash-UNCENSORED-FP8 を Modal で動かす設計メモ

作成日: 2026-10-19 / 対象: [dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8)
（revision `d61c59ea5e514e25d305b5850e8a432f7a9969f2`, 2026-09-11）

## 0. 結論（推奨構成）

| 項目 | 推奨 | 理由 |
|---|---|---|
| 推論エンジン | **SGLang**（V4.1 対応 nightly イメージを digest 固定） | モデル作者が検証した経路。Engram テーブルの CPU 退避など V4.1 固有のノブが揃う。公式 cookbook に 4×B200 の verified 設定あり（2026-10-16） |
| GPU | **`B200:4`**（TP4/EP4、Engram は GPU 上） | 重み 510 GB に対し VRAM 720 GB。CPU 退避なしで収まる唯一の 4 GPU 構成。$25/h |
| 第 2 候補 | `H200:4` + Engram テーブルを CPU 退避（ホスト RAM 約 200 GiB） | $18/h と最安だが、Modal のサンドボックス上で ~190 GiB の pinned memory が動くかは未検証 |
| 保険 | `H200:8`（TP8、退避なし） | 確実に収まるが $36/h |
| 重みの置き場 | Modal Volume **v2** ×2（weights / JIT キャッシュ） | 101 GB の shard が 2 本あり、v1 のファイル内書き込み再書込ペナルティを避ける。minimax-h3-modal と同じ |
| 実行形態 | `modal deploy` した `@app.cls`（起動時に SGLang サーバを subprocess で立てる）+ `@modal.method` ジェネレータでトークンをストリーム | minimax-h3-modal と同じ「Modal トークン認証 + `Cls.from_name`」方式。HTTP の 150 秒制限や Proxy Token の管理が不要 |
| CLI | `dsv4 generate / chat / download / up / down / status`（typer） | `up/down` で `min_containers` を切り替え、cold start（10〜25 分、$4〜10/回）の回数を管理する |
| OpenAI 互換 API | オプション（同じクラスに `@modal.web_server(..., requires_proxy_auth=True)`） | 他ツール（aider 等）から叩きたくなったときに追加 |

## 1. モデルの事実関係（調査結果）

- アーキテクチャ: DeepSeek V4.1 Flash。552B MoE（384 experts, top-6 + shared 1）、active 8B（prefill）/ 16B（decode）。1M context。Vision encoder 付き（`image-text-to-text`）。
- 重み: 510 GB / 48 shards（最大 shard 101.5 GB が 2 本）。FP8（ue8m0 block-32）+ 経路 experts は FP4 + Engram n-gram テーブル約 183 GiB（FP8）。base の 1017 更新は含まれない（0909 版を abliteration したもの）。
- chat template（Jinja）は同梱されない。エンジン内蔵のエンコーダを使うため **`--reasoning-parser deepseek-v41` を明示**しないと `<think>` が `content` に漏れる。`reasoning_effort` は毎リクエスト送る（`low/high/xhigh/max` または数値）。thinking を切るには `chat_template_kwargs: {"thinking": false}`。
- 推奨サンプリング: temperature 1.0 / top_p 0.95。`max` effort は数千トークン考えるので `max_tokens` は大きめ。
- DSpark（投機デコード）: 作者自身が「ablation 時に draft head を未処理」と認めており受理率が落ちている（2.94 vs 3.57 tokens/step）。**初期は無効**にし、計測後に `--speculative-algorithm DSPARK --speculative-dspark-block-size 5` を試す。
- VRAM の床: vLLM recipe で 614 GB（510 GB × 1.2）。4×H200（564 GB）は Engram を CPU に退避しないと載らない。4×B200（720 GB）と 8×H200（1128 GB）は載る。
- 参考性能: 4×H200 で single-stream 約 100 tok/s、同時 8 本で合計 126 tok/s（モデルカード）。

## 2. Modal 側の制約（確認済み）

- GPU 単価（$/h）: H100 3.95 / H200 4.54 / B200 6.25 / B300 7.10。1 コンテナ最大 8 GPU。3 GPU 以上は待ち時間が伸びる。Starter プランは同時 10 GPU まで（4 GPU 構成は収まる）。
- **作成時バリデーションの確認**（2026-10-19、deploy のみで実行はしていない）: `gpu="H200:4", memory=256 GiB, startup_timeout=3600, scaledown_window=1200`、`gpu="B200:4"`、`ephemeral_disk=600 GiB, timeout=24h` はすべて受理された。ホスト RAM 200 GiB 級の要求自体は通る。
- `scaledown_window` の上限は 20 分。Function の `timeout` は最大 24 h。`startup_timeout` は 3600 秒が通る。
- Volume: 読み出し最大 2.5 GB/s 目安。v2 は大きいファイルの部分書き込みが差分書き込みになる（v1 はファイル全体の再書込になり得る）。
- GPU メモリスナップショットは multi-GPU 非対応なので cold start 短縮には使えない。
- Web endpoint は 150 秒の HTTP 制限がある（ストリーミングで回避可能）。RPC（`remote_gen`）経路にはこの制限がない。
- gVisor サンドボックス上の `/dev/shm` サイズと pinned memory の上限は未検証。Modal 公式の vLLM / SGLang 例が TP 複数 GPU で動いているので NCCL 自体は問題ない見込み。
- `uvx modal` は Python 3.14 を拾うと `debian_slim` のビルダーに拒否される。`uvx --python 3.12 modal ...` かプロジェクトの venv（`uv run modal ...`）を使う。

## 3. 全体アーキテクチャ

```
ローカル                                         Modal (app "deepseek-v41")
dsv4 CLI (typer, modal/typer/pydantic のみ)
  generate / chat ── Cls.from_name → remote_gen ─▶ DeepSeekServer  @app.cls(gpu="B200:4", max_containers=1)
  up / down / status ── update_autoscaler ──▶    │  @modal.enter : sglang.launch_server を subprocess 起動
  download ── Function.from_name → spawn ──▶     │                 /health を待ち、warmup を 1 本流す
                                                 │  @modal.method generate(): 127.0.0.1:30000/v1/chat/completions
                                                 │                 を stream=True で読み、delta を yield
                                                 │  @modal.method complete(): 非ストリーム版（spawn / jobs 用）
                                                 │  (任意) @modal.web_server(30000, requires_proxy_auth=True)
                                                 ├─ download()  CPU 8 core, HF → Volume v2 に 510 GB
                                                 ├─ auto_down() CPU, N 時間後に min_containers=0 に戻す
                                                 └─ env_check() image 内のバージョン確認
Volumes (v2): deepseek-v41-weights → /models,  deepseek-v41-cache → /root/.cache
```

## 4. ディレクトリ構成（minimax-h3-modal を踏襲）

```
pyproject.toml              deps: modal, typer, pydantic / scripts: dsv4 = "deepseek_modal.cli:main"
src/deepseek_modal/
  config.py                 APP_NAME, MODEL_ID, REVISION, GPU, ENGRAM_HOST_TABLE, CONTEXT_LEN, MEM_FRACTION,
                            timeouts など。すべて DSV4_* 環境変数で上書き可能
  app.py                    image / volumes / download() / DeepSeekServer / auto_down() / local_entrypoint
  engine.py                 SGLang の起動コマンド組立・health 待ち・warmup（vLLM へ差し替える唯一の点）
  schemas.py                GenerationRequest(messages, effort, thinking, sampling, max_tokens), StreamChunk, GenerationResult
  cli.py                    dsv4: download / generate / chat / up / down / status / env-check
tests/                      schemas と engine のコマンド組立の単体テスト（GPU 不要）
docs/design.md              本書
```

## 5. 主要パラメータ（初期値案）

| 項目 | 値 | メモ |
|---|---|---|
| image | `lmsysorg/sglang:nightly-dev-cu13-20261014-684fea8a@sha256:ddae0e399d3d94f7827f16183301cf8dac4cf0eae0ef25590b7dac5881d713d8` + `.entrypoint([])` + `pip install ninja` | cookbook の B200 verified セルと同じ。予備は作者検証の `lmsysorg/sglang:dev-dsv41`（2026-09-11、`--reasoning-parser auto` が効かないので明示必須） |
| gpu / cpu / memory | `B200:4` / 16 core / 128 GiB（Engram 退避時は 256 GiB） | memory は $0.008/GiB/h、cpu は $0.047/core/h |
| TP / EP | 可視 GPU 数から自動決定（`nvidia-smi -L` の本数）。TP=EP | `gpu=["B200:4", "H200:8"]` のフォールバックにしても起動コードを変えずに済む。TP4 では `--ep-size` 必須 |
| context | 262144（必要時 1048576） | KV は 890 B/token なので context 長より同時実行数の方が効く |
| `--mem-fraction-static` | 0.85（Engram 退避なし）/ 0.76〜0.78（退避あり） | cookbook B200 TP4: 0.88（c1）, TP4·EP4: 0.76 |
| 同時実行 | `--max-running-requests 8`, `--cuda-graph-max-bs-decode 8`, `--chunked-prefill-size 4096` | 個人 CLI 用途。`@modal.concurrent(max_inputs=8)` を合わせる |
| parser | `--reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 --trust-remote-code` | |
| 読み込み高速化 | `--model-loader-extra-config '{"enable_multithread_load":true}' --weight-loader-prefetch-checkpoints --watchdog-timeout 3600` | Volume からの 510 GB 読み出しを並列化 |
| 投機デコード | なし（後から `DSPARK`, block 5） | 1 項参照 |
| env | `SGLANG_DEFAULT_THINKING=1`, `SGLANG_DSV41_REASONING_EFFORT=high`, `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0`（退避時 `1` + `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank`）, `SGLANG_TIMEOUT_KEEP_ALIVE=900`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `NCCL_CUMEM_ENABLE=1`, `TRITON_CACHE_DIR=/root/.cache/triton` | |
| timeouts | `startup_timeout` 60 min / 1 呼び出し `timeout` 60 min / `scaledown_window` 20 min（上限） | |
| 台数制限 | `max_containers=1` | 4 GPU が 2 組立つ事故を防ぐ |
| download | cpu 8 / memory 32 GiB / timeout 24 h / `snapshot_download(local_dir=/models/<repo>, revision=固定)` / `HF_XET_HIGH_PERFORMANCE=1` / `HF_XET_CACHE=/tmp/hf-xet-cache`（xet チャンクキャッシュは Volume に置かない） | 再実行で再開可能。目安 20〜60 分、$1 未満 |

Volume `/root/.cache` には DeepGEMM（`~/.cache/deep_gemm`）、FlashInfer、torch extensions、Triton（上記 env で `.cache` 配下へ）、SGLang の JIT 生成物を載せる。SGLang の JIT キャッシュの正確なディレクトリは公開ドキュメントにないので、初回起動後に `/root/.cache` の更新ファイルを見て確認する。

## 6. 起動シーケンスと時間・コストの見積

1. image pull（初回のみ、約 15 GB）: 数分。
2. 重み読み込み 510 GB @ 1〜2.5 GB/s: 4〜9 分。minimax-h3-modal の実測（144 GB で cold と warm の差 50〜100 秒）とも整合。
3. JIT コンパイル + CUDA graph 取得: 初回 数分〜10 分、2 回目以降はキャッシュ Volume で短縮。
4. 合計 cold start: **10〜25 分**（モデルカードはローカル NVMe で cold 28 分 / warm 10 分）。

| 費用項目 | B200:4（$25.0/h） | H200:4（$18.2/h） | H200:8（$36.3/h） |
|---|---|---|---|
| cold start 1 回（15〜25 分） | $6〜10 | $4.5〜7.5 | $9〜15 |
| アイドル 20 分（scaledown 上限） | $8.3 | $6.1 | $12.1 |
| 1k tokens 生成（100 tok/s 想定） | 約 $0.07 | 約 $0.05 | 約 $0.10 |

Volume 保管は 1 TiB までの無料枠を超えた分が $0.09/GiB/月。

## 7. CLI 仕様（案）

```
dsv4 download [--wait/--no-wait]                 Volume に重みを入れる（一度だけ）
dsv4 generate "prompt" [-s SYSTEM] [-f prompt.txt] [--effort low|high|xhigh|max|none|<int>]
              [--max-tokens N] [--temperature 1.0] [--top-p 0.95] [--hide-reasoning] [--json] [-o out.md]
dsv4 chat     [--effort ...]                      履歴をローカルに持つ REPL（毎ターン全履歴送信、prefix cache が効く）
dsv4 up   [--hours 2]                             min_containers=1 にして auto_down を spawn
dsv4 down                                         min_containers=0
dsv4 status                                       autoscaler 設定、稼働コンテナ数、health
dsv4 env-check                                    image 内の sglang / torch バージョン
```

- `generate` はデフォルトでストリーム表示。`reasoning_content` と `content` を色分けして出し、`--hide-reasoning` で後者のみ。
- 長い生成を切断に強くしたい場合は `complete` を `spawn` して `jobs fetch` する H3 と同じ経路を使う（ストリームなし）。
- 認証は Modal トークンのみ。web_server を有効にした場合だけ Proxy Token（`Authorization: Bearer <id>.<secret>`）を使う。
- 開発時は `modal run -m deepseek_modal.app --prompt "..."` で一回実行できるが、毎回 cold start を払う。

## 8. リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| B200:4 の空き待ち | 起動が数分〜遅れる | `gpu=["B200:4", "H200:8"]` のフォールバック + TP 自動決定。`nonpreemptible` は不要 |
| Engram 退避（H200:4）が gVisor 上で動かない | 第 2 候補が使えない | まず B200:4 で動かし、H200:4 は別途 1 回だけ実験（失敗しても cold start 1 回分の費用） |
| SGLang nightly の API 変化 | 起動フラグが通らない | image を digest 固定。予備タグ `dev-dsv41` を config で切替可能に |
| cold start が長い | 1 回 $4〜10 | `up/down` 運用、JIT キャッシュ Volume、`--weight-loader-prefetch-checkpoints` |
| `<think>` 漏れ | 出力が壊れる | parser 明示 + CLI が常に `reasoning_effort` を送る。テストで `reasoning_content` と `content` の分離を確認 |
| `max_tokens` 不足 | 空の content で `finish_reason=length` | 既定 32768、effort=max 時はさらに大きく |
| Volume v2 は beta | 再ダウンロードで復旧可能 | 重みは再取得できるので許容。download を冪等に |
| 画像入力 | 未検証 | 初期は text-only。SGLang 側の multimodal 対応を確認してから `--image` を追加 |
| DSpark 受理率低下 | 速度向上が小さい | 既定 off、計測して判断 |

モデルカードは「出力の責任は利用者にある」と明記している。MIT ライセンス。

## 9. 実装ステップ

1. プロジェクト骨組み（pyproject, config, schemas, tests）と `download()`。Volume に 510 GB を入れる（30〜60 分）。
2. `env_check()` で image と `sglang` の import を GPU なしで確認。
3. `DeepSeekServer` を `modal run` で起動し、B200:4 の cold start 時間と health を計測（1 回 $10 程度）。
4. `generate` / `chat` のストリーム表示、effort 制御、`up/down/status`。
5. tok/s と cold start を README に記録。`scaledown` と `max-running-requests` を調整。
6. 任意: web_server + Proxy Token、DSPARK、H200:4 の Engram 退避実験、画像入力。

## 10. 代替案: vLLM

vLLM 0.31.0 以降は V4.1-Flash を公式サポート（recipe: H200 は TP4 + `--engram-config '{"cpu_offload":true}'`、B200 は TP2 + 退避 または TP4）。`vllm/vllm-openai:v0.31.0`（2026-10-04）をベースに、`--tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 --language-model-only --max-model-len 262144`、`VLLM_ENGINE_READY_TIMEOUT_S=3600`、`/root/.cache/vllm` を Volume に載せる。Modal 公式の vLLM 例（`@app.server`）がそのまま下敷きになる。`engine.py` の差し替えだけで移行できるように、CLI とスキーマはエンジン非依存にしておく。

## 実装時に判明した差分（2026-10-19）

- Modal の既定イメージビルダー（2023.12）は SGLang イメージに古い依存を入れようとして失敗する。`MODAL_IMAGE_BUILDER_VERSION=2025.06` でデプロイし、Modal 側のコードは `add_python` の別インタプリタで動かす（SGLang の venv `/opt/sglang` には触れない）。
- Modal は中身のあるディレクトリに Volume をマウントできない。キャッシュ Volume は `/cache` にマウントし、イメージの `/root/.cache` は `/cache` へのシンボリックリンクに置き換える（元の内容は初回起動時に書き戻す）。
- 4×B200 は 30 分待っても確保できなかったため、既定の GPU 指定を `B200:4,H200:8` のフォールバックリストにした。TP/EP は見えた GPU 数で自動決定する。
- 重みのダウンロード実測: 475.3 GiB / 1470 秒（約 330 MB/s）、CPU コンテナ 1 台。
- 初回起動実測（4×B200、JIT キャッシュなし）: 重み読み込み 276 秒、スケジューラ起動（JIT 含む）792 秒、CUDA graph 49 秒、`/health` まで 871 秒。CLI から見た総時間 15 分 25 秒（GPU 代約 $6.5）。
- 推論実測（warm、単一ストリーム）: 約 290〜300 tok/s、TTFT 0.2〜0.3 秒。effort=high で 4096 トークンの生成が 14.4 秒。6 章の見積（100 tok/s）より大幅に速い。

## 参考リンク

- モデルカード: https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8
- SGLang cookbook (V4.1): https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1
- vLLM recipe (V4.1-Flash): https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash
- Modal: pricing https://modal.com/pricing / GPU https://modal.com/docs/guide/gpu / Volumes https://modal.com/docs/guide/volumes / cold start https://modal.com/docs/guide/cold-start / Servers https://modal.com/docs/guide/servers
- Modal vLLM 例: https://modal.com/docs/examples/vllm_inference / SGLang 例: https://modal.com/docs/examples/sgl_vlm
