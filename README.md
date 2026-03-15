# WISTERIA.md — Wisteria/Aquarius vLLM 構成記録

Qwen3.5-35B-A3B を東京大学 Wisteria/BDEC-01 Aquarius 上で
vLLM 0.17.1 + 4×A100-SXM4-40GB tensor parallel で動作させる構成。
2026-03-15 時点で動作確認済み。

## システム情報

| 項目 | 値 |
|---|---|
| HPC | Wisteria/BDEC-01 Aquarius |
| GPU | NVIDIA A100-SXM4-40GB × 4 (学生枠上限) |
| CUDA Driver | 12.2 (535.54) — forward compat で 12.9 相当 |
| OS | Red Hat 系 (GCC 8.3.1 がデフォルト) |
| ジョブスケジューラ | PJM |
| ホームディレクトリ | `/home/<user>` (50GB quota, 使用禁止推奨) |
| ワークディレクトリ | `/work/<project>/<user>/` (400GB quota) |

以下、パスの表記は:
- `$WORK` = `/work/<project>/<user>`
- `$PROJECT_ROOT` = `$WORK/wisteria-qwen35-4gpu`
- `<project>` = 計算科学アライアンスのプロジェクト名
- `<user>` = Wisteria ユーザ名

## ディスク構成 (~80GB)

```
$WORK/
├── miniconda3/                     (1.9GB)  Python 3.11 バイナリ提供のみ
└── wisteria-qwen35-4gpu/           プロジェクトルート
    ├── .venv/                      (11GB)   uv venv + vllm 0.17.1 + 全依存パッケージ
    ├── runtime/hf/                 (67GB)   Qwen3.5-35B-A3B モデルウェイト (14 safetensors)
    ├── .env                                 環境変数 (絶対パス)
    ├── pjm/
    │   ├── submit_vllm_4gpu.sh              本番ジョブ (share-short, 6h, ctx=4096)
    │   ├── submit_vllm_4gpu_debug.sh        デバッグジョブ (share-debug, 30min, ctx=2048)
    │   └── build_vllm.sh                    ソースビルドジョブ (share-short, 2h)
    ├── app/
    │   ├── client_smoke.py                  疎通テスト (OpenAI SDK)
    │   └── client_bench.py                  レイテンシ計測 (5リクエスト)
    ├── scripts/                             ユーティリティスクリプト
    ├── logs/                                PJM ジョブログ
    └── tmp/                                 Triton/CUDA キャッシュ (実行時生成)
```

## ソフトウェアスタック

| パッケージ | バージョン | インストール方法 |
|---|---|---|
| Python | 3.11.11 | miniconda3 (バイナリ提供のみ) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| vllm | 0.17.1 | **ソースビルド** (PyPI には 0.11.2 までしかない) |
| torch | 2.10.0+cu128 | `uv pip install` (CUDA 12.8 wheel) |
| transformers | 4.57.6 | vllm 依存で自動インストール |
| GCC | 12.2.0 | `module load gcc/12.2.0` |
| cmake | 4.2.3 | conda install → .venv/bin/ にシンボリックリンク |
| CUDA toolkit | 12.9 | `/work/opt/local/x86_64/cores/cuda/12.9/` |
| CUDA compat libs | 12.9 (driver 575相当) | `/work/opt/local/x86_64/cores/cuda/12.9/compat/` |

## 動作実績

| 項目 | 値 |
|---|---|
| モデル | Qwen/Qwen3.5-35B-A3B (MoE, 35B params, 3B active) |
| GPU メモリ使用 | 16.52 GiB (4GPU 分散) |
| Attention backend | FlashAttention v2 |
| MoE backend | Triton |
| モデルロード時間 | ~4分15秒 (14シャード, 67GB) |
| サーバ起動完了まで | ~10分 (KVキャッシュプロファイル含む) |
| API endpoint | `http://<compute-node>:8000/v1` (OpenAI互換) |

## 初回セットアップ手順

### 1. miniconda で Python 3.11 を確保

Wisteria のシステム Python は 3.6.8 で古すぎるため、miniconda で 3.11 を用意する。
パッケージ管理には uv を使い、conda は Python バイナリと cmake の提供のみ。

```bash
cd $WORK
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $WORK/miniconda3
# cmake も conda で入れる (システム cmake 3.20.2 では vllm ビルド不可)
$WORK/miniconda3/bin/conda install -y cmake
```

### 2. プロジェクトディレクトリ作成

```bash
mkdir -p $WORK/wisteria-qwen35-4gpu
cd $WORK/wisteria-qwen35-4gpu
```

ローカルの `wisteria/` から rsync でファイルを配置:
```bash
# ローカルから実行
rsync -avz --exclude='.venv' --exclude='runtime' --exclude='tmp' --exclude='logs' \
    wisteria/ wisteria:$WORK/wisteria-qwen35-4gpu/
```

### 3. .env を絶対パスで設定

PJM ジョブ環境では `$PROJECT_ROOT` 等の変数展開が効かないため、
`.env` 内は全て**絶対パス**で記述する。

```bash
cp .env.example .env
# エディタで全パスを $WORK/wisteria-qwen35-4gpu の実際の値に置換
```

### 4. uv で仮想環境作成

パッケージ管理は uv を使用。conda は Python 3.11 バイナリと cmake の提供のみ。

```bash
# uv インストール (未導入の場合)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# venv 作成 (miniconda の Python を使用)
uv venv .venv --python $WORK/miniconda3/bin/python3.11
source .venv/bin/activate

# pip を入れる (uv venv にはデフォルトで入らない。vllm ソースビルドで必要)
uv pip install pip setuptools>=82

# cmake シンボリックリンク (.venv/bin から使えるように)
ln -sf $WORK/miniconda3/bin/cmake .venv/bin/cmake
```

### 5. torch (CUDA 12.8) インストール

```bash
uv pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 6. Hugging Face ログイン & モデルダウンロード

```bash
uv pip install huggingface_hub
huggingface-cli login  # トークンを対話入力

# モデルダウンロード (~67GB, ~40秒)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-35B-A3B',
                  cache_dir='$PROJECT_ROOT/runtime/hf/hub')
"
```

### 7. vllm ソースビルド (計算ノード上で実行)

**ログインノードには GPU がないため、必ず PJM ジョブで実行する。**

vllm のソースビルドは `pip install . --no-build-isolation` で行う。
vllm のビルドシステムが内部的に pip を呼ぶため、この部分だけは uv ではなく pip を使用。
(Step 4 で `uv pip install pip` しておく必要があるのはこのため)

```bash
# ソース取得
cd $WORK
git clone --depth 1 --branch v0.17.1 https://github.com/vllm-project/vllm.git vllm-src

# ビルドジョブ投入 (share-short, 2時間, GPU×1)
cd $PROJECT_ROOT
pjsub pjm/build_vllm.sh

# ビルド完了後 (~1.5時間)、ソースは削除可
rm -rf $WORK/vllm-src
```

#### ビルド時の必須環境変数

PJM ジョブスクリプト (`pjm/build_vllm.sh`) 内で以下を設定する:

```bash
module load gcc/12.2.0
export CC=/work/opt/local/x86_64/cores/gcc/12.2.0/bin/gcc
export CXX=/work/opt/local/x86_64/cores/gcc/12.2.0/bin/g++
export CUDAHOSTCXX=/work/opt/local/x86_64/cores/gcc/12.2.0/bin/g++
export CUDA_HOME=/work/opt/local/x86_64/cores/cuda/12.9
export LD_LIBRARY_PATH="$CUDA_HOME/compat:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0"   # A100
export MAX_JOBS=4
```

**ビルドが失敗するパターンと対処:**

| 症状 | 原因 | 対処 |
|---|---|---|
| `#error GCC 9 or later` | nvcc が `/usr/bin/c++` (GCC 8.3.1) を使用 | `CC`, `CXX`, `CUDAHOSTCXX` を GCC 12 パスに設定 |
| `cmake 3.26+ required` | システム cmake 3.20.2 | conda cmake をシンボリックリンク |
| 30分で時間切れ | share-debug 枠が不足 | share-short (2h+) で実行 |
| setuptools エラー | SPDX license 未対応 | `uv pip install setuptools>=82` |

## ジョブ実行

### PJM ジョブスクリプトの要点

`#PJM` ディレクティブはシェル変数展開されないため、プロジェクト名やリソースグループは
スクリプト内にハードコーディングする。

また PJM ジョブ環境では `$HOME` が `/pjmhome/` になり書き込み不可なので、
以下の環境変数でキャッシュ先をプロジェクト内にリダイレクトする:

```bash
export HOME="$PROJECT_ROOT/tmp"
export TRITON_CACHE_DIR="$PROJECT_ROOT/tmp/triton"
export XDG_CACHE_HOME="$PROJECT_ROOT/tmp/cache"
```

### PJM ディレクティブ例

```bash
#PJM -L rscgrp=share-debug    # リソースグループ
#PJM -g <project>              # プロジェクト名
#PJM -L gpu=4                  # GPU 数
#PJM -L elapse=00:30:00        # 最大実行時間
#PJM -j                        # stdout/stderr 結合
#PJM -o logs/output.out        # 出力ファイル
```

### リソースグループ

| グループ | 最大時間 | 用途 |
|---|---|---|
| `share-debug` | 30分 | 疎通確認、短時間テスト |
| `share-short` | 6時間 | 本番推論、ソースビルド |

### デバッグジョブ (推奨: 初回はこちら)

```bash
cd $PROJECT_ROOT
pjsub pjm/submit_vllm_4gpu_debug.sh
pjstat                              # ジョブ状態確認
tail -f logs/pjm_vllm_4gpu_debug.out  # ログ確認
```

### 本番ジョブ

```bash
pjsub pjm/submit_vllm_4gpu.sh
pjstat
tail -f logs/pjm_vllm_4gpu.out
```

## ローカルからの接続 (SSH トンネル)

vLLM サーバは計算ノード上で動作するため、ローカルからは SSH トンネル経由でアクセスする。

```bash
# 1. 計算ノード名を確認 (ジョブログから)
ssh wisteria "grep 'hostname:' $PROJECT_ROOT/logs/pjm_vllm_4gpu_debug.out"
# → hostname: wa33

# 2. SSH トンネル
ssh -L 8000:wa33:8000 wisteria

# 3. ローカルから API 確認
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer EMPTY"

# 4. OpenAI 互換 API として使用
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{"model":"Qwen/Qwen3.5-35B-A3B","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

任意の OpenAI 互換クライアント (`--llm-url http://localhost:8000/v1`) から接続可能。

## トラブルシューティング

| 症状 | 原因 | 対処 |
|---|---|---|
| `PermissionError: '/pjmhome/.triton'` | PJM が HOME を `/pjmhome/` に変更 | `HOME`, `TRITON_CACHE_DIR` をプロジェクト内に設定 |
| `failed to be inspected` | Triton キャッシュ権限エラーの連鎖 | 上記と同じ |
| `Address family not supported` (NCCL) | IPv6 非対応の警告 | 無害、無視して良い |
| `CUDA out of memory` | コンテキスト長が大きすぎる | `MAX_MODEL_LEN` を 2048 に下げる |
| vllm import 失敗 | LD_LIBRARY_PATH に compat がない | `$CUDA_HOME/compat` を LD_LIBRARY_PATH 先頭に追加 |
| SSH トンネル接続拒否 | ジョブ未起動 or ポート不一致 | `pjstat` でジョブ確認、ログで `Application startup complete` を確認 |
| SSH passphrase エラー | ssh-agent にキー未登録 | `ssh-add ~/.ssh/<秘密鍵ファイル>` |

## CUDA Forward Compatibility

Wisteria の CUDA driver は 12.2 (535.54) だが、
`/work/opt/local/x86_64/cores/cuda/12.9/compat/` に **forward compatibility ライブラリ**
(driver 575 相当) が配置されている。これを `LD_LIBRARY_PATH` の先頭に置くことで、
CUDA 12.8/12.9 向けにビルドされた torch/vllm が動作する。

```bash
export LD_LIBRARY_PATH="/work/opt/local/x86_64/cores/cuda/12.9/compat:$LD_LIBRARY_PATH"
```

この仕組みにより、driver 更新なしで最新の PyTorch + vllm が利用可能。
