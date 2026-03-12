# Material Bench

Material science ベンチマーク（MaterialFigBench, MaterialBENCH）の推論・評価スクリプト

## 概要

- **MaterialFigBench**: 画像＋テキスト問題（137 問）- 結晶方位、相図、熱処理など
- **MaterialBENCH choice**: 4 択問題（164 問）- 原子量、結晶構造など
- **MaterialBENCH free**: 自由記述問題（144 問）- 計算問題、概念説明など

OpenAI 互換 API（vLLM など）を使用して推論を実行し、正答率などの評価指標を出力します。

## インストール

```bash
# pixi を使用
pixi install
```

## 使い方

### 基本コマンド

```bash
# MaterialBENCH choice を実行
pixi run python scripts/run_benchmark.py \
  --dataset materialbench-choice \
  --api-base http://localhost:8001

# MaterialFigBench を実行（画像キャッシュ付き）
pixi run python scripts/run_benchmark.py \
  --dataset material-figbench \
  --api-base http://localhost:8001

# MaterialBENCH free を実行
pixi run python scripts/run_benchmark.py \
  --dataset materialbench-free \
  --api-base http://localhost:8001

# すべて実行
pixi run python scripts/run_benchmark.py \
  --dataset all \
  --api-base http://localhost:8001

# ランダムサンプルでテスト
pixi run python scripts/run_benchmark.py \
  --dataset materialbench-choice \
  --api-base http://localhost:8001 \
  --max-samples 10 \
  --seed 42

# 並列実行で高速化（3 スレッド）
pixi run python scripts/run_benchmark.py \
  --dataset materialbench-choice \
  --api-base http://localhost:8001 \
  --max-samples 10 \
  --num-threads 3 \
  --output-dir ./results

# 既存の CSV 結果に LLM judge を適用（推論スキップ）
pixi run python scripts/run_benchmark.py \
  --llm-judge-only \
  --judge-csv results/materialbench-free_20260307_000330.csv \
  --api-base http://localhost:8001

# 最新の CSV を自動検出して LLM judge（データセット指定）
pixi run python scripts/run_benchmark.py \
  --llm-judge-only \
  --dataset materialbench-free \
  --api-base http://localhost:8001 \
  --judge-num-threads 20

# 全データセットの最新結果に LLM judge
pixi run python scripts/run_benchmark.py \
  --llm-judge-only \
  --dataset all \
  --api-base http://localhost:8001
```

### オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--dataset` | 対象データセット（material-figbench / materialbench-choice / materialbench-free / all） | 必須 |
| `--output-dir` | 結果保存ディレクトリ | `./results` |
| `--api-base` | OpenAI 互換 API ベース URL | `http://localhost:8001` |
| `--model` | モデル名 | `Qwen/Qwen3.5-397B-A17B-FP8` |
| `--max-samples` | ランダムに抽出するサンプル数（0 で全体） | `0` |
| `--seed` | ランダムシード | `42` |
| `--api-key` | API キー | `""` |
| `--hf-token` | HuggingFace トークン（レート制限回避） | `None` |
| `--timeout` | API タイムアウト（秒） | `300` |
| `--max-retries` | 最大リトライ回数 | `2` |
| `--num-threads` | 並列スレッド数（1 でシーケンシャル） | `1` |
| `--use-llm-judge` | 推論後に LLM 評価を実行するフラグ | `False` |
| `--llm-judge-only` | 推論をスキップして既存 CSV に LLM 評価のみ適用 | `False` |
| `--judge-csv` | LLM 評価する CSV ファイル（--llm-judge-only 使用時） | `None` |
| `--judge-api-base` | LLM judge 用 API ベース URL | `--api-base` と同じ |
| `--judge-model` | LLM judge 用モデル名 | `--model` と同じ |
| `--judge-num-threads` | LLM judge の並列スレッド数 | `--num-threads` と同じ |
| `--judge-timeout` | LLM judge の API タイムアウト（秒） | `60` |

### 出力ファイル

```
results/
├── material-figbench_YYYYMMDD_HHMMSS.csv    # MaterialFigBench の結果
├── materialbench-choice_YYYYMMDD_HHMMSS.csv  # MaterialBENCH choice の結果
├── materialbench-free_YYYYMMDD_HHMMSS.csv    # MaterialBENCH free の結果
├── summary_YYYYMMDD_HHMMSS.json              # 評価サマリー
└── *_judged.csv                              # LLM judge 結果
```

#### CSV カラム

**MaterialFigBench:**
- `question_id`: 問題 ID
- `problem_sentence`: 問題文
- `image_files`: 画像ファイル名
- `prediction`: モデルの回答
- `ground_truth`: 正解
- `answer_range_2`: 正解範囲（数値範囲用）
- `correct`: 正誤（True/False）
- `llm_judge_correct`: LLM による正誤判定（--use-llm-judge または --llm-judge-only オプション使用時）
- `llm_judge_reason`: 不正解時の理由（--use-llm-judge または --llm-judge-only オプション使用時）

**MaterialBENCH choice:**
- `question_id`: 問題 ID
- `problem_sentence`: 問題文
- `choices_a/b/c/d`: 選択肢
- `prediction`: モデルの回答（a/b/c/d）
- `correct_choice`: 正解
- `correct`: 正誤
- `llm_judge_correct`: LLM による正誤判定（--use-llm-judge または --llm-judge-only オプション使用時）
- `llm_judge_reason`: 不正解時の理由（--use-llm-judge または --llm-judge-only オプション使用時）

**MaterialBENCH free:**
- `question_id`: 問題 ID
- `problem_sentence`: 問題文
- `prediction`: モデルの回答
- `correct_answer`: 正解
- `correct`: 正誤（完全一致または数値許容誤差）
- `llm_judge_correct`: LLM による正誤判定（--use-llm-judge または --llm-judge-only オプション使用時）
- `llm_judge_reason`: 不正解時の理由（--use-llm-judge または --llm-judge-only オプション使用時）

#### サマリー JSON

```json
{
  "timestamp": "2026-03-05T16:28:10",
  "datasets": {
    "materialbench-choice": {
      "total": 164,
      "correct": 120,
      "accuracy": 0.732,
      "llm_judge_correct": 135,
      "llm_judge_accuracy": 0.823
    }
  },
  "overall": {
    "total": 164,
    "correct": 120,
    "accuracy": 0.732,
    "llm_judge_total": 164,
    "llm_judge_correct": 135,
    "llm_judge_accuracy": 0.823
  }
}
```

#### LLM Judge の理由出力

`--use-llm-judge` または `--llm-judge-only` オプションを使用すると、不正解だった場合に理由も出力されます。

**出力例:**
```
INCORRECT: The model predicted option (c), but the correct answer is (b).
INCORRECT: The predicted value 5.2 differs from the ground truth 3.8 by more than 1%.
```

**理由の表示:**
LLM judge 実行後、コンソールに不正解サンプルの理由が最大 5 件表示されます。CSV ファイルには全てのサンプルの `llm_judge_reason` カラムとして保存されます。

## データセット

### MaterialFigBench
- **形式**: 画像＋テキスト（マルチモーダル）
- **問題数**: 137 問
- **内容**: 結晶方位（ミラー指数）、相図、熱処理、機械的性質など
- **画像**: 最大 3 枚/問題、ローカルキャッシュ（~/.cache/material_bench/images/）

### MaterialBENCH
- **choice_dataset.json**: 4 選択肢問題（164 問）
- **free_dataset.json**: 自由記述問題（144 問）
- **内容**: 原子量計算、結晶構造、拡散、相平衡など

データソース：https://huggingface.co/datasets/omron-sinicx/

## 正解判定ロジック

- **大文字小文字無視**: `Martensite` == `martensite`
- **空白正規化**: 複数空白・タブ・改行を単一スペースに
- **数値範囲**: `0.0825-0.0975` 形式はモデル回答が範囲内か判定
- **完全一致**: 上記正規化後、完全一致で判定

## ベンチマーク結果

### 通常正解率（完全一致）

| モデル | FigBench | Choice | Free | Overall |
|--------|----------|--------|------|---------|
| Qwen3.5-397B-A17B | 22.14% (29/131) | 77.44% (127/164) | 52.08% (75/144) | 52.62% (231/439) |
| Qwen3.5-27B | 22.90% (30/131) | 67.68% (111/164) | 49.31% (71/144) | 48.29% (212/439) |

### LLM Judge 正解率

| モデル | FigBench | Choice | Free | Overall |
|--------|----------|--------|------|---------|
| Qwen3.5-397B-A17B | 41.22% (54/131) | 77.44% (127/164) | 86.11% (124/144) | 69.48% (305/439) |
| Qwen3.5-27B | 42.75% (56/131) | 67.68% (111/164) | 84.03% (121/144) | 65.60% (288/439) |

## TODO

### 短期

### 中期
- [x] バッチ処理の最適化（並列推論）
- [x] LLM による評価オプション（自由記述問題の柔軟な判定）
- [x] 既存結果への LLM judge 適用モード
- [ ] flexeval 統合の検討
- [ ] プログレス表示の改善（残り時間予測）

### 長期
- [ ] 評価指標の拡充（F1 スコア，BLEU など）


## ライセンス

MIT


**MaterialFigBench の課題:**

1. **複数画像の処理**: 2〜3 枚の画像を使用する問題は処理時間が長く、タイムアウトのリスク
2. **図表の正確な読み取り**: 相図、TTT 線図、応力 - ひずみ曲線などから数値を読み取る精度に課題
3. **ミラー指数の表記**: `[\bar{1}\bar{1}\bar{1}]` など特殊な表記の扱い
4. **段階的な計算問題**: 複数の図を参照して段階的に計算する問題の精度

**改善の余地:**
- [ ] 画像の事前処理（コントラスト強調など）
- [ ] 複数画像の重み付け
- [ ] 図表読み取り専用のプロンプト
- [ ] タイムアウト時間の調整（画像数に応じて可変）
