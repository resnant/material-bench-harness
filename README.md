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

# 並列実行で高速化（3スレッド）
pixi run python scripts/run_benchmark.py \
  --dataset materialbench-choice \
  --api-base http://localhost:8001 \
  --max-samples 10 \
  --num-threads 3 \
  --output-dir ./results
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

### 出力ファイル

```
results/
├── material-figbench_YYYYMMDD_HHMMSS.csv    # MaterialFigBench の結果
├── materialbench-choice_YYYYMMDD_HHMMSS.csv  # MaterialBENCH choice の結果
├── materialbench-free_YYYYMMDD_HHMMSS.csv    # MaterialBENCH free の結果
└── summary_YYYYMMDD_HHMMSS.json              # 評価サマリー
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

**MaterialBENCH choice:**
- `question_id`: 問題 ID
- `problem_sentence`: 問題文
- `choices_a/b/c/d`: 選択肢
- `prediction`: モデルの回答（a/b/c/d）
- `correct_choice`: 正解
- `correct`: 正誤

**MaterialBENCH free:**
- `question_id`: 問題 ID
- `problem_sentence`: 問題文
- `prediction`: モデルの回答
- `correct_answer`: 正解
- `correct`: 正誤

#### サマリー JSON

```json
{
  "timestamp": "2026-03-05T16:28:10",
  "datasets": {
    "materialbench-choice": {
      "total": 164,
      "correct": 120,
      "accuracy": 0.732
    }
  },
  "overall": {
    "total": 164,
    "correct": 120,
    "accuracy": 0.732
  }
}
```

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

データソース: https://huggingface.co/datasets/omron-sinicx/

## 正解判定ロジック

- **大文字小文字無視**: `Martensite` == `martensite`
- **空白正規化**: 複数空白・タブ・改行を単一スペースに
- **数値範囲**: `0.0825-0.0975` 形式はモデル回答が範囲内か判定
- **完全一致**: 上記正規化後、完全一致で判定

## TODO

### 短期
- [ ] 数値表記の正規化改善（`2.4x10^-9` ↔ `2.4 × 10^{-9}`）
- [ ] 単位の扱いの改善（`500 h` ↔ `500 hours`）
- [ ] 部分正解の判定（単語の包含関係など）
- [ ] エラーログの改善（失敗サンプルの理由記録）

### 中期
- [x] バッチ処理の最適化（並列推論）
- [ ] LLM による評価オプション（自由記述問題の柔軟な判定）
- [ ] flexeval 統合の検討
- [ ] プログレス表示の改善（残り時間予測）

### 長期
- [ ] 他のベンチマークへの拡張
- [ ] 評価指標の拡充（F1 スコア，BLEU など）
- [ ] Web UI による結果可視化
- [ ] CI 統合（自動ベンチマーク実行）

## 既知の問題

- MaterialFigBench の画像処理に時間（初回ダウンロード時）
- 正解判定が厳しすぎる場合あり（数値表記、単位の差異）
- vLLM API のベース URL は `/v1` を付与が必要

## ライセンス

MIT

## 性能参考

| データセット | 件数 | スレッド数 | 実行時間 |
|-------------|------|-----------|---------|
| materialbench-choice | 10 | 1 | ~4分 |
| materialbench-choice | 10 | 3 | ~2分44秒 |
| materialbench-choice | 10 | 5 | ~2分 |

## 実験結果と考察

### 改善の経過

| 時点 | 正答率 | サンプル数 | 改善内容 |
|------|--------|-----------|----------|
| 初期（完全一致） | 3.33% | 30 | 基本実装 |
| 回答抽出＋5% 許容 | 30% | 10 | **Answer:**形式の抽出、数値許容誤差 |
| 空白削除＋3% 許容 | **46%** | 50 | 空白完全削除、単位の扱い改善 |

### 50 問実験の詳細（MaterialBENCH free）

**実験条件:**
- データセット：MaterialBENCH free（自由記述問題）
- サンプル数：50 問（ランダム抽出）
- 並列スレッド：10
- 処理時間：約 7 分 48 秒（1 問あたり平均 9.4 秒）

**結果:**
- 正解数：23 問
- 正答率：46.0%

**正解した問題の例:**
- 問 29: 拡散束の計算（2.4 × 10^-9 kg/m²s）
- 問 7: FCC 充填率（0.74）
- 問 8,9: 密度計算（8.89 g/cm³, 7.31 g/cm³）
- 問 58: 応力集中係数（2404 MPa）
- 問 90: 電線直径（1.88 mm）

**不正解の主要原因:**

1. **複数値の形式の違い**
   - モデル：`Greater than, Less than`
   - 正解：`greater, less`
   - 課題：単語の省略・接続詞の扱い

2. **単位付きの数値**
   - モデル：`6.05x10^28 atoms/m³`
   - 正解：`6.05x10^28`
   - 課題：単位の自動削除

3. **有効数字の違い**
   - モデル：`1.58`
   - 正解：`1.6`
   - 課題：3% 許容でも厳しい場合あり

4. **記述形式の違い**
   - モデル：`72.5 at% Sn, 27.5 at% Pb`
   - 正解：`Sn: 72.5at%, Pb: 27.5at%`
   - 課題：コロン・百分率の順序

### 改善の余地

#### 短期的な改善
- [ ] 単位の自動削除：数値抽出時に単位（mm, MPa, kg/m³など）を除去
- [ ] 有効数字の柔軟化：2 桁以上の違いがある場合のみ不正解
- [ ] 複数値の順序正規化：カンマ区切り値をソートして比較
- [ ] 化学式の正規化：`at%` と `at %` の統一

#### 中長期的な改善
- [ ] LLM による意味的評価：数値以外の概念問題を LLM が評価
- [ ] 部分正解の導入：途中式が正しい場合に部分点
- [ ] flexeval 統合：評価フレームワークの活用
- [ ] エラー分析の自動化：不正解パターンの自動分類
