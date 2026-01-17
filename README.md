# PN 2017 インターン課題（機械学習・数理分野）提出物

## 構成
- `src/`: 課題1〜5の実装
- `tests/`: 単体テスト
- `params.txt`: 課題5のパラメータ出力形式サンプル
- `REPORT.md`: 課題6のレポート（サンプル）

## 実行例
```bash
python3 src/train_cartpole.py --command "./cartpole.out" --output params.txt
python3 src/evaluate_cartpole.py --command "./cartpole.out" --params params.txt
```

## テスト
```bash
python3 -m pytest
```
