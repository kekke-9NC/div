# SwiftUIフロントエンド

`swiftui/` にmacOS用SwiftUIフロントエンドを追加しました。動画・フォルダの入力、Finderからのドラッグ＆ドロップ、RTSP URL、検出開始/停止、進捗、ログ、保存先設定をSwiftUIで扱います。

## 起動

プロジェクトルートで次を実行すると、SwiftUI版が標準UIとして起動します。

```sh
./run_mac.command
```

ターミナルから起動する場合は `./run_swiftui.sh` または `./run_mac.sh` を使えます。`run_mac.sh` は `.venv-mac` の作成と依存関係の準備を行い、`METEOR_PYTHON` を指定した場合はそのPythonを使います。既存環境だけで起動したい場合は `run_swiftui.sh` を使ってください。旧Tkinter版へ戻す必要がある場合は `./run_legacy_mac.sh` を使います。

`run_mac.sh` は `METEOR_PYTHON` が指定されていればそれを優先し、未指定なら既存の `.venv-mac` を使い、未作成ならPython 3.11またはPython 3から作成します。`run_swiftui.sh` は既存の `.venv-mac` または `.venv` を使い、見つからない場合はブリッジ側がHomebrew PythonやシステムPythonを探索します。

## 構成

- `swiftui/Sources/MeteorDetectorApp/`: SwiftUI画面、状態管理、Pythonプロセス接続
- `swiftui/Sources/MeteorDetectorCore/`: UIから独立した共通モデルと検証用ロジック
- `swift_backend.py`: Tkinterを読み込まず、NDJSONでSwiftUIと既存Python処理を接続
- `app_settings.json`: 旧UIと共有する設定ファイル。未知の項目を保持したまま基本設定を更新

SwiftUIとPythonの境界は標準入力/標準出力のNDJSONです。標準出力はJSON専用、診断情報は標準エラー出力に限定しています。

## 検証

```sh
swift build --package-path swiftui
swift run --package-path swiftui MeteorDetectorCoreValidation
.venv-mac/bin/python -m pytest -q tests/test_swift_backend.py
```

旧Tkinter UIは、移行済み機能では再利用せず、未移行機能のフォールバックとして `./run_legacy_mac.sh` から起動できます。サマリー出力、検出マスクの適用と動画フレーム上でのマスク描画、検出モデルの選択、既存WCS／固定カメラ補正の適用、入力種別の優先順位、定期スキャン、RTSP運用、NoiseTwin／時間平均の設定はSwiftUI版で扱えます。補正データの新規作成、RTSP固定パターン補正、解析ツール、AIアシスタントなどは、既存動作を壊さず順次SwiftUIへ移行します。
