# SwiftUIフロントエンド

`swiftui/` にmacOS用SwiftUIフロントエンドを追加しました。動画・フォルダの入力、Finderからのドラッグ＆ドロップ、RTSP URL、検出開始/停止、進捗、ログ、保存先設定をSwiftUIで扱います。

## 起動

開発時はプロジェクトルートで次を実行します。

```sh
./run_swiftui.sh
```

初回は既存の `.venv-mac` をPython処理エンジンとして優先し、なければ `METEOR_PYTHON`、Homebrew Python、システムPythonの順で探します。

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

旧Tkinter UIは、機能移行中のフォールバックとして `./run_mac.command` から起動できます。マスク編集、プレートソルブ、NoiseTwin、定期スキャン、解析ツール、AIアシスタントなどは、既存動作を壊さず順次SwiftUIへ移行します。
