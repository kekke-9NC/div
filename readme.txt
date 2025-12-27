# Meteor Detector v0.8.27

## 概要

Meteor Detector は、動画ファイルやRTSPストリームから流星を自動検出し、解析するためのGUIアプリケーションです。深層学習モデル（ResNet系CNN）を使用して流星候補を分類し、Astrometry.net APIを利用したプレートソルブ機能により天球座標（赤経赤緯）を特定することができます。

## 主な機能

### 流星検出
- 動画ファイル処理: MP4, AVI, MOV 形式の動画から流星候補を検出
- RTSPストリーム対応: ネットワークカメラからのリアルタイム処理
- 定期スキャン: 指定フォルダを定期的に監視して新規動画を自動処理
- 時間制限機能: 指定時刻のみ処理を実行（天文薄明時間の自動設定対応）

### AI による判定
- ResNet 系アーキテクチャのCNN モデルによる流星判定
- TTA (Test-Time Augmentation) を適用した高精度予測
- 流星/非流星の自動分類と保存先振り分け

### 天体座標解析
- Astrometry.net API を使用したプレートソルブ
- 流星の出現位置（赤経赤緯）の自動算出
- 既存の WCS ファイルからの座標情報読み込み

### 解析可視化機能
- 複数の流星検出結果を天球図上にプロット
- カスタム座標点の追加管理（放射点等）
- 長時間輝線マップの作成
- ゆがみ補正機能
- 角度分布分析

### 出力オプション
- 動画クリップ (.mp4)
- 切り出し差分画像 (.jpg)
- 全体差分画像 (.jpg)
- 比較明合成画像 (.jpg)
- 検出情報テキストファイル (.txt)
- 概要動画 (.mp4)

## 必要環境

### システム要件
- Windows 10/11
- Python 3.8 以降
- CUDA 対応 GPU（推奨、CPU のみでも動作可能）

### 依存パッケージ
- torch
- torchvision
- opencv-python
- numpy
- Pillow
- astropy
- tkinter
- tkinterdnd2
- requests
- matplotlib

## ファイル構成

div/
 main_gui.py          # メインGUIアプリケーション
 config.py            # 設定パラメータ
 model.py             # 深層学習モデル定義推論
 video_processing.py  # 動画処理メイン
 image_processing.py  # 画像処理（線分検出等）
 astrometry.py        # プレートソルブ天体座標
 file_utils.py        # ファイル監視管理
 tracking.py          # オブジェクトトラッキング
 video_creation.py    # 概要動画作成
 download_pipeline.py # ダウンロード処理パイプライン
 network_copy.py      # ネットワークコピー
 meteor_sky_viewer.py # 天球図表示
 coordinate_manager.py# 座標点管理
 long_exposure_map.py # 長時間輝線マップ
 distortion_correction.py # ゆがみ補正
 meteor_angle_analysis.py # 角度分布分析
 location_utils.py    # 位置情報取得
 sun_times.py         # 日の出日の入り計算
 auto_time_updater.py # 時刻自動更新
 status_panel.py      # ステータスパネルUI
 ui_state.py          # UI状態管理
 utils.py             # ユーティリティ
 app_state.py         # アプリ状態管理
 model_latest_1.pth   # 学習済みモデル
 app_settings.json    # 保存された設定

## 使い方

### 1. アプリケーションの起動
    cd div
    python main_gui.py

### 2. ソースの選択
- フォルダ/動画ファイル: ドラッグ＆ドロップで追加
- RTSP ストリーム: URL を入力して「追加」ボタン

### 3. 設定
- 定期スキャン: 監視フォルダとスキャン間隔を設定
- 時間制限: 処理を実行する時間帯を指定（「自動で設定」で天文薄明時間を自動取得）
- 処理パラメータ: 同時処理数、差分作成間隔期間を調整
- 保存先: 流星/非流星の保存ディレクトリを指定

### 4. プレートソルブ（オプション）
1. 「動画から実行」で基準動画を選択
2. 「実行」ボタンでAstrometry.net に送信
3. 成功すると天球座標が自動計算される

### 5. マスク設定（オプション）
- 検出マスク: 検出対象外エリア（建物、樹木等）を描画
- プレートソルブ用マスク: プレートソルブ時に無視するエリアを設定

### 6. 処理開始
「開始」ボタンで処理を開始。進捗はログとプログレスバーで確認可能。

### 7. 解析タブ
- 検出結果の .txt ファイルをドロップ
- 「解析開始」で天球図にプロット
- 「座標点を追加」でカスタムポイント（放射点等）を追加

## 設定パラメータ (config.py)

### 主要パラメータ
| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| MIN_LINE_LENGTH | 25 | 検出する直線の最小長さ（ピクセル） |
| METEOR_PROBABILITY_THRESHOLD | 0.5 | 流星判定の確率閾値 |
| DEFAULT_CONCURRENCY | 4 | 同時処理数 |
| DEFAULT_INTERVAL | 1 | 差分作成間隔（秒） |
| DEFAULT_DURATION | 1 | 差分作成期間（秒） |
| ASTROMETRY_API_KEY | - | Astrometry.net API キー |

## 注意事項

- Astrometry.net の API キーは nova.astrometry.net で取得してください
- GPU を使用する場合は CUDA 対応の PyTorch をインストールしてください
- 大量の動画を処理する場合は十分なディスク容量を確保してください
- RTSP ストリームは安定したネットワーク接続が必要です

## トラブルシューティング

### モデルファイルが見つからない
config.py の MODEL_PATH を正しいパスに設定してください。

### プレートソルブが失敗する
- API キーが正しいか確認
- ネットワーク接続を確認
- 画像に十分な星が写っているか確認

### 処理が遅い
- DEFAULT_CONCURRENCY を調整
- GPU が認識されているか確認 (Using device: cuda と表示されれば OK)

## ライセンス

このソフトウェアは個人利用を目的としています。

## 更新履歴

### v0.8.27
- ステータスパネル UI の改善
- ネットワークコピー機能の追加
- 角度分布分析機能の追加
- 自動時刻更新機能の追加
