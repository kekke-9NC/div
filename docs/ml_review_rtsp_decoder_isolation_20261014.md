# レビュー動画とRTSP接続待ちの競合を解消（2026-10-14）

## v0.10.14.8で取得した本番の証拠

利用者から、動画が再生されず選別にも反応がないとの再報告があった。本番PID68328の新しい診断ログでは、preview開始からmetadata表示まで約15msだったが、`cv2.VideoCapture` のopenは **44.082秒** かかった。次frameの画像設定は開始から44.098秒だった。GUI描画だけの問題ではなく、復号器の開始待ちが残っていた。

同じPIDのOS sampleにはOpenCVのVideoCapture constructor→FFmpeg→native pollの停止があり、RTSPカメラ宛てTCPはSYN_SENTだった。サンプルだけではそのcaptureの入力を特定できないが、ログはレビューworkerのローカルclip openが止まっていた区間を示している。

インストール済みOpenCVは5.0.0。公式5.x実装にはprocess内のstatic mutexがあり、capture open時に通常そのlockを取る: [OpenCV公式実装](https://github.com/opencv/opencv/blob/5.x/modules/videoio/src/cap_ffmpeg_impl.hpp#L975)。共有mutexを無効化する環境変数は、録画側にも影響するため変更していない。

## 原因の再現診断

実カメラへ接続せず、localhostの診断用RTSP接続を4秒応答待ちにした。同じPython processで従来のOpenCV local clip openと、新しいPreviewStreamを並行実行した読み取り診断の結果:

| 対象 | 時間 |
| --- | ---: |
| RTSP openの応答待ち | 4.003秒 |
| 同一processの従来OpenCV local clip open | 4.002秒 |
| 独立FFmpeg readerの先頭frame取得 | 168.53ms |
| 独立reader停止・close完了 | 38.93ms |

従来のローカル動画はRTSP待ちと同じ時間待たされた。別processにしたreaderは、その待機中にframeを取得した。動画・判定データを変更しない診断。記録は `/private/tmp/meteor-review-rtsp-isolation-20261014/lock-diagnostic.json`。

## 修正

- レビュー動画のOpenCV VideoCaptureを撤去。専用ffprobe子processでサイズ・fpsを取得し、専用ffmpeg子processでローカルclipだけをRGB rawvideoへ復号する。
- FFmpegのstdoutからframe byte数を完全に読み取り、途中の短いpipe readを結合する。最大420px、元のaspect比とframe順序を保持し、loop再生する。frame queueは最大2件。
- stdin/stderrはDEVNULL、shellを使用せずargv配列で起動。decoder/filterのthreadsを1にする。RTSP URLは渡さない。
- stopはcancelと子processへのsignalだけ。workerでterminate→wait→必要ならkill→reap、stdout close後にclosedを通知する。保存workerはそのclosedを待ってから移動する。
- ffprobeは5秒timeout。復号のpipe readにも5秒のwatchdogを設け、停止中のreadを解除する。queueのbackpressure/ユーザーpause中はread timeoutを走らせない。
- Popen公開前後とstopの競合、cleanup例外時の参照解放を扱う。`preview_decoder_started` は子の起動時刻、最初のframe取得・Tk画像設定は別のstageとする。
- 「表示」メニューに既存と同じ目視レビュー画面を開く項目を追加し、native menuからもアクセス可能にした。

## 今回は実アプリの確定後の遷移を確認

v0.10.14.9 / PID93548を再起動し、CUAの「表示」メニューから原本データのレビューを開いた。本番アプリにはRTSPのSYN_SENT接続が同時にある状態だった。最初のpreviewは開始から **214.89ms** でframeをTk画像へ設定した。CUAの実画面でも動画・静止画・有効な確定ボタンを確認した。

21:32:58〜21:33:17 JSTに本番アプリへ記録された **21件の実際の確定操作** を読み取り集計した。こちらからM/Nキー・判定ボタンは操作していない。

| 確定callback開始→次動画の最初のframe設定 | 時間 |
| --- | ---: |
| 最短 | 174.83ms |
| 中央値 | 231.51ms |
| 最長 | 256.87ms |

次のeventに移る際のdecoder終了待ち、判定保存、GUI queue受領、新decoder起動、最初のframe画像設定を含む。物理的なマウス入力前の待ち、ディスプレイの提示時刻を計測したものではない。終了後の最後の動画も2,423フレームまで表示設定が継続し、`preview_render_error` は0件だった。保存workerが待っていた約44秒のopen競合は、これらの実操作では再発していない。

個別判定と動画表示の確認範囲であり、一括選別・undoの本番操作時間を追加計測したわけではない。GPT-6 Lunaが実装・子process停止・pipe読み取り・保存前closeの順序を監査。自動テストは追加・実行していない。

## 配備

ブランチ `codex/review-confirmation-latency`。処理コード `55455dc`、版表示 `ed3fa93`、**v0.10.14.9**。review関連3モジュール、gui_app、config、bootstrapの6ファイルを設置先・payloadへ配備し、計12ファイルのSHA256一致を確認した。

旧PID68328は開発中維持。切替前に一時録画PID93412を起動し、正常終了・lock解放後に更新。21:28:29 JSTに新PID93548を `--resume-rtsp` で起動した。切替要求から更新・再起動まで約28.12秒。v.9の実画面を確認した。既存データ・モデルは配備で置換していない。

本体・一時録画ともカメラTCPはSYN_SENT。一時録画は完成chunk 0件で正常終了した。新アプリの再接続待ちと、同時に動くレビュー専用ffmpeg子processを確認した。現時点でカメラ録画に成功したとは主張しない。レビュー停止済みdecoderの残存はなく、確認時点の子processは表示中の1つだけだった。

バックアップ: `/Users/keisukematsumoto/Library/Application Support/MeteorDetector/backups/review-decoder-20261014-212801`。
本番操作時間・配備記録: `/private/tmp/meteor-review-decoder-20261014`。

検索用web-searchのmwmbl/DuckDuckGoは0件で検索フェーズを終了した。上記の実装照合は既知のOpenCV公式ファイルを直接取得して行った。
