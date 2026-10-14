# v0.10.14.7の表示停滞とv0.10.14.8の修正

## 実操作で分かったこと

v0.10.14.7を反映後も利用者から「動画が再生されない」「選別しても反応がない」との報告があった。前回の対策では解決できていなかった。

PID58431の実操作ログ（19:39:23〜19:39:52 JST）では次の時刻が記録された。

| 区間 | 開始からの時間 |
| --- | ---: |
| 確定callback開始 | 0 |
| 保存worker完了 | 51.94ms |
| GUIが保存完了を受領 | 69.90ms |
| 次の動画frameをTk画像へ設定 | 29.170秒 |

保存自体は約0.62msで、worker開始までのdecoder終了待ちが約51ms。この操作には検出メディアの移動がなかった。遅延は保存完了受領後のプレビュー表示経路にある。

v.7はqueueから取り出したmetadata/frameを `after_idle` へ退避し、描画後に次のtimerを設定する処理も `after_idle` にしていた。最初の動画まで複数回のidle機会が必要となり、main GUI/nativeイベントが継続すると表示が後回しになる経路だった。v.7の実操作ログと整合する具体的な原因候補であり、ソース監査でも指摘された。ただしv.7にはdecoder開始・画像設定個別の時刻がなく、29秒をこの要因だけに完全分解できたわけではない。

また、インストール済みPillowの `ImageTk.PhotoImage.paste` の説明にも、表示中の画像では非常に遅くなり得る旨が記載されていた。前回の表示中画像をpasteで使い回す変更を撤去した。

## 修正

- previewの `after_idle` をすべて撤去。単一の通常timerでqueueを1件ずつ取り出し、そのcallback内で描画する。ready後5ms、queueが空なら10ms、frame後はfpsに応じた間隔（最低5ms）で次を設定する。
- 表示中の `PhotoImage.paste` を撤去。非表示の新画像に画素を設定してからlabelへ差し替える。動画の旧画像はlabelが新画像を参照するまで保持し、先に破棄して空になる瞬間を避ける。
- 次イベントへの移動時は旧動画を消して「読み込み中」と明示する。
- 描画例外でtimerが無言で止まらないよう、再読み込み案内を表示し、確認ボタンとM/Nキーを無効化する。失敗時に前の動画も残さない。
- decoder・metadata queue・metadata表示・最初のframe・5秒ごとの累計表示frame数を非同期診断ログへ追加。イベント原本のパス/画像は記録しない。
- decoder worker、最大2件queue、frameの順序保持、Windowsのファイル移動前のdecoder終了待ち、判定journal/fsync/rollbackは維持する。

## 診断と限界

GPT-6 Lunaが読み取り専用で監査。例外時にボタンだけ無効でもM/Nキーが判定できる問題を指摘し、metadata-valid flagもfalseにする修正を入れた。3モジュールを構文解析し、diffを確認した。自動テストは追加・実行していない。

実アプリと同じインストール済みPython runtime、同じレビューclassとスタイルを使用し、確認済みデータのコピーを開く診断アプリPID65902を起動した。実際のTk callbackが記録した結果:

- preview開始から最初のframeの画像設定まで **32.29ms**。
- metadata描画callbackは約3.19ms、最初のframe描画callbackは約0.43ms。
- **221.28秒・3,961フレーム** の画像設定が継続。`preview_render_error` は0件。
- 起動時の同一表示経路の診断であり、旧実操作の29秒と厳密な修正前後の倍率比較はしない。画面に実際に提示された時刻と、Tk画像設定完了時刻も区別する。

コピー診断アプリのCUA AX取得はtimeoutとなった。実アプリのnative menuとキーボードの画面切り替えは操作できたが、座標によるTkボタン操作は反映されなかった。よって原本データを選別せず、実アプリでの確定callback→次frameの時間や、取り消し/一括の操作実測は行っていない。これらの解消を実測済みとは主張しない。v.8では同一実アプリの次操作をdecoder/queue/描画の個別時刻で追跡できる。

## 配備

ブランチ `codex/review-confirmation-latency`。処理修正 `7680623`、版表示 `74999b4`、**v0.10.14.8**。

旧PID58431は開発中維持し、切替前に一時録画PID67991を起動。SIGTERMによる終了とlock解放を確認してからreview関連3モジュールとconfig/bootstrapを更新した。設置先・同梱payload計10ファイルのSHA256一致を確認。19:48:26 JSTに新PID68328を `--resume-rtsp` で起動した。切替要求から更新・再起動まで約22.54秒。CUAで実画面のv0.10.14.8、設定復元、RTSP開始ログを確認した。

カメラTCPは本体・一時録画ともSYN_SENT。一時録画の完成chunkは0件で正常終了した。新アプリは再接続待ち。カメラ録画が成功したとは主張しない。既存データ・判定ログ・モデルは配備で置換していない。

バックアップ: `/Users/keisukematsumoto/Library/Application Support/MeteorDetector/backups/review-stable-20261014-194803`。
診断と配備記録: `/private/tmp/meteor-review-stable-20261014`。
