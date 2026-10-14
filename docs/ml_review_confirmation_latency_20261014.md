# 確定ボタンから次のデータへの遷移（2026-10-14）

## 調査結果

利用者から、v0.10.14.6でも「非流星として確定」後に約10秒待つとの報告があった。

実際の最後の判定は18:45:36 JST。予測 not_meteor → 確定 not_meteor、`media_moves: []` だった。この操作には保存メディアの移動がなく、前回のメディア検索索引の改善だけでは待ち時間を説明できない。

次の未確認データ3件の読み取り診断では、metadataは0.078〜0.268ms、PNGは0.728〜5.626ms、動画openは0.497〜10.475ms、先頭frameは0.126〜2.790ms、releaseは0.009〜0.133msだった。原本の判定は変更していない。

18:47:08の実アプリPID31604のOS sampleでは、main threadの1504サンプル中875がQuartzCoreのtransaction flush、749が画像読み取り、約629が色変換にあった。動画readは12サンプルだった。GUIの画像描画負荷・イベント処理の競合が有力だが、これは利用者の10秒の操作そのものを計測した結果ではない。

## 修正

- metadata・静止画像読み取り・動画open/read/変換/releaseを専用workerへ移す。Tk APIは画面スレッドだけで呼ぶ。
- queueを2件に制限し、古い動画の通知はstream identityとgenerationで無効化する。画面を閉じる、次へ移動する、検索中にpause/resumeする場合も扱う。
- 動画のTk画像更新をidle priorityで実行し、描画後に次のtimerを設定する。ボタン操作と保存完了のpollを動画timerが連続で追い越す構造を避ける。フレームは捨てず、queueで復号を待たせる。
- 通常の確定では全操作ボタンを毎回disable/enableせず、2つの確定ボタンだけを更新する。他の操作は既存busy guardで競合を防ぐ。
- metadataと最初の動画frameが届くまで確定を無効にする。metadata読取失敗時は確定できない。動画なしは明示し、古い動画のまま確定しないようにする。
- ファイル移動前のdecoder終了待ちは保存・refresh・bulkのworkerで行う。Tkでjoin/releaseしない。判定metadata、ファイル移動、journal fsync、ロールバックの順序は維持する。
- 保存開始・worker完了・GUI受領・次preview表示の時刻を、別の非同期診断ログへ記録する。macOSでは `~/Library/Logs/MeteorDetector/review_latency.jsonl`。128件のqueue、約2MBでrotate、0600で作成。原本パスや画像は書かない。

## 確認の範囲

GPT-6 Lunaが静的監査を実施。refresh/bulkでdecoder終了待ちを取り落とす問題と、metadataエラー時のキー操作の状態差を修正した。3モジュールの構文解析とdiff確認を実施。自動テストは追加・実行していない。

確認済みデータのコピーを `/private/tmp/meteor-confirmation-delay-20261014` に用意し、検出メディアへの参照を外した。実アプリと同じPython runtimeでbackendだけを診断した2回の結果は、判定保存0.54〜0.91ms、次動画の先頭frame復号3.59〜8.73ms、合計4.13〜9.64msだった。両方とも同じbackendを使用したため、修正前後の高速化倍率ではない。原本の未確認データには分類操作を行っていない。

コピーのTk画面はCUAのアプリ取得がtimeoutとなり、実際のボタン入力から画面遷移までの測定はできなかった。修正版のコピー画面でも画像描画負荷は残っており、native描画のコスト自体が消えたとは主張しない。今回修正したのはGUI上の同期処理とイベント競合であり、「10秒の待ちが解消した」「何秒になった」という実測の主張はしない。次の実操作は追加した時刻ログで追跡できる。

## 配備

ブランチ `codex/review-confirmation-latency`。処理コード `89e6816`、読込失敗時の表示修正 `15e937d`、版表示 `e6e7cfc`、v0.10.14.7。別タスクのv6モデルcommitを含む既存HEADから分岐したが、配備したのはreview関連3モジュールとconfig/bootstrapの5ファイルだけ。

設置先・同梱payloadの計10ファイルのSHA256一致を確認した。旧PID31604の正常終了とlock解放後に更新し、19:14:29 JSTに新PID55268を `--resume-rtsp` で起動。CUAでv0.10.14.7の実画面と設定復元・RTSP開始ログを確認した。更新前のSIGTERM要求から再起動の実行まで約25秒。判定データ、選択モデルv5、録画原本は配備で置き換えていない。

切替前に一時録画PID54805を起動した。本体・一時録画ともカメラTCPがSYN_SENTのままで、完成chunkは0件。一時録画は正常終了し、新アプリの再接続待ちが動作していることを確認した。カメラへの接続や録画成功を確認したとは主張しない。

動画読込失敗時に前イベントの動画が残らないよう、追加修正を反映して19:20:30 JSTにPID58431へ再起動した。最終切替も一時録画PID57806を先に起動し、SIGTERM要求から更新・再起動の実行まで約23秒だった。最終版の10ファイル一致とCUAでのv0.10.14.7表示を確認。一時録画は完成chunk 0件で正常終了した。最終的に稼働しているのはPID58431で、カメラTCPは再接続待ち。原本review journalの最終書込みは18:45:36のままで、調査中に原本を分類していないことも確認した。

バックアップ: `/Users/keisukematsumoto/Library/Application Support/MeteorDetector/backups/confirmation-latency-20261014-191343`。
