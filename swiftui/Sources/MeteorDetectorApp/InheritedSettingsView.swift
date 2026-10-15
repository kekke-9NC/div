import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct SettingsLoadOverlay: View {
    @EnvironmentObject private var store: WorkspaceStore

    var body: some View {
        if !store.settingsLoaded {
            VStack(spacing: 12) {
                if let error = store.settingsLoadError {
                    Label("設定を読み込めませんでした", systemImage: "exclamationmark.triangle")
                    Text(error).font(.caption).textSelection(.enabled)
                    Button("再試行") { store.reloadSettings() }
                    Button("旧アプリの設定を選択…") { chooseSettingsFile(store) }
                        .disabled(store.isImportingSettings)
                } else {
                    ProgressView("設定を読み込んでいます…").controlSize(.small)
                }
            }
            .padding(20)
            .frame(maxWidth: 480)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
        }
    }
}

@MainActor
private func chooseSettingsFile(_ store: WorkspaceStore) {
    let panel = NSOpenPanel()
    panel.title = "旧アプリの設定を取り込む"
    panel.message = "選択した設定を優先して取り込みます。現在の設定はバックアップされます。"
    panel.allowedContentTypes = [.json]
    panel.allowsMultipleSelection = false
    panel.directoryURL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/MeteorDetector")
    if panel.runModal() == .OK, let url = panel.url { store.importLegacySettings(from: url) }
}

struct InheritedSettingsView: View {
    @EnvironmentObject private var store: WorkspaceStore

    var body: some View {
        VStack(spacing: 24) {
            GlassCard {
                VStack(alignment: .leading, spacing: 12) {
                    SectionTitle("旧アプリの設定", subtitle: "保存済みの全項目を保持し、対応する処理へ引き継ぎます")
                    Text("初回は旧アプリの設定も確認し、新しい値を優先して不足する項目を補完します。別の設定を優先したい場合はファイルを選択してください。")
                        .font(.caption).foregroundStyle(AppTheme.secondaryText)
                    Button { chooseSettingsFile(store) } label: {
                        Label(store.isImportingSettings ? "取り込み中…" : "旧アプリの設定を取り込む…", systemImage: "square.and.arrow.down")
                    }
                    .disabled(store.isBusy || store.isSavingSettings || store.isImportingSettings || store.cameraModelMonitorActive || store.cameraModelMonitorChanging)
                }
            }
            LegacySettingsCard(title: "検出の詳細条件", subtitle: "旧アプリのしきい値と追跡条件を復元します", fields: Self.advanced)
            LegacySettingsCard(title: "動画の時刻表示と圧縮", subtitle: "検出後に保存する動画に適用します", fields: Self.video)
            LegacySettingsCard(title: "学習データと自動マスク", subtitle: "流星判定・境界付近・一部の低スコア候補をレビュー用に保存します", fields: [
                .toggle("学習データを保存", "ml_training_export_enabled"),
                .text("学習データの保存先", "ml_training_data_root"),
                .text("判定閾値の直下を残す幅（既定0.30）", "ml_training_boundary_fraction", default: "0.30"),
                .toggle("動画からマスクを自動生成", "auto_video_mask_enabled"),
            ])
            LegacySettingsCard(title: "AI解析", subtitle: "旧アプリで選択した解析方法と接続先を使用します", fields: [
                .choice("解析方法", "ai_vlm_backend", default: "lmstudio_qwen3_5_2b", choices: ["lmstudio_qwen3_5_2b", "local_qwen3_vl_4b"]),
                .text("LM Studio の接続先", "lm_studio_vlm_url", default: "http://localhost:1234/v1"),
                .text("LM Studio のモデル", "lm_studio_vlm_model_id", default: "qwen/qwen3-vl-4b"),
                .secret("LM Studio のAPIキー", "lm_studio_vlm_api_key"),
            ])
            LegacySettingsCard(title: "カメラ制御と観測時間", subtitle: "接続先・露出目標・時間帯の自動更新設定を復元します", fields: [
                .text("カメラ制御の接続先", "camera_control_base_url"),
                .text("露出の目標値", "camera_control_ev_target", default: "0.0"),
                .toggle("観測時間帯を毎日自動更新", "auto_time_updater_enabled"),
            ])
            CameraControlSettingsCard()
            GlassCard {
                VStack(alignment: .leading, spacing: 12) {
                    SectionTitle("RTSPカメラ補正の自動監視", subtitle: "保存済みの更新間隔・雲量条件・AI接続先を使って補正モデルを更新します")
                    Button(store.cameraModelMonitorActive ? "自動監視を停止" : "自動監視を開始") { store.toggleCameraModelMonitor() }
                        .disabled(store.cameraModelMonitorChanging || store.cameraModelBuildActive)
                    Text(store.cameraModelMonitorStatus).font(.caption).foregroundStyle(AppTheme.secondaryText)
                }
            }
            LegacySettingsCard(title: "カメラ補正の保存済み情報", subtitle: "旧アプリのモデル・参照動画・解決方法も保持します", fields: [
                .text("登録済み補正モデル", "plate_solve_model_path"),
                .text("補正用の参照動画", "plate_solve_video_path"),
                .choice("プレートソルブの方法", "plate_solve_mode", default: "local", choices: ["local", "api"]),
                .secret("Astrometry.net のAPIキー", "astrometry_api_key"),
            ])
            GlassCard {
                VStack(alignment: .leading, spacing: 12) {
                    SectionTitle("追加の検出モデル", subtitle: "旧アプリで登録したモデル一覧を保持します。1行に1つのパスを入力してください")
                    TextEditor(text: Binding(
                        get: { (store.legacyValue(["custom_model_paths"]) as? [String] ?? []).joined(separator: "\n") },
                        set: { store.setLegacyValue($0.components(separatedBy: .newlines).filter { !$0.isEmpty }, at: ["custom_model_paths"]) }
                    ))
                    .font(.system(.caption, design: .monospaced))
                    .frame(height: 90)
                }
            }
        }
        .disabled(store.isImportingSettings)
    }

    private static let advanced: [LegacySettingField] = [
        .text("粗検出する線の最小長", "advanced_settings.min_line_length", default: "25"),
        .text("画像端の除外幅", "advanced_settings.border_size", default: "30"),
        .text("重複検出の距離", "advanced_settings.duplicate_thresh", default: "100"),
        .text("流星と判定する確率", "advanced_settings.meteor_prob", default: "0.5"),
        .text("詳細検出の時間幅（秒）", "advanced_settings.finer_window_sec", default: "4"),
        .text("詳細検出の合成間隔", "advanced_settings.finer_comp_step", default: "3"),
        .text("詳細検出の線の最小長", "advanced_settings.finer_min_length", default: "15"),
        .text("詳細検出の前後の余白（秒）", "advanced_settings.finer_padding_sec", default: "1"),
        .text("詳細検出の切り出しサイズ", "advanced_settings.finer_cutout_size", default: "384"),
        .text("飛行機判定の継続時間（秒）", "advanced_settings.airplane_dur_thresh", default: "7"),
        .text("飛行機判定のフレーム数", "advanced_settings.airplane_frame_thresh", default: "7"),
        .text("同一物体を追跡する距離", "advanced_settings.tracking_dist_thresh", default: "200"),
        .text("検出クリップの最大長（秒）", "advanced_settings.max_clip_dur", default: "2"),
        .text("クリップの目安の長さ（秒）", "advanced_settings.clip_dur_sec", default: "3"),
        .text("保存する切り出しサイズ", "advanced_settings.cutout_size", default: "256"),
        .text("RTSP視野角の下限（度）", "advanced_settings.rtsp_scale_lower", default: "85"),
        .text("RTSP視野角の上限（度）", "advanced_settings.rtsp_scale_upper", default: "100"),
    ]

    private static let video: [LegacySettingField] = [
        .toggle("フルサイズ動画に時刻を表示", "full_video_timestamp.enabled", default: true),
        .choice("時刻表示の位置", "full_video_timestamp.position", default: "右下", choices: ["右下", "左下", "右上", "左上"]),
        .text("時刻の文字サイズ（画面高の%）", "full_video_timestamp.size_percent", default: "1.8"),
        .choice("補正動画の圧縮形式", "processed_video_encoding.codec", default: "H.265 / HEVC (推奨)", choices: ["H.265 / HEVC (推奨)", "H.264 / AVC", "MPEG-4 (互換)"]),
        .choice("補正動画の品質", "processed_video_encoding.quality", default: "入力品質基準（推奨）", choices: ["入力品質基準（推奨）", "最高品質", "高品質", "標準", "容量優先", "カスタム", "可逆圧縮（低速）"]),
        .text("補正動画のビットレート（Mbps）", "processed_video_encoding.bitrate_mbps", default: "40"),
    ]
}

private struct LegacySettingField: Identifiable {
    enum Kind { case text, toggle, choice([String]), secret }
    let title: String
    let key: String
    let fallback: String
    let kind: Kind
    var id: String { key }
    var path: [String] { key.components(separatedBy: ".") }
    static func text(_ title: String, _ key: String, default fallback: String = "") -> Self {
        Self(title: title, key: key, fallback: fallback, kind: .text)
    }
    static func secret(_ title: String, _ key: String) -> Self {
        Self(title: title, key: key, fallback: "", kind: .secret)
    }
    static func toggle(_ title: String, _ key: String, default fallback: Bool = false) -> Self {
        Self(title: title, key: key, fallback: fallback ? "true" : "false", kind: .toggle)
    }
    static func choice(_ title: String, _ key: String, default fallback: String, choices: [String]) -> Self {
        Self(title: title, key: key, fallback: fallback, kind: .choice(choices))
    }
}

private struct LegacySettingsCard: View {
    let title: String
    let subtitle: String
    let fields: [LegacySettingField]
    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 14) {
                SectionTitle(title, subtitle: subtitle)
                ForEach(fields) { LegacySettingRow(field: $0) }
            }
        }
    }
}

private struct LegacySettingRow: View {
    @EnvironmentObject private var store: WorkspaceStore
    let field: LegacySettingField
    private var binding: Binding<String> {
        Binding(get: { store.legacyString(field.path, default: field.fallback) },
                set: { store.setLegacyValue($0, at: field.path) })
    }
    var body: some View {
        switch field.kind {
        case .toggle:
            Toggle(field.title, isOn: Binding(
                get: { store.legacyBool(field.path, default: field.fallback == "true") },
                set: { store.setLegacyValue($0, at: field.path) }
            ))
        case .choice(let choices):
            let selected = binding.wrappedValue
            Picker(field.title, selection: binding) {
                ForEach(choices.contains(selected) ? choices : choices + [selected], id: \.self) { choice in
                    Text(["local": "このMacで解決", "api": "Astrometry.net", "lmstudio_qwen3_5_2b": "LM Studio", "local_qwen3_vl_4b": "ローカルQwenモデル"][choice] ?? choice).tag(choice)
                }
            }
        case .text, .secret:
            HStack {
                Text(field.title).frame(width: 240, alignment: .leading)
                if case .secret = field.kind {
                    SecureField(field.title, text: binding).textFieldStyle(.roundedBorder)
                } else {
                    TextField(field.title, text: binding).textFieldStyle(.roundedBorder)
                }
            }
        }
    }
}

@MainActor
private final class CameraControlInputState: ObservableObject {
    @Published var error = ""
}

private struct CameraControlSettingsCard: View {
    @EnvironmentObject private var store: WorkspaceStore
    @StateObject private var input = CameraControlInputState()
    private let fields = [
        ("brightness", "明るさ"), ("contrast", "コントラスト"), ("saturation", "彩度"),
        ("sharpness", "シャープネス"), ("auto_exposureEx", "自動露出"), ("max_exposure", "最大露出"),
        ("exposure_time", "露出時間"), ("auto_gain_mode", "自動ゲイン"),
        ("manual_AGain_enable", "アナログゲインを有効化"), ("manual_AGain", "アナログゲイン"),
        ("manual_DGain_enable", "デジタルゲインを有効化"), ("manual_DGain", "デジタルゲイン"),
        ("wdr_level", "WDR"),
    ]
    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 12) {
                SectionTitle("カメラの手動設定", subtitle: "上の接続先を使用します。ボタンを押した時だけカメラへ接続します")
                Button("カメラ設定を読み込む") { input.error = ""; store.controlCamera(apply: false) }
                if !store.cameraControlDraft.isEmpty {
                    ForEach(fields, id: \.0) { key, title in
                        HStack {
                            Text(title).frame(width: 240, alignment: .leading)
                            TextField(title, text: Binding(
                                get: { store.cameraControlDraft[key] ?? "" },
                                set: { store.cameraControlDraft[key] = $0 }
                            )).textFieldStyle(.roundedBorder)
                        }
                    }
                    Button("編集した値をカメラへ反映") { applyChanges() }
                }
                Text(input.error.isEmpty ? store.cameraControlStatus : input.error)
                    .font(.caption).foregroundStyle(AppTheme.secondaryText)
            }
            .disabled(store.cameraControlBusy)
        }
    }
    private func applyChanges() {
        var basic: [String: Int] = [:]
        var extra: [String: Int] = [:]
        for (key, title) in fields {
            let text = (store.cameraControlDraft[key] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            guard let value = Int(text) else { input.error = "\(title)は整数で入力してください"; return }
            guard value != Int(store.cameraControlOriginal[key] ?? "") else { continue }
            if key == "wdr_level" { extra[key] = value } else { basic[key] = value }
        }
        input.error = ""
        store.controlCamera(apply: true, basic: basic, extra: extra)
    }
}
