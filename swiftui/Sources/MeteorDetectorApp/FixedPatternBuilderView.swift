import AppKit
import SwiftUI

enum FixedPatternSourceMode: String, CaseIterable, Identifiable {
    case url
    case video
    case directory

    var id: String { rawValue }

    var title: String {
        switch self {
        case .url: return "現在のRTSPカメラから"
        case .video: return "保存済み動画から"
        case .directory: return "録画フォルダから"
        }
    }

    var symbol: String {
        switch self {
        case .url: return "dot.radiowaves.left.and.right"
        case .video: return "film"
        case .directory: return "folder"
        }
    }
}

@MainActor
final class FixedPatternBuilderDraft: ObservableObject {
    @Published var mode: FixedPatternSourceMode = .directory
    @Published var sourcePath = ""
    @Published var isPresented = false
}

struct FixedPatternBuilderView: View {
    @EnvironmentObject private var store: WorkspaceStore
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var draft: FixedPatternBuilderDraft

    private var currentRTSPURL: String? {
        store.sources.first(where: { $0.kind == .rtsp })?.value
    }

    private var selectedSource: String {
        if draft.mode == .url {
            return currentRTSPURL ?? ""
        }
        return draft.sourcePath.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var canStart: Bool {
        !store.fixedPatternBuildActive && !selectedSource.isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Fixed Pattern")
                        .font(.system(size: 11, weight: .bold, design: .rounded))
                        .tracking(1.4)
                        .foregroundStyle(AppTheme.accent)
                    Text("固定パターン補正を作成")
                        .font(.system(size: 26, weight: .bold, design: .rounded))
                        .foregroundStyle(AppTheme.text)
                    Text("暗い映像の複数フレームから、センサー由来の細かなムラだけを抽出します。")
                        .font(.system(size: 13))
                        .foregroundStyle(AppTheme.secondaryText)
                }
                Spacer()
                Image(systemName: "wand.and.stars")
                    .font(.system(size: 24, weight: .semibold))
                    .foregroundStyle(AppTheme.cyan)
                    .padding(14)
                    .background(AppTheme.cyan.opacity(0.12), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            }

            GlassCard {
                VStack(alignment: .leading, spacing: 14) {
                    SectionTitle("入力を選択", subtitle: "既存の録画を使うか、キャップをしたRTSPカメラを約30秒撮影します")
                    Picker("作成方法", selection: $draft.mode) {
                        ForEach(FixedPatternSourceMode.allCases) { mode in
                            Label(mode.title, systemImage: mode.symbol).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(store.fixedPatternBuildActive)

                    if draft.mode == .url {
                        HStack(spacing: 10) {
                            Image(systemName: currentRTSPURL == nil ? "exclamationmark.triangle" : "dot.radiowaves.left.and.right")
                                .foregroundStyle(currentRTSPURL == nil ? AppTheme.warning : AppTheme.success)
                            if let currentRTSPURL {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text("使用するRTSPカメラ")
                                        .font(.system(size: 11, weight: .semibold))
                                        .foregroundStyle(AppTheme.secondaryText)
                                    Text(currentRTSPURL)
                                        .font(.system(size: 12, design: .monospaced))
                                        .foregroundStyle(AppTheme.text)
                                        .lineLimit(2)
                                }
                            } else {
                                Text("先に入力ソース画面からRTSPカメラを追加してください")
                                    .font(.system(size: 12, weight: .medium))
                                    .foregroundStyle(AppTheme.warning)
                            }
                        }
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(AppTheme.surfaceRaised, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                    } else {
                        HStack(spacing: 10) {
                            TextField(
                                draft.mode == .video ? "固定パターン用の動画" : "固定パターン用の録画フォルダ",
                                text: $draft.sourcePath
                            )
                            .textFieldStyle(.roundedBorder)
                            .disabled(store.fixedPatternBuildActive)
                            Button("選択") {
                                if draft.mode == .video {
                                    chooseVideo()
                                } else {
                                    chooseDirectory()
                                }
                            }
                            .buttonStyle(.bordered)
                            .disabled(store.fixedPatternBuildActive)
                        }
                    }

                    HStack(spacing: 20) {
                        Picker("サンプル数", selection: $store.rtspFixedPatternSamples) {
                            Text("90枚").tag(90)
                            Text("180枚").tag(180)
                            Text("360枚（推奨）").tag(360)
                        }
                        .pickerStyle(.menu)
                        .disabled(store.fixedPatternBuildActive)
                        Text("枚数が多いほど、変動する表示やノイズの影響を抑えられます。")
                            .font(.system(size: 11))
                            .foregroundStyle(AppTheme.tertiaryText)
                    }
                }
            }

            GlassCard {
                VStack(alignment: .leading, spacing: 12) {
                    Label(
                        store.fixedPatternBuildActive ? "作成中" : "作成ステータス",
                        systemImage: store.fixedPatternBuildActive ? "waveform" : "info.circle"
                    )
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(store.fixedPatternBuildActive ? AppTheme.accent : AppTheme.secondaryText)
                    Text(store.fixedPatternBuildStatus)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(AppTheme.text)
                    if !store.fixedPatternBuildProgress.isEmpty && store.fixedPatternBuildActive {
                        ProgressView()
                            .progressViewStyle(.linear)
                            .tint(AppTheme.accent)
                        Text(store.fixedPatternBuildProgress)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(AppTheme.secondaryText)
                    }
                    if !store.fixedPatternBuildPreviewPath.isEmpty {
                        HStack {
                            Text("プレビュー: \(store.fixedPatternBuildPreviewPath)")
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(AppTheme.tertiaryText)
                                .lineLimit(1)
                            Spacer()
                            Button("開く") {
                                NSWorkspace.shared.open(URL(fileURLWithPath: store.fixedPatternBuildPreviewPath))
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
            }

            HStack {
                Button("閉じる") { dismiss() }
                    .buttonStyle(.bordered)
                    .disabled(store.fixedPatternBuildActive)
                Spacer()
                if store.fixedPatternBuildActive {
                    Button("作成を停止", role: .destructive) {
                        store.cancelFixedPatternBuild()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(AppTheme.danger)
                } else {
                    Button {
                        store.startFixedPatternBuild(mode: draft.mode.rawValue, source: selectedSource)
                    } label: {
                        Label("補正マップを作成", systemImage: "sparkles")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(AppTheme.accent)
                    .disabled(!canStart)
                }
            }
        }
        .padding(28)
        .frame(minWidth: 720, minHeight: 560)
        .background(AppTheme.canvas)
        .preferredColorScheme(.dark)
        .interactiveDismissDisabled(store.fixedPatternBuildActive)
        .onAppear {
            if draft.mode == .directory && draft.sourcePath.isEmpty {
                let defaultDirectory = store.rootURL.appendingPathComponent("rtsp").path
                if FileManager.default.fileExists(atPath: defaultDirectory) {
                    draft.sourcePath = defaultDirectory
                }
            }
        }
    }

    private func chooseVideo() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.prompt = "選択"
        if panel.runModal() == .OK, let url = panel.url {
            draft.sourcePath = url.path
        }
    }

    private func chooseDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "選択"
        if panel.runModal() == .OK, let url = panel.url {
            draft.sourcePath = url.path
        }
    }
}
