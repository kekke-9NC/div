import AppKit
import SwiftUI
import UniformTypeIdentifiers

@MainActor
final class CameraModelBuilderDraft: ObservableObject {
    @Published var sourcePath = ""
    @Published var autoSelect = true
    @Published var start = ""
    @Published var end = ""
    @Published var useCloudFilter = false
    @Published var cloudThreshold = 0.10
    @Published var isPresented = false
}

struct CameraModelBuilderView: View {
    @EnvironmentObject private var store: WorkspaceStore
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var draft: CameraModelBuilderDraft

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("高精度カメラ補正データを作成")
                        .font(.system(size: 20, weight: .bold, design: .rounded))
                        .foregroundStyle(AppTheme.text)
                    Text("星の写り方を複数の動画から検証し、再利用できる固定カメラモデルを作成します。")
                        .font(.system(size: 12))
                        .foregroundStyle(AppTheme.secondaryText)
                }
                Spacer()
                Image(systemName: "scope")
                    .font(.system(size: 24, weight: .medium))
                    .foregroundStyle(AppTheme.cyan)
                    .padding(12)
                    .background(AppTheme.accent.opacity(0.12), in: Circle())
            }

            GlassCard {
                VStack(alignment: .leading, spacing: 14) {
                    SectionTitle("対象データ", subtitle: "動画ファイル、または同じ撮影日の動画が入ったフォルダを選択")
                    HStack(spacing: 10) {
                        TextField("動画またはフォルダ", text: $draft.sourcePath)
                            .textFieldStyle(.roundedBorder)
                            .disabled(store.cameraModelBuildActive)
                        Button("選択") { chooseSource() }
                            .buttonStyle(.bordered)
                            .disabled(store.cameraModelBuildActive)
                    }
                    if draft.sourcePath.isEmpty {
                        Label("対象を選択してください", systemImage: "exclamationmark.triangle.fill")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(AppTheme.warning)
                    } else if !sourceExists {
                        Label("選択した対象が見つかりません", systemImage: "exclamationmark.triangle.fill")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(AppTheme.warning)
                    } else {
                        Label("対象を確認しました", systemImage: "checkmark.circle.fill")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(AppTheme.success)
                    }
                }
            }

            GlassCard {
                VStack(alignment: .leading, spacing: 14) {
                    SectionTitle("動画の選び方", subtitle: "自動選択は同じ撮影日の夜間動画から星の多い時間を選びます")
                    Toggle("撮影日の動画を自動選択する", isOn: $draft.autoSelect)
                        .toggleStyle(.switch)
                        .tint(AppTheme.accent)
                        .disabled(store.cameraModelBuildActive)
                        .onChange(of: draft.autoSelect) { _, enabled in
                            if enabled {
                                draft.start = ""
                                draft.end = ""
                            }
                        }
                    if draft.autoSelect {
                        Label("タイムラプスの範囲外も含め、同じ撮影日の動画を調べます。", systemImage: "wand.and.stars")
                            .font(.system(size: 11))
                            .foregroundStyle(AppTheme.secondaryText)
                    } else {
                        HStack(spacing: 12) {
                            TextField("開始 HH:MM", text: $draft.start)
                                .textFieldStyle(.roundedBorder)
                            TextField("終了 HH:MM", text: $draft.end)
                                .textFieldStyle(.roundedBorder)
                        }
                        .disabled(store.cameraModelBuildActive)
                    }
                }
            }

            GlassCard {
                VStack(alignment: .leading, spacing: 14) {
                    SectionTitle("雲量フィルタ", subtitle: "必要な場合だけローカルの画像判定サービスを使って雲の多い動画を除外")
                    Toggle("雲量フィルタを使用する", isOn: $draft.useCloudFilter)
                        .toggleStyle(.switch)
                        .tint(AppTheme.accent)
                        .disabled(store.cameraModelBuildActive)
                    HStack(spacing: 12) {
                        Text("許容雲量")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(AppTheme.secondaryText)
                        Slider(value: $draft.cloudThreshold, in: 0...1, step: 0.01)
                            .disabled(!draft.useCloudFilter || store.cameraModelBuildActive)
                        Text(String(format: "%.0f%%以下", draft.cloudThreshold * 100))
                            .font(.system(size: 12, weight: .semibold, design: .monospaced))
                            .foregroundStyle(draft.useCloudFilter ? AppTheme.text : AppTheme.tertiaryText)
                            .frame(width: 72, alignment: .trailing)
                    }
                }
            }

            GlassCard(padding: 16) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Label(
                            store.cameraModelBuildActive ? "作成中" : "作成ステータス",
                            systemImage: store.cameraModelBuildActive ? "waveform" : "info.circle"
                        )
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(store.cameraModelBuildActive ? AppTheme.accent : AppTheme.secondaryText)
                        Spacer()
                        if store.cameraModelBuildActive {
                            ProgressView()
                                .controlSize(.small)
                        }
                    }
                    Text(store.cameraModelBuildStatus)
                        .font(.system(size: 12))
                        .foregroundStyle(AppTheme.text)
                    if !store.cameraModelBuildProgress.isEmpty && store.cameraModelBuildActive {
                        Text(store.cameraModelBuildProgress)
                            .font(.system(size: 11))
                            .foregroundStyle(AppTheme.secondaryText)
                            .lineLimit(2)
                    }
                    if !store.cameraModelBuildModelPath.isEmpty {
                        Text("保存先: \(store.cameraModelBuildModelPath)")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(AppTheme.tertiaryText)
                            .lineLimit(1)
                        Text(
                            String(format: "被覆率 %.0f%% / p95 %.2fpx", store.cameraModelBuildSupportFraction * 100, store.cameraModelBuildResidualP95)
                        )
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(store.cameraModelBuildTargetMet ? AppTheme.success : AppTheme.warning)
                    }
                }
            }

            HStack {
                Spacer()
                Button("閉じる") { dismiss() }
                    .buttonStyle(.bordered)
                    .disabled(store.cameraModelBuildActive)
                if store.cameraModelBuildActive {
                    Button("停止", role: .destructive) {
                        store.cancelCameraModelBuild()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(AppTheme.danger)
                } else {
                    Button {
                        store.startCameraModelBuild(
                            source: draft.sourcePath,
                            autoSelect: draft.autoSelect,
                            start: draft.start,
                            end: draft.end,
                            cloudThreshold: draft.cloudThreshold,
                            useCloudFilter: draft.useCloudFilter
                        )
                    } label: {
                        Label("作成を開始", systemImage: "wand.and.stars")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(AppTheme.accent)
                    .disabled(!sourceExists)
                }
            }
        }
        .padding(24)
        .frame(minWidth: 720, minHeight: 720)
        .background(AppTheme.canvas)
        .interactiveDismissDisabled(store.cameraModelBuildActive)
        .onAppear {
            if draft.sourcePath.isEmpty {
                draft.sourcePath = store.sources.first(where: { $0.kind != .rtsp })?.value ?? ""
            }
        }
    }

    private var sourceExists: Bool {
        guard !draft.sourcePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return false }
        return FileManager.default.fileExists(atPath: draft.sourcePath)
    }

    private func chooseSource() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.resolvesAliases = true
        panel.prompt = "選択"
        if panel.runModal() == .OK, let url = panel.url {
            draft.sourcePath = url.standardizedFileURL.path
        }
    }
}
