import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct ToolsView: View {
    @EnvironmentObject private var store: WorkspaceStore
    @StateObject private var viewState = ToolsViewState()

    private let bitrateOptions = ["Auto", "1000k", "2000k", "4000k", "8000k", "12000k", "16000k", "20000k"]
    private let fpsOptions = ["Auto", "15", "24", "25", "30", "60"]
    private let videoExtensions = ["mp4", "mov", "avi", "mkv", "wmv", "flv", "webm", "m4v", "ts", "mts", "m2ts"]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                PageHeader(
                    eyebrow: "Tools",
                    title: "動画ツール",
                    subtitle: "動画の順番を整え、安定した一本のファイルへ連結します。処理中も進捗と停止状態を確認できます。"
                )

                GlassCard(padding: 24) {
                    VStack(alignment: .leading, spacing: 16) {
                        HStack(alignment: .top) {
                            SectionTitle("動画を選ぶ", subtitle: "2本以上の動画を、Finderから順番どおりに追加します")
                            Spacer()
                            HStack(spacing: 8) {
                                Button {
                                    chooseVideoFiles()
                                } label: {
                                    Label("ファイルを追加", systemImage: "plus")
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(AppTheme.accent)
                                .disabled(store.videoConcatActive)
                                Button("すべて削除", role: .destructive) {
                                    store.clearVideoConcatFiles()
                                }
                                .buttonStyle(.bordered)
                                .disabled(store.videoConcatFiles.isEmpty || store.videoConcatActive)
                            }
                        }

                        Button {
                            chooseVideoFiles()
                        } label: {
                            ZStack {
                                RoundedRectangle(cornerRadius: 16, style: .continuous)
                                    .fill(viewState.isDropTargeted ? AppTheme.accent.opacity(0.18) : AppTheme.surfaceRaised.opacity(0.55))
                                    .overlay {
                                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                                            .stroke(
                                                viewState.isDropTargeted ? AppTheme.accent : AppTheme.border,
                                                style: StrokeStyle(lineWidth: 1.5, dash: [7, 6])
                                            )
                                    }
                                VStack(spacing: 8) {
                                    Image(systemName: "film.stack")
                                        .font(.system(size: 26, weight: .medium))
                                        .foregroundStyle(AppTheme.accent)
                                    Text(viewState.isDropTargeted ? "ここにドロップ" : "動画をここへドラッグ＆ドロップ")
                                        .font(.system(size: 15, weight: .semibold, design: .rounded))
                                        .foregroundStyle(AppTheme.text)
                                    Text("MP4 / MOV / AVI / MKV など")
                                        .font(.system(size: 12))
                                        .foregroundStyle(AppTheme.secondaryText)
                                }
                            }
                        }
                        .buttonStyle(.plain)
                        .frame(height: 126)
                        .onDrop(of: [.fileURL], isTargeted: $viewState.isDropTargeted) { providers in
                            acceptDrop(providers)
                        }
                        .accessibilityLabel("連結する動画を追加")

                        if store.videoConcatFiles.isEmpty {
                            EmptyState(
                                symbol: "film",
                                title: "動画がまだありません",
                                message: "ファイルを追加すると、ここで順番を確認できます"
                            )
                        } else {
                            VStack(spacing: 8) {
                                ForEach(Array(store.videoConcatFiles.enumerated()), id: \.element) { index, path in
                                    VideoConcatFileRow(
                                        index: index,
                                        path: path,
                                        canMoveUp: index > 0 && !store.videoConcatActive,
                                        canMoveDown: index + 1 < store.videoConcatFiles.count && !store.videoConcatActive,
                                        moveUp: { store.moveVideoConcatFile(from: index, by: -1) },
                                        moveDown: { store.moveVideoConcatFile(from: index, by: 1) },
                                        remove: { store.removeVideoConcatFile(at: index) },
                                        reveal: {
                                            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                                        }
                                    )
                                }
                            }
                        }
                    }
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: 16) {
                        SectionTitle("連結設定", subtitle: "安定性を優先した初期値です。必要なときだけ変更してください")

                        HStack(spacing: 18) {
                            Picker("ビットレート", selection: $store.videoConcatBitrate) {
                                ForEach(bitrateOptions, id: \.self) { option in
                                    Text(option).tag(option)
                                }
                            }
                            .pickerStyle(.menu)
                            .frame(width: 180, alignment: .leading)

                            Picker("コーデック", selection: $store.videoConcatCodec) {
                                Text("H.264").tag("h264")
                                Text("H.265 / HEVC").tag("h265")
                            }
                            .pickerStyle(.menu)
                            .frame(width: 180, alignment: .leading)

                            Picker("FPS", selection: $store.videoConcatFPS) {
                                ForEach(fpsOptions, id: \.self) { option in
                                    Text(option).tag(option)
                                }
                            }
                            .pickerStyle(.menu)
                            .frame(width: 140, alignment: .leading)
                        }
                        .disabled(store.videoConcatActive)
                        .onChange(of: store.videoConcatBitrate) { _, _ in store.saveSettings() }
                        .onChange(of: store.videoConcatCodec) { _, _ in store.saveSettings() }
                        .onChange(of: store.videoConcatFPS) { _, _ in store.saveSettings() }

                        Divider().overlay(AppTheme.border)

                        Toggle("セーフモード（タイムスタンプを補正して安定性を優先）", isOn: $store.videoConcatSafeMode)
                            .toggleStyle(.switch)
                            .tint(AppTheme.accent)
                            .disabled(store.videoConcatActive)
                        Toggle("固定パターン補正＋21フレーム平均を適用", isOn: $store.videoConcatEnhancement)
                            .toggleStyle(.switch)
                            .tint(AppTheme.accent)
                            .disabled(store.videoConcatActive)
                        if store.videoConcatEnhancement && !store.videoConcatEnhancementConfigurationIsValid {
                            Label("設定画面で固定パターン補正を有効にし、使用可能なマップを選択すると利用できます。", systemImage: "exclamationmark.triangle.fill")
                                .font(.system(size: 11, weight: .medium))
                                .foregroundStyle(AppTheme.warning)
                        } else if store.videoConcatEnhancement {
                            Text("使用する補正マップ: \(URL(fileURLWithPath: store.rtspFixedPatternPath).lastPathComponent)")
                                .font(.system(size: 11))
                                .foregroundStyle(AppTheme.tertiaryText)
                        }

                        Divider().overlay(AppTheme.border)

                        HStack(alignment: .center, spacing: 18) {
                            Toggle("実時刻を動画に表示", isOn: $store.videoConcatTimestampEnabled)
                                .toggleStyle(.switch)
                                .tint(AppTheme.accent)
                            Picker("位置", selection: $store.videoConcatTimestampPosition) {
                                Text("右下").tag("右下")
                                Text("左下").tag("左下")
                                Text("右上").tag("右上")
                                Text("左上").tag("左上")
                            }
                            .pickerStyle(.menu)
                            .frame(width: 100, alignment: .leading)
                            .disabled(store.videoConcatActive || !store.videoConcatTimestampEnabled)
                            Stepper(value: $store.videoConcatTimestampSizePercent, in: 0.8...4.0, step: 0.1) {
                                Text(String(format: "文字 %.1f%%", store.videoConcatTimestampSizePercent))
                                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                                    .foregroundStyle(AppTheme.secondaryText)
                            }
                            .disabled(store.videoConcatActive || !store.videoConcatTimestampEnabled)
                            TextField("補正秒", value: $store.videoConcatTimestampOffsetSeconds, format: .number.precision(.fractionLength(1)))
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 100)
                                .help("実時刻を前後へずらす秒数")
                                .disabled(store.videoConcatActive || !store.videoConcatTimestampEnabled)
                            Text("秒")
                                .font(.system(size: 11))
                                .foregroundStyle(AppTheme.tertiaryText)
                        }
                        .disabled(store.videoConcatActive)
                        .onChange(of: store.videoConcatSafeMode) { _, _ in store.saveSettings() }
                        .onChange(of: store.videoConcatEnhancement) { _, _ in store.saveSettings() }
                        .onChange(of: store.videoConcatTimestampEnabled) { _, _ in store.saveSettings() }
                        .onChange(of: store.videoConcatTimestampPosition) { _, _ in store.saveSettings() }
                        .onChange(of: store.videoConcatTimestampSizePercent) { _, _ in store.saveSettings() }
                        .onChange(of: store.videoConcatTimestampOffsetSeconds) { _, _ in store.saveSettings() }
                    }
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: 16) {
                        SectionTitle("保存先", subtitle: "元の動画は変更せず、新しい動画として保存します")
                        HStack(spacing: 10) {
                            TextField("連結後の動画ファイル", text: $store.videoConcatOutputPath)
                                .textFieldStyle(.roundedBorder)
                                .disabled(store.videoConcatActive)
                            Button("選択") {
                                chooseOutputFile()
                            }
                            .buttonStyle(.bordered)
                            .disabled(store.videoConcatActive)
                            if !store.videoConcatOutputPath.isEmpty {
                                Button("開く") {
                                    NSWorkspace.shared.open(URL(fileURLWithPath: store.videoConcatOutputPath))
                                }
                                .buttonStyle(.bordered)
                                .disabled(!FileManager.default.fileExists(atPath: store.videoConcatOutputPath))
                            }
                        }
                    }
                }

                GlassCard(padding: 22) {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack(alignment: .top) {
                            VStack(alignment: .leading, spacing: 5) {
                                Text("実行状態")
                                    .font(.system(size: 14, weight: .semibold, design: .rounded))
                                    .foregroundStyle(AppTheme.text)
                                Text(store.videoConcatReadinessMessage)
                                    .font(.system(size: 12))
                                    .foregroundStyle(statusColor)
                                    .lineLimit(2)
                            }
                            Spacer()
                            if store.videoConcatActive {
                                Button("停止", role: .destructive) {
                                    store.cancelVideoConcat()
                                }
                                .buttonStyle(.bordered)
                            } else {
                                Button {
                                    store.startVideoConcat()
                                } label: {
                                    Label("連結を開始", systemImage: "play.fill")
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(AppTheme.accent)
                                .disabled(!store.videoConcatCanStart)
                            }
                        }

                        if store.videoConcatActive || store.videoConcatProgress > 0 {
                            ProgressView(value: store.videoConcatProgress)
                                .tint(AppTheme.accent)
                            HStack {
                                Text(store.videoConcatProgressMessage.isEmpty ? store.videoConcatStatus : store.videoConcatProgressMessage)
                                    .font(.system(size: 11))
                                    .foregroundStyle(AppTheme.secondaryText)
                                    .lineLimit(2)
                                Spacer()
                                Text(String(format: "%.0f%%", store.videoConcatProgress * 100))
                                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                                    .foregroundStyle(AppTheme.text)
                            }
                        }

                        if !store.videoConcatLastResult.isEmpty {
                            Text(store.videoConcatLastResult)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(AppTheme.tertiaryText)
                                .lineLimit(4)
                                .textSelection(.enabled)
                        }
                    }
                }
            }
            .padding(34)
            .frame(maxWidth: 1100, alignment: .leading)
        }
        .allowsHitTesting(store.settingsLoaded)
        .overlay {
            if !store.settingsLoaded {
                ProgressView("設定を読み込んでいます…")
                    .controlSize(.small)
                    .padding(18)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            }
        }
    }

    private var statusColor: Color {
        if store.videoConcatActive { return AppTheme.accent }
        return store.videoConcatCanStart ? AppTheme.success : AppTheme.warning
    }

    private func chooseVideoFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.resolvesAliases = true
        panel.allowedContentTypes = videoExtensions.compactMap { UTType(filenameExtension: $0) }
        panel.prompt = "追加"
        if panel.runModal() == .OK {
            store.addVideoConcatFiles(panel.urls)
        }
    }

    private func chooseOutputFile() {
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.allowedContentTypes = [UTType.mpeg4Movie]
        panel.nameFieldStringValue = URL(fileURLWithPath: store.videoConcatOutputPath).lastPathComponent.isEmpty
            ? "concatenated.mp4"
            : URL(fileURLWithPath: store.videoConcatOutputPath).lastPathComponent
        panel.prompt = "保存"
        if panel.runModal() == .OK, let url = panel.url {
            store.videoConcatOutputPath = url.path
            store.saveSettings()
        }
    }

    private func acceptDrop(_ providers: [NSItemProvider]) -> Bool {
        for provider in providers {
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                Task { @MainActor in
                    guard !url.hasDirectoryPath, videoExtensions.contains(url.pathExtension.lowercased()) else { return }
                    store.addVideoConcatFiles([url])
                }
            }
        }
        return true
    }
}

@MainActor
private final class ToolsViewState: ObservableObject {
    @Published var isDropTargeted = false
}

private struct VideoConcatFileRow: View {
    let index: Int
    let path: String
    let canMoveUp: Bool
    let canMoveDown: Bool
    let moveUp: () -> Void
    let moveDown: () -> Void
    let remove: () -> Void
    let reveal: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Text(String(format: "%02d", index + 1))
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundStyle(AppTheme.accent)
                .frame(width: 30, height: 28)
                .background(AppTheme.accent.opacity(0.13), in: RoundedRectangle(cornerRadius: 8, style: .continuous))

            Image(systemName: "film")
                .foregroundStyle(fileExists ? AppTheme.accent : AppTheme.danger)
            VStack(alignment: .leading, spacing: 2) {
                Text(URL(fileURLWithPath: path).lastPathComponent)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(AppTheme.text)
                    .lineLimit(1)
                Text(path)
                    .font(.system(size: 10))
                    .foregroundStyle(AppTheme.tertiaryText)
                    .lineLimit(1)
            }
            Spacer()
            if !fileExists {
                Text("見つかりません")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(AppTheme.danger)
            }
            Button(action: moveUp) {
                Image(systemName: "chevron.up")
            }
            .buttonStyle(.borderless)
            .disabled(!canMoveUp)
            Button(action: moveDown) {
                Image(systemName: "chevron.down")
            }
            .buttonStyle(.borderless)
            .disabled(!canMoveDown)
            Button(action: reveal) {
                Image(systemName: "folder")
            }
            .buttonStyle(.borderless)
            .disabled(!fileExists)
            Button(action: remove) {
                Image(systemName: "minus.circle")
            }
            .buttonStyle(.borderless)
            .foregroundStyle(AppTheme.danger)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(AppTheme.surfaceRaised.opacity(0.55), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private var fileExists: Bool {
        FileManager.default.fileExists(atPath: path)
    }
}
