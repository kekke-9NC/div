import AppKit
import MeteorDetectorCore
import SwiftUI
import UniformTypeIdentifiers

struct RootView: View {
    @EnvironmentObject private var store: WorkspaceStore

    var body: some View {
        NavigationSplitView {
            SidebarView()
        } detail: {
            ZStack {
                AppTheme.canvas.ignoresSafeArea()
                detailView
            }
        }
        .navigationSplitViewStyle(.balanced)
        .frame(minWidth: 1080, minHeight: 720)
        .preferredColorScheme(.dark)
    }

    @ViewBuilder
    private var detailView: some View {
        switch store.selection {
        case .overview: DashboardView()
        case .capture: SourceView()
        case .analysis: AnalysisView()
        case .tools: ToolsView()
        case .results: ResultsView()
        case .settings: SettingsView()
        case .activity: ActivityView()
        }
    }
}

struct SidebarView: View {
    @EnvironmentObject private var store: WorkspaceStore

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(AppTheme.accent.opacity(0.18))
                    Image(systemName: "sparkles")
                        .font(.system(size: 20, weight: .bold))
                        .foregroundStyle(AppTheme.cyan)
                }
                .frame(width: 42, height: 42)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Meteor Detector")
                        .font(.system(size: 15, weight: .bold, design: .rounded))
                        .foregroundStyle(AppTheme.text)
                    Text("観測ワークスペース")
                        .font(.system(size: 11))
                        .foregroundStyle(AppTheme.secondaryText)
                }
            }
            .padding(.horizontal, 18)
            .padding(.top, 22)
            .padding(.bottom, 20)

            List(selection: $store.selection) {
                Section("ワークスペース") {
                    ForEach(AppSection.allCases) { section in
                        Label {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(section.title)
                                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                                Text(section.subtitle)
                                    .font(.system(size: 10))
                                    .foregroundStyle(AppTheme.tertiaryText)
                            }
                        } icon: {
                            Image(systemName: section.symbol)
                                .frame(width: 20)
                                .foregroundStyle(section == store.selection ? AppTheme.accent : AppTheme.secondaryText)
                        }
                        .tag(section)
                        .padding(.vertical, 5)
                    }
                }
            }
            .listStyle(.sidebar)
            .scrollContentBackground(.hidden)

            VStack(alignment: .leading, spacing: 10) {
                Divider().overlay(AppTheme.border)
                HStack(spacing: 8) {
                    Circle()
                        .fill(connectionColor)
                        .frame(width: 8, height: 8)
                    Text(store.connection.title)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(AppTheme.secondaryText)
                        .lineLimit(1)
                }
                Text("SwiftUI フロントエンド")
                    .font(.system(size: 10, weight: .medium, design: .rounded))
                    .foregroundStyle(AppTheme.tertiaryText)
            }
            .padding(.horizontal, 18)
            .padding(.bottom, 18)
        }
        .frame(minWidth: 238, idealWidth: 252)
        .background(AppTheme.surface.opacity(0.98))
    }

    private var connectionColor: Color {
        switch store.connection {
        case .connected: return AppTheme.success
        case .starting: return AppTheme.warning
        case .unavailable: return AppTheme.danger
        }
    }
}

struct DashboardView: View {
    @EnvironmentObject private var store: WorkspaceStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                PageHeader(
                    eyebrow: "Overview",
                    title: "観測ワークスペース",
                    subtitle: "入力を選び、検出を開始し、結果をひとつの流れで確認できます。"
                )

                GlassCard(padding: 26) {
                    HStack(alignment: .top, spacing: 24) {
                        VStack(alignment: .leading, spacing: 14) {
                            StatusPill(
                                title: store.runState.title,
                                color: runStateColor,
                                symbol: store.runState.isActive ? "waveform" : "circle.fill"
                            )
                            Text(heroTitle)
                                .font(.system(size: 26, weight: .bold, design: .rounded))
                                .foregroundStyle(AppTheme.text)
                            Text(store.readinessMessage)
                                .font(.system(size: 14))
                                .foregroundStyle(AppTheme.secondaryText)
                            HStack(spacing: 10) {
                                Button {
                                    store.selection = .capture
                                } label: {
                                    Label("入力を追加", systemImage: "plus")
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(AppTheme.accent)

                                Button {
                                    store.selection = .analysis
                                } label: {
                                    Label("解析画面を開く", systemImage: "scope")
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                        Spacer(minLength: 20)
                        Image(systemName: "moon.stars.fill")
                            .font(.system(size: 64, weight: .medium))
                            .foregroundStyle(AppTheme.cyan, AppTheme.accent.opacity(0.35))
                            .symbolRenderingMode(.hierarchical)
                            .padding(18)
                            .background(AppTheme.accent.opacity(0.10), in: Circle())
                    }
                }

                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
                    MetricCard(symbol: "tray.full.fill", title: "入力ソース", value: "\(store.sources.count)", detail: "動画・フォルダ・RTSP")
                    MetricCard(symbol: "checkmark.seal.fill", title: "準備状態", value: readinessShort, detail: store.connection.title)
                    MetricCard(symbol: "photo.on.rectangle.angled", title: "検出結果", value: "\(store.results.count)", detail: "候補ファイル")
                    MetricCard(symbol: "folder.fill", title: "保存先", value: outputFolderName, detail: "検出結果")
                }

                HStack(alignment: .top, spacing: 16) {
                    GlassCard {
                        VStack(alignment: .leading, spacing: 16) {
                            SectionTitle("次のステップ", subtitle: "迷ったときはこの順番で進めます")
                            WorkflowRow(number: "01", title: "入力ソースを追加", detail: "動画ファイル、フォルダ、またはRTSPカメラを選択")
                            WorkflowRow(number: "02", title: "設定を確認", detail: "保存先と処理条件を必要に応じて調整")
                            WorkflowRow(number: "03", title: "検出を開始", detail: "進捗とログはアクティビティで確認")
                        }
                    }
                    GlassCard {
                        VStack(alignment: .leading, spacing: 14) {
                            SectionTitle("最近のイベント", subtitle: "最新の状態")
                            if store.logs.isEmpty {
                                Text("イベントはまだありません")
                                    .font(.system(size: 13))
                                    .foregroundStyle(AppTheme.secondaryText)
                            } else {
                                ForEach(Array(store.logs.suffix(3).reversed())) { entry in
                                    CompactLogRow(entry: entry)
                                }
                            }
                            Button("すべてのイベントを見る") {
                                store.selection = .activity
                            }
                            .buttonStyle(.link)
                            .foregroundStyle(AppTheme.accent)
                        }
                    }
                }
            }
            .padding(34)
            .frame(maxWidth: 1100, alignment: .leading)
        }
    }

    private var heroTitle: String {
        switch store.runState {
        case .running: return "解析が進行中です"
        case .preparing: return "入力を準備しています"
        case .cancelling: return "停止処理を実行しています"
        case .completed: return "解析が完了しました"
        case .cancelled: return "解析を停止しました"
        case .failed: return "解析を開始できませんでした"
        case .idle:
            if store.periodicScanEnabled && store.canStart { return "定期スキャンを開始できます" }
            return store.sources.isEmpty ? "夜空の変化を見つける準備をしましょう" : "観測データの準備ができています"
        }
    }

    private var runStateColor: Color {
        switch store.runState {
        case .completed: return AppTheme.success
        case .failed: return AppTheme.danger
        case .cancelled: return AppTheme.warning
        case .preparing, .running, .cancelling: return AppTheme.accent
        case .idle: return AppTheme.secondaryText
        }
    }

    private var readinessShort: String {
        if store.connection != .connected || !store.settingsLoaded { return "接続中" }
        if store.isBusy { return "実行中" }
        return store.canStart ? "準備完了" : "要確認"
    }

    private var outputFolderName: String {
        URL(fileURLWithPath: store.meteorSavePath).lastPathComponent
    }
}

struct MetricCard: View {
    let symbol: String
    let title: String
    let value: String
    let detail: String

    var body: some View {
        GlassCard(padding: 18) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Image(systemName: symbol)
                        .foregroundStyle(AppTheme.accent)
                    Spacer()
                    Text(title)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(AppTheme.secondaryText)
                }
                Text(value)
                    .font(.system(size: 25, weight: .bold, design: .rounded))
                    .foregroundStyle(AppTheme.text)
                Text(detail)
                    .font(.system(size: 11))
                    .foregroundStyle(AppTheme.tertiaryText)
                    .lineLimit(1)
            }
        }
    }
}

struct WorkflowRow: View {
    let number: String
    let title: String
    let detail: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Text(number)
                .font(.system(size: 11, weight: .bold, design: .rounded))
                .foregroundStyle(AppTheme.accent)
                .frame(width: 28, height: 28)
                .background(AppTheme.accent.opacity(0.14), in: Circle())
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(AppTheme.text)
                Text(detail)
                    .font(.system(size: 12))
                    .foregroundStyle(AppTheme.secondaryText)
            }
        }
    }
}

struct SourceView: View {
    @EnvironmentObject private var store: WorkspaceStore
    @StateObject private var draft = RTSPDraft()
    @StateObject private var viewState = SourceViewState()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                PageHeader(
                    eyebrow: "Capture",
                    title: "入力ソース",
                    subtitle: "処理したい観測データを追加します。複数の動画やフォルダをまとめて選べます。"
                )

                GlassCard(padding: 24) {
                    VStack(alignment: .leading, spacing: 16) {
                        SectionTitle("動画・フォルダ", subtitle: "Finderからこの領域へドラッグ＆ドロップもできます")
                        HStack(spacing: 12) {
                            Button {
                                openSourcePanel()
                            } label: {
                                Label("動画 / フォルダを追加", systemImage: "plus")
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(AppTheme.accent)
                            Button("入力をすべて消去", role: .destructive) {
                                viewState.showClearConfirmation = true
                            }
                            .buttonStyle(.bordered)
                            .disabled(store.sources.isEmpty)
                            Spacer()
                            Text("対応形式: MP4 / AVI / MOV")
                                .font(.system(size: 11))
                                .foregroundStyle(AppTheme.tertiaryText)
                        }
                        Button(action: openSourcePanel) {
                            ZStack {
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .fill(store.isDropTargeted ? AppTheme.accent.opacity(0.18) : AppTheme.surfaceRaised.opacity(0.55))
                                .overlay {
                                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                                        .stroke(store.isDropTargeted ? AppTheme.accent : AppTheme.border, style: StrokeStyle(lineWidth: 1.5, dash: [7, 6]))
                                }
                            VStack(spacing: 8) {
                                Image(systemName: "arrow.down.doc.fill")
                                    .font(.system(size: 27, weight: .medium))
                                    .foregroundStyle(AppTheme.accent)
                                Text(store.isDropTargeted ? "ここにドロップ" : "動画やフォルダをここへ")
                                    .font(.system(size: 15, weight: .semibold, design: .rounded))
                                    .foregroundStyle(AppTheme.text)
                                Text("クリックしてFinderから選ぶこともできます")
                                    .font(.system(size: 12))
                                    .foregroundStyle(AppTheme.secondaryText)
                            }
                            }
                        }
                        .buttonStyle(.plain)
                        .frame(height: 132)
                        .onDrop(of: [.fileURL], isTargeted: $store.isDropTargeted) { providers in
                            acceptDrop(providers)
                        }
                        .accessibilityLabel("動画やフォルダを追加")
                        .accessibilityHint("クリックしてFinderを開くか、ファイルをここへドラッグします")
                    }
                }

                GlassCard(padding: 20) {
                    VStack(alignment: .leading, spacing: 14) {
                        SectionTitle("RTSPカメラ", subtitle: "ネットワークカメラは動画入力とは別に実行します")
                        HStack(spacing: 10) {
                            TextField("rtsp://ユーザー名:パスワード@ホスト/…", text: $draft.url)
                                .textFieldStyle(.roundedBorder)
                                .onSubmit { addRTSP() }
                            Button("追加") { addRTSP() }
                                .buttonStyle(.borderedProminent)
                                .tint(AppTheme.accent)
                        }
                        ForEach(store.sources.filter { $0.kind == .rtsp }) { source in
                            SourceRow(source: source) { store.removeSource(source) }
                        }
                    }
                }

                GlassCard(padding: 20) {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack {
                            SectionTitle("追加済みの入力", subtitle: "\(store.sources.count)件")
                            Spacer()
                            if store.activeSourceTypeCount > 1 {
                                StatusPill(title: "優先: \(store.selectedSourceLabel)", color: AppTheme.accent, symbol: "arrow.up.circle.fill")
                            }
                        }
                        let localSources = store.sources.filter { $0.kind != .rtsp }
                        if localSources.isEmpty {
                            EmptyState(symbol: "tray", title: "動画入力はまだありません", message: "動画またはフォルダを追加すると、ここに表示されます")
                        } else {
                            ForEach(localSources) { source in
                                SourceRow(source: source) { store.removeSource(source) }
                            }
                        }
                    }
                }
            }
            .padding(34)
            .frame(maxWidth: 1100, alignment: .leading)
        }
        .confirmationDialog(
            "入力ソースをすべて削除しますか？",
            isPresented: $viewState.showClearConfirmation,
            titleVisibility: .visible
        ) {
            Button("すべて削除", role: .destructive) { store.clearSources() }
            Button("キャンセル", role: .cancel) {}
        } message: {
            Text("登録済みの動画・フォルダ・RTSPカメラが一覧から削除されます。")
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

    private func openSourcePanel() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.resolvesAliases = true
        panel.prompt = "追加"
        if panel.runModal() == .OK {
            store.addSources(panel.urls)
        }
    }

    private func acceptDrop(_ providers: [NSItemProvider]) -> Bool {
        for provider in providers {
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                Task { @MainActor in
                    store.addSources([url])
                }
            }
        }
        return true
    }

    private func addRTSP() {
        if store.addRTSP(draft.url) { draft.url = "" }
    }
}

@MainActor
final class RTSPDraft: ObservableObject {
    @Published var url = ""
}

@MainActor
final class SourceViewState: ObservableObject {
    @Published var showClearConfirmation = false
}

struct SourceRow: View {
    let source: InputSource
    let remove: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: source.kind.symbol)
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(source.exists ? AppTheme.accent : AppTheme.danger)
                .frame(width: 30, height: 30)
                .background((source.exists ? AppTheme.accent : AppTheme.danger).opacity(0.13), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 8) {
                    Text(source.displayName)
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundStyle(AppTheme.text)
                        .lineLimit(1)
                    Text(source.kind.title)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(AppTheme.secondaryText)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(AppTheme.surfaceRaised, in: Capsule())
                }
                Text(source.detail)
                    .font(.system(size: 11))
                    .foregroundStyle(source.exists ? AppTheme.tertiaryText : AppTheme.danger)
                    .lineLimit(1)
            }
            Spacer()
            if !source.exists {
                Text("見つかりません")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(AppTheme.danger)
            }
            Button(action: remove) {
                Image(systemName: "trash")
                    .foregroundStyle(AppTheme.secondaryText)
            }
            .buttonStyle(.borderless)
            .help("この入力を削除")
            .accessibilityLabel("\(source.displayName)を削除")
        }
        .padding(.vertical, 5)
    }
}

struct AnalysisView: View {
    @EnvironmentObject private var store: WorkspaceStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                PageHeader(
                    eyebrow: "Analyze",
                    title: "検出と解析",
                    subtitle: "解析を開始し、進捗・キュー・結果をひとつの画面で確認します。"
                )

                if !store.unsupportedFeatures.isEmpty {
                    GlassCard(padding: 18) {
                        VStack(alignment: .leading, spacing: 10) {
                            Label("旧UIの設定に未対応項目があります", systemImage: "exclamationmark.triangle.fill")
                                .font(.system(size: 14, weight: .semibold, design: .rounded))
                                .foregroundStyle(AppTheme.warning)
                            Text(store.unsupportedFeatures.joined(separator: "、") + " はSwiftUI版のこの段階では適用されません。内容を変えずに実行する場合は、旧UIを使用してください。")
                                .font(.system(size: 12))
                                .foregroundStyle(AppTheme.secondaryText)
                            Toggle("未対応項目を無効にした基本解析を許可する", isOn: $store.allowReducedFeatureRun)
                                .toggleStyle(.switch)
                                .tint(AppTheme.warning)
                        }
                    }
                }

                GlassCard(padding: 24) {
                    VStack(alignment: .leading, spacing: 20) {
                        HStack {
                            SectionTitle("処理ステータス", subtitle: store.queueStatus)
                            Spacer()
                            StatusPill(title: store.runState.title, color: runStateColor, symbol: store.runState.isActive ? "waveform" : "circle.fill")
                        }
                        if let progress = store.progress {
                            VStack(alignment: .leading, spacing: 8) {
                                HStack {
                                    Text(progress.message.isEmpty ? "処理中" : progress.message)
                                        .font(.system(size: 13, weight: .medium))
                                        .foregroundStyle(AppTheme.text)
                                    Spacer()
                                    Text("\(progress.current) / \(progress.total)")
                                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                                        .foregroundStyle(AppTheme.secondaryText)
                                }
                                ProgressView(value: progress.fraction)
                                    .tint(AppTheme.accent)
                            }
                        } else if store.runState.isActive {
                            ProgressView()
                                .controlSize(.small)
                                .tint(AppTheme.accent)
                        } else {
                            Text(store.readinessMessage)
                                .font(.system(size: 13))
                                .foregroundStyle(AppTheme.secondaryText)
                        }
                        HStack(spacing: 10) {
                            if store.isBusy {
                                Button(role: .destructive) {
                                    store.cancelDetection()
                                } label: {
                                    Label("処理を停止", systemImage: "stop.fill")
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(AppTheme.danger)
                                .disabled(store.runState == .cancelling)
                            } else {
                                Button {
                                    store.startDetection()
                                } label: {
                                    Label("解析を開始", systemImage: "play.fill")
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(AppTheme.accent)
                                .disabled(!store.canStart)
                            }
                            if store.runState == .completed {
                                Button {
                                    store.refreshResults()
                                    store.selection = .results
                                } label: {
                                    Label("結果を見る", systemImage: "photo.on.rectangle")
                                }
                                .buttonStyle(.bordered)
                            }
                            Button("入力を見直す") { store.selection = .capture }
                                .buttonStyle(.bordered)
                        }
                    }
                }

                HStack(alignment: .top, spacing: 16) {
                    GlassCard {
                        VStack(alignment: .leading, spacing: 14) {
                            SectionTitle("今回の入力", subtitle: "解析対象")
                            if store.sources.isEmpty {
                                EmptyState(symbol: "tray", title: "入力がありません", message: "入力ソース画面から追加してください")
                            } else {
                                ForEach(store.sources) { source in
                                    HStack(spacing: 9) {
                                        Image(systemName: source.kind.symbol)
                                            .foregroundStyle(source.exists ? AppTheme.accent : AppTheme.danger)
                                        Text(source.displayName)
                                            .font(.system(size: 12, weight: .medium))
                                            .foregroundStyle(AppTheme.text)
                                            .lineLimit(1)
                                        Spacer()
                                    }
                                }
                            }
                        }
                    }
                    GlassCard {
                        VStack(alignment: .leading, spacing: 14) {
                            SectionTitle("処理条件", subtitle: "現在の設定")
                            KeyValueRow(key: "並列数", value: "\(store.maxWorkers) ワーカー")
                            KeyValueRow(key: "間隔", value: String(format: "%.2f 秒", store.interval))
                            KeyValueRow(key: "検出クリップ", value: String(format: "%.2f 秒", store.duration))
                            KeyValueRow(key: "保存先", value: URL(fileURLWithPath: store.meteorSavePath).lastPathComponent)
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

    private var runStateColor: Color {
        switch store.runState {
        case .completed: return AppTheme.success
        case .failed: return AppTheme.danger
        case .cancelled: return AppTheme.warning
        case .preparing, .running, .cancelling: return AppTheme.accent
        case .idle: return AppTheme.secondaryText
        }
    }
}

struct ResultsView: View {
    @EnvironmentObject private var store: WorkspaceStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
            PageHeader(
                eyebrow: "Results",
                title: "検出結果",
                subtitle: "候補ファイルを一覧で確認し、画像はこの画面でプレビューできます。"
            )
            .padding(34)

            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: 16) {
                    resultListPanel
                        .frame(minWidth: 390, idealWidth: 520, maxWidth: 620)
                    resultPreviewPanel
                        .frame(minWidth: 320, maxWidth: .infinity, minHeight: 470)
                }
                VStack(spacing: 16) {
                    resultListPanel
                        .frame(minHeight: 300)
                    resultPreviewPanel
                        .frame(minHeight: 380)
                }
            }
            .padding(.horizontal, 34)
            .padding(.bottom, 34)
            }
        }
        .onAppear { store.refreshResults() }
        .onChange(of: store.resultFilter) { _, _ in
            store.selectedResultID = nil
        }
    }

    private var resultListPanel: some View {
        GlassCard(padding: 16) {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Picker("結果の種類", selection: $store.resultFilter) {
                        ForEach(OutputCategory.allCases) { category in
                            Label(category.title, systemImage: category.symbol).tag(category)
                        }
                    }
                    .pickerStyle(.segmented)
                    Spacer()
                    Button {
                        store.refreshResults()
                    } label: {
                        if store.isRefreshingResults {
                            ProgressView()
                                .controlSize(.small)
                        } else {
                            Image(systemName: "arrow.clockwise")
                        }
                    }
                    .buttonStyle(.borderless)
                    .help("結果を再読み込み")
                    .accessibilityLabel("結果を再読み込み")
                    .disabled(store.isRefreshingResults)
                }

                HStack {
                    Text("\(store.filteredResults.count)件")
                        .font(.system(size: 12, weight: .semibold, design: .rounded))
                        .foregroundStyle(AppTheme.secondaryText)
                    Spacer()
                    Button("保存先を開く") {
                        store.openOutputFolder(store.resultFilter == .meteor ? store.meteorSavePath : store.notMeteorSavePath)
                    }
                    .buttonStyle(.link)
                    .foregroundStyle(AppTheme.accent)
                }

                Divider().overlay(AppTheme.border)

                if store.isRefreshingResults && store.filteredResults.isEmpty {
                    ProgressView("結果を読み込んでいます…")
                        .frame(maxWidth: .infinity, minHeight: 160)
                } else if store.filteredResults.isEmpty {
                    EmptyState(
                        symbol: store.resultFilter.symbol,
                        title: "まだ結果がありません",
                        message: "解析が完了すると、候補ファイルがここに表示されます。"
                    )
                } else {
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 4) {
                            ForEach(store.filteredResults) { item in
                                ResultRow(
                                    item: item,
                                    isSelected: item.id == store.selectedResultID
                                ) {
                                    store.selectedResultID = item.id
                                }
                            }
                        }
                    }
                    .frame(minHeight: 300)
                }
            }
        }
    }

    private var resultPreviewPanel: some View {
        GlassCard(padding: 20) {
            ResultPreviewPanel(item: store.results.first { $0.id == store.selectedResultID })
        }
    }
}

struct ResultRow: View {
    let item: OutputItem
    let isSelected: Bool
    let select: () -> Void

    var body: some View {
        Button(action: select) {
            HStack(spacing: 10) {
                Image(systemName: item.kind.symbol)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(AppTheme.accent)
                    .frame(width: 28, height: 28)
                    .background(AppTheme.accent.opacity(0.12), in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                VStack(alignment: .leading, spacing: 3) {
                    Text(item.displayName)
                        .font(.system(size: 12, weight: .semibold, design: .rounded))
                        .foregroundStyle(AppTheme.text)
                        .lineLimit(1)
                    Text(item.relativeDisplayName)
                        .font(.system(size: 10))
                        .foregroundStyle(AppTheme.tertiaryText)
                        .lineLimit(1)
                }
                Spacer(minLength: 8)
                VStack(alignment: .trailing, spacing: 3) {
                    Text(item.sizeLabel)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(AppTheme.secondaryText)
                    Text(item.modifiedAt, format: .dateTime.month().day().hour().minute())
                        .font(.system(size: 10))
                        .foregroundStyle(AppTheme.tertiaryText)
                }
            }
            .padding(.horizontal, 9)
            .padding(.vertical, 8)
            .background(isSelected ? AppTheme.surfaceSelected : .clear, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

struct ResultPreviewPanel: View {
    @EnvironmentObject private var store: WorkspaceStore
    @StateObject private var imageLoader = ResultImageLoader()
    let item: OutputItem?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionTitle("プレビュー", subtitle: item?.category.title ?? "結果を選択してください")
            if let item {
                if item.kind == .image {
                    if let image = imageLoader.image {
                        Image(nsImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(maxWidth: .infinity, maxHeight: 330)
                            .background(AppTheme.canvas, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                    } else if imageLoader.isLoading {
                        ProgressView()
                            .controlSize(.large)
                            .frame(maxWidth: .infinity, minHeight: 220)
                    } else {
                        previewPlaceholder(title: "画像を読み込めませんでした", message: "ファイルが移動または削除された可能性があります")
                    }
                } else {
                    previewPlaceholder(
                        title: item.kind == .video ? "動画ファイル" : "データファイル",
                        message: "下のボタンから開けます",
                        symbol: item.kind.symbol
                    )
                }

                Text(item.url.path)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(AppTheme.tertiaryText)
                    .textSelection(.enabled)
                    .lineLimit(3)
                HStack(spacing: 10) {
                    Button {
                        store.openResult(item)
                    } label: {
                        Label("ファイルを開く", systemImage: "arrow.up.right.square")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(AppTheme.accent)
                    Button("Finderで表示") {
                        store.revealResult(item)
                    }
                    .buttonStyle(.bordered)
                }
            } else {
                EmptyState(symbol: "photo", title: "結果を選択してください", message: "左の一覧から候補ファイルを選ぶと、ここにプレビューが表示されます")
                    .frame(maxHeight: .infinity)
            }
            Spacer(minLength: 0)
        }
        .onAppear { imageLoader.load(item: item) }
        .onChange(of: item?.id) { _, _ in
            imageLoader.load(item: item)
        }
    }

    @ViewBuilder
    private func previewPlaceholder(
        title: String,
        message: String,
        symbol: String = "photo"
    ) -> some View {
        VStack(spacing: 12) {
            Image(systemName: symbol)
                .font(.system(size: 42, weight: .medium))
                .foregroundStyle(AppTheme.accent)
            Text(title)
                .font(.system(size: 14, weight: .semibold, design: .rounded))
                .foregroundStyle(AppTheme.text)
            Text(message)
                .font(.system(size: 12))
                .foregroundStyle(AppTheme.secondaryText)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, minHeight: 220)
        .background(AppTheme.canvas, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

@MainActor
final class ResultImageLoader: ObservableObject {
    @Published private(set) var image: NSImage?
    @Published private(set) var isLoading = false

    private var loadedID: String?
    private var generation = 0

    func load(item: OutputItem?) {
        guard let item, item.kind == .image else {
            generation &+= 1
            loadedID = nil
            image = nil
            isLoading = false
            return
        }
        guard loadedID != item.id || image == nil else { return }

        generation &+= 1
        let currentGeneration = generation
        loadedID = item.id
        image = nil
        isLoading = true
        let url = item.url
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let data = try? Data(contentsOf: url)
            DispatchQueue.main.async { [weak self] in
                guard let self, self.generation == currentGeneration else { return }
                self.image = data.flatMap(NSImage.init(data:))
                self.isLoading = false
            }
        }
    }
}

struct KeyValueRow: View {
    let key: String
    let value: String

    var body: some View {
        HStack {
            Text(key)
                .font(.system(size: 12))
                .foregroundStyle(AppTheme.secondaryText)
            Spacer()
            Text(value)
                .font(.system(size: 12, weight: .semibold, design: .rounded))
                .foregroundStyle(AppTheme.text)
                .lineLimit(1)
        }
    }
}

struct SettingsView: View {
    @EnvironmentObject private var store: WorkspaceStore
    @StateObject private var maskEditor = MaskEditorState()
    @StateObject private var cameraModelBuilder = CameraModelBuilderDraft()
    @StateObject private var fixedPatternBuilder = FixedPatternBuilderDraft()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                PageHeader(
                    eyebrow: "System",
                    title: "設定",
                    subtitle: "検出条件と出力先を、処理を始める前にわかりやすく調整できます。"
                )

                GlassCard {
                    VStack(alignment: .leading, spacing: 18) {
                        SectionTitle("解析条件", subtitle: "まずは初期値のままでも動作します")
                        HStack(spacing: 28) {
                            Stepper(value: $store.maxWorkers, in: 1...6) {
                                SettingValue(title: "並列ワーカー", value: "\(store.maxWorkers)")
                            }
                            Stepper(value: $store.interval, in: 0.05...60, step: 0.05) {
                                SettingValue(title: "フレーム間隔", value: String(format: "%.2f 秒", store.interval))
                            }
                            Stepper(value: $store.duration, in: 0.05...30, step: 0.05) {
                                SettingValue(title: "クリップ長", value: String(format: "%.2f 秒", store.duration))
                            }
                        }
                        Toggle("日付フォルダに天文薄明フィルタを適用", isOn: $store.twilightFilterEnabled)
                            .toggleStyle(.switch)
                            .tint(AppTheme.accent)
                        HStack(spacing: 12) {
                            coordinateField(title: "観測地点の緯度", value: $store.latitude)
                            coordinateField(title: "観測地点の経度", value: $store.longitude)
                        }
                    }
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: 14) {
                        SectionTitle("処理対象の優先順位", subtitle: "複数の入力が有効な場合は、最上位の1種類だけを実行します")
                        ForEach(Array(store.sourcePriority.enumerated()), id: \.element) { index, sourceType in
                            HStack(spacing: 12) {
                                Text("\(index + 1)")
                                    .font(.system(size: 12, weight: .bold, design: .rounded))
                                    .foregroundStyle(AppTheme.accent)
                                    .frame(width: 26, height: 26)
                                    .background(AppTheme.accent.opacity(0.14), in: Circle())
                                Label(
                                    store.sourcePriorityTitle(for: sourceType),
                                    systemImage: store.sourcePrioritySymbol(for: sourceType)
                                )
                                .font(.system(size: 13, weight: .semibold, design: .rounded))
                                .foregroundStyle(AppTheme.text)
                                Spacer()
                                Button {
                                    store.moveSourcePriority(sourceType, by: -1)
                                } label: {
                                    Image(systemName: "chevron.up")
                                }
                                .buttonStyle(.borderless)
                                .disabled(index == 0)
                                .accessibilityLabel("上へ移動")
                                Button {
                                    store.moveSourcePriority(sourceType, by: 1)
                                } label: {
                                    Image(systemName: "chevron.down")
                                }
                                .buttonStyle(.borderless)
                                .disabled(index == store.sourcePriority.count - 1)
                                .accessibilityLabel("下へ移動")
                            }
                            .padding(.vertical, 4)
                            if index < store.sourcePriority.count - 1 {
                                Divider().overlay(AppTheme.border)
                            }
                        }
                        HStack {
                            if store.activeSourceTypeCount > 1 {
                                Label(
                                    "現在は「\(store.selectedSourceLabel)」を優先",
                                    systemImage: "arrow.up.circle.fill"
                                )
                                .font(.system(size: 11, weight: .medium))
                                .foregroundStyle(AppTheme.accent)
                            } else {
                                Text("入力種別を複数登録したときだけ、この順位が適用されます。")
                                    .font(.system(size: 11))
                                    .foregroundStyle(AppTheme.tertiaryText)
                            }
                            Spacer()
                            Button("標準に戻す") {
                                store.resetSourcePriority()
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: 14) {
                        SectionTitle("検出モデル", subtitle: "解析に使うPyTorchモデルを選択します")
                        HStack(spacing: 10) {
                            TextField("標準モデルを使用", text: $store.selectedModelPath)
                                .textFieldStyle(.roundedBorder)
                                .disabled(true)
                            Button("選択") {
                                chooseDetectionModel {
                                    store.selectedModelPath = $0
                                    store.saveSettings()
                                }
                            }
                            .buttonStyle(.bordered)
                            Button("開く") {
                                guard store.selectedModelConfigurationIsValid,
                                      !store.selectedModelPath.isEmpty else { return }
                                NSWorkspace.shared.open(URL(fileURLWithPath: store.selectedModelPath))
                            }
                            .buttonStyle(.bordered)
                            .disabled(!store.selectedModelConfigurationIsValid || store.selectedModelPath.isEmpty)
                        }
                        Label(
                            store.selectedModelStatusMessage,
                            systemImage: store.selectedModelConfigurationIsValid ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
                        )
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(store.selectedModelConfigurationIsValid ? AppTheme.success : AppTheme.warning)
                        Text("未選択の場合は、設定ファイルの標準モデルを使用します。")
                            .font(.system(size: 11))
                            .foregroundStyle(AppTheme.tertiaryText)
                    }
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: 14) {
                        SectionTitle("座標注釈", subtitle: "既存のWCS／固定カメラ補正データを検出結果へ適用します")
                        Toggle("プレートソルブ補正を使用する", isOn: $store.plateSolveEnabled)
                            .toggleStyle(.switch)
                            .tint(AppTheme.accent)
                            .onChange(of: store.plateSolveEnabled) { _, _ in
                                store.saveSettings()
                            }
                        HStack(spacing: 10) {
                            TextField("補正データ (.json / .wcs / .fits)", text: $store.plateSolvePath)
                                .textFieldStyle(.roundedBorder)
                                .disabled(true)
                            Button("選択") {
                                choosePlateSolveFile {
                                    store.plateSolvePath = $0
                                    store.plateSolveEnabled = true
                                    store.saveSettings()
                                }
                            }
                            .buttonStyle(.bordered)
                            Button("新規作成") {
                                cameraModelBuilder.isPresented = true
                            }
                            .buttonStyle(.bordered)
                            .disabled(store.cameraModelBuildActive)
                            Button("開く") {
                                guard store.plateSolveConfigurationIsValid else { return }
                                NSWorkspace.shared.open(URL(fileURLWithPath: store.plateSolvePath))
                            }
                            .buttonStyle(.bordered)
                            .disabled(!store.plateSolveConfigurationIsValid || !store.plateSolveEnabled)
                        }
                        Label(
                            store.plateSolveStatusMessage,
                            systemImage: store.plateSolveConfigurationIsValid ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
                        )
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(store.plateSolveConfigurationIsValid ? AppTheme.success : AppTheme.warning)
                        Text("既存データの適用と、高精度カメラ補正データの新規作成をSwiftUIから実行できます。")
                            .font(.system(size: 11))
                            .foregroundStyle(AppTheme.tertiaryText)
                    }
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: 14) {
                        SectionTitle("検出マスク", subtitle: "空やノイズの多い領域を解析対象から外します")
                        Toggle("検出マスクを適用する", isOn: $store.detectionMaskEnabled)
                            .toggleStyle(.switch)
                            .tint(AppTheme.accent)
                            .onChange(of: store.detectionMaskEnabled) { _, _ in
                                store.saveSettings()
                                store.validateDetectionMask()
                            }
                        HStack {
                            Text("動画の先頭フレーム上で、除外する領域を直接描けます。")
                                .font(.system(size: 11))
                                .foregroundStyle(AppTheme.tertiaryText)
                            Spacer()
                            Button("描く") {
                                maskEditor.present()
                            }
                            .buttonStyle(.bordered)
                        }
                        if store.detectionMaskEnabled {
                            HStack(spacing: 10) {
                                TextField("app_masks.npz", text: $store.detectionMaskPath)
                                    .textFieldStyle(.roundedBorder)
                                    .onSubmit {
                                        store.saveSettings()
                                        store.validateDetectionMask()
                                    }
                                Button("選択") {
                                    chooseMaskFile {
                                        store.detectionMaskPath = $0
                                        store.saveSettings()
                                        store.validateDetectionMask()
                                    }
                                }
                                .buttonStyle(.bordered)
                                Button("開く") {
                                    guard store.detectionMaskIsValid else { return }
                                    NSWorkspace.shared.open(URL(fileURLWithPath: store.detectionMaskPath))
                                }
                                .buttonStyle(.bordered)
                            }
                            Label(
                                store.detectionMaskStatusMessage,
                                systemImage: store.detectionMaskIsValid ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
                            )
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(store.detectionMaskIsValid ? AppTheme.success : AppTheme.warning)
                        }
                        Text("旧UIで作成した app_masks.npz をそのまま利用できます。")
                            .font(.system(size: 11))
                            .foregroundStyle(AppTheme.tertiaryText)
                    }
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: 14) {
                        SectionTitle("ノイズ前処理", subtitle: "モデルまたは短い時間平均で、検出前の揺らぎを抑えます")
                        Toggle("NoiseTwinモデルを適用する", isOn: $store.noiseTwinEnabled)
                            .toggleStyle(.switch)
                            .tint(AppTheme.accent)
                            .onChange(of: store.noiseTwinEnabled) { _, enabled in
                                if enabled { store.temporalMeanFrames = 0 }
                                store.saveSettings()
                            }
                        if store.noiseTwinEnabled {
                            HStack(spacing: 10) {
                                TextField("NoiseTwinモデル (.pth)", text: $store.noiseTwinModelPath)
                                    .textFieldStyle(.roundedBorder)
                                    .disabled(true)
                                Button("選択") {
                                    chooseNoiseTwinModel {
                                        store.noiseTwinModelPath = $0
                                        store.saveSettings()
                                    }
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                        HStack(spacing: 20) {
                            Picker("時間平均", selection: $store.temporalMeanFrames) {
                                Text("OFF").tag(0)
                                Text("3フレーム").tag(3)
                                Text("5フレーム").tag(5)
                            }
                            .pickerStyle(.segmented)
                            .disabled(store.noiseTwinEnabled)
                            .onChange(of: store.temporalMeanFrames) { _, frames in
                                if frames != 0 { store.noiseTwinEnabled = false }
                                store.saveSettings()
                            }
                            Toggle("RTSP保存動画にも適用", isOn: $store.saveTemporalMeanVideo)
                                .toggleStyle(.switch)
                                .tint(AppTheme.accent)
                                .disabled(store.temporalMeanFrames == 0 || store.noiseTwinEnabled)
                                .onChange(of: store.saveTemporalMeanVideo) { _, _ in
                                    store.saveSettings()
                                }
                        }
                        Label(
                            store.noiseTwinStatusMessage,
                            systemImage: store.noiseTwinConfigurationIsValid ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
                        )
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(store.noiseTwinConfigurationIsValid ? AppTheme.success : AppTheme.warning)
                        Text("NoiseTwinと時間平均は同時に使用できません。NoiseTwinモデルは検証済みのものだけ実行できます。")
                            .font(.system(size: 11))
                            .foregroundStyle(AppTheme.tertiaryText)
                    }
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: 14) {
                        SectionTitle("サマリー出力", subtitle: "検出後に保存する画像・動画を選びます")
                        Text("チェックを外した形式は作成されません。動画形式は候補を表示する時間も調整できます。")
                            .font(.system(size: 12))
                            .foregroundStyle(AppTheme.secondaryText)

                        ForEach($store.summaryVideoOptions) { $option in
                            VStack(alignment: .leading, spacing: 10) {
                                HStack(alignment: .top, spacing: 12) {
                                    Image(systemName: option.supportsDuration ? "film.stack" : "photo.stack")
                                        .font(.system(size: 16, weight: .medium))
                                        .foregroundStyle(AppTheme.accent)
                                        .frame(width: 30, height: 30)
                                        .background(AppTheme.accent.opacity(0.13), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(option.title)
                                            .font(.system(size: 13, weight: .semibold, design: .rounded))
                                            .foregroundStyle(AppTheme.text)
                                        Text(option.subtitle)
                                            .font(.system(size: 11))
                                            .foregroundStyle(AppTheme.tertiaryText)
                                    }
                                    Spacer()
                                    Toggle("", isOn: $option.enabled)
                                        .labelsHidden()
                                        .toggleStyle(.switch)
                                        .tint(AppTheme.accent)
                                }
                                if option.supportsDuration && option.enabled {
                                    HStack {
                                        Text("表示時間")
                                            .font(.system(size: 11, weight: .medium))
                                            .foregroundStyle(AppTheme.secondaryText)
                                        Spacer()
                                        Stepper(value: $option.duration, in: 0.05...60, step: 0.05) {
                                            Text(String(format: "%.2f 秒", option.duration))
                                                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                                                .foregroundStyle(AppTheme.text)
                                        }
                                    }
                                    .padding(.leading, 42)
                                }
                            }
                            .padding(.vertical, 5)
                        }

                        if !store.summaryVideoSelectionIsValid {
                            Label("少なくとも1つの出力形式を選択してください。", systemImage: "exclamationmark.triangle.fill")
                                .font(.system(size: 11, weight: .medium))
                                .foregroundStyle(AppTheme.warning)
                        }
                    }
                    .onChange(of: store.summaryVideoOptions) { _, _ in
                        store.saveSettings()
                    }
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: 16) {
                        SectionTitle("定期スキャン", subtitle: "指定したフォルダを監視し、新しい動画を自動で解析します")
                        Toggle("定期スキャンを有効にする", isOn: $store.periodicScanEnabled)
                            .toggleStyle(.switch)
                            .tint(AppTheme.accent)
                            .onChange(of: store.periodicScanEnabled) { _, _ in
                                store.saveSettings()
                            }
                        if store.periodicScanEnabled {
                            HStack(spacing: 10) {
                                TextField("監視フォルダ", text: $store.periodicScanDirectory)
                                    .textFieldStyle(.roundedBorder)
                                    .onSubmit { store.saveSettings() }
                                Button("選択") {
                                    chooseDirectory {
                                        store.periodicScanDirectory = $0
                                        store.saveSettings()
                                    }
                                }
                                .buttonStyle(.bordered)
                                Button("開く") {
                                    guard !store.periodicScanDirectory.isEmpty else { return }
                                    NSWorkspace.shared.open(URL(fileURLWithPath: store.periodicScanDirectory))
                                }
                                .buttonStyle(.bordered)
                            }
                            HStack(spacing: 28) {
                                Stepper(value: $store.periodicScanInterval, in: 5...3600, step: 5) {
                                    SettingValue(title: "監視間隔", value: "\(store.periodicScanInterval) 秒")
                                }
                                Toggle("時間帯を制限", isOn: $store.periodicTimeLimitEnabled)
                                    .toggleStyle(.switch)
                                    .tint(AppTheme.accent)
                            }
                            if store.periodicTimeLimitEnabled {
                                HStack(spacing: 12) {
                                    TimeStepperGroup(
                                        title: "開始",
                                        hour: $store.periodicStartHour,
                                        minute: $store.periodicStartMinute
                                    )
                                    TimeStepperGroup(
                                        title: "終了",
                                        hour: $store.periodicEndHour,
                                        minute: $store.periodicEndMinute
                                    )
                                }
                                if !store.periodicTimeWindowIsValid {
                                    Text("開始と終了が同じ場合は、時間制限をオフにしてください。")
                                        .font(.system(size: 11, weight: .medium))
                                        .foregroundStyle(AppTheme.warning)
                                }
                            }
                        }
                    }
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: 16) {
                        SectionTitle("RTSP運用", subtitle: "ネットワークカメラの録画時間帯と検出通知を設定します")
                        HStack(spacing: 28) {
                            Picker("検出感度", selection: $store.rtspPreset) {
                                Text("雲が多いとき（推奨）").tag("cloudy")
                                Text("雲が少ないとき").tag("clear")
                            }
                            .pickerStyle(.menu)
                            .onChange(of: store.rtspPreset) { _, _ in
                                store.saveSettings()
                            }
                            Stepper(value: $store.rtspFPS, in: 1...120) {
                                SettingValue(title: "録画FPS", value: "\(store.rtspFPS) fps")
                            }
                            .onChange(of: store.rtspFPS) { _, _ in
                                store.saveSettings()
                            }
                        }
                        Toggle("録画時間帯を制限する", isOn: $store.rtspTimeLimitEnabled)
                            .toggleStyle(.switch)
                            .tint(AppTheme.accent)
                        if store.rtspTimeLimitEnabled {
                            HStack(spacing: 12) {
                                TimeStepperGroup(
                                    title: "開始",
                                    hour: $store.rtspStartHour,
                                    minute: $store.rtspStartMinute
                                )
                                TimeStepperGroup(
                                    title: "終了",
                                    hour: $store.rtspEndHour,
                                    minute: $store.rtspEndMinute
                                )
                            }
                            Text("時間外は録画を停止し、時間帯内に再開します。")
                                .font(.system(size: 11))
                                .foregroundStyle(AppTheme.tertiaryText)
                            if !store.rtspTimeWindowIsValid {
                                Text("開始と終了が同じ場合は、時間制限をオフにしてください。")
                                    .font(.system(size: 11, weight: .medium))
                                    .foregroundStyle(AppTheme.warning)
                            }
                        }
                        Toggle("流星検出時に通知音を鳴らす", isOn: $store.rtspNotificationSound)
                            .toggleStyle(.switch)
                            .tint(AppTheme.accent)

                        Divider().overlay(AppTheme.border)
                        HStack(alignment: .top) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("固定パターン補正")
                                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                                    .foregroundStyle(AppTheme.text)
                                Text("センサーの細かなムラを検出前に補正します。既存の旧形式NPZも利用できます。")
                                    .font(.system(size: 11))
                                    .foregroundStyle(AppTheme.tertiaryText)
                            }
                            Spacer()
                            Toggle("適用", isOn: $store.rtspFixedPatternEnabled)
                                .labelsHidden()
                                .toggleStyle(.switch)
                                .tint(AppTheme.accent)
                                .onChange(of: store.rtspFixedPatternEnabled) { _, _ in
                                    store.saveSettings()
                                    store.validateFixedPattern()
                                }
                        }
                        HStack(spacing: 10) {
                            TextField("rtsp_dark_frame.npz", text: $store.rtspFixedPatternPath)
                                .textFieldStyle(.roundedBorder)
                                .disabled(true)
                            Button("選択") {
                                chooseFixedPatternFile {
                                    store.rtspFixedPatternPath = $0
                                    store.rtspFixedPatternEnabled = true
                                    store.saveSettings()
                                    store.validateFixedPattern()
                                }
                            }
                            .buttonStyle(.bordered)
                            .disabled(store.fixedPatternBuildActive)
                            Button("作成") {
                                fixedPatternBuilder.mode = store.rtspSourceCount > 0 ? .url : .directory
                                fixedPatternBuilder.isPresented = true
                            }
                            .buttonStyle(.bordered)
                            .disabled(store.fixedPatternBuildActive)
                            Button("開く") {
                                guard store.fixedPatternConfigurationIsValid else { return }
                                NSWorkspace.shared.open(URL(fileURLWithPath: store.rtspFixedPatternPath))
                            }
                            .buttonStyle(.bordered)
                            .disabled(store.fixedPatternBuildActive || !store.fixedPatternConfigurationIsValid)
                        }
                        Label(
                            store.fixedPatternStatusMessage,
                            systemImage: store.fixedPatternConfigurationIsValid ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
                        )
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(store.fixedPatternConfigurationIsValid ? AppTheme.success : AppTheme.warning)
                    }
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: 16) {
                        SectionTitle("保存先", subtitle: "既存のapp_settings.jsonと互換性を保って保存します")
                        OutputPathRow(title: "流星候補", path: $store.meteorSavePath) {
                            chooseDirectory {
                                store.meteorSavePath = $0
                                store.saveSettings()
                                store.refreshResults()
                            }
                        } open: {
                            store.openOutputFolder(store.meteorSavePath)
                        } commit: {
                            store.saveSettings()
                            store.refreshResults()
                        }
                        OutputPathRow(title: "非流星候補", path: $store.notMeteorSavePath) {
                            chooseDirectory {
                                store.notMeteorSavePath = $0
                                store.saveSettings()
                                store.refreshResults()
                            }
                        } open: {
                            store.openOutputFolder(store.notMeteorSavePath)
                        } commit: {
                            store.saveSettings()
                            store.refreshResults()
                        }
                    }
                }

                GlassCard {
                    VStack(alignment: .leading, spacing: 14) {
                        SectionTitle("保存する成果物", subtitle: "不要な項目はオフにして処理時間と容量を抑えられます")
                        SaveOptionToggle(title: "検出動画", key: "video")
                        SaveOptionToggle(title: "カットアウト差分", key: "cutout")
                        SaveOptionToggle(title: "フル差分", key: "full")
                        SaveOptionToggle(title: "比較明合成", key: "composite")
                        SaveOptionToggle(title: "検出情報ファイル", key: "info")
                        SaveOptionToggle(title: "サマリー", key: "summary")
                        SaveOptionToggle(title: "フルサイズ動画", key: "full_video")
                    }
                }

                HStack {
                    Spacer()
                    Button {
                        store.saveSettings(showSuccessLog: true)
                    } label: {
                        if store.isSavingSettings {
                            ProgressView()
                                .controlSize(.small)
                            Text("保存中…")
                        } else {
                            Label("設定を保存", systemImage: "checkmark")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(AppTheme.accent)
                    .disabled(store.isSavingSettings || !store.settingsLoaded)
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
        .sheet(isPresented: $maskEditor.isPresented) {
            MaskEditorView(editor: maskEditor)
                .environmentObject(store)
        }
        .sheet(isPresented: $cameraModelBuilder.isPresented) {
            CameraModelBuilderView(draft: cameraModelBuilder)
                .environmentObject(store)
        }
        .sheet(isPresented: $fixedPatternBuilder.isPresented) {
            FixedPatternBuilderView(draft: fixedPatternBuilder)
                .environmentObject(store)
        }
    }

    private func coordinateField(title: String, value: Binding<Double>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(AppTheme.secondaryText)
            TextField(title, value: value, format: .number.precision(.fractionLength(6)))
                .textFieldStyle(.roundedBorder)
        }
        .frame(maxWidth: 220)
    }

    private func chooseDirectory(_ completion: @escaping (String) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "選択"
        if panel.runModal() == .OK, let url = panel.url {
            completion(url.path)
        }
    }

    private func chooseMaskFile(_ completion: @escaping (String) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [UTType(filenameExtension: "npz")].compactMap { $0 }
        panel.prompt = "選択"
        if panel.runModal() == .OK, let url = panel.url {
            completion(url.path)
        }
    }

    private func chooseFixedPatternFile(_ completion: @escaping (String) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [UTType(filenameExtension: "npz")].compactMap { $0 }
        panel.prompt = "選択"
        if panel.runModal() == .OK, let url = panel.url {
            completion(url.path)
        }
    }

    private func chooseNoiseTwinModel(_ completion: @escaping (String) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [
            UTType(filenameExtension: "pth"),
            UTType(filenameExtension: "pt"),
        ].compactMap { $0 }
        panel.prompt = "選択"
        if panel.runModal() == .OK, let url = panel.url {
            completion(url.path)
        }
    }

    private func chooseDetectionModel(_ completion: @escaping (String) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [
            UTType(filenameExtension: "pth"),
            UTType(filenameExtension: "pt"),
        ].compactMap { $0 }
        panel.prompt = "選択"
        if panel.runModal() == .OK, let url = panel.url {
            completion(url.path)
        }
    }

    private func choosePlateSolveFile(_ completion: @escaping (String) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [
            UTType(filenameExtension: "json"),
            UTType(filenameExtension: "wcs"),
            UTType(filenameExtension: "fits"),
            UTType(filenameExtension: "fit"),
        ].compactMap { $0 }
        panel.prompt = "選択"
        if panel.runModal() == .OK, let url = panel.url {
            completion(url.path)
        }
    }
}

struct SettingValue: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(AppTheme.secondaryText)
            Text(value)
                .font(.system(size: 17, weight: .bold, design: .rounded))
                .foregroundStyle(AppTheme.text)
        }
    }
}

struct TimeStepperGroup: View {
    let title: String
    @Binding var hour: Int
    @Binding var minute: Int

    var body: some View {
        HStack(spacing: 8) {
            Text(title)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(AppTheme.secondaryText)
            Stepper(value: $hour, in: 0...23) {
                Text(String(format: "%02d", hour))
                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    .foregroundStyle(AppTheme.text)
            }
            Text(":")
                .foregroundStyle(AppTheme.secondaryText)
            Stepper(value: $minute, in: 0...59, step: 5) {
                Text(String(format: "%02d", minute))
                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    .foregroundStyle(AppTheme.text)
            }
        }
    }
}

struct OutputPathRow: View {
    let title: String
    @Binding var path: String
    let choose: () -> Void
    let open: () -> Void
    let commit: () -> Void
    @StateObject private var commitScheduler = PathCommitScheduler()

    var body: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(AppTheme.text)
                TextField("保存先", text: $path)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { commitScheduler.commitNow(commit) }
                    .onChange(of: path) { _, _ in
                        commitScheduler.schedule(commit)
                    }
            }
            Button("選択", action: choose)
                .buttonStyle(.bordered)
            Button("開く", action: open)
                .buttonStyle(.bordered)
        }
    }
}

@MainActor
final class PathCommitScheduler: ObservableObject {
    private var pendingWork: DispatchWorkItem?

    func schedule(_ action: @escaping () -> Void) {
        pendingWork?.cancel()
        let work = DispatchWorkItem(block: action)
        pendingWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.75, execute: work)
    }

    func commitNow(_ action: @escaping () -> Void) {
        pendingWork?.cancel()
        pendingWork = nil
        action()
    }

    deinit {
        pendingWork?.cancel()
    }
}

struct SaveOptionToggle: View {
    @EnvironmentObject private var store: WorkspaceStore
    let title: String
    let key: String

    var body: some View {
        Toggle(title, isOn: Binding(
            get: { store.saveOptions[key] ?? true },
            set: { store.saveOptions[key] = $0 }
        ))
        .toggleStyle(.switch)
        .tint(AppTheme.accent)
    }
}

struct ActivityView: View {
    @EnvironmentObject private var store: WorkspaceStore

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                PageHeader(
                    eyebrow: "Activity",
                    title: "アクティビティ",
                    subtitle: "バックエンドの状態と、処理中に発生したイベントを確認できます。"
                )
                Spacer()
                Button("ログを消去", role: .destructive) {
                    store.logs.removeAll()
                            }
                            .buttonStyle(.bordered)
                            .disabled(store.fixedPatternBuildActive)
                        }
            .padding(34)
            Divider().overlay(AppTheme.border)
            if store.logs.isEmpty {
                EmptyState(symbol: "waveform.path", title: "イベントはまだありません", message: "入力の追加や解析を行うと、ここに記録されます")
                    .frame(maxHeight: .infinity)
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 0) {
                            ForEach(store.logs.reversed()) { entry in
                                LogRow(entry: entry)
                                    .id(entry.id)
                            }
                        }
                        .padding(.horizontal, 34)
                        .padding(.vertical, 18)
                    }
                    .onChange(of: store.logs.count) { _, _ in
                        if let first = store.logs.last {
                            withAnimation { proxy.scrollTo(first.id, anchor: .top) }
                        }
                    }
                }
            }
        }
    }
}

struct LogRow: View {
    let entry: ActivityEntry

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Circle()
                .fill(color)
                .frame(width: 7, height: 7)
                .padding(.top, 6)
            Text(entry.date, format: .dateTime.hour().minute().second())
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(AppTheme.tertiaryText)
                .frame(width: 68, alignment: .leading)
            Text(entry.message)
                .font(.system(size: 12))
                .foregroundStyle(entry.level == .error ? AppTheme.danger : AppTheme.text)
                .textSelection(.enabled)
            Spacer()
        }
        .padding(.vertical, 9)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(AppTheme.border.opacity(0.45))
                .frame(height: 1)
        }
    }

    private var color: Color {
        switch entry.level {
        case .info: return AppTheme.accent
        case .warning: return AppTheme.warning
        case .error: return AppTheme.danger
        }
    }
}

struct CompactLogRow: View {
    let entry: ActivityEntry

    var body: some View {
        HStack(spacing: 8) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(entry.message)
                .font(.system(size: 11))
                .foregroundStyle(AppTheme.secondaryText)
                .lineLimit(1)
        }
    }

    private var color: Color {
        switch entry.level {
        case .info: return AppTheme.accent
        case .warning: return AppTheme.warning
        case .error: return AppTheme.danger
        }
    }
}
