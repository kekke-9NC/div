import AppKit
import Combine
import Foundation
import MeteorDetectorCore

enum DetectionMaskValidation: Equatable {
    case unknown
    case validating
    case valid
    case invalid(String)
}

@MainActor
final class WorkspaceStore: ObservableObject {
    @Published var selection: AppSection = .overview
    @Published var sources: [InputSource] = []
    @Published var sourcePriority = ["periodic", "rtsp", "folder"]
    @Published var connection: BackendConnectionState = .starting
    @Published var runState: RunState = .idle
    @Published var progress: PipelineProgress?
    @Published var logs: [ActivityEntry] = []
    @Published private(set) var results: [OutputItem] = []
    @Published var resultFilter: OutputCategory = .meteor
    @Published var selectedResultID: String?
    @Published var queueStatus = "待機中"
    @Published var isDropTargeted = false
    @Published private(set) var isRefreshingResults = false
    @Published private(set) var unsupportedFeatures: [String] = []
    @Published var allowReducedFeatureRun = false
    @Published private(set) var settingsLoaded = false
    @Published private(set) var isSavingSettings = false

    @Published var meteorSavePath: String
    @Published var notMeteorSavePath: String
    @Published var maxWorkers = 4
    @Published var interval = 1.0
    @Published var duration = 1.0
    @Published var twilightFilterEnabled = true
    @Published var latitude = 35.0
    @Published var longitude = 135.0
    @Published var saveOptions: [String: Bool]
    @Published var summaryVideoOptions: [SummaryVideoOption] = WorkspaceStore.defaultSummaryVideoOptions
    @Published var detectionMaskEnabled = false {
        didSet {
            detectionMaskValidation = .unknown
            detectionMaskValidationGeneration &+= 1
        }
    }
    @Published var detectionMaskPath = "" {
        didSet {
            detectionMaskValidation = .unknown
            detectionMaskValidationGeneration &+= 1
        }
    }
    @Published private(set) var detectionMaskValidation: DetectionMaskValidation = .unknown
    @Published var plateSolveEnabled = false
    @Published var plateSolvePath = ""
    @Published var periodicScanEnabled = false
    @Published var periodicScanDirectory = ""
    @Published var periodicScanInterval = 60
    @Published var periodicTimeLimitEnabled = false
    @Published var periodicStartHour = 17
    @Published var periodicStartMinute = 0
    @Published var periodicEndHour = 7
    @Published var periodicEndMinute = 0
    @Published var rtspTimeLimitEnabled = false
    @Published var rtspStartHour = 17
    @Published var rtspStartMinute = 0
    @Published var rtspEndHour = 7
    @Published var rtspEndMinute = 0
    @Published var rtspNotificationSound = true
    @Published var rtspPreset = "cloudy"
    @Published var rtspFPS = 25
    @Published var selectedModelPath = ""
    @Published var noiseTwinEnabled = false
    @Published var noiseTwinModelPath = ""
    @Published var temporalMeanFrames = 0
    @Published var saveTemporalMeanVideo = true

    let rootURL: URL
    private let bridge: PythonBridge
    private var cancellables = Set<AnyCancellable>()
    private var resultsRefreshGeneration = 0
    private var detectionMaskValidationGeneration = 0
    private var settingsSaveGeneration = 0
    private var settingsSaveCompletions: [(Result<Void, BridgeError>) -> Void] = []
    private var activeRunToken = UUID()

    private static let defaultSourcePriority = ["periodic", "rtsp", "folder"]

    private static func normalizeSourcePriority(_ priority: [String]) -> [String] {
        var normalized: [String] = []
        for item in priority where defaultSourcePriority.contains(item) {
            if !normalized.contains(item) { normalized.append(item) }
        }
        normalized.append(contentsOf: defaultSourcePriority.filter { !normalized.contains($0) })
        return normalized
    }

    private var normalizedSourcePriority: [String] {
        Self.normalizeSourcePriority(sourcePriority)
    }

    private static var defaultSummaryVideoOptions: [SummaryVideoOption] {
        [
            SummaryVideoOption(name: "Composite Image", enabled: true, duration: 1.0, supportsDuration: true),
            SummaryVideoOption(name: "Annotated Image", enabled: false, duration: 2.0, supportsDuration: true),
            SummaryVideoOption(name: "Full Size Video", enabled: true, supportsDuration: false),
            SummaryVideoOption(name: "Zoom Sequence", enabled: false, duration: 2.0, supportsDuration: true),
            SummaryVideoOption(name: "Cutout Video", enabled: true, supportsDuration: false),
        ]
    }

    init(rootURL: URL? = nil) {
        let resolvedRoot = rootURL ?? Self.resolveRoot()
        self.rootURL = resolvedRoot
        self.meteorSavePath = resolvedRoot.appendingPathComponent("meteor").path
        self.notMeteorSavePath = resolvedRoot.appendingPathComponent("not_meteor").path
        self.saveOptions = LegacySettings().saveOptions
        self.detectionMaskPath = resolvedRoot.appendingPathComponent("app_masks.npz").path

        let bridge = PythonBridge(rootURL: resolvedRoot)
        self.bridge = bridge

        bridge.onEnvelope = { [weak self] envelope in
            self?.handle(envelope)
        }
        bridge.onDiagnostic = { [weak self] message in
            self?.appendLog(message, level: .warning)
        }

        appendLog("SwiftUIインターフェースを起動しています…")
        bridge.start()
        bridge.request("ping") { [weak self] result in
            switch result {
            case .success(let payload):
                self?.connection = .connected
                if let name = payload["name"] as? String {
                    self?.appendLog("\(name) の処理エンジンに接続しました。")
                }
            case .failure(let error):
                self?.connection = .unavailable(error.localizedDescription)
                self?.appendLog(error.localizedDescription, level: .error)
            }
        }
        bridge.request("load_settings") { [weak self] result in
            switch result {
            case .success(let payload):
                self?.applySettings(
                    payload["settings"] as? [String: Any] ?? [:],
                    unsupported: self?.stringArray(payload["unsupportedFeatures"]) ?? []
                )
            case .failure(let error):
                self?.settingsLoaded = true
                self?.appendLog(error.localizedDescription, level: .error)
            }
        }
    }

    deinit {
        let bridge = bridge
        Task { @MainActor in
            bridge.stop()
        }
    }

    var isBusy: Bool { runState.isActive }

    var canStart: Bool {
        guard settingsLoaded, connection == .connected, !isBusy else { return false }
        guard detectionMaskIsValid, summaryVideoSelectionIsValid else { return false }
        guard selectedModelConfigurationIsValid else { return false }
        guard plateSolveConfigurationIsValid else { return false }
        guard noiseTwinConfigurationIsValid else { return false }
        guard unsupportedFeatures.isEmpty || allowReducedFeatureRun else { return false }
        guard let sourceType = selectedSourceType else { return false }
        switch sourceType {
        case "periodic":
            return periodicDirectoryIsValid && periodicTimeWindowIsValid
        case "rtsp":
            let rtspSources = sources.filter { $0.kind == .rtsp }
            return rtspSources.count == 1
                && rtspSources.allSatisfy(\.exists)
                && rtspTimeWindowIsValid
        case "folder":
            let localSources = sources.filter { $0.kind != .rtsp }
            return !localSources.isEmpty && localSources.allSatisfy(\.exists)
        default:
            return false
        }
    }

    var localSourceCount: Int {
        sources.filter { $0.kind != .rtsp }.count
    }

    var rtspSourceCount: Int {
        sources.filter { $0.kind == .rtsp }.count
    }

    var selectedSourceType: String? {
        let active: [String: Bool] = [
            "periodic": periodicScanEnabled,
            "rtsp": rtspSourceCount > 0,
            "folder": localSourceCount > 0,
        ]
        for sourceType in normalizedSourcePriority where active[sourceType] == true {
            return sourceType
        }
        return nil
    }

    var activeSourceTypeCount: Int {
        [periodicScanEnabled, rtspSourceCount > 0, localSourceCount > 0].filter { $0 }.count
    }

    var selectedSourceLabel: String {
        sourcePriorityTitle(for: selectedSourceType ?? "")
    }

    var selectedModelConfigurationIsValid: Bool {
        let path = selectedModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty, FileManager.default.fileExists(atPath: path) else {
            return path.isEmpty
        }
        return (try? URL(fileURLWithPath: path)
            .resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
    }

    var selectedModelStatusMessage: String {
        let path = selectedModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else { return "標準モデルを使用します" }
        guard selectedModelConfigurationIsValid else {
            return "選択した検出モデルが見つかりません"
        }
        return "このモデルを実行時に適用します"
    }

    var plateSolveConfigurationIsValid: Bool {
        guard plateSolveEnabled else { return true }
        let path = plateSolvePath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty,
              FileManager.default.fileExists(atPath: path),
              (try? URL(fileURLWithPath: path)
                  .resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else {
            return false
        }
        return ["json", "wcs", "fits", "fit"].contains(URL(fileURLWithPath: path).pathExtension.lowercased())
    }

    var plateSolveStatusMessage: String {
        guard plateSolveEnabled else { return "座標注釈は無効です" }
        guard !plateSolvePath.isEmpty else { return "既存のWCS／カメラ補正データを選択してください" }
        guard plateSolveConfigurationIsValid else { return "カメラ補正データが見つからないか形式が不正です" }
        return "解析結果に座標注釈を付けます"
    }

    func sourcePriorityTitle(for sourceType: String) -> String {
        switch sourceType {
        case "periodic": return "定期スキャン"
        case "rtsp": return "RTSPストリーム"
        case "folder": return "フォルダ / 動画ファイル"
        default: return "入力"
        }
    }

    func sourcePrioritySymbol(for sourceType: String) -> String {
        switch sourceType {
        case "periodic": return "clock.arrow.2.circlepath"
        case "rtsp": return "dot.radiowaves.left.and.right"
        case "folder": return "film.stack"
        default: return "line.3.horizontal"
        }
    }

    func moveSourcePriority(_ sourceType: String, by offset: Int) {
        guard let index = sourcePriority.firstIndex(of: sourceType) else { return }
        let target = index + offset
        guard sourcePriority.indices.contains(target) else { return }
        sourcePriority.swapAt(index, target)
        saveSettings()
    }

    func resetSourcePriority() {
        sourcePriority = Self.defaultSourcePriority
        saveSettings()
    }

    var filteredResults: [OutputItem] {
        results.filter { $0.category == resultFilter }
    }

    var periodicDirectoryIsValid: Bool {
        guard !periodicScanDirectory.isEmpty else { return false }
        return (try? URL(fileURLWithPath: periodicScanDirectory)
            .resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
    }

    var periodicTimeWindowIsValid: Bool {
        !periodicTimeLimitEnabled || periodicStartHour != periodicEndHour || periodicStartMinute != periodicEndMinute
    }

    var rtspTimeWindowIsValid: Bool {
        !rtspTimeLimitEnabled || rtspStartHour != rtspEndHour || rtspStartMinute != rtspEndMinute
    }

    var summaryVideoSelectionIsValid: Bool {
        summaryVideoOptions.contains(where: { $0.enabled })
    }

    var noiseTwinConfigurationIsValid: Bool {
        if noiseTwinEnabled {
            return !noiseTwinModelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                && FileManager.default.fileExists(atPath: noiseTwinModelPath)
        }
        return temporalMeanFrames == 0 || temporalMeanFrames == 3 || temporalMeanFrames == 5
    }

    var noiseTwinStatusMessage: String {
        if noiseTwinEnabled {
            guard !noiseTwinModelPath.isEmpty else { return "NoiseTwinモデルを選択してください" }
            guard FileManager.default.fileExists(atPath: noiseTwinModelPath) else {
                return "NoiseTwinモデルが見つかりません"
            }
            return "実行開始時にモデルの検証を確認します"
        }
        switch temporalMeanFrames {
        case 3, 5: return "検出時に\(temporalMeanFrames)フレーム平均を使用します"
        default: return "ノイズ前処理は無効です"
        }
    }

    var detectionMaskIsValid: Bool {
        guard detectionMaskEnabled else { return true }
        guard URL(fileURLWithPath: detectionMaskPath).pathExtension.lowercased() == "npz" else {
            return false
        }
        guard FileManager.default.fileExists(atPath: detectionMaskPath) else { return false }
        return detectionMaskValidation == .valid
    }

    var detectionMaskStatusMessage: String {
        guard detectionMaskEnabled else { return "無効" }
        guard URL(fileURLWithPath: detectionMaskPath).pathExtension.lowercased() == "npz" else {
            return ".npz形式の検出マスクを選択してください"
        }
        guard FileManager.default.fileExists(atPath: detectionMaskPath) else {
            return "検出マスクファイルが見つかりません"
        }
        switch detectionMaskValidation {
        case .unknown, .validating:
            return "検出マスクを確認しています…"
        case .valid:
            return "検出マスクを使用できます"
        case .invalid(let message):
            return message
        }
    }

    var readinessMessage: String {
        if connection != .connected { return "処理エンジンを接続しています" }
        if !unsupportedFeatures.isEmpty && !allowReducedFeatureRun {
            return "旧UIの未対応設定を確認してください"
        }
        if detectionMaskEnabled && !detectionMaskIsValid {
            return detectionMaskStatusMessage
        }
        if !selectedModelConfigurationIsValid { return selectedModelStatusMessage }
        if !plateSolveConfigurationIsValid { return plateSolveStatusMessage }
        if !noiseTwinConfigurationIsValid { return noiseTwinStatusMessage }
        if !summaryVideoSelectionIsValid { return "出力構成を1つ以上選択してください" }
        guard let sourceType = selectedSourceType else {
            return "入力ソースを追加するか、定期スキャンを有効にしてください"
        }
        let priorityNote = activeSourceTypeCount > 1 ? "\(selectedSourceLabel)を優先します。" : ""
        switch sourceType {
        case "periodic":
            if periodicScanDirectory.isEmpty { return "監視フォルダを設定してください" }
            if !periodicDirectoryIsValid { return "監視フォルダが見つかりません" }
            if !periodicTimeWindowIsValid { return "時間制限の開始と終了を変えてください" }
            return priorityNote.isEmpty ? "定期スキャンを開始できます" : "\(priorityNote)定期スキャンを開始できます"
        case "rtsp":
            if rtspSourceCount > 1 { return "RTSPは1台ずつ実行してください" }
            if !rtspTimeWindowIsValid { return "RTSP時間制限の開始と終了を変えてください" }
            return priorityNote.isEmpty ? "RTSP解析を開始できます" : "\(priorityNote)RTSP解析を開始できます"
        case "folder":
            let localSources = sources.filter { $0.kind != .rtsp }
            if localSources.isEmpty { return "動画・フォルダを追加してください" }
            if localSources.contains(where: { !$0.exists }) { return "存在しない動画入力があります" }
            return priorityNote.isEmpty ? "解析を開始できます" : "\(priorityNote)動画解析を開始できます"
        default:
            break
        }
        if !unsupportedFeatures.isEmpty { return "基本解析モードで開始できます" }
        return "解析を開始できます"
    }

    func appendLog(_ message: String, level: ActivityEntry.Level = .info) {
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        logs.append(ActivityEntry(date: Date(), message: trimmed, level: level))
        if logs.count > 500 {
            logs.removeFirst(logs.count - 500)
        }
    }

    func addSources(_ urls: [URL]) {
        var added = 0
        for url in urls {
            let path = url.standardizedFileURL.path
            guard FileManager.default.fileExists(atPath: path) else {
                appendLog("入力が見つかりません: \(path)", level: .warning)
                continue
            }
            let kind: SourceKind = url.hasDirectoryPath ? .folder : .file
            guard !sources.contains(where: { $0.kind == kind && $0.value == path }) else { continue }
            sources.append(InputSource(kind: kind, value: path))
            added += 1
        }
        if added > 0 {
            selection = .capture
            appendLog("入力ソースを\(added)件追加しました。")
            saveSettings()
        }
    }

    @discardableResult
    func addRTSP(_ rawValue: String) -> Bool {
        let value = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: value),
              let scheme = url.scheme?.lowercased(),
              (scheme == "rtsp" || scheme == "rtsps"),
              let host = url.host,
              !host.isEmpty else {
            appendLog("RTSP URLの形式を確認してください。", level: .warning)
            return false
        }
        guard !sources.contains(where: { $0.kind == .rtsp && $0.value == value }) else { return true }
        sources.append(InputSource(kind: .rtsp, value: value))
        selection = .capture
        appendLog("RTSPカメラを追加しました。")
        saveSettings()
        return true
    }

    func removeSource(_ source: InputSource) {
        sources.removeAll { $0.id == source.id }
        saveSettings()
    }

    func clearSources() {
        sources.removeAll()
        saveSettings()
    }

    func startDetection() {
        guard settingsLoaded else {
            appendLog("設定を読み込んでいます。少し待ってから再試行してください。", level: .warning)
            return
        }
        guard connection == .connected else {
            appendLog("処理エンジンに接続できていません。", level: .error)
            selection = .activity
            return
        }
        guard !isBusy else { return }
        guard detectionMaskIsValid else {
            appendLog("検出マスクファイルを選択してください。", level: .warning)
            selection = .settings
            return
        }
        guard summaryVideoSelectionIsValid else {
            appendLog("出力構成を1つ以上選択してください。", level: .warning)
            selection = .settings
            return
        }
        guard selectedModelConfigurationIsValid else {
            appendLog(selectedModelStatusMessage, level: .warning)
            selection = .settings
            return
        }
        guard plateSolveConfigurationIsValid else {
            appendLog(plateSolveStatusMessage, level: .warning)
            selection = .settings
            return
        }
        guard noiseTwinConfigurationIsValid else {
            appendLog(noiseTwinStatusMessage, level: .warning)
            selection = .settings
            return
        }
        guard unsupportedFeatures.isEmpty || allowReducedFeatureRun else {
            appendLog("未対応設定があるため停止しました。確認後に基本解析モードを許可してください。", level: .warning)
            selection = .analysis
            return
        }
        guard let sourceType = selectedSourceType else {
            appendLog("動画・フォルダ・RTSPカメラを追加するか、定期スキャンを有効にしてください。", level: .warning)
            selection = .capture
            return
        }

        if activeSourceTypeCount > 1 {
            appendLog("複数の入力が有効です。優先順位により「\(selectedSourceLabel)」だけを実行します。")
        }

        if sourceType == "periodic" {
            guard !periodicScanDirectory.isEmpty,
                  periodicDirectoryIsValid else {
                appendLog("定期スキャン用の監視フォルダを設定してください。", level: .warning)
                selection = .settings
                return
            }
            guard periodicTimeWindowIsValid else {
                appendLog("定期スキャン時間制限の開始と終了を変えてください。", level: .warning)
                selection = .settings
                return
            }
            selection = .analysis
            runState = .preparing
            progress = nil
            queueStatus = "定期スキャンを開始しています"
            activeRunToken = UUID()
            saveSettings()
            startPeriodicRun()
            return
        }

        let selectedSources: [InputSource]
        if sourceType == "rtsp" {
            let rtspSources = sources.filter { $0.kind == .rtsp }
            guard rtspSources.count == 1 else {
                appendLog("RTSPカメラは一度に1台だけ実行できます。", level: .warning)
                selection = .capture
                return
            }
            selectedSources = rtspSources
        } else {
            selectedSources = sources.filter { $0.kind != .rtsp }
        }

        guard !selectedSources.isEmpty else {
            appendLog("先に動画・フォルダを追加してください。", level: .warning)
            selection = .capture
            return
        }
        guard selectedSources.allSatisfy({ $0.exists }) else {
            appendLog("存在しない入力ソースを削除または選び直してください。", level: .error)
            selection = .capture
            return
        }
        guard sourceType != "rtsp" || rtspTimeWindowIsValid else {
            appendLog("RTSP時間制限の開始と終了を変えてください。", level: .warning)
            selection = .settings
            return
        }

        selection = .analysis
        runState = .preparing
        progress = nil
        queueStatus = "入力を確認しています"
        let runToken = UUID()
        activeRunToken = runToken
        saveSettings()

        if sourceType == "rtsp", let rtsp = selectedSources.first?.value {
            startRTSPRun(url: rtsp)
            return
        }

        let localPaths = selectedSources.map(\.value)

        bridge.request(
            "discover_sources",
            payload: [
                "paths": localPaths,
                "twilightFilter": twilightFilterEnabled,
                "latitude": latitude,
                "longitude": longitude,
            ]
        ) { [weak self] result in
            guard let self else { return }
            guard self.activeRunToken == runToken, self.runState != .cancelling else { return }
            switch result {
            case .failure(let error):
                self.runState = .failed(error.localizedDescription)
                self.appendLog(error.localizedDescription, level: .error)
            case .success(let payload):
                if self.boolValue(payload["cancelled"]) {
                    self.runState = .cancelled
                    self.queueStatus = "走査をキャンセルしました"
                    return
                }
                let discovered = (payload["sources"] as? [[String: Any]] ?? []).compactMap { item -> [String: Any]? in
                    guard let path = item["path"] as? String, !path.isEmpty else { return nil }
                    return ["path": path, "is_rtsp": false]
                }
                guard !discovered.isEmpty else {
                    self.runState = .idle
                    self.queueStatus = "入力なし"
                    self.appendLog("処理対象の動画が見つかりませんでした。", level: .warning)
                    return
                }
                self.startLocalRun(sources: discovered)
            }
        }
    }

    func cancelDetection() {
        guard isBusy else { return }
        activeRunToken = UUID()
        runState = .cancelling
        queueStatus = "停止要求を送信しました"
        bridge.request("cancel") { [weak self] result in
            if case .failure(let error) = result {
                self?.appendLog(error.localizedDescription, level: .warning)
            }
        }
    }

    func saveSettings(
        showSuccessLog: Bool = false,
        completion: ((Result<Void, BridgeError>) -> Void)? = nil
    ) {
        guard settingsLoaded else {
            completion?(.success(()))
            return
        }
        settingsSaveGeneration &+= 1
        let generation = settingsSaveGeneration
        if let completion {
            settingsSaveCompletions.append(completion)
        }
        isSavingSettings = true
        let localPaths = sources.filter { $0.kind != .rtsp }.map(\.value)
        let rtspURLs = sources.filter { $0.kind == .rtsp }.map(\.value)
        bridge.request(
            "save_settings",
            payload: [
                "settings": [
                    "folder_paths": localPaths,
                    "rtsp_urls": rtspURLs,
                    "processing_source_priority": sourcePriority,
                    "meteor_save_path": meteorSavePath,
                    "not_meteor_save_path": notMeteorSavePath,
                    "concurrency": String(maxWorkers),
                    "interval": String(interval),
                    "duration": String(duration),
                    "date_folder_twilight_filter_enabled": twilightFilterEnabled,
                    "observation_latitude": String(latitude),
                    "observation_longitude": String(longitude),
                    "save_options": saveOptions,
                    "summary_video_config": summaryVideoConfigPayload,
                    "apply_mask": detectionMaskEnabled,
                    "detection_mask_path": detectionMaskPath,
                    "mask_path_or_status": detectionMaskPath,
                    "has_mask_image": FileManager.default.fileExists(atPath: detectionMaskPath),
                    "use_plate_solve": plateSolveEnabled,
                    "plate_solve_wcs_path": plateSolvePath,
                    "periodic_scan_enabled": periodicScanEnabled,
                    "periodic_scan_directory": periodicScanDirectory,
                    "periodic_scan_interval": String(periodicScanInterval),
                    "periodic_time_limit_enabled": periodicTimeLimitEnabled,
                    "periodic_start_hour": String(periodicStartHour),
                    "periodic_start_minute": String(periodicStartMinute),
                    "periodic_end_hour": String(periodicEndHour),
                    "periodic_end_minute": String(periodicEndMinute),
                    "rtsp_time_limit_enabled": rtspTimeLimitEnabled,
                    "rtsp_start_hour": String(rtspStartHour),
                    "rtsp_start_minute": String(rtspStartMinute),
                    "rtsp_end_hour": String(rtspEndHour),
                    "rtsp_end_minute": String(rtspEndMinute),
                    "rtsp_notification_sound": rtspNotificationSound,
                    "rtsp_preset": rtspPreset,
                    "rtsp_fps": String(rtspFPS),
                    "selected_model_path": selectedModelPath,
                    "noise_twin_enabled": noiseTwinEnabled,
                    "noise_twin_model_path": noiseTwinModelPath,
                    "temporal_mean_frames": temporalMeanFrames,
                    "rtsp_save_temporal_mean": saveTemporalMeanVideo,
                ],
            ]
        ) { [weak self] result in
            guard let self, self.settingsSaveGeneration == generation else { return }
            self.isSavingSettings = false
            let completionResult: Result<Void, BridgeError>
            if case .failure(let error) = result {
                self.appendLog("設定を保存できませんでした: \(error.localizedDescription)", level: .warning)
                completionResult = .failure(error)
            } else if showSuccessLog {
                self.appendLog("設定を保存しました。")
                completionResult = .success(())
            } else {
                completionResult = .success(())
            }
            let completions = self.settingsSaveCompletions
            self.settingsSaveCompletions.removeAll()
            completions.forEach { $0(completionResult) }
        }
    }

    func validateDetectionMask() {
        guard detectionMaskEnabled else {
            detectionMaskValidation = .valid
            return
        }
        guard URL(fileURLWithPath: detectionMaskPath).pathExtension.lowercased() == "npz",
              FileManager.default.fileExists(atPath: detectionMaskPath) else {
            detectionMaskValidation = .invalid(detectionMaskStatusMessage)
            return
        }
        detectionMaskValidation = .validating
        detectionMaskValidationGeneration &+= 1
        let generation = detectionMaskValidationGeneration
        bridge.request("validate_mask", payload: ["maskPath": detectionMaskPath]) { [weak self] result in
            guard let self else { return }
            guard self.detectionMaskValidationGeneration == generation else { return }
            switch result {
            case .success:
                self.detectionMaskValidation = .valid
            case .failure(let error):
                self.detectionMaskValidation = .invalid(error.localizedDescription)
            }
        }
    }

    func saveMaskFromStrokes(
        width: Int,
        height: Int,
        brushSize: Double,
        strokes: [[String: Any]],
        completion: @escaping (Result<Void, BridgeError>) -> Void
    ) {
        bridge.request(
            "save_mask",
            payload: [
                "maskPath": detectionMaskPath,
                "width": width,
                "height": height,
                "brushSize": brushSize,
                "strokes": strokes,
            ]
        ) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success:
                self.detectionMaskEnabled = true
                self.validateDetectionMask()
                self.saveSettings { settingsResult in
                    completion(settingsResult)
                }
            case .failure(let error):
                self.appendLog("検出マスクを保存できませんでした: \(error.localizedDescription)", level: .error)
                completion(.failure(error))
            }
        }
    }

    func openOutputFolder(_ path: String) {
        let url = URL(fileURLWithPath: path)
        if !FileManager.default.fileExists(atPath: url.path) {
            do {
                try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            } catch {
                appendLog("出力フォルダを作成できませんでした: \(error.localizedDescription)", level: .error)
                return
            }
        }
        NSWorkspace.shared.open(url)
    }

    func openResult(_ item: OutputItem) {
        NSWorkspace.shared.open(item.url)
    }

    func revealResult(_ item: OutputItem) {
        NSWorkspace.shared.activateFileViewerSelecting([item.url])
    }

    func refreshResults() {
        resultsRefreshGeneration &+= 1
        let generation = resultsRefreshGeneration
        isRefreshingResults = true
        let meteorPath = URL(fileURLWithPath: meteorSavePath)
        let notMeteorPath = URL(fileURLWithPath: notMeteorSavePath)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let items = WorkspaceStore.collectResults(meteorPath: meteorPath, notMeteorPath: notMeteorPath)
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                guard self.resultsRefreshGeneration == generation else { return }
                self.isRefreshingResults = false
                self.results = items
                if let selected = self.selectedResultID,
                   !items.contains(where: { $0.id == selected }) {
                    self.selectedResultID = nil
                }
            }
        }
    }

    func shutdown() {
        guard settingsLoaded else {
            bridge.stop()
            return
        }
        saveSettings { [weak self] _ in
            self?.bridge.stop()
        }
        bridge.stop(after: 1.0)
    }

    private func startLocalRun(sources: [[String: Any]]) {
        queueStatus = "解析を開始しています"
        bridge.request(
            "run_detection",
            payload: runPayload(merging: ["sources": sources])
        ) { [weak self] result in
            if case .failure(let error) = result {
                self?.runState = .failed(error.localizedDescription)
                self?.appendLog(error.localizedDescription, level: .error)
            }
        }
    }

    private func startRTSPRun(url: String) {
        queueStatus = "RTSP接続を開始しています"
        bridge.request(
            "run_rtsp",
            payload: runPayload(
                merging: [
                    "url": url,
                    "timeLimitEnabled": rtspTimeLimitEnabled,
                    "startHour": rtspStartHour,
                    "startMinute": rtspStartMinute,
                    "endHour": rtspEndHour,
                    "endMinute": rtspEndMinute,
                    "notifyOnDetection": rtspNotificationSound,
                    "rtspPreset": rtspPreset,
                    "rtspFps": rtspFPS,
                ]
            )
        ) { [weak self] result in
            if case .failure(let error) = result {
                self?.runState = .failed(error.localizedDescription)
                self?.appendLog(error.localizedDescription, level: .error)
            }
        }
    }

    private func startPeriodicRun() {
        bridge.request(
            "run_periodic",
            payload: runPayload(
                merging: [
                    "directory": periodicScanDirectory,
                    "scanInterval": periodicScanInterval,
                    "timeLimitEnabled": periodicTimeLimitEnabled,
                    "startHour": periodicStartHour,
                    "startMinute": periodicStartMinute,
                    "endHour": periodicEndHour,
                    "endMinute": periodicEndMinute,
                ]
            )
        ) { [weak self] result in
            if case .failure(let error) = result {
                self?.runState = .failed(error.localizedDescription)
                self?.appendLog(error.localizedDescription, level: .error)
            }
        }
    }

    private func runPayload(merging additional: [String: Any]) -> [String: Any] {
        var payload: [String: Any] = [
            "maxWorkers": maxWorkers,
            "interval": interval,
            "duration": duration,
            "meteorSavePath": meteorSavePath,
            "notMeteorSavePath": notMeteorSavePath,
            "saveOptions": saveOptions,
            "summaryConfig": summaryVideoConfigPayload,
            "applyMask": detectionMaskEnabled,
            "maskPath": detectionMaskPath,
            "modelPath": selectedModelPath,
            "plateSolveWCSPath": plateSolveEnabled ? plateSolvePath : "",
            "noiseTwinOptions": [
                "enabled": noiseTwinEnabled,
                "modelPath": noiseTwinModelPath,
                "temporalMeanFrames": temporalMeanFrames,
                "saveTemporalMeanVideo": saveTemporalMeanVideo,
            ],
        ]
        for (key, value) in additional {
            payload[key] = value
        }
        return payload
    }

    private var summaryVideoConfigPayload: [[String: Any]] {
        summaryVideoOptions.map { option in
            var item: [String: Any] = [
                "name": option.name,
                "enabled": option.enabled,
            ]
            if option.supportsDuration {
                item["duration"] = option.duration
            }
            return item
        }
    }

    private func applySummaryVideoConfig(_ value: Any?) {
        guard let rawItems = value as? [[String: Any]] else { return }
        var restored: [SummaryVideoOption] = []
        var seen = Set<String>()

        for item in rawItems {
            guard let name = item["name"] as? String,
                  !name.isEmpty,
                  !seen.contains(name) else { continue }
            guard let template = Self.defaultSummaryVideoOptions.first(where: { $0.name == name }) else {
                appendLog("未対応のサマリー出力をスキップしました: \(name)", level: .warning)
                continue
            }
            let supportsDuration = template.supportsDuration
            let duration = max(
                0.05,
                min(60, doubleValue(item["duration"], default: template.duration))
            )
            restored.append(
                SummaryVideoOption(
                    name: name,
                    enabled: boolValue(item["enabled"], default: template.enabled),
                    duration: duration,
                    supportsDuration: supportsDuration
                )
            )
            seen.insert(name)
        }

        for option in Self.defaultSummaryVideoOptions where !seen.contains(option.name) {
            restored.append(option)
        }
        if !restored.isEmpty {
            summaryVideoOptions = restored
        }
    }

    private func handle(_ envelope: BridgeEnvelope) {
        guard envelope.type == "event" else { return }
        switch envelope.event {
        case "log":
            let message = envelope.payload["message"] as? String ?? ""
            let levelString = envelope.payload["level"] as? String ?? "info"
            let level: ActivityEntry.Level = levelString == "error" ? .error : levelString == "warning" ? .warning : .info
            appendLog(message, level: level)
        case "progress":
            let current = intValue(envelope.payload["current"], default: 0)
            let total = intValue(envelope.payload["total"], default: 0)
            let message = envelope.payload["message"] as? String ?? ""
            progress = PipelineProgress(current: current, total: total, message: message)
            queueStatus = message.isEmpty ? "\(current)/\(total) 本" : message
        case "status":
            let pending = intValue(envelope.payload["pending_sources"], default: 0)
            let queue = intValue(envelope.payload["download_queue_size"], default: 0)
            let busy = (envelope.payload["processors_busy"] as? [Any] ?? []).filter { boolValue($0) }.count
            queueStatus = "処理中 \(busy) / 待機 \(pending + queue)"
        case "run_state":
            let state = envelope.payload["state"] as? String ?? "idle"
            switch state {
            case "preparing":
                runState = .preparing
                progress = nil
            case "running": runState = .running
            case "cancelling": runState = .cancelling
            case "completed":
                runState = .completed
                progress = nil
                refreshResults()
            case "cancelled":
                runState = .cancelled
                progress = nil
            case "failed":
                runState = .failed(envelope.payload["error"] as? String ?? "処理に失敗しました")
                progress = nil
            case "discovery_completed":
                if case .cancelling = runState {
                    runState = .cancelled
                    progress = nil
                }
            case "idle":
                if case .cancelling = runState {
                    runState = .cancelled
                } else {
                    runState = .idle
                }
                progress = nil
            default:
                break
            }
        default:
            break
        }
    }

    private func applySettings(_ settings: [String: Any], unsupported: [String]) {
        unsupportedFeatures = unsupported
        if !unsupported.isEmpty {
            appendLog("SwiftUI版では未対応の旧設定があります: \(unsupported.joined(separator: "、"))", level: .warning)
        }
        let folderPaths = stringArray(settings["folder_paths"])
        let rtspURLs = stringArray(settings["rtsp_urls"])
        sourcePriority = Self.normalizeSourcePriority(stringArray(settings["processing_source_priority"]))
        var restored: [InputSource] = []
        restored += folderPaths.map {
            let isDirectory = (try? URL(fileURLWithPath: $0).resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
            return InputSource(kind: isDirectory ? .folder : .file, value: $0)
        }
        restored += rtspURLs.map { InputSource(kind: .rtsp, value: $0) }
        sources = restored

        if let value = settings["meteor_save_path"] as? String, !value.isEmpty { meteorSavePath = value }
        if let value = settings["not_meteor_save_path"] as? String, !value.isEmpty { notMeteorSavePath = value }
        maxWorkers = max(1, min(6, intValue(settings["concurrency"], default: 4)))
        interval = max(0.05, min(60, doubleValue(settings["interval"], default: 1)))
        duration = max(0.05, min(30, doubleValue(settings["duration"], default: 1)))
        twilightFilterEnabled = boolValue(settings["date_folder_twilight_filter_enabled"], default: true)
        latitude = doubleValue(settings["observation_latitude"], default: latitude)
        longitude = doubleValue(settings["observation_longitude"], default: longitude)
        applySummaryVideoConfig(settings["summary_video_config"])
        detectionMaskEnabled = boolValue(settings["apply_mask"], default: false)
        if let configuredMaskPath = settings["detection_mask_path"] as? String,
           !configuredMaskPath.isEmpty {
            detectionMaskPath = configuredMaskPath
        } else if let legacyMaskPath = settings["mask_path_or_status"] as? String,
                  FileManager.default.fileExists(atPath: legacyMaskPath) {
            detectionMaskPath = legacyMaskPath
        }
        plateSolvePath = settings["plate_solve_wcs_path"] as? String ?? ""
        plateSolveEnabled = boolValue(settings["use_plate_solve"], default: false)
        if plateSolveEnabled && !plateSolveConfigurationIsValid {
            plateSolveEnabled = false
            appendLog("保存されていたカメラ補正データを利用できないため、座標注釈を無効にしました。", level: .warning)
        }
        periodicScanEnabled = boolValue(settings["periodic_scan_enabled"], default: false)
        periodicScanDirectory = settings["periodic_scan_directory"] as? String ?? ""
        periodicScanInterval = max(5, min(3600, intValue(settings["periodic_scan_interval"], default: 60)))
        periodicTimeLimitEnabled = boolValue(settings["periodic_time_limit_enabled"], default: false)
        periodicStartHour = max(0, min(23, intValue(settings["periodic_start_hour"], default: 17)))
        periodicStartMinute = max(0, min(59, intValue(settings["periodic_start_minute"], default: 0)))
        periodicEndHour = max(0, min(23, intValue(settings["periodic_end_hour"], default: 7)))
        periodicEndMinute = max(0, min(59, intValue(settings["periodic_end_minute"], default: 0)))
        rtspTimeLimitEnabled = boolValue(settings["rtsp_time_limit_enabled"], default: false)
        rtspStartHour = max(0, min(23, intValue(settings["rtsp_start_hour"], default: 17)))
        rtspStartMinute = max(0, min(59, intValue(settings["rtsp_start_minute"], default: 0)))
        rtspEndHour = max(0, min(23, intValue(settings["rtsp_end_hour"], default: 7)))
        rtspEndMinute = max(0, min(59, intValue(settings["rtsp_end_minute"], default: 0)))
        rtspNotificationSound = boolValue(settings["rtsp_notification_sound"], default: true)
        let savedPreset = settings["rtsp_preset"] as? String ?? "cloudy"
        rtspPreset = savedPreset == "clear" ? "clear" : "cloudy"
        rtspFPS = max(1, min(120, intValue(settings["rtsp_fps"], default: 25)))
        let savedModelPath = (settings["selected_model_path"] as? String ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !savedModelPath.isEmpty,
           (try? URL(fileURLWithPath: savedModelPath)
               .resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
            selectedModelPath = savedModelPath
        } else {
            selectedModelPath = ""
            if !savedModelPath.isEmpty {
                appendLog("保存されていた検出モデルが見つからないため、標準モデルへ戻しました。", level: .warning)
            }
        }
        noiseTwinModelPath = settings["noise_twin_model_path"] as? String ?? ""
        noiseTwinEnabled = boolValue(settings["noise_twin_enabled"], default: false)
        let savedMeanFrames = intValue(settings["temporal_mean_frames"], default: 0)
        temporalMeanFrames = [0, 3, 5].contains(savedMeanFrames) ? savedMeanFrames : 0
        saveTemporalMeanVideo = boolValue(settings["rtsp_save_temporal_mean"], default: true)
        if noiseTwinEnabled { temporalMeanFrames = 0 }
        if let options = settings["save_options"] as? [String: Any] {
            for key in saveOptions.keys {
                if let value = options[key] { saveOptions[key] = boolValue(value, default: saveOptions[key] ?? true) }
            }
        }
        if !restored.isEmpty {
            appendLog("前回の入力設定を復元しました。")
        }
        validateDetectionMask()
        refreshResults()
        settingsLoaded = true
    }

    private nonisolated static func collectResults(meteorPath: URL, notMeteorPath: URL) -> [OutputItem] {
        let fileManager = FileManager.default
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .fileSizeKey, .contentModificationDateKey]
        let allowedExtensions: Set<String> = [
            "jpg", "jpeg", "png", "gif", "tif", "tiff", "webp",
            "mp4", "mov", "avi", "m4v", "mkv", "json", "txt", "csv",
        ]

        func scan(_ root: URL, category: OutputCategory) -> [OutputItem] {
            guard fileManager.fileExists(atPath: root.path),
                  let enumerator = fileManager.enumerator(
                      at: root,
                      includingPropertiesForKeys: Array(keys),
                      options: [.skipsHiddenFiles, .skipsPackageDescendants]
                  ) else { return [] }

            var found: [OutputItem] = []
            for case let url as URL in enumerator {
                let ext = url.pathExtension.lowercased()
                guard allowedExtensions.contains(ext), let kind = OutputKind.from(fileExtension: ext) else { continue }
                guard let values = try? url.resourceValues(forKeys: keys), values.isRegularFile == true else { continue }
                found.append(
                    OutputItem(
                        url: url,
                        category: category,
                        kind: kind,
                        byteCount: Int64(values.fileSize ?? 0),
                        modifiedAt: values.contentModificationDate ?? .distantPast
                    )
                )
            }
            return found
        }

        return (scan(meteorPath, category: .meteor) + scan(notMeteorPath, category: .notMeteor))
            .sorted { lhs, rhs in
                if lhs.modifiedAt != rhs.modifiedAt { return lhs.modifiedAt > rhs.modifiedAt }
                return lhs.id < rhs.id
            }
            .prefix(300)
            .map { $0 }
    }

    private static func resolveRoot() -> URL {
        let fileManager = FileManager.default
        var candidates: [URL] = []
        if let configured = ProcessInfo.processInfo.environment["METEOR_DETECTOR_ROOT"] {
            candidates.append(URL(fileURLWithPath: configured))
        }
        if let resourceURL = Bundle.main.resourceURL {
            candidates.append(resourceURL)
            candidates.append(resourceURL.deletingLastPathComponent())
        }
        if let executableURL = Bundle.main.executableURL {
            candidates.append(executableURL.deletingLastPathComponent())
        }
        candidates.append(URL(fileURLWithPath: fileManager.currentDirectoryPath))

        for base in candidates {
            var candidate = base.standardizedFileURL
            for _ in 0..<6 {
                if fileManager.fileExists(atPath: candidate.appendingPathComponent("swift_backend.py").path) {
                    return candidate
                }
                let parent = candidate.deletingLastPathComponent()
                if parent == candidate { break }
                candidate = parent
            }
        }
        return URL(fileURLWithPath: fileManager.currentDirectoryPath).standardizedFileURL
    }

    private func stringArray(_ value: Any?) -> [String] {
        (value as? [Any] ?? []).compactMap { $0 as? String }.filter { !$0.isEmpty }
    }

    private func intValue(_ value: Any?, default fallback: Int) -> Int {
        if let value = value as? Int { return value }
        if let value = value as? NSNumber { return value.intValue }
        if let value = value as? String, let parsed = Int(value) { return parsed }
        return fallback
    }

    private func doubleValue(_ value: Any?, default fallback: Double) -> Double {
        if let value = value as? Double { return value }
        if let value = value as? NSNumber { return value.doubleValue }
        if let value = value as? String, let parsed = Double(value) { return parsed }
        return fallback
    }

    private func boolValue(_ value: Any?, default fallback: Bool = false) -> Bool {
        if let value = value as? Bool { return value }
        if let value = value as? NSNumber { return value.boolValue }
        if let value = value as? String { return (value as NSString).boolValue }
        return fallback
    }
}
