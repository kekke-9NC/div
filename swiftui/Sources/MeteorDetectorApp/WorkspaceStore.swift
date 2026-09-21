import AppKit
import Combine
import Foundation
import MeteorDetectorCore

@MainActor
final class WorkspaceStore: ObservableObject {
    @Published var selection: AppSection = .overview
    @Published var sources: [InputSource] = []
    @Published var connection: BackendConnectionState = .starting
    @Published var runState: RunState = .idle
    @Published var progress: PipelineProgress?
    @Published var logs: [ActivityEntry] = []
    @Published private(set) var results: [OutputItem] = []
    @Published var resultFilter: OutputCategory = .meteor
    @Published var selectedResultID: String?
    @Published var queueStatus = "待機中"
    @Published var isDropTargeted = false
    @Published private(set) var unsupportedFeatures: [String] = []
    @Published var allowReducedFeatureRun = false

    @Published var meteorSavePath: String
    @Published var notMeteorSavePath: String
    @Published var maxWorkers = 4
    @Published var interval = 1.0
    @Published var duration = 1.0
    @Published var twilightFilterEnabled = true
    @Published var latitude = 35.0
    @Published var longitude = 135.0
    @Published var saveOptions: [String: Bool]

    let rootURL: URL
    private let bridge: PythonBridge
    private var cancellables = Set<AnyCancellable>()
    private var resultsRefreshGeneration = 0

    init(rootURL: URL? = nil) {
        let resolvedRoot = rootURL ?? Self.resolveRoot()
        self.rootURL = resolvedRoot
        self.meteorSavePath = resolvedRoot.appendingPathComponent("meteor").path
        self.notMeteorSavePath = resolvedRoot.appendingPathComponent("not_meteor").path
        self.saveOptions = LegacySettings().saveOptions

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

    var localSourceCount: Int {
        sources.filter { $0.kind != .rtsp }.count
    }

    var rtspSourceCount: Int {
        sources.filter { $0.kind == .rtsp }.count
    }

    var filteredResults: [OutputItem] {
        results.filter { $0.category == resultFilter }
    }

    var readinessMessage: String {
        if connection != .connected { return "処理エンジンを接続しています" }
        if !unsupportedFeatures.isEmpty && !allowReducedFeatureRun {
            return "旧UIの未対応設定を確認してください"
        }
        if sources.isEmpty { return "入力ソースを追加すると解析を開始できます" }
        if sources.contains(where: { !$0.exists }) { return "存在しない入力があります" }
        if localSourceCount > 0 && rtspSourceCount > 0 { return "動画とRTSPは同時に実行できません" }
        if rtspSourceCount > 1 { return "RTSPは1台ずつ実行してください" }
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
        guard connection == .connected else {
            appendLog("処理エンジンに接続できていません。", level: .error)
            selection = .activity
            return
        }
        guard !isBusy else { return }
        guard unsupportedFeatures.isEmpty || allowReducedFeatureRun else {
            appendLog("未対応設定があるため停止しました。確認後に基本解析モードを許可してください。", level: .warning)
            selection = .analysis
            return
        }
        guard !sources.isEmpty else {
            appendLog("先に動画・フォルダ・RTSPカメラを追加してください。", level: .warning)
            selection = .capture
            return
        }
        guard sources.allSatisfy({ $0.exists }) else {
            appendLog("存在しない入力ソースを削除または選び直してください。", level: .error)
            selection = .capture
            return
        }
        guard localSourceCount == 0 || rtspSourceCount == 0 else {
            appendLog("動画/フォルダとRTSPカメラは同時に実行できません。どちらかを選んでください。", level: .warning)
            selection = .capture
            return
        }
        guard rtspSourceCount <= 1 else {
            appendLog("RTSPカメラは一度に1台だけ実行できます。", level: .warning)
            selection = .capture
            return
        }

        selection = .analysis
        runState = .preparing
        progress = nil
        queueStatus = "入力を確認しています"
        saveSettings()

        let localPaths = sources.filter { $0.kind != .rtsp }.map(\.value)
        if localPaths.isEmpty, let rtsp = sources.first(where: { $0.kind == .rtsp })?.value {
            startRTSPRun(url: rtsp)
            return
        }

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
        runState = .cancelling
        queueStatus = "停止要求を送信しました"
        bridge.request("cancel") { [weak self] result in
            if case .failure(let error) = result {
                self?.appendLog(error.localizedDescription, level: .warning)
            }
        }
    }

    func saveSettings() {
        let localPaths = sources.filter { $0.kind != .rtsp }.map(\.value)
        let rtspURLs = sources.filter { $0.kind == .rtsp }.map(\.value)
        bridge.request(
            "save_settings",
            payload: [
                "settings": [
                    "folder_paths": localPaths,
                    "rtsp_urls": rtspURLs,
                    "meteor_save_path": meteorSavePath,
                    "not_meteor_save_path": notMeteorSavePath,
                    "concurrency": String(maxWorkers),
                    "interval": String(interval),
                    "duration": String(duration),
                    "date_folder_twilight_filter_enabled": twilightFilterEnabled,
                    "observation_latitude": String(latitude),
                    "observation_longitude": String(longitude),
                    "save_options": saveOptions,
                ],
            ]
        ) { [weak self] result in
            if case .failure(let error) = result {
                self?.appendLog("設定を保存できませんでした: \(error.localizedDescription)", level: .warning)
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
        let meteorPath = URL(fileURLWithPath: meteorSavePath)
        let notMeteorPath = URL(fileURLWithPath: notMeteorSavePath)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let items = WorkspaceStore.collectResults(meteorPath: meteorPath, notMeteorPath: notMeteorPath)
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                guard self.resultsRefreshGeneration == generation else { return }
                self.results = items
                if let selected = self.selectedResultID,
                   !items.contains(where: { $0.id == selected }) {
                    self.selectedResultID = nil
                }
            }
        }
    }

    func shutdown() {
        saveSettings()
        bridge.stop(after: 0.25)
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
            payload: runPayload(merging: ["url": url])
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
            "summaryConfig": [
                ["name": "Composite Image", "enabled": true, "duration": 1.0],
                ["name": "Annotated Image", "enabled": false, "duration": 2.0],
                ["name": "Full Size Video", "enabled": true],
                ["name": "Zoom Sequence", "enabled": false, "duration": 2.0],
                ["name": "Cutout Video", "enabled": true],
            ],
        ]
        for (key, value) in additional {
            payload[key] = value
        }
        return payload
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
            case "preparing": runState = .preparing
            case "running": runState = .running
            case "cancelling": runState = .cancelling
            case "completed":
                runState = .completed
                refreshResults()
            case "cancelled": runState = .cancelled
            case "failed": runState = .failed(envelope.payload["error"] as? String ?? "処理に失敗しました")
            default: runState = .idle
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
        if let options = settings["save_options"] as? [String: Any] {
            for key in saveOptions.keys {
                if let value = options[key] { saveOptions[key] = boolValue(value, default: saveOptions[key] ?? true) }
            }
        }
        if !restored.isEmpty {
            appendLog("前回の入力設定を復元しました。")
        }
        refreshResults()
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
