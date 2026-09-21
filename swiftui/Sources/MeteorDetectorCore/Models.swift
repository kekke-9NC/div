import Foundation

public enum AppSection: String, CaseIterable, Identifiable, Hashable {
    case overview
    case capture
    case analysis
    case results
    case settings
    case activity

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .overview: return "概要"
        case .capture: return "入力ソース"
        case .analysis: return "検出と解析"
        case .results: return "結果"
        case .settings: return "設定"
        case .activity: return "アクティビティ"
        }
    }

    public var subtitle: String {
        switch self {
        case .overview: return "観測ワークスペース"
        case .capture: return "動画・フォルダ・RTSP"
        case .analysis: return "処理の開始と進捗"
        case .results: return "候補一覧とプレビュー"
        case .settings: return "保存先と処理条件"
        case .activity: return "イベントログと状態"
        }
    }

    public var symbol: String {
        switch self {
        case .overview: return "sparkles"
        case .capture: return "tray.and.arrow.down"
        case .analysis: return "scope"
        case .results: return "photo.on.rectangle.angled"
        case .settings: return "slider.horizontal.3"
        case .activity: return "waveform.path.ecg"
        }
    }
}

public enum SourceKind: String, Codable {
    case file
    case folder
    case rtsp

    public var title: String {
        switch self {
        case .file: return "動画ファイル"
        case .folder: return "フォルダ"
        case .rtsp: return "RTSPカメラ"
        }
    }

    public var symbol: String {
        switch self {
        case .file: return "film"
        case .folder: return "folder"
        case .rtsp: return "dot.radiowaves.left.and.right"
        }
    }
}

public struct InputSource: Identifiable, Hashable, Codable {
    public let id: UUID
    public let kind: SourceKind
    public let value: String

    public init(id: UUID = UUID(), kind: SourceKind, value: String) {
        self.id = id
        self.kind = kind
        self.value = value
    }

    public var displayName: String {
        if kind == .rtsp { return value }
        let url = URL(fileURLWithPath: value)
        return url.lastPathComponent.isEmpty ? value : url.lastPathComponent
    }

    public var detail: String {
        if kind == .rtsp { return "ネットワークストリーム" }
        return value
    }

    public var exists: Bool {
        kind == .rtsp || FileManager.default.fileExists(atPath: value)
    }
}

public enum BackendConnectionState: Equatable {
    case starting
    case connected
    case unavailable(String)

    public var title: String {
        switch self {
        case .starting: return "接続中"
        case .connected: return "処理エンジン接続済み"
        case .unavailable: return "処理エンジン未接続"
        }
    }
}

public enum RunState: Equatable {
    case idle
    case preparing
    case running
    case cancelling
    case completed
    case cancelled
    case failed(String)

    public var title: String {
        switch self {
        case .idle: return "待機中"
        case .preparing: return "準備中"
        case .running: return "解析中"
        case .cancelling: return "停止中"
        case .completed: return "完了"
        case .cancelled: return "キャンセル済み"
        case .failed: return "エラー"
        }
    }

    public var isActive: Bool {
        switch self {
        case .preparing, .running, .cancelling: return true
        default: return false
        }
    }
}

public struct PipelineProgress: Equatable {
    public var current: Int = 0
    public var total: Int = 0
    public var message: String = ""

    public init(current: Int = 0, total: Int = 0, message: String = "") {
        self.current = current
        self.total = total
        self.message = message
    }

    public var fraction: Double {
        guard total > 0 else { return 0 }
        return min(1, max(0, Double(current) / Double(total)))
    }
}

public struct ActivityEntry: Identifiable, Equatable {
    public let id = UUID()
    public let date: Date
    public let message: String
    public let level: Level

    public enum Level: Equatable {
        case info
        case warning
        case error
    }

    public init(date: Date, message: String, level: Level) {
        self.date = date
        self.message = message
        self.level = level
    }
}

public enum OutputCategory: String, CaseIterable, Identifiable, Hashable {
    case meteor
    case notMeteor

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .meteor: return "流星候補"
        case .notMeteor: return "非流星候補"
        }
    }

    public var symbol: String {
        switch self {
        case .meteor: return "sparkles"
        case .notMeteor: return "archivebox"
        }
    }
}

public enum OutputKind: String, Hashable {
    case image
    case video
    case data

    public var symbol: String {
        switch self {
        case .image: return "photo"
        case .video: return "film"
        case .data: return "doc.text"
        }
    }

    public static func from(fileExtension: String) -> OutputKind? {
        switch fileExtension.lowercased() {
        case "jpg", "jpeg", "png", "gif", "tif", "tiff", "webp": return .image
        case "mp4", "mov", "avi", "m4v", "mkv": return .video
        case "json", "txt", "csv": return .data
        default: return nil
        }
    }
}

public struct OutputItem: Identifiable, Hashable {
    public let id: String
    public let url: URL
    public let category: OutputCategory
    public let kind: OutputKind
    public let byteCount: Int64
    public let modifiedAt: Date

    public init(
        url: URL,
        category: OutputCategory,
        kind: OutputKind,
        byteCount: Int64 = 0,
        modifiedAt: Date = .distantPast
    ) {
        self.id = url.standardizedFileURL.path
        self.url = url
        self.category = category
        self.kind = kind
        self.byteCount = byteCount
        self.modifiedAt = modifiedAt
    }

    public var displayName: String { url.lastPathComponent }

    public var relativeDisplayName: String {
        let parent = url.deletingLastPathComponent().lastPathComponent
        return parent.isEmpty ? displayName : "\(parent) / \(displayName)"
    }

    public var sizeLabel: String {
        ByteCountFormatter.string(fromByteCount: byteCount, countStyle: .file)
    }
}

public struct LegacySettings {
    public var folderPaths: [String] = []
    public var rtspURLs: [String] = []
    public var meteorSavePath: String = ""
    public var notMeteorSavePath: String = ""
    public var concurrency: Int = 4
    public var interval: Double = 1
    public var duration: Double = 1
    public var twilightFilterEnabled: Bool = true
    public var latitude: Double = 35.0
    public var longitude: Double = 135.0
    public var saveOptions: [String: Bool] = [
        "video": true,
        "cutout": true,
        "full": false,
        "composite": true,
        "info": true,
        "summary": true,
        "full_video": false,
    ]

    public init() {}
}
