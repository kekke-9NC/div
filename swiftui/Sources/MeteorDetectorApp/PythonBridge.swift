import Foundation

struct BridgeEnvelope {
    let type: String
    let id: String?
    let event: String?
    let ok: Bool
    let payload: [String: Any]
    let error: String?
}

enum BridgeError: LocalizedError {
    case unavailable(String)
    case encoding(String)
    case backend(String)

    var errorDescription: String? {
        switch self {
        case .unavailable(let message), .encoding(let message), .backend(let message):
            return message
        }
    }
}

@MainActor
final class PythonBridge {
    typealias Completion = (Result<[String: Any], BridgeError>) -> Void

    let rootURL: URL
    var onEnvelope: ((BridgeEnvelope) -> Void)?
    var onDiagnostic: ((String) -> Void)?

    private var process: Process?
    private var inputPipe: Pipe?
    private var outputHandle: FileHandle?
    private var errorHandle: FileHandle?
    private var outputBuffer = Data()
    private var pending: [String: Completion] = [:]
    private var startupError: BridgeError?
    private var scheduledStop: DispatchWorkItem?

    init(rootURL: URL) {
        self.rootURL = rootURL
    }

    var isRunning: Bool {
        process?.isRunning == true
    }

    func start() {
        scheduledStop?.cancel()
        scheduledStop = nil
        guard process?.isRunning != true else { return }
        outputBuffer.removeAll(keepingCapacity: true)
        guard let pythonURL = resolvePython() else {
            let error = BridgeError.unavailable(
                "Python実行環境が見つかりません。METEOR_PYTHONまたは.venv-macを確認してください。"
            )
            startupError = error
            onDiagnostic?(error.localizedDescription)
            return
        }

        let scriptURL = rootURL.appendingPathComponent("swift_backend.py")
        guard FileManager.default.fileExists(atPath: scriptURL.path) else {
            let error = BridgeError.unavailable("SwiftUI用のPythonブリッジが見つかりません: \(scriptURL.path)")
            startupError = error
            onDiagnostic?(error.localizedDescription)
            return
        }

        let input = Pipe()
        let output = Pipe()
        let errorOutput = Pipe()
        let child = Process()
        child.executableURL = pythonURL
        child.arguments = [scriptURL.path, "--stdio", "--root", rootURL.path]
        child.currentDirectoryURL = rootURL
        var environment = ProcessInfo.processInfo.environment
        environment["METEOR_DETECTOR_ROOT"] = rootURL.path
        environment["PYTHONUNBUFFERED"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        child.environment = environment
        child.standardInput = input
        child.standardOutput = output
        child.standardError = errorOutput

        let stdout = output.fileHandleForReading
        let stderr = errorOutput.fileHandleForReading
        stdout.readabilityHandler = { [weak self] handle in
            do {
                guard let data = try handle.read(upToCount: 64 * 1024), !data.isEmpty else { return }
                Task { @MainActor [weak self] in
                    self?.consume(data)
                }
            } catch {
                Task { @MainActor [weak self] in
                    self?.onDiagnostic?("Python出力の読み取りに失敗しました: \(error.localizedDescription)")
                }
            }
        }
        stderr.readabilityHandler = { [weak self] handle in
            do {
                guard let data = try handle.read(upToCount: 64 * 1024), !data.isEmpty else { return }
                let message = String(decoding: data, as: UTF8.self)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !message.isEmpty else { return }
                Task { @MainActor [weak self] in
                    self?.onDiagnostic?(message)
                }
            } catch {
                // The JSON stdout channel remains authoritative for UI state.
            }
        }

        child.terminationHandler = { [weak self] terminatedProcess in
            Task { @MainActor [weak self] in
                guard let self else { return }
                guard self.process === terminatedProcess else { return }
                self.outputHandle?.readabilityHandler = nil
                self.errorHandle?.readabilityHandler = nil
                self.failPendingRequests(
                    .unavailable("Python処理エンジンが終了したため、保留中の操作を完了できませんでした。")
                )
                if terminatedProcess.terminationStatus != 0 && self.startupError == nil {
                    self.onDiagnostic?("Python処理エンジンが終了しました (code \(terminatedProcess.terminationStatus))")
                }
            }
        }

        do {
            try child.run()
        } catch {
            let bridgeError = BridgeError.unavailable("Python処理エンジンを起動できませんでした: \(error.localizedDescription)")
            startupError = bridgeError
            onDiagnostic?(bridgeError.localizedDescription)
            return
        }

        process = child
        inputPipe = input
        outputHandle = stdout
        errorHandle = stderr
        startupError = nil
    }

    func stop(after delay: TimeInterval = 0) {
        if delay > 0 {
            scheduledStop?.cancel()
            let work = DispatchWorkItem { [weak self] in
                self?.scheduledStop = nil
                self?.stop()
            }
            scheduledStop = work
            DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: work)
            return
        }
        scheduledStop?.cancel()
        scheduledStop = nil
        outputHandle?.readabilityHandler = nil
        errorHandle?.readabilityHandler = nil
        if process?.isRunning == true {
            process?.terminate()
        }
        process = nil
        inputPipe = nil
        outputHandle = nil
        errorHandle = nil
        failPendingRequests(.unavailable("Python処理エンジンを停止しました。"))
    }

    func request(
        _ command: String,
        payload: [String: Any] = [:],
        completion: Completion? = nil
    ) {
        if process?.isRunning != true {
            start()
        }
        guard let inputPipe, process?.isRunning == true else {
            completion?(.failure(startupError ?? .unavailable("Python処理エンジンに接続できません。")))
            return
        }

        let requestID = UUID().uuidString
        if let completion {
            pending[requestID] = completion
        }
        let request: [String: Any] = [
            "id": requestID,
            "command": command,
            "payload": payload,
        ]
        do {
            let data = try JSONSerialization.data(withJSONObject: request, options: [])
            var line = data
            line.append(0x0A)
            try inputPipe.fileHandleForWriting.write(contentsOf: line)
        } catch {
            pending.removeValue(forKey: requestID)
            completion?(.failure(.encoding("Pythonへのリクエスト送信に失敗しました: \(error.localizedDescription)")))
        }
    }

    private func consume(_ data: Data) {
        outputBuffer.append(data)
        while let newline = outputBuffer.firstIndex(of: 0x0A) {
            let lineData = outputBuffer.prefix(upTo: newline)
            outputBuffer.removeSubrange(...newline)
            guard !lineData.isEmpty else { continue }
            do {
                guard let object = try JSONSerialization.jsonObject(with: lineData, options: []) as? [String: Any] else {
                    failProtocol("Pythonからオブジェクト形式ではない応答を受信しました。")
                    continue
                }
                let envelope = BridgeEnvelope(
                    type: object["type"] as? String ?? "",
                    id: object["id"] as? String,
                    event: object["event"] as? String,
                    ok: object["ok"] as? Bool ?? true,
                    payload: object["payload"] as? [String: Any] ?? [:],
                    error: object["error"] as? String
                )
                if envelope.type == "response", let id = envelope.id, let completion = pending.removeValue(forKey: id) {
                    if envelope.ok {
                        completion(.success(envelope.payload))
                    } else {
                        completion(.failure(.backend(envelope.error ?? "Python処理エンジンでエラーが発生しました。")))
                    }
                }
                onEnvelope?(envelope)
            } catch {
                failProtocol("Pythonから不正な応答を受信しました: \(error.localizedDescription)")
                return
            }
        }
    }

    private func failProtocol(_ message: String) {
        onDiagnostic?(message)
        failPendingRequests(.backend(message))
    }

    private func failPendingRequests(_ error: BridgeError) {
        let callbacks = pending.values
        pending.removeAll()
        callbacks.forEach { $0(.failure(error)) }
    }

    private func resolvePython() -> URL? {
        let fileManager = FileManager.default
        var candidates: [String] = []
        if let configured = ProcessInfo.processInfo.environment["METEOR_PYTHON"], !configured.isEmpty {
            candidates.append(configured)
        }
        candidates.append(rootURL.appendingPathComponent(".venv-mac/bin/python").path)
        candidates.append(rootURL.appendingPathComponent(".venv/bin/python").path)
        candidates.append("/opt/homebrew/bin/python3")
        candidates.append("/usr/local/bin/python3")
        candidates.append("/usr/bin/python3")
        return candidates
            .map { URL(fileURLWithPath: $0) }
            .first { fileManager.isExecutableFile(atPath: $0.path) }
    }
}
