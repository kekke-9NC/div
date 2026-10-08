import Foundation

@main
struct PythonBridgeValidation {
    @MainActor
    static func main() async {
        let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        let mode = CommandLine.arguments[2]
        let bridge = PythonBridge(rootURL: root)
        var responses: [String: Result<[String: Any], BridgeError>] = [:]
        var diagnostics: [String] = []
        bridge.onDiagnostic = { diagnostics.append($0) }
        bridge.request("ping") { responses["ping"] = $0 }
        bridge.request("load_settings") { responses["settings"] = $0 }

        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if responses.count == 2 && (mode != "diagnostic" || !diagnostics.isEmpty) { break }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        for (command, result) in responses {
            if case .failure(let error) = result {
                fputs("\(command): \(error.localizedDescription)\n", stderr)
            }
        }
        if responses.count < 2 { fputs("diagnostics: \(diagnostics)\n", stderr) }
        require(responses.count == 2, "short responses must arrive without waiting for EOF")
        let ping = try? responses["ping"]?.get()
        let settings = try? responses["settings"]?.get()["settings"] as? [String: Any]
        require(ping?["name"] as? String == "Meteor Detector", "ping response")
        require(settings?["interval"] as? Double == 2.5, "saved settings response")
        if mode == "fragmented" {
            require(settings?["label"] as? String == String(repeating: "流星🌠", count: 20_000), "fragmented UTF-8 response")
        }
        if mode == "diagnostic" {
            require(diagnostics.joined().contains("診断メッセージ"), "short diagnostics must arrive without EOF")
        }
        require(bridge.isRunning, "backend must remain alive after responding")
        bridge.stop()
        print("PythonBridgeValidation: \(mode) ok")
    }

    private static func require(_ condition: @autoclosure () -> Bool, _ message: String) {
        if !condition() {
            fputs("validation failed: \(message)\n", stderr)
            exit(1)
        }
    }
}
