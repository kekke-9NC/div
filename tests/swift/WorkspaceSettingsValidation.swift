import Foundation
import MeteorDetectorCore

@main
struct WorkspaceSettingsValidation {
    @MainActor
    static func main() async {
        let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        let store = WorkspaceStore(rootURL: root)
        let deadline = Date().addingTimeInterval(5)
        while !store.settingsLoaded && store.settingsLoadError == nil && Date() < deadline {
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        if CommandLine.arguments.last == "invalid" {
            require(!store.settingsLoaded, "invalid settings must not enable saving defaults")
            require(store.settingsLoadError != nil, "invalid settings must display a load error")
            var failed = false
            store.saveSettings { result in if case .failure = result { failed = true } }
            require(failed, "saving before a successful load must fail")
        } else {
            require(store.settingsLoaded, "settings must finish loading")
            require(store.connection == .connected, "backend connection")
            require(store.duration == 60, "legacy duration above the previous UI cap")
            require(store.periodicScanInterval == 1, "legacy one-second scan interval")
            require(store.selectedModelPath == "/missing/selected.pth", "missing selected models must not be reset")
            require(store.plateSolveEnabled, "missing calibration must not erase the on/off preference")
            require(store.plateSolveConfigurationIsValid, "local calibration may be created from inputs without an existing file")
            let legacyCalibration = root.appendingPathComponent("legacy-camera.wcs").path
            if FileManager.default.fileExists(atPath: legacyCalibration) {
                require(store.plateSolvePath == legacyCalibration, "calibration_path must be used when other saved paths are missing")
            }
            require(store.rtspFixedPatternEnabled, "missing correction must not erase the on/off preference")
            require(store.legacyString(["camera_model_method"]) == "静止画プレートソルブ", "camera model method")
            store.setLegacyValue("0.8", at: ["advanced_settings", "meteor_prob"])
            var completed = false
            store.saveSettings { result in
                if case .failure(let error) = result { fputs("\(error)\n", stderr); exit(1) }
                completed = true
            }
            while !completed && Date() < deadline { try? await Task.sleep(nanoseconds: 10_000_000) }
            require(completed, "save completion")
        }
        store.shutdown()
        try? await Task.sleep(nanoseconds: 200_000_000)
        print("WorkspaceSettingsValidation: ok")
    }

    private static func require(_ condition: @autoclosure () -> Bool, _ message: String) {
        if !condition() { fputs("validation failed: \(message)\n", stderr); exit(1) }
    }
}
