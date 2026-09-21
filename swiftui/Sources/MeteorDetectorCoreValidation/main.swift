import Foundation
import MeteorDetectorCore

func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    if !condition() {
        fputs("validation failed: \(message)\n", stderr)
        exit(1)
    }
}

let source = InputSource(kind: .file, value: "/tmp/meteor-sample.mp4")
require(source.displayName == "meteor-sample.mp4", "InputSource displayName")
require(!source.exists, "missing files must be reported as unavailable")
require(PipelineProgress(current: 2, total: 4).fraction == 0.5, "progress fraction")
require(PipelineProgress(current: 9, total: 4).fraction == 1, "progress upper bound")
require(LegacySettings().saveOptions["video"] == true, "legacy save defaults")
print("MeteorDetectorCoreValidation: ok")
