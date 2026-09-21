// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "MeteorDetectorSwiftUI",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(
            name: "MeteorDetectorSwiftUI",
            targets: ["MeteorDetectorSwiftUI"]
        ),
        .library(
            name: "MeteorDetectorCore",
            targets: ["MeteorDetectorCore"]
        ),
        .executable(
            name: "MeteorDetectorCoreValidation",
            targets: ["MeteorDetectorCoreValidation"]
        ),
    ],
    targets: [
        .target(
            name: "MeteorDetectorCore",
            path: "Sources/MeteorDetectorCore"
        ),
        .executableTarget(
            name: "MeteorDetectorSwiftUI",
            dependencies: ["MeteorDetectorCore"],
            path: "Sources/MeteorDetectorApp"
        ),
        .executableTarget(
            name: "MeteorDetectorCoreValidation",
            dependencies: ["MeteorDetectorCore"],
            path: "Sources/MeteorDetectorCoreValidation"
        ),
    ]
)
