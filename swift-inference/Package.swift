// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "gemma-swift",
    platforms: [
        .macOS(.v12)
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "gemma",
            dependencies: [],
            resources: []
        )
    ]
)
