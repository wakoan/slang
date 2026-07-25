// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "g4",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(name: "g4", path: "Sources/g4")
    ]
)
