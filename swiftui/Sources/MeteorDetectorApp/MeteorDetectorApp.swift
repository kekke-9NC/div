import AppKit
import SwiftUI

@main
struct MeteorDetectorApp: App {
    @StateObject private var store = WorkspaceStore()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(store)
                .background(AppTheme.canvas)
                .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in
                    store.shutdown()
                }
        }
        .commands {
            CommandGroup(after: .newItem) {
                Button("入力ソースを追加") {
                    store.selection = .capture
                }
                .keyboardShortcut("o", modifiers: [.command])
                Button("解析画面を開く") {
                    store.selection = .analysis
                }
                .keyboardShortcut("r", modifiers: [.command, .shift])
            }
            CommandGroup(after: .appSettings) {
                Button("設定を開く") {
                    store.selection = .settings
                }
                .keyboardShortcut(",", modifiers: [.command])
            }
        }
        .onChange(of: scenePhase) { _, phase in
            if (phase == .background || phase == .inactive) && store.settingsLoaded {
                store.saveSettings()
            }
        }
    }
}
