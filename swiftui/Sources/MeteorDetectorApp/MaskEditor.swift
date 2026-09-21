import AVFoundation
import AppKit
import SwiftUI
import UniformTypeIdentifiers

enum MaskStrokeMode: String {
    case exclude
    case restore
}

struct MaskStroke: Identifiable {
    let id = UUID()
    let mode: MaskStrokeMode
    var points: [CGPoint]
}

@MainActor
final class MaskEditorState: ObservableObject {
    @Published var isPresented = false
    @Published private(set) var image: NSImage?
    @Published private(set) var pixelSize = CGSize(width: 1920, height: 1080)
    @Published var strokes: [MaskStroke] = []
    @Published var brushSize = 0.04
    @Published var mode: MaskStrokeMode = .exclude
    @Published private(set) var isLoading = false
    @Published var isSaving = false
    @Published var status = "動画の先頭フレームを選択してください"

    private var loadGeneration = 0
    private var saveGeneration = 0

    var serializedStrokes: [[String: Any]] {
        strokes.map { stroke in
            [
                "mode": stroke.mode.rawValue,
                "points": stroke.points.map { [$0.x, $0.y] },
            ]
        }
    }

    func present() {
        loadGeneration &+= 1
        saveGeneration &+= 1
        image = nil
        strokes.removeAll()
        isLoading = false
        isSaving = false
        brushSize = 0.04
        mode = .exclude
        status = "動画の先頭フレームを選択してください"
        isPresented = true
    }

    func loadVideo(_ url: URL) {
        guard !isSaving else { return }
        loadGeneration &+= 1
        let generation = loadGeneration
        saveGeneration &+= 1
        isSaving = false
        image = nil
        strokes.removeAll()
        isLoading = true
        status = "先頭フレームを読み込んでいます…"
        let assetURL = url
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                let asset = AVAsset(url: assetURL)
                let generator = AVAssetImageGenerator(asset: asset)
                generator.appliesPreferredTrackTransform = true
                generator.requestedTimeToleranceBefore = .zero
                generator.requestedTimeToleranceAfter = .zero
                let cgImage = try generator.copyCGImage(
                    at: CMTime(seconds: 0, preferredTimescale: 600),
                    actualTime: nil
                )
                let image = NSImage(
                    cgImage: cgImage,
                    size: NSSize(width: cgImage.width, height: cgImage.height)
                )
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    guard self.isPresented, self.loadGeneration == generation else { return }
                    self.image = image
                    self.pixelSize = CGSize(width: cgImage.width, height: cgImage.height)
                    self.strokes.removeAll()
                    self.isLoading = false
                    self.status = "左ドラッグで除外領域を塗ります"
                }
            } catch {
                Task { @MainActor [weak self] in
                    guard let self,
                          self.isPresented,
                          self.loadGeneration == generation else { return }
                    self.isLoading = false
                    self.status = "フレームを読み込めませんでした: \(error.localizedDescription)"
                }
            }
        }
    }

    @discardableResult
    func beginSave() -> Int {
        saveGeneration &+= 1
        isSaving = true
        return saveGeneration
    }

    func acceptsSave(_ generation: Int) -> Bool {
        isPresented && saveGeneration == generation
    }

    func appendPoint(_ location: CGPoint, in size: CGSize) {
        guard !isSaving else { return }
        guard size.width > 0, size.height > 0 else { return }
        let point = CGPoint(
            x: max(0, min(1, location.x / size.width)),
            y: max(0, min(1, location.y / size.height))
        )
        if strokes.last?.mode == mode {
            strokes[strokes.count - 1].points.append(point)
        } else {
            strokes.append(MaskStroke(mode: mode, points: [point]))
        }
    }

    func undo() {
        guard !strokes.isEmpty else { return }
        strokes.removeLast()
    }

    func clear() {
        strokes.removeAll()
    }

    func close() {
        guard !isSaving else { return }
        loadGeneration &+= 1
        saveGeneration &+= 1
        isLoading = false
        isSaving = false
        isPresented = false
    }
}

struct MaskEditorView: View {
    @ObservedObject var editor: MaskEditorState
    @EnvironmentObject private var store: WorkspaceStore

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("検出マスクエディタ")
                        .font(.system(size: 20, weight: .bold, design: .rounded))
                        .foregroundStyle(AppTheme.text)
                    Text("塗った領域を解析から除外します。保存形式は旧UI互換の app_masks.npz です。")
                        .font(.system(size: 12))
                        .foregroundStyle(AppTheme.secondaryText)
                }
                Spacer()
                Button("動画を選択") { chooseVideo() }
                    .buttonStyle(.bordered)
                    .disabled(editor.isSaving)
            }

            HStack(spacing: 12) {
                Picker("描画モード", selection: $editor.mode) {
                    Text("除外を塗る").tag(MaskStrokeMode.exclude)
                    Text("復元する").tag(MaskStrokeMode.restore)
                }
                .pickerStyle(.segmented)
                .disabled(editor.isSaving)
                Slider(value: $editor.brushSize, in: 0.01...0.20)
                    .frame(width: 180)
                    .disabled(editor.isSaving)
                Text("ブラシ \(Int(editor.brushSize * 100))%")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(AppTheme.secondaryText)
                Spacer()
                Button("元に戻す") { editor.undo() }
                    .buttonStyle(.borderless)
                    .disabled(editor.strokes.isEmpty || editor.isSaving)
                Button("クリア", role: .destructive) { editor.clear() }
                    .buttonStyle(.borderless)
                    .disabled(editor.strokes.isEmpty || editor.isSaving)
            }

            if editor.isLoading {
                ProgressView(editor.status)
                    .frame(maxWidth: .infinity, minHeight: 420)
            } else if editor.image != nil {
                drawingCanvas
            } else {
                EmptyState(
                    symbol: "pencil.and.outline",
                    title: "背景動画を選択してください",
                    message: editor.status
                )
                .frame(maxWidth: .infinity, minHeight: 420)
                .background(AppTheme.canvas, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            }

            HStack {
                Label(editor.status, systemImage: editor.isSaving ? "arrow.triangle.2.circlepath" : "info.circle")
                    .font(.system(size: 11))
                    .foregroundStyle(AppTheme.secondaryText)
                Spacer()
                Button("キャンセル") { editor.close() }
                    .buttonStyle(.bordered)
                    .disabled(editor.isSaving)
                Button {
                    saveMask()
                } label: {
                    if editor.isSaving {
                        ProgressView().controlSize(.small)
                        Text("保存中…")
                    } else {
                        Label("マスクを保存", systemImage: "checkmark")
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(AppTheme.accent)
                .disabled(editor.image == nil || editor.isSaving)
            }
        }
        .padding(24)
        .frame(minWidth: 900, minHeight: 660)
        .background(AppTheme.canvas)
        .interactiveDismissDisabled(editor.isSaving)
    }

    private var drawingCanvas: some View {
        GeometryReader { proxy in
            ZStack {
                if let image = editor.image {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFill()
                }
                Canvas { context, size in
                    for stroke in editor.strokes {
                        guard let first = stroke.points.first else { continue }
                        var path = Path()
                        path.move(to: CGPoint(x: first.x * size.width, y: first.y * size.height))
                        for point in stroke.points.dropFirst() {
                            path.addLine(to: CGPoint(x: point.x * size.width, y: point.y * size.height))
                        }
                        let color = stroke.mode == .exclude ? Color.red.opacity(0.42) : Color.green.opacity(0.42)
                        let width = max(4, editor.brushSize * min(size.width, size.height))
                        context.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: width, lineCap: .round, lineJoin: .round))
                    }
                }
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            editor.appendPoint(value.location, in: proxy.size)
                        }
                )
            }
            .clipped()
            .background(Color.black)
        }
        .aspectRatio(editor.pixelSize.width / max(1, editor.pixelSize.height), contentMode: .fit)
        .frame(maxHeight: 470)
        .background(Color.black, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private func chooseVideo() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.movie]
        panel.prompt = "選択"
        if panel.runModal() == .OK, let url = panel.url {
            editor.loadVideo(url)
        }
    }

    private func saveMask() {
        let generation = editor.beginSave()
        store.saveMaskFromStrokes(
            width: max(1, Int(editor.pixelSize.width)),
            height: max(1, Int(editor.pixelSize.height)),
            brushSize: editor.brushSize,
            strokes: editor.serializedStrokes
        ) { result in
            guard editor.acceptsSave(generation) else { return }
            editor.isSaving = false
            switch result {
            case .success:
                editor.status = "検出マスクを保存しました"
                editor.close()
            case .failure(let error):
                editor.status = error.localizedDescription
            }
        }
    }
}
