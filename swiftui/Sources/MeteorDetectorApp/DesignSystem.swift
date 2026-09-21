import AppKit
import SwiftUI

enum AppTheme {
    static let canvas = Color(nsColor: NSColor(calibratedRed: 0.035, green: 0.047, blue: 0.075, alpha: 1))
    static let surface = Color(nsColor: NSColor(calibratedRed: 0.065, green: 0.090, blue: 0.135, alpha: 1))
    static let surfaceRaised = Color(nsColor: NSColor(calibratedRed: 0.090, green: 0.125, blue: 0.185, alpha: 1))
    static let surfaceSelected = Color(nsColor: NSColor(calibratedRed: 0.145, green: 0.220, blue: 0.340, alpha: 1))
    static let border = Color(nsColor: NSColor(calibratedRed: 0.170, green: 0.230, blue: 0.330, alpha: 1))
    static let text = Color(nsColor: NSColor(calibratedRed: 0.950, green: 0.970, blue: 1.000, alpha: 1))
    static let secondaryText = Color(nsColor: NSColor(calibratedRed: 0.620, green: 0.690, blue: 0.790, alpha: 1))
    static let tertiaryText = Color(nsColor: NSColor(calibratedRed: 0.420, green: 0.490, blue: 0.600, alpha: 1))
    static let accent = Color(nsColor: NSColor(calibratedRed: 0.400, green: 0.650, blue: 1.000, alpha: 1))
    static let cyan = Color(nsColor: NSColor(calibratedRed: 0.360, green: 0.840, blue: 0.930, alpha: 1))
    static let success = Color(nsColor: NSColor(calibratedRed: 0.360, green: 0.870, blue: 0.650, alpha: 1))
    static let warning = Color(nsColor: NSColor(calibratedRed: 0.980, green: 0.750, blue: 0.360, alpha: 1))
    static let danger = Color(nsColor: NSColor(calibratedRed: 1.000, green: 0.410, blue: 0.500, alpha: 1))
}

struct PageHeader: View {
    let eyebrow: String
    let title: String
    let subtitle: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(eyebrow.uppercased())
                .font(.system(size: 11, weight: .bold, design: .rounded))
                .tracking(1.6)
                .foregroundStyle(AppTheme.accent)
            Text(title)
                .font(.system(size: 30, weight: .bold, design: .rounded))
                .foregroundStyle(AppTheme.text)
            Text(subtitle)
                .font(.system(size: 14))
                .foregroundStyle(AppTheme.secondaryText)
        }
    }
}

struct GlassCard<Content: View>: View {
    let padding: CGFloat
    @ViewBuilder let content: Content

    init(padding: CGFloat = 20, @ViewBuilder content: () -> Content) {
        self.padding = padding
        self.content = content()
    }

    var body: some View {
        content
            .padding(padding)
            .background {
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(AppTheme.surface.opacity(0.96))
                    .overlay {
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .stroke(AppTheme.border.opacity(0.9), lineWidth: 1)
                    }
            }
    }
}

struct StatusPill: View {
    let title: String
    let color: Color
    var symbol: String = "circle.fill"

    var body: some View {
        Label(title, systemImage: symbol)
            .font(.system(size: 12, weight: .semibold, design: .rounded))
            .foregroundStyle(color)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(color.opacity(0.13), in: Capsule())
    }
}

struct SectionTitle: View {
    let title: String
    let subtitle: String?

    init(_ title: String, subtitle: String? = nil) {
        self.title = title
        self.subtitle = subtitle
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(size: 16, weight: .bold, design: .rounded))
                .foregroundStyle(AppTheme.text)
            if let subtitle {
                Text(subtitle)
                    .font(.system(size: 12))
                    .foregroundStyle(AppTheme.secondaryText)
            }
        }
    }
}

struct EmptyState: View {
    let symbol: String
    let title: String
    let message: String

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: symbol)
                .font(.system(size: 26, weight: .medium))
                .foregroundStyle(AppTheme.accent)
            Text(title)
                .font(.system(size: 15, weight: .semibold, design: .rounded))
                .foregroundStyle(AppTheme.text)
            Text(message)
                .font(.system(size: 13))
                .foregroundStyle(AppTheme.secondaryText)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 28)
    }
}
