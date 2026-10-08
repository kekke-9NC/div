import Foundation

/// Retain settings that have no dedicated control, including nested extension keys.
public struct LegacySettingsDocument {
    public private(set) var values: [String: Any]

    public init(_ values: [String: Any] = [:]) {
        self.values = values
    }

    public func value(at path: [String]) -> Any? {
        var current: Any = values
        for key in path {
            guard let next = (current as? [String: Any])?[key] else { return nil }
            current = next
        }
        return current
    }

    public mutating func set(_ value: Any, at path: [String]) {
        guard let key = path.first else { return }
        if path.count == 1 {
            values[key] = value
        } else {
            var child = LegacySettingsDocument(values[key] as? [String: Any] ?? [:])
            child.set(value, at: Array(path.dropFirst()))
            values[key] = child.values
        }
    }

    public func merging(_ updates: [String: Any]) -> [String: Any] {
        var result = values
        for (key, value) in updates {
            if let nested = value as? [String: Any], let existing = result[key] as? [String: Any] {
                result[key] = LegacySettingsDocument(existing).merging(nested)
            } else {
                result[key] = value
            }
        }
        return result
    }
}
