"""Exercise the production Swift bridge with live pipes, without requiring XCTest."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def swift_bridge_validation(tmp_path_factory):
    if sys.platform != "darwin" or not shutil.which("swiftc"):
        pytest.skip("The macOS Swift compiler is required")
    executable = tmp_path_factory.mktemp("swift-bridge-build") / "validate"
    subprocess.run(
        [
            "swiftc", "-parse-as-library",
            str(ROOT / "swiftui/Sources/MeteorDetectorApp/PythonBridge.swift"),
            str(ROOT / "tests/swift/PythonBridgeValidation.swift"),
            "-o", str(executable),
        ],
        check=True, capture_output=True, text=True, timeout=120,
    )
    return executable


@pytest.mark.parametrize("mode", ["short", "diagnostic", "fragmented", "real_backend"])
def test_live_bridge_responses(swift_bridge_validation, tmp_path, monkeypatch, mode):
    # The backend deliberately stays alive with stdin open after each response.
    # subprocess.run(input=...) tests close stdin and cannot expose this deadlock.
    monkeypatch.setenv("METEOR_PYTHON", sys.executable)
    if mode == "real_backend":
        for name in ("swift_backend.py", "process_lock.py"):
            shutil.copy2(ROOT / name, tmp_path / name)
        (tmp_path / "app_settings.json").write_text(
            json.dumps({"interval": 2.5}), encoding="utf-8"
        )
    else:
        (tmp_path / "swift_backend.py").write_text(
            "import json, sys, time\n"
            f"mode = {mode!r}\n"
            "if mode == 'diagnostic':\n"
            "    print('診断メッセージ', file=sys.stderr, flush=True)\n"
            "for line in sys.stdin:\n"
            "    request = json.loads(line)\n"
            "    settings = {'interval': 2.5}\n"
            "    if mode == 'fragmented':\n"
            "        settings['label'] = '流星🌠' * 20000\n"
            "    payload = {'name': 'Meteor Detector'} if request['command'] == 'ping' else {'settings': settings}\n"
            "    data = (json.dumps({'type': 'response', 'id': request['id'], 'ok': True, 'payload': payload}, ensure_ascii=False) + chr(10)).encode()\n"
            "    if mode == 'fragmented':\n"
            "        for start in range(0, len(data), 997):\n"
            "            sys.stdout.buffer.write(data[start:start + 997])\n"
            "            sys.stdout.buffer.flush()\n"
            "            time.sleep(0.001)\n"
            "    else:\n"
            "        sys.stdout.buffer.write(data)\n"
            "        sys.stdout.buffer.flush()\n",
            encoding="utf-8",
        )
    result = subprocess.run(
        [str(swift_bridge_validation), str(tmp_path), mode],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
