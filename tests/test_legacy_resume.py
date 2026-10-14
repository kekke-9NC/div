import sys
import legacy_resume


def test_explicit_restart_flag_only(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['main_gui.py'])
    assert not legacy_resume.resume_rtsp_requested()
    monkeypatch.setattr(sys, 'argv', ['main_gui.py', '--resume-rtsp'])
    assert legacy_resume.resume_rtsp_requested()
    monkeypatch.setattr(sys, 'argv', ['--resume-rtsp'])
    assert not legacy_resume.resume_rtsp_requested()
    monkeypatch.setattr(sys, 'argv', ['main_gui.py', '--resume-rtsp=false'])
    assert not legacy_resume.resume_rtsp_requested()
