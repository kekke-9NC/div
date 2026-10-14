"""Explicit one-shot restart mode for the legacy application's launcher."""
import sys


def resume_rtsp_requested():
    return '--resume-rtsp' in sys.argv[1:]
