import ast
from pathlib import Path
from unittest.mock import Mock
import pytest
import runtime_scheduling


@pytest.mark.parametrize('value,expected',[(None,.001),('.002',.002),('.005',.005),('invalid',.005),('nan',.005),('0',.005),('.00001',.005)])
def test_validated_owner_policy(monkeypatch,value,expected):
    monkeypatch.setattr(runtime_scheduling.sys,'platform','darwin')
    monkeypatch.setattr(runtime_scheduling.sys,'getswitchinterval',lambda:.005)
    setter=Mock();monkeypatch.setattr(runtime_scheduling.sys,'setswitchinterval',setter)
    if value is None:monkeypatch.delenv('METEOR_GIL_SWITCH_INTERVAL',raising=False)
    else:monkeypatch.setenv('METEOR_GIL_SWITCH_INTERVAL',value)
    assert runtime_scheduling.configure_native_return_scheduling()==expected
    if expected<.005:setter.assert_called_once_with(expected)
    else:setter.assert_not_called()


def test_non_mac_and_already_short_intervals_are_preserved(monkeypatch):
    monkeypatch.setattr(runtime_scheduling.sys,'getswitchinterval',lambda:.0008)
    setter=Mock();monkeypatch.setattr(runtime_scheduling.sys,'setswitchinterval',setter)
    monkeypatch.setattr(runtime_scheduling.sys,'platform','linux')
    assert runtime_scheduling.configure_native_return_scheduling()==.0008
    monkeypatch.setattr(runtime_scheduling.sys,'platform','darwin');monkeypatch.delenv('METEOR_GIL_SWITCH_INTERVAL',raising=False)
    assert runtime_scheduling.configure_native_return_scheduling()==.0008
    setter.assert_not_called()


def test_scheduler_only_runs_after_freeze_support_in_owner_entrypoint():
    tree=ast.parse(Path('main_gui.py').read_text())
    guard=next(node for node in tree.body if isinstance(node,ast.If) and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='configure_native_return_scheduling' for n in ast.walk(node)))
    assert ast.unparse(guard.test)=="__name__ == '__main__'"
    calls=[ast.unparse(node) for node in guard.body]
    assert calls.index('multiprocessing.freeze_support()')<calls.index('configure_native_return_scheduling()')
