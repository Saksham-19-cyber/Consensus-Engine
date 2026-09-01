import pytest
from src.protocol.rules import ProtocolRules


def test_terminate_on_agreed():
    rules = ProtocolRules()
    should_stop, reason = rules.should_terminate({"status": "agreed", "round_number": 3})
    assert should_stop
    assert reason == "agreed"


def test_terminate_on_impasse():
    rules = ProtocolRules()
    should_stop, reason = rules.should_terminate({"status": "impasse", "round_number": 5})
    assert should_stop
    assert reason == "impasse"


def test_terminate_on_max_rounds():
    rules = ProtocolRules(max_rounds=10)
    should_stop, reason = rules.should_terminate({"status": "in_progress", "round_number": 10})
    assert should_stop
    assert reason == "max_rounds"


def test_continue_in_progress():
    rules = ProtocolRules(max_rounds=10)
    should_stop, reason = rules.should_terminate({"status": "in_progress", "round_number": 5})
    assert not should_stop
    assert reason == "in_progress"
