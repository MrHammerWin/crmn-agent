import json

import pytest

from crmn_agent.codex_provider import reserve_call
from crmn_agent.providers import BudgetExceeded


def test_subscription_limit_survives_new_process_state(tmp_path):
    ledger = tmp_path/'budget.json'
    for i in range(30):
        assert reserve_call(ledger) == i + 1
    with pytest.raises(BudgetExceeded):
        reserve_call(ledger)
    assert json.loads(ledger.read_text())['used'] == 30


def test_subscription_authorization_cannot_be_expanded(tmp_path):
    with pytest.raises(ValueError):
        reserve_call(tmp_path/'budget.json', 31)


def test_reduced_limit_cannot_be_reset_by_different_config(tmp_path):
    ledger = tmp_path/'budget.json'
    reserve_call(ledger, 1)
    with pytest.raises(BudgetExceeded):
        reserve_call(ledger, 30)
