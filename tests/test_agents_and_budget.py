import json

import pytest

from crmn_agent.agents import Agent
from crmn_agent.providers import BudgetExceeded, ModelOutputError, Provider


class ScriptedProvider(Provider):
    """UNIT TEST FIXTURE ONLY. Not exposed by the scientific evaluation CLI."""
    is_real_llm = False

    def __init__(self, config=None):
        super().__init__(config or {'backend': 'fixture', 'max_calls': 100})
        self.payloads = []

    def _generate(self, system, user, schema, cap):
        data = json.loads(user)
        self.payloads.append((schema.__name__, data))
        if schema.__name__ == 'Grounding':
            response = {'facts': [{'key': 'carry', 'value': 'red ball', 'evidence': 'red ball'}]}
        elif schema.__name__ == 'Representation':
            response = {'summary': 'Carrying red ball', 'facts': [{'key': 'carry', 'value': 'red ball'}],
                        'subgoals': ['move'], 'action': 'go forward', 'expected': ['moved'], 'value': 0.5}
        elif schema.__name__ == 'Reconstruction':
            response = {'facts': [{'key': 'carry', 'value': 'red ball'}]}
        elif schema.__name__ == 'Audit':
            response = {'verdicts': []}
        else:
            response = {'reflection': 'move', 'action': 'go forward'}
        return json.dumps(response), 100, 30


def test_decoder_has_no_observation_history_or_mission():
    provider = ScriptedProvider()
    agent = Agent(provider, 'crmn')
    turn = agent.act('You carry a red ball. SECRET_RAW_LINE', 'MISSION_SECRET', ['go forward'], 0)
    assert turn.action == 'go forward'
    for name, payload in provider.payloads:
        if name == 'Reconstruction':
            assert set(payload) == {'representation'}
            assert 'SECRET_RAW_LINE' not in json.dumps(payload)
            assert 'MISSION_SECRET' not in json.dumps(payload)
    assert provider.usage.calls == 4


@pytest.mark.parametrize('variant', ['react', 'reflact', 'naive'])
def test_prompt_baselines_share_action_interface(variant):
    agent = Agent(ScriptedProvider(), variant)
    assert agent.act('red ball', 'go', ['go forward'], 0).action == 'go forward'


def test_unavailable_action_is_not_replaced_with_oracle():
    with pytest.raises(ModelOutputError):
        Agent(ScriptedProvider(), 'react').act('red ball', 'go', ['turn left'], 0)


def test_call_budget_checked_before_generation():
    provider = ScriptedProvider({'backend': 'fixture', 'max_calls': 0})
    with pytest.raises(BudgetExceeded):
        Agent(provider, 'react').act('red ball', 'go', [], 0)
    assert provider.payloads == []


def test_zero_paid_budget_blocks_before_network():
    provider = ScriptedProvider({'backend': 'openai', 'max_usd': 0, 'allow_paid': False})
    with pytest.raises(BudgetExceeded):
        Agent(provider, 'react').act('red ball', 'go', [], 0)
    assert provider.payloads == []


def test_positive_budget_requires_separate_authorization_flag():
    config = {'backend': 'openai', 'max_usd': 10, 'allow_paid': False}
    with pytest.raises(BudgetExceeded):
        Agent(ScriptedProvider(config), 'react').act('red ball', 'go', [], 0)


def test_token_reservation_blocks_overshoot():
    provider = ScriptedProvider({'backend': 'fixture', 'max_total_tokens': 50})
    with pytest.raises(BudgetExceeded):
        Agent(provider, 'react').act('red ball', 'go', [], 0)
    assert not provider.payloads


def test_controller_history_does_not_cross_episodes():
    first = Agent(ScriptedProvider(), 'crmn')
    first.act('red ball', 'go', [], 0)
    second = Agent(ScriptedProvider(), 'crmn')
    assert second.history == []
    assert second.previous is None


def test_no_cycle_ablation_removes_decoder_calls():
    provider = ScriptedProvider()
    Agent(provider, 'no_cycle').act('red ball', 'go', [], 0)
    assert all(name != 'Reconstruction' for name, _ in provider.payloads)


def test_reward_prediction_error_uses_terminal_binary_reward():
    agent = Agent(ScriptedProvider(), 'crmn')
    agent.act('red ball', 'go', [], 0)
    agent.feedback(success=True, terminated=True)
    assert agent.last_reward_error == pytest.approx(0.5)
    assert agent.reward_ema[0] == pytest.approx(0.15)


def test_nonterminal_td_error_bootstraps_next_value():
    agent = Agent(ScriptedProvider(), 'crmn')
    agent.act('red ball', 'go', [], 0)
    agent.feedback(success=False, terminated=False)
    agent.act('red ball', 'go', [], 1)
    assert agent.last_reward_error == pytest.approx(-0.025)


def test_observation_anchors_are_literal_not_hallucinated_by_model():
    agent = Agent(ScriptedProvider(), 'crmn')
    turn = agent.act('blue key', 'go', [], 0)
    assert turn.diagnostics['anchors']
    assert all(a['evidence'] in 'blue key' for a in turn.diagnostics['anchors'])


def test_failed_api_call_reserves_potential_charge():
    class FailingProvider(ScriptedProvider):
        def _generate(self, system, user, schema, cap):
            raise TimeoutError('Do not log the exception body')
    config = {'backend': 'openai', 'max_usd': 1, 'allow_paid': True,
              'input_usd_per_million': 0.1, 'output_usd_per_million': 0.5}
    provider = FailingProvider(config)
    with pytest.raises(TimeoutError):
        Agent(provider, 'react').act('red ball', 'go', [], 0)
    assert provider.usage.estimated_usd > 0
    assert provider.usage.calls == 1

