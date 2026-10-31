import pytest

from crmn_agent.analysis import paired_bootstrap, wilson
from crmn_agent.runner import validate_config


def episode(seed, success):
    return {'benchmark': 'babyai', 'task': 'goto', 'seed': seed, 'success': success}


def test_wilson_respects_small_sample_uncertainty():
    lo, hi = wilson(5, 5)
    assert lo < 0.6
    assert hi == pytest.approx(1)


def test_pairs_are_matched_by_task_and_seed_not_row_order():
    a = [episode(1, True), episode(2, False)]
    b = [episode(2, False), episode(1, False)]
    assert paired_bootstrap(a, b, samples=100)['difference'] == 0.5


def test_duplicate_episodes_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap([episode(1, True), episode(1, False)], [episode(1, True)])


def test_unpaired_tasks_not_silently_counted():
    result = paired_bootstrap([episode(1, True), episode(2, True)], [episode(1, False)], samples=100)
    assert result['paired_n'] == 1
    assert result['a_only'] == 1


def test_final_evaluation_requires_frozen_task_manifest():
    with pytest.raises(ValueError, match='manifest'):
        validate_config({'split': 'test', 'seeds': [1], 'variants': ['crmn'], 'max_steps': 5,
                         'provider': {'backend': 'mlx'}})

