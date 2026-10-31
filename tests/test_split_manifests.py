import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('check_splits', Path(__file__).parents[1]/'scripts/check_splits.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_seed_difference_does_not_hide_goal_leakage(tmp_path):
    a, b = tmp_path/'train.json', tmp_path/'test.json'
    a.write_text(json.dumps({'split': 'train', 'tasks': {'textcraft/default/1': 'same_goal'}}))
    b.write_text(json.dumps({'split': 'test', 'tasks': {'textcraft/default/20001': 'same_goal'}}))
    with pytest.raises(ValueError, match='shared'):
        module.check([a, b])


def test_disjoint_tasks_accepted(tmp_path):
    a, b = tmp_path/'train.json', tmp_path/'test.json'
    a.write_text(json.dumps({'split': 'train', 'tasks': {'textcraft/default/1': 'goal_a'}}))
    b.write_text(json.dumps({'split': 'test', 'tasks': {'textcraft/default/20001': 'goal_b'}}))
    assert module.check([a, b])['test']['unique_tasks'] == 1
