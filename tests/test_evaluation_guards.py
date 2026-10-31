import hashlib
import json

import pytest
from test_agents_and_budget import ScriptedProvider

from crmn_agent.agents import Agent
from crmn_agent.environments import tasks_for
from crmn_agent.providers import EpisodeBudgetExceeded
from crmn_agent.runner import code_hash, run, validate_config


def test_episode_budget_is_checked_before_model_generation():
    provider = ScriptedProvider()
    provider.begin_episode(token_limit=10)
    with pytest.raises(EpisodeBudgetExceeded):
        Agent(provider, 'react').act('wall', 'go', [], 0)
    assert provider.usage.calls == 0
    provider.begin_episode(token_limit=10000)
    assert Agent(provider, 'react').act('wall', 'go', [], 0).action == 'go forward'


def test_named_task_subset_is_explicit():
    assert tasks_for('babyai_text', {'tasks': {'babyai_text': ['goto']}}) == ['goto']
    with pytest.raises(ValueError):
        tasks_for('babyai_text', {'tasks': {'babyai_text': ['imaginary_task']}})


def test_missing_backend_is_clear_validation_error():
    with pytest.raises(ValueError):
        validate_config({})


def test_secrets_in_configuration_rejected_without_echo():
    config = {'split':'validation','seeds':[1],'variants':['react'],'max_steps':1,
              'provider': {'backend':'mlx','api_key':'synthetic-do-not-echo'}}
    with pytest.raises(ValueError) as result:
        validate_config(config)
    assert 'synthetic-do-not-echo' not in str(result.value)


def test_resume_detects_changed_prior_before_model_is_loaded(tmp_path):
    checkpoint = tmp_path/'prior.json'
    checkpoint.write_text('{"weights":[],"training_rows":0}')
    config = {'split':'validation','seeds':[1],'variants':['react'],'max_steps':1,
              'benchmarks':['textcraft'], 'provider':{'backend':'mlx'},
              'prior_path':str(checkpoint),'results_dir':str(tmp_path/'results')}
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(config))
    (tmp_path/'results').mkdir()
    (tmp_path/'results/run.json').write_text(json.dumps({
        'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        'code_sha256':code_hash(),'prior_sha256':'changed'}))
    with pytest.raises(ValueError,match='checkpoint'):
        run(config_path,resume=True)
