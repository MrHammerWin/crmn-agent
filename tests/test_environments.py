import pytest

from crmn_agent.environments import BABY_TASKS, make_env

CONFIG = {'split': 'validation', 'babyai_protocol': 'balrog_mixed', 'alfworld_data': 'data/alfworld'}


def test_textcraft_resets_reproducibly_and_rejects_impossible_get():
    env = make_env('textcraft', 'default', 10001, CONFIG)
    try:
        a = env.reset()
        b = env.reset()
        assert a.text == b.text
        assert a.background == b.background
        result = env.step('get 1 impossible imaginary item')
        assert not result.success
        assert not result.terminated
    finally:
        env.close()


@pytest.mark.parametrize('task', BABY_TASKS)
def test_babyai_all_official_task_types_have_text_and_same_seed_reset(task):
    env = make_env('babyai_text', task, 10001, CONFIG)
    try:
        a = env.reset()
        identity = env.task_id
        b = env.reset()
        assert a.text == b.text
        assert identity == env.task_id
        assert len(a.actions) == 6
        result = env.step('turn left')
        assert result.text
    finally:
        env.close()


def test_alfworld_adapter_can_reset_and_step_without_expert_data():
    env = make_env('alfworld', 'default', 10001, CONFIG)
    try:
        obs = env.reset()
        assert obs.mission and obs.text
        assert 'extra.expert_plan' not in obs.__dict__
        assert obs.actions
        following = env.step(obs.actions[0])
        assert isinstance(following.text, str)
    finally:
        env.close()


def test_textcraft_handwritten_positive_control_reaches_native_success():
    """Environment/reward test only. This is NOT an LLM-generated successful trajectory."""
    env = make_env('textcraft', 'default', 10001, CONFIG)
    try:
        initial = env.reset()
        if initial.mission != 'craft orange bed.':
            pytest.skip('The pinned validation fixture changed; inspect the task manifest')
        commands = [
            'get 12 string',
            'craft 1 white wool using 4 string',
            'craft 1 white wool using 4 string',
            'craft 1 white wool using 4 string',
            'get 1 oak logs',
            'craft 4 oak planks using 1 oak logs',
            'craft 1 white bed using 3 white wool, 3 oak planks',
            'get 1 orange tulip',
            'craft 1 orange dye using 1 orange tulip',
            'craft 1 orange bed using 1 white bed, 1 orange dye',
        ]
        for command in commands:
            final = env.step(command)
            assert not final.text.startswith('Could not'), command
        assert final.terminated
        assert final.success
        assert final.reward == 1
    finally:
        env.close()
