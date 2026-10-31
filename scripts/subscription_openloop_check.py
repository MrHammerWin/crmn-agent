"""A separately labelled, one-call full-task feasibility check. Not a CRMN evaluation."""
import json
from dataclasses import asdict
from pathlib import Path

from pydantic import Field

from crmn_agent.codex_provider import CodexProvider
from crmn_agent.environments import make_env
from crmn_agent.schemas import Schema


class Plan(Schema):
    actions: list[str] = Field(max_length=32)


def main():
    out = Path('results/subscription_openloop')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'check.json').exists():
        raise ValueError('Do not overwrite an existing feasibility result')
    env = make_env('textcraft', 'default', 10001, {'split': 'validation'})
    provider = CodexProvider({
        'backend': 'codex', 'model': 'gpt-5.6-luna', 'allow_subscription': True,
        'subscription_max_launches': 30, 'max_calls': 1, 'max_total_tokens': 100000,
    }, out/'model_calls.jsonl')
    try:
        obs = env.reset()
        plan = provider.complete(
            Plan,
            'Make a complete executable crafting plan, at most 32 commands. Inventory starts empty. '
            'Only get base ingredients, never get a craftable intermediate. Follow the exact recipe '
            'commands and quantities shown. Account for recipe output quantities and resource consumption. '
            'Prefer one simple dependency chain; collect all necessary base resources first. '
            'Return no prose, only JSON with actions.',
            {'mission': obs.mission, 'observation': obs.text, 'background': obs.background}, 768,
        )
        trajectory = []
        for action in plan.actions:
            obs = env.step(action)
            trajectory.append({'action': action, 'observation': obs.text,
                               'reward': obs.reward, 'done': obs.terminated})
            if obs.terminated:
                break
        record = {
            'kind': 'open_loop_environment_feasibility_not_CRMN', 'is_real_llm': True,
            'benchmark': 'textcraft', 'split': 'validation', 'seed': 10001,
            'task_id': env.task_id, 'mission': obs.mission, 'model': provider.model,
            'planned_actions': plan.actions, 'executed_steps': len(trajectory),
            'success': obs.success, 'trajectory': trajectory, 'usage': asdict(provider.usage),
        }
        (out/'check.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(json.dumps({'kind': record['kind'], 'steps': record['executed_steps'],
                          'success': record['success']}, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
