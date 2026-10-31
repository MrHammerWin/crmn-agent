"""Explicit offline campaign. No OpenAI API requests, automatic downloads or publication."""
import json
from pathlib import Path

from crmn_agent.analysis import summarize
from crmn_agent.controller import Prior
from crmn_agent.runner import run


def main():
    root = Path(__file__).resolve().parents[1]
    import os
    os.chdir(root)
    training = Path('configs/local_train_v2.json')
    validation = [Path('configs/local_pilot.json'), Path('configs/local_ablations.json')]
    for config_path in [training, *validation]:
        config = json.loads(config_path.read_text())
        if config['provider']['backend'] != 'mlx':
            raise ValueError('This script accepts only the local MLX backend')
    episodes = run(training)
    summarize(episodes)
    rows = []
    for line in episodes.read_text().splitlines():
        episode = json.loads(line)
        for turn in episode['trajectory']:
            if 'responsibilities' in turn['diagnostics']:
                rows.append({**turn['diagnostics'], 'split': episode['split']})
    prior = Prior()
    prior.fit(rows)
    checkpoint = Path('local/checkpoints/prior.json')
    prior.save(checkpoint)
    print('Prior training steps:', len(rows), flush=True)
    for config_path in validation:
        config = json.loads(config_path.read_text())
        config['prior_path'] = str(checkpoint)
        config_path.write_text(json.dumps(config, indent=2))
        summarize(run(config_path))


if __name__ == '__main__':
    main()

