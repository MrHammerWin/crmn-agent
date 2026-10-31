"""Synthetic corruption diagnostic, explicitly NOT an LLM or benchmark evaluation."""
import json
from pathlib import Path

import numpy as np

from crmn_agent.controller import cycle_error, responsibilities
from crmn_agent.schemas import Anchor, Fact, Reconstruction


def main():
    rng = np.random.default_rng(1729)
    correct = 0
    rows = []
    for i in range(500):
        n = int(rng.integers(2, 5))
        anchors = [Anchor(key=f'object_{j}', value=f'known_{j}', evidence=f'known_{j}') for j in range(n)]
        intact = Reconstruction(facts=[Fact(key=a.key, value=a.value) for a in anchors])
        damaged = Reconstruction(facts=[Fact(key=a.key, value=a.value) for a in anchors[:-1]])
        good_branch = int(rng.integers(2))
        candidates = [intact, damaged] if good_branch == 0 else [damaged, intact]
        errors = [cycle_error(anchors, candidate) for candidate in candidates]
        weights = responsibilities(errors, [0.5, 0.5], [0.5, 0.5])
        selected = int(np.argmax(weights))
        correct += selected == good_branch
        rows.append({'case': i, 'errors': errors, 'weights': weights, 'selected': selected,
                     'uncorrupted_branch': good_branch})
    out = Path('results/synthetic_diagnostic')
    out.mkdir(parents=True, exist_ok=True)
    report = {'kind': 'synthetic_controller_diagnostic', 'is_real_llm': False,
              'benchmark': None, 'seed': 1729, 'cases': 500, 'correct': correct,
              'interpretation': 'Checks gate direction on explicit fact loss, not metacognitive ability'}
    (out/'summary.json').write_text(json.dumps(report, indent=2))
    (out/'cases.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
