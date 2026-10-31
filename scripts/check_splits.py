"""Reject task-identity overlap across frozen training, validation and test manifests."""
import argparse
import json
from pathlib import Path


def identities(manifest: dict) -> set[tuple[str, str]]:
    return {(key.split('/')[0], value) for key, value in manifest['tasks'].items()}


def check(paths: list[Path]) -> dict:
    seen = {}
    report = {}
    splits = set()
    for path in paths:
        manifest = json.loads(path.read_text())
        if manifest['split'] in splits:
            raise ValueError('Repeated split manifest')
        splits.add(manifest['split'])
        tasks = identities(manifest)
        overlap = tasks & seen.keys()
        if overlap:
            raise ValueError(f'Found {len(overlap)} task identities shared between splits')
        for task in tasks:
            seen[task] = manifest['split']
        report[manifest['split']] = {'rows': len(manifest['tasks']), 'unique_tasks': len(tasks)}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifests', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.manifests), indent=2))

