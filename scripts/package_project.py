"""Package source, public experiment outputs and docs; exclude credentials, weights and environments."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root.parent/'outputs'/'crmn-agent-local.zip'
    destination.parent.mkdir(parents=True, exist_ok=True)
    excluded = {'.git', '.venv', '__pycache__', '.pytest_cache', '.ruff_cache',
                'models', 'data', 'external', 'local'}
    files = []
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        if any(part in excluded or part.endswith('.egg-info') for part in relative.parts):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if path.name.startswith('.env') and path.name != '.env.example':
            continue
        if path.suffix in {'.pyc', '.log'} or relative.parts[:2] == ('results', 'private'):
            continue
        files.append(path)
    with ZipFile(destination, 'w', compression=ZIP_DEFLATED) as zipped:
        for path in files:
            zipped.write(path, arcname='crmn-agent/'+str(path.relative_to(root)))
    print(f'Packaged {len(files)} files: {destination}')


if __name__ == '__main__':
    main()
