"""Explicitly authorized ChatGPT-subscription backend, isolated from repository tools."""
import fcntl
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .providers import BudgetExceeded, ModelOutputError, Provider


def reserve_call(path: Path, authorized_limit: int = 30) -> int:
    if not 0 < authorized_limit <= 30:
        raise ValueError('This session authorized at most 30 subscription launches')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+', encoding='utf-8') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        data = json.loads(handle.read() or '{"used":0,"limit":30}')
        limit = min(data['limit'], authorized_limit)
        if data['used'] >= limit:
            raise BudgetExceeded('The persistent subscription launch allowance is exhausted')
        data['used'] += 1
        data['limit'] = limit
        handle.seek(0)
        handle.truncate()
        json.dump(data, handle)
        handle.flush()
        os.fsync(handle.fileno())
        return data['used']


class CodexProvider(Provider):
    def __init__(self, config: dict, call_log: Path | None = None):
        super().__init__(config, call_log)
        if not config.get('allow_subscription', False):
            raise BudgetExceeded('Subscription use has not been explicitly enabled')
        if not shutil.which('codex'):
            raise RuntimeError('Codex CLI is not installed')
        self.process_env = dict(os.environ)
        for name in ['OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL']:
            self.process_env.pop(name, None)
        status = subprocess.run(['codex', 'login', 'status'], capture_output=True, text=True,
                                env=self.process_env, timeout=20)
        if status.returncode or 'ChatGPT' not in (status.stdout + status.stderr):
            raise RuntimeError('A ChatGPT sign-in, not API-key authentication, is required')
        self.root = Path(config.get('isolated_dir', 'local/codex-isolated')).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = Path(config.get('subscription_ledger', 'local/subscription_budget.json'))

    def _generate(self, system, user, schema, cap):
        reserve_call(self.ledger, self.config.get('subscription_max_launches', 30))
        with tempfile.TemporaryDirectory(prefix='call-', dir=self.root) as name:
            directory = Path(name)
            schema_path, output_path = directory/'schema.json', directory/'response.json'
            schema_path.write_text(json.dumps(schema.model_json_schema()))
            arguments = [
                'codex', '-a', 'never', 'exec', '--ignore-user-config', '--ephemeral',
                '--skip-git-repo-check', '--sandbox', 'read-only', '--color', 'never', '--json',
                '--disable', 'shell_tool', '--disable', 'apps',
                '-c', 'web_search="disabled"', '-c', 'mcp_servers={}',
                '-c', 'project_doc_max_bytes=0', '-c', 'model_reasoning_effort="low"',
                '--model', self.model, '--cd', str(directory),
                '--output-schema', str(schema_path), '--output-last-message', str(output_path), '-',
            ]
            prompt = (
                'This is an isolated text-game experiment. Return JSON using ONLY the supplied data. '
                'Do not call tools, run commands, read files, inspect the workspace or use the internet. '
                'Do not perform coding work.\n' + system + '\nINPUT:\n' + user
            )
            result = subprocess.run(arguments, input=prompt, capture_output=True, text=True,
                                    env=self.process_env, timeout=self.config.get('timeout_seconds', 180))
            events = []
            for line in result.stdout.splitlines():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
            forbidden = {'command_execution', 'mcp_tool_call', 'web_search', 'file_change', 'tool_call'}
            if any(e.get('item', {}).get('type') in forbidden for e in events):
                raise ModelOutputError('Rejected: Codex used tools instead of the supplied observations')
            if result.returncode or not output_path.exists():
                raise ModelOutputError('Codex did not complete; raw stderr was suppressed to protect credentials')
            usage = [e.get('usage', {}) for e in events if e.get('type') == 'turn.completed']
            if not usage:
                raise ModelOutputError('Codex completion is missing usage evidence')
            inputs = sum(u.get('input_tokens', 0) for u in usage)
            outputs = sum(u.get('output_tokens', 0) for u in usage)
            return output_path.read_text(), inputs, outputs

