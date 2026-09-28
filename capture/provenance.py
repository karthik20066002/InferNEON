"""Shared provenance for capture scripts: versions, kernel path, clean tree.

Every script that produces paper numbers must record exactly what ran:
which package versions (including the DeltaNet fast kernels), which kernel
path the DeltaNet layers actually took, and the git state. Refsusing to run
on a dirty tree (tracked modifications) is the default; --allow-dirty
exists for debugging and is recorded in the output.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path

VERSION_PACKAGES = ('torch', 'transformers', 'safetensors',
                    'flash-linear-attention', 'fla-core',
                    'causal-conv1d', 'einops')


def package_versions(names=VERSION_PACKAGES) -> dict:
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = 'not installed'
    return out


def delta_kernel_path() -> str:
    """Which DeltaNet implementation the model will take: fast or reference.

    Presence check only (no import side effects): transformers falls back to
    its reference PyTorch implementation when these packages are absent.
    A run that forces the fallback despite installed packages must label
    itself 'reference (forced)' via its own flag.
    """
    fla = importlib.util.find_spec('fla') is not None
    conv = importlib.util.find_spec('causal_conv1d') is not None
    return 'fast' if (fla and conv) else 'reference'


def block_fast_kernels():
    """Force transformers onto the reference DeltaNet path (fresh process).

    Must run before `transformers.models.qwen3_5_moe...` is first imported
    (i.e. before `capture.stream_capture`): the fallback decorators bind at
    modeling-module import time. Raises ImportError for the fast-kernel
    packages so the try/except fallback takes the reference implementation.
    Verify via the reappearing fallback warnings in the log.
    """
    import importlib.abc

    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name in ('fla', 'causal_conv1d') or name.startswith(
                    ('fla.', 'causal_conv1d.')):
                raise ImportError(f'blocked (force-reference-kernels): {name}')
            return None

    sys.meta_path.insert(0, _Block())


def git_state(root: Path) -> dict:
    """Commit + dirtiness. Tracked-only dirt is what the refusal enforces.

    `git_dirty` (any porcelain output, incl. untracked new reports) is kept
    for continuity with older files. `git_tracked_dirty` covers staged and
    unstaged modifications to tracked files: that is what a commit hash
    actually describes, so that is what the refusal checks.
    """
    def run(*args):
        return subprocess.run(['git', '-C', str(root), *args],
                              capture_output=True, text=True)

    commit = run('rev-parse', '--verify', 'HEAD')
    dirty_full = run('status', '--porcelain')
    tracked = run('diff', '--quiet', 'HEAD')
    return {
        'git_commit': commit.stdout.strip() if commit.returncode == 0 else None,
        'git_dirty': bool(dirty_full.stdout.strip()) if dirty_full.returncode == 0 else None,
        # diff --quiet exits 1 on tracked modifications, 0 when clean,
        # 128 when there is no HEAD yet (fail open with a warning upstream).
        'git_tracked_dirty': None if tracked.returncode == 128 else bool(tracked.returncode),
    }


def require_clean_tree(root: Path, allow_dirty: bool, script: str) -> dict:
    """Refuse to run paper-number scripts on a dirty tree. Returns git_state."""
    state = git_state(root)
    if state['git_tracked_dirty'] and not allow_dirty:
        raise SystemExit(
            f'{script}: refusing to run: tracked files are modified '
            f'(commit {state["git_commit"]} does not describe this tree). '
            f'Commit first, or pass --allow-dirty (recorded in the output).')
    if state['git_tracked_dirty'] is None:
        print(f'{script}: WARNING: no git HEAD; provenance is unenforceable.',
              flush=True)
    return state


def file_hashes(root: Path, relpaths: list[str]) -> dict:
    out = {}
    for rel in relpaths:
        out[rel] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    return out
