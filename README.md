# Weight, what?

Exposing LLM uncertainty, unfairness, and other related trustworthiness metrics to users at response time.

# Contributing guide

## Pixi installation

### Why pixi (vs conda)

Pixi is increasingly preferred by ML researchers for its speed and reproducibility.

| Aspect | Conda/Mamba | Pixi |
|--------|-------------|------|
| Environment location | Global (`~/miniconda3/envs/`) | Project-local (`.pixi/envs/`) |
| Lock file | None native (env.yml is loose) | Built-in, cross-platform |
| Speed | Slow (conda) / Fast (mamba) | ~10x faster than conda |
| Reproducibility | Poor (no lock) | Excellent |
| Task runner | None | Built-in |
| Global state | Has base env | Clean - no global state |
| Installation | Needs installer | Single binary |

### Key files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project config + all dependencies + task definitions |
| `pixi.lock` | Exact versions for reproducibility (auto-generated, commit this) |
| `.pixi/` | Local environment folder (gitignored, don't commit) |

### Installation (one-time)

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Restart your terminal after installation, e.g with `exec bash`.

### Running tasks

```bash
pixi install              # Install dependencies (first time / after pulling)
pixi run <task>           # Run a task (lint, test, serve, etc.)
pixi run python script.py # Run any python file
pixi shell                # Enter activated environment (for IDE/manual commands)
```

Tasks are defined in `[tool.pixi.tasks]` in `pyproject.toml`.

### Environments

| Environment | Use case |
|-------------|----------|
| `default` | CPU-only, used by CI and teammates without GPU |
| `cuda` | For teammates with NVIDIA GPU (CUDA 12.4) |

Most commands use the default environment automatically. To use CUDA:

```bash
pixi install -e cuda      # Install the CUDA environment (one-time)
pixi run -e cuda serve    # Run app with CUDA support
pixi run -e cuda python train.py
```

## Git workflow

### New branches

- Create from main: `git checkout -b feat/your-feature main`
- Naming conventions: `feat/`, `fix/`, `docs/`, `refactor/` prefixes
- Keep branches focused on one thing

### Syncing with main

Run this regularly (ideally daily) to stay up to date:

```bash
git checkout main
git pull --rebase             # Update local main
pixi install                  # In case dependencies changed
git checkout feat/my-feature
git rebase main               # Rebase feature onto updated main
# Resolve conflicts if any
```

This keeps local main fresh so new branches always start from the right place.

#### Resolving pixi.lock conflicts

This will happen often - someone else updated dependencies on main while you were working. It is a simple fix:

```bash
# During rebase, you see: CONFLICT (content): Merge conflict in pixi.lock

# 1. Accept main's version of the lock file
git checkout --theirs pixi.lock

# 2. Regenerate lock with your dependencies included
pixi install

# 3. Stage the resolved lock file and continue
git add pixi.lock
git rebase --continue
```

The key insight: `pixi.lock` is auto-generated, so you never manually edit it. Just take theirs and let pixi regenerate it.

### Integrating with pixi

- Always run `pixi install` after pulling/rebasing (dependencies may have changed)
- Use `pixi run check` before pushing to catch issues early

### PR titles

PRs get squashed when merging into main, so **PR titles should follow conventional commit format**:

```
type: short description
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `build`, `ci`

Examples:
- `feat: add uncertainty visualization component`
- `fix: handle empty response from LLM`
- `docs: update installation instructions`

Individual commits within a branch don't need to follow this strictly - just keep them reasonably descriptive for your own sanity during development.

## Extra tools

### Linting (ruff)

Catches common errors and style issues.

```bash
pixi run lint         # Check for issues
pixi run lint-fix     # Auto-fix what it can
```

### Formatting (ruff)

Keeps code style consistent across the team.

```bash
pixi run format       # Format all files
pixi run format-check # Check without changing (used in CI)
```

### Type checking (pyrefly)

Catches type errors before runtime. Add type hints to new code.

```bash
pixi run typecheck
```

### Testing (pytest)

Put tests in the `tests/` directory. Name test files `test_*.py`.

```bash
pixi run test
```

### Run all checks

Before pushing, run everything to avoid CI failures:

```bash
pixi run check  # Runs lint, format-check, typecheck, test
```
