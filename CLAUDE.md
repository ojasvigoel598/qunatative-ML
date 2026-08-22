# CLAUDE.md

Working notes for AI coding agents in this repository.

## Standing rules (apply to every task)

- **Skills first.** Before executing any task, read the request, look at the
  available skills (the skill tool, plus anything installed under
  `~/.agents/skills/`), decide which skill(s) would help, and use them. Do this
  proactively on every task, every time, without being asked.
- **GitHub identity.** All commits and pushes are made under the personal GitHub
  account **ojasvigoel598** (Ojasvi goel <ojasvigoel598@gmail.com>), never
  @codebuff-team. Never change the repository owner or transfer the repo.

## Git Identity and Contribution Policy

- `ojasvigoel598` is the **sole contributor identity** for this repository.
- AI agents are development tools, not repository contributors.
- **Never** create commits under `codebuff-team`, Codebuff, Claude, Claude Code,
  bot, AI, GitHub Actions, or another agent identity.
- **Never** change Git identity automatically.
- Before **every** commit, verify:
  ```bash
  git config user.name
  git config user.email
  ```
- **Never** create an empty, placeholder, generated, unrelated, or fake commit.
- Every commit must correspond to an intentional repository change.
- **Never** create a commit merely to increase contribution/commit counts.
- **Never** rewrite authorship to hide an agent identity; investigate the actual
  Git configuration instead.
- If the identity is wrong or ambiguous, **stop and report** rather than guessing.
- **Sync after every change.** One logical change: test it, update the README or
  docs if needed, commit, push, verify GitHub reflects the change. Never batch
  unrelated changes into one commit. Never leave GitHub in a broken state.

## Project

Quantitative sports-betting research and simulation: Poisson/ELO model, an ML
layer, RL staking, and an agent-based simulation. Ongoing work-in-progress; the
README documents what works, what is experimental, and known limitations.

## Commands

- Tests: `.venv/Scripts/python.exe -m pytest tests/ -q`
- Full pipeline: `.venv/Scripts/python.exe run_full_project.py`
- ML+RL pipeline: `.venv/Scripts/python.exe run_full_ml_rl.py`
- Notebook build: `.venv/Scripts/python.exe notebooks/build_notebook.py`
- Simulation demo: `.venv/Scripts/python.exe demo/simulation.py --trials 25 --matches 300`
