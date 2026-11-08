# Repository Guidelines

## Project Structure & Module Organization

This Windows-oriented Python repository cleans Revo Scan frames, aligns scans, and produces point clouds and meshes.

- `pipeline/`: standalone processing and evaluation scripts; `revo_frames/revo_projeto.py` handles Revo project data, `creality/` contains scanner-specific workflows, and `revo_antigo/` holds older implementations.
- `gui/`: PySide6 application, script catalog, generated forms, and execution queue. Local state lives in ignored `gui/estado/`.
- `aprendizado/`: learned rules and evaluation JSON files; generated caches are ignored.
- `diario/GUIA_PIPELINE.md`: workflow rules and validated recipes. Read it before changing processing behavior; record measured findings in `diario/DIARIO.md` without duplication.

## Build, Test, and Development Commands

Run from the repository root in PowerShell using the local environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe gui\app.py
.\.venv\Scripts\python.exe pipeline\rodar_tudo.py --help
.\.venv\Scripts\python.exe pipeline\limpar_raw.py --help
.\.venv\Scripts\python.exe -m ruff check pipeline gui tests setup.py scripts
.\.venv\Scripts\python.exe -m ruff format --check pipeline gui tests setup.py scripts
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\testes_sem_gpu.py
```

`Abrir_GUI.bat` launches the GUI. Dependencies and lint configuration live in `pyproject.toml`; `setup.py` exposes the editable source paths. Install the editable project before running scripts. The CI runner checks a limited set of tests without CUDA; the full numerical suite requires a working GPU. Revo automation requires installed Revo Scan 6 and control of the real mouse.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions and modules, `PascalCase` classes, and uppercase constants. Follow existing Portuguese terminology and nearby code style. Add concise module docstrings with CLI examples. Expose script options through `argparse`: the GUI discovers scripts and generates forms from their argument definitions. Use Ruff (`pyproject.toml`). Console progress uses `comum.configurar_log` on stdout; preserve machine-readable JSON outputs. Reuse shared helpers and the GPU libraries instead of copying functions. Do not add `sys.path.insert` to work around packaging errors.

## Testing Guidelines

Use `unittest` discovery; there is no coverage threshold. Validate changed scripts with `--help`, then run a representative small scan into a new output directory. Compare frame/point counts and relevant evaluation metrics; for alignment, use signed bias per 30 mm cell. Check GUI discovery and forms when changing CLI arguments. CUDA tests may skip only when the device/driver is absent, never to hide kernel or numerical failures. Report commands, measured results, and unverified behavior. Never replace regression baselines just to make a failing test pass.

## Commit & Pull Request Guidelines

History uses short Portuguese, component-prefixed subjects, such as `pipeline: ...`, `gui: ...`, and `creality: ...`. Follow that pattern. PRs should explain the processing change, affected scripts, validation data, and linked issues when applicable; include screenshots for GUI changes.

## Data Protection & Processing Rules

Never modify original scan projects; write copies to new destinations. Clean Revo frames rather than only exported PLY files. Keep hole filling opt-in. Use the resource budget in `pipeline/recursos.py` (default since 2026-11-08: high priority and all logical processors; `REVO_RECURSOS=conservador` restores low priority and at most 12 threads). Exclude scan data, generated outputs, caches, logs, and local GUI state from commits.

Heavy numerical processing uses CUDA through `gpu.py` and its geometry helpers. `REVO_PERMITIR_CPU=1` is for explicit reference/debug work only. CPU exceptions include file/SQLite I/O, UI automation, scalar optimizers with GPU costs, small pose systems and Open3D/MeshLib operations without a GPU equivalent. Serialize benchmarks to make their timings meaningful. Estimate support planes before removing the table; `limpar_por_placa --planos-apoio` reuses them, and excessive loss returns partial status (2). Never approve a partial revision for merge. Desktop `*_ENTREGA` folders are read-only references. Push and PR require an explicit user request.
