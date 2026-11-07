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
.\.venv\Scripts\python.exe gui\app.py
.\.venv\Scripts\python.exe pipeline\rodar_tudo.py --help
.\.venv\Scripts\python.exe pipeline\limpar_raw.py --help
```

The first command launches the GUI; `Abrir_GUI.bat` is the desktop equivalent. The help commands describe pipeline and cleaning options. There is no separate build step or dependency manifest. PySide6 is required for the GUI; processing dependencies vary by script. Revo automation requires installed Revo Scan 6 and control of the real mouse.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions and modules, `PascalCase` classes, and uppercase constants. Follow existing Portuguese terminology and nearby code style. Add concise module docstrings with CLI examples. Expose script options through `argparse`: the GUI discovers scripts and generates forms from their argument definitions. No formatter or linter configuration is currently provided.

## Testing Guidelines

No automated test framework or coverage threshold is configured. Validate changed scripts with `--help`, then run a representative small scan into a new output directory. Compare frame/point counts and relevant evaluation metrics; for alignment, follow the guide's signed bias per 30 mm cell. Check GUI discovery and forms when changing CLI arguments. Report commands, measured results, and unverified behavior in the PR.

## Commit & Pull Request Guidelines

History uses short Portuguese, component-prefixed subjects, such as `pipeline: ...`, `gui: ...`, and `creality: ...`. Follow that pattern. PRs should explain the processing change, affected scripts, validation data, and linked issues when applicable; include screenshots for GUI changes.

## Data Protection & Processing Rules

Never modify original scan projects; write copies to new destinations. Clean Revo frames rather than only exported PLY files. Keep hole filling opt-in. Use low process priority and at most 12 threads, following the pipeline guide. Exclude scan data, generated outputs, caches, logs, and local GUI state from commits.