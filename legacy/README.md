# Legacy Scripts

These scripts are preserved for reference only. They predate the current
`run_all.py` / `crop.py` / `scan2.py` pipeline and may contain local paths,
manual debug assumptions, and import-time side effects.

Supported entry points are:

- `run_all.py`
- `gui_app.py`
- `gui_app_en.py`

Do not use files in this folder as production entry points without first
converting paths to `app_paths.py` and moving setup work into `main()`.
