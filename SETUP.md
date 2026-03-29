# Setup Guide

Read this file first before working on the project.

## 1) Install VS Code extensions

When VS Code asks to install recommended extensions, choose **Install All**.

You can also open command palette:

- `Ctrl+Shift+P`
- `Show Recommended Extensions`

## 2) Create virtual environment and install dependencies

Use terminal in VS Code (`Ctrl + \``):

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Select Python interpreter

- Press `Ctrl+Shift+P`
- Run `Python: Select Interpreter`
- Choose `.venv`

## 4) Verify configuration

```bash
python config.py
```

Expected output should include project paths and `Config OK!`.

## 5) Run project tasks

### Option A: Run and Debug

- Open **Run and Debug** (`Ctrl+Shift+D`)
- Choose a configuration
- Press `F5`

### Option B: Run tasks

- Open command palette (`Ctrl+Shift+P`)
- Run `Run Task`
- Choose the task you need

## Project structure

```text
Computer-Vision-masked-face-recognition/
|-- .vscode/
|-- src/
|   |-- data_preparation.py
|   |-- train_model.py
|   `-- app.py
|-- tests/
|   `-- test_data_preparation.py
|-- data/
|   |-- raw/
|   |   |-- with_mask/
|   |   `-- without_mask/
|   `-- processed/
|-- models/
|-- results/
|   |-- plots/
|   |-- logs/
|   `-- screenshots/
|-- config.py
|-- requirements.txt
|-- pytest.ini
|-- .gitignore
`-- SETUP.md
```

## Team workflow

```bash
# Sync latest main branch
git pull origin main

# Create your feature branch
git checkout -b yourname/feature-name

# Commit with clear message
git add <files>
git commit -m "short clear message"

# Push and create PR
git push origin yourname/feature-name
```

## Coding rules

- Do not hardcode paths, use `config.py`.
- Run tests before commit:

```bash
pytest tests/ -v
```

- Add docstrings for new functions.
- Add comments only where logic is non-trivial.

## Common issues

- `ModuleNotFoundError: tensorflow`:
  - Activate `.venv`
  - Re-run `pip install -r requirements.txt`

- `FileNotFoundError: mask_detector_final.keras`:
  - Train model first with `python src/train_model.py`

- Camera does not open:
  - Update `CFG.CAMERA_ID` in `config.py` (try `0`, `1`, `2`)

## Team contacts

Fill your team information here:

| Member | Role | GitHub |
|---|---|---|
| TV1 | Data | @... |
| TV2 | Model | @... |
| TV3 | App | @... |
