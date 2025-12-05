# Package Management Guide

This project supports multiple modern Python package managers. Choose based on your needs:

## 🚀 Quick Start

### Using uv (Recommended - Fastest)
```bash
./setup_uv.sh
```

### Using Poetry
```bash
./setup_poetry.sh
```

### Using pip
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
```

## 📦 Package Manager Comparison

### uv (⚡ Blazing Fast)
**Best for:** Speed-focused development, large projects, CI/CD

**Pros:**
- 10-100x faster than pip
- Drop-in pip replacement
- Built-in virtual environment management
- Excellent dependency resolution
- No configuration needed

**Installation:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: pip install uv
```

**Usage:**
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Poetry (🎨 Developer-Friendly)
**Best for:** Library development, publishing packages, team projects

**Pros:**
- Excellent dependency management
- Automatic lock files (poetry.lock)
- Easy PyPI publishing
- Built-in virtual environment management
- Great for reproducible builds

**Installation:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Usage:**
```bash
poetry install
poetry shell
poetry run python script.py
```

**Note:** Requires updating `[build-system]` in `pyproject.toml`

### pip (🔧 Standard)
**Best for:** Simple projects, universal compatibility

**Pros:**
- Standard Python tool
- No additional installation
- Works everywhere
- Well-documented

**Usage:**
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## 🎯 Which Should You Use?

| Use Case | Recommended Tool |
|----------|-----------------|
| Want fastest installs | **uv** |
| Publishing to PyPI | **Poetry** |
| Simple project setup | **pip** |
| CI/CD pipelines | **uv** or **pip** |
| Team collaboration | **Poetry** (lock files) |
| Legacy compatibility | **pip + requirements.txt** |

## 🔄 Migration Between Tools

### From pip to uv
```bash
# Just replace pip with uv pip
uv pip install -r requirements.txt
uv pip install -e .
```

### From pip to Poetry
```bash
# Poetry will use pyproject.toml
poetry install
```

### Generate requirements.txt from any
```bash
# From uv
uv pip freeze > requirements.txt

# From Poetry
poetry export -f requirements.txt -o requirements.txt

# From pip
pip freeze > requirements.txt
```

## 📝 Files Included

- `pyproject.toml` - Main project config (pip/uv compatible)
- `pyproject.poetry.toml` - Poetry-specific config reference
- `requirements.txt` - Legacy pip requirements
- `setup.py` - Backward compatibility
- `setup_uv.sh` - Automated uv setup
- `setup_poetry.sh` - Automated Poetry setup
- `SETUP.md` - Detailed installation guide

## 🐳 Docker/Container Usage

For containerized environments, we recommend **uv** for speed:

```dockerfile
FROM python:3.10-slim

# Install uv
RUN pip install uv

WORKDIR /app
COPY pyproject.toml .

# Fast dependency installation
RUN uv pip install -e .

COPY . .
```

## 🤝 Contributing

When contributing:
1. **uv users**: Run `uv pip freeze > requirements-lock.txt` 
2. **Poetry users**: Commit `poetry.lock`
3. **pip users**: Keep `requirements.txt` updated

This ensures reproducibility across all setups.
