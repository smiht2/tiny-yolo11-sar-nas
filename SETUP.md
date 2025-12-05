# Setup Instructions

## Installation Options

This project supports multiple package managers. Choose the one that fits your workflow:

### Option 1: Using uv (Fastest ⚡ - Recommended)

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Linux/Mac
# or: .venv\Scripts\activate  # On Windows

# Install all dependencies (much faster than pip!)
uv pip install -e .

# With CUDA support
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
```

**Benefits:**
- 10-100x faster than pip
- Better dependency resolution
- Built-in virtual environment management
- Drop-in replacement for pip

### Option 2: Using Poetry (Modern & Popular)

[Poetry](https://python-poetry.org/) provides robust dependency management and packaging.

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Copy Poetry configuration
# (Replace [build-system] in pyproject.toml with the Poetry version from pyproject.poetry.toml)

# Install dependencies
poetry install

# With dev dependencies
poetry install --with dev

# For CUDA support, add PyTorch index first:
poetry source add pytorch-cu121 https://download.pytorch.org/whl/cu121
poetry add torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --source pytorch-cu121

# Activate Poetry's virtual environment
poetry shell

# Or run commands directly
poetry run python nas/nas_yolo11_ea.py --help
```

**Benefits:**
- Automatic virtual environment management
- Dependency locking (poetry.lock)
- Easy publishing to PyPI
- Great for library development

### Option 3: Using pip with pyproject.toml (Standard)

Standard Python installation using pip.

```bash
# Create and activate a virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or: venv\Scripts\activate  # On Windows

# Install the project with all dependencies
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

**Benefits:**
- Standard Python tooling
- No additional tools required
- Works everywhere

### Option 4: Using requirements.txt (Legacy)

1. Create and activate a virtual environment (as above)

2. Generate requirements.txt from pyproject.toml:
```bash
pip install pip-tools
pip-compile pyproject.toml -o requirements.txt
```

3. Install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Quick Comparison

| Tool | Speed | Lock File | Ease of Use | Best For |
|------|-------|-----------|-------------|----------|
| **uv** | ⚡⚡⚡ Fastest | ✅ Yes | ⭐⭐⭐ Easy | Speed-focused workflows |
| **Poetry** | ⚡⚡ Fast | ✅ Yes | ⭐⭐⭐ Easy | Library development |
| **pip** | ⚡ Standard | ❌ No | ⭐⭐ Moderate | Standard projects |
| **requirements.txt** | ⚡ Standard | ❌ No | ⭐ Basic | Legacy compatibility |

## CUDA Support (All Methods)

**uv:**
```bash
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
```

**Poetry:**
```bash
poetry source add pytorch-cu121 https://download.pytorch.org/whl/cu121
poetry add torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --source pytorch-cu121
```

**pip:**
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### Option 3: With CUDA Support

If you need specific CUDA version (e.g., CUDA 12.1):

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## Development Installation

To install with development tools (pytest, black, flake8, mypy):

```bash
pip install -e ".[dev]"
```

## Exporting Dependencies

To export exact versions for reproducibility:

```bash
pip freeze > requirements-lock.txt
```

## Usage

After installation, you can run the NAS scripts:

```bash
# Run validation
sh nas_tiny.sh

# Or run NAS algorithms directly
python nas/nas_yolo11_ea.py --help
python nas/nas_yolo11_nsga2.py --help
```
