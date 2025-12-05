#!/bin/bash
# Quick setup script using Poetry

echo "🚀 Setting up project with Poetry..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Note: You need to update pyproject.toml to use Poetry's build system
echo "⚠️  Note: Make sure to update [build-system] in pyproject.toml"
echo "   Replace with Poetry configuration from pyproject.poetry.toml"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
poetry install --with dev

# Add PyTorch source for CUDA support
echo "🔧 Adding PyTorch CUDA repository..."
poetry source add pytorch-cu121 https://download.pytorch.org/whl/cu121 2>/dev/null || true

# Install PyTorch with CUDA
echo "📥 Installing PyTorch with CUDA 12.1..."
poetry add torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --source pytorch-cu121

echo ""
echo "✨ Setup complete! Activate the environment with:"
echo "   poetry shell"
echo ""
echo "Or run commands directly with:"
echo "   poetry run python nas/nas_yolo11_ea.py"
