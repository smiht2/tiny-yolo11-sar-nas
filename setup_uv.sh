#!/bin/bash
# Quick setup script using uv (fastest option)

echo "🚀 Setting up project with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
uv venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source .venv/bin/activate

# Install dependencies with CUDA support (change cu121 to your CUDA version if needed)
echo "📥 Installing PyTorch with CUDA 12.1..."
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
echo "📥 Installing project dependencies..."
uv pip install -e .

echo ""
echo "✨ Setup complete! Activate the environment with:"
echo "   source .venv/bin/activate"
echo ""
echo "Then run validation with:"
echo "   sh nas_tiny.sh"
