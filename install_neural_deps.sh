#!/bin/bash

echo "Installing PyTorch and neural network dependencies..."

# Install PyTorch (CPU version for compatibility)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Alternative for GPU if available:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "✅ Installation complete!"
echo ""
echo "Run the neural network with:"
echo "python3 src/neural_model.py --epochs 50 --batch_size 32"