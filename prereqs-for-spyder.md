# ML for Finance Workshop: Stock Price Prediction with Artificial Neural Networks

## Prerequisites

### Python Version
- Python 3.7 or higher recommended
- Anaconda distribution recommended for Spyder users

### Installing TensorFlow in Spyder

#### Method 1: Using Anaconda Navigator (Recommended for Beginners)
1. Open Anaconda Navigator
2. Select your environment (or create a new one)
3. Search for "tensorflow" in the package list
4. Click "Apply" to install

#### Method 2: Using Conda (Recommended)
```bash
# Create a new environment (optional but recommended)
conda create -n ml_finance python=3.8
conda activate ml_finance

# Install Spyder and TensorFlow
conda install spyder tensorflow keras

# Install other required packages
conda install pandas numpy scikit-learn matplotlib
pip install yfinance plotly
```

#### Method 3: Using Pip in Spyder Console
```python
# Run this in Spyder's IPython console
import sys
!{sys.executable} -m pip install tensorflow keras yfinance pandas numpy scikit-learn matplotlib plotly
```

#### Method 4: Bash Functions for Different OS

**For Windows (PowerShell):**
```powershell
# Function to install TensorFlow for Spyder
function Install-TensorFlowSpyder {
    Write-Host "Installing TensorFlow and dependencies for Spyder..." -ForegroundColor Green
    
    # Check if conda is available
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        conda install -y tensorflow keras pandas numpy scikit-learn matplotlib
        pip install yfinance plotly
    } else {
        pip install tensorflow keras yfinance pandas numpy scikit-learn matplotlib plotly
    }
    
    Write-Host "Installation complete!" -ForegroundColor Green
}

# Run the function
Install-TensorFlowSpyder
```

**For macOS/Linux (Bash):**
```bash
#!/bin/bash

# Function to install TensorFlow for Spyder
install_tensorflow_spyder() {
    echo "Installing TensorFlow and dependencies for Spyder..."
    
    # Check if conda is available
    if command -v conda &> /dev/null; then
        echo "Using conda for installation..."
        conda install -y tensorflow keras pandas numpy scikit-learn matplotlib
        pip install yfinance plotly
    else
        echo "Using pip for installation..."
        pip install tensorflow keras yfinance pandas numpy scikit-learn matplotlib plotly
    fi
    
    echo "Installation complete!"
}

# Function to create dedicated environment
create_ml_finance_env() {
    echo "Creating ML Finance environment..."
    
    if command -v conda &> /dev/null; then
        conda create -n ml_finance python=3.8 -y
        conda activate ml_finance
        conda install -y spyder tensorflow keras pandas numpy scikit-learn matplotlib
        pip install yfinance plotly
        echo "Environment 'ml_finance' created successfully!"
        echo "To activate: conda activate ml_finance"
    else
        echo "Conda not found. Please install Anaconda first."
    fi
}

# Run the installation
install_tensorflow_spyder
```

**One-line Installation Commands:**
```bash
# For conda users (recommended)
conda install -c conda-forge tensorflow keras spyder pandas numpy scikit-learn matplotlib -y && pip install yfinance plotly

# For pip users
pip install tensorflow keras yfinance pandas numpy scikit-learn matplotlib plotly
```

### Verifying Installation in Spyder

Run this code in Spyder to verify all packages are installed correctly:

```python
import sys
print(f"Python version: {sys.version}")

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not installed")

try:
    import keras
    print(f"Keras version: {keras.__version__}")
except ImportError:
    print("Keras not installed")

# Check other packages
packages = ['yfinance', 'pandas', 'numpy', 'sklearn', 'matplotlib', 'plotly']
for package in packages:
    try:
        __import__(package)
        print(f"✓ {package} installed")
    except ImportError:
        print(f"✗ {package} NOT installed")
```

### Troubleshooting Spyder Installation Issues

1. **ImportError in Spyder but not in terminal:**
   - Ensure Spyder is using the correct Python environment
   - In Spyder: Tools → Preferences → Python Interpreter
   - Select the environment where you installed TensorFlow

2. **Permission Errors:**
   ```bash
   # Use --user flag
   pip install --user tensorflow keras yfinance
   ```

3. **Conflicting Versions:**
   ```bash
   # Create a fresh environment
   conda create -n ml_finance_fresh python=3.8
   conda activate ml_finance_fresh
   conda install spyder tensorflow keras
   ```

4. **GPU Support (Optional):**
   ```bash
   # For NVIDIA GPU support
   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
   pip install tensorflow-gpu
   ```

### Required Package Versions (Tested Configuration)
```
tensorflow>=2.10.0
keras>=2.10.0
yfinance>=0.2.18
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.5.0
plotly>=5.14.0
```
