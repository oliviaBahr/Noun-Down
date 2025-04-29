#!/bin/bash

# Script to install all necessary dependencies for the visualizations.
# Usage: bash install_dependencies.sh

set -e  # Exit immediately if a command exits with a non-zero status

echo "===================================================="
echo "Installing dependencies for CSV data visualization"
echo "===================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Installing Python 3..."
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        sudo yum install -y python3 python3-pip
    elif command -v brew &> /dev/null; then
        # macOS with Homebrew
        brew install python3
    else
        echo "Error: Could not detect package manager. Please install Python 3 manually."
        exit 1
    fi
fi

# Create and activate virtual environment (optional but recommended)
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required Python packages
echo "Installing required Python packages..."
pip install polars matplotlib seaborn numpy scikit-learn

# Verify installations
echo "Verifying installations..."
python3 -c "import polars; import matplotlib; import seaborn; import numpy; print('All dependencies successfully installed!')"

# Create a simple README
echo "Creating README file..."
cat > README.md << 'EOL'
# csv Visualization Setup

This project is set up to visualize CSV data using Polars, Matplotlib, and Seaborn.

## Dependencies Installed
- polars: Fast DataFrame library (similar to pandas)
- matplotlib: Basic plotting library
- seaborn: Statistical data visualization
- numpy: Numerical computing library


echo "===================================================="
echo "Setup complete! Virtual environment is now activated."
echo "You can run your visualization script now."
echo "===================================================="

# Usage instructions
echo "To use the visualization script:"
echo "1. Make sure the results_summary.py file is accessible"
echo "2. Run the Python script with: python visualizations.py"
echo "3. Follow the prompts to select a column for visualization"
echo ""
echo "When you're done, deactivate the virtual environment with: deactivate"
