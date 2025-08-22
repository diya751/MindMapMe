# Python 3.13 Installation Guide for MindMapMe

## Quick Fix for Python 3.13 Users

If you're using Python 3.13 and encountering the `distutils` error, follow these steps:

### Option 1: Use the Updated Setup Script (Recommended)

1. **Run the updated setup script:**
   ```bash
   python setup.py
   ```
   The script will automatically detect Python 3.13 and use compatible package versions.

### Option 2: Manual Installation for Python 3.13

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # macOS/Linux
   ```

2. **Upgrade pip and install build tools:**
   ```bash
   pip install --upgrade pip
   pip install setuptools>=68.0.0 wheel>=0.40.0
   ```

3. **Install dependencies using the Python 3.13 requirements:**
   ```bash
   pip install -r requirements_python313.txt
   ```

4. **Create .env file:**
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   FLASK_SECRET_KEY=your_secret_key_here
   ```

5. **Run the application:**
   ```bash
   python app.py
   ```

## Why This Happens

Python 3.13 removed the `distutils` module, which was previously included in the standard library. Many packages that haven't been updated yet still depend on `distutils` for their build process.

## Alternative Solutions

### Option 3: Use Python 3.12 (If Available)

If you have Python 3.12 installed, you can use it instead:

```bash
# Create virtual environment with Python 3.12
python3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Option 4: Install distutils Separately (Not Recommended)

You can install `setuptools` which includes `distutils`:

```bash
pip install setuptools>=68.0.0
```

However, this is not recommended as it may cause other compatibility issues.

## Troubleshooting

### If you still get errors:

1. **Clear pip cache:**
   ```bash
   pip cache purge
   ```

2. **Install packages one by one:**
   ```bash
   pip install setuptools>=68.0.0 wheel>=0.40.0
   pip install flask==3.0.0 flask-cors==4.0.0
   pip install langchain==0.1.0 langchain-community==0.0.10 langchain-groq==0.0.1
   pip install chromadb==0.4.22
   pip install pypdf2==3.0.1
   pip install numpy>=1.26.0 pandas>=2.1.0
   pip install scikit-learn>=1.3.0 sentence-transformers>=2.2.0
   pip install matplotlib>=3.8.0 plotly>=5.17.0 networkx>=3.2.0
   pip install requests>=2.31.0 Pillow>=10.1.0 python-dotenv==1.0.0
   ```

3. **Use conda instead of pip:**
   ```bash
   conda create -n mindmapme python=3.12
   conda activate mindmapme
   pip install -r requirements.txt
   ```

## Package Versions for Python 3.13

The `requirements_python313.txt` file uses these specific versions that are known to work with Python 3.13:

- `setuptools>=68.0.0` - Includes distutils replacement
- `wheel>=0.40.0` - Modern wheel building
- `numpy>=1.26.0` - Python 3.13 compatible
- `pandas>=2.1.0` - Python 3.13 compatible
- `scikit-learn>=1.3.0` - Python 3.13 compatible
- `matplotlib>=3.8.0` - Python 3.13 compatible

## Support

If you continue to have issues:

1. Check that you're using the latest version of pip
2. Ensure you have sufficient disk space
3. Try installing in a fresh virtual environment
4. Consider using Python 3.12 if available

For additional help, refer to the main README.md and INSTALLATION.md files.
