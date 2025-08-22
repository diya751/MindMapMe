#!/usr/bin/env python3
"""
MindMapMe Setup Script
Automates the installation and setup process for the MindMapMe application.
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print the application banner"""
    print("=" * 60)
    print("           MindMapMe - AI Learning Assistant")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    
    # Special handling for Python 3.13
    if version.major == 3 and version.minor == 13:
        print("⚠️  Python 3.13 detected - using compatible package versions")
        return "python313"
    
    return True

def create_virtual_environment():
    """Create a virtual environment"""
    print("\nCreating virtual environment...")
    try:
        if os.path.exists("venv"):
            print("✅ Virtual environment already exists")
            return True
        
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error: Failed to create virtual environment")
        return False

def get_activate_command():
    """Get the appropriate activation command based on OS"""
    system = platform.system().lower()
    if system == "windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def install_dependencies(python_version=None):
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    try:
        # Determine the pip command for the virtual environment
        if platform.system().lower() == "windows":
            pip_cmd = "venv\\Scripts\\pip"
        else:
            pip_cmd = "venv/bin/pip"
        
        # Choose requirements file based on Python version
        requirements_file = "requirements_python313.txt" if python_version == "python313" else "requirements.txt"
        
        print(f"Using requirements file: {requirements_file}")
        subprocess.run([pip_cmd, "install", "-r", requirements_file], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error: Failed to install dependencies")
        return False

def create_env_file():
    """Create .env file with template"""
    print("\nSetting up environment variables...")
    env_content = """# MindMapMe Environment Variables
# Replace with your actual GROQ API key from https://console.groq.com/
GROQ_API_KEY=your_groq_api_key_here

# Flask secret key (you can generate a random one)
FLASK_SECRET_KEY=your_secret_key_here
"""
    
    if os.path.exists(".env"):
        print("✅ .env file already exists")
        return True
    
    try:
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ .env file created")
        print("⚠️  Please edit .env file and add your GROQ API key")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    directories = ["uploads", "chroma_db"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created {directory}/ directory")
        else:
            print(f"✅ {directory}/ directory already exists")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Edit the .env file and add your GROQ API key:")
    print("   - Get your API key from: https://console.groq.com/")
    print("   - Replace 'your_groq_api_key_here' with your actual key")
    print()
    print("2. Activate the virtual environment:")
    activate_cmd = get_activate_command()
    print(f"   {activate_cmd}")
    print()
    print("3. Run the application:")
    print("   python app.py")
    print()
    print("4. Open your browser and go to: http://localhost:5000")
    print()
    print("For more information, see README.md")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    python_version = check_python_version()
    if not python_version:
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies(python_version):
        sys.exit(1)
    
    # Create .env file
    if not create_env_file():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
