#!/usr/bin/env python3
"""
Setup helper for RAG Chat Assistant
This script helps you set up the environment and verify everything is working.
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} is installed")
    return True

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies!")
        return False

def check_env_file():
    """Check if .env file exists and has API key"""
    print("\n🔑 Checking environment configuration...")
    
    if not os.path.exists(".env"):
        print("⚠️  .env file not found!")
        print("   Creating .env from template...")
        
        # Copy from template
        if os.path.exists("env_example.txt"):
            with open("env_example.txt", "r") as src, open(".env", "w") as dst:
                dst.write(src.read())
            print("✅ Created .env file")
            print("⚠️  Please add your OpenAI API key to the .env file!")
            return False
    
    # Check if API key is set
    with open(".env", "r") as f:
        content = f.read()
        if "your-openai-api-key-here" in content or "OPENAI_API_KEY=" not in content:
            print("⚠️  OpenAI API key not set in .env file!")
            print("   Please add your API key to the .env file")
            return False
    
    print("✅ Environment configuration looks good!")
    return True

def main():
    """Main setup function"""
    print("🚀 RAG Chat Assistant Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check environment
    env_ready = check_env_file()
    
    print("\n" + "=" * 40)
    if env_ready:
        print("✅ Setup complete! You can now run:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Setup incomplete!")
        print("   Please add your OpenAI API key to .env file")
        print("   Then run: streamlit run app.py")
    
    print("\n📚 Happy chatting with your documents!")

if __name__ == "__main__":
    main() 