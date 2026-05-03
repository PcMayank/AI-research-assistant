#!/usr/bin/env python3
"""
setup.py — One-click project setup script.

It will:
  1. Check Python version
  2. Create a virtual environment (.venv)
  3. Install all dependencies
  4. Copy .env.example → .env (if not exists)
  5. Create required directories
  6. Print next steps
"""
import sys
import subprocess
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
VENV = ROOT / ".venv"
PYTHON = sys.executable
PIP = VENV / "Scripts" / "pip.exe" if sys.platform == "win32" else VENV / "bin" / "pip"
PYTHON_VENV = VENV / "Scripts" / "python.exe" if sys.platform == "win32" else VENV / "bin" / "python"


def run(cmd: list, **kwargs):
    print(f"\n  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"\n❌ Command failed: {' '.join(str(c) for c in cmd)}")
        sys.exit(1)


def main():
    print("\n" + "="*60)
    print("  🔬 AI Research Assistant — Setup")
    print("="*60)

    # 1. Python version check
    major, minor = sys.version_info[:2]
    print(f"\n✅ Python {major}.{minor} detected")
    if major < 3 or minor < 9:
        print(f"❌ Python 3.9+ is required (you have {major}.{minor})")
        sys.exit(1)

    # 2. Create virtual environment
    if VENV.exists():
        print(f"\n⚡ Virtual environment already exists at {VENV}")
    else:
        print(f"\n🔧 Creating virtual environment at {VENV}…")
        run([PYTHON, "-m", "venv", str(VENV)])
        print("✅ Virtual environment created")

    # 3. Upgrade pip
    print("\n📦 Upgrading pip…")
    run([str(PYTHON_VENV), "-m", "pip", "install", "--upgrade", "pip", "-q"])

    # 4. Install requirements
    print("\n📦 Installing dependencies (this may take 2-5 minutes)…")
    run([
        str(PIP), "install", "-r", str(ROOT / "requirements.txt"), "-q"
    ])
    print("✅ Dependencies installed")

    # 5. Copy .env
    env_file = ROOT / ".env"
    env_example = ROOT / ".env.example"
    if not env_file.exists():
        shutil.copy(env_example, env_file)
        print("\n📄 .env file created from .env.example")
        print("   ⚠️  IMPORTANT: Open .env and add your API key!")
    else:
        print("\n📄 .env already exists — skipping")

    # 6. Create directories
    for d in ["uploads", "vectorstore", "logs", "data"]:
        (ROOT / d).mkdir(exist_ok=True)
    print("✅ Directories created")

    # 7. Print next steps
    print("\n" + "="*60)
    print("  🎉 Setup Complete!")
    print("="*60)
    print("""
NEXT STEPS:
  1. Edit .env and set your API key:
       • Groq (FREE):  https://console.groq.com  → GROQ_API_KEY
       • OpenAI:       https://platform.openai.com → OPENAI_API_KEY

  2. Activate the virtual environment:
       Windows:  .venv\\Scripts\\activate
       Mac/Linux: source .venv/bin/activate

  3. Run the app:
       streamlit run app.py

  4. Open browser at:
       http://localhost:8501
""")


if __name__ == "__main__":
    main()
