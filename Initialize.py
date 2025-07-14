import os
import sys
import subprocess

def ensure_latest_python():
    print(f"Current Python version: {sys.version}")
    # Actual interpreter upgrade must be done outside Python (e.g., via installer or package manager).

def install_requirements():
    req_file = 'requirements.txt'
    if os.path.exists(req_file):
        print(f"Installing requirements from {req_file}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_file])
    else:
        print(f"{req_file} not found.")

if __name__ == "__main__":
    clear_memory()
    ensure_latest_python()
    install_requirements()