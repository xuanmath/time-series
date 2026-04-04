"""
Auto Git Push Script
Automatically commits and pushes changes to GitHub
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_command(cmd, cwd=None):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"

def git_push(commit_message=None, repo_path="."):
    """Commit and push changes to GitHub."""
    repo_path = Path(repo_path)
    
    if commit_message is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        commit_message = f"Auto update: {timestamp}"
    
    print("=" * 50)
    print("Git Auto Push")
    print("=" * 50)
    
    # Check git status
    print("\n[1/4] Checking status...")
    code, stdout, stderr = run_command("git status --short", cwd=repo_path)
    
    if not stdout.strip():
        print("    No changes to commit.")
        return True
    
    print(f"    Changes found:\n{stdout}")
    
    # Add all
    print("\n[2/4] Adding files...")
    code, stdout, stderr = run_command("git add .", cwd=repo_path)
    if code != 0:
        print(f"    Error: {stderr}")
        return False
    print("    Files added.")
    
    # Commit
    print("\n[3/4] Committing...")
    code, stdout, stderr = run_command(f'git commit -m "{commit_message}"', cwd=repo_path)
    if code != 0:
        print(f"    Error: {stderr}")
        return False
    print(f"    Committed: {commit_message}")
    
    # Push
    print("\n[4/4] Pushing to GitHub...")
    code, stdout, stderr = run_command("git push origin main", cwd=repo_path)
    if code != 0:
        print(f"    Error: {stderr}")
        return False
    print("    Pushed successfully!")
    
    print("\n" + "=" * 50)
    print("Push completed!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto Git Push")
    parser.add_argument("--message", "-m", type=str, help="Commit message")
    parser.add_argument("--path", "-p", type=str, default=".", help="Repository path")
    args = parser.parse_args()
    
    success = git_push(args.message, args.path)
    sys.exit(0 if success else 1)
