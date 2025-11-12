"""
GITHUB UPLOAD HELPER
====================

This script helps you upload all Dynamic RNN files to your GitHub repository.

SETUP (One-time):
1. Install GitHub CLI: https://cli.github.com/
2. Run: gh auth login
3. Create a new repo or use existing one

USAGE:
1. Edit GITHUB_REPO below with your repo name
2. Run: python UPLOAD_TO_GITHUB.py

Or use the manual git commands shown at the bottom.
"""

import os
import subprocess
from pathlib import Path

# ============================================================================
# CONFIGURATION - EDIT THIS
# ============================================================================
GITHUB_USERNAME = "YOUR_USERNAME"  # Change this to your GitHub username
GITHUB_REPO = "dynamic-rnn"         # Change this to your repo name
# ============================================================================

def setup_git_repo():
    """Initialize git repo and push to GitHub."""
    
    source_dir = Path("/mnt/user-data/outputs")
    
    # Files to upload
    files = [
        "ENHANCED_DYNAMIC_RNN.py",
        "test_dynamic_modeling.py",
        "INDEX.md",
        "QUICK_REFERENCE.md",
        "COMPREHENSIVE_SUMMARY.md",
        "IMPLEMENTATION_GUIDE.md",
        "synthetic_dynamic_data.csv",
    ] + [
        f"test_{name}.png" for name in [
            "training_summary",
            "freq_response_Output_1",
            "prediction_grid_Output_1",
            "error_analysis_Output_1",
            "pca_analysis_Output_1",
            "freq_response_Output_2",
            "prediction_grid_Output_2",
            "error_analysis_Output_2",
            "pca_analysis_Output_2",
        ]
    ]
    
    print("="*80)
    print("GITHUB UPLOAD HELPER")
    print("="*80)
    print(f"\nTarget repo: {GITHUB_USERNAME}/{GITHUB_REPO}")
    print(f"Files to upload: {len(files)}\n")
    
    # Check if git is available
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        print("✓ Git is installed")
    except:
        print("✗ Git not found. Please install git first.")
        print_manual_instructions()
        return
    
    # Check if gh CLI is available
    try:
        subprocess.run(["gh", "--version"], check=True, capture_output=True)
        print("✓ GitHub CLI is installed")
        has_gh = True
    except:
        print("⚠ GitHub CLI not found. Will provide manual instructions.")
        has_gh = False
    
    print("\n" + "="*80)
    print("AUTOMATED UPLOAD")
    print("="*80)
    
    if has_gh and GITHUB_USERNAME != "YOUR_USERNAME":
        print("\nAttempting automated upload...")
        try:
            # Create temp directory
            temp_dir = Path.home() / "temp_dynamic_rnn"
            temp_dir.mkdir(exist_ok=True)
            os.chdir(temp_dir)
            
            # Initialize git
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "branch", "-M", "main"], check=True)
            
            # Copy files
            for filename in files:
                src = source_dir / filename
                if src.exists():
                    subprocess.run(["cp", str(src), str(temp_dir)], check=True)
            
            # Create README
            with open("README.md", "w") as f:
                f.write("# Dynamic RNN Package\n\n")
                f.write("Complete implementation of dynamic frequency response prediction.\n\n")
                f.write("## Quick Start\n\n")
                f.write("1. Read `QUICK_REFERENCE.md`\n")
                f.write("2. Review `INDEX.md` for all files\n")
                f.write("3. Import `ENHANCED_DYNAMIC_RNN.py` in your project\n\n")
                f.write("See `COMPREHENSIVE_SUMMARY.md` for complete documentation.\n")
            
            # Git commands
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit: Dynamic RNN package"], check=True)
            
            # Create repo and push (using gh CLI)
            subprocess.run([
                "gh", "repo", "create", f"{GITHUB_USERNAME}/{GITHUB_REPO}",
                "--public", "--source=.", "--push"
            ], check=True)
            
            print(f"\n✓ SUCCESS! Repository created and files uploaded!")
            print(f"\nView your repo at:")
            print(f"https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}")
            
        except Exception as e:
            print(f"\n✗ Automated upload failed: {e}")
            print("\nPlease use manual instructions below.")
            print_manual_instructions()
    else:
        print("\nAutomated upload not available.")
        print_manual_instructions()

def print_manual_instructions():
    """Print manual git/GitHub instructions."""
    
    print("\n" + "="*80)
    print("MANUAL UPLOAD INSTRUCTIONS")
    print("="*80)
    
    print("""
OPTION 1: Using GitHub Web Interface
-------------------------------------
1. Go to https://github.com/new
2. Create a new repository (e.g., "dynamic-rnn")
3. Click "Add file" > "Upload files"
4. Drag and drop all files from /mnt/user-data/outputs/
5. Click "Commit changes"

OPTION 2: Using Git Command Line
---------------------------------
# 1. Download all files to your computer first
# 2. Open terminal in that folder
# 3. Run these commands:

git init
git add .
git commit -m "Initial commit: Dynamic RNN package"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main

OPTION 3: Using GitHub Desktop
-------------------------------
1. Download GitHub Desktop: https://desktop.github.com/
2. Create new repository
3. Copy all files to the repository folder
4. Commit and push

OPTION 4: Using GitHub CLI (Recommended)
-----------------------------------------
# Install: https://cli.github.com/
gh auth login
gh repo create dynamic-rnn --public
cd /path/to/files
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/dynamic-rnn.git
git push -u origin main
""")

    print("="*80)
    print("FILES TO UPLOAD")
    print("="*80)
    print("""
Core Files:
  ✓ ENHANCED_DYNAMIC_RNN.py
  ✓ test_dynamic_modeling.py
  
Documentation:
  ✓ INDEX.md
  ✓ QUICK_REFERENCE.md
  ✓ COMPREHENSIVE_SUMMARY.md
  ✓ IMPLEMENTATION_GUIDE.md
  
Data & Visualizations:
  ✓ synthetic_dynamic_data.csv
  ✓ All PNG files (12 files)
""")
    print("="*80)

if __name__ == "__main__":
    print("\n⚠ IMPORTANT: Edit GITHUB_USERNAME and GITHUB_REPO at top of this script first!\n")
    
    response = input("Have you edited the configuration? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        setup_git_repo()
    else:
        print("\nPlease edit the script first:")
        print("1. Open UPLOAD_TO_GITHUB.py")
        print("2. Change GITHUB_USERNAME to your username")
        print("3. Change GITHUB_REPO to your repository name")
        print("4. Run the script again")
        print("\nOr see manual instructions below:")
        print_manual_instructions()
