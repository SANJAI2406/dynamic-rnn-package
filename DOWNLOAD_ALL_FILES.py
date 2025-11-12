"""
AUTOMATED FILE DOWNLOADER
==========================

This script will automatically download all Dynamic RNN files to your local computer.

USAGE:
1. Download THIS script to your computer
2. Run: python DOWNLOAD_ALL_FILES.py
3. All files will be saved to: ./dynamic_rnn_package/

Alternatively, you can run this directly in Claude.ai interface.
"""

import os
import shutil
from pathlib import Path

def download_all_files():
    """Download all Dynamic RNN files to local directory."""
    
    # Source directory (in Claude environment)
    source_dir = "/mnt/user-data/outputs"
    
    # Destination directory (your local computer)
    dest_dir = Path.home() / "Downloads" / "dynamic_rnn_package"
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to download
    files_to_copy = [
        # Core Python files
        "ENHANCED_DYNAMIC_RNN.py",
        "test_dynamic_modeling.py",
        
        # Documentation
        "INDEX.md",
        "QUICK_REFERENCE.md",
        "COMPREHENSIVE_SUMMARY.md",
        "IMPLEMENTATION_GUIDE.md",
        
        # Data
        "synthetic_dynamic_data.csv",
        
        # Visualizations
        "test_training_summary.png",
        "test_freq_response_Output_1.png",
        "test_prediction_grid_Output_1.png",
        "test_error_analysis_Output_1.png",
        "test_pca_analysis_Output_1.png",
        "test_freq_response_Output_2.png",
        "test_prediction_grid_Output_2.png",
        "test_error_analysis_Output_2.png",
        "test_pca_analysis_Output_2.png",
    ]
    
    print("="*80)
    print("DOWNLOADING DYNAMIC RNN PACKAGE")
    print("="*80)
    print(f"\nDestination: {dest_dir}\n")
    
    success_count = 0
    error_count = 0
    
    for filename in files_to_copy:
        source_path = Path(source_dir) / filename
        dest_path = dest_dir / filename
        
        try:
            if source_path.exists():
                shutil.copy2(source_path, dest_path)
                print(f"✓ {filename}")
                success_count += 1
            else:
                print(f"✗ {filename} (not found)")
                error_count += 1
        except Exception as e:
            print(f"✗ {filename} (error: {e})")
            error_count += 1
    
    print("\n" + "="*80)
    print(f"DOWNLOAD COMPLETE")
    print("="*80)
    print(f"✓ Success: {success_count} files")
    print(f"✗ Errors:  {error_count} files")
    print(f"\nAll files saved to:\n{dest_dir}")
    print("="*80)
    
    # Create a README in the destination
    readme_path = dest_dir / "README.txt"
    with open(readme_path, 'w') as f:
        f.write("DYNAMIC RNN PACKAGE\n")
        f.write("="*80 + "\n\n")
        f.write("Start with:\n")
        f.write("1. QUICK_REFERENCE.md (2-page overview)\n")
        f.write("2. INDEX.md (file index with links)\n")
        f.write("3. COMPREHENSIVE_SUMMARY.md (complete guide)\n\n")
        f.write("Core Files:\n")
        f.write("- ENHANCED_DYNAMIC_RNN.py (main module)\n")
        f.write("- test_dynamic_modeling.py (demo script)\n\n")
        f.write("Package downloaded: " + str(dest_dir) + "\n")
    
    print(f"\n✓ Created README.txt with instructions")
    
    return dest_dir

if __name__ == "__main__":
    try:
        download_dir = download_all_files()
        print(f"\n🎉 SUCCESS! Open this folder:\n{download_dir}\n")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nIf this doesn't work, use the alternative methods below.")
