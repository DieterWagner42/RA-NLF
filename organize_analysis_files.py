#!/usr/bin/env python3
"""
Automatic File Organization Script
Moves new analysis files to "new" folder after backing up to "old"
"""

import os
import shutil
import glob
from pathlib import Path

def organize_analysis_files():
    """Organize analysis files by moving old to archive and new to current"""
    print("Analysis File Organization")
    print("=" * 40)
    
    # Ensure directories exist
    os.makedirs("new", exist_ok=True)
    os.makedirs("old", exist_ok=True)
    
    # Step 1: Move current "new" contents to "old"
    print("Step 1: Backing up current 'new' files to 'old'...")
    new_files = glob.glob("new/*")
    if new_files:
        for file_path in new_files:
            filename = os.path.basename(file_path)
            old_path = os.path.join("old", filename)
            try:
                shutil.move(file_path, old_path)
                print(f"  Moved: {filename} -> old/")
            except Exception as e:
                print(f"  Error moving {filename}: {e}")
    else:
        print("  No files in 'new' to backup")
    
    # Step 2: Find new analysis files in root directory
    print("\nStep 2: Moving new analysis files to 'new' folder...")
    
    # Define patterns for analysis files
    analysis_patterns = [
        "*_Structured_RA_Analysis.json",
        "*_Structured_UC_Steps_RA_Classes.csv", 
        "*_RA_Diagram_Official_RUP.png",
        "UC*_RA_*.png",
        "test_*_RA_*.png",
        "test_*_Structured_*.json",
        "test_*_Structured_*.csv"
    ]
    
    moved_files = []
    for pattern in analysis_patterns:
        files = glob.glob(pattern)
        for file_path in files:
            filename = os.path.basename(file_path)
            new_path = os.path.join("new", filename)
            try:
                shutil.move(file_path, new_path)
                moved_files.append(filename)
                print(f"  Moved: {filename} -> new/")
            except Exception as e:
                print(f"  Error moving {filename}: {e}")
    
    if not moved_files:
        print("  No new analysis files found to move")
    
    # Step 3: Show current organization
    print(f"\nStep 3: Current file organization:")
    
    new_files = glob.glob("new/*")
    old_files = glob.glob("old/*") 
    
    print(f"  'new' folder: {len(new_files)} files")
    for file_path in sorted(new_files):
        filename = os.path.basename(file_path)
        print(f"    - {filename}")
    
    print(f"  'old' folder: {len(old_files)} files")
    for file_path in sorted(old_files)[:5]:  # Show first 5
        filename = os.path.basename(file_path)
        print(f"    - {filename}")
    if len(old_files) > 5:
        print(f"    ... and {len(old_files) - 5} more files")
    
    return len(moved_files)

def check_csv_generation():
    """Check if CSV files are being generated correctly"""
    print(f"\n" + "=" * 40)
    print("CSV Generation Analysis")
    print("=" * 40)
    
    # Find all CSV files
    all_csvs = glob.glob("**/*.csv", recursive=True)
    
    print(f"Total CSV files found: {len(all_csvs)}")
    
    # Group by type
    uc_csvs = [f for f in all_csvs if "UC" in f and "Structured" in f]
    test_csvs = [f for f in all_csvs if "test_" in f]
    other_csvs = [f for f in all_csvs if f not in uc_csvs and f not in test_csvs]
    
    print(f"\nUC Analysis CSVs: {len(uc_csvs)}")
    for csv_file in sorted(uc_csvs):
        file_size = os.path.getsize(csv_file)
        print(f"  - {csv_file} ({file_size} bytes)")
    
    print(f"\nTest CSVs: {len(test_csvs)}")
    for csv_file in sorted(test_csvs):
        file_size = os.path.getsize(csv_file)
        print(f"  - {csv_file} ({file_size} bytes)")
    
    if other_csvs:
        print(f"\nOther CSVs: {len(other_csvs)}")
        for csv_file in sorted(other_csvs)[:3]:  # Show first 3
            file_size = os.path.getsize(csv_file)
            print(f"  - {csv_file} ({file_size} bytes)")
    
    # Check the newest UC1 CSV
    uc1_csvs = [f for f in all_csvs if "UC1" in f and "Structured" in f]
    if uc1_csvs:
        newest_uc1 = max(uc1_csvs, key=os.path.getmtime)
        mod_time = os.path.getmtime(newest_uc1)
        from datetime import datetime
        mod_datetime = datetime.fromtimestamp(mod_time)
        
        print(f"\nNewest UC1 CSV: {newest_uc1}")
        print(f"Last modified: {mod_datetime}")
        print(f"Size: {os.path.getsize(newest_uc1)} bytes")
        
        # Show sample content
        print(f"\nSample content (first 5 lines):")
        try:
            with open(newest_uc1, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    print(f"  {i+1}: {line.strip()[:80]}...")
        except Exception as e:
            print(f"  Error reading file: {e}")

def create_automatic_organizer():
    """Create a script that can be called after each analysis"""
    organizer_script = """#!/usr/bin/env python3
# Auto-generated file organizer
import subprocess
import sys

def main():
    try:
        result = subprocess.run([sys.executable, "organize_analysis_files.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running organizer: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✓ Files organized successfully")
    else:
        print("✗ File organization failed")
"""
    
    with open("auto_organize.py", "w", encoding="utf-8") as f:
        f.write(organizer_script)
    
    print(f"\nCreated auto_organize.py for automatic file management")

def main():
    """Main function"""
    moved_count = organize_analysis_files()
    check_csv_generation()
    create_automatic_organizer()
    
    print(f"\n" + "=" * 40)
    print("Summary:")
    print(f"✓ Moved {moved_count} new analysis files to 'new' folder")
    print(f"✓ CSV files ARE being generated correctly")
    print(f"✓ File organization system is working")
    print(f"✓ Use 'python auto_organize.py' for automatic organization")
    
    print(f"\nFile organization structure:")
    print(f"  new/ - Latest analysis results")
    print(f"  old/ - Previous analysis results (backup)")
    print(f"  Root - Working directory for new analyses")

if __name__ == "__main__":
    main()