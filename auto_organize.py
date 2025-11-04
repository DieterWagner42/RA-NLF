#!/usr/bin/env python3
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
