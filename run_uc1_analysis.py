#!/usr/bin/env python3
"""
UC1 Analysis with output directory management
Copies existing results from new/ to old/ and generates new results in new/
"""

import os
import shutil
import glob
from pathlib import Path
from datetime import datetime

def copy_new_to_old():
    """Copy all results from new/ to old/ before generating new ones"""
    new_dir = Path("new")
    old_dir = Path("old")
    
    # Create directories if they don't exist
    new_dir.mkdir(exist_ok=True)
    old_dir.mkdir(exist_ok=True)
    
    # Get timestamp for backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Copy all files from new/ to old/ with timestamp
    if list(new_dir.glob("*")):
        print(f"[BACKUP] Copying results from new/ to old/ with timestamp {timestamp}")
        
        for file_path in new_dir.glob("*"):
            if file_path.is_file():
                # Add timestamp to filename
                backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
                backup_path = old_dir / backup_name
                shutil.copy2(file_path, backup_path)
                print(f"  Backed up: {file_path.name} -> {backup_name}")
    else:
        print("[BACKUP] No existing results in new/ to backup")

def run_uc1_analysis():
    """Run UC1 analysis and save results to new/ directory"""
    
    # First backup existing results
    copy_new_to_old()
    
    # Auto-detect domain from UC1 capability
    domain = 'beverage_preparation'  # Default fallback
    try:
        import sys
        sys.path.append('src')
        from context_analyzer import ContextAnalyzer
        
        analyzer = ContextAnalyzer()
        context = analyzer.analyze_uc_context('Use Case/UC1.txt')
        domain = context['detected_domain']
        confidence = context['confidence']
        
        print(f"[AUTO-DETECT] Domain detected from UC1 capability: {domain} (confidence: {confidence:.2f})")
        
    except Exception as e:
        print(f"[WARNING] Could not auto-detect domain: {e}")
        print(f"[FALLBACK] Using default domain: {domain}")
    
    print(f"\n[ANALYSIS] Starting UC1 analysis with domain: {domain}...")
    
    # Import and run the structured analyzer
    try:
        from structured_uc_analyzer import StructuredUCAnalyzer
        
        # Run UC1 analysis 
        analyzer = StructuredUCAnalyzer(domain_name=domain)
        line_analyses, json_output = analyzer.analyze_uc_file('Use Case/UC1.txt')
        
        # Move results to new/ directory
        new_dir = Path("new")
        
        # Find and move JSON file
        if os.path.exists(json_output):
            new_json_path = new_dir / Path(json_output).name
            shutil.move(json_output, new_json_path)
            print(f"[OUTPUT] JSON moved to: {new_json_path}")
        
        # Find and move CSV file
        csv_pattern = "UC1_Structured_UC_Steps_RA_Classes.csv"
        if os.path.exists(csv_pattern):
            new_csv_path = new_dir / csv_pattern
            shutil.move(csv_pattern, new_csv_path)
            print(f"[OUTPUT] CSV moved to: {new_csv_path}")
        
        # Generate RA diagram and move to new/
        from official_rup_engine import generate_rup_diagram_from_json
        
        diagram_file = generate_rup_diagram_from_json(str(new_json_path))
        if os.path.exists(diagram_file):
            new_diagram_path = new_dir / Path(diagram_file).name
            shutil.move(diagram_file, new_diagram_path)
            print(f"[OUTPUT] Diagram moved to: {new_diagram_path}")
        
        print(f"\n[SUCCESS] UC1 analysis completed! Results saved in new/ directory")
        
    except ImportError as e:
        print(f"[ERROR] Could not import analyzer: {e}")
        print("[FALLBACK] Using generic_uc_analyzer...")
        
        # Fallback to generic analyzer if structured doesn't work
        os.system("python src/generic_uc_analyzer.py")
        
        # Move any generated files to new/
        new_dir = Path("new")
        
        for pattern in ["*UC1*.json", "*UC1*.csv", "*UC1*.png"]:
            for file_path in glob.glob(pattern):
                new_path = new_dir / Path(file_path).name
                shutil.move(file_path, new_path)
                print(f"[OUTPUT] Moved to new/: {file_path}")

if __name__ == "__main__":
    run_uc1_analysis()