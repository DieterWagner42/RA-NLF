#!/usr/bin/env python3
"""
Universal UC Analysis with output directory management
Copies existing results from new/ to old/ and generates new results in new/
"""

import os
import sys
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

def run_uc_analysis(uc_file=None, domain=None):
    """Run UC analysis and save results to new/ directory"""
    
    # Set defaults
    if uc_file is None:
        uc_file = 'Use Case/UC1.txt'
    
    if domain is None:
        # Auto-detect domain using context analysis
        try:
            import sys
            sys.path.append('src')
            from context_analyzer import ContextAnalyzer
            
            analyzer = ContextAnalyzer()
            context = analyzer.analyze_uc_context(uc_file)
            domain = context['detected_domain']
            confidence = context['confidence']
            
            print(f"[AUTO-DETECT] Domain detected from capability: {domain} (confidence: {confidence:.2f})")
            if confidence < 0.5:
                print(f"[WARNING] Low confidence in domain detection!")
            
        except Exception as e:
            print(f"[WARNING] Could not auto-detect domain: {e}")
            print("[FALLBACK] Using common_domain")
            domain = 'common_domain'
    
    # First backup existing results
    copy_new_to_old()
    
    print(f"\\n[ANALYSIS] Starting analysis of {uc_file} with domain {domain}...")
    
    # Import and run the structured analyzer
    try:
        import sys
        sys.path.append('src')
        
        from structured_uc_analyzer import StructuredUCAnalyzer
        
        # Run UC analysis 
        analyzer = StructuredUCAnalyzer(domain_name=domain)
        line_analyses, json_output = analyzer.analyze_uc_file(uc_file)
        
        # Move results to new/ directory
        new_dir = Path("new")
        
        # Get UC name for file patterns
        uc_name = Path(uc_file).stem
        
        # Find and move JSON file
        if os.path.exists(json_output):
            new_json_path = new_dir / Path(json_output).name
            shutil.move(json_output, new_json_path)
            print(f"[OUTPUT] JSON moved to: {new_json_path}")
        
        # Find and move CSV file
        csv_pattern = f"{uc_name}_Structured_UC_Steps_RA_Classes.csv"
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
        
        print(f"\\n[SUCCESS] {uc_name} analysis completed! Results saved in new/ directory")
        
    except ImportError as e:
        print(f"[ERROR] Could not import structured analyzer: {e}")
        print("[FALLBACK] Using generic_uc_analyzer...")
        
        # Fallback to generic analyzer if structured doesn't work
        os.system(f"python src/generic_uc_analyzer.py \"{uc_file}\" {domain}")
        
        # Move any generated files to new/
        new_dir = Path("new")
        
        for pattern in [f"*{Path(uc_file).stem}*.json", f"*{Path(uc_file).stem}*.csv", f"*{Path(uc_file).stem}*.png"]:
            for file_path in glob.glob(pattern):
                new_path = new_dir / Path(file_path).name
                shutil.move(file_path, new_path)
                print(f"[OUTPUT] Moved to new/: {file_path}")

def main():
    """Main function with command line argument support"""
    if len(sys.argv) > 1:
        uc_file = sys.argv[1]
        domain = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        uc_file = None
        domain = None
    
    run_uc_analysis(uc_file, domain)

if __name__ == "__main__":
    main()