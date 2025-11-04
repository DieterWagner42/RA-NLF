#!/usr/bin/env python3
"""
Analyze ONLY UC1 with Enhanced Generative Context System
Fresh analysis of UC1.txt with latest generative features
"""

import sys
import os
import shutil
sys.path.append('src')

from structured_uc_analyzer import StructuredUCAnalyzer
from official_rup_engine import OfficialRUPEngine

def clean_previous_uc1_files():
    """Clean any previous UC1 analysis files"""
    uc1_files = [
        "UC1_Structured_RA_Analysis.json",
        "UC1_Structured_UC_Steps_RA_Classes.csv", 
        "UC1_RA_Diagram_Official_RUP.png"
    ]
    
    print("Cleaning previous UC1 analysis files...")
    for filename in uc1_files:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"  Removed: {filename}")

def analyze_uc1_fresh():
    """Perform fresh UC1 analysis"""
    uc1_path = "Use Case/UC1.txt"
    
    if not os.path.exists(uc1_path):
        print(f"ERROR: UC1 file not found: {uc1_path}")
        return None, None
    
    print(f"\nAnalyzing UC1 with Enhanced Generative Context System")
    print(f"File: {uc1_path}")
    print("-" * 60)
    
    # Initialize enhanced analyzer
    analyzer = StructuredUCAnalyzer("beverage_preparation")
    
    # Perform analysis
    line_analyses, output_json = analyzer.analyze_uc_file(uc1_path)
    
    print(f"\nUC1 Analysis Completed!")
    print(f"JSON Output: {output_json}")
    
    return line_analyses, analyzer

def generate_uc1_diagram():
    """Generate UC1 RA diagram"""
    json_file = "UC1_Structured_RA_Analysis.json"
    
    if not os.path.exists(json_file):
        print(f"ERROR: JSON file not found: {json_file}")
        return None
    
    print(f"\nGenerating UC1 RA Diagram...")
    print(f"Using: {json_file}")
    
    # Initialize RUP engine
    engine = OfficialRUPEngine(figure_size=(24, 20))
    
    # Generate diagram
    diagram_path = engine.create_official_rup_diagram_from_json(json_file)
    
    print(f"UC1 Diagram Generated: {diagram_path}")
    return diagram_path

def show_uc1_analysis_summary():
    """Show summary of UC1 analysis results"""
    json_file = "UC1_Structured_RA_Analysis.json"
    csv_file = "UC1_Structured_UC_Steps_RA_Classes.csv"
    
    if not os.path.exists(json_file):
        print("No JSON analysis file found")
        return
    
    import json
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n" + "=" * 60)
    print("UC1 Analysis Summary")
    print("=" * 60)
    
    meta = data['meta']
    components = data['components']
    
    print(f"Use Case: {meta['capability']} - {meta['feature']}")
    print(f"Goal: {meta['goal']}")
    print(f"Domain: {meta['domain']}")
    print(f"Total RA Classes: {meta['total_ra_classes']}")
    print(f"Analysis Engine: {meta['analysis_engine']}")
    print(f"Generated: {meta['generated_at']}")
    
    print(f"\nComponent Breakdown:")
    print(f"  Actors: {len(components.get('actors', []))}")
    print(f"  Boundaries: {len(components.get('boundaries', []))}")
    print(f"  Controllers: {len(components.get('controllers', []))}")
    print(f"  Entities: {len(components.get('entities', []))}")
    
    # Show enhanced features
    enhanced_controllers = [c for c in components.get('controllers', []) 
                          if any(keyword in c.get('name', '').lower() 
                               for keyword in ['safety', 'hygiene', 'food', 'haccp'])]
    
    material_entities = [e for e in components.get('entities', []) 
                        if 'material' in e.get('name', '').lower()]
    
    supply_boundaries = [b for b in components.get('boundaries', []) 
                        if 'supply' in b.get('name', '').lower()]
    
    print(f"\nEnhanced Features (Generative Context):")
    print(f"  Safety/Hygiene Controllers: {len(enhanced_controllers)}")
    print(f"  Operational Material Entities: {len(material_entities)}")
    print(f"  Supply Chain Boundaries: {len(supply_boundaries)}")
    
    # Show file sizes
    if os.path.exists(csv_file):
        csv_size = os.path.getsize(csv_file)
        print(f"\nGenerated Files:")
        print(f"  JSON: {json_file} ({os.path.getsize(json_file)} bytes)")
        print(f"  CSV: {csv_file} ({csv_size} bytes)")

def move_to_new_folder():
    """Move UC1 results to new folder"""
    # Ensure new folder exists
    os.makedirs("new", exist_ok=True)
    
    uc1_files = [
        "UC1_Structured_RA_Analysis.json",
        "UC1_Structured_UC_Steps_RA_Classes.csv", 
        "UC1_RA_Diagram_Official_RUP.png"
    ]
    
    print(f"\nMoving UC1 files to 'new' folder...")
    for filename in uc1_files:
        if os.path.exists(filename):
            dest_path = os.path.join("new", filename)
            shutil.move(filename, dest_path)
            print(f"  Moved: {filename} -> new/")

def main():
    """Main UC1 analysis function"""
    print("UC1 Enhanced Analysis - Generative Context System")
    print("=" * 60)
    
    try:
        # Step 1: Clean previous files
        clean_previous_uc1_files()
        
        # Step 2: Perform fresh UC1 analysis
        line_analyses, analyzer = analyze_uc1_fresh()
        
        if not analyzer:
            print("UC1 analysis failed!")
            return
        
        # Step 3: Generate RA diagram
        diagram_path = generate_uc1_diagram()
        
        # Step 4: Show summary
        show_uc1_analysis_summary()
        
        # Step 5: Move to new folder
        move_to_new_folder()
        
        print(f"\n" + "=" * 60)
        print("UC1 Analysis Complete!")
        print("=" * 60)
        print("Generated Files (in new/ folder):")
        print("  - UC1_Structured_RA_Analysis.json")
        print("  - UC1_Structured_UC_Steps_RA_Classes.csv")
        print("  - UC1_RA_Diagram_Official_RUP.png")
        print()
        print("Features:")
        print("  ✓ Enhanced generative context system")
        print("  ✓ Operational materials with safety/hygiene")
        print("  ✓ No hardcoded contexts - all from domain JSON")
        print("  ✓ NLP-based semantic analysis")
        print("  ✓ Universal operational materials framework")
        
    except Exception as e:
        print(f"Error during UC1 analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()