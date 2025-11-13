#!/usr/bin/env python3
"""
Generate UC3 RA Diagram with Data Flows
========================================

Generates complete Robustness Analysis diagram for UC3 Rocket Launch with:
- All RA classes (Actor, Boundary, Controller, Entity)
- Control flows following UC-Methode Rules 1-5
- Data flows (USE/PROVIDE relationships)
- Implementation element violations
- Actor validation violations
"""

import sys
from pathlib import Path
sys.path.append('src')
sys.path.append('visualizers')

from oldpy.generic_uc_analyzer import GenericUCAnalyzer
from enhanced_rup_engine import EnhancedRUPEngine

def main():
    print("Generating UC3 RA Diagram with Data Flows...")
    
    # Initialize analyzer with rocket science domain
    analyzer = GenericUCAnalyzer(domain_name="rocket_science")
    
    # Analyze UC3 file
    uc_file = "Use Case/UC3_Rocket_Launch_Improved.txt"
    if not Path(uc_file).exists():
        print(f"ERROR: UC file not found: {uc_file}")
        return
    
    print(f"Analyzing {uc_file}...")
    verb_analyses, ra_classes = analyzer.analyze_uc_file(uc_file)
    
    print(f"Analysis completed:")
    print(f"  - {len(verb_analyses)} verb analyses")
    print(f"  - {len(ra_classes)} RA classes")
    
    # Generate RA diagram
    engine = EnhancedRUPEngine()
    
    # Create diagram data structure
    diagram_data = {
        "uc_name": "UC3_Rocket_Launch_Improved",
        "ra_classes": ra_classes,
        "verb_analyses": verb_analyses,
        "control_flows": getattr(analyzer, 'control_flows', []),
        "data_flows": getattr(analyzer, 'data_flows', [])
    }
    
    try:
        # Generate both PNG and SVG
        png_file = engine.generate_ra_diagram(
            diagram_data, 
            output_file="UC3_RA_Diagram_With_DataFlows.png",
            title="UC3: Rocket Launch Mission - RA Diagram with Data Flows",
            show_data_flows=True,
            show_legend=True
        )
        
        svg_file = engine.generate_ra_diagram(
            diagram_data, 
            output_file="UC3_RA_Diagram_With_DataFlows.svg",
            title="UC3: Rocket Launch Mission - RA Diagram with Data Flows", 
            show_data_flows=True,
            show_legend=True
        )
        
        print(f"\nRA Diagram with Data Flows for UC3 created successfully!")
        print(f"Files saved:")
        print(f"- UC3_RA_Diagram_With_DataFlows.png")
        print(f"- UC3_RA_Diagram_With_DataFlows.svg")
        print(f"\nData Flow Legend:")
        print(f"Blue dashed lines (--use-->): Controller uses Entity as input")
        print(f"Red dashed lines (--provide-->): Controller provides Entity as output")
        print(f"Black solid lines: Control flow (UC-Methode rules)")
        
    except Exception as e:
        print(f"Error generating diagram: {e}")
        print("Note: This may be due to missing Graphviz installation")
        print("Install Graphviz from: https://graphviz.org/download/")

if __name__ == "__main__":
    main()