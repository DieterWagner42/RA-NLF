#!/usr/bin/env python3
"""
Generate UC1 Specific RA Diagram using official RUP engine
"""

import sys
import os
sys.path.append('src')

from official_rup_engine import OfficialRUPEngine

def generate_uc1_diagram():
    """Generate UC1 RA diagram from the generated JSON"""
    json_file = "UC1_Structured_RA_Analysis.json"
    
    if not os.path.exists(json_file):
        print(f"UC1 JSON file not found: {json_file}")
        return None
    
    print(f"Generating UC1 RA diagram from: {json_file}")
    
    # Initialize official RUP engine
    engine = OfficialRUPEngine(figure_size=(24, 20))
    
    # Generate the diagram
    output_file = engine.create_official_rup_diagram_from_json(json_file)
    
    print(f"UC1 RA diagram generated: {output_file}")
    return output_file

def main():
    print("UC1 Enhanced RA Diagram with Generative Context")
    print("=" * 60)
    
    # Generate the diagram
    diagram_path = generate_uc1_diagram()
    
    if diagram_path:
        print("\nSUCCESS!")
        print(f"UC1 Enhanced RA Diagram: {diagram_path}")
        print("\nFeatures shown in diagram:")
        print("- Original UC1 analyzed with generative context system")
        print("- Operational materials (Water, Milk, Sugar) with food-grade classification")
        print("- Safety/hygiene controllers (FoodSafetyController, HACCPController)")
        print("- Supply chain boundaries for material traceability")
        print("- Technical context controllers (Time, Thermal, Mechanical, etc.)")
        print("- No hardcoded contexts - all generated from domain knowledge")
    else:
        print("Failed to generate UC1 diagram")

if __name__ == "__main__":
    main()