#!/usr/bin/env python3
"""
Simple test to demonstrate the integrated HMI architecture analysis
"""

from oldpy.generic_uc_analyzer import GenericUCAnalyzer
from pathlib import Path

def test_hmi_integration():
    """Test the integrated HMI architecture analysis"""
    print("TESTING INTEGRATED HMI ARCHITECTURE ANALYSIS")
    print("=" * 60)
    
    # UC files from beverage domain
    beverage_ucs = [
        "D:\\KI\\RA-NLF\\Use Case\\UC1.txt",
        "D:\\KI\\RA-NLF\\Use Case\\UC2.txt"
    ]
    
    # Check which files exist
    existing_ucs = [uc for uc in beverage_ucs if Path(uc).exists()]
    
    if len(existing_ucs) < 2:
        print("ERROR: Need both UC1 and UC2 files")
        return
    
    print(f"Found {len(existing_ucs)} UC files:")
    for uc in existing_ucs:
        print(f"  - {Path(uc).name}")
    
    # Use the new convenience method
    analyzer = GenericUCAnalyzer()
    all_verb_analyses, combined_ra_classes = analyzer.analyze_domain_with_hmi(
        existing_ucs, domain_name="beverage_preparation"
    )
    
    print(f"\n" + "="*60)
    print("INTEGRATION TEST COMPLETED SUCCESSFULLY!")
    print(f"Final Statistics:")
    print(f"  Total Verb Analyses: {len(all_verb_analyses)}")
    print(f"  Total RA Classes: {len(combined_ra_classes)}")
    
    # Show HMI components count
    hmi_controllers = [ra for ra in combined_ra_classes if ra.type == "Controller" and "HMI" in ra.name]
    hmi_boundaries = [ra for ra in combined_ra_classes if ra.type == "Boundary" and "HMI" in ra.name]
    
    print(f"  HMI Controllers: {len(hmi_controllers)}")
    print(f"  HMI Boundaries: {len(hmi_boundaries)}")
    print("="*60)

if __name__ == "__main__":
    test_hmi_integration()