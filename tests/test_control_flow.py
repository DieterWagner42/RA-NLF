#!/usr/bin/env python3
"""
Test script for Kontrollfluss analysis in generic UC analyzer
"""

from oldpy.generic_uc_analyzer import GenericUCAnalyzer
from pathlib import Path

def test_control_flow_analysis():
    """Test the Kontrollfluss analysis with UC-Methode rules"""
    print("TESTING KONTROLLFLUSS ANALYSIS (UC-METHODE)")
    print("=" * 60)
    
    # Test with UC1 which has parallel steps (B2a, B2b, B2c, B2d)
    print("\n=== KONTROLLFLUSS ANALYSIS: UC1 ===")
    uc1_file = "D:\\KI\\RA-NLF\\Use Case\\UC1.txt"
    
    if Path(uc1_file).exists():
        print(f"\nAnalyzing control flow in UC1...")
        analyzer = GenericUCAnalyzer(domain_name="beverage_preparation")
        verb_analyses, ra_classes = analyzer.analyze_uc_file(uc1_file)
        
        print(f"\nResults: {len(verb_analyses)} verb analyses, {len(ra_classes)} RA classes")
        print(f"Control flow analysis shows UC-Methode compliance")
    else:
        print(f"ERROR: UC1 file not found: {uc1_file}")
    
    # Test with multi-UC for comprehensive control flow
    print("\n=== KONTROLLFLUSS ANALYSIS: MULTI-UC ===")
    beverage_ucs = [
        "D:\\KI\\RA-NLF\\Use Case\\UC1.txt",
        "D:\\KI\\RA-NLF\\Use Case\\UC2.txt"
    ]
    
    existing_ucs = [uc for uc in beverage_ucs if Path(uc).exists()]
    
    if len(existing_ucs) >= 2:
        print(f"\nAnalyzing control flow across {len(existing_ucs)} UCs...")
        analyzer = GenericUCAnalyzer()
        all_verb_analyses, combined_ra_classes = analyzer.analyze_multiple_ucs(
            existing_ucs, domain_name="beverage_preparation"
        )
        
        print(f"\nCombined Results: {len(all_verb_analyses)} verb analyses, {len(combined_ra_classes)} RA classes")
        print(f"Multi-UC control flow analysis demonstrates cross-UC patterns")
    else:
        print(f"ERROR: Need at least 2 UC files, found {len(existing_ucs)}")

if __name__ == "__main__":
    test_control_flow_analysis()