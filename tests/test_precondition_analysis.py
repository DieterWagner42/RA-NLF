#!/usr/bin/env python3
"""
Test script for enhanced precondition analysis in generic UC analyzer
"""

from generic_uc_analyzer import GenericUCAnalyzer
from pathlib import Path

def test_precondition_analysis():
    """Test the enhanced precondition analysis functionality"""
    print("TESTING ENHANCED PRECONDITION ANALYSIS")
    print("=" * 60)
    
    # Test single UC with preconditions
    print("\n=== TEST 1: SINGLE UC WITH PRECONDITIONS ===")
    uc1_file = "D:\\KI\\RA-NLF\\Use Case\\UC1.txt"
    
    if Path(uc1_file).exists():
        print(f"\nAnalyzing UC1 with precondition parsing...")
        analyzer = GenericUCAnalyzer(domain_name="beverage_preparation")
        verb_analyses, ra_classes = analyzer.analyze_uc_file(uc1_file)
        
        print(f"\nResults: {len(verb_analyses)} verb analyses, {len(ra_classes)} RA classes")
        
        # Show precondition-derived RA classes
        precondition_classes = [ra for ra in ra_classes if ra.source == "precondition"]
        print(f"Precondition-derived RA classes: {len(precondition_classes)}")
        
        for ra_class in precondition_classes:
            print(f"  - {ra_class.type}: {ra_class.name}")
            print(f"    Description: {ra_class.description}")
    else:
        print(f"ERROR: UC1 file not found: {uc1_file}")
    
    # Test multi-UC analysis with preconditions
    print("\n=== TEST 2: MULTI-UC WITH PRECONDITIONS ===")
    beverage_ucs = [
        "D:\\KI\\RA-NLF\\Use Case\\UC1.txt",
        "D:\\KI\\RA-NLF\\Use Case\\UC2.txt"
    ]
    
    existing_ucs = [uc for uc in beverage_ucs if Path(uc).exists()]
    
    if len(existing_ucs) >= 2:
        print(f"\nAnalyzing {len(existing_ucs)} UCs with precondition parsing...")
        analyzer = GenericUCAnalyzer()
        all_verb_analyses, combined_ra_classes = analyzer.analyze_multiple_ucs(
            existing_ucs, domain_name="beverage_preparation"
        )
        
        print(f"\nCombined Results: {len(all_verb_analyses)} verb analyses, {len(combined_ra_classes)} RA classes")
        
        # Show all precondition-derived RA classes
        precondition_classes = [ra for ra in combined_ra_classes if ra.source == "precondition"]
        step_classes = [ra for ra in combined_ra_classes if ra.source == "step"]
        
        print(f"\nRA Class Sources:")
        print(f"  From UC steps: {len(step_classes)}")
        print(f"  From preconditions: {len(precondition_classes)}")
        
        if precondition_classes:
            print(f"\nPrecondition-derived components:")
            controllers = [ra for ra in precondition_classes if ra.type == "Controller"]
            entities = [ra for ra in precondition_classes if ra.type == "Entity"]
            boundaries = [ra for ra in precondition_classes if ra.type == "Boundary"]
            
            print(f"  Supply Controllers: {[c.name for c in controllers]}")
            print(f"  Resource Entities: {[e.name for e in entities]}")
            print(f"  Supply Boundaries: {[b.name for b in boundaries]}")
    else:
        print(f"ERROR: Need at least 2 UC files, found {len(existing_ucs)}")

if __name__ == "__main__":
    test_precondition_analysis()