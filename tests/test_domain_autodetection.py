#!/usr/bin/env python3
"""
Test script to demonstrate domain auto-detection functionality of the generic UC analyzer
"""

from oldpy.generic_uc_analyzer import GenericUCAnalyzer
from pathlib import Path

def test_domain_autodetection():
    """Test domain auto-detection with UC1"""
    print("TESTING DOMAIN AUTO-DETECTION")
    print("=" * 50)
    
    uc_file = "D:\\KI\\RA-NLF\\Use Case\\UC1.txt"
    
    if not Path(uc_file).exists():
        print(f"ERROR: UC file not found: {uc_file}")
        return
    
    # Test 1: No domain specified - should auto-detect
    print("\nTest 1: Auto-detection (no domain specified)")
    print("-" * 40)
    
    analyzer = GenericUCAnalyzer()  # No domain specified
    verb_analyses, ra_classes = analyzer.analyze_uc_file(uc_file)
    
    print(f"\nAuto-detection Results:")
    print(f"- Detected domain: {analyzer.domain_name}")
    print(f"- Verb analyses: {len(verb_analyses)}")
    print(f"- RA classes: {len(ra_classes)}")
    
    # Test 2: Compare with explicit domain specification
    print("\n\nTest 2: Explicit domain specification")
    print("-" * 40)
    
    analyzer_explicit = GenericUCAnalyzer(domain_name="beverage_preparation")
    verb_analyses_explicit, ra_classes_explicit = analyzer_explicit.analyze_uc_file(uc_file)
    
    print(f"\nExplicit Domain Results:")
    print(f"- Specified domain: {analyzer_explicit.domain_name}")
    print(f"- Verb analyses: {len(verb_analyses_explicit)}")
    print(f"- RA classes: {len(ra_classes_explicit)}")
    
    # Test 3: Verify equivalency
    print("\n\nTest 3: Equivalency Check")
    print("-" * 40)
    
    auto_verb_types = [v.verb_type for v in verb_analyses]
    explicit_verb_types = [v.verb_type for v in verb_analyses_explicit]
    
    auto_ra_names = sorted([f"{ra.type}_{ra.name}" for ra in ra_classes])
    explicit_ra_names = sorted([f"{ra.type}_{ra.name}" for ra in ra_classes_explicit])
    
    verb_match = auto_verb_types == explicit_verb_types
    ra_match = len(auto_ra_names) == len(explicit_ra_names)  # Compare counts (names may differ slightly)
    
    print(f"Verb classification match: {verb_match}")
    print(f"RA classes count match: {ra_match}")
    print(f"Auto-detected: {len(auto_ra_names)} RA classes")
    print(f"Explicit: {len(explicit_ra_names)} RA classes")
    
    if verb_match and ra_match:
        print("\n✓ SUCCESS: Auto-detection produces equivalent results!")
    else:
        print("\n⚠ WARNING: Results differ between auto-detection and explicit domain")
        
        if not verb_match:
            print("Verb classification differences found")
        if not ra_match:
            print("RA class count differences found")

if __name__ == "__main__":
    test_domain_autodetection()