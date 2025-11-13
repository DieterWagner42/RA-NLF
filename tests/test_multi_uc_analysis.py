#!/usr/bin/env python3
"""
Test script to demonstrate multi-UC analysis capability
Shows how to analyze multiple UCs from the same domain
"""

from oldpy.generic_uc_analyzer import GenericUCAnalyzer
from pathlib import Path

def test_multi_uc_beverage_domain():
    """Test multi-UC analysis for beverage preparation domain"""
    print("MULTI-UC ANALYSIS DEMO - BEVERAGE PREPARATION DOMAIN")
    print("=" * 60)
    
    # UC files from beverage domain
    beverage_ucs = [
        "D:\\KI\\RA-NLF\\Use Case\\UC1.txt",  # Milk Coffee
        "D:\\KI\\RA-NLF\\Use Case\\UC2.txt"   # Espresso
    ]
    
    # Check which files exist
    existing_ucs = [uc for uc in beverage_ucs if Path(uc).exists()]
    
    if len(existing_ucs) < 2:
        print("ERROR: Need at least 2 UC files for multi-UC analysis")
        for uc in beverage_ucs:
            status = "✓ Found" if Path(uc).exists() else "✗ Missing"
            print(f"  {status}: {uc}")
        return
    
    print(f"Found {len(existing_ucs)} UC files for analysis:")
    for i, uc in enumerate(existing_ucs, 1):
        print(f"  {i}. {Path(uc).name}")
    
    # Perform multi-UC analysis
    print(f"\nAnalyzing {len(existing_ucs)} UCs from beverage_preparation domain...")
    analyzer = GenericUCAnalyzer()
    all_verb_analyses, combined_ra_classes = analyzer.analyze_multiple_ucs(
        existing_ucs, domain_name="beverage_preparation"
    )
    
    # Show key results
    print("\n" + "=" * 60)
    print("KEY RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"Total verb analyses across all UCs: {len(all_verb_analyses)}")
    print(f"Unique RA classes identified: {len(combined_ra_classes)}")
    
    # Show verb type distribution
    verb_types = {}
    for va in all_verb_analyses:
        vt = va.verb_type.value
        verb_types[vt] = verb_types.get(vt, 0) + 1
    
    print(f"\nVerb Type Distribution:")
    for verb_type, count in sorted(verb_types.items()):
        print(f"  {verb_type}: {count}")
    
    # Show RA class type distribution
    ra_types = {}
    for ra in combined_ra_classes:
        rt = ra.type
        ra_types[rt] = ra_types.get(rt, 0) + 1
    
    print(f"\nRA Class Type Distribution:")
    for ra_type, count in sorted(ra_types.items()):
        print(f"  {ra_type}: {count}")
    
    # Find shared components (appearing in both UCs)
    shared_classes = []
    for ra_class in combined_ra_classes:
        uc_sources = set()
        for step_ref in ra_class.step_references:
            for verb_analysis in all_verb_analyses:
                if step_ref == verb_analysis.step_id and verb_analysis.uc_name:
                    uc_sources.add(verb_analysis.uc_name)
        
        if len(uc_sources) > 1:
            shared_classes.append((ra_class, uc_sources))
    
    print(f"\nShared Components (appear in multiple UCs): {len(shared_classes)}")
    for ra_class, uc_sources in shared_classes[:5]:  # Show first 5
        print(f"  {ra_class.type}: {ra_class.name} -> {', '.join(sorted(uc_sources))}")
    if len(shared_classes) > 5:
        print(f"  ... and {len(shared_classes) - 5} more")
    
    print(f"\nAnalysis demonstrates domain-wide component reuse and systematic patterns!")

def compare_single_vs_multi_analysis():
    """Compare single UC analysis vs multi-UC analysis"""
    print("\n" + "=" * 60)
    print("COMPARISON: SINGLE vs MULTI-UC ANALYSIS")
    print("=" * 60)
    
    uc_files = [
        "D:\\KI\\RA-NLF\\Use Case\\UC1.txt",
        "D:\\KI\\RA-NLF\\Use Case\\UC2.txt"
    ]
    
    existing_files = [f for f in uc_files if Path(f).exists()]
    if len(existing_files) < 2:
        print("Skipping comparison - not enough UC files")
        return
    
    # Single UC analyses
    analyzer = GenericUCAnalyzer(domain_name="beverage_preparation")
    single_results = []
    
    for uc_file in existing_files:
        verb_analyses, ra_classes = analyzer.analyze_uc_file(uc_file)
        single_results.append((Path(uc_file).stem, len(verb_analyses), len(ra_classes)))
    
    # Multi-UC analysis
    multi_verb_analyses, multi_ra_classes = analyzer.analyze_multiple_ucs(
        existing_files, domain_name="beverage_preparation"
    )
    
    print(f"\nSingle UC Analysis Results:")
    total_single_verbs = 0
    total_single_ras = 0
    for uc_name, verb_count, ra_count in single_results:
        print(f"  {uc_name}: {verb_count} verbs, {ra_count} RA classes")
        total_single_verbs += verb_count
        total_single_ras += ra_count
    
    print(f"  TOTAL (sum): {total_single_verbs} verbs, {total_single_ras} RA classes")
    
    print(f"\nMulti-UC Analysis Results:")
    print(f"  COMBINED: {len(multi_verb_analyses)} verbs, {len(multi_ra_classes)} unique RA classes")
    
    # Calculate savings from component reuse
    ra_savings = total_single_ras - len(multi_ra_classes)
    savings_percent = (ra_savings / total_single_ras) * 100 if total_single_ras > 0 else 0
    
    print(f"\nComponent Reuse Benefits:")
    print(f"  RA classes saved through reuse: {ra_savings}")
    print(f"  Reduction percentage: {savings_percent:.1f}%")
    print(f"  This shows systematic patterns across UCs in the same domain!")

if __name__ == "__main__":
    test_multi_uc_beverage_domain()
    compare_single_vs_multi_analysis()