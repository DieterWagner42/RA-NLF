#!/usr/bin/env python3
"""
Extract and display boundary components from multi-UC analysis
Shows how boundaries are identified for beverage preparation domain
"""

from generic_uc_analyzer import GenericUCAnalyzer
from pathlib import Path

def show_boundaries_analysis():
    """Extract and display boundary analysis for UC1 and UC2"""
    print("BOUNDARY ANALYSIS - BEVERAGE PREPARATION DOMAIN")
    print("=" * 60)
    
    # UC files from beverage domain
    beverage_ucs = [
        "D:\\KI\\RA-NLF\\Use Case\\UC1.txt",  # Milk Coffee
        "D:\\KI\\RA-NLF\\Use Case\\UC2.txt"   # Espresso
    ]
    
    # Check which files exist
    existing_ucs = [uc for uc in beverage_ucs if Path(uc).exists()]
    
    if len(existing_ucs) < 2:
        print("ERROR: Need UC1 and UC2 files")
        return
    
    # Perform multi-UC analysis
    analyzer = GenericUCAnalyzer()
    all_verb_analyses, combined_ra_classes = analyzer.analyze_multiple_ucs(
        existing_ucs, domain_name="beverage_preparation"
    )
    
    # Filter for boundary components only
    boundaries = [ra for ra in combined_ra_classes if ra.type == "Boundary"]
    
    print(f"\nFOUND {len(boundaries)} BOUNDARY COMPONENTS:")
    print("=" * 60)
    
    for boundary in boundaries:
        print(f"\nBoundary: {boundary.name}")
        print(f"  Stereotype: {boundary.stereotype}")
        print(f"  Element Type: {boundary.element_type.value}")
        print(f"  Description: {boundary.description}")
        print(f"  Steps Referenced: {', '.join(boundary.step_references)}")
        
        # Show which UCs use this boundary
        uc_sources = set()
        for step_ref in boundary.step_references:
            for verb_analysis in all_verb_analyses:
                if step_ref == verb_analysis.step_id and verb_analysis.uc_name:
                    uc_sources.add(verb_analysis.uc_name)
        
        print(f"  Used in UCs: {', '.join(sorted(uc_sources))}")
        
        # Show the actual verb activities that led to this boundary
        related_verbs = []
        for step_ref in boundary.step_references:
            for verb_analysis in all_verb_analyses:
                if step_ref == verb_analysis.step_id:
                    related_verbs.append(f"{step_ref}: {verb_analysis.verb} - {verb_analysis.original_text}")
        
        print(f"  Related Activities:")
        for verb_info in related_verbs:
            print(f"    {verb_info}")
    
    # Show transaction verbs that led to boundaries
    print(f"\n" + "=" * 60)
    print("TRANSACTION VERBS (Lead to Boundary Identification)")
    print("=" * 60)
    
    transaction_verbs = [v for v in all_verb_analyses if v.verb_type.value == "transaction"]
    
    for verb_analysis in transaction_verbs:
        print(f"\n{verb_analysis.step_id} ({verb_analysis.uc_name}): {verb_analysis.verb}")
        print(f"  Text: {verb_analysis.original_text}")
        print(f"  Direct Object: {verb_analysis.direct_object}")
        print(f"  Prepositional Objects: {verb_analysis.prepositional_objects}")
        
        # Show if this led to a boundary
        related_boundaries = []
        for boundary in boundaries:
            if verb_analysis.step_id in boundary.step_references:
                related_boundaries.append(boundary.name)
        
        if related_boundaries:
            print(f"  → Generated Boundary: {', '.join(related_boundaries)}")
        else:
            print(f"  → No boundary generated (may be actor or other)")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("BOUNDARY SUMMARY")
    print("=" * 60)
    print(f"Total Boundaries Identified: {len(boundaries)}")
    print(f"Total Transaction Verbs: {len(transaction_verbs)}")
    
    # Show boundary types
    boundary_types = {}
    for boundary in boundaries:
        desc_key = boundary.name.replace("Boundary", "").strip()
        boundary_types[desc_key] = boundary_types.get(desc_key, 0) + 1
    
    print(f"\nBoundary Types:")
    for boundary_type, count in boundary_types.items():
        print(f"  {boundary_type}: {count}")
    
    # Show coverage across UCs
    print(f"\nBoundary Coverage:")
    for boundary in boundaries:
        uc_sources = set()
        for step_ref in boundary.step_references:
            for verb_analysis in all_verb_analyses:
                if step_ref == verb_analysis.step_id and verb_analysis.uc_name:
                    uc_sources.add(verb_analysis.uc_name)
        
        coverage = "Shared" if len(uc_sources) > 1 else "Specific"
        print(f"  {boundary.name}: {coverage} ({', '.join(sorted(uc_sources))})")

if __name__ == "__main__":
    show_boundaries_analysis()