#!/usr/bin/env python3
"""
Test UC1 + UC2 multi-UC analysis with generic analyzer
"""

from pathlib import Path
from oldpy.generic_uc_analyzer import GenericUCAnalyzer

def test_uc1_uc2_analysis():
    """Test UC1 + UC2 multi-UC analysis"""
    
    print("=== MULTI-UC ANALYSIS: UC1 + UC2 ===")
    print("="*60)
    
    # UC files for same domain (beverage_preparation)
    uc_files = [
        "D:\\KI\\RA-NLF\\Use Case\\UC1.txt",
        "D:\\KI\\RA-NLF\\Use Case\\UC2.txt"
    ]
    
    # Check if files exist
    existing_ucs = [uc for uc in uc_files if Path(uc).exists()]
    
    if len(existing_ucs) < 2:
        print(f"ERROR: Only {len(existing_ucs)} UC files found")
        for uc in uc_files:
            if not Path(uc).exists():
                print(f"  Missing: {uc}")
        return
    
    print(f"Found {len(existing_ucs)} UC files:")
    for uc in existing_ucs:
        print(f"  - {Path(uc).name}")
    
    # Create analyzer and run multi-UC analysis
    print(f"\nRunning multi-UC analysis for beverage_preparation domain...")
    analyzer = GenericUCAnalyzer()
    
    try:
        all_verb_analyses, combined_ra_classes = analyzer.analyze_domain_with_hmi(
            existing_ucs, domain_name="beverage_preparation"
        )
        
        print(f"\n=== MULTI-UC ANALYSIS RESULTS ===")
        print(f"Total verb analyses: {len(all_verb_analyses)}")
        print(f"Total unique RA classes: {len(combined_ra_classes)}")
        
        # Count by UC
        uc1_analyses = [va for va in all_verb_analyses if va.uc_name == "UC1"]
        uc2_analyses = [va for va in all_verb_analyses if va.uc_name == "UC2"]
        
        print(f"\nBreakdown by UC:")
        print(f"  UC1: {len(uc1_analyses)} verb analyses")
        print(f"  UC2: {len(uc2_analyses)} verb analyses")
        
        # Count RA classes by type
        actors = [ra for ra in combined_ra_classes if ra.type == "Actor"]
        boundaries = [ra for ra in combined_ra_classes if ra.type == "Boundary"]
        controllers = [ra for ra in combined_ra_classes if ra.type == "Controller"]
        entities = [ra for ra in combined_ra_classes if ra.type == "Entity"]
        
        print(f"\nRA Classes by type:")
        print(f"  Actors: {len(actors)}")
        print(f"  Boundaries: {len(boundaries)}")
        print(f"  Controllers: {len(controllers)}")
        print(f"  Entities: {len(entities)}")
        
        # Show shared components
        print(f"\n=== SHARED COMPONENTS BETWEEN UC1 + UC2 ===")
        
        # Find shared controllers
        shared_controllers = []
        for controller in controllers:
            uc1_steps = [step for step in controller.steps if step.startswith(('B', 'A', 'E')) and 'UC1' in str(controller.description)]
            uc2_steps = [step for step in controller.steps if step.startswith(('B', 'A', 'E')) and 'UC2' in str(controller.description)]
            
            if 'supply' in controller.name.lower() or 'supply' in controller.description.lower():
                shared_controllers.append(controller.name)
            elif controller.name in ['HMIController', 'HeaterManager', 'FilterManager', 'AmountManager', 'CupManager']:
                shared_controllers.append(controller.name)
        
        print(f"Shared Controllers ({len(set(shared_controllers))}):")
        for controller_name in sorted(set(shared_controllers)):
            print(f"  - {controller_name}")
        
        # Show UC2-specific components
        print(f"\n=== UC2-SPECIFIC COMPONENTS ===")
        
        # Look for compressor/pressure related components
        uc2_specific = []
        for controller in controllers:
            if 'compressor' in controller.name.lower() or 'pressure' in controller.name.lower():
                uc2_specific.append(f"Controller: {controller.name}")
        
        for entity in entities:
            if 'compressor' in entity.name.lower() or 'pressure' in entity.name.lower():
                uc2_specific.append(f"Entity: {entity.name}")
        
        if uc2_specific:
            for component in uc2_specific:
                print(f"  - {component}")
        else:
            print("  - No UC2-specific components detected (may need domain knowledge update)")
        
        print(f"\n=== SUCCESS: Multi-UC analysis completed ===")
        
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_uc1_uc2_analysis()