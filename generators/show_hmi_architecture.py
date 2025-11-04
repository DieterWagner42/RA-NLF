#!/usr/bin/env python3
"""
Show HMI architecture: HMI Controller and connected boundaries
"""

from generic_uc_analyzer import GenericUCAnalyzer
from pathlib import Path

def show_hmi_architecture():
    """Show the HMI Controller and connected input/output boundaries"""
    print("HMI ARCHITECTURE ANALYSIS - BEVERAGE PREPARATION DOMAIN")
    print("=" * 70)
    
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
    
    # Filter for HMI-related components
    hmi_controllers = [ra for ra in combined_ra_classes if ra.type == "Controller" and "HMI" in ra.name]
    hmi_boundaries = [ra for ra in combined_ra_classes if ra.type == "Boundary" and "HMI" in ra.name]
    other_boundaries = [ra for ra in combined_ra_classes if ra.type == "Boundary" and "HMI" not in ra.name]
    
    print(f"\nHMI CONTROLLER:")
    print("=" * 50)
    
    if hmi_controllers:
        for controller in hmi_controllers:
            print(f"\nController: {controller.name}")
            print(f"  Description: {controller.description}")
            print(f"  Steps Referenced: {', '.join(controller.step_references)}")
            
            # Show which UCs use this controller
            uc_sources = set()
            for step_ref in controller.step_references:
                for verb_analysis in all_verb_analyses:
                    if step_ref == verb_analysis.step_id and verb_analysis.uc_name:
                        uc_sources.add(verb_analysis.uc_name)
            
            print(f"  Used in UCs: {', '.join(sorted(uc_sources))}")
            
            # Show related activities
            related_activities = []
            for step_ref in controller.step_references:
                for verb_analysis in all_verb_analyses:
                    if step_ref == verb_analysis.step_id:
                        related_activities.append(f"{step_ref}: {verb_analysis.original_text}")
            
            print(f"  Related HMI Activities:")
            for activity in related_activities:
                print(f"    - {activity}")
    else:
        print("No HMI Controllers found")
    
    print(f"\nHMI INPUT BOUNDARIES (User -> System):")
    print("=" * 50)
    
    input_boundaries = [b for b in hmi_boundaries if "Input" in b.name or "Selection" in b.name]
    
    for boundary in input_boundaries:
        print(f"\nBoundary: {boundary.name}")
        print(f"  Description: {boundary.description}")
        print(f"  Steps Referenced: {', '.join(boundary.step_references)}")
        
        # Show specific user inputs
        for step_ref in boundary.step_references:
            for verb_analysis in all_verb_analyses:
                if step_ref == verb_analysis.step_id:
                    print(f"  User Input: {verb_analysis.original_text}")
    
    print(f"\nHMI OUTPUT BOUNDARIES (System -> User):")
    print("=" * 50)
    
    output_boundaries = [b for b in hmi_boundaries if "Display" in b.name or "Output" in b.name]
    
    for boundary in output_boundaries:
        print(f"\nBoundary: {boundary.name}")
        print(f"  Description: {boundary.description}")
        print(f"  Steps Referenced: {', '.join(boundary.step_references)}")
        
        # Show specific system outputs
        for step_ref in boundary.step_references:
            for verb_analysis in all_verb_analyses:
                if step_ref == verb_analysis.step_id:
                    print(f"  System Output: {verb_analysis.original_text}")
    
    print(f"\nOTHER SYSTEM BOUNDARIES (Non-HMI):")
    print("=" * 50)
    
    for boundary in other_boundaries:
        print(f"\nBoundary: {boundary.name}")
        print(f"  Description: {boundary.description}")
        print(f"  Element Type: {boundary.element_type.value}")
    
    # HMI Architecture Summary
    print(f"\n" + "=" * 70)
    print("HMI ARCHITECTURE SUMMARY")
    print("=" * 70)
    
    print(f"\nHMI Components Found:")
    print(f"  HMI Controllers: {len(hmi_controllers)}")
    print(f"  HMI Input Boundaries: {len(input_boundaries)}")
    print(f"  HMI Output Boundaries: {len(output_boundaries)}")
    print(f"  Other System Boundaries: {len(other_boundaries)}")
    
    print(f"\nHMI Interaction Patterns:")
    
    # Analyze user inputs
    user_inputs = []
    for boundary in input_boundaries:
        for step_ref in boundary.step_references:
            for verb_analysis in all_verb_analyses:
                if step_ref == verb_analysis.step_id:
                    if "espresso" in verb_analysis.original_text.lower():
                        user_inputs.append("Beverage Type Selection (Espresso)")
                    elif "sugar" in verb_analysis.original_text.lower():
                        user_inputs.append("Additive Selection (Sugar)")
    
    # Analyze system outputs
    system_outputs = []
    for boundary in output_boundaries:
        for step_ref in boundary.step_references:
            for verb_analysis in all_verb_analyses:
                if step_ref == verb_analysis.step_id:
                    if "error" in verb_analysis.original_text.lower():
                        if "water" in verb_analysis.original_text.lower():
                            system_outputs.append("Error Message: 'Water Low'")
                        elif "milk" in verb_analysis.original_text.lower():
                            system_outputs.append("Error Message: 'No Milk'")
                        else:
                            system_outputs.append("Error Message: General")
                    elif "message" in verb_analysis.original_text.lower():
                        system_outputs.append("Status Message: 'Coffee Ready'")
    
    print(f"\n  User Input Types:")
    for input_type in set(user_inputs):
        print(f"    - {input_type}")
    
    print(f"\n  System Output Types:")
    for output_type in set(system_outputs):
        print(f"    - {output_type}")
    
    print(f"\nExample HMI Interactions:")
    print(f"  Input:  User selects 'Espresso' -> HMIBeverageSelectionBoundary -> HMIController")
    print(f"  Output: HMIController -> HMIErrorDisplayBoundary -> Display 'No Milk'")
    print(f"  Input:  User selects 'Add Sugar' -> HMIAdditiveInputBoundary -> HMIController")
    print(f"  Output: HMIController -> HMIStatusDisplayBoundary -> Display 'Coffee Ready'")

if __name__ == "__main__":
    show_hmi_architecture()