"""
UC1 Robustness Analysis (RA) Diagram Generator
Creates ASCII diagram representation of UC1 Coffee preparation RA analysis results
"""

import json

def generate_uc1_ra_diagram():
    """Generate ASCII RA diagram for UC1 based on analysis results"""
    
    # Load analysis results
    with open("Zwischenprodukte/UC1_Coffee_phase2_analysis.json", "r", encoding="utf-8") as f:
        phase2_data = json.load(f)
    
    with open("Zwischenprodukte/UC1_Coffee_phase3_analysis.json", "r", encoding="utf-8") as f:
        phase3_data = json.load(f)
    
    print("=" * 80)
    print("   UC1: Prepare Milk Coffee - ROBUSTNESS ANALYSIS DIAGRAM")
    print("=" * 80)
    print()
    
    # 1. ACTOR SECTION
    print("ACTORS:")
    print("+-------------+        +-------------+")
    print("|    User     |        |    Timer    |")
    print("|  (Human)    |        | (Non-Human) |") 
    print("+-------------+        +-------------+")
    print("      |                       |")
    print("      | B5: presents          | B1: 7:00h trigger")
    print("      v                       v")
    print()
    
    # 2. BOUNDARY OBJECTS SECTION
    print("BOUNDARY OBJECTS:")
    print("+-----------------------------------------------------------------+")
    print("|                        INPUT BOUNDARIES                         |")
    print("+-----------------+-----------------+-----------------------------+")
    print("| <<boundary>>    | <<boundary>>    | <<boundary>>                |")
    print("| CoffeeBeans     | Water Input     | Milk Input                  |")
    print("| Input           |                 |                             |")
    print("| + accept_input  | + accept_input  | + accept_input              |")
    print("| + validate      | + validate      | + validate                  |")
    print("| + feedback      | + feedback      | + feedback                  |")
    print("+-----------------+-----------------+-----------------------------+")
    print("         |                 |                       |")
    print("         v                 v                       v")
    print()
    
    # 3. CONTROLLER OBJECTS SECTION  
    print("CONTROLLER OBJECTS:")
    print("+-----------------------------------------------------------------+")
    print("|                     ORCHESTRATION LAYER                        |")
    print("| +-------------------------------------------------------------+ |")
    print("| |          <<controller>> GetraenkeOrchestrator               | |")
    print("| |          + coordinate_parallel_steps()                     | |") 
    print("| |          + manage_sequence()                               | |")
    print("| +-------------------------------------------------------------+ |")
    print("+-----------------------------------------------------------------+")
    print("                               |")
    print("           +-------------------+-------------------+")
    print("           v                   v                   v")
    print()
    
    print("+-----------------+-----------------+-----------------------------+")
    print("| <<controller>>  | <<controller>>  | <<controller>>              |")
    print("| CoffeeBeansMan. | WaterManager    | MilkManager                 |")
    print("|                 |                 |                             |")
    print("| + store_beans   | + store_water   | + store_milk                |")
    print("| + monitor_level | + monitor_level | + monitor_level             |")
    print("| + provide_beans | + provide_water | + provide_milk              |")
    print("| + grind_coffee  | + heat_water    | + cool_milk                 |")
    print("| + schedule_use  | + schedule_use  | + steam_milk                |")
    print("| + clean_equip   | + clean_equip   | + schedule_use              |")
    print("|                 |                 | + clean_equip               |")
    print("+-----------------+-----------------+-----------------------------+")
    print("| UC Flow:        | UC Flow:        | UC Flow:                    |")
    print("| B2c: grinds     | B2a: activates  | B3b: adds                   |")
    print("| B3a: begins     | B3a: begins     |                             |")
    print("+-----------------+-----------------+-----------------------------+")
    print("         |                 |                       |")
    print("         v                 v                       v")
    print()
    
    # 4. ENTITY FLOWS
    print("ENTITY FLOWS & TRANSFORMATIONS:")
    print("+-----------------+    transformation    +-----------------+")
    print("| <<entity>>      | --------------------> | <<entity>>      |")
    print("| Kaffeebohnen    |   beans -> grinding   | Kaffeemehl      |")
    print("|                 |   -> ground coffee    |                 |")
    print("+-----------------+                      +-----------------+")
    print("                                                   |")
    print("                                                   v")
    print("+-----------------+                      +-----------------+")
    print("| <<controller>>  |                      | <<controller>>  |") 
    print("| FilterManager   | <-------------------- | KaffeeManager   |")
    print("| (internal)      |                      | (B2c reference) |")
    print("+-----------------+                      +-----------------+")
    print()
    
    # 5. OUTPUT BOUNDARIES
    print("OUTPUT BOUNDARIES:")
    print("+-----------------+-------------------------------------------------+")
    print("| <<boundary>>    | <<boundary>>                                    |")
    print("| CoffeeBeans     | User Presentation                               |")
    print("| Waste Output    |                                                 |")
    print("|                 | B4: outputs message                            |")
    print("| + dispose_waste | B5: presents cup                               |")
    print("| + manage_waste  |                                                 |")
    print("| + constraints   |                                                 |")
    print("+-----------------+-------------------------------------------------+")
    print()
    
    # 6. INTERACTION FLOWS
    print("INTERACTION FLOWS (Phase 3):")
    print("-" * 80)
    print("1. TRIGGER (B1): Zeit --> ZeitManager")
    print("   +-- Time trigger activates system at 7:00h")
    print()
    print("2. CONTROL FLOWS:")
    print("   B3a --> B3b: CoffeeBeansAreManager --> MilkIsManager")
    print("   B3a --> B3b: WaterIsManager --> MilkIsManager")
    print()
    print("3. ORCHESTRATION PATTERN:")
    print("   GetraenkeOrchestrator coordinates:")
    print("   +-- WaterIsManager (B2a, B3a)")
    print("   +-- CoffeeBeansAreManager (B3a)") 
    print("   +-- MilkIsManager (B3b)")
    print()
    
    # 7. UC STEP MAPPING
    print("UC STEP TO RA OBJECT MAPPING:")
    print("-" * 80)
    step_mappings = [
        ("B1", "Zeit -> ZeitManager", "Trigger"),
        ("B2a", "WaterIsManager.activates()", "Parallel"),
        ("B2b", "FilterManager (internal)", "Parallel"),
        ("B2c", "CoffeeManager.grinds() -> Kaffeemehl", "Parallel"),
        ("B2d", "TassenManager (internal)", "Parallel"),
        ("B3a", "WaterManager + CoffeeManager", "Parallel"),
        ("B3b", "MilkManager.adds()", "Parallel"),
        ("B4", "HMI.output_message()", "Sequential"),
        ("B5", "HMI.present_cup()", "Sequential"),
        ("B6", "End", "Sequential")
    ]
    
    for step, mapping, step_type in step_mappings:
        print(f"{step:4} | {mapping:35} | {step_type}")
    
    print()
    print("=" * 80)
    print("RA OBJECTS SUMMARY:")
    print(f"Input Boundaries:  6 objects")
    print(f"Manager Controllers: 6 objects") 
    print(f"Output Boundaries: 2 objects")
    print(f"Entities: {len(phase3_data.get('entity_flows', []))} flows")
    print(f"Interactions: {len(phase3_data.get('interactions', []))} patterns")
    print(f"Total RA Objects: 14 classes")
    print()
    print("PHASE SUMMARIES:")
    print(f"Phase 1: {phase3_data['phase1_summary']}")
    print(f"Phase 2: {phase3_data['phase2_summary']}")
    print(f"Phase 3: {phase3_data['phase3_summary']}")
    print("=" * 80)

if __name__ == "__main__":
    generate_uc1_ra_diagram()