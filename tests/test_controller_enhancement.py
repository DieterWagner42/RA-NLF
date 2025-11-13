#!/usr/bin/env python3
"""
Test Controller Enhancement with Safety/Hygiene Functions
Demonstrates automatic generalization and function assignment
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path
from oldpy.generic_uc_analyzer import GenericUCAnalyzer

def test_controller_enhancement():
    """Test controller enhancement with safety/hygiene functions"""
    
    print("="*80)
    print("CONTROLLER ENHANCEMENT TEST")
    print("="*80)
    
    print("\nTesting UC1 with enhanced controller analysis...")
    
    uc_file_path = "D:\\KI\\RA-NLF\\Use Case\\UC1.txt"
    
    if not Path(uc_file_path).exists():
        print(f"ERROR: UC file not found: {uc_file_path}")
        return
    
    try:
        # Initialize analyzer 
        analyzer = GenericUCAnalyzer(domain_name="beverage_preparation")
        
        # STEP 1: Standard analysis (before enhancement)
        print(f"\n{'='*60}")
        print("STEP 1: STANDARD ANALYSIS (Before Enhancement)")
        print(f"{'='*60}")
        
        verb_analyses, original_ra_classes = analyzer.analyze_uc_file(uc_file_path)
        
        original_controllers = [ra for ra in original_ra_classes if ra.type == "Controller"]
        print(f"\nOriginal Controllers ({len(original_controllers)}):")
        for controller in original_controllers:
            print(f"  - {controller.name}: {controller.description}")
        
        # STEP 2: Enhanced analysis (with safety/hygiene functions)
        print(f"\n{'='*60}")
        print("STEP 2: ENHANCED ANALYSIS (With Safety/Hygiene Functions)")
        print(f"{'='*60}")
        
        verb_analyses, enhanced_ra_classes, operational_materials, safety_constraints, hygiene_requirements = \
            analyzer.analyze_uc_with_safety_hygiene(uc_file_path)
        
        enhanced_controllers = [ra for ra in enhanced_ra_classes if ra.type == "Controller"]
        
        # STEP 3: Compare results
        print(f"\n{'='*60}")
        print("STEP 3: COMPARISON RESULTS")
        print(f"{'='*60}")
        
        print(f"\nController Count:")
        print(f"  Before: {len(original_controllers)} controllers")
        print(f"  After:  {len(enhanced_controllers)} controllers")
        
        # Show generalization mapping
        print(f"\nController Generalizations:")
        original_names = {c.name for c in original_controllers}
        enhanced_names = {c.name for c in enhanced_controllers}
        
        for original_controller in original_controllers:
            enhanced_controller = next((c for c in enhanced_controllers 
                                      if c.step_references == original_controller.step_references), None)
            if enhanced_controller and enhanced_controller.name != original_controller.name:
                print(f"  {original_controller.name} -> {enhanced_controller.name}")
        
        # Show new controllers
        new_controllers = [c for c in enhanced_controllers if c.source == "safety_hygiene_analysis"]
        if new_controllers:
            print(f"\nNew Safety/Hygiene Controllers ({len(new_controllers)}):")
            for controller in new_controllers:
                print(f"  + {controller.name}: {controller.description}")
        
        # Show enhanced functions
        print(f"\nEnhanced Controllers with Safety/Hygiene Functions:")
        for controller in enhanced_controllers:
            if "Safety/Hygiene:" in (controller.description or ""):
                print(f"  {controller.name}:")
                print(f"    {controller.description}")
        
        # STEP 4: Demonstrate the milk refrigeration case
        print(f"\n{'='*60}")
        print("STEP 4: MILK REFRIGERATION CASE STUDY")
        print(f"{'='*60}")
        
        milk_material = next((m for m in operational_materials if m.material_name.lower() == "milk"), None)
        if milk_material:
            print(f"\nMilk Material Analysis:")
            print(f"  Safety Class: {milk_material.safety_class}")
            print(f"  Hygiene Level: {milk_material.hygiene_level}")
            print(f"  Storage Conditions: {milk_material.storage_conditions}")
            print(f"  Special Requirements: {milk_material.special_requirements}")
            
            # Find controller responsible for milk temperature control
            temp_controllers = [c for c in enhanced_controllers 
                              if "temperature" in c.name.lower() or 
                                 ("temperature" in (c.description or "").lower() and "milk" in (c.description or "").lower())]
            
            if temp_controllers:
                temp_controller = temp_controllers[0]
                print(f"\n✅ SOLUTION FOUND:")
                print(f"  Controller: {temp_controller.name}")
                print(f"  Description: {temp_controller.description}")
                print(f"  Functions: Temperature control for milk storage")
                print(f"  Step References: {temp_controller.step_references}")
            else:
                print(f"\n❌ NO TEMPERATURE CONTROLLER FOUND")
                print(f"  The system should have created a TemperatureController for milk!")
        
        # STEP 5: Show specific examples
        print(f"\n{'='*60}")
        print("STEP 5: SPECIFIC EXAMPLES")
        print(f"{'='*60}")
        
        examples = [
            ("MilkManager -> StorageManager", "Milk requires refrigerated storage"),
            ("HeaterManager -> TemperatureController", "Temperature control generalization"),
            ("CoffeeBeansSupplyController -> SupplyController", "Generic supply management"),
            ("Sugar hygiene functions", "Food-grade handling requirements")
        ]
        
        print(f"\nExpected Enhancements:")
        for example, explanation in examples:
            print(f"  {example}: {explanation}")
        
        # STEP 6: Generate enhanced graph
        print(f"\n{'='*60}")
        print("STEP 6: GRAPH GENERATION")
        print(f"{'='*60}")
        
        try:
            dot_file = analyzer.generate_complete_graph(uc_file_path, "output")
            print(f"\n✅ Enhanced graph generated: {dot_file}")
            print(f"   Includes generalized controllers and safety/hygiene functions")
        except Exception as e:
            print(f"❌ Graph generation failed: {e}")
        
        print(f"\n{'='*80}")
        print("CONTROLLER ENHANCEMENT TEST COMPLETED")
        print(f"{'='*80}")
        
        return enhanced_controllers, operational_materials
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def demonstrate_generalization_patterns():
    """Demonstrate the controller generalization patterns"""
    
    print(f"\n{'='*80}")
    print("CONTROLLER GENERALIZATION PATTERNS")
    print(f"{'='*80}")
    
    patterns = [
        ("MilkManager", "StorageManager", "Milk needs refrigerated storage"),
        ("SugarManager", "AdditiveManager", "Sugar is a beverage additive"),
        ("CoffeeManager", "ProcessController", "Coffee processing operations"),
        ("HeaterManager", "TemperatureController", "Temperature control systems"),
        ("FilterManager", "ProcessController", "Filter processing operations"),
        ("CupManager", "ContainerManager", "Container handling operations"),
        ("WaterSupplyController", "SupplyController", "Generic supply management"),
        ("CoffeeBeansSupplyController", "SupplyController", "Generic supply management"),
        ("A1ConditionManager", "ConditionManager", "Generic condition handling"),
        ("ActionsManager", "ProcessController", "Process control operations")
    ]
    
    print(f"\n{'Original Controller':<25} {'Generalized Controller':<20} {'Rationale'}")
    print("-" * 80)
    
    for original, generalized, rationale in patterns:
        print(f"{original:<25} {generalized:<20} {rationale}")
    
    print(f"\nBenefits of Generalization:")
    print(f"  ✅ Reduces controller proliferation")
    print(f"  ✅ Groups related functions logically")
    print(f"  ✅ Enables function reuse across materials")
    print(f"  ✅ Simplifies system architecture")
    print(f"  ✅ Facilitates safety/hygiene function assignment")

def demonstrate_function_assignment():
    """Demonstrate safety/hygiene function assignment logic"""
    
    print(f"\n{'='*80}")
    print("SAFETY/HYGIENE FUNCTION ASSIGNMENT")
    print(f"{'='*80}")
    
    function_examples = [
        {
            "material": "Milk",
            "safety_class": "standard", 
            "hygiene_level": "food_grade",
            "functions": [
                "maintain_food_safety_standards_milk",
                "monitor_temperature_control_milk",
                "control_contamination_prevention_milk",
                "validate_cleaning_procedures_milk"
            ],
            "target_controller": "StorageManager (generalized from MilkManager)"
        },
        {
            "material": "Water",
            "safety_class": "standard",
            "hygiene_level": "food_grade", 
            "functions": [
                "maintain_food_safety_standards_water",
                "monitor_temperature_control_water",
                "control_contamination_prevention_water",
                "validate_cleaning_procedures_water"
            ],
            "target_controller": "TemperatureController (generalized from HeaterManager)"
        }
    ]
    
    for example in function_examples:
        print(f"\n{example['material']} ({example['safety_class']}/{example['hygiene_level']}):")
        print(f"  Target Controller: {example['target_controller']}")
        print(f"  Functions Added:")
        for func in example['functions']:
            print(f"    + {func}")
    
    print(f"\nFunction Assignment Rules:")
    print(f"  🎯 Temperature functions -> TemperatureController")
    print(f"  🎯 Storage functions -> StorageManager") 
    print(f"  🎯 Monitoring functions -> existing material controller")
    print(f"  🎯 Pressure functions -> PressureController")
    print(f"  🎯 Contamination functions -> ContaminationController")

if __name__ == "__main__":
    # Run the main test
    enhanced_controllers, operational_materials = test_controller_enhancement()
    
    # Show demonstration patterns
    demonstrate_generalization_patterns()
    demonstrate_function_assignment()
    
    print(f"\n🎉 DEMO COMPLETED: Controller enhancement with safety/hygiene functions!")