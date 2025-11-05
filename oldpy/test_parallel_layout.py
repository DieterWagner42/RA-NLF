#!/usr/bin/env python3
"""
Test script to verify that parallel controller layout is correct
"""

import pandas as pd
from src.official_rup_engine import OfficialRUPEngine

def test_parallel_layout():
    """Test that controllers are correctly assigned to parallel groups"""
    print("=== Testing Parallel Layout Correction ===")
    
    csv_file = 'new/UC1_Structured_UC_Steps_RA_Classes.csv'
    df = pd.read_csv(csv_file, sep=';')
    
    # Extract unique controllers per step
    controllers = df[df['RA_Typ'] == 'Controller'][['UC_Schritt', 'RA_Klasse']].drop_duplicates()
    step_to_controller = {}
    for _, row in controllers.iterrows():
        step = row['UC_Schritt']
        ctrl = row['RA_Klasse']
        if step not in step_to_controller:
            step_to_controller[step] = []
        if ctrl not in step_to_controller[step]:
            step_to_controller[step].append(ctrl)
    
    print("Step to Controller mapping:")
    for step, ctrls in step_to_controller.items():
        print(f"  {step}: {ctrls}")
    
    # Expected parallel groups
    expected_b2_parallel = ['WaterManager', 'FilterManager', 'CoffeeManager', 'CupManager']
    expected_b3_parallel = ['BrewingManager', 'MilkManager']
    
    # Actual parallel groups
    actual_b2_controllers = []
    for step in ['B2a', 'B2b', 'B2c', 'B2d']:
        if step in step_to_controller:
            for ctrl in step_to_controller[step]:
                if ctrl not in actual_b2_controllers:
                    actual_b2_controllers.append(ctrl)
    
    actual_b3_controllers = []
    for step in ['B3a', 'B3b']:
        if step in step_to_controller:
            for ctrl in step_to_controller[step]:
                if ctrl not in actual_b3_controllers:
                    actual_b3_controllers.append(ctrl)
    
    print("\n=== Parallel Group Verification ===")
    print(f"Expected B2 parallel: {expected_b2_parallel}")
    print(f"Actual B2 parallel: {actual_b2_controllers}")
    b2_correct = set(expected_b2_parallel) == set(actual_b2_controllers)
    print(f"B2 parallel group correct: {'✅' if b2_correct else '❌'}")
    
    print(f"\nExpected B3 parallel: {expected_b3_parallel}")
    print(f"Actual B3 parallel: {actual_b3_controllers}")
    b3_correct = set(expected_b3_parallel) == set(actual_b3_controllers)
    print(f"B3 parallel group correct: {'✅' if b3_correct else '❌'}")
    
    # Check brewing correction specifically
    b3a_controller = step_to_controller.get('B3a', [])
    brewing_correct = 'BrewingManager' in b3a_controller and 'WaterManager' not in b3a_controller
    print(f"\nB3a brewing correction: {'✅' if brewing_correct else '❌'}")
    print(f"B3a controller: {b3a_controller}")
    
    # Overall result
    all_correct = b2_correct and b3_correct and brewing_correct
    print(f"\n=== Overall Result: {'✅ PASS' if all_correct else '❌ FAIL'} ===")
    
    return all_correct

if __name__ == "__main__":
    test_parallel_layout()