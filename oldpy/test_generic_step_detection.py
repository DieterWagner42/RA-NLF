#!/usr/bin/env python3
"""
Test script for the generic step detection implementation
"""

import pandas as pd
from src.official_rup_engine import OfficialRUPEngine

def test_generic_step_detection():
    """Test the generic step detection to ensure no hard-coding"""
    print("=== Testing Generic Step Detection ===")
    
    csv_file = 'new/UC1_Structured_UC_Steps_RA_Classes.csv'
    df = pd.read_csv(csv_file, sep=';')
    
    # Get all unique steps
    steps = df[df['UC_Schritt'].notna() & (df['UC_Schritt'] != '')]['UC_Schritt'].unique()
    step_list = list(steps)
    print(f"Found {len(step_list)} unique steps: {step_list}")
    
    # Test the RUP engine
    engine = OfficialRUPEngine('UC1')
    
    # Test automatic parallel group detection
    print("\n=== Testing Auto Parallel Group Detection ===")
    groups = engine._auto_detect_parallel_groups(step_list, {})
    print("Detected groups:")
    for group_name, group_steps in groups.items():
        print(f"  {group_name}: {group_steps}")
    
    # Test step parallel candidate detection
    print("\n=== Testing Parallel Candidate Detection ===")
    test_pairs = [
        ('B2a', 'B2b'),
        ('B2c', 'B2d'), 
        ('B3a', 'B3b'),
        ('B1', 'B2a'),  # Should not be parallel
        ('B4', 'B5')    # Should not be parallel
    ]
    
    for step1, step2 in test_pairs:
        is_parallel = engine._steps_are_parallel_candidates(step1, step2)
        print(f"  {step1} + {step2}: {'PARALLEL' if is_parallel else 'SEQUENTIAL'}")
    
    # Test dynamic layout generation
    print("\n=== Testing Dynamic Layout Generation ===")
    try:
        layout_sequence = engine._generate_dynamic_layout_sequence(csv_file)
        print("Generated layout sequence:")
        for i, item in enumerate(layout_sequence[:10]):  # Show first 10 items
            print(f"  {i+1}. {item}")
        if len(layout_sequence) > 10:
            print(f"  ... and {len(layout_sequence) - 10} more items")
        print(f"Total layout items: {len(layout_sequence)}")
    except Exception as e:
        print(f"Error in layout generation: {e}")
    
    print("\n=== Generic Step Detection Test Complete ===")

if __name__ == "__main__":
    test_generic_step_detection()