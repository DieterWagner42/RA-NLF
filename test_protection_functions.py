"""
Test script for implicit protection functions
"""

import sys
import os
sys.path.insert(0, 'src')

from domain_verb_loader import DomainVerbLoader
import json

def test_load_protection_functions():
    """Test loading implicit protection functions from domain JSON"""
    print("=" * 80)
    print("TEST: Loading Implicit Protection Functions")
    print("=" * 80)

    loader = DomainVerbLoader()
    protection_funcs = loader.get_implicit_protection_functions('beverage_preparation')

    if not protection_funcs:
        print("ERROR: No protection functions loaded!")
        return

    print(f"\nLoaded protection functions for {len(protection_funcs)} materials:")
    print(json.dumps(protection_funcs, indent=2))

    # Test trigger pattern matching
    print("\n" + "=" * 80)
    print("TEST: Trigger Pattern Matching")
    print("=" * 80)

    test_cases = [
        ("Water", "The system activates the water heater", ["OverheatProtection"]),
        ("Water", "The system starts water compressor", ["OverpressureProtection"]),
        ("Water", "Water is available in the system", ["WaterQualityProtection"]),
        ("Water", "Water heater has too little water", ["DryRunProtection"]),
        ("Milk", "Milk is available in the system", ["TemperatureProtection", "FreshnessProtection"]),
        ("Milk", "The system adds milk to the cup", ["LineHygieneProtection"]),
        ("Sugar", "Sugar is available in the system", ["MoistureProtection"]),
        ("Sugar", "The system adds defined amount of sugar", ["DosingPrecisionProtection"]),
        ("CoffeeBeans", "Coffee beans are available in the system", ["FreshnessProtection"]),
    ]

    for material, text, expected_functions in test_cases:
        print(f"\nMaterial: {material}")
        print(f"Text: '{text}'")
        print(f"Expected Functions: {expected_functions}")

        # Find matching protection functions
        matched_functions = find_matching_protection_functions(
            material, text, protection_funcs
        )

        print(f"Matched Functions: {matched_functions}")

        # Check if all expected functions were matched
        for expected in expected_functions:
            if expected in matched_functions:
                print(f"  [OK] {expected} - MATCHED")
            else:
                print(f"  [FAIL] {expected} - NOT MATCHED")


def find_matching_protection_functions(material: str, text: str, protection_funcs: dict) -> list:
    """
    Find protection functions that should be triggered based on text patterns.

    Args:
        material: Material name (e.g., 'water', 'milk')
        text: UC step text to analyze
        protection_funcs: Dict of protection functions from domain JSON

    Returns:
        List of matched protection function names
    """
    matched = []
    text_lower = text.lower()
    material_lower = material.lower()

    # Map material names to protection function keys
    material_mapping = {
        'water': 'water',
        'milk': 'milk',
        'sugar': 'sugar',
        'coffeebeans': 'coffee_beans',
        'coffee beans': 'coffee_beans',
        'filter': 'filter'
    }

    protection_key = material_mapping.get(material_lower)
    if not protection_key or protection_key not in protection_funcs:
        return matched

    material_protections = protection_funcs[protection_key]

    # Check all function types (safety_functions, hygiene_functions, quality_functions)
    for func_type in ['safety_functions', 'hygiene_functions', 'quality_functions']:
        if func_type not in material_protections:
            continue

        for protection_func in material_protections[func_type]:
            func_name = protection_func['name']
            trigger_patterns = protection_func.get('trigger_patterns', [])

            # Check if any trigger pattern matches the text
            for pattern in trigger_patterns:
                if pattern.lower() in text_lower:
                    matched.append(func_name)
                    break  # Don't match same function multiple times

    return matched


if __name__ == "__main__":
    test_load_protection_functions()
