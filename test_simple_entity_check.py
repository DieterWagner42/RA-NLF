#!/usr/bin/env python3
"""
Simple test that only tests the domain loader part to verify implementation element checking
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test just the domain loader
from domain_verb_loader import DomainVerbLoader

def test_modified_analyzer_concept():
    """
    Test the concept of what our modifications should do
    This simulates what the _derive_generic_entities method should do
    """
    
    print("Testing the implementation element filtering concept...")
    print("=" * 60)
    
    # Initialize domain loader  
    loader = DomainVerbLoader()
    domain_name = 'beverage_preparation'
    
    # Test objects that might appear in UC1
    test_objects = [
        "clock",              # Should be filtered - appears in UC1 B1
        "radio clock",        # Should be filtered - appears in UC1 B1  
        "time",              # Should be created - functional concept
        "water",             # Should be created - functional entity
        "coffee",            # Should be created - functional entity
        "heater",            # Should be filtered - implementation element
        "system",            # Should be created - functional concept
        "filter",            # Should be created - functional entity
        "cup",               # Should be created - functional entity
        "mill",              # Might be filtered if it's implementation
        "grinder"            # Might be filtered if it's implementation
    ]
    
    print(f"Testing entity creation logic for UC text objects:")
    print(f"Domain: {domain_name}")
    print()
    
    entities_that_would_be_created = []
    entities_that_would_be_filtered = []
    
    for obj_text in test_objects:
        print(f"Testing: '{obj_text}'")
        
        # Check if this would be filtered as implementation element
        impl_info = loader.get_implementation_element_info(obj_text, domain_name)
        
        if impl_info:
            # This would be filtered out by our modification
            entities_that_would_be_filtered.append(obj_text)
            print(f"  -> FILTERED (Implementation Element)")
            print(f"     Warning: {impl_info['warning']}")
            if impl_info.get('functional_suggestion'):
                print(f"     Suggestion: {impl_info['functional_suggestion']}")
        else:
            # This would be created as an entity
            entities_that_would_be_created.append(obj_text)
            print(f"  -> CREATE ENTITY (Functional)")
        
        print()
    
    print("=" * 60)
    print("SUMMARY:")
    print(f"Entities that WOULD BE CREATED: {len(entities_that_would_be_created)}")
    for entity in entities_that_would_be_created:
        print(f"  - {entity}")
    
    print(f"\nEntities that WOULD BE FILTERED: {len(entities_that_would_be_filtered)}")
    for entity in entities_that_would_be_filtered:
        print(f"  - {entity}")
    
    print("\nOur modifications to _derive_generic_entities() will:")
    print("1. Check each object against get_implementation_element_info()")
    print("2. Skip creating RAClass entities for implementation elements")
    print("3. Print warnings when implementation elements are found")
    print("4. Suggest functional alternatives when available")

if __name__ == "__main__":
    test_modified_analyzer_concept()