#!/usr/bin/env python3
"""
Simple test script to verify implementation element checking in entity creation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test the key parts without loading the full analyzer
from domain_verb_loader import DomainVerbLoader

def test_implementation_element_detection():
    """Test if implementation elements like 'clock' and 'radio clock' are detected"""
    
    print("Testing implementation element detection...")
    
    # Initialize domain loader
    loader = DomainVerbLoader()
    
    # Test various clock-related terms
    test_terms = [
        'clock',
        'radio clock', 
        'system clock',
        'timer',
        'heater',  # known implementation element
        'water',   # should not be implementation element
        'coffee'   # should not be implementation element
    ]
    
    domain_name = 'beverage_preparation'
    
    print(f"\nTesting with domain: {domain_name}")
    print("=" * 50)
    
    for term in test_terms:
        impl_info = loader.get_implementation_element_info(term, domain_name)
        if impl_info:
            print(f"[IMPL] '{term}' -> IMPLEMENTATION ELEMENT")
            print(f"  Warning: {impl_info['warning']}")
            if impl_info.get('functional_suggestion'):
                print(f"  Suggestion: {impl_info['functional_suggestion']}")
        else:
            print(f"[OK] '{term}' -> functional entity (OK)")
        print()

if __name__ == "__main__":
    test_implementation_element_detection()