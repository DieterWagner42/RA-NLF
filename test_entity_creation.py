#!/usr/bin/env python3
"""
Test the modified _derive_generic_entities method to verify implementation element filtering
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import necessary components
from generic_uc_analyzer import GenericUCAnalyzer, UCStep, VerbAnalysis, VerbType

def test_entity_creation_with_implementation_check():
    """Test entity creation with implementation element checking"""
    
    print("Testing entity creation with implementation element filtering...")
    
    try:
        # Initialize analyzer
        analyzer = GenericUCAnalyzer(domain_name='beverage_preparation')
        print("Analyzer initialized successfully")
        
        # Create test data
        test_step = UCStep(
            step_id="B1",
            step_text="System clock reaches the user defined time",
            flow_type="main"
        )
        
        test_verb_analysis = VerbAnalysis(
            step_id="B1",
            original_text="System clock reaches the user defined time",
            verb="reaches",
            verb_lemma="reach",
            verb_type=VerbType.FUNCTION_VERB,
            direct_object="time"
        )
        
        # Test cases for entity creation
        test_objects = [
            "clock",           # Should be filtered out
            "radio clock",     # Should be filtered out  
            "time",           # Should be created
            "water",          # Should be created
            "heater"          # Should be filtered out
        ]
        
        print("\nTesting entity creation for different objects:")
        print("=" * 60)
        
        for obj_text in test_objects:
            print(f"\nTesting object: '{obj_text}'")
            print("-" * 30)
            
            # Call the method we modified
            entities = analyzer._derive_generic_entities(
                obj_text, test_verb_analysis, test_step
            )
            
            if entities:
                print(f"  Created {len(entities)} entity(ies):")
                for entity in entities:
                    print(f"    - {entity.name} ({entity.type})")
            else:
                print(f"  No entities created (likely filtered as implementation element)")
                
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_entity_creation_with_implementation_check()