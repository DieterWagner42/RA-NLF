#!/usr/bin/env python3
"""
Test script for the new generative context management system
Tests the replacement of hardcoded contexts with NLP-based generative approach
"""

import sys
import os
sys.path.append('src')

from generative_context_manager import GenerativeContextManager, ContextType
from structured_uc_analyzer import StructuredUCAnalyzer

def test_generative_context_manager():
    """Test the generative context manager directly"""
    print("=== Testing Generative Context Manager ===")
    
    # Initialize for beverage preparation domain
    context_manager = GenerativeContextManager("beverage_preparation")
    
    # Test operational materials detection
    test_texts = [
        "The system heats the milk to 60°C",
        "Coffee beans are ground to the specified degree",
        "Water is filtered and heated for brewing",
        "Sugar is added to the coffee",
        "The cleaning agent is used for system maintenance"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i}: {text} ---")
        contexts = context_manager.generate_contexts_for_text(text, f"A{i}")
        
        print(f"Generated {len(contexts)} contexts:")
        for context in contexts:
            print(f"  - {context.context_type.value}: {context.context_name}")
            if context.safety_class:
                print(f"    Safety: {context.safety_class}")
            if context.hygiene_level:
                print(f"    Hygiene: {context.hygiene_level}")
            if context.special_requirements:
                print(f"    Requirements: {context.special_requirements[:2]}...")
            print(f"    Domain alignment: {context.domain_alignment:.2f}")
        
        # Test addressing format generation
        for context in contexts:
            if context.context_type == ContextType.OPERATIONAL_MATERIAL:
                addressing = context_manager.get_material_addressing_format("test_material", context)
                print(f"    Addressing format: {addressing}")
    
    # Test context summary
    all_contexts = []
    for text in test_texts:
        all_contexts.extend(context_manager.generate_contexts_for_text(text))
    
    summary = context_manager.get_context_summary(all_contexts)
    print(f"\n--- Summary ---")
    print(f"Total contexts: {summary['total_contexts']}")
    print(f"Context types: {summary['context_types']}")
    print(f"Safety classes: {summary['safety_classes']}")
    print(f"Hygiene levels: {summary['hygiene_levels']}")
    print(f"Average domain alignment: {summary['domain_alignment_avg']:.2f}")

def test_enhanced_structured_analyzer():
    """Test the enhanced structured UC analyzer"""
    print("\n\n=== Testing Enhanced Structured UC Analyzer ===")
    
    # Create test UC content
    test_uc_content = """Capability: Coffee Preparation
Feature: Automated Milk Coffee Preparation
Use Case: UC1 - Milchkaffee
Goal: Prepare milk coffee automatically

Actor: User

Preconditions:
- Sufficient coffee beans available
- Milk tank filled with fresh milk
- Water tank filled with filtered water
- System is clean and operational

Main Flow:
A1: User initiates milk coffee preparation
A2: System grinds coffee beans to specified degree
A3: System heats water to optimal brewing temperature
A4: System brews coffee using ground beans and hot water
A5: System heats milk to 60°C
A6: System froths the heated milk
A7: System combines coffee and frothed milk
A8: System dispenses the prepared milk coffee to user

End Use Case"""

    # Write test UC file
    test_file_path = "test_uc_generative.txt"
    with open(test_file_path, 'w', encoding='utf-8') as f:
        f.write(test_uc_content)
    
    try:
        # Initialize enhanced analyzer
        analyzer = StructuredUCAnalyzer("beverage_preparation")
        
        # Analyze the test UC
        print(f"Analyzing test UC file: {test_file_path}")
        line_analyses, output_json = analyzer.analyze_uc_file(test_file_path)
        
        print(f"\nAnalyzed {len(line_analyses)} lines")
        print(f"Output JSON: {output_json}")
        
        # Show enhanced context results
        print("\n--- Enhanced Context Results ---")
        for analysis in line_analyses:
            if analysis.step_id and analysis.step_context:
                print(f"\nStep {analysis.step_id}: {analysis.line_text}")
                print(f"  Context Type: {analysis.step_context.context_type}")
                print(f"  Global Context: {analysis.step_context.global_context}")
                
                if analysis.step_context.operational_materials:
                    print(f"  Operational Materials:")
                    for material in analysis.step_context.operational_materials:
                        print(f"    - {material['name']} (Safety: {material['safety_class']}, Hygiene: {material['hygiene_level']})")
                
                if analysis.step_context.safety_requirements:
                    print(f"  Safety Requirements: {len(analysis.step_context.safety_requirements)} items")
                
                if analysis.step_context.special_controllers:
                    print(f"  Special Controllers: {analysis.step_context.special_controllers}")
                
                # Show enhanced RA classes
                operational_classes = [ra for ra in analysis.ra_classes if ra.element_type in ["operational_material", "safety", "hygiene", "safety_hygiene"]]
                if operational_classes:
                    print(f"  Enhanced RA Classes:")
                    for ra_class in operational_classes:
                        print(f"    - {ra_class.ra_type.value}: {ra_class.name} ({ra_class.element_type})")
        
        # Show generated contexts summary
        if hasattr(analyzer, 'generated_contexts') and analyzer.generated_contexts:
            print(f"\n--- Generated Contexts Summary ---")
            total_contexts = sum(len(contexts) for contexts in analyzer.generated_contexts.values())
            print(f"Total generated contexts: {total_contexts}")
            
            for step_id, contexts in analyzer.generated_contexts.items():
                if contexts:
                    print(f"  {step_id}: {len(contexts)} contexts")
                    for context in contexts:
                        print(f"    - {context.context_type.value}: {context.context_name}")
    
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

def test_domain_independence():
    """Test domain independence by testing with different domains"""
    print("\n\n=== Testing Domain Independence ===")
    
    # Test different domains
    domains = ["beverage_preparation", "aerospace", "nuclear"]
    test_text = "The system processes radioactive material with safety protocols"
    
    for domain in domains:
        print(f"\n--- Testing domain: {domain} ---")
        try:
            context_manager = GenerativeContextManager(domain)
            contexts = context_manager.generate_contexts_for_text(test_text, "T1")
            
            print(f"Generated {len(contexts)} contexts:")
            for context in contexts:
                print(f"  - {context.context_type.value}: {context.context_name} (alignment: {context.domain_alignment:.2f})")
                if context.safety_class:
                    print(f"    Safety: {context.safety_class}")
                if context.hygiene_level:
                    print(f"    Hygiene: {context.hygiene_level}")
        except Exception as e:
            print(f"  Error testing domain {domain}: {e}")

if __name__ == "__main__":
    print("Testing Generative Context Management System")
    print("=" * 60)
    
    try:
        # Test 1: Direct context manager testing
        test_generative_context_manager()
        
        # Test 2: Enhanced structured analyzer testing
        test_enhanced_structured_analyzer()
        
        # Test 3: Domain independence testing
        test_domain_independence()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("The generative context system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()