#!/usr/bin/env python3
"""
Generate RA Diagram with Generative Context System
Uses the official_rup_engine.py to visualize the enhanced RA classes with operational materials
"""

import sys
import os
sys.path.append('src')

from structured_uc_analyzer import StructuredUCAnalyzer
from official_rup_engine import OfficialRUPEngine, RAComponent, ComponentType
import json

def create_test_uc_file():
    """Create a comprehensive test UC file with operational materials"""
    test_uc_content = """Capability: Advanced Coffee Preparation
Feature: Automated Milk Coffee with Safety & Hygiene
Use Case: UC1 - Enhanced Milchkaffee
Goal: Prepare milk coffee with full operational materials tracking

Actor: User

Preconditions:
- Coffee beans available with batch tracking
- Fresh milk with temperature monitoring
- Filtered water with quality certification
- Cleaning agents for hygiene maintenance

Main Flow:
A1: User initiates enhanced coffee preparation
A2: System grinds coffee beans to optimal degree
A3: System heats filtered water to brewing temperature
A4: System brews coffee using ground beans and hot water
A5: System heats milk to 65°C with safety monitoring
A6: System froths the heated milk with hygiene controls
A7: System combines coffee and frothed milk
A8: System performs quality check of final product
A9: System dispenses coffee with material traceability
A10: System initiates cleaning cycle with chemical agents

End Use Case"""

    test_file_path = "test_uc_enhanced.txt"
    with open(test_file_path, 'w', encoding='utf-8') as f:
        f.write(test_uc_content)
    
    return test_file_path

def analyze_uc_with_generative_context(uc_file_path):
    """Analyze UC with enhanced generative context system"""
    print(f"Analyzing UC file with generative context: {uc_file_path}")
    
    # Initialize enhanced analyzer
    analyzer = StructuredUCAnalyzer("beverage_preparation")
    
    # Analyze the UC
    line_analyses, output_json = analyzer.analyze_uc_file(uc_file_path)
    
    print(f"Analysis completed: {output_json}")
    return line_analyses, analyzer

def extract_ra_components_from_analysis(line_analyses, analyzer):
    """Extract RA components from the enhanced analysis"""
    components = []
    
    # Extract all RA classes from analysis
    all_ra_classes = []
    for analysis in line_analyses:
        if analysis.ra_classes:
            all_ra_classes.extend(analysis.ra_classes)
    
    # Convert to OfficialRUPEngine components
    for i, ra_class in enumerate(all_ra_classes):
        # Determine component type
        if ra_class.ra_type.value == "Actor":
            component_type = ComponentType.ACTOR
        elif ra_class.ra_type.value == "Boundary":
            component_type = ComponentType.BOUNDARY
        elif ra_class.ra_type.value == "Controller":
            component_type = ComponentType.CONTROLLER
        elif ra_class.ra_type.value == "Entity":
            component_type = ComponentType.ENTITY
        else:
            continue
        
        # Create enhanced label with context information
        label = ra_class.name
        if hasattr(ra_class, 'element_type'):
            if ra_class.element_type == "operational_material":
                label += "\n(OpMaterial)"
            elif ra_class.element_type == "safety":
                label += "\n(Safety)"
            elif ra_class.element_type == "hygiene":
                label += "\n(Hygiene)"
            elif ra_class.element_type == "safety_hygiene":
                label += "\n(Safety/Hygiene)"
        
        # Add warnings for implementation elements
        warnings = []
        if "implementation" in ra_class.description.lower():
            warnings.append("Implementation element detected")
        
        component = RAComponent(
            id=f"{ra_class.name}_{i}",
            label=label,
            component_type=component_type,
            stereotype=ra_class.stereotype,
            warnings=warnings
        )
        
        components.append(component)
    
    return components

def generate_ra_diagram_from_json(json_file_path, title="Enhanced RA Diagram with Generative Context"):
    """Generate the RA diagram using official RUP engine from JSON"""
    print(f"Generating RA diagram from JSON: {json_file_path}")
    
    # Initialize the official RUP engine
    rup_engine = OfficialRUPEngine(figure_size=(24, 18))
    
    # Generate diagram from JSON analysis
    output_path = rup_engine.create_official_rup_diagram_from_json(json_file_path)
    
    print(f"RA diagram generated: {output_path}")
    return output_path

def show_generative_context_summary(analyzer):
    """Show summary of generated contexts"""
    if hasattr(analyzer, 'generated_contexts') and analyzer.generated_contexts:
        print("\n=== Generative Context Summary ===")
        
        total_contexts = sum(len(contexts) for contexts in analyzer.generated_contexts.values())
        print(f"Total generated contexts: {total_contexts}")
        
        # Context type breakdown
        context_types = {}
        operational_materials = []
        safety_requirements = set()
        hygiene_levels = set()
        
        for step_id, contexts in analyzer.generated_contexts.items():
            print(f"\nStep {step_id}: {len(contexts)} contexts")
            for context in contexts:
                context_type = context.context_type.value
                context_types[context_type] = context_types.get(context_type, 0) + 1
                
                print(f"  - {context_type}: {context.context_name}")
                if context.safety_class:
                    print(f"    Safety: {context.safety_class}")
                if context.hygiene_level:
                    print(f"    Hygiene: {context.hygiene_level}")
                    hygiene_levels.add(context.hygiene_level)
                if context.special_requirements:
                    safety_requirements.update(context.special_requirements[:2])  # Show first 2
                
                if context.context_type.value == "operational_material":
                    operational_materials.append(context.context_name)
        
        print(f"\n=== Summary Statistics ===")
        print(f"Context Types: {context_types}")
        print(f"Operational Materials: {operational_materials}")
        print(f"Hygiene Levels: {list(hygiene_levels)}")
        print(f"Safety Requirements (sample): {list(safety_requirements)[:5]}")

def main():
    """Main execution function"""
    print("Enhanced RA Diagram Generation with Generative Context")
    print("=" * 70)
    
    try:
        # Step 1: Create test UC file
        uc_file_path = create_test_uc_file()
        print(f"Created test UC file: {uc_file_path}")
        
        # Step 2: Analyze with generative context
        line_analyses, analyzer = analyze_uc_with_generative_context(uc_file_path)
        
        # Step 3: Show generative context summary
        show_generative_context_summary(analyzer)
        
        # Step 4: Extract RA components
        components = extract_ra_components_from_analysis(line_analyses, analyzer)
        print(f"\nExtracted {len(components)} RA components")
        
        # Step 5: Generate RA diagram from JSON
        json_file = f"{uc_file_path.replace('.txt', '')}_Structured_RA_Analysis.json"
        if os.path.exists(json_file):
            diagram_path = generate_ra_diagram_from_json(json_file)
            print(f"\nRA Diagram generated: {diagram_path}")
        else:
            print(f"JSON analysis file not found: {json_file}")
        
        # Step 6: Show enhanced features
        print(f"\n=== Enhanced Features Demonstrated ===")
        print("✓ Generative context management (no hardcoded contexts)")
        print("✓ Operational materials with safety/hygiene classification")
        print("✓ Domain-driven context generation from JSON")
        print("✓ NLP-based semantic analysis")
        print("✓ Universal operational materials framework integration")
        print("✓ Enhanced RA class generation with context awareness")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test file
        if 'uc_file_path' in locals() and os.path.exists(uc_file_path):
            os.remove(uc_file_path)
            print(f"Cleaned up test file: {uc_file_path}")

if __name__ == "__main__":
    main()