#!/usr/bin/env python3
"""
Detailed Analysis of UC1 Step B1 with Generative Context System
Shows comprehensive breakdown of what the system generated for step B1
"""

import json
import sys
sys.path.append('src')

def analyze_step_b1_from_json():
    """Analyze step B1 from the generated JSON"""
    try:
        with open('UC1_Structured_RA_Analysis.json', 'r', encoding='utf-8') as f:
            uc1_data = json.load(f)
        
        print("UC1 Step B1 Detailed Analysis")
        print("=" * 50)
        print("Step B1: System clock reaches set time of 7:00h (Radio clock)")
        print()
        
        # Find B1-related components
        components = uc1_data['components']
        
        # B1 Controllers
        b1_controllers = []
        for controller in components.get('controllers', []):
            if 'B1' in controller.get('description', '') or 'TimeManager' in controller.get('name', ''):
                b1_controllers.append(controller)
        
        print("=== B1 Controllers ===")
        for controller in b1_controllers:
            print(f"- {controller['name']}")
            print(f"  Description: {controller['description']}")
            print(f"  Parallel Group: {controller.get('parallel_group', 0)}")
        print()
        
        # B1 Entities
        b1_entities = []
        for entity in components.get('entities', []):
            name = entity.get('name', '').lower()
            if any(keyword in name for keyword in ['time', 'clock', 'radio']):
                b1_entities.append(entity)
        
        print("=== B1 Entities ===")
        for entity in b1_entities:
            print(f"- {entity['name']}")
            print(f"  Description: {entity['description']}")
        print()
        
        # Look for data flows involving B1
        relationships = uc1_data.get('relationships', {})
        data_flows = relationships.get('data_flows', [])
        
        b1_data_flows = []
        for flow in data_flows:
            if 'B1' in flow.get('step_id', '') or 'Time' in flow.get('controller', ''):
                b1_data_flows.append(flow)
        
        print("=== B1 Data Flows ===")
        if b1_data_flows:
            for flow in b1_data_flows:
                print(f"- {flow.get('flow_type', '')}: {flow.get('controller', '')} -> {flow.get('entity', '')}")
                print(f"  Step: {flow.get('step_id', '')}")
                print(f"  Description: {flow.get('description', '')}")
        else:
            print("No specific B1 data flows found in relationships")
        print()
        
        return True
        
    except Exception as e:
        print(f"Error analyzing B1 from JSON: {e}")
        return False

def analyze_step_b1_with_generative_context():
    """Re-analyze step B1 with detailed generative context output"""
    from structured_uc_analyzer import StructuredUCAnalyzer
    
    print("\n" + "=" * 50)
    print("B1 Generative Context Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = StructuredUCAnalyzer("beverage_preparation")
    
    # Analyze just the B1 line
    b1_text = "B1 (trigger) System clock reaches set time of 7:00h (Radio clock)"
    
    # Generate contexts for B1
    contexts = analyzer.context_manager.generate_contexts_for_text(b1_text, "B1")
    
    print(f"B1 Text: {b1_text}")
    print(f"Generated {len(contexts)} contexts for B1:")
    print()
    
    for i, context in enumerate(contexts, 1):
        print(f"Context {i}: {context.context_type.value}")
        print(f"  Name: {context.context_name}")
        print(f"  Source: {context.source_text}")
        print(f"  Semantic Features: {context.semantic_features[:3]}...")  # First 3 features
        print(f"  Domain Alignment: {context.domain_alignment:.2f}")
        
        if context.safety_class:
            print(f"  Safety Class: {context.safety_class}")
        if context.hygiene_level:
            print(f"  Hygiene Level: {context.hygiene_level}")
        if context.special_requirements:
            print(f"  Requirements: {context.special_requirements[:2]}...")  # First 2 requirements
        if context.controllers:
            print(f"  Special Controllers: {context.controllers}")
        print()
    
    # Show context summary for B1
    summary = analyzer.context_manager.get_context_summary(contexts)
    print("=== B1 Context Summary ===")
    print(f"Total contexts: {summary['total_contexts']}")
    print(f"Context types: {summary['context_types']}")
    print(f"Average domain alignment: {summary['domain_alignment_avg']:.2f}")
    
    return contexts

def analyze_b1_grammatical_analysis():
    """Show grammatical analysis for B1"""
    print("\n" + "=" * 50)
    print("B1 Grammatical Analysis")
    print("=" * 50)
    
    from structured_uc_analyzer import StructuredUCAnalyzer
    
    analyzer = StructuredUCAnalyzer("beverage_preparation")
    b1_text = "B1 (trigger) System clock reaches set time of 7:00h (Radio clock)"
    
    # Perform grammatical analysis
    grammatical = analyzer._perform_grammatical_analysis(b1_text)
    
    print(f"B1 Text: {b1_text}")
    print()
    print("=== Grammatical Analysis Results ===")
    print(f"Main Verb: {grammatical.main_verb}")
    print(f"Verb Lemma: {grammatical.verb_lemma}")
    print(f"Verb Type: {grammatical.verb_type}")
    print(f"Direct Object: {grammatical.direct_object}")
    print(f"Compound Nouns: {grammatical.compound_nouns}")
    print(f"Gerunds: {grammatical.gerunds}")
    print(f"Weak Verbs: {grammatical.weak_verbs}")
    print(f"Prepositional Objects: {grammatical.prepositional_objects}")
    
    return grammatical

def show_b1_step_context():
    """Show step context for B1"""
    print("\n" + "=" * 50)
    print("B1 Enhanced Step Context")
    print("=" * 50)
    
    from structured_uc_analyzer import StructuredUCAnalyzer, LineType
    
    analyzer = StructuredUCAnalyzer("beverage_preparation")
    b1_text = "B1 (trigger) System clock reaches set time of 7:00h (Radio clock)"
    
    # Generate contexts
    contexts = analyzer.context_manager.generate_contexts_for_text(b1_text, "B1")
    
    # Perform grammatical analysis
    grammatical = analyzer._perform_grammatical_analysis(b1_text)
    
    # Generate enhanced step context
    step_context = analyzer._determine_step_context_enhanced(
        "B1", b1_text, LineType.STEP, grammatical, contexts
    )
    
    print(f"B1 Text: {b1_text}")
    print()
    print("=== Enhanced Step Context ===")
    print(f"Step ID: {step_context.step_id}")
    print(f"Step Type: {step_context.step_type}")
    print(f"Domain: {step_context.domain}")
    print(f"Context Type: {step_context.context_type}")
    print(f"Global Context: {step_context.global_context}")
    print(f"Description: {step_context.description}")
    print(f"Business Context: {step_context.business_context}")
    print(f"Technical Context: {step_context.technical_context}")
    
    if step_context.operational_materials:
        print(f"Operational Materials: {len(step_context.operational_materials)}")
        for material in step_context.operational_materials:
            print(f"  - {material}")
    
    if step_context.safety_requirements:
        print(f"Safety Requirements: {len(step_context.safety_requirements)} items")
        for req in step_context.safety_requirements[:3]:  # First 3
            print(f"  - {req}")
    
    if step_context.special_controllers:
        print(f"Special Controllers: {step_context.special_controllers}")
    
    return step_context

def show_b1_ra_classes():
    """Show generated RA classes for B1"""
    print("\n" + "=" * 50)
    print("B1 Generated RA Classes")
    print("=" * 50)
    
    from structured_uc_analyzer import StructuredUCAnalyzer, LineType
    
    analyzer = StructuredUCAnalyzer("beverage_preparation")
    b1_text = "B1 (trigger) System clock reaches set time of 7:00h (Radio clock)"
    
    # Generate contexts
    contexts = analyzer.context_manager.generate_contexts_for_text(b1_text, "B1")
    
    # Perform grammatical analysis
    grammatical = analyzer._perform_grammatical_analysis(b1_text)
    
    # Generate enhanced step context
    step_context = analyzer._determine_step_context_enhanced(
        "B1", b1_text, LineType.STEP, grammatical, contexts
    )
    
    # Generate RA classes
    ra_classes = analyzer._generate_ra_classes_for_line_enhanced(
        b1_text, LineType.STEP, "B1", grammatical, step_context, contexts
    )
    
    print(f"B1 Text: {b1_text}")
    print(f"Generated {len(ra_classes)} RA classes:")
    print()
    
    # Group by type
    actors = [c for c in ra_classes if c.ra_type.value == "Actor"]
    boundaries = [c for c in ra_classes if c.ra_type.value == "Boundary"]
    controllers = [c for c in ra_classes if c.ra_type.value == "Controller"]
    entities = [c for c in ra_classes if c.ra_type.value == "Entity"]
    
    if actors:
        print("=== B1 Actors ===")
        for actor in actors:
            print(f"- {actor.name}")
            print(f"  Description: {actor.description}")
            print(f"  Element Type: {actor.element_type}")
    
    if boundaries:
        print("=== B1 Boundaries ===")
        for boundary in boundaries:
            print(f"- {boundary.name}")
            print(f"  Description: {boundary.description}")
            print(f"  Element Type: {boundary.element_type}")
    
    if controllers:
        print("=== B1 Controllers ===")
        for controller in controllers:
            print(f"- {controller.name}")
            print(f"  Description: {controller.description}")
            print(f"  Element Type: {controller.element_type}")
            print(f"  Parallel Group: {controller.parallel_group}")
    
    if entities:
        print("=== B1 Entities ===")
        for entity in entities:
            print(f"- {entity.name}")
            print(f"  Description: {entity.description}")
            print(f"  Element Type: {entity.element_type}")
    
    print(f"\n=== B1 RA Class Summary ===")
    print(f"Actors: {len(actors)}")
    print(f"Boundaries: {len(boundaries)}")
    print(f"Controllers: {len(controllers)}")
    print(f"Entities: {len(entities)}")
    print(f"Total: {len(ra_classes)}")
    
    return ra_classes

def main():
    """Main analysis function"""
    print("UC1 Step B1 Comprehensive Analysis")
    print("Using Enhanced Generative Context System")
    print("=" * 60)
    
    # 1. Analyze from existing JSON
    analyze_step_b1_from_json()
    
    # 2. Live generative context analysis
    contexts = analyze_step_b1_with_generative_context()
    
    # 3. Grammatical analysis
    grammatical = analyze_b1_grammatical_analysis()
    
    # 4. Enhanced step context
    step_context = show_b1_step_context()
    
    # 5. Generated RA classes
    ra_classes = show_b1_ra_classes()
    
    print("\n" + "=" * 60)
    print("B1 Analysis Complete!")
    print(f"Step B1 demonstrates the full power of generative context analysis")
    print(f"- {len(contexts)} contexts generated from NLP analysis")
    print(f"- {len(ra_classes)} RA classes automatically created")
    print(f"- Enhanced with operational materials and safety considerations")
    print(f"- No hardcoded rules - all generated from domain knowledge")

if __name__ == "__main__":
    main()