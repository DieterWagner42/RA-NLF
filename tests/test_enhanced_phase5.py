"""
Test Enhanced Phase 5 with improved Phases 2-4
Check if validation state management fixes Phase 5 violations
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict
from phase2_enhanced_analyzer import ValidationState
from phase5_datenfluss_analyzer import DatenflussAnalyzer


@dataclass
class MockEnhancedPhase2Analysis:
    resource_name: str
    transformations: List[str]
    validation_flow: List[str]
    manager_controller: object = None

@dataclass 
class MockManagerController:
    name: str

@dataclass
class MockEnhancedPhase2Result:
    resource_analyses: List
    validation_flow_map: Dict = field(default_factory=dict)

@dataclass
class MockEnhancedPhase3Result:
    phase2_result: MockEnhancedPhase2Result
    all_objects: Dict = field(default_factory=dict)

@dataclass
class MockEnhancedPhase4Result:
    phase3_result: MockEnhancedPhase3Result


def create_enhanced_mock_data():
    """Create mock data with proper validation states"""
    
    # Enhanced transformations with validation states
    mock_analyses = [
        MockEnhancedPhase2Analysis(
            resource_name="coffee beans",
            transformations=["validated beans -> grinding -> processed ground coffee"],
            validation_flow=["coffee beans: RAW -> VALIDATED", "coffee beans: VALIDATED -> PROCESSED"],
            manager_controller=MockManagerController("CoffeeBeansManager")
        ),
        MockEnhancedPhase2Analysis(
            resource_name="water",
            transformations=["validated water -> heating -> processed hot water"],
            validation_flow=["water: RAW -> VALIDATED", "water: VALIDATED -> PROCESSED"], 
            manager_controller=MockManagerController("WaterManager")
        ),
        MockEnhancedPhase2Analysis(
            resource_name="milk",
            transformations=["validated milk -> steaming -> processed steamed milk"],
            validation_flow=["milk: RAW -> VALIDATED", "milk: VALIDATED -> PROCESSED"],
            manager_controller=MockManagerController("MilkManager")
        ),
        MockEnhancedPhase2Analysis(
            resource_name="cup",
            transformations=["validated cup -> usage -> processed cup"],
            validation_flow=["cup: RAW -> VALIDATED", "cup: VALIDATED -> PROCESSED"],
            manager_controller=MockManagerController("CupManager")
        )
    ]
    
    # Enhanced validation flow map
    validation_flow_map = {
        "coffee beans": [ValidationState.RAW, ValidationState.VALIDATED, ValidationState.PROCESSED],
        "water": [ValidationState.RAW, ValidationState.VALIDATED, ValidationState.PROCESSED],
        "milk": [ValidationState.RAW, ValidationState.VALIDATED, ValidationState.PROCESSED],
        "cup": [ValidationState.RAW, ValidationState.VALIDATED, ValidationState.PROCESSED]
    }
    
    phase2_result = MockEnhancedPhase2Result(
        resource_analyses=mock_analyses,
        validation_flow_map=validation_flow_map
    )
    
    phase3_result = MockEnhancedPhase3Result(
        phase2_result=phase2_result,
        all_objects={"controllers": ["GeträenkeOrchestrator", "WaterManager", "CoffeeBeansManager", "MilkManager", "CupManager"]}
    )
    
    phase4_result = MockEnhancedPhase4Result(phase3_result=phase3_result)
    
    return phase4_result


def test_enhanced_phase5():
    """Test Phase 5 with enhanced validation state management"""
    
    print("Testing Enhanced Phase 5 with improved validation states...")
    
    # Create enhanced mock data
    enhanced_phase4_result = create_enhanced_mock_data()
    
    # Initialize Phase 5 analyzer
    analyzer = DatenflussAnalyzer()
    
    # Perform Phase 5 analysis with enhanced data
    try:
        phase5_result = analyzer.perform_phase5_analysis(enhanced_phase4_result)
        
        print(f"Phase 5 Analysis completed!")
        print(f"Summary: {phase5_result.summary}")
        print(f"Data Entities: {len(phase5_result.data_entities)}")
        print(f"Data Transformations: {len(phase5_result.data_transformations)}")
        print(f"Data Flows: {len(phase5_result.data_flows)}")
        print(f"Data Violations: {len(phase5_result.data_violations)}")
        
        if phase5_result.data_violations:
            print("\nRemaining violations:")
            for violation in phase5_result.data_violations:
                print(f"  - {violation}")
        else:
            print("\n✅ NO VIOLATIONS FOUND! Enhanced phases fixed the issues!")
        
        # Save enhanced result
        output_file = "Zwischenprodukte/UC1_Coffee_enhanced_phase5_analysis.json"
        
        serializable_result = {
            "summary": phase5_result.summary,
            "data_entities": [
                {
                    "entity_id": e.entity_id,
                    "name": e.name,
                    "data_type": e.data_type,
                    "initial_state": e.initial_state.value,
                    "current_state": e.current_state.value,
                    "managed_by": e.managed_by,
                    "dependencies": e.dependencies
                } for e in phase5_result.data_entities
            ],
            "data_transformations": [
                {
                    "transformation_id": t.transformation_id,
                    "name": t.name,
                    "transformation_type": t.transformation_type,
                    "responsible_controller": t.responsible_controller,
                    "input_entities": t.input_entities,
                    "output_entities": t.output_entities,
                    "validation_rules": t.validation_rules
                } for t in phase5_result.data_transformations
            ],
            "violations": phase5_result.data_violations,
            "enhanced_features": [
                "Validation state management integrated",
                "RAW -> VALIDATED -> PROCESSED -> READY flow",
                "Proper entity state tracking",
                "Enhanced transformation validation"
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        print(f"\nEnhanced Phase 5 result saved to: {output_file}")
        
        return phase5_result
        
    except Exception as e:
        print(f"Error in enhanced Phase 5 analysis: {e}")
        return None


def compare_original_vs_enhanced():
    """Compare original Phase 5 violations with enhanced version"""
    
    print("\nCOMPARISON: Original vs Enhanced Phase 5")
    print("="*50)
    
    # Load original Phase 5 result
    try:
        with open('Zwischenprodukte/UC1_Coffee_phase5_analysis.json', 'r', encoding='utf-8') as f:
            original_result = json.load(f)
        
        original_violations = original_result.get("violations", [])
        print(f"Original Phase 5 violations: {len(original_violations)}")
        for violation in original_violations:
            print(f"  - {violation}")
    except FileNotFoundError:
        print("Original Phase 5 result not found")
        original_violations = []
    
    # Test enhanced Phase 5
    enhanced_result = test_enhanced_phase5()
    
    if enhanced_result:
        enhanced_violations = enhanced_result.data_violations
        print(f"\nEnhanced Phase 5 violations: {len(enhanced_violations)}")
        for violation in enhanced_violations:
            print(f"  - {violation}")
        
        print(f"\nIMPROVEMENT:")
        print(f"Violations reduced from {len(original_violations)} to {len(enhanced_violations)}")
        
        if len(enhanced_violations) == 0:
            print("🎉 PERFECT! All violations fixed by enhanced phases 2-4!")
        elif len(enhanced_violations) < len(original_violations):
            print(f"✅ IMPROVEMENT! Reduced violations by {len(original_violations) - len(enhanced_violations)}")
        else:
            print("⚠️  Still need further improvements")
    
    return enhanced_result


if __name__ == "__main__":
    result = compare_original_vs_enhanced()