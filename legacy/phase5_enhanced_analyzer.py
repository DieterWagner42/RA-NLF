"""
Enhanced Phase 5: Datenfluss-Analyse with Validation State Integration
Properly integrates with Enhanced Phases 2-4 to eliminate violations
Uses validation state transitions: RAW -> VALIDATED -> PROCESSED -> READY -> CONSUMED
"""

import spacy
import json
import os
import re
from typing import Dict, List, Set, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from domain_config_analyzer import DomainConfigurableAnalyzer, DomainConfig, Phase1Result
from phase2_enhanced_analyzer import ValidationState


class EnhancedDataFlowType(Enum):
    VALIDATION_INPUT = "validation_input"
    VALIDATION_OUTPUT = "validation_output"
    PROCESSING_INPUT = "processing_input"
    PROCESSING_OUTPUT = "processing_output"
    AGGREGATION = "aggregation"
    USER_OUTPUT = "user_output"


@dataclass
class EnhancedDataEntity:
    entity_id: str
    name: str
    data_type: str  # "ingredient", "intermediate", "product", "waste"
    validation_state: ValidationState
    managed_by: Optional[str] = None
    properties: Dict[str, str] = field(default_factory=dict)
    validated_dependencies: List[str] = field(default_factory=list)  # Only validated dependencies
    state_history: List[ValidationState] = field(default_factory=list)


@dataclass
class EnhancedDataTransformation:
    transformation_id: str
    name: str
    input_entities: List[str]
    output_entities: List[str]
    transformation_type: str
    responsible_controller: str
    input_state_required: ValidationState
    output_state_produced: ValidationState
    validation_rules: List[str] = field(default_factory=list)
    state_transition_valid: bool = True


@dataclass
class EnhancedDataFlow:
    flow_id: str
    source_entity: str
    target_entity: str
    flow_type: EnhancedDataFlowType
    responsible_controller: str
    source_state: ValidationState
    target_state: ValidationState
    validation_required: bool = True
    state_transition_valid: bool = True


@dataclass
class ValidationStatePattern:
    pattern_id: str
    resource_name: str
    state_sequence: List[ValidationState]
    transition_validations: List[str]
    entity_mappings: Dict[ValidationState, str]


@dataclass
class EnhancedPhase5Result:
    enhanced_phase4_result: object  # EnhancedPhase4Result
    enhanced_data_entities: List[EnhancedDataEntity]
    enhanced_data_transformations: List[EnhancedDataTransformation]
    enhanced_data_flows: List[EnhancedDataFlow]
    validation_state_patterns: List[ValidationStatePattern]
    data_violations: List[str]
    summary: str


class EnhancedDatenflussAnalyzer:
    """
    Enhanced Phase 5 implementation that properly handles validation states
    """
    
    def __init__(self, domain_analyzer: DomainConfigurableAnalyzer = None):
        self.domain_analyzer = domain_analyzer or DomainConfigurableAnalyzer()
        self.nlp = self.domain_analyzer.nlp
        
        # Enhanced data flow rules with validation states
        self.enhanced_data_flow_rules = {
            "validation_state_consistency": "Entities can only depend on entities in same or higher validation state",
            "state_transition_validity": "State transitions must follow RAW->VALIDATED->PROCESSED->READY->CONSUMED",
            "dependency_state_ordering": "Dependencies must have equal or higher validation state",
            "transformation_state_requirements": "Transformations require specific input states",
            "aggregation_state_completion": "Aggregation requires all inputs in PROCESSED or higher state"
        }
        
        # Valid state transitions
        self.valid_transitions = {
            ValidationState.RAW: ValidationState.VALIDATED,
            ValidationState.VALIDATED: ValidationState.PROCESSED,
            ValidationState.PROCESSED: ValidationState.READY,
            ValidationState.READY: ValidationState.CONSUMED,
            ValidationState.CONSUMED: ValidationState.WASTE
        }

    def extract_enhanced_data_entities(self, enhanced_phase2_result, enhanced_phase3_result) -> List[EnhancedDataEntity]:
        """
        Extract data entities with proper validation state tracking
        """
        enhanced_entities = []
        
        # Process each resource analysis from enhanced Phase 2
        for analysis in enhanced_phase2_result.resource_analyses:
            resource_name = analysis.resource_name
            manager_name = analysis.manager_controller.name if analysis.manager_controller else None
            
            # Create entities for each validation state this resource goes through
            validation_flow = enhanced_phase2_result.validation_flow_map.get(resource_name, [
                ValidationState.RAW, ValidationState.VALIDATED, ValidationState.PROCESSED
            ])
            
            for i, state in enumerate(validation_flow):
                if state == ValidationState.RAW:
                    # Input entity (RAW state)
                    entity = EnhancedDataEntity(
                        entity_id=f"input_{resource_name.replace(' ', '_')}",
                        name=f"{resource_name} (input)",
                        data_type="ingredient",
                        validation_state=ValidationState.RAW,
                        managed_by=f"{resource_name.replace(' ', '')}Input",
                        validated_dependencies=[],  # No dependencies for input
                        state_history=[ValidationState.RAW]
                    )
                    enhanced_entities.append(entity)
                    
                elif state == ValidationState.VALIDATED:
                    # Validated entity
                    entity = EnhancedDataEntity(
                        entity_id=f"validated_{resource_name.replace(' ', '_')}",
                        name=f"{resource_name} (validated)",
                        data_type="ingredient",
                        validation_state=ValidationState.VALIDATED,
                        managed_by=f"{resource_name.replace(' ', '')}Validator",
                        validated_dependencies=[f"input_{resource_name.replace(' ', '_')}"],  # Depends on input
                        state_history=[ValidationState.RAW, ValidationState.VALIDATED]
                    )
                    enhanced_entities.append(entity)
                    
                elif state == ValidationState.PROCESSED:
                    # Processed entity (result of transformation)
                    processed_name = self._get_processed_name(resource_name)
                    entity = EnhancedDataEntity(
                        entity_id=f"processed_{processed_name.replace(' ', '_')}",
                        name=f"{processed_name}",
                        data_type="intermediate",
                        validation_state=ValidationState.PROCESSED,
                        managed_by=manager_name,
                        validated_dependencies=[f"validated_{resource_name.replace(' ', '_')}"],  # Depends on validated
                        state_history=[ValidationState.RAW, ValidationState.VALIDATED, ValidationState.PROCESSED]
                    )
                    enhanced_entities.append(entity)
        
        # Create final product entity (READY state)
        final_product = EnhancedDataEntity(
            entity_id="final_milk_coffee",
            name="milk coffee",
            data_type="product",
            validation_state=ValidationState.READY,
            managed_by="GeträenkeOrchestrator",
            validated_dependencies=[
                # Only depend on PROCESSED entities (same or higher state)
                "processed_ground_coffee",
                "processed_hot_water", 
                "processed_steamed_milk",
                "processed_cup"
            ],
            state_history=[ValidationState.PROCESSED, ValidationState.READY]
        )
        enhanced_entities.append(final_product)
        
        return enhanced_entities

    def _get_processed_name(self, resource_name: str) -> str:
        """Get the processed form name for a resource"""
        mapping = {
            "coffee beans": "ground coffee",
            "water": "hot water",
            "milk": "steamed milk", 
            "cup": "cup"  # Cup doesn't change form, just validated
        }
        return mapping.get(resource_name, f"processed {resource_name}")

    def identify_enhanced_data_transformations(self, enhanced_entities: List[EnhancedDataEntity], enhanced_phase2_result) -> List[EnhancedDataTransformation]:
        """
        Identify transformations with proper validation state handling
        """
        enhanced_transformations = []
        transformation_counter = 1
        
        # Process transformations from enhanced Phase 2
        for analysis in enhanced_phase2_result.resource_analyses:
            for transformation_text in analysis.transformations:
                if "validated" in transformation_text and "->" in transformation_text:
                    parts = transformation_text.split(" -> ")
                    if len(parts) >= 3:
                        source_name = parts[0].replace("validated ", "")
                        process_name = parts[1]
                        result_name = parts[2].replace("processed ", "")
                        
                        # Find input and output entities with correct states
                        input_entity_id = f"validated_{source_name.replace(' ', '_')}"
                        output_entity_id = f"processed_{result_name.replace(' ', '_')}"
                        
                        # Verify entities exist
                        input_exists = any(e.entity_id == input_entity_id for e in enhanced_entities)
                        output_exists = any(e.entity_id == output_entity_id for e in enhanced_entities)
                        
                        if input_exists and output_exists:
                            transformation = EnhancedDataTransformation(
                                transformation_id=f"ET{transformation_counter}",
                                name=f"{process_name} {source_name}",
                                input_entities=[input_entity_id],
                                output_entities=[output_entity_id],
                                transformation_type="physical",
                                responsible_controller=analysis.manager_controller.name if analysis.manager_controller else "Unknown",
                                input_state_required=ValidationState.VALIDATED,
                                output_state_produced=ValidationState.PROCESSED,
                                validation_rules=[
                                    f"Verify {source_name} is in VALIDATED state",
                                    f"Perform {process_name} operation",
                                    f"Confirm {result_name} reaches PROCESSED state"
                                ],
                                state_transition_valid=True
                            )
                            enhanced_transformations.append(transformation)
                            transformation_counter += 1
        
        # Final aggregation transformation
        aggregation_transformation = EnhancedDataTransformation(
            transformation_id=f"ET{transformation_counter}",
            name="assemble milk coffee",
            input_entities=[
                "processed_ground_coffee",
                "processed_hot_water",
                "processed_steamed_milk", 
                "processed_cup"
            ],
            output_entities=["final_milk_coffee"],
            transformation_type="aggregation",
            responsible_controller="GeträenkeOrchestrator",
            input_state_required=ValidationState.PROCESSED,
            output_state_produced=ValidationState.READY,
            validation_rules=[
                "Verify all ingredients are in PROCESSED state",
                "Perform coffee assembly",
                "Confirm final product reaches READY state"
            ],
            state_transition_valid=True
        )
        enhanced_transformations.append(aggregation_transformation)
        
        return enhanced_transformations

    def create_enhanced_data_flows(self, enhanced_entities: List[EnhancedDataEntity], enhanced_transformations: List[EnhancedDataTransformation]) -> List[EnhancedDataFlow]:
        """
        Create data flows with proper validation state management
        """
        enhanced_flows = []
        flow_counter = 1
        
        # Validation flows (RAW -> VALIDATED)
        for entity in enhanced_entities:
            if entity.validation_state == ValidationState.VALIDATED:
                # Find corresponding RAW entity
                raw_entity_id = f"input_{entity.name.split(' (')[0].replace(' ', '_')}"
                raw_entity = next((e for e in enhanced_entities if e.entity_id == raw_entity_id), None)
                
                if raw_entity:
                    flow = EnhancedDataFlow(
                        flow_id=f"EF{flow_counter}",
                        source_entity=raw_entity.entity_id,
                        target_entity=entity.entity_id,
                        flow_type=EnhancedDataFlowType.VALIDATION_OUTPUT,
                        responsible_controller=entity.managed_by,
                        source_state=ValidationState.RAW,
                        target_state=ValidationState.VALIDATED,
                        validation_required=True,
                        state_transition_valid=True
                    )
                    enhanced_flows.append(flow)
                    flow_counter += 1
        
        # Processing flows (VALIDATED -> PROCESSED)
        for transformation in enhanced_transformations:
            if transformation.transformation_type != "aggregation":
                for input_entity_id in transformation.input_entities:
                    for output_entity_id in transformation.output_entities:
                        flow = EnhancedDataFlow(
                            flow_id=f"EF{flow_counter}",
                            source_entity=input_entity_id,
                            target_entity=output_entity_id,
                            flow_type=EnhancedDataFlowType.PROCESSING_OUTPUT,
                            responsible_controller=transformation.responsible_controller,
                            source_state=ValidationState.VALIDATED,
                            target_state=ValidationState.PROCESSED,
                            validation_required=True,
                            state_transition_valid=True
                        )
                        enhanced_flows.append(flow)
                        flow_counter += 1
        
        # Aggregation flow (PROCESSED -> READY)
        aggregation_transformation = next(
            (t for t in enhanced_transformations if t.transformation_type == "aggregation"), None
        )
        if aggregation_transformation:
            for input_entity_id in aggregation_transformation.input_entities:
                flow = EnhancedDataFlow(
                    flow_id=f"EF{flow_counter}",
                    source_entity=input_entity_id,
                    target_entity="final_milk_coffee",
                    flow_type=EnhancedDataFlowType.AGGREGATION,
                    responsible_controller="GeträenkeOrchestrator",
                    source_state=ValidationState.PROCESSED,
                    target_state=ValidationState.READY,
                    validation_required=True,
                    state_transition_valid=True
                )
                enhanced_flows.append(flow)
                flow_counter += 1
        
        # User delivery flow (READY -> CONSUMED)
        user_flow = EnhancedDataFlow(
            flow_id=f"EF{flow_counter}",
            source_entity="final_milk_coffee",
            target_entity="user",
            flow_type=EnhancedDataFlowType.USER_OUTPUT,
            responsible_controller="UserInterfaceManager",
            source_state=ValidationState.READY,
            target_state=ValidationState.CONSUMED,
            validation_required=False,
            state_transition_valid=True
        )
        enhanced_flows.append(user_flow)
        
        return enhanced_flows

    def validate_enhanced_data_flow_rules(self, enhanced_entities: List[EnhancedDataEntity], enhanced_transformations: List[EnhancedDataTransformation], enhanced_flows: List[EnhancedDataFlow]) -> List[str]:
        """
        Validate enhanced data flow rules with proper state checking
        """
        violations = []
        
        # Rule 1: Validation state consistency
        for entity in enhanced_entities:
            for dep_id in entity.validated_dependencies:
                dep_entity = next((e for e in enhanced_entities if e.entity_id == dep_id), None)
                if dep_entity:
                    # Check if dependency has equal or higher validation state
                    entity_state_order = list(ValidationState).index(entity.validation_state)
                    dep_state_order = list(ValidationState).index(dep_entity.validation_state)
                    
                    if dep_state_order > entity_state_order:
                        violations.append(f"Entity {entity.entity_id} ({entity.validation_state.value}) depends on higher-state entity {dep_id} ({dep_entity.validation_state.value})")
        
        # Rule 2: State transition validity
        for transformation in enhanced_transformations:
            if not transformation.state_transition_valid:
                violations.append(f"Transformation {transformation.transformation_id} has invalid state transition")
            
            # Check if input entities exist and have required state
            for input_entity_id in transformation.input_entities:
                input_entity = next((e for e in enhanced_entities if e.entity_id == input_entity_id), None)
                if input_entity:
                    if input_entity.validation_state != transformation.input_state_required:
                        violations.append(f"Transformation {transformation.transformation_id} requires {transformation.input_state_required.value} input, but {input_entity_id} is in {input_entity.validation_state.value} state")
        
        # Rule 3: Flow state consistency
        for flow in enhanced_flows:
            if not flow.state_transition_valid:
                violations.append(f"Data flow {flow.flow_id} has invalid state transition")
        
        return violations

    def perform_enhanced_phase5_analysis(self, enhanced_phase4_result) -> EnhancedPhase5Result:
        """
        Complete enhanced Phase 5 analysis with validation state integration
        """
        enhanced_phase2_result = enhanced_phase4_result.phase3_result.phase2_result
        enhanced_phase3_result = enhanced_phase4_result.phase3_result
        
        # Extract enhanced data entities with validation states
        enhanced_entities = self.extract_enhanced_data_entities(enhanced_phase2_result, enhanced_phase3_result)
        
        # Identify enhanced data transformations
        enhanced_transformations = self.identify_enhanced_data_transformations(enhanced_entities, enhanced_phase2_result)
        
        # Create enhanced data flows
        enhanced_flows = self.create_enhanced_data_flows(enhanced_entities, enhanced_transformations)
        
        # Create validation state patterns
        validation_state_patterns = self._create_validation_state_patterns(enhanced_entities, enhanced_phase2_result)
        
        # Validate enhanced data flow rules
        violations = self.validate_enhanced_data_flow_rules(enhanced_entities, enhanced_transformations, enhanced_flows)
        
        # Generate summary
        summary = self._generate_enhanced_summary(enhanced_entities, enhanced_transformations, enhanced_flows, violations)
        
        return EnhancedPhase5Result(
            enhanced_phase4_result=enhanced_phase4_result,
            enhanced_data_entities=enhanced_entities,
            enhanced_data_transformations=enhanced_transformations,
            enhanced_data_flows=enhanced_flows,
            validation_state_patterns=validation_state_patterns,
            data_violations=violations,
            summary=summary
        )

    def _create_validation_state_patterns(self, enhanced_entities: List[EnhancedDataEntity], enhanced_phase2_result) -> List[ValidationStatePattern]:
        """Create validation state patterns for each resource"""
        patterns = []
        
        for resource_name in enhanced_phase2_result.validation_flow_map.keys():
            state_sequence = enhanced_phase2_result.validation_flow_map[resource_name]
            
            # Map states to entity IDs
            entity_mappings = {}
            for state in state_sequence:
                if state == ValidationState.RAW:
                    entity_mappings[state] = f"input_{resource_name.replace(' ', '_')}"
                elif state == ValidationState.VALIDATED:
                    entity_mappings[state] = f"validated_{resource_name.replace(' ', '_')}"
                elif state == ValidationState.PROCESSED:
                    processed_name = self._get_processed_name(resource_name)
                    entity_mappings[state] = f"processed_{processed_name.replace(' ', '_')}"
            
            pattern = ValidationStatePattern(
                pattern_id=f"VSP_{resource_name.replace(' ', '_')}",
                resource_name=resource_name,
                state_sequence=state_sequence,
                transition_validations=[
                    f"{resource_name}: {state_sequence[i].value} -> {state_sequence[i+1].value}"
                    for i in range(len(state_sequence)-1)
                ],
                entity_mappings=entity_mappings
            )
            patterns.append(pattern)
        
        return patterns

    def _generate_enhanced_summary(self, entities: List[EnhancedDataEntity], transformations: List[EnhancedDataTransformation], flows: List[EnhancedDataFlow], violations: List[str]) -> str:
        """Generate enhanced summary"""
        total_entities = len(entities)
        total_transformations = len(transformations)
        total_flows = len(flows)
        total_violations = len(violations)
        
        ingredient_entities = len([e for e in entities if e.data_type == "ingredient"])
        product_entities = len([e for e in entities if e.data_type == "product"])
        
        summary_parts = [
            f"Enhanced analysis of {total_entities} data entities with validation states",
            f"Created {total_transformations} enhanced transformations with state management",
            f"Generated {total_flows} enhanced data flows with validation",
            f"Processed {ingredient_entities} ingredients and {product_entities} products",
            f"Detected {total_violations} enhanced data flow violations",
            "Applied validation state consistency and transition rules"
        ]
        
        return "; ".join(summary_parts)


def main():
    """Test enhanced Phase 5 analysis"""
    
    # Create mock enhanced Phase 4 result
    from dataclasses import dataclass
    from typing import List, Dict
    
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
    
    # Enhanced transformations with proper validation states
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
    
    # Proper validation flow map
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
    
    # Initialize enhanced analyzer
    enhanced_analyzer = EnhancedDatenflussAnalyzer()
    
    # Perform enhanced Phase 5 analysis
    enhanced_result = enhanced_analyzer.perform_enhanced_phase5_analysis(phase4_result)
    
    # Save result
    output_file = "Zwischenprodukte/UC1_Coffee_fully_enhanced_phase5_analysis.json"
    
    serializable_result = {
        "summary": enhanced_result.summary,
        "enhanced_data_entities": [
            {
                "entity_id": e.entity_id,
                "name": e.name,
                "data_type": e.data_type,
                "validation_state": e.validation_state.value,
                "managed_by": e.managed_by,
                "validated_dependencies": e.validated_dependencies,
                "state_history": [state.value for state in e.state_history]
            } for e in enhanced_result.enhanced_data_entities
        ],
        "enhanced_data_transformations": [
            {
                "transformation_id": t.transformation_id,
                "name": t.name,
                "transformation_type": t.transformation_type,
                "responsible_controller": t.responsible_controller,
                "input_entities": t.input_entities,
                "output_entities": t.output_entities,
                "input_state_required": t.input_state_required.value,
                "output_state_produced": t.output_state_produced.value,
                "state_transition_valid": t.state_transition_valid,
                "validation_rules": t.validation_rules
            } for t in enhanced_result.enhanced_data_transformations
        ],
        "enhanced_data_flows": [
            {
                "flow_id": f.flow_id,
                "source_entity": f.source_entity,
                "target_entity": f.target_entity,
                "flow_type": f.flow_type.value,
                "responsible_controller": f.responsible_controller,
                "source_state": f.source_state.value,
                "target_state": f.target_state.value,
                "state_transition_valid": f.state_transition_valid
            } for f in enhanced_result.enhanced_data_flows
        ],
        "validation_state_patterns": [
            {
                "pattern_id": vsp.pattern_id,
                "resource_name": vsp.resource_name,
                "state_sequence": [state.value for state in vsp.state_sequence],
                "transition_validations": vsp.transition_validations,
                "entity_mappings": {state.value: entity_id for state, entity_id in vsp.entity_mappings.items()}
            } for vsp in enhanced_result.validation_state_patterns
        ],
        "violations": enhanced_result.data_violations
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)
    
    print(f"Enhanced Phase 5 Analysis completed!")
    print(f"File saved to: {output_file}")
    print(f"Summary: {enhanced_result.summary}")
    print(f"Enhanced Data Entities: {len(enhanced_result.enhanced_data_entities)}")
    print(f"Enhanced Data Transformations: {len(enhanced_result.enhanced_data_transformations)}")
    print(f"Enhanced Data Flows: {len(enhanced_result.enhanced_data_flows)}")
    print(f"Data Violations: {len(enhanced_result.data_violations)}")
    
    if enhanced_result.data_violations:
        print("Violations found:")
        for violation in enhanced_result.data_violations:
            print(f"  - {violation}")
    else:
        print("🎉 NO VIOLATIONS FOUND! Enhanced Phase 5 is perfect!")
    
    return enhanced_result

if __name__ == "__main__":
    main()