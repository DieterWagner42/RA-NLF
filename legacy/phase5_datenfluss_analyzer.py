"""
Phase 5: Datenfluss-Analyse (Data Flow Analysis) - UC-Methode.txt Implementation
Analyzes data flow patterns, transformations, and information exchange
Ensures proper data flow rules are followed based on entities and transformations
"""

import spacy
import json
import os
import re
from typing import Dict, List, Set, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from domain_config_analyzer import DomainConfigurableAnalyzer, DomainConfig, Phase1Result
from phase2_betriebsmittel_analyzer import Phase2Result
from phase3_interaktion_analyzer import Phase3Result
from phase4_kontrollfluss_analyzer import Phase4Result


class DataFlowType(Enum):
    INPUT = "input"
    TRANSFORMATION = "transformation"
    OUTPUT = "output"
    STORAGE = "storage"
    VALIDATION = "validation"
    AGGREGATION = "aggregation"


class DataStateType(Enum):
    RAW = "raw"
    VALIDATED = "validated"
    PROCESSED = "processed"
    TRANSFORMED = "transformed"
    READY = "ready"
    CONSUMED = "consumed"
    WASTE = "waste"


@dataclass
class DataEntity:
    entity_id: str
    name: str
    data_type: str  # "ingredient", "product", "intermediate", "waste"
    initial_state: DataStateType
    current_state: DataStateType
    properties: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    managed_by: Optional[str] = None  # Controller responsible


@dataclass
class DataTransformation:
    transformation_id: str
    name: str
    input_entities: List[str]
    output_entities: List[str]
    transformation_type: str  # "physical", "logical", "state_change"
    responsible_controller: str
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)


@dataclass
class DataFlow:
    flow_id: str
    source_entity: str
    target_entity: str
    flow_type: DataFlowType
    responsible_controller: str
    data_volume: Optional[str] = None
    flow_conditions: List[str] = field(default_factory=list)
    validation_required: bool = False


@dataclass
class DataFlowPattern:
    pattern_id: str
    pattern_type: str  # "input_processing", "transformation_chain", "output_delivery"
    entities: List[DataEntity]
    transformations: List[DataTransformation]
    flows: List[DataFlow]
    data_integrity_rules: List[str] = field(default_factory=list)


@dataclass
class Phase5Result:
    phase4_result: Phase4Result
    data_entities: List[DataEntity]
    data_transformations: List[DataTransformation]
    data_flows: List[DataFlow]
    data_flow_patterns: List[DataFlowPattern]
    data_violations: List[str]
    summary: str


class DatenflussAnalyzer:
    """
    Phase 5 implementation: Analyzes data flow patterns and ensures proper data flow rules
    """
    
    def __init__(self, domain_analyzer: DomainConfigurableAnalyzer = None):
        """
        Initialize with optional domain analyzer for context
        """
        self.domain_analyzer = domain_analyzer or DomainConfigurableAnalyzer()
        self.nlp = self.domain_analyzer.nlp
        
        # Data transformation patterns
        self.transformation_patterns = {
            "physical": ["grinding", "heating", "steaming", "brewing", "mixing"],
            "logical": ["measuring", "calculating", "determining", "selecting"],
            "state_change": ["activating", "preparing", "presenting", "storing"]
        }
        
        # Data flow rules for UC-Methode
        self.data_flow_rules = {
            "input_validation": "All input data must be validated before processing",
            "transformation_integrity": "Data transformations must preserve integrity",
            "state_consistency": "Entity states must be consistent throughout flow",
            "dependency_ordering": "Data dependencies must determine processing order",
            "output_completeness": "All required output data must be produced",
            "waste_management": "Waste data must be properly handled and disposed"
        }
        
        # Domain-specific data characteristics
        self.domain_data_characteristics = {
            "beverage_preparation": {
                "ingredients": ["coffee beans", "water", "milk", "sugar"],
                "containers": ["cup", "filter", "reservoir"],
                "products": ["ground coffee", "brewed coffee", "steamed milk"],
                "waste": ["used grounds", "filter waste", "excess water"]
            }
        }

    def extract_data_entities(self, phase2_result: Phase2Result, phase3_result: Phase3Result) -> List[DataEntity]:
        """
        Extract data entities from Phase 2 resources and Phase 3 interactions
        """
        entities = []
        
        # Extract entities from Phase 2 resource analyses
        for analysis in phase2_result.resource_analyses:
            # Input entities (raw resources)
            input_entity = DataEntity(
                entity_id=f"input_{analysis.resource_name.replace(' ', '_')}",
                name=analysis.resource_name,
                data_type="ingredient",
                initial_state=DataStateType.RAW,
                current_state=DataStateType.RAW,
                managed_by=analysis.manager_controller.name if analysis.manager_controller else None
            )
            entities.append(input_entity)
            
            # RUP-KORREKTUR: Keine Entity-Entity Abhängigkeiten!
            # Entities haben keine dependencies - nur Controller haben use/provide Beziehungen!
            
            # Transformation entities (processed forms)
            for transformation in analysis.transformations:
                parts = transformation.split(" -> ")
                if len(parts) >= 3:  # source -> process -> result
                    source_name = parts[0]
                    result_name = parts[2]
                    
                    # RUP-KORREKTUR: Entities haben KEINE dependencies!
                    # Controller use/provide Entities, aber Entities haben keine Abhängigkeiten untereinander
                    
                    result_entity = DataEntity(
                        entity_id=f"processed_{result_name.replace(' ', '_')}",
                        name=result_name,
                        data_type="intermediate",
                        initial_state=DataStateType.PROCESSED,
                        current_state=DataStateType.PROCESSED,
                        dependencies=[],  # RUP: Keine Entity-Entity dependencies!
                        managed_by=analysis.manager_controller.name if analysis.manager_controller else None
                    )
                    entities.append(result_entity)
        
        # RUP-KORREKTUR: Final product hat KEINE dependencies!
        # GeträenkeOrchestrator use-> processed entities und provide-> final product
        
        final_product = DataEntity(
            entity_id="final_milk_coffee",
            name="milk coffee",
            data_type="product",
            initial_state=DataStateType.READY,
            current_state=DataStateType.READY,
            dependencies=[],  # RUP: Keine Entity-Entity dependencies!
            managed_by="GeträenkeOrchestrator"
        )
        entities.append(final_product)
        
        return entities

    def identify_data_transformations(self, entities: List[DataEntity], phase2_result: Phase2Result) -> List[DataTransformation]:
        """
        Identify data transformations based on Phase 2 transformation contexts
        """
        transformations = []
        transformation_counter = 1
        
        for analysis in phase2_result.resource_analyses:
            for transformation_text in analysis.transformations:
                parts = transformation_text.split(" -> ")
                if len(parts) >= 3:  # source -> process -> result
                    source_name = parts[0]
                    process_name = parts[1]
                    result_name = parts[2]
                    
                    # Find input and output entities
                    input_entities = [e.entity_id for e in entities if source_name in e.name]
                    output_entities = [e.entity_id for e in entities if result_name in e.name]
                    
                    # Determine transformation type
                    transformation_type = self._classify_transformation_type(process_name)
                    
                    transformation = DataTransformation(
                        transformation_id=f"T{transformation_counter}",
                        name=f"{process_name} {source_name}",
                        input_entities=input_entities,
                        output_entities=output_entities,
                        transformation_type=transformation_type,
                        responsible_controller=analysis.manager_controller.name if analysis.manager_controller else "Unknown",
                        preconditions=[f"{source_name} available and validated"],
                        postconditions=[f"{result_name} produced and ready"],
                        validation_rules=[f"Validate {source_name} quality", f"Verify {process_name} completion"]
                    )
                    transformations.append(transformation)
                    transformation_counter += 1
        
        # Final assembly transformation
        final_transformation = DataTransformation(
            transformation_id=f"T{transformation_counter}",
            name="assemble milk coffee",
            input_entities=[e.entity_id for e in entities if e.data_type in ["intermediate", "ingredient"]],
            output_entities=["final_milk_coffee"],
            transformation_type="aggregation",
            responsible_controller="GeträenkeOrchestrator",
            preconditions=["All ingredients processed", "Cup available"],
            postconditions=["Milk coffee ready for user"],
            validation_rules=["Verify all components present", "Check temperature", "Confirm quality"]
        )
        transformations.append(final_transformation)
        
        return transformations

    def _classify_transformation_type(self, process_name: str) -> str:
        """
        Classify the type of transformation based on the process
        """
        process_lower = process_name.lower()
        
        for trans_type, indicators in self.transformation_patterns.items():
            if any(indicator in process_lower for indicator in indicators):
                return trans_type
        
        return "physical"  # Default

    def create_data_flows(self, entities: List[DataEntity], transformations: List[DataTransformation]) -> List[DataFlow]:
        """
        Create data flows between entities through transformations
        """
        flows = []
        flow_counter = 1
        
        for transformation in transformations:
            # Input flows (entities -> transformation)
            for input_entity_id in transformation.input_entities:
                flow = DataFlow(
                    flow_id=f"F{flow_counter}",
                    source_entity=input_entity_id,
                    target_entity=transformation.transformation_id,
                    flow_type=DataFlowType.INPUT,
                    responsible_controller=transformation.responsible_controller,
                    validation_required=True,
                    flow_conditions=transformation.preconditions
                )
                flows.append(flow)
                flow_counter += 1
            
            # Output flows (transformation -> entities)
            for output_entity_id in transformation.output_entities:
                flow = DataFlow(
                    flow_id=f"F{flow_counter}",
                    source_entity=transformation.transformation_id,
                    target_entity=output_entity_id,
                    flow_type=DataFlowType.OUTPUT,
                    responsible_controller=transformation.responsible_controller,
                    flow_conditions=transformation.postconditions
                )
                flows.append(flow)
                flow_counter += 1
        
        # User delivery flow
        user_flow = DataFlow(
            flow_id=f"F{flow_counter}",
            source_entity="final_milk_coffee",
            target_entity="user",
            flow_type=DataFlowType.OUTPUT,
            responsible_controller="UserInterfaceManager",
            flow_conditions=["Coffee ready", "User present"]
        )
        flows.append(user_flow)
        
        return flows

    def create_data_flow_patterns(self, entities: List[DataEntity], transformations: List[DataTransformation], flows: List[DataFlow]) -> List[DataFlowPattern]:
        """
        Create data flow patterns for different processing stages
        """
        patterns = []
        
        # Input processing pattern
        input_entities = [e for e in entities if e.data_type == "ingredient"]
        input_transformations = [t for t in transformations if t.transformation_type in ["physical", "validation"]]
        input_flows = [f for f in flows if f.flow_type == DataFlowType.INPUT]
        
        input_pattern = DataFlowPattern(
            pattern_id="input_processing",
            pattern_type="input_processing",
            entities=input_entities,
            transformations=input_transformations,
            flows=input_flows,
            data_integrity_rules=[
                "Validate all ingredient quality",
                "Check ingredient availability",
                "Ensure proper storage conditions"
            ]
        )
        patterns.append(input_pattern)
        
        # Transformation chain pattern
        intermediate_entities = [e for e in entities if e.data_type == "intermediate"]
        transformation_chain = [t for t in transformations if t.transformation_type == "physical"]
        transformation_flows = [f for f in flows if f.flow_type == DataFlowType.TRANSFORMATION]
        
        transformation_pattern = DataFlowPattern(
            pattern_id="transformation_chain",
            pattern_type="transformation_chain",
            entities=intermediate_entities,
            transformations=transformation_chain,
            flows=transformation_flows,
            data_integrity_rules=[
                "Maintain ingredient properties during transformation",
                "Ensure transformation completeness",
                "Preserve quality throughout process"
            ]
        )
        patterns.append(transformation_pattern)
        
        # Output delivery pattern
        output_entities = [e for e in entities if e.data_type == "product"]
        output_transformations = [t for t in transformations if t.transformation_type in ["aggregation", "state_change"]]
        output_flows = [f for f in flows if f.flow_type == DataFlowType.OUTPUT and f.target_entity == "user"]
        
        output_pattern = DataFlowPattern(
            pattern_id="output_delivery",
            pattern_type="output_delivery",
            entities=output_entities,
            transformations=output_transformations,
            flows=output_flows,
            data_integrity_rules=[
                "Ensure final product quality",
                "Verify complete assembly",
                "Confirm user delivery readiness"
            ]
        )
        patterns.append(output_pattern)
        
        return patterns

    def validate_data_flow_rules(self, entities: List[DataEntity], transformations: List[DataTransformation], flows: List[DataFlow]) -> List[str]:
        """
        Validate that data flow follows UC-Methode rules
        """
        violations = []
        
        # Rule 1: Input validation
        input_flows = [f for f in flows if f.flow_type == DataFlowType.INPUT]
        for flow in input_flows:
            if not flow.validation_required:
                violations.append(f"Input flow {flow.flow_id} lacks required validation")
        
        # Rule 2: Transformation integrity
        for transformation in transformations:
            if not transformation.validation_rules:
                violations.append(f"Transformation {transformation.transformation_id} lacks validation rules")
            
            # Check if all inputs have corresponding entities
            for input_id in transformation.input_entities:
                if not any(e.entity_id == input_id for e in entities):
                    violations.append(f"Transformation {transformation.transformation_id} references missing input entity {input_id}")
        
        # Rule 3: State consistency
        for entity in entities:
            if entity.initial_state != entity.current_state and entity.data_type == "ingredient":
                violations.append(f"Entity {entity.entity_id} has inconsistent state: {entity.initial_state} -> {entity.current_state}")
        
        # Rule 4: RUP-KORREKT - Entities haben KEINE dependencies!
        for entity in entities:
            if entity.dependencies:
                violations.append(f"RUP VIOLATION: Entity {entity.entity_id} has dependencies {entity.dependencies} - Entities dürfen keine dependencies haben!")
        
        # Rule 5: Output completeness
        user_flows = [f for f in flows if f.target_entity == "user"]
        if not user_flows:
            violations.append("No data flow to user found - output completeness violated")
        
        return violations

    def perform_phase5_analysis(self, phase4_result: Phase4Result) -> Phase5Result:
        """
        Complete Phase 5 data flow analysis
        
        Args:
            phase4_result: Result from Phase 4 analysis
            
        Returns:
            Complete Phase5Result
        """
        phase2_result = phase4_result.phase3_result.phase2_result
        phase3_result = phase4_result.phase3_result
        
        # Extract data entities from previous phases
        entities = self.extract_data_entities(phase2_result, phase3_result)
        
        # Identify data transformations
        transformations = self.identify_data_transformations(entities, phase2_result)
        
        # Create data flows
        flows = self.create_data_flows(entities, transformations)
        
        # Create data flow patterns
        patterns = self.create_data_flow_patterns(entities, transformations, flows)
        
        # Validate data flow rules
        violations = self.validate_data_flow_rules(entities, transformations, flows)
        
        # Generate summary
        summary = self._generate_phase5_summary(entities, transformations, flows, violations)
        
        return Phase5Result(
            phase4_result=phase4_result,
            data_entities=entities,
            data_transformations=transformations,
            data_flows=flows,
            data_flow_patterns=patterns,
            data_violations=violations,
            summary=summary
        )

    def _generate_phase5_summary(self, entities: List[DataEntity], transformations: List[DataTransformation], flows: List[DataFlow], violations: List[str]) -> str:
        """
        Generate summary of Phase 5 analysis
        """
        total_entities = len(entities)
        total_transformations = len(transformations)
        total_flows = len(flows)
        total_violations = len(violations)
        
        ingredient_entities = len([e for e in entities if e.data_type == "ingredient"])
        product_entities = len([e for e in entities if e.data_type == "product"])
        
        summary_parts = [
            f"Analyzed {total_entities} data entities ({ingredient_entities} ingredients, {product_entities} products)",
            f"Identified {total_transformations} data transformations",
            f"Created {total_flows} data flows",
            f"Detected {total_violations} data flow violations",
            "Applied UC-Methode data flow rules"
        ]
        
        return "; ".join(summary_parts)


def main():
    """Example usage with UC1"""
    # Load Phase 4 result (mock for testing)
    from dataclasses import dataclass
    from typing import Dict
    
    @dataclass
    class MockPhase2Analysis:
        resource_name: str
        transformations: List[str]
        manager_controller: object = None
    
    @dataclass 
    class MockManagerController:
        name: str
    
    @dataclass
    class MockPhase2Result:
        resource_analyses: List
    
    @dataclass
    class MockPhase3Result:
        phase2_result: MockPhase2Result
        all_objects: Dict = field(default_factory=dict)
    
    @dataclass
    class MockPhase4Result:
        phase3_result: MockPhase3Result
    
    # Create mock data
    mock_analyses = [
        MockPhase2Analysis(
            resource_name="coffee beans",
            transformations=["beans -> grinding -> ground coffee"],
            manager_controller=MockManagerController("CoffeeBeansManager")
        ),
        MockPhase2Analysis(
            resource_name="water",
            transformations=["water -> heating -> hot water"],
            manager_controller=MockManagerController("WaterManager")
        ),
        MockPhase2Analysis(
            resource_name="milk",
            transformations=["milk -> steaming -> steamed milk"],
            manager_controller=MockManagerController("MilkManager")
        )
    ]
    
    phase2_result = MockPhase2Result(resource_analyses=mock_analyses)
    phase3_result = MockPhase3Result(phase2_result=phase2_result)
    phase4_result = MockPhase4Result(phase3_result=phase3_result)
    
    # Initialize analyzer
    analyzer = DatenflussAnalyzer()
    
    # Perform Phase 5 analysis
    phase5_result = analyzer.perform_phase5_analysis(phase4_result)
    
    # Save result
    output_file = "Zwischenprodukte/UC1_Coffee_phase5_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
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
            "data_flows": [
                {
                    "flow_id": f.flow_id,
                    "source_entity": f.source_entity,
                    "target_entity": f.target_entity,
                    "flow_type": f.flow_type.value,
                    "responsible_controller": f.responsible_controller,
                    "validation_required": f.validation_required
                } for f in phase5_result.data_flows
            ],
            "violations": phase5_result.data_violations
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Phase 5 Data Flow Analysis completed!")
    print(f"File saved to: {output_file}")
    print(f"Summary: {phase5_result.summary}")
    if phase5_result.data_violations:
        print(f"Violations found: {len(phase5_result.data_violations)}")
        for violation in phase5_result.data_violations:
            print(f"  - {violation}")
    else:
        print("No data flow violations detected!")

if __name__ == "__main__":
    main()