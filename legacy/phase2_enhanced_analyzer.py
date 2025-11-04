"""
Enhanced Phase 2: Betriebsmittel-Analyse with Validation Integration
Fixes Phase 5 violations by implementing proper validation states:
RAW -> VALIDATED -> PROCESSED -> READY -> CONSUMED
"""

import spacy
import json
import os
import re
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from domain_config_analyzer import DomainConfigurableAnalyzer, DomainConfig, Phase1Result


class ResourceType(Enum):
    INPUT_BOUNDARY = "input_boundary"
    VALIDATION_BOUNDARY = "validation_boundary"  # NEW: Explicit validation
    MANAGER_CONTROLLER = "manager_controller"  
    OUTPUT_BOUNDARY = "output_boundary"
    ENTITY = "entity"


class ValidationState(Enum):
    RAW = "raw"
    VALIDATED = "validated"
    PROCESSED = "processed"
    READY = "ready"
    CONSUMED = "consumed"
    WASTE = "waste"


@dataclass
class EnhancedResourceFunction:
    name: str
    source: str  # "explicit", "implicit", "validation", "context"
    description: str
    input_state: Optional[ValidationState] = None
    output_state: Optional[ValidationState] = None


@dataclass
class ValidationController:
    name: str
    resource_name: str
    validation_functions: List[EnhancedResourceFunction]
    quality_checks: List[str]
    quantity_checks: List[str]
    safety_checks: List[str]


@dataclass
class EnhancedResourceObject:
    name: str
    type: ResourceType
    resource_name: str
    functions: List[EnhancedResourceFunction] = field(default_factory=list)
    validation_controller: Optional[ValidationController] = None
    context_derived: bool = False
    domain_specific: bool = False
    current_state: ValidationState = ValidationState.RAW


@dataclass
class EnhancedBetriebsmittelAnalysis:
    resource_name: str
    input_boundary: Optional[EnhancedResourceObject] = None
    validation_boundary: Optional[EnhancedResourceObject] = None  # NEW
    manager_controller: Optional[EnhancedResourceObject] = None
    output_boundary: Optional[EnhancedResourceObject] = None
    entities: List[EnhancedResourceObject] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)
    validation_flow: List[str] = field(default_factory=list)  # NEW


@dataclass
class EnhancedPhase2Result:
    phase1_result: Phase1Result
    resource_analyses: List[EnhancedBetriebsmittelAnalysis]
    validation_controllers: List[ValidationController]  # NEW
    all_objects: Dict[str, List[EnhancedResourceObject]]
    validation_flow_map: Dict[str, List[ValidationState]]  # NEW
    summary: str


class EnhancedBetriebsmittelAnalyzer:
    """
    Enhanced Phase 2 implementation with explicit validation states
    Fixes Phase 5 violations by proper state management
    """
    
    def __init__(self, domain_analyzer: DomainConfigurableAnalyzer = None):
        self.domain_analyzer = domain_analyzer or DomainConfigurableAnalyzer()
        self.nlp = self.domain_analyzer.nlp
        
        # Enhanced validation patterns for different domains
        self.validation_requirements = {
            "beverage_preparation": {
                "coffee": ["quality_check", "freshness_check", "quantity_check"],
                "water": ["purity_check", "temperature_check", "quantity_check"],
                "milk": ["freshness_check", "fat_content_check", "quantity_check"],
                "cup": ["cleanliness_check", "material_check", "availability_check"]
            }
        }
        
        # Validation state transitions
        self.state_transitions = {
            ValidationState.RAW: [ValidationState.VALIDATED],
            ValidationState.VALIDATED: [ValidationState.PROCESSED],
            ValidationState.PROCESSED: [ValidationState.READY],
            ValidationState.READY: [ValidationState.CONSUMED, ValidationState.WASTE],
            ValidationState.CONSUMED: [],
            ValidationState.WASTE: []
        }

    def create_validation_controller(self, resource_name: str, domain: str) -> ValidationController:
        """
        Create explicit validation controller for resource
        """
        validation_checks = self.validation_requirements.get(domain, {}).get(
            resource_name.split()[0].lower(), ["basic_quality_check", "basic_quantity_check"]
        )
        
        validation_functions = []
        
        # Create validation functions with state transitions
        for check in validation_checks:
            function = EnhancedResourceFunction(
                name=f"perform_{check}",
                source="validation",
                description=f"Perform {check.replace('_', ' ')} on {resource_name}",
                input_state=ValidationState.RAW,
                output_state=ValidationState.VALIDATED
            )
            validation_functions.append(function)
        
        # Add confirmation function
        confirmation_function = EnhancedResourceFunction(
            name="confirm_validation",
            source="validation",
            description=f"Confirm {resource_name} passes all validation checks",
            input_state=ValidationState.VALIDATED,
            output_state=ValidationState.VALIDATED
        )
        validation_functions.append(confirmation_function)
        
        return ValidationController(
            name=f"{resource_name.replace(' ', '')}Validator",
            resource_name=resource_name,
            validation_functions=validation_functions,
            quality_checks=[check for check in validation_checks if "quality" in check],
            quantity_checks=[check for check in validation_checks if "quantity" in check],
            safety_checks=[check for check in validation_checks if "safety" in check or "purity" in check]
        )

    def create_enhanced_input_boundary(self, resource_name: str, domain_config: DomainConfig, phase1_result: Phase1Result) -> EnhancedResourceObject:
        """
        Create enhanced input boundary with explicit validation integration
        """
        normalized_resource = resource_name.replace(' ', '').title()
        
        functions = [
            EnhancedResourceFunction(
                name="accept_input",
                source="explicit",
                description=f"Accept {resource_name} from external source",
                input_state=None,  # External input
                output_state=ValidationState.RAW
            ),
            EnhancedResourceFunction(
                name="register_arrival",
                source="implicit",
                description=f"Register {resource_name} arrival in system",
                input_state=ValidationState.RAW,
                output_state=ValidationState.RAW
            ),
            EnhancedResourceFunction(
                name="forward_to_validation",
                source="validation",
                description=f"Forward {resource_name} to validation controller",
                input_state=ValidationState.RAW,
                output_state=ValidationState.RAW
            )
        ]
        
        # Add human feedback if applicable
        if phase1_result and any(actor.type.value == "human" for actor in phase1_result.actors):
            functions.append(EnhancedResourceFunction(
                name="provide_user_feedback",
                source="phase1_integration",
                description=f"Provide feedback to human actors about {resource_name} arrival",
                input_state=ValidationState.RAW,
                output_state=ValidationState.RAW
            ))
        
        return EnhancedResourceObject(
            name=f"{normalized_resource} Input",
            type=ResourceType.INPUT_BOUNDARY,
            resource_name=resource_name,
            functions=functions,
            current_state=ValidationState.RAW
        )

    def create_enhanced_manager_controller(self, resource_name: str, domain_config: DomainConfig, validation_controller: ValidationController) -> EnhancedResourceObject:
        """
        Create enhanced manager controller that properly handles validation states
        """
        normalized_resource = resource_name.replace(' ', '').title()
        
        functions = [
            EnhancedResourceFunction(
                name=f"receive_validated_{resource_name.replace(' ', '_')}",
                source="validation",
                description=f"Receive validated {resource_name} from validation controller",
                input_state=ValidationState.VALIDATED,
                output_state=ValidationState.VALIDATED
            ),
            EnhancedResourceFunction(
                name=f"store_{resource_name.replace(' ', '_')}",
                source="implicit",
                description=f"Store validated {resource_name} safely",
                input_state=ValidationState.VALIDATED,
                output_state=ValidationState.VALIDATED
            ),
            EnhancedResourceFunction(
                name=f"monitor_{resource_name.replace(' ', '_')}_level",
                source="implicit",
                description=f"Monitor available quantity of validated {resource_name}",
                input_state=ValidationState.VALIDATED,
                output_state=ValidationState.VALIDATED
            ),
            EnhancedResourceFunction(
                name=f"provide_{resource_name.replace(' ', '_')}",
                source="implicit",
                description=f"Provide validated {resource_name} for processing",
                input_state=ValidationState.VALIDATED,
                output_state=ValidationState.PROCESSED
            )
        ]
        
        # Add domain-specific processing functions
        domain = domain_config.domain_name if domain_config else "generic"
        if domain == "beverage_preparation":
            if "coffee" in resource_name.lower():
                functions.append(EnhancedResourceFunction(
                    name="grind_coffee",
                    source="context",
                    description="Grind validated coffee beans to required consistency",
                    input_state=ValidationState.VALIDATED,
                    output_state=ValidationState.PROCESSED
                ))
            elif "water" in resource_name.lower():
                functions.append(EnhancedResourceFunction(
                    name="heat_water",
                    source="context",
                    description="Heat validated water to brewing temperature",
                    input_state=ValidationState.VALIDATED,
                    output_state=ValidationState.PROCESSED
                ))
            elif "milk" in resource_name.lower():
                functions.append(EnhancedResourceFunction(
                    name="steam_milk",
                    source="context",
                    description="Steam validated milk for coffee drinks",
                    input_state=ValidationState.VALIDATED,
                    output_state=ValidationState.PROCESSED
                ))
        
        return EnhancedResourceObject(
            name=f"{normalized_resource}Manager",
            type=ResourceType.MANAGER_CONTROLLER,
            resource_name=resource_name,
            functions=functions,
            validation_controller=validation_controller,
            domain_specific=True,
            current_state=ValidationState.VALIDATED
        )

    def analyze_enhanced_resource(self, resource_name: str, domain_config: DomainConfig, phase1_result: Phase1Result) -> EnhancedBetriebsmittelAnalysis:
        """
        Enhanced resource analysis with proper validation flow
        """
        domain = domain_config.domain_name if domain_config else "generic"
        
        # 1. Create validation controller
        validation_controller = self.create_validation_controller(resource_name, domain)
        
        # 2. Create input boundary
        input_boundary = self.create_enhanced_input_boundary(resource_name, domain_config, phase1_result)
        
        # 3. Create validation boundary
        validation_boundary = EnhancedResourceObject(
            name=f"{resource_name.replace(' ', '').title()} Validation",
            type=ResourceType.VALIDATION_BOUNDARY,
            resource_name=resource_name,
            functions=validation_controller.validation_functions,
            validation_controller=validation_controller,
            current_state=ValidationState.RAW
        )
        
        # 4. Create enhanced manager controller
        manager_controller = self.create_enhanced_manager_controller(resource_name, domain_config, validation_controller)
        
        # 5. Create output boundary if needed
        output_boundary = None
        if any(waste_term in resource_name.lower() for waste_term in ["coffee", "bean"]):
            output_boundary = EnhancedResourceObject(
                name=f"{resource_name.replace(' ', '').title()} Waste Output",
                type=ResourceType.OUTPUT_BOUNDARY,
                resource_name=resource_name,
                functions=[
                    EnhancedResourceFunction(
                        name="dispose_waste",
                        source="implicit",
                        description=f"Dispose of waste products from {resource_name}",
                        input_state=ValidationState.PROCESSED,
                        output_state=ValidationState.WASTE
                    )
                ],
                current_state=ValidationState.WASTE
            )
        
        # 6. Create enhanced transformations with state information
        transformations = []
        validation_flow = [
            f"{resource_name}: RAW -> VALIDATED (validation)",
            f"{resource_name}: VALIDATED -> PROCESSED (processing)",
        ]
        
        if domain == "beverage_preparation":
            if "coffee" in resource_name.lower():
                transformations.append("validated beans -> grinding -> processed ground coffee")
                validation_flow.append(f"{resource_name}: grinding transformation (VALIDATED -> PROCESSED)")
            elif "water" in resource_name.lower():
                transformations.append("validated water -> heating -> processed hot water")
                validation_flow.append(f"{resource_name}: heating transformation (VALIDATED -> PROCESSED)")
            elif "milk" in resource_name.lower():
                transformations.append("validated milk -> steaming -> processed steamed milk")
                validation_flow.append(f"{resource_name}: steaming transformation (VALIDATED -> PROCESSED)")
        
        return EnhancedBetriebsmittelAnalysis(
            resource_name=resource_name,
            input_boundary=input_boundary,
            validation_boundary=validation_boundary,
            manager_controller=manager_controller,
            output_boundary=output_boundary,
            transformations=transformations,
            validation_flow=validation_flow
        )

    def _derive_context_resources(self, phase1_result: Phase1Result, existing_resources: List[str]) -> List[str]:
        """
        Derive implicit resources from domain context (same as before)
        """
        context_resources = []
        domain = phase1_result.capability_context.domain
        
        if domain == "beverage_preparation":
            has_beverage_ingredients = any(
                beverage_term in " ".join(existing_resources).lower() 
                for beverage_term in ["coffee", "tea", "beans", "leaves", "milk", "water"]
            )
            
            has_container = any(
                container_term in " ".join(existing_resources).lower()
                for container_term in ["cup", "tasse", "mug", "glass", "container"]
            )
            
            if has_beverage_ingredients and not has_container:
                context_resources.append("cup")
                print(f"CONTEXT RULE APPLIED: Beverages need containers -> derived 'cup'")
        
        return context_resources

    def perform_enhanced_phase2_analysis(self, phase1_result: Phase1Result, preconditions: List[str]) -> EnhancedPhase2Result:
        """
        Enhanced Phase 2 analysis with validation integration
        """
        # Extract resources from preconditions
        from phase2_betriebsmittel_analyzer import BetriebsmittelAnalyzer
        basic_analyzer = BetriebsmittelAnalyzer()
        resources = basic_analyzer.extract_resources_from_preconditions(preconditions)
        
        # Derive context-based resources
        context_resources = self._derive_context_resources(phase1_result, resources)
        all_resources = resources + context_resources
        
        # Enhanced analysis for each resource
        resource_analyses = []
        validation_controllers = []
        all_objects = {
            "input_boundaries": [],
            "validation_boundaries": [],
            "manager_controllers": [], 
            "output_boundaries": [],
            "entities": []
        }
        validation_flow_map = {}
        
        for resource in all_resources:
            analysis = self.analyze_enhanced_resource(
                resource, 
                phase1_result.capability_context.domain_config,
                phase1_result
            )
            resource_analyses.append(analysis)
            
            # Collect validation controllers
            if analysis.validation_boundary and analysis.validation_boundary.validation_controller:
                validation_controllers.append(analysis.validation_boundary.validation_controller)
            
            # Collect all objects
            if analysis.input_boundary:
                all_objects["input_boundaries"].append(analysis.input_boundary)
            if analysis.validation_boundary:
                all_objects["validation_boundaries"].append(analysis.validation_boundary)
            if analysis.manager_controller:
                all_objects["manager_controllers"].append(analysis.manager_controller)
            if analysis.output_boundary:
                all_objects["output_boundaries"].append(analysis.output_boundary)
            
            # Map validation flow
            validation_flow_map[resource] = [
                ValidationState.RAW,
                ValidationState.VALIDATED,
                ValidationState.PROCESSED
            ]
        
        # Generate enhanced summary
        summary = self._generate_enhanced_summary(resource_analyses, validation_controllers, phase1_result)
        
        return EnhancedPhase2Result(
            phase1_result=phase1_result,
            resource_analyses=resource_analyses,
            validation_controllers=validation_controllers,
            all_objects=all_objects,
            validation_flow_map=validation_flow_map,
            summary=summary
        )

    def _generate_enhanced_summary(self, analyses: List[EnhancedBetriebsmittelAnalysis], validators: List[ValidationController], phase1_result: Phase1Result) -> str:
        """Generate enhanced summary"""
        total_resources = len(analyses)
        total_validators = len(validators)
        total_controllers = sum(1 for a in analyses if a.manager_controller)
        total_boundaries = sum(1 for a in analyses if a.input_boundary or a.output_boundary)
        total_validations = sum(1 for a in analyses if a.validation_boundary)
        
        summary_parts = [
            f"Enhanced analysis of {total_resources} resources with validation",
            f"Created {total_validators} validation controllers",
            f"Generated {total_validations} validation boundaries", 
            f"Established {total_controllers} manager controllers",
            f"Identified {total_boundaries} boundary objects",
            f"Domain: {phase1_result.capability_context.domain}",
            "Applied validation state transitions: RAW->VALIDATED->PROCESSED"
        ]
        
        return "; ".join(summary_parts)


def main():
    """Test enhanced Phase 2 analysis"""
    
    # Mock Phase 1 result for testing
    from dataclasses import dataclass
    from typing import List
    
    @dataclass
    class MockActor:
        type: object
        
    @dataclass
    class MockActorType:
        value: str = "human"
    
    @dataclass
    class MockCapabilityContext:
        domain: str = "beverage_preparation"
        domain_config: object = None
    
    @dataclass
    class MockPhase1Result:
        capability_context: MockCapabilityContext = field(default_factory=MockCapabilityContext)
        actors: List = field(default_factory=list)
        
        def __post_init__(self):
            if not self.actors:
                actor_type = MockActorType()
                actor = MockActor(type=actor_type)
                self.actors = [actor]
    
    # Initialize enhanced analyzer
    enhanced_analyzer = EnhancedBetriebsmittelAnalyzer()
    
    # Mock phase 1 result
    phase1_result = MockPhase1Result()
    
    # Test preconditions
    preconditions = [
        'Coffee beans are available in the system',
        'Water is available in the system',
        'Milk is available in the system'
    ]
    
    # Perform enhanced analysis
    enhanced_result = enhanced_analyzer.perform_enhanced_phase2_analysis(phase1_result, preconditions)
    
    # Save result
    output_file = "Zwischenprodukte/UC1_Coffee_enhanced_phase2_analysis.json"
    
    # Convert to serializable format
    serializable_result = {
        "summary": enhanced_result.summary,
        "validation_controllers": [
            {
                "name": vc.name,
                "resource_name": vc.resource_name,
                "quality_checks": vc.quality_checks,
                "quantity_checks": vc.quantity_checks,
                "safety_checks": vc.safety_checks,
                "validation_functions": [
                    {
                        "name": vf.name,
                        "description": vf.description,
                        "input_state": vf.input_state.value if vf.input_state else None,
                        "output_state": vf.output_state.value if vf.output_state else None
                    } for vf in vc.validation_functions
                ]
            } for vc in enhanced_result.validation_controllers
        ],
        "resource_analyses": [
            {
                "resource_name": ra.resource_name,
                "has_validation_boundary": ra.validation_boundary is not None,
                "validation_flow": ra.validation_flow,
                "transformations": ra.transformations
            } for ra in enhanced_result.resource_analyses
        ],
        "validation_flow_map": {
            resource: [state.value for state in states]
            for resource, states in enhanced_result.validation_flow_map.items()
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)
    
    print(f"Enhanced Phase 2 Analysis completed!")
    print(f"File saved to: {output_file}")
    print(f"Summary: {enhanced_result.summary}")
    print(f"Validation Controllers: {len(enhanced_result.validation_controllers)}")
    print(f"Resources with validation: {len([ra for ra in enhanced_result.resource_analyses if ra.validation_boundary])}")
    
    # Show validation flow
    print("\nValidation Flow Map:")
    for resource, states in enhanced_result.validation_flow_map.items():
        print(f"  {resource}: {' -> '.join([state.value for state in states])}")
    
    return enhanced_result

if __name__ == "__main__":
    main()