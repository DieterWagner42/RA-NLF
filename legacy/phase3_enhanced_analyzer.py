"""
Enhanced Phase 3: Interaktions-Analyse with Validation Integration
Fixes Phase 5 violations by modeling validation interactions explicitly
Integrates with Enhanced Phase 2 validation controllers
"""

import spacy
import json
import os
import re
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from domain_config_analyzer import DomainConfigurableAnalyzer, DomainConfig, Phase1Result
from phase2_enhanced_analyzer import EnhancedPhase2Result, ValidationState, ValidationController


class InteractionType(Enum):
    VALIDATION = "validation"
    COORDINATION = "coordination"
    TRANSFORMATION = "transformation"
    USER_COMMUNICATION = "user_communication"
    ERROR_HANDLING = "error_handling"


@dataclass
class ValidationInteraction:
    interaction_id: str
    validator_controller: str
    resource_name: str
    input_state: ValidationState
    output_state: ValidationState
    validation_steps: List[str]
    failure_handling: List[str]


@dataclass
class EnhancedUCStep:
    step_id: str
    description: str
    step_type: str  # "basic", "alternative", "extension"
    responsible_controller: str
    validation_requirements: List[ValidationInteraction] = field(default_factory=list)
    data_state_requirements: List[ValidationState] = field(default_factory=list)
    coordination_dependencies: List[str] = field(default_factory=list)


@dataclass
class EnhancedInteraction:
    interaction_id: str
    interaction_type: InteractionType
    source_controller: str
    target_controller: str
    data_exchanged: str
    validation_state: Optional[ValidationState] = None
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)


@dataclass
class ValidationOrchestrationPattern:
    pattern_id: str
    coordinator_controller: str
    validation_sequence: List[ValidationInteraction]
    processing_sequence: List[str]
    error_handling_sequence: List[str]
    state_transition_map: Dict[str, List[ValidationState]]


@dataclass
class EnhancedPhase3Result:
    enhanced_phase2_result: EnhancedPhase2Result
    uc_steps: List[EnhancedUCStep]
    validation_interactions: List[ValidationInteraction]
    enhanced_interactions: List[EnhancedInteraction]
    validation_orchestration_patterns: List[ValidationOrchestrationPattern]
    all_objects: Dict[str, List[str]]
    violations: List[str]
    summary: str


class EnhancedInteraktionAnalyzer:
    """
    Enhanced Phase 3 implementation with explicit validation modeling
    """
    
    def __init__(self, domain_analyzer: DomainConfigurableAnalyzer = None):
        self.domain_analyzer = domain_analyzer or DomainConfigurableAnalyzer()
        self.nlp = self.domain_analyzer.nlp
        
        # Enhanced interaction patterns
        self.validation_patterns = {
            "quality_validation": ["validate", "check", "verify", "inspect", "test"],
            "quantity_validation": ["measure", "count", "weigh", "assess amount"],
            "state_transition": ["accept", "approve", "confirm", "pass", "ready"]
        }
        
        # Coordination patterns with validation
        self.coordination_patterns = {
            "validation_coordination": "Coordinator ensures all resources validated before processing",
            "processing_coordination": "Coordinator sequences processing after validation",
            "error_coordination": "Coordinator handles validation failures and errors"
        }

    def create_validation_interactions(self, validation_controllers: List[ValidationController]) -> List[ValidationInteraction]:
        """
        Create explicit validation interactions for each validation controller
        """
        validation_interactions = []
        
        for i, validator in enumerate(validation_controllers, 1):
            # Main validation interaction
            validation_interaction = ValidationInteraction(
                interaction_id=f"VI{i}",
                validator_controller=validator.name,
                resource_name=validator.resource_name,
                input_state=ValidationState.RAW,
                output_state=ValidationState.VALIDATED,
                validation_steps=[
                    f"Receive {validator.resource_name} in RAW state",
                    f"Perform quality checks: {', '.join(validator.quality_checks)}",
                    f"Perform quantity checks: {', '.join(validator.quantity_checks)}",
                    f"Perform safety checks: {', '.join(validator.safety_checks)}",
                    f"Confirm {validator.resource_name} passes all validations",
                    f"Transition {validator.resource_name} to VALIDATED state",
                    f"Forward validated {validator.resource_name} to manager"
                ],
                failure_handling=[
                    f"If validation fails, reject {validator.resource_name}",
                    f"Log validation failure for {validator.resource_name}",
                    f"Notify coordinator of validation failure",
                    f"Request replacement {validator.resource_name} if available"
                ]
            )
            validation_interactions.append(validation_interaction)
        
        return validation_interactions

    def parse_enhanced_uc_steps(self, uc_text: str, validation_interactions: List[ValidationInteraction]) -> List[EnhancedUCStep]:
        """
        Parse UC steps with validation requirements
        """
        enhanced_steps = []
        lines = uc_text.split('\n')
        
        # Basic step pattern matching
        basic_pattern = r"B(\d+[a-z]?)\s+(.*)"
        alt_pattern = r"A(\d+)\.?(\d+)?\s+at\s+B(\d+[a-z]?)\s+(.*)"
        ext_pattern = r"E(\d+)\.?(\d+)?\s+B(\d+[a-z]?)-?B?(\d+[a-z]?)?\s+\(trigger\)\s+(.*)"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse basic steps
            basic_match = re.search(basic_pattern, line)
            if basic_match:
                step_id = f"B{basic_match.group(1)}"
                description = basic_match.group(2)
                
                # Determine responsible controller and validation requirements
                controller, validation_reqs, state_reqs = self._analyze_step_requirements(
                    step_id, description, validation_interactions
                )
                
                enhanced_step = EnhancedUCStep(
                    step_id=step_id,
                    description=description,
                    step_type="basic",
                    responsible_controller=controller,
                    validation_requirements=validation_reqs,
                    data_state_requirements=state_reqs
                )
                enhanced_steps.append(enhanced_step)
                continue
            
            # Parse alternative steps
            alt_match = re.search(alt_pattern, line)
            if alt_match:
                step_id = f"A{alt_match.group(1)}"
                if alt_match.group(2):
                    step_id += f".{alt_match.group(2)}"
                description = alt_match.group(4)
                
                controller, validation_reqs, state_reqs = self._analyze_step_requirements(
                    step_id, description, validation_interactions
                )
                
                enhanced_step = EnhancedUCStep(
                    step_id=step_id,
                    description=description,
                    step_type="alternative",
                    responsible_controller=controller,
                    validation_requirements=validation_reqs,
                    data_state_requirements=state_reqs
                )
                enhanced_steps.append(enhanced_step)
        
        return enhanced_steps

    def _analyze_step_requirements(self, step_id: str, description: str, validation_interactions: List[ValidationInteraction]) -> Tuple[str, List[ValidationInteraction], List[ValidationState]]:
        """
        Analyze step requirements for controller, validation, and state
        """
        desc_lower = description.lower()
        
        # Determine responsible controller
        controller = "GeträenkeOrchestrator"  # Default coordinator
        validation_reqs = []
        state_reqs = []
        
        # Map specific activities to controllers and requirements
        if "activate" in desc_lower or "heat" in desc_lower:
            if "water" in desc_lower:
                controller = "WaterManager"
                # Require validated water
                water_validation = [vi for vi in validation_interactions if "water" in vi.resource_name.lower()]
                validation_reqs.extend(water_validation)
                state_reqs.append(ValidationState.VALIDATED)
        
        elif "prepare" in desc_lower and "filter" in desc_lower:
            controller = "FilterManager"
            # No specific validation needed for filter preparation
            
        elif "grind" in desc_lower:
            controller = "CoffeeBeansManager"
            # Require validated coffee beans
            coffee_validation = [vi for vi in validation_interactions if "coffee" in vi.resource_name.lower()]
            validation_reqs.extend(coffee_validation)
            state_reqs.append(ValidationState.VALIDATED)
            
        elif "retrieve" in desc_lower and "cup" in desc_lower:
            controller = "CupManager"
            # Require validated cup
            cup_validation = [vi for vi in validation_interactions if "cup" in vi.resource_name.lower()]
            validation_reqs.extend(cup_validation)
            state_reqs.append(ValidationState.VALIDATED)
            
        elif "brewing" in desc_lower or "begin" in desc_lower:
            controller = "BrewingManager"
            # Require validated ground coffee and hot water
            state_reqs.extend([ValidationState.PROCESSED, ValidationState.PROCESSED])  # Ground coffee + hot water
            
        elif "add" in desc_lower and "milk" in desc_lower:
            controller = "MilkManager"
            # Require validated milk
            milk_validation = [vi for vi in validation_interactions if "milk" in vi.resource_name.lower()]
            validation_reqs.extend(milk_validation)
            state_reqs.append(ValidationState.VALIDATED)
            
        elif "output" in desc_lower or "message" in desc_lower or "present" in desc_lower:
            controller = "UserInterfaceManager"
            # Require ready product
            state_reqs.append(ValidationState.READY)
        
        return controller, validation_reqs, state_reqs

    def create_enhanced_interactions(self, enhanced_steps: List[EnhancedUCStep], validation_interactions: List[ValidationInteraction]) -> List[EnhancedInteraction]:
        """
        Create enhanced interactions including validation flows
        """
        enhanced_interactions = []
        interaction_counter = 1
        
        # 1. Validation interactions
        for val_interaction in validation_interactions:
            enhanced_interaction = EnhancedInteraction(
                interaction_id=f"EI{interaction_counter}",
                interaction_type=InteractionType.VALIDATION,
                source_controller=f"{val_interaction.resource_name.replace(' ', '')}Input",
                target_controller=val_interaction.validator_controller,
                data_exchanged=f"{val_interaction.resource_name} (RAW)",
                validation_state=ValidationState.RAW,
                preconditions=[f"{val_interaction.resource_name} received from external source"],
                postconditions=[f"{val_interaction.resource_name} validated or rejected"]
            )
            enhanced_interactions.append(enhanced_interaction)
            interaction_counter += 1
            
            # Validation to Manager interaction
            manager_name = f"{val_interaction.resource_name.replace(' ', '')}Manager"
            enhanced_interaction = EnhancedInteraction(
                interaction_id=f"EI{interaction_counter}",
                interaction_type=InteractionType.COORDINATION,
                source_controller=val_interaction.validator_controller,
                target_controller=manager_name,
                data_exchanged=f"{val_interaction.resource_name} (VALIDATED)",
                validation_state=ValidationState.VALIDATED,
                preconditions=[f"{val_interaction.resource_name} passes all validation checks"],
                postconditions=[f"{val_interaction.resource_name} ready for processing"]
            )
            enhanced_interactions.append(enhanced_interaction)
            interaction_counter += 1
        
        # 2. Coordination interactions
        coordinator_interaction = EnhancedInteraction(
            interaction_id=f"EI{interaction_counter}",
            interaction_type=InteractionType.COORDINATION,
            source_controller="ZeitManager",
            target_controller="GeträenkeOrchestrator",
            data_exchanged="Time trigger event",
            validation_state=None,
            preconditions=["System clock reaches 7:00h"],
            postconditions=["Coffee preparation process initiated"]
        )
        enhanced_interactions.append(coordinator_interaction)
        interaction_counter += 1
        
        # 3. Processing interactions (Manager to Manager)
        processing_interaction = EnhancedInteraction(
            interaction_id=f"EI{interaction_counter}",
            interaction_type=InteractionType.TRANSFORMATION,
            source_controller="GeträenkeOrchestrator",
            target_controller="BrewingManager",
            data_exchanged="Validated ingredients for brewing",
            validation_state=ValidationState.PROCESSED,
            preconditions=["All ingredients validated and processed"],
            postconditions=["Milk coffee ready for user"]
        )
        enhanced_interactions.append(processing_interaction)
        interaction_counter += 1
        
        # 4. User communication
        user_interaction = EnhancedInteraction(
            interaction_id=f"EI{interaction_counter}",
            interaction_type=InteractionType.USER_COMMUNICATION,
            source_controller="GeträenkeOrchestrator",
            target_controller="UserInterfaceManager",
            data_exchanged="Ready milk coffee",
            validation_state=ValidationState.READY,
            preconditions=["Milk coffee preparation completed"],
            postconditions=["User receives coffee"]
        )
        enhanced_interactions.append(user_interaction)
        
        return enhanced_interactions

    def create_validation_orchestration_pattern(self, validation_interactions: List[ValidationInteraction], enhanced_steps: List[EnhancedUCStep]) -> ValidationOrchestrationPattern:
        """
        Create comprehensive validation orchestration pattern
        """
        # Build state transition map
        state_transition_map = {}
        for val_interaction in validation_interactions:
            state_transition_map[val_interaction.resource_name] = [
                ValidationState.RAW,
                ValidationState.VALIDATED,
                ValidationState.PROCESSED,
                ValidationState.READY
            ]
        
        # Build processing sequence
        processing_sequence = [
            "1. Validate all input resources (RAW -> VALIDATED)",
            "2. Process validated resources (VALIDATED -> PROCESSED)", 
            "3. Aggregate processed ingredients (PROCESSED -> READY)",
            "4. Deliver ready product to user (READY -> CONSUMED)"
        ]
        
        # Build error handling sequence
        error_handling_sequence = [
            "1. Detect validation failure",
            "2. Log failure details",
            "3. Notify coordinator",
            "4. Request replacement resource if available",
            "5. Abort process if critical resource unavailable"
        ]
        
        return ValidationOrchestrationPattern(
            pattern_id="ValidationOrchestration_UC1",
            coordinator_controller="GeträenkeOrchestrator",
            validation_sequence=validation_interactions,
            processing_sequence=processing_sequence,
            error_handling_sequence=error_handling_sequence,
            state_transition_map=state_transition_map
        )

    def validate_enhanced_interactions(self, enhanced_interactions: List[EnhancedInteraction], validation_orchestration: ValidationOrchestrationPattern) -> List[str]:
        """
        Validate that interactions follow proper validation patterns
        """
        violations = []
        
        # Check validation state consistency
        for interaction in enhanced_interactions:
            if interaction.interaction_type == InteractionType.VALIDATION:
                if interaction.validation_state != ValidationState.RAW:
                    violations.append(f"Validation interaction {interaction.interaction_id} should start with RAW state")
            
            elif interaction.interaction_type == InteractionType.TRANSFORMATION:
                if interaction.validation_state not in [ValidationState.VALIDATED, ValidationState.PROCESSED]:
                    violations.append(f"Transformation interaction {interaction.interaction_id} requires VALIDATED or PROCESSED state")
        
        # Check coordinator presence
        coordinator_interactions = [i for i in enhanced_interactions if "GeträenkeOrchestrator" in [i.source_controller, i.target_controller]]
        if not coordinator_interactions:
            violations.append("No coordinator interactions found - violates coordination rule")
        
        # Check validation completeness
        validation_interactions = [i for i in enhanced_interactions if i.interaction_type == InteractionType.VALIDATION]
        expected_validations = len(validation_orchestration.validation_sequence)
        if len(validation_interactions) < expected_validations:
            violations.append(f"Incomplete validation interactions: {len(validation_interactions)} found, {expected_validations} expected")
        
        return violations

    def perform_enhanced_phase3_analysis(self, enhanced_phase2_result: EnhancedPhase2Result, uc_text: str) -> EnhancedPhase3Result:
        """
        Complete enhanced Phase 3 analysis with validation integration
        """
        # Create validation interactions from Phase 2 validation controllers
        validation_interactions = self.create_validation_interactions(enhanced_phase2_result.validation_controllers)
        
        # Parse UC steps with validation requirements
        enhanced_steps = self.parse_enhanced_uc_steps(uc_text, validation_interactions)
        
        # Create enhanced interactions
        enhanced_interactions = self.create_enhanced_interactions(enhanced_steps, validation_interactions)
        
        # Create validation orchestration pattern
        validation_orchestration = self.create_validation_orchestration_pattern(validation_interactions, enhanced_steps)
        
        # Collect all objects
        all_objects = {
            "controllers": [],
            "boundaries": [],
            "entities": [],
            "validators": []
        }
        
        # Add controllers from Phase 2
        for analysis in enhanced_phase2_result.resource_analyses:
            if analysis.manager_controller:
                all_objects["controllers"].append(analysis.manager_controller.name)
            if analysis.input_boundary:
                all_objects["boundaries"].append(analysis.input_boundary.name)
            if analysis.validation_boundary:
                all_objects["boundaries"].append(analysis.validation_boundary.name)
                all_objects["validators"].append(analysis.validation_boundary.validation_controller.name)
        
        # Add coordinator
        all_objects["controllers"].append("GeträenkeOrchestrator")
        all_objects["controllers"].append("ZeitManager")
        all_objects["controllers"].append("UserInterfaceManager")
        
        # Validate interactions
        violations = self.validate_enhanced_interactions(enhanced_interactions, validation_orchestration)
        
        # Generate summary
        summary = self._generate_enhanced_summary(enhanced_steps, validation_interactions, enhanced_interactions, violations)
        
        return EnhancedPhase3Result(
            enhanced_phase2_result=enhanced_phase2_result,
            uc_steps=enhanced_steps,
            validation_interactions=validation_interactions,
            enhanced_interactions=enhanced_interactions,
            validation_orchestration_patterns=[validation_orchestration],
            all_objects=all_objects,
            violations=violations,
            summary=summary
        )

    def _generate_enhanced_summary(self, steps: List[EnhancedUCStep], validations: List[ValidationInteraction], interactions: List[EnhancedInteraction], violations: List[str]) -> str:
        """Generate enhanced summary"""
        total_steps = len(steps)
        total_validations = len(validations)
        total_interactions = len(interactions)
        total_violations = len(violations)
        
        validation_steps = len([s for s in steps if s.validation_requirements])
        
        summary_parts = [
            f"Enhanced analysis of {total_steps} UC steps with validation integration",
            f"Created {total_validations} explicit validation interactions",
            f"Generated {total_interactions} enhanced interactions",
            f"Identified {validation_steps} steps requiring validation",
            f"Detected {total_violations} interaction violations",
            "Applied validation state management and coordination rules"
        ]
        
        return "; ".join(summary_parts)


def main():
    """Test enhanced Phase 3 analysis"""
    
    # Load UC text
    with open('Use Case/UC1.txt', 'r', encoding='utf-8') as f:
        uc1_text = f.read()
    
    # Load enhanced Phase 2 result
    with open('Zwischenprodukte/UC1_Coffee_enhanced_phase2_analysis.json', 'r', encoding='utf-8') as f:
        phase2_data = json.load(f)
    
    # Mock enhanced Phase 2 result for testing
    from phase2_enhanced_analyzer import EnhancedBetriebsmittelAnalyzer
    
    # Create mock Phase 2 result (simplified)
    from dataclasses import dataclass
    
    @dataclass
    class MockValidationController:
        name: str
        resource_name: str
        quality_checks: List[str]
        quantity_checks: List[str] 
        safety_checks: List[str]
    
    # Create validation controllers from JSON data
    validation_controllers = []
    for vc_data in phase2_data["validation_controllers"]:
        vc = MockValidationController(
            name=vc_data["name"],
            resource_name=vc_data["resource_name"],
            quality_checks=vc_data["quality_checks"],
            quantity_checks=vc_data["quantity_checks"],
            safety_checks=vc_data["safety_checks"]
        )
        validation_controllers.append(vc)
    
    @dataclass
    class MockEnhancedPhase2Result:
        validation_controllers: List[MockValidationController]
        resource_analyses: List = field(default_factory=list)
    
    enhanced_phase2_result = MockEnhancedPhase2Result(validation_controllers=validation_controllers)
    
    # Initialize enhanced analyzer
    enhanced_analyzer = EnhancedInteraktionAnalyzer()
    
    # Perform enhanced Phase 3 analysis
    enhanced_result = enhanced_analyzer.perform_enhanced_phase3_analysis(enhanced_phase2_result, uc1_text)
    
    # Save result
    output_file = "Zwischenprodukte/UC1_Coffee_enhanced_phase3_analysis.json"
    
    serializable_result = {
        "summary": enhanced_result.summary,
        "uc_steps": [
            {
                "step_id": step.step_id,
                "description": step.description,
                "responsible_controller": step.responsible_controller,
                "validation_requirements": len(step.validation_requirements),
                "data_state_requirements": [state.value for state in step.data_state_requirements]
            } for step in enhanced_result.uc_steps
        ],
        "validation_interactions": [
            {
                "interaction_id": vi.interaction_id,
                "validator_controller": vi.validator_controller,
                "resource_name": vi.resource_name,
                "input_state": vi.input_state.value,
                "output_state": vi.output_state.value,
                "validation_steps": vi.validation_steps
            } for vi in enhanced_result.validation_interactions
        ],
        "enhanced_interactions": [
            {
                "interaction_id": ei.interaction_id,
                "interaction_type": ei.interaction_type.value,
                "source_controller": ei.source_controller,
                "target_controller": ei.target_controller,
                "data_exchanged": ei.data_exchanged,
                "validation_state": ei.validation_state.value if ei.validation_state else None
            } for ei in enhanced_result.enhanced_interactions
        ],
        "violations": enhanced_result.violations,
        "all_objects": enhanced_result.all_objects
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)
    
    print(f"Enhanced Phase 3 Analysis completed!")
    print(f"File saved to: {output_file}")
    print(f"Summary: {enhanced_result.summary}")
    print(f"Validation Interactions: {len(enhanced_result.validation_interactions)}")
    print(f"Enhanced Interactions: {len(enhanced_result.enhanced_interactions)}")
    print(f"Violations: {len(enhanced_result.violations)}")
    
    if enhanced_result.violations:
        print("Violations found:")
        for violation in enhanced_result.violations:
            print(f"  - {violation}")
    else:
        print("No violations found!")
    
    return enhanced_result

if __name__ == "__main__":
    main()