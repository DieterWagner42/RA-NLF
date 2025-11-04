"""
Enhanced Phase 4: Kontrollfluss-Analyse with Validation Integration
Fixes Phase 5 violations by integrating validation control flow
Models validation sequences, dependencies, and state transitions
"""

import spacy
import json
import os
import re
from typing import Dict, List, Set, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from domain_config_analyzer import DomainConfigurableAnalyzer, DomainConfig, Phase1Result
from phase2_enhanced_analyzer import EnhancedPhase2Result, ValidationState
from phase3_enhanced_analyzer import EnhancedPhase3Result, ValidationInteraction


class EnhancedControlFlowType(Enum):
    VALIDATION_SEQUENTIAL = "validation_sequential"
    VALIDATION_PARALLEL = "validation_parallel"
    PROCESSING_SEQUENTIAL = "processing_sequential"
    PROCESSING_PARALLEL = "processing_parallel"
    CONDITIONAL = "conditional"
    ERROR_HANDLING = "error_handling"


class ValidationFlowType(Enum):
    MANDATORY_VALIDATION = "mandatory_validation"
    OPTIONAL_VALIDATION = "optional_validation"
    BATCH_VALIDATION = "batch_validation"
    REAL_TIME_VALIDATION = "real_time_validation"


@dataclass
class ValidationControlStep:
    step_id: str
    description: str
    validation_state_input: ValidationState
    validation_state_output: ValidationState
    responsible_controller: str
    validation_dependencies: List[str] = field(default_factory=list)
    parallel_validations: List[str] = field(default_factory=list)
    failure_handling: List[str] = field(default_factory=list)


@dataclass
class EnhancedControlFlowStep:
    step_id: str
    step_type: str
    description: str
    validation_control_steps: List[ValidationControlStep] = field(default_factory=list)
    responsible_controller: str = ""
    flow_type: EnhancedControlFlowType = EnhancedControlFlowType.PROCESSING_SEQUENTIAL
    validation_flow_type: ValidationFlowType = ValidationFlowType.MANDATORY_VALIDATION
    state_preconditions: List[ValidationState] = field(default_factory=list)
    state_postconditions: List[ValidationState] = field(default_factory=list)
    parallel_with: List[str] = field(default_factory=list)


@dataclass
class ValidationFlowPattern:
    pattern_id: str
    pattern_type: str
    validation_sequence: List[ValidationControlStep]
    processing_sequence: List[str]
    state_transition_map: Dict[str, List[ValidationState]]
    coordination_rules: List[str]
    error_recovery: List[str]


@dataclass
class EnhancedControlFlowPattern:
    pattern_id: str
    pattern_type: str
    enhanced_steps: List[EnhancedControlFlowStep]
    validation_flow_patterns: List[ValidationFlowPattern]
    coordination_sequence: List[str]
    error_handling: List[str] = field(default_factory=list)


@dataclass
class EnhancedPhase4Result:
    enhanced_phase3_result: EnhancedPhase3Result
    enhanced_control_flow_patterns: List[EnhancedControlFlowPattern]
    validation_flow_patterns: List[ValidationFlowPattern]
    coordination_rules: Dict[str, List[str]]
    flow_violations: List[str]
    summary: str


class EnhancedKontrollflussAnalyzer:
    """
    Enhanced Phase 4 implementation with validation control flow integration
    """
    
    def __init__(self, domain_analyzer: DomainConfigurableAnalyzer = None):
        self.domain_analyzer = domain_analyzer or DomainConfigurableAnalyzer()
        self.nlp = self.domain_analyzer.nlp
        
        # Enhanced control flow rules with validation
        self.enhanced_control_flow_rules = {
            "validation_before_processing": "All resources must be validated before processing",
            "validation_state_consistency": "State transitions must follow RAW->VALIDATED->PROCESSED->READY",
            "validation_parallelization": "Independent resource validations can be parallel",
            "validation_dependency_sequencing": "Dependent validations must be sequential",
            "coordinator_validation_orchestration": "Coordinator orchestrates validation completion before processing",
            "validation_error_propagation": "Validation errors must propagate through coordination hierarchy"
        }
        
        # Validation flow patterns
        self.validation_flow_patterns = {
            "ingredient_validation": ["accept", "validate_quality", "validate_quantity", "confirm", "forward"],
            "container_validation": ["accept", "validate_cleanliness", "validate_availability", "confirm", "forward"],
            "batch_validation": ["collect_all", "validate_batch", "confirm_batch", "release_batch"]
        }

    def create_validation_control_steps(self, validation_interactions: List[ValidationInteraction]) -> List[ValidationControlStep]:
        """
        Create validation control steps from validation interactions
        """
        validation_control_steps = []
        
        for i, val_interaction in enumerate(validation_interactions, 1):
            # Step 1: Accept input
            accept_step = ValidationControlStep(
                step_id=f"V{i}.1",
                description=f"Accept {val_interaction.resource_name} from input boundary",
                validation_state_input=ValidationState.RAW,
                validation_state_output=ValidationState.RAW,
                responsible_controller=val_interaction.validator_controller,
                validation_dependencies=[],
                failure_handling=[f"Reject invalid {val_interaction.resource_name}"]
            )
            validation_control_steps.append(accept_step)
            
            # Step 2: Perform validations
            validate_step = ValidationControlStep(
                step_id=f"V{i}.2",
                description=f"Perform all validation checks on {val_interaction.resource_name}",
                validation_state_input=ValidationState.RAW,
                validation_state_output=ValidationState.VALIDATED,
                responsible_controller=val_interaction.validator_controller,
                validation_dependencies=[f"V{i}.1"],
                failure_handling=[
                    f"Log validation failure for {val_interaction.resource_name}",
                    f"Notify coordinator of failure",
                    f"Request replacement if available"
                ]
            )
            validation_control_steps.append(validate_step)
            
            # Step 3: Forward to manager
            forward_step = ValidationControlStep(
                step_id=f"V{i}.3",
                description=f"Forward validated {val_interaction.resource_name} to manager",
                validation_state_input=ValidationState.VALIDATED,
                validation_state_output=ValidationState.VALIDATED,
                responsible_controller=val_interaction.validator_controller,
                validation_dependencies=[f"V{i}.2"],
                failure_handling=[f"Retry forwarding to manager"]
            )
            validation_control_steps.append(forward_step)
        
        return validation_control_steps

    def analyze_enhanced_uc_steps(self, uc_text: str, validation_control_steps: List[ValidationControlStep]) -> List[EnhancedControlFlowStep]:
        """
        Analyze UC steps with enhanced validation control flow
        """
        enhanced_steps = []
        lines = uc_text.split('\n')
        
        # UC step patterns
        basic_pattern = r"B(\d+[a-z]?)\s+(.*)"
        alt_pattern = r"A(\d+)\.?(\d+)?\s+at\s+B(\d+[a-z]?)\s+(.*)"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse basic steps
            basic_match = re.search(basic_pattern, line)
            if basic_match:
                step_id = f"B{basic_match.group(1)}"
                description = basic_match.group(2)
                
                enhanced_step = self._create_enhanced_control_flow_step(
                    step_id, description, validation_control_steps
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
                
                enhanced_step = self._create_enhanced_control_flow_step(
                    step_id, description, validation_control_steps, is_alternative=True
                )
                enhanced_steps.append(enhanced_step)
        
        return enhanced_steps

    def _create_enhanced_control_flow_step(self, step_id: str, description: str, validation_control_steps: List[ValidationControlStep], is_alternative: bool = False) -> EnhancedControlFlowStep:
        """
        Create enhanced control flow step with validation integration
        """
        desc_lower = description.lower()
        
        # Determine controller and validation requirements
        controller = "GeträenkeOrchestrator"  # Default
        required_validations = []
        state_preconditions = []
        state_postconditions = []
        flow_type = EnhancedControlFlowType.PROCESSING_SEQUENTIAL
        validation_flow_type = ValidationFlowType.MANDATORY_VALIDATION
        parallel_with = []
        
        # Map UC steps to enhanced control flow
        if step_id == "B1":  # Timer trigger
            controller = "ZeitManager"
            flow_type = EnhancedControlFlowType.CONDITIONAL
            validation_flow_type = ValidationFlowType.REAL_TIME_VALIDATION
            state_postconditions = [ValidationState.RAW]  # Triggers validation sequence
            
        elif step_id in ["B2a", "B2b", "B2c", "B2d"]:  # Parallel preparation steps
            flow_type = EnhancedControlFlowType.VALIDATION_PARALLEL
            parallel_with = ["B2a", "B2b", "B2c", "B2d"]
            parallel_with.remove(step_id)  # Remove self
            
            if step_id == "B2a":  # Water heating
                controller = "WaterManager"
                # Requires validated water
                water_validations = [vcs for vcs in validation_control_steps if "water" in vcs.description.lower()]
                required_validations.extend(water_validations)
                state_preconditions = [ValidationState.VALIDATED]
                state_postconditions = [ValidationState.PROCESSED]
                
            elif step_id == "B2c":  # Grinding
                controller = "CoffeeBeansManager"
                # Requires validated coffee beans
                coffee_validations = [vcs for vcs in validation_control_steps if "coffee" in vcs.description.lower()]
                required_validations.extend(coffee_validations)
                state_preconditions = [ValidationState.VALIDATED]
                state_postconditions = [ValidationState.PROCESSED]
                
            elif step_id == "B2d":  # Cup retrieval
                controller = "CupManager"
                # Requires validated cup
                cup_validations = [vcs for vcs in validation_control_steps if "cup" in vcs.description.lower()]
                required_validations.extend(cup_validations)
                state_preconditions = [ValidationState.VALIDATED]
                state_postconditions = [ValidationState.VALIDATED]  # Cup stays validated
                
        elif step_id in ["B3a", "B3b"]:  # Sequential processing steps
            flow_type = EnhancedControlFlowType.PROCESSING_SEQUENTIAL
            
            if step_id == "B3a":  # Brewing
                controller = "BrewingManager"
                state_preconditions = [ValidationState.PROCESSED, ValidationState.PROCESSED]  # Ground coffee + hot water
                state_postconditions = [ValidationState.PROCESSED]  # Brewed coffee
                
            elif step_id == "B3b":  # Add milk
                controller = "MilkManager"
                # Requires validated milk
                milk_validations = [vcs for vcs in validation_control_steps if "milk" in vcs.description.lower()]
                required_validations.extend(milk_validations)
                state_preconditions = [ValidationState.VALIDATED, ValidationState.PROCESSED]  # Milk + brewed coffee
                state_postconditions = [ValidationState.READY]  # Ready milk coffee
                
        elif step_id in ["B4", "B5"]:  # Output steps
            controller = "UserInterfaceManager"
            flow_type = EnhancedControlFlowType.PROCESSING_SEQUENTIAL
            state_preconditions = [ValidationState.READY]
            state_postconditions = [ValidationState.CONSUMED]
            
        elif is_alternative:  # Error handling
            flow_type = EnhancedControlFlowType.ERROR_HANDLING
            validation_flow_type = ValidationFlowType.OPTIONAL_VALIDATION
        
        return EnhancedControlFlowStep(
            step_id=step_id,
            step_type="alternative" if is_alternative else "basic",
            description=description,
            validation_control_steps=required_validations,
            responsible_controller=controller,
            flow_type=flow_type,
            validation_flow_type=validation_flow_type,
            state_preconditions=state_preconditions,
            state_postconditions=state_postconditions,
            parallel_with=parallel_with
        )

    def create_validation_flow_patterns(self, validation_control_steps: List[ValidationControlStep], enhanced_steps: List[EnhancedControlFlowStep]) -> List[ValidationFlowPattern]:
        """
        Create validation flow patterns from control steps
        """
        validation_flow_patterns = []
        
        # Pattern 1: Input Validation Pattern
        input_validation_steps = [vcs for vcs in validation_control_steps if vcs.validation_state_input == ValidationState.RAW]
        
        input_pattern = ValidationFlowPattern(
            pattern_id="InputValidationFlow",
            pattern_type="input_validation",
            validation_sequence=input_validation_steps,
            processing_sequence=[
                "1. Accept all input resources simultaneously",
                "2. Validate each resource in parallel (quality, quantity, safety)",
                "3. Confirm validation completion for each resource",
                "4. Forward all validated resources to respective managers"
            ],
            state_transition_map={
                "coffee_beans": [ValidationState.RAW, ValidationState.VALIDATED],
                "water": [ValidationState.RAW, ValidationState.VALIDATED],
                "milk": [ValidationState.RAW, ValidationState.VALIDATED],
                "cup": [ValidationState.RAW, ValidationState.VALIDATED]
            },
            coordination_rules=[
                "GeträenkeOrchestrator waits for all validations to complete",
                "No processing starts until all resources validated",
                "Validation failures trigger error handling"
            ],
            error_recovery=[
                "Request replacement for failed resources",
                "Abort if critical resources unavailable",
                "Log all validation failures"
            ]
        )
        validation_flow_patterns.append(input_pattern)
        
        # Pattern 2: Processing Flow Pattern
        processing_steps = [step for step in enhanced_steps if step.flow_type in [
            EnhancedControlFlowType.PROCESSING_SEQUENTIAL,
            EnhancedControlFlowType.PROCESSING_PARALLEL
        ]]
        
        processing_pattern = ValidationFlowPattern(
            pattern_id="ProcessingValidationFlow",
            pattern_type="processing_validation",
            validation_sequence=[],  # No validation steps, uses pre-validated resources
            processing_sequence=[
                "1. Begin parallel processing of validated resources (B2a||B2b||B2c||B2d)",
                "2. Transition validated resources to processed state",
                "3. Sequential brewing with processed ingredients (B3a)",
                "4. Final assembly with validated milk (B3b)",
                "5. Transition to ready state for user delivery"
            ],
            state_transition_map={
                "processing": [ValidationState.VALIDATED, ValidationState.PROCESSED, ValidationState.READY]
            },
            coordination_rules=[
                "GeträenkeOrchestrator coordinates parallel processing",
                "Sequential dependencies respected (grinding before brewing)",
                "State transitions monitored throughout"
            ],
            error_recovery=[
                "Retry processing operations on failure",
                "Validate intermediate states",
                "Escalate to coordinator on repeated failures"
            ]
        )
        validation_flow_patterns.append(processing_pattern)
        
        return validation_flow_patterns

    def create_enhanced_control_flow_patterns(self, enhanced_steps: List[EnhancedControlFlowStep], validation_flow_patterns: List[ValidationFlowPattern]) -> List[EnhancedControlFlowPattern]:
        """
        Create enhanced control flow patterns with validation integration
        """
        enhanced_patterns = []
        
        # Main flow pattern with validation
        main_steps = [step for step in enhanced_steps if step.step_type == "basic"]
        
        main_pattern = EnhancedControlFlowPattern(
            pattern_id="MainFlowWithValidation",
            pattern_type="main_flow_validated",
            enhanced_steps=main_steps,
            validation_flow_patterns=validation_flow_patterns,
            coordination_sequence=[
                "1. Timer triggers validation sequence",
                "2. Parallel validation of all input resources",
                "3. Coordinator confirms all validations complete",
                "4. Parallel processing of validated resources",
                "5. Sequential assembly and delivery"
            ],
            error_handling=[
                "Validation failures abort process",
                "Processing errors trigger retry",
                "Critical failures escalate to user notification"
            ]
        )
        enhanced_patterns.append(main_pattern)
        
        # Alternative flow patterns
        alt_steps = [step for step in enhanced_steps if step.step_type == "alternative"]
        if alt_steps:
            alt_pattern = EnhancedControlFlowPattern(
                pattern_id="AlternativeFlowWithValidation",
                pattern_type="alternative_flow_validated",
                enhanced_steps=alt_steps,
                validation_flow_patterns=[],
                coordination_sequence=[
                    "1. Detect validation or processing failure",
                    "2. Execute appropriate error handling",
                    "3. Attempt recovery if possible",
                    "4. Notify user of failure if recovery impossible"
                ],
                error_handling=alt_steps[0].validation_control_steps[0].failure_handling if alt_steps and alt_steps[0].validation_control_steps else []
            )
            enhanced_patterns.append(alt_pattern)
        
        return enhanced_patterns

    def validate_enhanced_control_flow_rules(self, enhanced_patterns: List[EnhancedControlFlowPattern], validation_flow_patterns: List[ValidationFlowPattern]) -> List[str]:
        """
        Validate enhanced control flow rules
        """
        violations = []
        
        # Rule 1: Validation before processing
        for pattern in enhanced_patterns:
            if pattern.pattern_type == "main_flow_validated":
                validation_steps = [step for step in pattern.enhanced_steps if step.validation_control_steps]
                processing_steps = [step for step in pattern.enhanced_steps if step.flow_type in [
                    EnhancedControlFlowType.PROCESSING_SEQUENTIAL,
                    EnhancedControlFlowType.PROCESSING_PARALLEL
                ]]
                
                # Check if all processing steps have proper validation preconditions
                for proc_step in processing_steps:
                    if not proc_step.state_preconditions:
                        violations.append(f"Processing step {proc_step.step_id} lacks validation state preconditions")
        
        # Rule 2: State consistency
        for pattern in enhanced_patterns:
            for step in pattern.enhanced_steps:
                if step.state_preconditions and step.state_postconditions:
                    # Check for valid state transitions
                    for pre_state in step.state_preconditions:
                        valid_transitions = {
                            ValidationState.RAW: [ValidationState.VALIDATED],
                            ValidationState.VALIDATED: [ValidationState.PROCESSED, ValidationState.VALIDATED],
                            ValidationState.PROCESSED: [ValidationState.READY, ValidationState.PROCESSED],
                            ValidationState.READY: [ValidationState.CONSUMED]
                        }
                        
                        valid_next_states = valid_transitions.get(pre_state, [])
                        if not any(post_state in valid_next_states for post_state in step.state_postconditions):
                            violations.append(f"Invalid state transition in step {step.step_id}: {pre_state} -> {step.state_postconditions}")
        
        # Rule 3: Coordinator orchestration
        coordinator_steps = []
        for pattern in enhanced_patterns:
            coordinator_steps.extend([step for step in pattern.enhanced_steps if "orchestrator" in step.responsible_controller.lower() or "coordinator" in step.responsible_controller.lower()])
        
        if not coordinator_steps:
            violations.append("No coordinator orchestration found in control flow")
        
        return violations

    def perform_enhanced_phase4_analysis(self, enhanced_phase3_result: EnhancedPhase3Result, uc_text: str) -> EnhancedPhase4Result:
        """
        Complete enhanced Phase 4 analysis with validation control flow
        """
        # Create validation control steps from Phase 3 validation interactions
        validation_control_steps = self.create_validation_control_steps(enhanced_phase3_result.validation_interactions)
        
        # Analyze UC steps with enhanced validation control flow
        enhanced_steps = self.analyze_enhanced_uc_steps(uc_text, validation_control_steps)
        
        # Create validation flow patterns
        validation_flow_patterns = self.create_validation_flow_patterns(validation_control_steps, enhanced_steps)
        
        # Create enhanced control flow patterns
        enhanced_patterns = self.create_enhanced_control_flow_patterns(enhanced_steps, validation_flow_patterns)
        
        # Build coordination rules
        coordination_rules = self._build_enhanced_coordination_rules(enhanced_patterns, validation_flow_patterns)
        
        # Validate enhanced control flow rules
        violations = self.validate_enhanced_control_flow_rules(enhanced_patterns, validation_flow_patterns)
        
        # Generate summary
        summary = self._generate_enhanced_summary(enhanced_patterns, validation_flow_patterns, violations)
        
        return EnhancedPhase4Result(
            enhanced_phase3_result=enhanced_phase3_result,
            enhanced_control_flow_patterns=enhanced_patterns,
            validation_flow_patterns=validation_flow_patterns,
            coordination_rules=coordination_rules,
            flow_violations=violations,
            summary=summary
        )

    def _build_enhanced_coordination_rules(self, enhanced_patterns: List[EnhancedControlFlowPattern], validation_flow_patterns: List[ValidationFlowPattern]) -> Dict[str, List[str]]:
        """Build enhanced coordination rules"""
        rules = {
            "validation_coordination": [
                "GeträenkeOrchestrator ensures all resources validated before processing",
                "Validation failures trigger coordinated error handling",
                "Parallel validations coordinated for efficiency"
            ],
            "processing_coordination": [
                "GeträenkeOrchestrator sequences processing after validation completion",
                "Parallel processing coordinated for independent resources",
                "Sequential dependencies respected for dependent operations"
            ],
            "state_coordination": [
                "State transitions coordinated throughout system",
                "Invalid state transitions prevented by coordinator",
                "State consistency maintained across all resources"
            ],
            "error_coordination": [
                "Validation errors propagate through coordination hierarchy",
                "Processing errors trigger appropriate recovery actions",
                "Critical failures escalate to user notification"
            ]
        }
        return rules

    def _generate_enhanced_summary(self, enhanced_patterns: List[EnhancedControlFlowPattern], validation_flow_patterns: List[ValidationFlowPattern], violations: List[str]) -> str:
        """Generate enhanced summary"""
        total_patterns = len(enhanced_patterns)
        total_validation_patterns = len(validation_flow_patterns)
        total_violations = len(violations)
        
        total_steps = sum(len(pattern.enhanced_steps) for pattern in enhanced_patterns)
        validation_steps = sum(len(step.validation_control_steps) for pattern in enhanced_patterns for step in pattern.enhanced_steps)
        
        summary_parts = [
            f"Enhanced control flow analysis with validation integration",
            f"Created {total_patterns} enhanced control flow patterns",
            f"Generated {total_validation_patterns} validation flow patterns",
            f"Analyzed {total_steps} steps with {validation_steps} validation requirements",
            f"Detected {total_violations} enhanced control flow violations",
            "Applied validation state management and coordination rules"
        ]
        
        return "; ".join(summary_parts)


def main():
    """Test enhanced Phase 4 analysis"""
    
    # Load UC text
    with open('Use Case/UC1.txt', 'r', encoding='utf-8') as f:
        uc1_text = f.read()
    
    # Load enhanced Phase 3 result
    with open('Zwischenprodukte/UC1_Coffee_enhanced_phase3_analysis.json', 'r', encoding='utf-8') as f:
        phase3_data = json.load(f)
    
    # Create mock enhanced Phase 3 result
    from dataclasses import dataclass
    
    @dataclass
    class MockValidationInteraction:
        interaction_id: str
        validator_controller: str
        resource_name: str
        input_state: ValidationState
        output_state: ValidationState
    
    validation_interactions = []
    for vi_data in phase3_data["validation_interactions"]:
        vi = MockValidationInteraction(
            interaction_id=vi_data["interaction_id"],
            validator_controller=vi_data["validator_controller"],
            resource_name=vi_data["resource_name"],
            input_state=ValidationState(vi_data["input_state"]),
            output_state=ValidationState(vi_data["output_state"])
        )
        validation_interactions.append(vi)
    
    @dataclass
    class MockEnhancedPhase3Result:
        validation_interactions: List[MockValidationInteraction]
    
    enhanced_phase3_result = MockEnhancedPhase3Result(validation_interactions=validation_interactions)
    
    # Initialize enhanced analyzer
    enhanced_analyzer = EnhancedKontrollflussAnalyzer()
    
    # Perform enhanced Phase 4 analysis
    enhanced_result = enhanced_analyzer.perform_enhanced_phase4_analysis(enhanced_phase3_result, uc1_text)
    
    # Save result
    output_file = "Zwischenprodukte/UC1_Coffee_enhanced_phase4_analysis.json"
    
    serializable_result = {
        "summary": enhanced_result.summary,
        "enhanced_control_flow_patterns": [
            {
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "steps_count": len(pattern.enhanced_steps),
                "validation_patterns_count": len(pattern.validation_flow_patterns),
                "coordination_sequence": pattern.coordination_sequence
            } for pattern in enhanced_result.enhanced_control_flow_patterns
        ],
        "validation_flow_patterns": [
            {
                "pattern_id": vfp.pattern_id,
                "pattern_type": vfp.pattern_type,
                "processing_sequence": vfp.processing_sequence,
                "coordination_rules": vfp.coordination_rules
            } for vfp in enhanced_result.validation_flow_patterns
        ],
        "coordination_rules": enhanced_result.coordination_rules,
        "violations": enhanced_result.flow_violations
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)
    
    print(f"Enhanced Phase 4 Analysis completed!")
    print(f"File saved to: {output_file}")
    print(f"Summary: {enhanced_result.summary}")
    print(f"Enhanced Control Flow Patterns: {len(enhanced_result.enhanced_control_flow_patterns)}")
    print(f"Validation Flow Patterns: {len(enhanced_result.validation_flow_patterns)}")
    print(f"Violations: {len(enhanced_result.flow_violations)}")
    
    if enhanced_result.flow_violations:
        print("Violations found:")
        for violation in enhanced_result.flow_violations:
            print(f"  - {violation}")
    else:
        print("No violations found!")
    
    return enhanced_result

if __name__ == "__main__":
    main()