"""
Phase 4: Kontrollfluss-Analyse (Control Flow Analysis) - UC-Methode.txt Implementation
Analyzes control flow patterns, decision points, and execution sequences
Ensures proper control flow rules are followed based on UC steps and interactions
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


class ControlFlowType(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    EXCEPTION = "exception"


class DecisionType(Enum):
    RESOURCE_CHECK = "resource_check"
    STATE_CHECK = "state_check"
    USER_INPUT = "user_input"
    SYSTEM_CONDITION = "system_condition"
    ERROR_HANDLING = "error_handling"


@dataclass
class ControlFlowStep:
    step_id: str
    step_type: str  # "basic", "alternative", "extension"
    description: str
    trigger_condition: Optional[str] = None
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    responsible_controller: Optional[str] = None
    flow_type: ControlFlowType = ControlFlowType.SEQUENTIAL
    parallel_with: List[str] = field(default_factory=list)


@dataclass
class DecisionPoint:
    decision_id: str
    step_location: str
    condition: str
    decision_type: DecisionType
    true_path: List[str]  # Steps if condition is true
    false_path: List[str]  # Steps if condition is false
    responsible_controller: str
    resource_dependencies: List[str] = field(default_factory=list)


@dataclass
class ControlFlowPattern:
    pattern_id: str
    pattern_type: str  # "main_flow", "alternative_flow", "extension_flow"
    steps: List[ControlFlowStep]
    decision_points: List[DecisionPoint]
    coordination_sequence: List[str]
    error_handling: List[str] = field(default_factory=list)


@dataclass
class Phase4Result:
    phase3_result: Phase3Result
    control_flow_patterns: List[ControlFlowPattern]
    decision_points: List[DecisionPoint]
    coordination_rules: Dict[str, List[str]]
    flow_violations: List[str]
    summary: str


class KontrollflussAnalyzer:
    """
    Phase 4 implementation: Analyzes control flow patterns and ensures proper flow rules
    """
    
    def __init__(self, domain_analyzer: DomainConfigurableAnalyzer = None):
        """
        Initialize with optional domain analyzer for context
        """
        self.domain_analyzer = domain_analyzer or DomainConfigurableAnalyzer()
        self.nlp = self.domain_analyzer.nlp
        
        # Control flow patterns for step identification
        self.step_patterns = {
            "basic": r"B(\d+[a-z]?)\s+(.*)",
            "alternative": r"A(\d+)\.?(\d+)?\s+at\s+B(\d+[a-z]?)\s+(.*)",
            "extension": r"E(\d+)\.?(\d+)?\s+B(\d+[a-z]?)-?B?(\d+[a-z]?)?\s+\(trigger\)\s+(.*)"
        }
        
        # Decision pattern recognition
        self.decision_patterns = [
            r"(?:too little|insufficient|not enough|empty|low)\s+(\w+)",
            r"(?:if|when)\s+(.*?)\s+(?:then|:)",
            r"system\s+(?:checks?|validates?|verifies?)\s+(.*)",
            r"(?:has|contains|available)\s+(.*?)\s+(?:in|within)\s+system"
        ]
        
        # Control flow rules for UC-Methode
        self.control_flow_rules = {
            "coordinator_orchestration": "Coordinator controllers must orchestrate all parallel activities",
            "event_handler_limitation": "Event handlers (Timer, Sensor, etc.) must NOT coordinate business logic",
            "sequential_dependency": "Resource-dependent steps must execute sequentially",
            "parallel_independence": "Independent resource operations can execute in parallel",
            "error_propagation": "Errors must propagate through coordination hierarchy",
            "decision_responsibility": "Decision points must be handled by appropriate controllers"
        }

    def analyze_uc_steps(self, uc_text: str) -> List[ControlFlowStep]:
        """
        Extract and analyze UC steps from text to identify control flow patterns
        """
        steps = []
        lines = uc_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for basic steps (B1, B2a, etc.)
            basic_match = re.search(self.step_patterns["basic"], line)
            if basic_match:
                step_id = f"B{basic_match.group(1)}"
                description = basic_match.group(2)
                
                # Determine flow type based on step characteristics
                flow_type = self._determine_flow_type(step_id, description)
                parallel_with = self._identify_parallel_steps(step_id, description)
                
                step = ControlFlowStep(
                    step_id=step_id,
                    step_type="basic",
                    description=description,
                    flow_type=flow_type,
                    parallel_with=parallel_with
                )
                steps.append(step)
                continue
            
            # Check for alternative flows (A1.1, A2, etc.)
            alt_match = re.search(self.step_patterns["alternative"], line)
            if alt_match:
                step_id = f"A{alt_match.group(1)}"
                if alt_match.group(2):
                    step_id += f".{alt_match.group(2)}"
                trigger_step = f"B{alt_match.group(3)}"
                description = alt_match.group(4)
                
                step = ControlFlowStep(
                    step_id=step_id,
                    step_type="alternative",
                    description=description,
                    trigger_condition=f"Alternative at {trigger_step}",
                    flow_type=ControlFlowType.CONDITIONAL
                )
                steps.append(step)
                continue
            
            # Check for extension flows (E1.1, etc.)
            ext_match = re.search(self.step_patterns["extension"], line)
            if ext_match:
                step_id = f"E{ext_match.group(1)}"
                if ext_match.group(2):
                    step_id += f".{ext_match.group(2)}"
                trigger_range = f"B{ext_match.group(3)}"
                if ext_match.group(4):
                    trigger_range += f"-B{ext_match.group(4)}"
                description = ext_match.group(5)
                
                step = ControlFlowStep(
                    step_id=step_id,
                    step_type="extension",
                    description=description,
                    trigger_condition=f"Extension during {trigger_range}",
                    flow_type=ControlFlowType.CONDITIONAL
                )
                steps.append(step)
        
        return steps

    def _determine_flow_type(self, step_id: str, description: str) -> ControlFlowType:
        """
        Determine the control flow type for a step based on its characteristics
        """
        # Parallel indicators - multiple independent activities
        parallel_indicators = [
            "activates", "prepares", "begins", "starts",
            "simultaneously", "at the same time", "in parallel"
        ]
        
        # Sequential indicators - dependent activities
        sequential_indicators = [
            "after", "then", "subsequently", "following",
            "into the", "to the", "with the"
        ]
        
        # Conditional indicators
        conditional_indicators = [
            "if", "when", "checks", "validates", "verifies"
        ]
        
        description_lower = description.lower()
        
        # Check for conditional flow
        if any(indicator in description_lower for indicator in conditional_indicators):
            return ControlFlowType.CONDITIONAL
        
        # Check for parallel flow based on step patterns
        if step_id.startswith("B2") and any(indicator in description_lower for indicator in parallel_indicators):
            return ControlFlowType.PARALLEL
        
        # Check for sequential dependencies
        if any(indicator in description_lower for indicator in sequential_indicators):
            return ControlFlowType.SEQUENTIAL
        
        # Default to sequential
        return ControlFlowType.SEQUENTIAL

    def _identify_parallel_steps(self, step_id: str, description: str) -> List[str]:
        """
        Identify steps that can execute in parallel with the given step
        """
        parallel_with = []
        
        # UC1 specific parallel patterns
        if step_id == "B2a":  # Water heater activation
            parallel_with = ["B2b", "B2c"]  # Filter prep and grinding can be parallel
        elif step_id == "B2b":  # Filter preparation
            parallel_with = ["B2a", "B2c"]
        elif step_id == "B2c":  # Grinding
            parallel_with = ["B2a", "B2b"]
        
        return parallel_with

    def identify_decision_points(self, steps: List[ControlFlowStep], uc_text: str) -> List[DecisionPoint]:
        """
        Identify decision points in the control flow
        """
        decision_points = []
        lines = uc_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for alternative flow triggers (error conditions)
            alt_match = re.search(r"A(\d+)\s+at\s+B(\d+[a-z]?)\s+(.*)", line)
            if alt_match:
                decision_id = f"D{alt_match.group(1)}"
                step_location = f"B{alt_match.group(2)}"
                condition = alt_match.group(3)
                
                # Determine decision type
                decision_type = self._classify_decision_type(condition)
                
                # Find the alternative steps
                alt_steps = self._find_alternative_steps(alt_match.group(1), lines)
                
                decision_point = DecisionPoint(
                    decision_id=decision_id,
                    step_location=step_location,
                    condition=condition,
                    decision_type=decision_type,
                    true_path=alt_steps,
                    false_path=["continue_main_flow"],
                    responsible_controller=self._determine_responsible_controller(condition)
                )
                decision_points.append(decision_point)
        
        return decision_points

    def _classify_decision_type(self, condition: str) -> DecisionType:
        """
        Classify the type of decision based on the condition
        """
        condition_lower = condition.lower()
        
        if any(term in condition_lower for term in ["too little", "insufficient", "not enough", "empty", "low"]):
            return DecisionType.RESOURCE_CHECK
        elif any(term in condition_lower for term in ["user", "wants", "input"]):
            return DecisionType.USER_INPUT
        elif any(term in condition_lower for term in ["system", "state", "status"]):
            return DecisionType.SYSTEM_CONDITION
        elif any(term in condition_lower for term in ["error", "failure", "problem"]):
            return DecisionType.ERROR_HANDLING
        else:
            return DecisionType.STATE_CHECK

    def _find_alternative_steps(self, alt_number: str, lines: List[str]) -> List[str]:
        """
        Find all steps belonging to an alternative flow
        """
        alt_steps = []
        in_alternative = False
        
        for line in lines:
            line = line.strip()
            if re.match(rf"A{alt_number}\.?\d*\s+", line):
                in_alternative = True
                # Extract step ID
                match = re.search(rf"A{alt_number}\.?(\d*)", line)
                if match:
                    step_id = f"A{alt_number}"
                    if match.group(1):
                        step_id += f".{match.group(1)}"
                    alt_steps.append(step_id)
            elif in_alternative and (line.startswith("A") or line.startswith("E") or line.startswith("Postcondition")):
                break
        
        return alt_steps

    def _determine_responsible_controller(self, condition: str) -> str:
        """
        Determine which controller is responsible for handling the decision
        """
        condition_lower = condition.lower()
        
        if "water" in condition_lower:
            return "WaterManager"
        elif "milk" in condition_lower:
            return "MilkManager"
        elif "coffee" in condition_lower or "bean" in condition_lower:
            return "CoffeeBeansManager"
        elif "cup" in condition_lower or "container" in condition_lower:
            return "CupManager"
        else:
            return "GeträenkeOrchestrator"  # Default to coordinator

    def create_control_flow_patterns(self, steps: List[ControlFlowStep], decision_points: List[DecisionPoint]) -> List[ControlFlowPattern]:
        """
        Create control flow patterns from steps and decision points
        """
        patterns = []
        
        # Main flow pattern
        main_steps = [step for step in steps if step.step_type == "basic"]
        main_pattern = ControlFlowPattern(
            pattern_id="main_flow",
            pattern_type="main_flow",
            steps=main_steps,
            decision_points=[dp for dp in decision_points if any(step.step_id == dp.step_location for step in main_steps)],
            coordination_sequence=self._build_coordination_sequence(main_steps)
        )
        patterns.append(main_pattern)
        
        # Alternative flow patterns
        alt_steps_by_number = {}
        for step in steps:
            if step.step_type == "alternative":
                alt_number = step.step_id.split('.')[0]  # A1, A2, etc.
                if alt_number not in alt_steps_by_number:
                    alt_steps_by_number[alt_number] = []
                alt_steps_by_number[alt_number].append(step)
        
        for alt_number, alt_steps in alt_steps_by_number.items():
            alt_pattern = ControlFlowPattern(
                pattern_id=f"alternative_flow_{alt_number}",
                pattern_type="alternative_flow",
                steps=alt_steps,
                decision_points=[dp for dp in decision_points if dp.decision_id.endswith(alt_number[1:])],
                coordination_sequence=self._build_coordination_sequence(alt_steps),
                error_handling=[step.step_id for step in alt_steps]
            )
            patterns.append(alt_pattern)
        
        # Extension flow patterns
        ext_steps_by_number = {}
        for step in steps:
            if step.step_type == "extension":
                ext_number = step.step_id.split('.')[0]  # E1, E2, etc.
                if ext_number not in ext_steps_by_number:
                    ext_steps_by_number[ext_number] = []
                ext_steps_by_number[ext_number].append(step)
        
        for ext_number, ext_steps in ext_steps_by_number.items():
            ext_pattern = ControlFlowPattern(
                pattern_id=f"extension_flow_{ext_number}",
                pattern_type="extension_flow",
                steps=ext_steps,
                decision_points=[],
                coordination_sequence=self._build_coordination_sequence(ext_steps)
            )
            patterns.append(ext_pattern)
        
        return patterns

    def _build_coordination_sequence(self, steps: List[ControlFlowStep]) -> List[str]:
        """
        Build the coordination sequence for a set of steps
        """
        coordination_sequence = []
        
        for step in steps:
            if step.flow_type == ControlFlowType.PARALLEL:
                if step.parallel_with:
                    coordination_sequence.append(f"Coordinate parallel: {step.step_id} with {', '.join(step.parallel_with)}")
                else:
                    coordination_sequence.append(f"Execute parallel: {step.step_id}")
            elif step.flow_type == ControlFlowType.CONDITIONAL:
                coordination_sequence.append(f"Decision point: {step.step_id}")
            else:
                coordination_sequence.append(f"Execute sequential: {step.step_id}")
        
        return coordination_sequence

    def validate_control_flow_rules(self, patterns: List[ControlFlowPattern], phase3_result: Phase3Result) -> List[str]:
        """
        Validate that control flow follows UC-Methode rules
        """
        violations = []
        
        # Rule 1: Coordinator orchestration
        coordinator = None
        for controller in phase3_result.all_objects.get("controllers", []):
            if "orchestrator" in controller.lower() or "coordinator" in controller.lower():
                coordinator = controller
                break
        
        if not coordinator:
            violations.append("No coordinator controller found for orchestration")
        
        # Rule 2: Event handler limitation
        event_handlers = ["ZeitManager", "TimerManager", "SensorManager"]
        for pattern in patterns:
            for step in pattern.steps:
                if (step.responsible_controller in event_handlers and 
                    step.flow_type != ControlFlowType.SEQUENTIAL):
                    violations.append(f"Event handler {step.responsible_controller} should not coordinate complex flow in {step.step_id}")
        
        # Rule 3: Resource dependency validation
        for pattern in patterns:
            for i, step in enumerate(pattern.steps[:-1]):
                next_step = pattern.steps[i + 1]
                if self._has_resource_dependency(step, next_step) and step.flow_type == ControlFlowType.PARALLEL:
                    violations.append(f"Resource-dependent steps {step.step_id} and {next_step.step_id} should not be parallel")
        
        return violations

    def _has_resource_dependency(self, step1: ControlFlowStep, step2: ControlFlowStep) -> bool:
        """
        Check if step2 depends on resources from step1
        """
        # Simple heuristic: if step1 produces something that step2 uses
        step1_desc = step1.description.lower()
        step2_desc = step2.description.lower()
        
        # Check for common dependencies
        if "grind" in step1_desc and ("brewing" in step2_desc or "coffee" in step2_desc):
            return True
        if "heat" in step1_desc and "brewing" in step2_desc:
            return True
        if "brewing" in step1_desc and "add" in step2_desc:
            return True
        
        return False

    def perform_phase4_analysis(self, phase3_result: Phase3Result, uc_text: str) -> Phase4Result:
        """
        Complete Phase 4 control flow analysis
        
        Args:
            phase3_result: Result from Phase 3 analysis
            uc_text: Complete use case text
            
        Returns:
            Complete Phase4Result
        """
        # Analyze UC steps for control flow
        steps = self.analyze_uc_steps(uc_text)
        
        # Identify decision points
        decision_points = self.identify_decision_points(steps, uc_text)
        
        # Create control flow patterns
        patterns = self.create_control_flow_patterns(steps, decision_points)
        
        # Assign responsible controllers from Phase 3
        self._assign_controllers_to_steps(steps, phase3_result)
        
        # Build coordination rules
        coordination_rules = self._build_coordination_rules(patterns, phase3_result)
        
        # Validate control flow rules
        violations = self.validate_control_flow_rules(patterns, phase3_result)
        
        # Generate summary
        summary = self._generate_phase4_summary(patterns, decision_points, violations)
        
        return Phase4Result(
            phase3_result=phase3_result,
            control_flow_patterns=patterns,
            decision_points=decision_points,
            coordination_rules=coordination_rules,
            flow_violations=violations,
            summary=summary
        )

    def _assign_controllers_to_steps(self, steps: List[ControlFlowStep], phase3_result: Phase3Result):
        """
        Assign responsible controllers to steps based on Phase 3 analysis
        """
        for step in steps:
            step.responsible_controller = self._determine_step_controller(step, phase3_result)

    def _determine_step_controller(self, step: ControlFlowStep, phase3_result: Phase3Result) -> str:
        """
        Determine which controller is responsible for a step
        """
        desc_lower = step.description.lower()
        
        # Map step activities to controllers
        if any(term in desc_lower for term in ["activate", "heat", "water"]):
            return "WaterManager"
        elif any(term in desc_lower for term in ["prepare", "filter"]):
            return "FilterManager"
        elif any(term in desc_lower for term in ["grind", "bean", "coffee"]):
            return "CoffeeBeansManager"
        elif any(term in desc_lower for term in ["retrieve", "cup", "place"]):
            return "CupManager"
        elif any(term in desc_lower for term in ["brewing", "begin"]):
            return "BrewingManager"
        elif any(term in desc_lower for term in ["milk", "add"]):
            return "MilkManager"
        elif any(term in desc_lower for term in ["output", "message", "present"]):
            return "UserInterfaceManager"
        else:
            return "GeträenkeOrchestrator"  # Default coordinator

    def _build_coordination_rules(self, patterns: List[ControlFlowPattern], phase3_result: Phase3Result) -> Dict[str, List[str]]:
        """
        Build coordination rules based on control flow patterns
        """
        rules = {
            "parallel_coordination": [],
            "sequential_coordination": [],
            "error_handling": [],
            "decision_coordination": []
        }
        
        for pattern in patterns:
            for step in pattern.steps:
                if step.flow_type == ControlFlowType.PARALLEL:
                    rules["parallel_coordination"].append(
                        f"GeträenkeOrchestrator coordinates {step.responsible_controller} in parallel with {step.parallel_with}"
                    )
                elif step.flow_type == ControlFlowType.SEQUENTIAL:
                    rules["sequential_coordination"].append(
                        f"GeträenkeOrchestrator ensures {step.responsible_controller} executes after dependencies"
                    )
                elif step.flow_type == ControlFlowType.CONDITIONAL:
                    rules["decision_coordination"].append(
                        f"GeträenkeOrchestrator handles decision from {step.responsible_controller}"
                    )
            
            if pattern.error_handling:
                rules["error_handling"].extend([
                    f"GeträenkeOrchestrator handles error flow {step_id}" for step_id in pattern.error_handling
                ])
        
        return rules

    def _generate_phase4_summary(self, patterns: List[ControlFlowPattern], decision_points: List[DecisionPoint], violations: List[str]) -> str:
        """
        Generate summary of Phase 4 analysis
        """
        total_patterns = len(patterns)
        total_decisions = len(decision_points)
        total_violations = len(violations)
        
        parallel_steps = sum(1 for pattern in patterns for step in pattern.steps if step.flow_type == ControlFlowType.PARALLEL)
        sequential_steps = sum(1 for pattern in patterns for step in pattern.steps if step.flow_type == ControlFlowType.SEQUENTIAL)
        
        summary_parts = [
            f"Analyzed {total_patterns} control flow patterns",
            f"Identified {total_decisions} decision points",
            f"Found {parallel_steps} parallel and {sequential_steps} sequential steps",
            f"Detected {total_violations} control flow violations",
            "Applied UC-Methode control flow rules"
        ]
        
        return "; ".join(summary_parts)


def main():
    """Example usage with UC1"""
    # Load UC1 text
    with open('Use Case/UC1.txt', 'r', encoding='utf-8') as f:
        uc1_text = f.read()
    
    # Load Phase 3 result
    with open('Zwischenprodukte/UC1_Coffee_phase3_analysis.json', 'r', encoding='utf-8') as f:
        phase3_data = json.load(f)
    
    # Initialize analyzer
    analyzer = KontrollflussAnalyzer()
    
    # Mock Phase 3 result for testing
    from dataclasses import dataclass
    @dataclass
    class MockPhase3Result:
        all_objects: Dict = None
    
    phase3_result = MockPhase3Result(all_objects={"controllers": ["GeträenkeOrchestrator", "WaterManager", "CoffeeBeansManager", "MilkManager", "CupManager"]})
    
    # Perform Phase 4 analysis
    phase4_result = analyzer.perform_phase4_analysis(phase3_result, uc1_text)
    
    # Save result
    output_file = "Zwischenprodukte/UC1_Coffee_phase4_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": phase4_result.summary,
            "control_flow_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "pattern_type": p.pattern_type,
                    "steps": [
                        {
                            "step_id": s.step_id,
                            "description": s.description,
                            "flow_type": s.flow_type.value,
                            "responsible_controller": s.responsible_controller,
                            "parallel_with": s.parallel_with
                        } for s in p.steps
                    ],
                    "coordination_sequence": p.coordination_sequence
                } for p in phase4_result.control_flow_patterns
            ],
            "decision_points": [
                {
                    "decision_id": dp.decision_id,
                    "condition": dp.condition,
                    "decision_type": dp.decision_type.value,
                    "responsible_controller": dp.responsible_controller
                } for dp in phase4_result.decision_points
            ],
            "coordination_rules": phase4_result.coordination_rules,
            "violations": phase4_result.flow_violations
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Phase 4 Control Flow Analysis completed!")
    print(f"File saved to: {output_file}")
    print(f"Summary: {phase4_result.summary}")
    if phase4_result.flow_violations:
        print(f"Violations found: {len(phase4_result.flow_violations)}")
        for violation in phase4_result.flow_violations:
            print(f"  - {violation}")
    else:
        print("No control flow violations detected!")

if __name__ == "__main__":
    main()