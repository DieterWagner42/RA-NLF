"""
Phase 3: Interaktions-Analyse - UC-Methode.txt Implementation
Analyzes use case flow step-by-step to create interaction patterns between RA objects

Integrates Phase 1 (context & actors) and Phase 2 (resources) to build complete interaction model
"""

import spacy
import json
import os
import re
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from domain_config_analyzer import DomainConfigurableAnalyzer, Phase1Result
from phase2_betriebsmittel_analyzer import BetriebsmittelAnalyzer, Phase2Result


class InteractionType(Enum):
    TRIGGER = "trigger"
    CONTROL_FLOW = "control_flow"
    DATA_FLOW = "data_flow"
    ORCHESTRATION = "orchestration"


class StepType(Enum):
    TRIGGER_STEP = "trigger_step"      # B1 - trigger from actor or time
    NORMAL_STEP = "normal_step"        # B2a, B2b etc - system actions
    PARALLEL_STEP = "parallel_step"    # B2a-d - can run in parallel
    SEQUENCE_STEP = "sequence_step"    # B3, B4 - sequential after parallel


@dataclass
class UCStep:
    step_id: str  # e.g., "B1", "B2a", "B3"
    step_text: str
    step_type: StepType
    verb: Optional[str] = None
    objects: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    controllers: List[str] = field(default_factory=list)


@dataclass
class Interaction:
    interaction_id: str
    interaction_type: InteractionType
    source_object: str
    target_object: str
    data_object: Optional[str] = None
    description: str = ""
    step_reference: str = ""


@dataclass
class ControllerFunction:
    controller_name: str
    function_name: str
    source: str  # "uc_flow", "phase2_integration", "orchestration"
    description: str
    step_references: List[str] = field(default_factory=list)


@dataclass
class EntityFlow:
    entity_name: str
    source_controller: str
    target_controller: str
    transformation: Optional[str] = None
    step_reference: str = ""


@dataclass
class Phase3Result:
    phase1_result: Phase1Result
    phase2_result: Phase2Result
    uc_steps: List[UCStep]
    interactions: List[Interaction]
    controller_functions: List[ControllerFunction]
    entity_flows: List[EntityFlow]
    orchestration_pattern: Dict[str, List[str]]
    missing_preconditions: List[str]
    summary: str


class InteraktionAnalyzer:
    """
    Phase 3 implementation: Analyzes use case flows step-by-step following UC-Methode.txt
    """
    
    def __init__(self, domain_analyzer: Optional[DomainConfigurableAnalyzer] = None,
                 phase2_analyzer: Optional[BetriebsmittelAnalyzer] = None):
        """Initialize with existing analyzers from previous phases"""
        self.domain_analyzer = domain_analyzer or DomainConfigurableAnalyzer()
        self.phase2_analyzer = phase2_analyzer or BetriebsmittelAnalyzer()
        self.nlp = self.domain_analyzer.nlp
        
        # UC flow parsing patterns
        self.step_patterns = [
            r"^([B]?\d+[a-z]*)\s*\(trigger\)\s*(.*)",      # B1 (trigger) ...
            r"^([B]?\d+[a-z]*)\s+(.*)",                     # B2a ...
            r"^([A]?\d+\.?\d*)\s+(.*)",                     # A1.1 ... (alternative)
            r"^([E]?\d+\.?\d*)\s+(.*)"                      # E1.1 ... (extension)
        ]
        
        # Verb patterns for action identification
        self.action_verbs = [
            "aktiviert", "bereitet", "mahlt", "holt", "beginnt", "gibt", "ausgabe", "präsentiert",
            "schaltet", "stoppt", "adds", "starts", "activates", "prepares", "grinds", "retrieves",
            "begins", "outputs", "presents", "switches", "stops", "provides"
        ]
        
        # Object identification patterns
        self.object_patterns = [
            r"(\w+(?:\s+\w+)*)\s+(?:aktiviert|bereitet|mahlt|holt)",  # object before verb
            r"(?:aktiviert|bereitet|mahlt|holt)\s+(?:den|die|das)?\s*(\w+(?:\s+\w+)*)",  # object after verb
            r"(?:in|aus|mit|von|zu)\s+(?:den|die|das|dem|der)?\s*(\w+(?:\s+\w+)*)"  # object with preposition
        ]

    def parse_uc_steps(self, uc_text: str) -> List[UCStep]:
        """Parse use case text into structured steps"""
        steps = []
        lines = uc_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to match step patterns
            for pattern in self.step_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    step_id = match.group(1)
                    step_text = match.group(2).strip()
                    
                    # Determine step type
                    step_type = self._determine_step_type(step_id, step_text)
                    
                    # Extract verb and objects
                    verb = self._extract_verb(step_text)
                    objects = self._extract_objects(step_text)
                    
                    step = UCStep(
                        step_id=step_id,
                        step_text=step_text,
                        step_type=step_type,
                        verb=verb,
                        objects=objects
                    )
                    steps.append(step)
                    break
        
        return steps

    def _determine_step_type(self, step_id: str, step_text: str) -> StepType:
        """Determine the type of UC step following UC-Methode.txt rules"""
        if "trigger" in step_text.lower() or step_id.endswith("1"):
            return StepType.TRIGGER_STEP
        elif re.match(r".*[a-z]$", step_id):  # B2a, B2b, etc.
            return StepType.PARALLEL_STEP
        else:
            return StepType.NORMAL_STEP

    def _extract_verb(self, step_text: str) -> Optional[str]:
        """Extract action verb from step text"""
        doc = self.nlp(step_text.lower())
        
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ in [v.lower() for v in self.action_verbs]:
                return token.lemma_
        
        # Fallback: check for known action verbs directly
        for verb in self.action_verbs:
            if verb.lower() in step_text.lower():
                return verb.lower()
        
        return None

    def _extract_objects(self, step_text: str) -> List[str]:
        """Extract objects mentioned in step text"""
        objects = []
        
        # Use regex patterns to find objects
        for pattern in self.object_patterns:
            matches = re.findall(pattern, step_text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                
                if match and len(match) > 2:
                    # Clean up the match
                    clean_match = re.sub(r'^(den|die|das|dem|der)\s+', '', match.strip())
                    if clean_match not in objects:
                        objects.append(clean_match)
        
        # Also extract nouns using NLP
        doc = self.nlp(step_text)
        for token in doc:
            if token.pos_ == "NOUN" and not token.is_stop and len(token.text) > 2:
                noun = token.text.lower()
                if noun not in objects and noun not in ["system", "das", "die", "der"]:
                    objects.append(noun)
        
        return objects

    def map_steps_to_controllers(self, steps: List[UCStep], phase2_result: Phase2Result) -> List[UCStep]:
        """Map UC steps to controllers from Phase 2 analysis - COMPLETE SYSTEM BEHAVIOR ANALYSIS"""
        
        print("\n=== PHASE 3: ANALYZING ALL UC STEPS FOR RA CLASSES ===")
        
        # Create mapping from objects to controllers
        object_to_controller = {}
        
        for analysis in phase2_result.resource_analyses:
            resource_name = analysis.resource_name.lower()
            if analysis.manager_controller:
                controller_name = analysis.manager_controller.name
                object_to_controller[resource_name] = controller_name
                
                # Also map variations
                for word in resource_name.split():
                    object_to_controller[word] = controller_name

        # COMPLETE SYSTEM BEHAVIOR ANALYSIS - JEDEN SCHRITT ANALYSIEREN
        for step in steps:
            step.controllers = []
            step.entities = []
            
            print(f"\n--- Analyzing {step.step_id}: {step.step_text} ---")
            print(f"Step Type: {step.step_type.value}")
            print(f"Verb: {step.verb}")
            print(f"Objects: {step.objects}")
            
            # SCHRITT 1: Basis-Mapping von erkannten Objekten
            for obj in step.objects:
                # Try direct mapping
                if obj in object_to_controller:
                    step.controllers.append(object_to_controller[obj])
                    print(f"  MAPPED OBJECT: {obj} -> {object_to_controller[obj]}")
                else:
                    # Try partial matches
                    for resource_key, controller in object_to_controller.items():
                        if any(word in obj for word in resource_key.split()) or \
                           any(word in resource_key for word in obj.split()):
                            if controller not in step.controllers:
                                step.controllers.append(controller)
                                print(f"  PARTIAL MATCH: {obj} -> {controller}")
            
            # SCHRITT 2: VOLLSTÄNDIGE SYSTEMVERHALTEN-ANALYSE 
            # Jeder UC-Schritt beschreibt ein Systemverhalten und benötigt Controller!
            step = self._analyze_system_behavior(step)
            
            print(f"  FOUND CONTROLLERS: {step.controllers}")
            print(f"  FOUND ENTITIES: {step.entities}")
        
        # ERGÄNZUNG: Identifiziere fehlende Output-Controller für UC-Schritte
        steps = self._identify_missing_output_controllers(steps)
        
        return steps
    
    def _analyze_system_behavior(self, step: UCStep) -> UCStep:
        """VOLLSTÄNDIGE ANALYSE: Jeder UC-Schritt beschreibt Systemverhalten und benötigt Controller + Entities"""
        
        step_text = step.step_text.lower()
        step_id = step.step_id
        
        # B1: Time/Clock triggers
        if "clock" in step_text or "time" in step_text or "uhr" in step_text:
            if "ZeitManager" not in step.controllers:
                step.controllers.append("ZeitManager")
                print(f"  SYSTEM BEHAVIOR: {step_id} -> ZeitManager (time coordination)")
            # Zeit-Entity
            if "Zeit" not in step.entities:
                step.entities.append("Zeit")
                print(f"  ENTITY: {step_id} -> Zeit")
        
        # B2a: Water heating activation
        if "activates" in step_text and "water" in step_text:
            if "WaterIsManager" not in step.controllers:
                step.controllers.append("WaterIsManager")
                print(f"  SYSTEM BEHAVIOR: {step_id} -> WaterIsManager (water heating)")
            # Water heating entities
            if "Water" not in step.entities:
                step.entities.append("Water")
            if "Heater" not in step.entities:
                step.entities.append("Heater")
                print(f"  ENTITIES: {step_id} -> Water, Heater")
        
        # B2b: Filter preparation - WURDE IGNORIERT!
        if "prepares" in step_text and "filter" in step_text:
            if "FilterManager" not in step.controllers:
                step.controllers.append("FilterManager")
                print(f"  SYSTEM BEHAVIOR: {step_id} -> FilterManager (filter preparation)")
            # Filter entity
            if "Filter" not in step.entities:
                step.entities.append("Filter")
                print(f"  ENTITY: {step_id} -> Filter")
        
        # B2c: Coffee grinding
        if "grinds" in step_text or "grinding" in step_text:
            if "CoffeeBeansAreManager" not in step.controllers:
                step.controllers.append("CoffeeBeansAreManager")
                print(f"  SYSTEM BEHAVIOR: {step_id} -> CoffeeBeansAreManager (grinding)")
            # Coffee grinding entities
            if "CoffeeBeans" not in step.entities:
                step.entities.append("CoffeeBeans")
            if "GroundCoffee" not in step.entities:
                step.entities.append("GroundCoffee")
                print(f"  ENTITIES: {step_id} -> CoffeeBeans, GroundCoffee")
        
        # B2d: Cup retrieval and placement
        if "retrieves" in step_text and "cup" in step_text:
            if "CupManager" not in step.controllers:
                step.controllers.append("CupManager")
                print(f"  SYSTEM BEHAVIOR: {step_id} -> CupManager (cup handling)")
            # Cup storage entities
            if "Cup" not in step.entities:
                step.entities.append("Cup")
            if "Storage" not in step.entities:
                step.entities.append("Storage")
                print(f"  ENTITIES: {step_id} -> Cup, Storage")
        
        # B3a: Coffee brewing
        if "brewing" in step_text or ("begins" in step_text and "coffee" in step_text):
            # Multiple controllers for brewing coordination
            controllers_needed = ["CoffeeBeansAreManager", "WaterIsManager", "CupManager"]
            for controller in controllers_needed:
                if controller not in step.controllers:
                    step.controllers.append(controller)
            print(f"  SYSTEM BEHAVIOR: {step_id} -> Multiple controllers for brewing coordination")
            # Brewing entities
            entities_needed = ["Coffee", "Water", "Cup"]
            for entity in entities_needed:
                if entity not in step.entities:
                    step.entities.append(entity)
                    print(f"  ENTITY: {step_id} -> {entity}")
        
        # B3b: Milk addition
        if "adds" in step_text and "milk" in step_text:
            controllers_needed = ["MilkIsManager", "CupManager"]
            for controller in controllers_needed:
                if controller not in step.controllers:
                    step.controllers.append(controller)
            print(f"  SYSTEM BEHAVIOR: {step_id} -> Milk addition coordination")
            # Milk entities
            entities_needed = ["Milk", "Cup"]
            for entity in entities_needed:
                if entity not in step.entities:
                    step.entities.append(entity)
                    print(f"  ENTITY: {step_id} -> {entity}")
        
        # B3: Espresso pressure (UC2)
        if "compressor" in step_text and "pressure" in step_text:
            if "WaterIsManager" not in step.controllers:
                step.controllers.append("WaterIsManager")
                print(f"  SYSTEM BEHAVIOR: {step_id} -> WaterIsManager (pressure control)")
            # Pressure entities
            if "Compressor" not in step.entities:
                step.entities.append("Compressor")
            if "Pressure" not in step.entities:
                step.entities.append("Pressure")
                print(f"  ENTITIES: {step_id} -> Compressor, Pressure")
        
        # B4: Hot water pressing (UC2)
        if "pressing" in step_text and "water" in step_text:
            controllers_needed = ["WaterIsManager", "CoffeeBeansAreManager"]
            for controller in controllers_needed:
                if controller not in step.controllers:
                    step.controllers.append(controller)
            print(f"  SYSTEM BEHAVIOR: {step_id} -> Water pressing coordination")
            # Pressing entities
            entities_needed = ["HotWater", "CoffeeGrounds"]
            for entity in entities_needed:
                if entity not in step.entities:
                    step.entities.append(entity)
                    print(f"  ENTITY: {step_id} -> {entity}")
        
        return step
    
    def _identify_missing_output_controllers(self, steps: List[UCStep]) -> List[UCStep]:
        """Identifiziere fehlende Output-Controller für UC-Schritte wie B4, B5"""
        
        for step in steps:
            step_text = step.step_text.lower()
            
            # B4: Message Output Controller
            if ("output" in step_text and "message" in step_text) or \
               ("message" in step.objects and "user" in step.objects):
                if not step.controllers:  # Nur wenn noch keine Controller
                    step.controllers.append("UserNotificationManager")
                    print(f"PHASE 3 FIX: Added UserNotificationManager for {step.step_id}")
            
            # B5: User Delivery Controller  
            if ("present" in step_text and "user" in step_text) or \
               ("deliver" in step_text and "user" in step_text):
                if "UserDeliveryManager" not in step.controllers:
                    step.controllers.append("UserDeliveryManager")
                    print(f"PHASE 3 FIX: Added UserDeliveryManager for {step.step_id}")
            
            # Weitere Output-Patterns erkennen
            if "user" in step.objects and not step.controllers:
                if any(verb in step_text for verb in ["output", "present", "deliver", "provide", "give"]):
                    step.controllers.append("UserInterfaceManager")
                    print(f"PHASE 3 FIX: Added UserInterfaceManager for {step.step_id}")
        
        return steps

    def identify_orchestration_pattern(self, steps: List[UCStep]) -> Dict[str, List[str]]:
        """Identify orchestration patterns from parallel steps following UC-Methode.txt"""
        orchestration = {}
        
        # Group parallel steps (B2a, B2b, B2c, etc.)
        parallel_groups = {}
        for step in steps:
            if step.step_type == StepType.PARALLEL_STEP:
                # Extract base step number (B2 from B2a)
                base_step = re.sub(r'[a-z]$', '', step.step_id)
                if base_step not in parallel_groups:
                    parallel_groups[base_step] = []
                parallel_groups[base_step].append(step)
        
        # Create orchestration patterns
        for base_step, parallel_steps in parallel_groups.items():
            if len(parallel_steps) > 1:
                orchestrator_name = f"GetränkeOrchestrator"  # or domain-specific name
                controlled_managers = []
                
                for step in parallel_steps:
                    controlled_managers.extend(step.controllers)
                
                orchestration[orchestrator_name] = list(set(controlled_managers))
        
        return orchestration

    def generate_interactions(self, steps: List[UCStep], phase1_result: Phase1Result, 
                            phase2_result: Phase2Result) -> List[Interaction]:
        """Generate interaction patterns from analyzed steps"""
        interactions = []
        interaction_counter = 1
        
        for i, step in enumerate(steps):
            step_id = step.step_id
            
            # Trigger interactions (Step 3.1 from UC-Methode.txt)
            if step.step_type == StepType.TRIGGER_STEP:
                if any("zeit" in obj or "time" in obj or "uhr" in obj for obj in step.objects):
                    # Time-based trigger
                    interactions.append(Interaction(
                        interaction_id=f"INT_{interaction_counter:03d}",
                        interaction_type=InteractionType.TRIGGER,
                        source_object="Zeit",
                        target_object="ZeitManager",
                        description=f"Time trigger activates system at specified time",
                        step_reference=step_id
                    ))
                    interaction_counter += 1
                else:
                    # Actor-based trigger
                    for actor in phase1_result.actors:
                        if actor.type.value == "human":
                            interactions.append(Interaction(
                                interaction_id=f"INT_{interaction_counter:03d}",
                                interaction_type=InteractionType.TRIGGER,
                                source_object=actor.name,
                                target_object="HMI",
                                description=f"Human actor {actor.name} triggers use case",
                                step_reference=step_id
                            ))
                            interaction_counter += 1
            
            # Control flow interactions (Step 3.2 from UC-Methode.txt)
            elif step.step_type in [StepType.NORMAL_STEP, StepType.PARALLEL_STEP]:
                for controller in step.controllers:
                    # Previous step to current step control flow
                    if i > 0:
                        prev_step = steps[i-1]
                        if prev_step.controllers:
                            for prev_controller in prev_step.controllers:
                                interactions.append(Interaction(
                                    interaction_id=f"INT_{interaction_counter:03d}",
                                    interaction_type=InteractionType.CONTROL_FLOW,
                                    source_object=prev_controller,
                                    target_object=controller,
                                    description=f"Control flow from {prev_step.step_id} to {step_id}",
                                    step_reference=f"{prev_step.step_id}->{step_id}"
                                ))
                                interaction_counter += 1
        
        return interactions

    def identify_missing_preconditions(self, steps: List[UCStep], phase2_result: Phase2Result) -> List[str]:
        """Identify missing preconditions as described in UC-Methode.txt Step 3.1 and 3.4"""
        missing = []
        
        # Check for configuration entities used but not in preconditions
        configuration_objects = []
        for step in steps:
            for obj in step.objects:
                if any(config_word in obj for config_word in ["eingestellt", "set", "configured", "amount", "level"]):
                    configuration_objects.append(obj)
        
        # These should have corresponding preconditions
        for config_obj in configuration_objects:
            if "menge" in config_obj or "amount" in config_obj:
                missing.append(f"Kaffeestärke ist eingestellt ({config_obj})")
            elif "grad" in config_obj or "level" in config_obj:
                missing.append(f"Kaffee-Aroma ist eingestellt ({config_obj})")
            elif "uhrzeit" in config_obj or "time" in config_obj:
                missing.append(f"Uhrzeit für automatische Zubereitung ist eingestellt ({config_obj})")
        
        return missing

    def generate_controller_functions(self, steps: List[UCStep], phase2_result: Phase2Result) -> List[ControllerFunction]:
        """Generate controller functions from UC flow analysis"""
        functions = []
        
        for step in steps:
            for controller in step.controllers:
                if step.verb:
                    # Create function from verb and context
                    function_name = f"{step.verb.replace('iert', 'ieren').replace('et', 'en')}"
                    
                    functions.append(ControllerFunction(
                        controller_name=controller,
                        function_name=function_name,
                        source="uc_flow",
                        description=f"Function derived from UC step {step.step_id}: {step.step_text}",
                        step_references=[step.step_id]
                    ))
        
        return functions

    def perform_phase3_analysis(self, phase1_result: Phase1Result, phase2_result: Phase2Result, 
                               uc_main_flow: str) -> Phase3Result:
        """
        Complete Phase 3 analysis integrating Phase 1 and Phase 2 results
        
        Args:
            phase1_result: Result from Phase 1 analysis
            phase2_result: Result from Phase 2 analysis  
            uc_main_flow: Main flow text from use case
            
        Returns:
            Complete Phase3Result
        """
        # Step 1: Parse UC steps
        uc_steps = self.parse_uc_steps(uc_main_flow)
        
        # Step 2: Map steps to controllers from Phase 2
        uc_steps = self.map_steps_to_controllers(uc_steps, phase2_result)
        
        # Step 3: Identify orchestration patterns
        orchestration_pattern = self.identify_orchestration_pattern(uc_steps)
        
        # Step 4: Generate interactions
        interactions = self.generate_interactions(uc_steps, phase1_result, phase2_result)
        
        # Step 5: Generate controller functions from UC flow
        controller_functions = self.generate_controller_functions(uc_steps, phase2_result)
        
        # Step 6: Identify entity flows (transformations)
        entity_flows = self._identify_entity_flows(uc_steps, phase2_result)
        
        # Step 7: Identify missing preconditions
        missing_preconditions = self.identify_missing_preconditions(uc_steps, phase2_result)
        
        # Step 8: Generate summary
        summary = self._generate_phase3_summary(uc_steps, interactions, orchestration_pattern, missing_preconditions)
        
        return Phase3Result(
            phase1_result=phase1_result,
            phase2_result=phase2_result,
            uc_steps=uc_steps,
            interactions=interactions,
            controller_functions=controller_functions,
            entity_flows=entity_flows,
            orchestration_pattern=orchestration_pattern,
            missing_preconditions=missing_preconditions,
            summary=summary
        )

    def _identify_entity_flows(self, steps: List[UCStep], phase2_result: Phase2Result) -> List[EntityFlow]:
        """Identify entity flows between controllers following UC-Methode.txt rules"""
        entity_flows = []
        
        # Map transformations from Phase 2 to UC steps
        for step in steps:
            if step.verb and any("mahl" in step.verb or "grind" in step.verb for _ in [step.verb]):
                # Coffee grinding transformation
                entity_flows.append(EntityFlow(
                    entity_name="Kaffeemehl",
                    source_controller="KaffeeManager",
                    target_controller="FilterManager",
                    transformation="beans -> grinding -> ground coffee",
                    step_reference=step.step_id
                ))
            elif step.verb and any("erhitz" in step.verb or "heat" in step.verb for _ in [step.verb]):
                # Water heating transformation
                entity_flows.append(EntityFlow(
                    entity_name="Heißes Wasser",
                    source_controller="WasserManager",
                    target_controller="GetränkeManager",
                    transformation="water -> heating -> hot water",
                    step_reference=step.step_id
                ))
        
        return entity_flows

    def _generate_phase3_summary(self, steps: List[UCStep], interactions: List[Interaction], 
                                orchestration: Dict[str, List[str]], missing_preconditions: List[str]) -> str:
        """Generate summary of Phase 3 analysis"""
        summary_parts = [
            f"Analyzed {len(steps)} UC steps",
            f"Generated {len(interactions)} interactions",
            f"Identified {len(orchestration)} orchestration patterns",
            f"Found {len(missing_preconditions)} missing preconditions"
        ]
        
        return "; ".join(summary_parts)


def main():
    """Example usage with UC1 and UC2"""
    import json
    import os
    
    # Initialize analyzers
    domain_analyzer = DomainConfigurableAnalyzer()
    phase2_analyzer = BetriebsmittelAnalyzer(domain_analyzer)
    phase3_analyzer = InteraktionAnalyzer(domain_analyzer, phase2_analyzer)
    
    # Create output directory
    output_dir = "Zwischenprodukte"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test cases with main flows
    test_cases = [
        {
            "name": "UC1_Coffee",
            "capability_name": "Coffee Preparation",
            "uc_title": "Prepare Milk Coffee",
            "goal_text": "User can drink their milk coffee every morning at 7am",
            "actors_text": "User, Timer",
            "preconditions": [
                "Coffee beans are available in the system",
                "Water is available in the system", 
                "Milk is available in the system"
            ],
            "main_flow": """
B1 (trigger) System clock reaches set time of 7:00h (Radio clock)
B2a The system activates the water heater
B2b The system prepares filter
B2c The system grinds the set amount at the set grinding degree directly into the filter
B2d The system retrieves cup from storage container and places it under the filter
B3a The system begins brewing coffee with the set water amount into the cup
B3b The system adds milk to the cup
B4 The system outputs a message to user
B5 The system presents cup to user
B6 End UC
"""
        },
        {
            "name": "UC2_Espresso",
            "capability_name": "Coffee Preparation", 
            "uc_title": "Prepare Espresso",
            "goal_text": "User wants to drink an espresso",
            "actors_text": "User, Brew Timer",
            "preconditions": [
                "Coffee beans are available in the system",
                "Water is available in the system"
            ],
            "main_flow": """
B1 (trigger) User request a espresso
B2a The system activates the water heater
B2b The system prepares filter
B2c The system grinds the set amount at the set grinding degree directly into the filter
B2d The system retrieves cup from storage container and places it under the filter
B3 The system starts water compressor to generate appropriate water pressure for espresso
B4 The system begins pressing hot water through the coffee grounds
B5 The system outputs a message to user
B6 End UC
"""
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        print(f"\n=== {test_case['name']} Phase 3 Analysis ===")
        
        # Get Phase 1 and Phase 2 results
        phase1_result = domain_analyzer.perform_phase1_analysis(
            capability_name=test_case["capability_name"],
            uc_title=test_case["uc_title"],
            goal_text=test_case["goal_text"],
            actors_text=test_case["actors_text"]
        )
        
        phase2_result = phase2_analyzer.perform_phase2_analysis(
            phase1_result, test_case["preconditions"]
        )
        
        # Perform Phase 3 analysis
        phase3_result = phase3_analyzer.perform_phase3_analysis(
            phase1_result, phase2_result, test_case["main_flow"]
        )
        
        print(f"Summary: {phase3_result.summary}")
        print(f"UC Steps: {len(phase3_result.uc_steps)}")
        print(f"Interactions: {len(phase3_result.interactions)}")
        print(f"Orchestration patterns: {len(phase3_result.orchestration_pattern)}")
        
        print(f"\nUC Steps Analysis:")
        for step in phase3_result.uc_steps:
            print(f"  {step.step_id}: {step.step_text[:60]}...")
            print(f"    Type: {step.step_type.value}, Verb: {step.verb}")
            print(f"    Objects: {step.objects[:3]}...")  # Show first 3 objects
            print(f"    Controllers: {step.controllers}")
        
        print(f"\nInteractions:")
        for interaction in phase3_result.interactions[:5]:  # Show first 5
            print(f"  {interaction.interaction_id}: {interaction.source_object} -> {interaction.target_object}")
            print(f"    Type: {interaction.interaction_type.value}, Step: {interaction.step_reference}")
        
        print(f"\nOrchestration patterns:")
        for orchestrator, managers in phase3_result.orchestration_pattern.items():
            print(f"  {orchestrator} -> {managers}")
        
        if phase3_result.missing_preconditions:
            print(f"\nMissing preconditions:")
            for missing in phase3_result.missing_preconditions:
                print(f"  - {missing}")
        
        # Convert result to dictionary for JSON serialization
        result_dict = {
            "phase1_summary": phase1_result.context_summary,
            "phase2_summary": phase2_result.summary,
            "phase3_summary": phase3_result.summary,
            "uc_steps": [
                {
                    "step_id": step.step_id,
                    "step_text": step.step_text,
                    "step_type": step.step_type.value,
                    "verb": step.verb,
                    "objects": step.objects,
                    "controllers": step.controllers,
                    "entities": step.entities
                }
                for step in phase3_result.uc_steps
            ],
            "interactions": [
                {
                    "interaction_id": interaction.interaction_id,
                    "interaction_type": interaction.interaction_type.value,
                    "source_object": interaction.source_object,
                    "target_object": interaction.target_object,
                    "data_object": interaction.data_object,
                    "description": interaction.description,
                    "step_reference": interaction.step_reference
                }
                for interaction in phase3_result.interactions
            ],
            "orchestration_pattern": phase3_result.orchestration_pattern,
            "controller_functions": [
                {
                    "controller_name": func.controller_name,
                    "function_name": func.function_name,
                    "source": func.source,
                    "description": func.description,
                    "step_references": func.step_references
                }
                for func in phase3_result.controller_functions
            ],
            "entity_flows": [
                {
                    "entity_name": flow.entity_name,
                    "source_controller": flow.source_controller,
                    "target_controller": flow.target_controller,
                    "transformation": flow.transformation,
                    "step_reference": flow.step_reference
                }
                for flow in phase3_result.entity_flows
            ],
            "missing_preconditions": phase3_result.missing_preconditions
        }
        
        results[test_case["name"]] = result_dict
        
        # Save individual result
        output_file = os.path.join(output_dir, f"{test_case['name']}_phase3_analysis.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\nPhase 3 results saved to {output_file}")
    
    # Save combined results
    combined_output_file = os.path.join(output_dir, "all_phase3_analyses.json")
    with open(combined_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nAll Phase 3 results saved to {combined_output_file}")


if __name__ == "__main__":
    main()