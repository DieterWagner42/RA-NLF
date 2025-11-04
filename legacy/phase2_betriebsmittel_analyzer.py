"""
Phase 2: Betriebsmittel-Analyse (Resources Analysis) - UC-Methode.txt Implementation
Analyzes resources from preconditions using the 3-step schema:
1. Input determination (how resources enter the system)
2. Storage and Processing (manager controllers and context-derived functions)  
3. Output determination (how resources or products leave the system)
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
    MANAGER_CONTROLLER = "manager_controller"  
    OUTPUT_BOUNDARY = "output_boundary"
    ENTITY = "entity"


@dataclass
class ResourceFunction:
    name: str
    source: str  # "explicit" from UC text, "implicit" from context
    description: str


@dataclass
class ResourceObject:
    name: str
    type: ResourceType
    resource_name: str  # The original resource this relates to
    functions: List[ResourceFunction] = field(default_factory=list)
    context_derived: bool = False
    domain_specific: bool = False


@dataclass
class BetriebsmittelAnalysis:
    resource_name: str
    input_boundary: Optional[ResourceObject] = None
    manager_controller: Optional[ResourceObject] = None
    output_boundary: Optional[ResourceObject] = None
    entities: List[ResourceObject] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)


@dataclass
class Phase2Result:
    phase1_result: Phase1Result
    resource_analyses: List[BetriebsmittelAnalysis]
    all_objects: Dict[str, List[ResourceObject]]
    summary: str


class BetriebsmittelAnalyzer:
    """
    Phase 2 implementation: Analyzes resources from preconditions using 3-step schema
    """
    
    def __init__(self, domain_analyzer: DomainConfigurableAnalyzer = None):
        """
        Initialize with optional domain analyzer for context
        """
        self.domain_analyzer = domain_analyzer or DomainConfigurableAnalyzer()
        self.nlp = self.domain_analyzer.nlp
        
        # Pattern for recognizing resources in preconditions
        self.resource_patterns = [
            r"(\w+(?:\s+\w+)*)\s+(?:is|are)\s+(?:available|present|loaded|ready)\s+in\s+(?:the\s+)?system",
            r"(\w+(?:\s+\w+)*)\s+(?:is|are)\s+(?:in|within)\s+(?:the\s+)?system",
            r"system\s+(?:has|contains)\s+(\w+(?:\s+\w+)*)",
            r"(\w+(?:\s+\w+)*)\s+(?:exists?|available)"
        ]
        
        # Context-based transformation knowledge
        self.transformation_contexts = {
            "beverage_preparation": {
                "coffee": {
                    "beans": ("grinding", "ground coffee"),
                    "water": ("heating", "hot water"),
                    "milk": ("steaming", "steamed milk"),
                    "ground": ("brewing", "brewed coffee")
                },
                "tea": {
                    "leaves": ("steeping", "brewed tea"),
                    "water": ("heating", "hot water")
                }
            },
            "rocket_science": {
                "rocket": {
                    "fuel": ("combustion", "thrust"),
                    "oxidizer": ("combustion", "exhaust"),
                    "trajectory": ("calculation", "guidance commands")
                }
            },
            "nuclear": {
                "reactor": {
                    "uranium": ("fission", "heat"),
                    "coolant": ("circulation", "heat transfer"),
                    "control_rods": ("positioning", "neutron absorption")
                }
            },
            "robotics": {
                "manufacturing": {
                    "components": ("assembly", "finished product"),
                    "materials": ("processing", "shaped parts"),
                    "sensors": ("data_collection", "sensor data")
                }
            },
            "automotive": {
                "vehicle": {
                    "fuel": ("combustion", "mechanical energy"),
                    "battery": ("discharge", "electrical energy"),
                    "sensor_data": ("processing", "control signals")
                }
            },
            "aerospace": {
                "aircraft": {
                    "fuel": ("combustion", "thrust"),
                    "flight_data": ("processing", "navigation commands"),
                    "weather_data": ("analysis", "route adjustments")
                }
            }
        }

    def extract_resources_from_preconditions(self, preconditions: List[str]) -> List[str]:
        """
        Extract resources from preconditions using pattern matching
        
        Args:
            preconditions: List of precondition strings
            
        Returns:
            List of identified resource names
        """
        resources = []
        
        for precond in preconditions:
            precond_lower = precond.lower().strip()
            
            # Try each pattern
            for pattern in self.resource_patterns:
                matches = re.findall(pattern, precond_lower, re.IGNORECASE)
                for match in matches:
                    # Clean up the match
                    resource = match.strip()
                    if len(resource) > 2 and resource not in resources:
                        resources.append(resource)
        
        return resources

    def analyze_single_resource(self, resource_name: str, domain_config: Optional[DomainConfig] = None, 
                               phase1_result: Optional[Phase1Result] = None) -> BetriebsmittelAnalysis:
        """
        Analyze a single resource using the 3-step schema, integrating Phase 1 results
        
        Args:
            resource_name: Name of the resource to analyze
            domain_config: Domain configuration for context
            phase1_result: Phase 1 analysis results for enhanced integration
            
        Returns:
            Complete BetriebsmittelAnalysis for the resource
        """
        analysis = BetriebsmittelAnalysis(resource_name=resource_name)
        
        # Step 2.1: Input determination (enhanced with Phase 1 boundaries)
        analysis.input_boundary = self._determine_input(resource_name, domain_config, phase1_result)
        
        # Step 2.2: Storage and Processing (enhanced with Phase 1 controllers)
        analysis.manager_controller = self._determine_manager_controller(resource_name, domain_config, phase1_result)
        
        # Step 2.3: Output determination (enhanced with Phase 1 constraints)
        analysis.output_boundary = self._determine_output(resource_name, domain_config, phase1_result)
        
        # Additional entities (transformations)
        analysis.entities = self._identify_transformation_entities(resource_name, domain_config)
        analysis.transformations = self._identify_transformations(resource_name, domain_config)
        
        return analysis

    def _determine_input(self, resource_name: str, domain_config: Optional[DomainConfig], 
                        phase1_result: Optional[Phase1Result] = None) -> ResourceObject:
        """
        Step 2.1: Determine how the resource enters the system, integrating Phase 1 boundaries
        """
        # Normalize resource name
        normalized_resource = self._normalize_resource_name(resource_name)
        
        # Check if Phase 1 identified related boundaries
        existing_boundaries = []
        if phase1_result and phase1_result.goal_analysis.identified_boundaries:
            # Check if any Phase 1 boundaries relate to this resource
            for boundary in phase1_result.goal_analysis.identified_boundaries:
                if any(word in boundary.lower() for word in resource_name.lower().split()):
                    existing_boundaries.append(boundary)
        
        # Determine input type based on Phase 1 actors and resource characteristics
        is_manual_input = self._requires_manual_input(resource_name, domain_config, phase1_result)
        is_consumable = self._is_consumable_resource(resource_name, domain_config)
        
        if is_consumable or is_manual_input:
            # Manual input for consumables or user-driven resources
            if existing_boundaries:
                boundary_name = f"{normalized_resource} Input (extends {', '.join(existing_boundaries)})"
            else:
                boundary_name = f"{normalized_resource} Input"
            
            functions = [
                ResourceFunction(
                    name="accept_input",
                    source="implicit",
                    description=f"Accept {resource_name} from external source"
                ),
                ResourceFunction(
                    name="validate_input",
                    source="implicit", 
                    description=f"Validate {resource_name} quality and quantity"
                )
            ]
            
            # Add human actor considerations from Phase 1
            if phase1_result and any(actor.type.value == "human" for actor in phase1_result.actors):
                functions.append(ResourceFunction(
                    name="provide_user_feedback",
                    source="phase1_integration",
                    description=f"Provide feedback to human actors about {resource_name} status"
                ))
        else:
            # Automatic or system-generated input
            boundary_name = f"{normalized_resource} Interface"
            functions = [
                ResourceFunction(
                    name="receive_data",
                    source="implicit",
                    description=f"Receive {resource_name} from system"
                )
            ]
        
        return ResourceObject(
            name=boundary_name,
            type=ResourceType.INPUT_BOUNDARY,
            resource_name=resource_name,
            functions=functions,
            context_derived=True
        )

    def _determine_manager_controller(self, resource_name: str, domain_config: Optional[DomainConfig], 
                                     phase1_result: Optional[Phase1Result] = None) -> ResourceObject:
        """
        Step 2.2: Create Manager Controller with context-derived functions, integrating Phase 1 controllers
        """
        normalized_resource = self._normalize_resource_name(resource_name)
        
        # Check if Phase 1 identified related controllers
        related_controllers = []
        if phase1_result and phase1_result.goal_analysis.identified_controllers:
            for controller in phase1_result.goal_analysis.identified_controllers:
                if any(word in controller.lower() for word in resource_name.lower().split()) or \
                   any(word in resource_name.lower() for word in controller.lower().split()):
                    related_controllers.append(controller)
        
        # Name controller considering Phase 1 context
        if related_controllers:
            controller_name = f"{normalized_resource}Manager (integrates with {', '.join(related_controllers)})"
        else:
            controller_name = f"{normalized_resource}Manager"
        
        # Basic functions every manager has
        functions = [
            ResourceFunction(
                name=f"store_{normalized_resource.lower()}",
                source="implicit",
                description=f"Store and manage {resource_name}"
            ),
            ResourceFunction(
                name=f"monitor_{normalized_resource.lower()}_level",
                source="implicit", 
                description=f"Monitor available quantity of {resource_name}"
            ),
            ResourceFunction(
                name=f"provide_{normalized_resource.lower()}",
                source="implicit",
                description=f"Provide {resource_name} when requested"
            )
        ]
        
        # Add temporal functions if Phase 1 identified temporal requirements
        if phase1_result and phase1_result.goal_analysis.temporal_requirements:
            functions.append(ResourceFunction(
                name=f"schedule_{normalized_resource.lower()}_usage",
                source="phase1_integration",
                description=f"Schedule {resource_name} usage based on temporal requirements: {', '.join(phase1_result.goal_analysis.temporal_requirements)}"
            ))
        
        # Add context-derived functions
        context_functions = self._derive_context_functions(resource_name, domain_config)
        functions.extend(context_functions)
        
        return ResourceObject(
            name=controller_name,
            type=ResourceType.MANAGER_CONTROLLER,
            resource_name=resource_name,
            functions=functions,
            context_derived=True,
            domain_specific=bool(context_functions)
        )

    def _determine_output(self, resource_name: str, domain_config: Optional[DomainConfig], 
                         phase1_result: Optional[Phase1Result] = None) -> Optional[ResourceObject]:
        """
        Step 2.3: Determine how resource/products leave the system, considering Phase 1 constraints
        """
        # Check if this resource produces waste/byproducts
        waste_products = self._identify_waste_products(resource_name, domain_config)
        
        # Consider Phase 1 solution constraints that might affect output handling
        constraint_considerations = []
        if phase1_result and phase1_result.uc_title_analysis.solution_constraints:
            constraint_considerations = phase1_result.uc_title_analysis.solution_constraints
        
        if waste_products:
            normalized_resource = self._normalize_resource_name(resource_name)
            boundary_name = f"{normalized_resource} Waste Output"
            
            functions = [
                ResourceFunction(
                    name="dispose_waste",
                    source="implicit",
                    description=f"Dispose of waste products from {resource_name}"
                ),
                ResourceFunction(
                    name="manage_waste",
                    source="implicit", 
                    description=f"Manage waste disposal processes"
                )
            ]
            
            # Add constraint-based output functions from Phase 1
            if constraint_considerations:
                functions.append(ResourceFunction(
                    name="apply_constraints",
                    source="phase1_integration",
                    description=f"Apply solution constraints to output: {', '.join(constraint_considerations)}"
                ))
            
            return ResourceObject(
                name=boundary_name,
                type=ResourceType.OUTPUT_BOUNDARY,
                resource_name=resource_name,
                functions=functions,
                context_derived=True
            )
        
        # If resource becomes part of end product, no separate output boundary
        return None

    def _identify_transformation_entities(self, resource_name: str, domain_config: Optional[DomainConfig]) -> List[ResourceObject]:
        """
        Identify entities created through resource transformation
        """
        entities = []
        transformations = self._identify_transformations(resource_name, domain_config)
        
        for transformation in transformations:
            # Create entity for transformed product
            if " -> " in transformation:
                parts = transformation.split(" -> ")
                if len(parts) == 2:
                    output_product = parts[1]
                    entity = ResourceObject(
                        name=self._normalize_resource_name(output_product),
                        type=ResourceType.ENTITY,
                        resource_name=resource_name,
                        functions=[],
                        context_derived=True
                    )
                    entities.append(entity)
        
        return entities

    def _identify_transformations(self, resource_name: str, domain_config: Optional[DomainConfig]) -> List[str]:
        """
        Identify possible transformations for the resource based on domain context
        """
        transformations = []
        
        if not domain_config:
            return transformations
        
        domain_name = domain_config.domain_name
        resource_lower = resource_name.lower()
        
        # Check transformation contexts
        if domain_name in self.transformation_contexts:
            domain_transforms = self.transformation_contexts[domain_name]
            
            # Find relevant transformations - FIX: Use validated input to avoid RAW dependency violations
            for category, transforms in domain_transforms.items():
                for input_resource, (process, output) in transforms.items():
                    if input_resource in resource_lower or resource_lower in input_resource:
                        # RUP-COMPLIANT FIX: Manager processes validated entities, not raw
                        transformation = f"validated {input_resource} -> {process} -> {output}"
                        transformations.append(transformation)
        
        return transformations

    def _derive_context_functions(self, resource_name: str, domain_config: Optional[DomainConfig]) -> List[ResourceFunction]:
        """
        Derive additional functions based on domain context and resource properties
        """
        functions = []
        resource_lower = resource_name.lower()
        
        if not domain_config:
            return functions
        
        # Domain-specific function derivation
        if domain_config.domain_name == "beverage_preparation":
            if "coffee" in resource_lower or "bean" in resource_lower:
                functions.append(ResourceFunction(
                    name="grind_coffee",
                    source="context",
                    description="Grind coffee beans to required consistency"
                ))
            elif "water" in resource_lower:
                functions.append(ResourceFunction(
                    name="heat_water",
                    source="context", 
                    description="Heat water to brewing temperature"
                ))
            elif "milk" in resource_lower:
                functions.extend([
                    ResourceFunction(
                        name="cool_milk",
                        source="context",
                        description="Keep milk at safe temperature"
                    ),
                    ResourceFunction(
                        name="steam_milk",
                        source="context",
                        description="Steam milk for coffee drinks"
                    )
                ])
        
        elif domain_config.domain_name == "rocket_science":
            if "fuel" in resource_lower:
                functions.extend([
                    ResourceFunction(
                        name="maintain_fuel_pressure",
                        source="context",
                        description="Maintain fuel at correct pressure"
                    ),
                    ResourceFunction(
                        name="control_fuel_flow",
                        source="context",
                        description="Control fuel flow rate to engines"
                    )
                ])
            elif "telemetry" in resource_lower or "data" in resource_lower:
                functions.extend([
                    ResourceFunction(
                        name="process_telemetry",
                        source="context",
                        description="Process real-time telemetry data"
                    ),
                    ResourceFunction(
                        name="validate_data",
                        source="context",
                        description="Validate data integrity and accuracy"
                    )
                ])
        
        elif domain_config.domain_name == "nuclear":
            if "uranium" in resource_lower or "fuel" in resource_lower:
                functions.extend([
                    ResourceFunction(
                        name="monitor_radiation",
                        source="context",
                        description="Monitor radiation levels continuously"
                    ),
                    ResourceFunction(
                        name="control_reaction",
                        source="context", 
                        description="Control nuclear reaction rate"
                    )
                ])
            elif "coolant" in resource_lower:
                functions.extend([
                    ResourceFunction(
                        name="circulate_coolant",
                        source="context",
                        description="Circulate coolant through reactor"
                    ),
                    ResourceFunction(
                        name="monitor_temperature",
                        source="context",
                        description="Monitor coolant temperature"
                    )
                ])
        
        elif domain_config.domain_name == "robotics":
            if "component" in resource_lower or "part" in resource_lower:
                functions.extend([
                    ResourceFunction(
                        name="position_components",
                        source="context",
                        description="Position components for assembly"
                    ),
                    ResourceFunction(
                        name="verify_quality",
                        source="context",
                        description="Verify component quality and specifications"
                    )
                ])
        
        # Add hygiene/safety functions for appropriate resources
        if self._requires_hygiene(resource_name, domain_config):
            functions.extend([
                ResourceFunction(
                    name="clean_equipment",
                    source="context", 
                    description=f"Clean equipment used with {resource_name}"
                ),
                ResourceFunction(
                    name="sanitize_surfaces",
                    source="context",
                    description="Sanitize surfaces in contact with resource"
                )
            ])
        
        return functions

    def _normalize_resource_name(self, resource_name: str) -> str:
        """Normalize resource name for consistent naming"""
        # Convert to title case and remove articles
        words = resource_name.lower().split()
        filtered_words = [w for w in words if w not in ['the', 'a', 'an']]
        return ''.join(word.capitalize() for word in filtered_words)

    def _is_consumable_resource(self, resource_name: str, domain_config: Optional[DomainConfig]) -> bool:
        """Check if resource is consumable (needs manual input)"""
        consumable_indicators = [
            "bean", "beans", "water", "milk", "sugar", "fuel", "oil", "gas",
            "component", "parts", "material", "ingredient", "chemical"
        ]
        resource_lower = resource_name.lower()
        return any(indicator in resource_lower for indicator in consumable_indicators)

    def _identify_waste_products(self, resource_name: str, domain_config: Optional[DomainConfig]) -> List[str]:
        """Identify waste products from resource usage"""
        waste_products = []
        resource_lower = resource_name.lower()
        
        # Domain-specific waste products
        if domain_config and domain_config.domain_name == "beverage_preparation":
            if "coffee" in resource_lower or "bean" in resource_lower:
                waste_products.extend(["coffee grounds", "used filter"])
            elif "tea" in resource_lower:
                waste_products.extend(["used tea leaves", "tea bag"])
        
        elif domain_config and domain_config.domain_name == "nuclear":
            if "uranium" in resource_lower or "fuel" in resource_lower:
                waste_products.extend(["spent fuel", "radioactive waste"])
        
        return waste_products

    def _requires_hygiene(self, resource_name: str, domain_config: Optional[DomainConfig]) -> bool:
        """Check if resource requires hygiene functions"""
        if not domain_config:
            return False
        
        hygiene_domains = ["beverage_preparation", "food_preparation", "healthcare"]
        food_contact_resources = ["coffee", "tea", "milk", "water", "beans", "ingredient"]
        
        if domain_config.domain_name in hygiene_domains:
            resource_lower = resource_name.lower()
            return any(indicator in resource_lower for indicator in food_contact_resources)
        
        return False

    def _requires_manual_input(self, resource_name: str, domain_config: Optional[DomainConfig], 
                              phase1_result: Optional[Phase1Result]) -> bool:
        """Determine if resource requires manual input based on Phase 1 actor analysis"""
        if not phase1_result:
            return False
        
        # Check if there are human actors who would handle this resource
        human_actors = [actor for actor in phase1_result.actors if actor.type.value == "human"]
        
        # If there are human actors and input requirements from Phase 1, likely manual
        if human_actors and phase1_result.goal_analysis.input_requirements:
            return True
        
        # Check if resource relates to user input from Phase 1 boundaries
        if phase1_result.goal_analysis.identified_boundaries:
            for boundary in phase1_result.goal_analysis.identified_boundaries:
                if "user" in boundary.lower() or "input" in boundary.lower():
                    if any(word in resource_name.lower() for word in boundary.lower().split()):
                        return True
        
        return False

    def _derive_context_resources(self, phase1_result: Phase1Result, existing_resources: List[str]) -> List[str]:
        """
        Derive implicit resources from domain context that are not in preconditions
        
        DOMAIN RULE: Getränke benötigen immer einen Behälter!
        """
        context_resources = []
        domain = phase1_result.capability_context.domain
        
        # Rule for beverage_preparation domain: beverages need containers
        if domain == "beverage_preparation":
            # Check if we have beverage ingredients (coffee, tea, etc.)
            has_beverage_ingredients = any(
                beverage_term in " ".join(existing_resources).lower() 
                for beverage_term in ["coffee", "tea", "beans", "leaves", "milk", "water"]
            )
            
            # Check if container already in resources
            has_container = any(
                container_term in " ".join(existing_resources).lower()
                for container_term in ["cup", "tasse", "mug", "glass", "container"]
            )
            
            # Apply rule: if beverage ingredients exist but no container -> derive cup
            if has_beverage_ingredients and not has_container:
                context_resources.append("cup")
                print(f"CONTEXT RULE APPLIED: Beverages need containers -> derived 'cup'")
        
        return context_resources

    def perform_phase2_analysis(self, phase1_result: Phase1Result, preconditions: List[str]) -> Phase2Result:
        """
        Complete Phase 2 analysis using the 3-step schema
        
        Args:
            phase1_result: Result from Phase 1 analysis
            preconditions: List of precondition strings from use case
            
        Returns:
            Complete Phase2Result
        """
        # Extract resources from preconditions
        resources = self.extract_resources_from_preconditions(preconditions)
        
        # ADD: Derive context-based resources from domain knowledge
        context_resources = self._derive_context_resources(phase1_result, resources)
        all_resources = resources + context_resources
        
        # Analyze each resource
        resource_analyses = []
        all_objects = {
            "input_boundaries": [],
            "manager_controllers": [], 
            "output_boundaries": [],
            "entities": []
        }
        
        for resource in all_resources:
            analysis = self.analyze_single_resource(
                resource, 
                phase1_result.capability_context.domain_config,
                phase1_result
            )
            resource_analyses.append(analysis)
            
            # Collect all objects
            if analysis.input_boundary:
                all_objects["input_boundaries"].append(analysis.input_boundary)
            if analysis.manager_controller:
                all_objects["manager_controllers"].append(analysis.manager_controller)
            if analysis.output_boundary:
                all_objects["output_boundaries"].append(analysis.output_boundary)
            all_objects["entities"].extend(analysis.entities)
        
        # Generate summary
        summary = self._generate_phase2_summary(resource_analyses, phase1_result)
        
        return Phase2Result(
            phase1_result=phase1_result,
            resource_analyses=resource_analyses,
            all_objects=all_objects,
            summary=summary
        )

    def _generate_phase2_summary(self, analyses: List[BetriebsmittelAnalysis], phase1_result: Phase1Result) -> str:
        """Generate summary of Phase 2 analysis"""
        total_resources = len(analyses)
        total_controllers = sum(1 for a in analyses if a.manager_controller)
        total_boundaries = sum(1 for a in analyses if a.input_boundary or a.output_boundary)
        total_transformations = sum(len(a.transformations) for a in analyses)
        
        summary_parts = [
            f"Analyzed {total_resources} resources from preconditions",
            f"Created {total_controllers} Manager Controllers",
            f"Identified {total_boundaries} Boundary Objects",
            f"Found {total_transformations} context-based transformations",
            f"Domain: {phase1_result.capability_context.domain}"
        ]
        
        return "; ".join(summary_parts)


def main():
    """Example usage with UC1 and UC2"""
    import json
    import os
    
    # Initialize analyzers
    domain_analyzer = DomainConfigurableAnalyzer()
    phase2_analyzer = BetriebsmittelAnalyzer(domain_analyzer)
    
    # Create output directory
    output_dir = "Zwischenprodukte"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test cases with preconditions
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
            ]
        },
        {
            "name": "UC3_Rocket_Launch",
            "capability_name": "Rocket Launch",
            "uc_title": "Execute Satellite Launch",
            "goal_text": "Mission control can launch satellite into orbit at scheduled time",
            "actors_text": "Mission Control, Launch Sequencer, Range Safety System",
            "preconditions": [
                "Rocket fuel is available in the system",
                "Oxidizer is available in the system",
                "Telemetry data is available in the system",
                "Launch trajectory is calculated"
            ]
        },
        {
            "name": "UC4_Nuclear_Shutdown", 
            "capability_name": "Nuclear Reactor Control",
            "uc_title": "Emergency Reactor Shutdown",
            "goal_text": "Reactor can be safely shut down in emergency",
            "actors_text": "Operator, Reactor Protection System",
            "preconditions": [
                "Uranium fuel is present in reactor",
                "Coolant is available in system",
                "Control rods are positioned correctly"
            ]
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        print(f"\n=== {test_case['name']} Phase 2 Analysis ===")
        
        # First get Phase 1 result
        phase1_result = domain_analyzer.perform_phase1_analysis(
            capability_name=test_case["capability_name"],
            uc_title=test_case["uc_title"],
            goal_text=test_case["goal_text"], 
            actors_text=test_case["actors_text"]
        )
        
        # Then perform Phase 2 analysis
        phase2_result = phase2_analyzer.perform_phase2_analysis(
            phase1_result, 
            test_case["preconditions"]
        )
        
        print(f"Summary: {phase2_result.summary}")
        print(f"Resources analyzed: {len(phase2_result.resource_analyses)}")
        
        for analysis in phase2_result.resource_analyses:
            print(f"\n  Resource: {analysis.resource_name}")
            if analysis.input_boundary:
                print(f"    Input: {analysis.input_boundary.name}")
            if analysis.manager_controller:
                print(f"    Manager: {analysis.manager_controller.name}")
                print(f"    Functions: {len(analysis.manager_controller.functions)}")
            if analysis.output_boundary:
                print(f"    Output: {analysis.output_boundary.name}")
            if analysis.transformations:
                print(f"    Transformations: {analysis.transformations}")
        
        print(f"\n  Total Objects Created:")
        print(f"    Input Boundaries: {len(phase2_result.all_objects['input_boundaries'])}")
        print(f"    Manager Controllers: {len(phase2_result.all_objects['manager_controllers'])}")
        print(f"    Output Boundaries: {len(phase2_result.all_objects['output_boundaries'])}")
        print(f"    Entities: {len(phase2_result.all_objects['entities'])}")
        
        # Convert result to dictionary for JSON serialization
        result_dict = {
            "phase1_summary": phase1_result.context_summary,
            "phase2_summary": phase2_result.summary,
            "resource_analyses": [
                {
                    "resource_name": analysis.resource_name,
                    "input_boundary": {
                        "name": analysis.input_boundary.name,
                        "functions": [{"name": f.name, "source": f.source, "description": f.description} 
                                    for f in analysis.input_boundary.functions]
                    } if analysis.input_boundary else None,
                    "manager_controller": {
                        "name": analysis.manager_controller.name,
                        "functions": [{"name": f.name, "source": f.source, "description": f.description}
                                    for f in analysis.manager_controller.functions],
                        "domain_specific": analysis.manager_controller.domain_specific
                    } if analysis.manager_controller else None,
                    "output_boundary": {
                        "name": analysis.output_boundary.name,
                        "functions": [{"name": f.name, "source": f.source, "description": f.description}
                                    for f in analysis.output_boundary.functions]
                    } if analysis.output_boundary else None,
                    "entities": [{"name": entity.name, "type": entity.type.value} 
                               for entity in analysis.entities],
                    "transformations": analysis.transformations
                }
                for analysis in phase2_result.resource_analyses
            ],
            "all_objects": {
                "input_boundaries": [obj.name for obj in phase2_result.all_objects["input_boundaries"]],
                "manager_controllers": [obj.name for obj in phase2_result.all_objects["manager_controllers"]],
                "output_boundaries": [obj.name for obj in phase2_result.all_objects["output_boundaries"]], 
                "entities": [obj.name for obj in phase2_result.all_objects["entities"]]
            }
        }
        
        results[test_case["name"]] = result_dict
        
        # Save individual result
        output_file = os.path.join(output_dir, f"{test_case['name']}_phase2_analysis.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    # Save combined results  
    combined_output_file = os.path.join(output_dir, "all_phase2_analyses.json")
    with open(combined_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nPhase 2 results saved to {output_dir}/ directory")


if __name__ == "__main__":
    main()