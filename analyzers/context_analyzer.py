"""
Generic Context Analyzer for UC Robustness Analysis - Phase 1
Implementation of UC-Methode.txt Phase 1: Context building and understanding

Uses spaCy medium model for NLP analysis across any domain.
"""

import spacy
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class ActorType(Enum):
    HUMAN = "human"
    NON_HUMAN = "non_human"


@dataclass
class ActorInfo:
    name: str
    type: ActorType
    description: Optional[str] = None


@dataclass
class CapabilityContext:
    capability_name: str
    domain: str
    domain_knowledge: List[str]
    excluded_aspects: List[str]
    related_capabilities: List[str]


@dataclass
class UCTitleAnalysis:
    uc_title: str
    solution_constraints: List[str]
    possible_variants: List[str]
    scope_restrictions: List[str]


@dataclass
class GoalAnalysis:
    goal_text: str
    temporal_requirements: List[str]
    input_requirements: List[str]
    identified_controllers: List[str]
    identified_boundaries: List[str]


@dataclass
class Phase1Result:
    capability_context: CapabilityContext
    uc_title_analysis: UCTitleAnalysis
    goal_analysis: GoalAnalysis
    actors: List[ActorInfo]
    context_summary: str


class ContextAnalyzer:
    """
    Generic context analyzer implementing Phase 1 of UC-Methode.txt
    """
    
    def __init__(self, model_name: str = "en_core_web_md"):
        """
        Initialize with spaCy medium model
        
        Args:
            model_name: spaCy model name (default: en_core_web_md)
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise RuntimeError(f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}")
        
        # Domain knowledge base - expandable for any domain
        self.domain_patterns = {
            "food_preparation": ["cooking", "preparation", "recipe", "ingredient", "kitchen"],
            "beverage_preparation": ["brewing", "coffee", "tea", "drink", "beverage", "liquid"],
            "manufacturing": ["production", "assembly", "manufacturing", "fabrication"],
            "transportation": ["movement", "transport", "delivery", "logistics", "vehicle"],
            "communication": ["messaging", "communication", "transmission", "signal"],
            "financial": ["payment", "transaction", "financial", "money", "banking"],
            "healthcare": ["treatment", "medical", "patient", "diagnosis", "therapy"],
            "education": ["learning", "teaching", "training", "education", "instruction"],
            "rocket_science": ["rocket", "launch", "spacecraft", "propulsion", "orbital", "space", "mission", "trajectory"],
            "aerospace": ["aircraft", "flight", "aviation", "aerodynamics", "navigation", "altitude", "pilot"],
            "automotive": ["vehicle", "engine", "driving", "car", "automotive", "brake", "acceleration"],
            "robotics": ["robot", "automation", "actuator", "sensor", "manipulation", "autonomous"],
            "energy": ["power", "energy", "generation", "grid", "electrical", "renewable", "solar", "wind"],
            "telecommunications": ["network", "wireless", "cellular", "satellite", "antenna", "frequency"],
            "nuclear": ["nuclear", "reactor", "radiation", "uranium", "fusion", "fission", "containment"],
            "chemical_processing": ["chemical", "reaction", "catalyst", "process", "compound", "synthesis"],
            "biotechnology": ["bio", "genetic", "dna", "protein", "cell", "molecular", "organism"],
            "mining": ["mining", "extraction", "ore", "drilling", "excavation", "mineral"],
            "maritime": ["ship", "vessel", "marine", "navigation", "port", "cargo", "sailing"],
            "agriculture": ["farming", "crop", "harvest", "irrigation", "livestock", "agricultural"],
            "construction": ["building", "construction", "concrete", "structural", "architecture"],
            "security": ["security", "surveillance", "protection", "access", "threat", "monitoring"],
            "gaming": ["game", "player", "score", "level", "character", "simulation"],
            "logistics": ["warehouse", "inventory", "supply", "distribution", "storage", "tracking"]
        }
        
        # Temporal indicators
        self.temporal_patterns = [
            r"\d{1,2}:\d{2}[ap]?m?",  # 7:00am, 14:30
            r"\d{1,2}[ap]m",          # 7am, 2pm
            r"at \d+",                # at 7
            r"every \w+",             # every morning
            r"daily|weekly|monthly",
            r"automatically",
            r"scheduled",
            r"timer"
        ]
        
        # Non-human actor indicators
        self.non_human_indicators = [
            "system", "clock", "timer", "sensor", "device", "machine", 
            "automatic", "scheduler", "monitor", "detector", "controller"
        ]

    def analyze_capability(self, capability_name: str) -> CapabilityContext:
        """
        Step 1.1: Analyze capability and identify domain
        
        Args:
            capability_name: Name of the capability (e.g., "Coffee Preparation")
            
        Returns:
            CapabilityContext with domain identification and knowledge
        """
        doc = self.nlp(capability_name.lower())
        
        # Extract key terms
        key_terms = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        # Identify domain
        domain = self._identify_domain(key_terms)
        
        # Build domain knowledge
        domain_knowledge = self._build_domain_knowledge(domain, key_terms)
        
        # Identify excluded aspects (heuristics)
        excluded_aspects = self._identify_excluded_aspects(domain, key_terms)
        
        # Find related capabilities
        related_capabilities = self._find_related_capabilities(domain, key_terms)
        
        return CapabilityContext(
            capability_name=capability_name,
            domain=domain,
            domain_knowledge=domain_knowledge,
            excluded_aspects=excluded_aspects,
            related_capabilities=related_capabilities
        )

    def analyze_uc_title(self, uc_title: str, capability_context: CapabilityContext) -> UCTitleAnalysis:
        """
        Step 1.2: Analyze UC title for solution space constraints
        
        Args:
            uc_title: Use case title (e.g., "Prepare Milk Coffee")
            capability_context: Previously analyzed capability context
            
        Returns:
            UCTitleAnalysis with constraints and variants
        """
        doc = self.nlp(uc_title.lower())
        
        # Extract solution constraints
        constraints = self._extract_solution_constraints(doc, capability_context)
        
        # Identify possible variants
        variants = self._identify_possible_variants(doc, capability_context)
        
        # Determine scope restrictions
        scope_restrictions = self._analyze_scope_restrictions(doc, capability_context)
        
        return UCTitleAnalysis(
            uc_title=uc_title,
            solution_constraints=constraints,
            possible_variants=variants,
            scope_restrictions=scope_restrictions
        )

    def analyze_goal(self, goal_text: str) -> GoalAnalysis:
        """
        Step 1.3: Analyze goal for temporal and input requirements
        
        Args:
            goal_text: Goal description (e.g., "User can drink milk coffee at 7am")
            
        Returns:
            GoalAnalysis with temporal requirements and identified components
        """
        doc = self.nlp(goal_text.lower())
        
        # Extract temporal requirements
        temporal_reqs = self._extract_temporal_requirements(goal_text)
        
        # Extract input requirements
        input_reqs = self._extract_input_requirements(doc)
        
        # Identify initial controllers
        controllers = self._identify_initial_controllers(temporal_reqs, input_reqs)
        
        # Identify initial boundaries
        boundaries = self._identify_initial_boundaries(temporal_reqs, input_reqs)
        
        return GoalAnalysis(
            goal_text=goal_text,
            temporal_requirements=temporal_reqs,
            input_requirements=input_reqs,
            identified_controllers=controllers,
            identified_boundaries=boundaries
        )

    def analyze_actors(self, actors_text: str) -> List[ActorInfo]:
        """
        Step 1.4: Analyze and classify actors
        
        Args:
            actors_text: Actor description or list
            
        Returns:
            List of ActorInfo with type classification
        """
        actors = []
        
        # Split and clean actor names
        actor_names = [name.strip() for name in re.split(r'[,;]', actors_text) if name.strip()]
        
        for actor_name in actor_names:
            actor_type = self._classify_actor_type(actor_name)
            actors.append(ActorInfo(
                name=actor_name,
                type=actor_type,
                description=f"{'Human' if actor_type == ActorType.HUMAN else 'Non-human'} actor"
            ))
        
        return actors

    def perform_phase1_analysis(self, capability_name: str, uc_title: str, 
                               goal_text: str, actors_text: str) -> Phase1Result:
        """
        Complete Phase 1 analysis following UC-Methode.txt
        
        Args:
            capability_name: Capability name
            uc_title: Use case title
            goal_text: Goal description
            actors_text: Actors list/description
            
        Returns:
            Complete Phase1Result
        """
        # Step 1.1: Capability analysis
        capability_context = self.analyze_capability(capability_name)
        
        # Step 1.2: UC title analysis
        uc_title_analysis = self.analyze_uc_title(uc_title, capability_context)
        
        # Step 1.3: Goal analysis
        goal_analysis = self.analyze_goal(goal_text)
        
        # Step 1.4: Actor analysis
        actors = self.analyze_actors(actors_text)
        
        # Generate context summary
        context_summary = self._generate_context_summary(
            capability_context, uc_title_analysis, goal_analysis, actors
        )
        
        return Phase1Result(
            capability_context=capability_context,
            uc_title_analysis=uc_title_analysis,
            goal_analysis=goal_analysis,
            actors=actors,
            context_summary=context_summary
        )

    # Private helper methods
    
    def _identify_domain(self, key_terms: List[str]) -> str:
        """Identify domain based on key terms"""
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = sum(1 for term in key_terms if any(pattern in term for pattern in patterns))
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.keys(), key=lambda k: domain_scores[k])
        
        return "generic"

    def _build_domain_knowledge(self, domain: str, key_terms: List[str]) -> List[str]:
        """Build domain-specific knowledge base"""
        knowledge = []
        
        domain_knowledge_base = {
            "beverage_preparation": [
                "Beverages require liquid ingredients",
                "Temperature control is important for quality",
                "Timing affects taste and safety",
                "Equipment cleaning is necessary for hygiene",
                "Different beverages have different preparation methods"
            ],
            "food_preparation": [
                "Food safety and hygiene are critical",
                "Temperature control prevents spoilage",
                "Timing affects quality and safety",
                "Ingredients have storage requirements"
            ],
            "rocket_science": [
                "Safety is paramount in rocket operations",
                "Precise timing is critical for launches",
                "Multiple redundant systems are required",
                "Environmental conditions affect operations",
                "Fuel handling requires extreme caution",
                "Trajectory calculations must be precise",
                "Abort procedures are essential for safety"
            ],
            "aerospace": [
                "Flight safety protocols are mandatory",
                "Weather conditions affect operations",
                "Navigation systems require calibration",
                "Maintenance schedules are critical",
                "Altitude affects system performance",
                "Emergency procedures must be available"
            ],
            "automotive": [
                "Vehicle safety systems are essential",
                "Engine performance depends on maintenance",
                "Driver input affects vehicle behavior",
                "Environmental conditions impact performance",
                "Braking systems require regular inspection"
            ],
            "robotics": [
                "Safety interlocks prevent accidents",
                "Sensor feedback enables autonomous operation",
                "Calibration ensures precision",
                "Error handling prevents damage",
                "Real-time control is often required"
            ],
            "energy": [
                "Grid stability requires load balancing",
                "Safety systems prevent overloads",
                "Environmental monitoring is essential",
                "Redundancy prevents power outages",
                "Renewable sources are weather dependent"
            ],
            "nuclear": [
                "Safety is the highest priority",
                "Radiation monitoring is continuous",
                "Containment systems are critical",
                "Emergency shutdown procedures exist",
                "Personnel protection is mandatory"
            ],
            "telecommunications": [
                "Signal quality affects performance",
                "Network redundancy prevents outages",
                "Interference can disrupt communications",
                "Security protocols protect data",
                "Bandwidth limits affect capacity"
            ],
            "healthcare": [
                "Patient safety is paramount",
                "Sterile conditions prevent infection",
                "Accurate diagnosis requires proper procedures",
                "Emergency protocols save lives",
                "Documentation is legally required"
            ],
            "security": [
                "Access control prevents unauthorized entry",
                "Surveillance systems enable monitoring",
                "Threat detection requires real-time analysis",
                "Emergency response procedures are critical",
                "Data protection prevents breaches"
            ]
        }
        
        # Get domain-specific knowledge
        if domain in domain_knowledge_base:
            knowledge.extend(domain_knowledge_base[domain])
        
        # Add specific knowledge based on key terms
        term_specific_knowledge = {
            "coffee": [
                "Coffee beans need grinding",
                "Hot water is required for brewing",
                "Milk is perishable and needs cooling",
                "Filters are used for brewing process"
            ],
            "rocket": [
                "Fuel systems require precise control",
                "Launch windows are time-critical",
                "Telemetry monitors all systems",
                "Range safety ensures public protection"
            ],
            "spacecraft": [
                "Life support systems are critical",
                "Communication with ground is essential",
                "Orbital mechanics determine trajectories",
                "Thermal control maintains temperatures"
            ],
            "robot": [
                "Motion planning prevents collisions",
                "Sensors provide environmental awareness",
                "Control loops maintain stability",
                "Safety zones protect operators"
            ],
            "reactor": [
                "Control rods regulate reactions",
                "Cooling systems prevent overheating",
                "Pressure vessels contain reactions",
                "Monitoring systems track parameters"
            ]
        }
        
        for term in key_terms:
            if term in term_specific_knowledge:
                knowledge.extend(term_specific_knowledge[term])
        
        return knowledge

    def _identify_excluded_aspects(self, domain: str, key_terms: List[str]) -> List[str]:
        """Identify aspects to exclude from analysis"""
        excluded = []
        
        exclusion_patterns = {
            "beverage_preparation": {
                "coffee": [
                    "Manual coffee preparation",
                    "Bean sourcing and purchasing",
                    "Equipment manufacturing details"
                ]
            },
            "rocket_science": {
                "rocket": [
                    "Rocket manufacturing",
                    "Component procurement",
                    "Historical mission analysis",
                    "Theoretical physics research"
                ],
                "spacecraft": [
                    "Spacecraft design process",
                    "Material science research",
                    "Vendor negotiations"
                ]
            },
            "automotive": {
                "vehicle": [
                    "Vehicle manufacturing",
                    "Parts procurement",
                    "Marketing and sales"
                ]
            },
            "healthcare": {
                "medical": [
                    "Medical research",
                    "Drug development",
                    "Insurance processing"
                ]
            },
            "nuclear": {
                "reactor": [
                    "Reactor design",
                    "Nuclear waste disposal",
                    "Regulatory compliance"
                ]
            }
        }
        
        if domain in exclusion_patterns:
            for term in key_terms:
                if term in exclusion_patterns[domain]:
                    excluded.extend(exclusion_patterns[domain][term])
        
        return excluded

    def _find_related_capabilities(self, domain: str, key_terms: List[str]) -> List[str]:
        """Find related capabilities in the same domain"""
        related = []
        
        related_capabilities = {
            "beverage_preparation": {
                "coffee": ["Tea Preparation", "Hot Chocolate Preparation", "Espresso Preparation"],
                "tea": ["Coffee Preparation", "Herbal Tea Preparation"],
                "drink": ["Cocktail Preparation", "Smoothie Preparation"]
            },
            "rocket_science": {
                "rocket": ["Satellite Deployment", "Space Station Operations", "Planetary Missions"],
                "spacecraft": ["Orbital Maneuvers", "Re-entry Operations", "Deep Space Navigation"],
                "launch": ["Mission Planning", "Range Operations", "Recovery Operations"]
            },
            "automotive": {
                "vehicle": ["Traffic Management", "Parking Systems", "Vehicle Maintenance"],
                "engine": ["Fuel Systems", "Emission Control", "Performance Tuning"],
                "driving": ["Navigation Systems", "Safety Systems", "Driver Assistance"]
            },
            "aerospace": {
                "aircraft": ["Air Traffic Control", "Weather Systems", "Maintenance Operations"],
                "flight": ["Navigation Systems", "Communication Systems", "Emergency Procedures"]
            },
            "robotics": {
                "robot": ["Human-Robot Interaction", "Autonomous Navigation", "Task Planning"],
                "automation": ["Process Control", "Quality Assurance", "Predictive Maintenance"]
            },
            "energy": {
                "power": ["Grid Management", "Load Balancing", "Fault Detection"],
                "renewable": ["Weather Forecasting", "Energy Storage", "Grid Integration"]
            },
            "nuclear": {
                "reactor": ["Waste Management", "Safety Systems", "Maintenance Operations"],
                "radiation": ["Environmental Monitoring", "Personnel Protection", "Emergency Response"]
            },
            "healthcare": {
                "medical": ["Patient Monitoring", "Emergency Response", "Equipment Management"],
                "patient": ["Appointment Scheduling", "Medical Records", "Treatment Planning"]
            }
        }
        
        if domain in related_capabilities:
            for term in key_terms:
                if term in related_capabilities[domain]:
                    related.extend(related_capabilities[domain][term])
        
        return related

    def _extract_solution_constraints(self, doc, capability_context: CapabilityContext) -> List[str]:
        """Extract constraints from UC title"""
        constraints = []
        
        # Look for specific variants or modifiers
        for token in doc:
            if token.pos_ == "ADJ" and token.dep_ in ["amod", "compound"]:
                constraints.append(f"Specific variant: {token.text}")
        
        return constraints

    def _identify_possible_variants(self, doc, capability_context: CapabilityContext) -> List[str]:
        """Identify possible UC variants"""
        variants = []
        
        # Extract main action and object
        main_verb = None
        main_object = None
        
        for token in doc:
            if token.pos_ == "VERB" and not main_verb:
                main_verb = token.lemma_
            elif token.pos_ in ["NOUN"] and not main_object:
                main_object = token.text
        
        if main_verb and main_object:
            if "coffee" in main_object.lower():
                variants.extend(["Prepare Espresso", "Prepare Americano", "Prepare Cappuccino"])
        
        return variants

    def _analyze_scope_restrictions(self, doc, capability_context: CapabilityContext) -> List[str]:
        """Analyze scope restrictions from title"""
        restrictions = []
        
        # Look for restrictive adjectives
        for token in doc:
            if token.pos_ == "ADJ":
                restrictions.append(f"Restricted to {token.text} variant")
        
        return restrictions

    def _extract_temporal_requirements(self, goal_text: str) -> List[str]:
        """Extract temporal requirements using regex patterns"""
        temporal_reqs = []
        
        for pattern in self.temporal_patterns:
            matches = re.findall(pattern, goal_text.lower())
            temporal_reqs.extend(matches)
        
        # Add semantic temporal analysis
        doc = self.nlp(goal_text.lower())
        for token in doc:
            if token.pos_ == "NUM" and "time" in [child.lemma_ for child in token.children]:
                temporal_reqs.append(f"Time: {token.text}")
        
        return temporal_reqs

    def _extract_input_requirements(self, doc) -> List[str]:
        """Extract input requirements from goal"""
        input_reqs = []
        
        # Look for input-related patterns
        for token in doc:
            if token.lemma_ in ["input", "enter", "select", "choose", "set"]:
                input_reqs.append(f"Input required: {token.text}")
        
        return input_reqs

    def _identify_initial_controllers(self, temporal_reqs: List[str], input_reqs: List[str]) -> List[str]:
        """Identify initial controllers from requirements"""
        controllers = []
        
        if temporal_reqs:
            controllers.append("TimeManager")
        
        if input_reqs:
            controllers.append("InputManager")
        
        return controllers

    def _identify_initial_boundaries(self, temporal_reqs: List[str], input_reqs: List[str]) -> List[str]:
        """Identify initial boundaries from requirements"""
        boundaries = []
        
        if temporal_reqs:
            if any("automatic" in req.lower() for req in temporal_reqs):
                boundaries.append("Time")
            else:
                boundaries.append("Time Input")
        
        if input_reqs:
            boundaries.append("User Input")
        
        return boundaries

    def _classify_actor_type(self, actor_name: str) -> ActorType:
        """Classify actor as human or non-human"""
        actor_lower = actor_name.lower()
        
        # Check for non-human indicators
        if any(indicator in actor_lower for indicator in self.non_human_indicators):
            return ActorType.NON_HUMAN
        
        # Check for human indicators
        if any(human_word in actor_lower for human_word in ["user", "customer", "person", "operator"]):
            return ActorType.HUMAN
        
        # Default to human for ambiguous cases
        return ActorType.HUMAN

    def _generate_context_summary(self, capability_context: CapabilityContext,
                                 uc_title_analysis: UCTitleAnalysis,
                                 goal_analysis: GoalAnalysis,
                                 actors: List[ActorInfo]) -> str:
        """Generate comprehensive context summary"""
        summary_parts = [
            f"Domain: {capability_context.domain}",
            f"Capability: {capability_context.capability_name}",
            f"Use Case: {uc_title_analysis.uc_title}",
            f"Actors: {', '.join([f'{a.name} ({a.type.value})' for a in actors])}"
        ]
        
        if goal_analysis.temporal_requirements:
            summary_parts.append(f"Temporal Requirements: {', '.join(goal_analysis.temporal_requirements)}")
        
        if capability_context.excluded_aspects:
            summary_parts.append(f"Excluded: {', '.join(capability_context.excluded_aspects)}")
        
        return "; ".join(summary_parts)


def main():
    """Example usage with UC1, UC2 and additional domain examples"""
    import json
    import os
    
    analyzer = ContextAnalyzer()
    
    # Create output directory
    output_dir = "Zwischenprodukte"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test cases from different domains
    test_cases = [
        {
            "name": "UC1_Coffee",
            "capability_name": "Coffee Preparation",
            "uc_title": "Prepare Milk Coffee",
            "goal_text": "User can drink their milk coffee every morning at 7am",
            "actors_text": "User"
        },
        {
            "name": "UC2_Espresso", 
            "capability_name": "Coffee Preparation",
            "uc_title": "Prepare Espresso",
            "goal_text": "User wants to drink an espresso",
            "actors_text": "User"
        },
        {
            "name": "UC3_Rocket_Launch",
            "capability_name": "Rocket Launch",
            "uc_title": "Execute Satellite Launch",
            "goal_text": "Mission control can launch satellite into orbit at scheduled time",
            "actors_text": "Mission Control, Launch Sequencer"
        },
        {
            "name": "UC4_Spacecraft_Docking",
            "capability_name": "Spacecraft Operations", 
            "uc_title": "Dock with Space Station",
            "goal_text": "Spacecraft can autonomously dock with ISS",
            "actors_text": "Flight Controller, Automated Docking System"
        },
        {
            "name": "UC5_Nuclear_Shutdown",
            "capability_name": "Nuclear Reactor Control",
            "uc_title": "Emergency Reactor Shutdown",
            "goal_text": "Reactor can be safely shut down in emergency",
            "actors_text": "Operator, Safety System"
        },
        {
            "name": "UC6_Robot_Assembly",
            "capability_name": "Robotic Manufacturing",
            "uc_title": "Assemble Product Components", 
            "goal_text": "Robot can assemble components with precision",
            "actors_text": "Operator, Quality Control System"
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        print(f"=== {test_case['name']} Analysis ===")
        
        result = analyzer.perform_phase1_analysis(
            capability_name=test_case["capability_name"],
            uc_title=test_case["uc_title"], 
            goal_text=test_case["goal_text"],
            actors_text=test_case["actors_text"]
        )
        
        print(f"Context Summary: {result.context_summary}")
        print(f"Domain: {result.capability_context.domain}")
        print(f"Domain Knowledge: {result.capability_context.domain_knowledge[:3]}...")  # Show first 3
        print(f"Controllers: {result.goal_analysis.identified_controllers}")
        print(f"Boundaries: {result.goal_analysis.identified_boundaries}")
        print(f"Related Capabilities: {result.capability_context.related_capabilities}")
        print("")
        
        # Convert result to dictionary for JSON serialization
        result_dict = {
            "capability_context": {
                "capability_name": result.capability_context.capability_name,
                "domain": result.capability_context.domain,
                "domain_knowledge": result.capability_context.domain_knowledge,
                "excluded_aspects": result.capability_context.excluded_aspects,
                "related_capabilities": result.capability_context.related_capabilities
            },
            "uc_title_analysis": {
                "uc_title": result.uc_title_analysis.uc_title,
                "solution_constraints": result.uc_title_analysis.solution_constraints,
                "possible_variants": result.uc_title_analysis.possible_variants,
                "scope_restrictions": result.uc_title_analysis.scope_restrictions
            },
            "goal_analysis": {
                "goal_text": result.goal_analysis.goal_text,
                "temporal_requirements": result.goal_analysis.temporal_requirements,
                "input_requirements": result.goal_analysis.input_requirements,
                "identified_controllers": result.goal_analysis.identified_controllers,
                "identified_boundaries": result.goal_analysis.identified_boundaries
            },
            "actors": [
                {
                    "name": actor.name,
                    "type": actor.type.value,
                    "description": actor.description
                }
                for actor in result.actors
            ],
            "context_summary": result.context_summary
        }
        
        results[test_case["name"]] = result_dict
        
        # Save individual result
        output_file = os.path.join(output_dir, f"{test_case['name']}_phase1_analysis.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    # Save combined results
    combined_output_file = os.path.join(output_dir, "all_phase1_analyses.json")
    with open(combined_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_dir}/ directory")


if __name__ == "__main__":
    main()