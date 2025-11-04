"""
Domain-Configurable Context Analyzer for UC Robustness Analysis - Phase 1
Uses external JSON configuration files for domain-specific knowledge

Enhanced version that loads domain configurations from JSON files
"""

import spacy
import json
import os
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
class DomainConfig:
    domain_name: str
    description: str
    keywords: List[str]
    non_human_indicators: List[str]
    domain_knowledge: List[str]
    term_specific_knowledge: Dict[str, List[str]]
    exclusion_patterns: Dict[str, List[str]]
    related_capabilities: Dict[str, List[str]]


@dataclass
class CapabilityContext:
    capability_name: str
    domain: str
    domain_knowledge: List[str]
    excluded_aspects: List[str]
    related_capabilities: List[str]
    domain_config: Optional[DomainConfig] = None


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


class DomainConfigurableAnalyzer:
    """
    Context analyzer that uses domain configuration files for domain-specific knowledge
    """
    
    def __init__(self, model_name: str = "en_core_web_md", domains_path: str = "domains"):
        """
        Initialize with spaCy model and domain configurations
        
        Args:
            model_name: spaCy model name (default: en_core_web_md)
            domains_path: Path to domain configuration files
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise RuntimeError(f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}")
        
        self.domains_path = domains_path
        self.domain_configs: Dict[str, DomainConfig] = {}
        self._load_domain_configurations()
        
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

    def _load_domain_configurations(self):
        """Load all domain configuration files from the domains directory"""
        if not os.path.exists(self.domains_path):
            print(f"Warning: Domains path '{self.domains_path}' not found")
            return
        
        print(f"Loading domain configurations from {self.domains_path}/")
        
        for filename in os.listdir(self.domains_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.domains_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    
                    domain_config = DomainConfig(
                        domain_name=config_data['domain_name'],
                        description=config_data['description'],
                        keywords=config_data['keywords'],
                        non_human_indicators=config_data['non_human_indicators'],
                        domain_knowledge=config_data['domain_knowledge'],
                        term_specific_knowledge=config_data['term_specific_knowledge'],
                        exclusion_patterns=config_data['exclusion_patterns'],
                        related_capabilities=config_data['related_capabilities']
                    )
                    
                    self.domain_configs[domain_config.domain_name] = domain_config
                    print(f"  + Loaded {domain_config.domain_name}: {len(domain_config.keywords)} keywords, {len(domain_config.non_human_indicators)} non-human indicators")
                    
                except Exception as e:
                    print(f"  - Error loading {filename}: {e}")
        
        print(f"Total domains loaded: {len(self.domain_configs)}")

    def analyze_capability(self, capability_name: str) -> CapabilityContext:
        """
        Step 1.1: Analyze capability and identify domain using loaded configurations
        
        Args:
            capability_name: Name of the capability (e.g., "Coffee Preparation")
            
        Returns:
            CapabilityContext with domain identification and knowledge
        """
        doc = self.nlp(capability_name.lower())
        
        # Extract key terms
        key_terms = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        # Identify domain using loaded configurations
        domain, domain_config = self._identify_domain_from_config(key_terms)
        
        # Build domain knowledge from configuration
        domain_knowledge = []
        if domain_config:
            domain_knowledge.extend(domain_config.domain_knowledge)
            
            # Add specific knowledge based on key terms
            for term in key_terms:
                if term in domain_config.term_specific_knowledge:
                    domain_knowledge.extend(domain_config.term_specific_knowledge[term])
        
        # Identify excluded aspects from configuration
        excluded_aspects = []
        if domain_config:
            for term in key_terms:
                if term in domain_config.exclusion_patterns:
                    excluded_aspects.extend(domain_config.exclusion_patterns[term])
        
        # Find related capabilities from configuration
        related_capabilities = []
        if domain_config:
            for term in key_terms:
                if term in domain_config.related_capabilities:
                    related_capabilities.extend(domain_config.related_capabilities[term])
        
        return CapabilityContext(
            capability_name=capability_name,
            domain=domain,
            domain_knowledge=domain_knowledge,
            excluded_aspects=excluded_aspects,
            related_capabilities=related_capabilities,
            domain_config=domain_config
        )

    def _identify_domain_from_config(self, key_terms: List[str]) -> Tuple[str, Optional[DomainConfig]]:
        """Identify domain based on key terms using loaded configurations"""
        domain_scores = {}
        
        for domain_name, config in self.domain_configs.items():
            score = sum(1 for term in key_terms if any(keyword in term for keyword in config.keywords))
            if score > 0:
                domain_scores[domain_name] = score
        
        if domain_scores:
            best_domain = max(domain_scores.keys(), key=lambda k: domain_scores[k])
            return best_domain, self.domain_configs[best_domain]
        
        return "generic", None

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
        
        # Identify possible variants using domain configuration
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

    def analyze_actors(self, actors_text: str, capability_context: CapabilityContext) -> List[ActorInfo]:
        """
        Step 1.4: Analyze and classify actors using domain-specific non-human indicators
        
        Args:
            actors_text: Actor description or list
            capability_context: Context with domain configuration
            
        Returns:
            List of ActorInfo with type classification
        """
        actors = []
        
        # Split and clean actor names
        actor_names = [name.strip() for name in re.split(r'[,;]', actors_text) if name.strip()]
        
        for actor_name in actor_names:
            actor_type = self._classify_actor_type(actor_name, capability_context)
            actors.append(ActorInfo(
                name=actor_name,
                type=actor_type,
                description=f"{'Human' if actor_type == ActorType.HUMAN else 'Non-human'} actor in {capability_context.domain} domain"
            ))
        
        return actors

    def _classify_actor_type(self, actor_name: str, capability_context: CapabilityContext) -> ActorType:
        """Classify actor as human or non-human using domain-specific indicators"""
        actor_lower = actor_name.lower()
        
        # Check domain-specific non-human indicators first
        if capability_context.domain_config:
            for indicator in capability_context.domain_config.non_human_indicators:
                if indicator.lower() in actor_lower:
                    return ActorType.NON_HUMAN
        
        # Check for general non-human indicators
        general_non_human = ["system", "automatic", "timer", "sensor", "computer", "controller"]
        if any(indicator in actor_lower for indicator in general_non_human):
            return ActorType.NON_HUMAN
        
        # Check for human indicators
        if any(human_word in actor_lower for human_word in ["user", "customer", "person", "operator", "pilot", "driver"]):
            return ActorType.HUMAN
        
        # Default to human for ambiguous cases
        return ActorType.HUMAN

    def perform_phase1_analysis(self, capability_name: str, uc_title: str, 
                               goal_text: str, actors_text: str) -> Phase1Result:
        """
        Complete Phase 1 analysis using domain configurations
        
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
        
        # Step 1.4: Actor analysis (with domain context)
        actors = self.analyze_actors(actors_text, capability_context)
        
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

    # Helper methods (same logic as before but using domain configurations)
    
    def _extract_solution_constraints(self, doc, capability_context: CapabilityContext) -> List[str]:
        """Extract constraints from UC title"""
        constraints = []
        
        # Look for specific variants or modifiers
        for token in doc:
            if token.pos_ == "ADJ" and token.dep_ in ["amod", "compound"]:
                constraints.append(f"Specific variant: {token.text}")
        
        # Add domain-specific constraints if available
        if capability_context.domain_config:
            for token in doc:
                if token.lemma_ in capability_context.domain_config.keywords:
                    constraints.append(f"Domain-specific constraint: {token.text}")
        
        return constraints

    def _identify_possible_variants(self, doc, capability_context: CapabilityContext) -> List[str]:
        """Identify possible UC variants using domain configuration"""
        variants = []
        
        # Extract main action and object
        main_verb = None
        main_object = None
        
        for token in doc:
            if token.pos_ == "VERB" and not main_verb:
                main_verb = token.lemma_
            elif token.pos_ in ["NOUN"] and not main_object:
                main_object = token.text
        
        # Use domain configuration for variants
        if capability_context.domain_config and main_object:
            main_object_lower = main_object.lower()
            for term, related_caps in capability_context.domain_config.related_capabilities.items():
                if term in main_object_lower:
                    variants.extend(related_caps)
                    break
        
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
            summary_parts.append(f"Excluded: {', '.join(capability_context.excluded_aspects[:2])}...")  # Limit to first 2
        
        return "; ".join(summary_parts)

    def get_loaded_domains(self) -> List[str]:
        """Get list of loaded domain names"""
        return list(self.domain_configs.keys())

    def get_domain_info(self, domain_name: str) -> Optional[DomainConfig]:
        """Get detailed information about a specific domain"""
        return self.domain_configs.get(domain_name)


def main():
    """Example usage with UC1, UC2 and examples from different domains"""
    import json
    import os
    
    analyzer = DomainConfigurableAnalyzer()
    
    # Create output directory
    output_dir = "Zwischenprodukte"
    os.makedirs(output_dir, exist_ok=True)
    
    # Show loaded domains
    print("Loaded domains:")
    for domain in analyzer.get_loaded_domains():
        domain_info = analyzer.get_domain_info(domain)
        if domain_info:
            print(f"  * {domain}: {domain_info.description}")
            print(f"    - Keywords: {', '.join(domain_info.keywords[:5])}...")
            print(f"    - Non-human indicators: {len(domain_info.non_human_indicators)}")
    
    # Test cases from different domains
    test_cases = [
        {
            "name": "UC1_Coffee",
            "capability_name": "Coffee Preparation",
            "uc_title": "Prepare Milk Coffee",
            "goal_text": "User can drink their milk coffee every morning at 7am",
            "actors_text": "User, Timer"
        },
        {
            "name": "UC2_Espresso", 
            "capability_name": "Coffee Preparation",
            "uc_title": "Prepare Espresso",
            "goal_text": "User wants to drink an espresso",
            "actors_text": "User, Brew Timer"
        },
        {
            "name": "UC3_Rocket_Launch",
            "capability_name": "Rocket Launch",
            "uc_title": "Execute Satellite Launch",
            "goal_text": "Mission control can launch satellite into orbit at scheduled time",
            "actors_text": "Mission Control, Launch Sequencer, Range Safety System"
        },
        {
            "name": "UC4_Nuclear_Shutdown",
            "capability_name": "Nuclear Reactor Control",
            "uc_title": "Emergency Reactor Shutdown",
            "goal_text": "Reactor can be safely shut down in emergency",
            "actors_text": "Operator, Reactor Protection System, Emergency Core Cooling System"
        },
        {
            "name": "UC5_Robot_Assembly",
            "capability_name": "Robotic Manufacturing",
            "uc_title": "Assemble Product Components", 
            "goal_text": "Robot can assemble components with precision",
            "actors_text": "Operator, Quality Control System, Vision System"
        },
        {
            "name": "UC6_Aircraft_Navigation",
            "capability_name": "Aircraft Operations",
            "uc_title": "Navigate to Destination",
            "goal_text": "Aircraft can navigate safely to destination",
            "actors_text": "Pilot, Autopilot System, Flight Management System"
        },
        {
            "name": "UC7_Network_Management",
            "capability_name": "Network Operations",
            "uc_title": "Monitor Network Performance",
            "goal_text": "Network operations center can monitor network performance in real-time",
            "actors_text": "Network Operator, Network Management System, Telemetry Processor, Command and Control System"
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        print(f"\n=== {test_case['name']} Analysis ===")
        
        result = analyzer.perform_phase1_analysis(
            capability_name=test_case["capability_name"],
            uc_title=test_case["uc_title"], 
            goal_text=test_case["goal_text"],
            actors_text=test_case["actors_text"]
        )
        
        print(f"Context Summary: {result.context_summary}")
        print(f"Domain: {result.capability_context.domain}")
        if result.capability_context.domain_config:
            print(f"Domain Config: {len(result.capability_context.domain_config.non_human_indicators)} non-human indicators")
        print(f"Domain Knowledge: {result.capability_context.domain_knowledge[:2]}...")
        print(f"Controllers: {result.goal_analysis.identified_controllers}")
        print(f"Boundaries: {result.goal_analysis.identified_boundaries}")
        print(f"Actor Types: {[(a.name, a.type.value) for a in result.actors]}")
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
        output_file = os.path.join(output_dir, f"{test_case['name']}_domain_config_analysis.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    # Save combined results
    combined_output_file = os.path.join(output_dir, "all_domain_config_analyses.json")
    with open(combined_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_dir}/ directory")


if __name__ == "__main__":
    main()