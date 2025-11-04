"""
Domain Verb Configuration Loader
Loads common and domain-specific verb classifications from JSON files
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

class VerbType(Enum):
    TRANSACTION_VERB = "transaction"
    TRANSFORMATION_VERB = "transformation"
    FUNCTION_VERB = "function"

@dataclass
class VerbConfiguration:
    transaction_verbs: Dict[str, str]
    transformation_verbs: Dict[str, str]
    function_verbs: Dict[str, str]
    implementation_elements: Dict[str, Dict[str, str]]
    coffee_additives: Dict[str, Dict[str, str]]
    contextual_entities: Optional[Dict[str, List[Dict]]] = None
    excluded_entity_terms: Optional[List[str]] = None

class DomainVerbLoader:
    """
    Loads verb classifications from domain JSON configurations
    Combines common domain verbs with domain-specific verbs
    """
    
    def __init__(self, domains_directory: str = "domains"):
        self.domains_directory = domains_directory
        self.common_config = None
        self.domain_configs = {}
        self._load_configurations()
    
    def _load_configurations(self):
        """Load common and all domain-specific configurations"""
        
        # Load common domain configuration
        common_path = os.path.join(self.domains_directory, "common_domain.json")
        if os.path.exists(common_path):
            with open(common_path, 'r', encoding='utf-8') as f:
                self.common_config = json.load(f)
                print(f"Loaded common domain configuration")
        else:
            print(f"WARNING: Common domain file not found: {common_path}")
            self.common_config = {"verb_classification": {}, "implementation_elements": {}}
        
        # Load all domain-specific configurations
        if os.path.exists(self.domains_directory):
            for filename in os.listdir(self.domains_directory):
                if filename.endswith('.json') and filename != 'common_domain.json':
                    domain_name = filename.replace('.json', '')
                    domain_path = os.path.join(self.domains_directory, filename)
                    
                    try:
                        with open(domain_path, 'r', encoding='utf-8') as f:
                            domain_config = json.load(f)
                            self.domain_configs[domain_name] = domain_config
                            print(f"Loaded domain configuration: {domain_name}")
                    except Exception as e:
                        print(f"ERROR loading domain {domain_name}: {e}")
    
    def get_verb_configuration(self, domain_name: Optional[str] = None) -> VerbConfiguration:
        """
        Get combined verb configuration for a specific domain
        
        Args:
            domain_name: Domain name (e.g., 'beverage_preparation'). If None, returns only common verbs.
        
        Returns:
            VerbConfiguration with merged common + domain-specific verbs
        """
        
        # Start with common verbs
        transaction_verbs = {}
        transformation_verbs = {}
        function_verbs = {}
        implementation_elements = {}
        coffee_additives = {}
        
        # Load common verbs
        if self.common_config and "verb_classification" in self.common_config:
            common_verbs = self.common_config["verb_classification"]
            
            if "transaction_verbs" in common_verbs:
                transaction_verbs.update(common_verbs["transaction_verbs"].get("verbs", {}))
            
            if "transformation_verbs" in common_verbs:
                transformation_verbs.update(common_verbs["transformation_verbs"].get("verbs", {}))
            
            if "function_verbs" in common_verbs:
                function_verbs.update(common_verbs["function_verbs"].get("verbs", {}))
        
        # Load common implementation elements
        if self.common_config and "implementation_elements" in self.common_config:
            implementation_elements.update(self.common_config["implementation_elements"].get("elements", {}))
        
        # Load common coffee additives
        if self.common_config and "coffee_additives" in self.common_config:
            coffee_additives.update(self.common_config["coffee_additives"].get("additives", {}))
        
        # Add domain-specific verbs if domain is specified
        if domain_name and domain_name in self.domain_configs:
            domain_config = self.domain_configs[domain_name]
            
            if "verb_classification" in domain_config:
                domain_verbs = domain_config["verb_classification"]
                
                # Merge domain-specific verbs (they override common ones if same key)
                if "transaction_verbs" in domain_verbs:
                    transaction_verbs.update(domain_verbs["transaction_verbs"].get("verbs", {}))
                
                if "transformation_verbs" in domain_verbs:
                    transformation_verbs.update(domain_verbs["transformation_verbs"].get("verbs", {}))
                
                if "function_verbs" in domain_verbs:
                    function_verbs.update(domain_verbs["function_verbs"].get("verbs", {}))
            
            # Add domain-specific implementation elements
            if "implementation_elements" in domain_config:
                implementation_elements.update(domain_config["implementation_elements"].get("elements", {}))
            
            # Add domain-specific coffee additives
            if "coffee_additives" in domain_config:
                coffee_additives.update(domain_config["coffee_additives"].get("additives", {}))
        
        # Load contextual entities from domain configuration
        contextual_entities = None
        excluded_entity_terms = None
        if domain_name and domain_name in self.domain_configs:
            domain_config = self.domain_configs[domain_name]
            if "contextual_entities" in domain_config:
                contextual_entities = domain_config["contextual_entities"]
            if "excluded_entity_terms" in domain_config:
                excluded_entity_terms = domain_config["excluded_entity_terms"]
        
        return VerbConfiguration(
            transaction_verbs=transaction_verbs,
            transformation_verbs=transformation_verbs,
            function_verbs=function_verbs,
            implementation_elements=implementation_elements,
            coffee_additives=coffee_additives,
            contextual_entities=contextual_entities,
            excluded_entity_terms=excluded_entity_terms
        )
    
    def categorize_verb(self, verb_lemma: str, domain_name: Optional[str] = None) -> VerbType:
        """
        Categorize a verb using domain-specific + common verb classifications
        
        Args:
            verb_lemma: Lemmatized verb (e.g., "activate", "grind", "output")
            domain_name: Domain context (e.g., 'beverage_preparation')
        
        Returns:
            VerbType classification
        """
        config = self.get_verb_configuration(domain_name)
        
        if verb_lemma in config.transaction_verbs:
            return VerbType.TRANSACTION_VERB
        elif verb_lemma in config.transformation_verbs:
            return VerbType.TRANSFORMATION_VERB
        else:
            return VerbType.FUNCTION_VERB
    
    def get_transformation_description(self, verb_lemma: str, domain_name: Optional[str] = None) -> Optional[str]:
        """Get transformation description for a transformation verb"""
        config = self.get_verb_configuration(domain_name)
        return config.transformation_verbs.get(verb_lemma)
    
    def get_transformation_for_verb(self, verb_lemma: str, domain_name: Optional[str] = None) -> Optional[str]:
        """Alias for get_transformation_description - used by data flow analysis"""
        return self.get_transformation_description(verb_lemma, domain_name)
    
    def get_transaction_description(self, verb_lemma: str, domain_name: Optional[str] = None) -> Optional[str]:
        """Get transaction description for a transaction verb"""
        config = self.get_verb_configuration(domain_name)
        return config.transaction_verbs.get(verb_lemma)
    
    def get_implementation_element_info(self, element_name: str, domain_name: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Get implementation element warning and suggestion"""
        config = self.get_verb_configuration(domain_name)
        element_name_lower = element_name.lower()
        
        # First try exact match
        result = config.implementation_elements.get(element_name_lower)
        if result:
            return result
        
        # Then try word-by-word for compound terms like "water heater"
        words = element_name_lower.split()
        for word in words:
            if word in config.implementation_elements:
                return config.implementation_elements[word]
        
        return None
    
    def get_coffee_additive_info(self, additive_name: str, domain_name: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Get coffee additive information for generalized coffee modeling"""
        config = self.get_verb_configuration(domain_name)
        return config.coffee_additives.get(additive_name.lower())
    
    def is_coffee_additive(self, additive_name: str, domain_name: Optional[str] = None) -> bool:
        """Check if an ingredient is a recognized coffee additive"""
        return self.get_coffee_additive_info(additive_name, domain_name) is not None
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domain names"""
        return list(self.domain_configs.keys())
    
    def print_configuration_summary(self, domain_name: Optional[str] = None):
        """Print summary of loaded verb configuration"""
        config = self.get_verb_configuration(domain_name)
        
        domain_label = domain_name if domain_name else "Common Only"
        print(f"\n=== VERB CONFIGURATION SUMMARY: {domain_label} ===")
        
        print(f"\nTransaction Verbs ({len(config.transaction_verbs)}):")
        for verb, desc in sorted(config.transaction_verbs.items()):
            print(f"  {verb}: {desc}")
        
        print(f"\nTransformation Verbs ({len(config.transformation_verbs)}):")
        for verb, desc in sorted(config.transformation_verbs.items()):
            print(f"  {verb}: {desc}")
        
        print(f"\nFunction Verbs ({len(config.function_verbs)}):")
        for verb, desc in sorted(config.function_verbs.items()):
            print(f"  {verb}: {desc}")
        
        print(f"\nImplementation Elements ({len(config.implementation_elements)}):")
        for element, info in sorted(config.implementation_elements.items()):
            print(f"  {element}: {info.get('warning', 'No warning')}")
        
        print(f"\nCoffee Additives ({len(config.coffee_additives)}):")
        for additive, info in sorted(config.coffee_additives.items()):
            print(f"  {additive}: {info.get('transformation', 'No transformation')}")
    
    def detect_domain_from_text(self, text: str) -> Optional[str]:
        """
        Auto-detect domain from text content using domain keywords
        
        Args:
            text: Text to analyze for domain detection
        
        Returns:
            Domain name or None if no clear domain detected
        """
        text_lower = text.lower()
        domain_scores = {}
        
        for domain_name, domain_config in self.domain_configs.items():
            score = 0
            
            # Check domain keywords
            if "keywords" in domain_config:
                for keyword in domain_config["keywords"]:
                    if keyword.lower() in text_lower:
                        score += 2
            
            # Check domain-specific verbs
            if "verb_classification" in domain_config:
                all_domain_verbs = []
                for verb_type in domain_config["verb_classification"].values():
                    if "verbs" in verb_type:
                        all_domain_verbs.extend(verb_type["verbs"].keys())
                
                for verb in all_domain_verbs:
                    if verb in text_lower:
                        score += 1
            
            domain_scores[domain_name] = score
        
        # Return domain with highest score (if > 0)
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 0:
                return best_domain
        
        return None
    
    def classify_step_phase(self, step_text: str, domain_name: Optional[str] = None) -> str:
        """
        Classify the phase of a UC step using domain-specific context classification
        
        Args:
            step_text: Text of the UC step
            domain_name: Domain context
            
        Returns:
            Phase classification (initialization, validation, execution, completion, error_handling)
        """
        if not domain_name or domain_name not in self.domain_configs:
            return "execution"  # Default
        
        domain_config = self.domain_configs[domain_name]
        context_classification = domain_config.get("context_classification", {})
        step_phases = context_classification.get("step_phases", {})
        
        step_lower = step_text.lower()
        
        # Score each phase based on keywords and patterns
        phase_scores = {}
        
        for phase_name, phase_config in step_phases.items():
            score = 0
            
            # Check keywords
            keywords = phase_config.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in step_lower:
                    score += 2
            
            # Check patterns (regex)
            import re
            patterns = phase_config.get("patterns", [])
            for pattern in patterns:
                if re.search(pattern.lower(), step_lower):
                    score += 3  # Patterns are more specific
            
            phase_scores[phase_name] = score
        
        # Return highest scoring phase
        if phase_scores:
            best_phase = max(phase_scores, key=phase_scores.get)
            if phase_scores[best_phase] > 0:
                return best_phase
        
        return "execution"  # Default
    
    def classify_business_context(self, step_text: str, domain_name: Optional[str] = None) -> str:
        """
        Classify the business context of a UC step using domain-specific classification
        
        Args:
            step_text: Text of the UC step
            domain_name: Domain context
            
        Returns:
            Business context classification
        """
        if not domain_name or domain_name not in self.domain_configs:
            return "General Operations"
        
        domain_config = self.domain_configs[domain_name]
        context_classification = domain_config.get("context_classification", {})
        business_contexts = context_classification.get("business_contexts", {})
        
        step_lower = step_text.lower()
        
        # Score each context based on keywords and patterns
        context_scores = {}
        
        for context_name, context_config in business_contexts.items():
            score = 0
            
            # Check keywords
            keywords = context_config.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in step_lower:
                    score += 2
            
            # Check patterns (regex)
            import re
            patterns = context_config.get("patterns", [])
            for pattern in patterns:
                if re.search(pattern.lower(), step_lower):
                    score += 3  # Patterns are more specific
            
            context_scores[context_name] = score
        
        # Return highest scoring context
        if context_scores:
            best_context = max(context_scores, key=context_scores.get)
            if context_scores[best_context] > 0:
                return best_context
        
        return "General Operations"  # Default
    
    def classify_technical_context(self, step_text: str, domain_name: Optional[str] = None) -> str:
        """
        Classify the technical context of a UC step using domain-specific classification
        
        Args:
            step_text: Text of the UC step
            domain_name: Domain context
            
        Returns:
            Technical context classification
        """
        if not domain_name or domain_name not in self.domain_configs:
            return "General system operation"
        
        domain_config = self.domain_configs[domain_name]
        context_classification = domain_config.get("context_classification", {})
        technical_contexts = context_classification.get("technical_contexts", {})
        
        step_lower = step_text.lower()
        
        # Score each context based on keywords and patterns
        context_scores = {}
        
        for context_name, context_config in technical_contexts.items():
            score = 0
            
            # Check keywords
            keywords = context_config.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in step_lower:
                    score += 2
            
            # Check patterns (regex)
            import re
            patterns = context_config.get("patterns", [])
            for pattern in patterns:
                if re.search(pattern.lower(), step_lower):
                    score += 3  # Patterns are more specific
            
            context_scores[context_name] = score
        
        # Return highest scoring context
        if context_scores:
            best_context = max(context_scores, key=context_scores.get)
            if context_scores[best_context] > 0:
                return best_context
        
        return "General system operation"  # Default
    
    def get_context_specific_controller_name(self, step_text: str, technical_context: str, 
                                           direct_object: str = "", verb_lemma: str = "", 
                                           domain_name: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate context-specific controller name and description based on technical context
        
        Args:
            step_text: Text of the UC step
            technical_context: Technical context classification
            direct_object: Direct object from grammatical analysis
            verb_lemma: Lemmatized verb
            domain_name: Domain context
            
        Returns:
            Tuple of (controller_name, description)
        """
        if not domain_name or domain_name not in self.domain_configs:
            # Fallback to generic naming
            if direct_object:
                obj_clean = self._clean_entity_name(direct_object)
                return f"{obj_clean}Manager", f"Manages {verb_lemma} function"
            else:
                verb_clean = verb_lemma.capitalize()
                return f"{verb_clean}Manager", f"Manages {verb_lemma} function"
        
        domain_config = self.domain_configs[domain_name]
        context_classification = domain_config.get("context_classification", {})
        controller_mapping = context_classification.get("controller_mapping", {})
        
        # Check if we have specific mapping for this technical context
        if technical_context in controller_mapping:
            mapping_config = controller_mapping[technical_context]
            pattern = mapping_config.get("controller_pattern", "{Entity}Manager")
            description_pattern = mapping_config.get("description_pattern", "Manages {entity} in {step_id}")
            
            # Determine the entity for the pattern
            entity = self._determine_entity_for_controller(step_text, direct_object, mapping_config)
            
            # Apply pattern
            controller_name = pattern.replace("{Entity}", entity)
            description = description_pattern.replace("{entity}", entity.lower()).replace("{step_id}", "step")
            
            return controller_name, description
        
        # Fallback to generic naming
        if direct_object:
            obj_clean = self._clean_entity_name(direct_object)
            return f"{obj_clean}Manager", f"Manages {verb_lemma} function"
        else:
            verb_clean = verb_lemma.capitalize()
            return f"{verb_clean}Manager", f"Manages {verb_lemma} function"
    
    def _determine_entity_for_controller(self, step_text: str, direct_object: str, mapping_config: dict) -> str:
        """
        Determine the best entity name for controller pattern based on step text and examples
        
        Args:
            step_text: UC step text
            direct_object: Direct object from grammatical analysis
            mapping_config: Controller mapping configuration
            
        Returns:
            Entity name for controller pattern
        """
        step_lower = step_text.lower()
        
        # Check examples first for specific matches
        examples = mapping_config.get("examples", {})
        for keyword, controller_name in examples.items():
            if keyword in step_lower:
                # Extract entity from example controller name
                # E.g., "LaunchWindowTriggerController" -> "LaunchWindow"
                entity = controller_name.replace("TriggerController", "").replace("ProcessingController", "")
                entity = entity.replace("ValidationController", "").replace("ActuationController", "")
                entity = entity.replace("CommunicationController", "").replace("MonitoringController", "")
                entity = entity.replace("RecoveryController", "").replace("Controller", "")
                return entity
        
        # If no specific match, use direct object or derive from step
        if direct_object:
            return self._clean_entity_name(direct_object)
        
        # Try to extract meaningful entity from step text
        keywords = mapping_config.get("keywords", [])
        for keyword in keywords:
            if keyword in step_lower:
                # Look for noun after the keyword
                import re
                pattern = fr"{keyword}\s+(\w+)"
                match = re.search(pattern, step_lower)
                if match:
                    return self._clean_entity_name(match.group(1))
        
        # Last resort: use generic names based on context
        if "trigger" in step_lower:
            return "Event"
        elif "window" in step_lower:
            return "Window"
        elif "system" in step_lower:
            return "System"
        
        return "Operation"
    
    def _clean_entity_name(self, entity_name: str) -> str:
        """Clean and format entity name for use in controller names"""
        # Remove articles and common words
        words = entity_name.split()
        cleaned_words = []
        
        for word in words:
            word_clean = word.strip().lower()
            if word_clean not in {'the', 'a', 'an', 'this', 'that', 'all', 'some', 'any'}:
                # Capitalize first letter
                cleaned_words.append(word.strip().capitalize())
        
        return "".join(cleaned_words) if cleaned_words else "Entity"


def test_domain_verb_loader():
    """Test the domain verb loader functionality"""
    print("="*80)
    print("TESTING DOMAIN VERB LOADER")
    print("="*80)
    
    # Initialize loader
    loader = DomainVerbLoader()
    
    print(f"\nAvailable domains: {loader.get_available_domains()}")
    
    # Test common verbs only
    print("\n" + "="*50)
    loader.print_configuration_summary()
    
    # Test domain-specific configurations
    test_domains = ["beverage_preparation", "rocket_science", "nuclear", "robotics"]
    
    for domain in test_domains:
        if domain in loader.get_available_domains():
            print("\n" + "="*50)
            loader.print_configuration_summary(domain)
    
    # Test verb categorization
    print("\n" + "="*50)
    print("VERB CATEGORIZATION TESTS")
    print("="*50)
    
    test_verbs = [
        ("output", "beverage_preparation"),
        ("grind", "beverage_preparation"),
        ("activate", "beverage_preparation"),
        ("deploy", "rocket_science"),
        ("ignite", "rocket_science"),
        ("monitor", "rocket_science"),
        ("insert", "nuclear"),
        ("cool", "nuclear"),
        ("shutdown", "nuclear"),
        ("assemble", "robotics"),
        ("pick", "robotics"),
        ("calibrate", "robotics")
    ]
    
    for verb, domain in test_verbs:
        verb_type = loader.categorize_verb(verb, domain)
        print(f"{domain:20} | {verb:10} -> {verb_type.value}")
    
    # Test domain detection
    print("\n" + "="*50)
    print("DOMAIN DETECTION TESTS")
    print("="*50)
    
    test_texts = [
        "The system activates the water heater and grinds coffee beans",
        "The system deploys satellite and ignites the rocket engine",
        "The system inserts control rods and cools the reactor",
        "The robot picks components and assembles the product"
    ]
    
    for text in test_texts:
        detected_domain = loader.detect_domain_from_text(text)
        print(f"Text: {text[:50]}...")
        print(f"Detected Domain: {detected_domain}")
        print()
    
    # Test implementation element detection
    print("\n" + "="*50)
    print("IMPLEMENTATION ELEMENT TESTS")
    print("="*50)
    
    test_elements = [
        ("heater", "beverage_preparation"),
        ("engine", "rocket_science"), 
        ("pump", "nuclear"),
        ("gripper", "robotics"),
        ("sensor", None)  # Common element
    ]
    
    for element, domain in test_elements:
        info = loader.get_implementation_element_info(element, domain)
        if info:
            print(f"{element} ({domain or 'common'}):")
            print(f"  Warning: {info.get('warning', 'None')}")
            print(f"  Suggestion: {info.get('functional_suggestion', 'None')}")
        else:
            print(f"{element} ({domain or 'common'}): No configuration found")


if __name__ == "__main__":
    test_domain_verb_loader()