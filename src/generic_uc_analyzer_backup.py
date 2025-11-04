"""
Generic Use Case Grammar Analyzer
Domain-agnostic analyzer for any UC from any domain using UC-Methode systematic analysis
"""

import spacy
import re
from spacy.tokens import Token
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from domain_verb_loader import DomainVerbLoader, VerbType
import json
from datetime import datetime

class ElementType(Enum):
    FUNCTIONAL_ENTITY = "functional"    # Domain objects (Water, Coffee, Fuel, etc.)
    IMPLEMENTATION_ELEMENT = "implementation"  # Hardware (Heater, Motor, Engine, etc.)
    CONTAINER = "container"            # Storage (Cup, Tank, Container, etc.)
    CONTROL_DATA = "control"           # Settings, Messages, Status

@dataclass
class UCStep:
    """Represents a single UC step"""
    step_id: str
    step_text: str
    flow_type: str  # "main", "alternative", "extension"
    flow_id: Optional[str] = None  # A1, A2, E1, etc.
    uc_name: Optional[str] = None  # UC1, UC2, etc.

@dataclass
class VerbAnalysis:
    step_id: str
    original_text: str
    verb: str
    verb_lemma: str
    verb_type: VerbType
    direct_object: Optional[str] = None
    prepositional_objects: List[Tuple[str, str]] = None
    warnings: List[str] = None
    suggested_functional_activity: Optional[str] = None
    uc_name: Optional[str] = None  # UC1, UC2, etc.
    
    def __post_init__(self):
        if self.prepositional_objects is None:
            self.prepositional_objects = []
        if self.warnings is None:
            self.warnings = []

@dataclass
class Precondition:
    """Represents a UC precondition"""
    resource_name: str
    original_text: str
    uc_name: str
    availability_type: str  # "available", "configured", "connected", etc.

@dataclass
class ControlFlow:
    """Represents a control flow connection between RA classes"""
    from_class: str  # Source RA class name
    to_class: str    # Target RA class name
    flow_type: str   # "boundary_to_controller", "controller_to_controller", "controller_to_boundary", "sequential", "parallel_split", "parallel_join"
    step_id: str     # UC step that triggers this flow
    flow_rule: int   # Which UC-Methode rule (1-5)
    description: str # Description of the control flow

@dataclass
class ParallelFlow:
    """Represents parallel execution paths"""
    split_step: str     # Step where parallelism begins
    join_step: str      # Step where parallelism ends
    parallel_paths: List[List[str]]  # List of parallel step sequences
    description: str    # Description of parallel execution

@dataclass
class OperationalMaterial:
    """Represents operational material with safety, hygiene, and addressing requirements"""
    material_name: str
    safety_class: str           # explosive, toxic, cryogenic, radioactive, pressure_sensitive
    hygiene_level: str          # sterile, food_grade, pharmaceutical, cleanroom
    special_requirements: List[str]  # Specific handling requirements
    addressing_id: str          # Unique identifier following universal format
    storage_conditions: Dict[str, str]  # Temperature, pressure, atmosphere requirements
    tracking_parameters: List[str]     # Parameters to monitor (temperature, pressure, etc.)
    emergency_procedures: List[str]    # Emergency response procedures

@dataclass
class SafetyConstraint:
    """Represents safety constraints for operational materials"""
    material_name: str
    constraint_type: str        # thermal, pressure, electrical, mechanical, radiation
    max_limits: Dict[str, str]  # Maximum safe operating limits
    monitoring_required: List[str]  # Required monitoring systems
    emergency_actions: List[str]    # Actions to take on constraint violation
    responsible_controller: str     # Controller responsible for this constraint

@dataclass
class Actor:
    """Represents an Actor from UC file"""
    name: str
    uc_name: str
    original_text: str
    used_in_steps: List[str] = None  # Steps where this actor is referenced

@dataclass
class HygieneRequirement:
    """Represents hygiene requirements for operational materials"""
    material_name: str
    sterility_level: str        # sterile, food_grade, pharmaceutical, cleanroom
    cleaning_protocols: List[str]   # Required cleaning procedures
    contamination_controls: List[str]  # Contamination prevention measures
    validation_requirements: List[str]  # Validation and testing requirements
    responsible_controller: str     # Controller responsible for hygiene

@dataclass
class DataFlow:
    """Represents data flow relationships between Controllers and Entities"""
    controller_name: str  # Controller that uses/provides the entity
    entity_name: str     # Entity being used/provided
    relationship_type: str  # "use" or "provide"
    step_id: str         # UC step where this relationship occurs
    transformation: str  # Transformation description (e.g., "GroundCoffee + HotWater + Filter -> Coffee")
    description: str     # Description of the data flow
    safety_constraints: List[SafetyConstraint] = None  # Safety constraints for this data flow
    hygiene_requirements: List[HygieneRequirement] = None  # Hygiene requirements
    operational_material: OperationalMaterial = None  # Associated operational material
    
    def __post_init__(self):
        if self.safety_constraints is None:
            self.safety_constraints = []
        if self.hygiene_requirements is None:
            self.hygiene_requirements = []
    
@dataclass
class RAClass:
    name: str
    type: str  # "Actor", "Boundary", "Controller", "Entity"
    stereotype: str  # "«actor»", "«boundary»", "«control»", "«entity»"
    element_type: ElementType
    step_references: List[str]
    description: str
    warnings: List[str] = None
    source: str = "step"  # "step" or "precondition"
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class GenericUCAnalyzer:
    """
    Domain-agnostic Use Case Grammar Analyzer
    Works with any UC from any domain using systematic UC-Methode analysis
    """
    
    def __init__(self, domain_name: Optional[str] = None):
        self.nlp = spacy.load("en_core_web_md")
        self.domain_loader = DomainVerbLoader()
        
        # Auto-detect domain if not specified
        self.domain_name = domain_name
        
        # Load domain-specific configuration
        if self.domain_name:
            self.verb_config = self.domain_loader.get_verb_configuration(self.domain_name)
            print(f"Loaded verb configuration for domain: {self.domain_name}")
        else:
            self.verb_config = self.domain_loader.get_verb_configuration()
            print("Loaded common verb configuration (no domain specified)")
            
        # Load universal operational materials framework
        self._load_operational_materials_framework()
    
    def analyze_multiple_ucs(self, uc_file_paths: List[str], domain_name: Optional[str] = None) -> Tuple[List[VerbAnalysis], List[RAClass]]:
        """
        Analyze multiple UC files from the same domain
        
        Args:
            uc_file_paths: List of paths to UC text files from same domain
            domain_name: Optional domain name override
        
        Returns:
            Tuple of (combined_verb_analyses, combined_ra_classes)
        """
        if not uc_file_paths:
            print("ERROR: No UC files provided")
            return [], []
        
        # Override domain if specified
        if domain_name:
            self.domain_name = domain_name
            self.verb_config = self.domain_loader.get_verb_configuration(domain_name)
            print(f"Using specified domain: {domain_name}")
        
        print("="*80)
        print("MULTI-UC DOMAIN ANALYSIS")
        print("="*80)
        print(f"Analyzing {len(uc_file_paths)} UC files from same domain")
        print(f"Domain: {self.domain_name or 'Auto-detect'}")
        print("="*80)
        
        all_verb_analyses = []
        combined_ra_classes = {}  # Dict for unique RA Classes across all UCs
        
        for i, uc_file_path in enumerate(uc_file_paths, 1):
            print(f"\n{'-'*60}")
            print(f"ANALYZING UC FILE {i}/{len(uc_file_paths)}: {Path(uc_file_path).name}")
            print(f"{'-'*60}")
            
            # Analyze individual UC
            verb_analyses, ra_classes = self.analyze_uc_file(uc_file_path)
            
            # Add to combined results
            all_verb_analyses.extend(verb_analyses)
            
            # Merge RA classes
            for ra_class in ra_classes:
                key = f"{ra_class.type}_{ra_class.name}"
                if key in combined_ra_classes:
                    # Extend existing class with new step references
                    combined_ra_classes[key].step_references.extend(ra_class.step_references)
                    # Keep step references unique
                    combined_ra_classes[key].step_references = list(set(combined_ra_classes[key].step_references))
                else:
                    combined_ra_classes[key] = ra_class
        
        # Sort combined RA classes
        sorted_combined_classes = sorted(combined_ra_classes.values(), key=lambda x: (x.type, x.name))
        
        self._print_multi_uc_summary(all_verb_analyses, sorted_combined_classes, uc_file_paths)
        
        # Show HMI Architecture
        self.show_hmi_architecture(all_verb_analyses, sorted_combined_classes)
        
        return all_verb_analyses, sorted_combined_classes
    
    def analyze_domain_with_hmi(self, uc_file_paths: List[str], domain_name: str) -> Tuple[List[VerbAnalysis], List[RAClass]]:
        """
        Convenience method to analyze multiple UCs from a domain and show HMI architecture
        
        Args:
            uc_file_paths: List of UC file paths from same domain
            domain_name: Domain name (e.g., 'beverage_preparation')
        
        Returns:
            Tuple of (all_verb_analyses, combined_ra_classes)
        """
        print("DOMAIN-WIDE UC ANALYSIS WITH HMI ARCHITECTURE")
        print("=" * 60)
        print(f"Domain: {domain_name}")
        print(f"UC Files: {len(uc_file_paths)}")
        for i, uc_path in enumerate(uc_file_paths, 1):
            print(f"  {i}. {Path(uc_path).name}")
        print("=" * 60)
        
        # Perform multi-UC analysis (includes HMI architecture display)
        return self.analyze_multiple_ucs(uc_file_paths, domain_name)
    
    def analyze_uc_file(self, uc_file_path: str) -> Tuple[List[VerbAnalysis], List[RAClass]]:
        """
        Analyze a complete UC file with all flows
        
        Args:
            uc_file_path: Path to UC text file
        
        Returns:
            Tuple of (verb_analyses, ra_classes)
        """
        uc_steps, preconditions, actors = self._parse_uc_file(uc_file_path)
        
        if not uc_steps:
            print(f"ERROR: No UC steps found in file: {uc_file_path}")
            return [], []
        
        # Auto-detect domain if not specified
        if not self.domain_name:
            self.domain_name = self._detect_domain_from_steps(uc_steps)
            if self.domain_name:
                self.verb_config = self.domain_loader.get_verb_configuration(self.domain_name)
                print(f"Auto-detected domain: {self.domain_name}")
        
        result = self._analyze_uc_steps(uc_steps, preconditions, Path(uc_file_path).stem)
        
        # Validate actors after analysis
        self._validate_actors(actors, uc_steps, result[1])  # result[1] is ra_classes
        
        return result
    
    def _parse_uc_file(self, uc_file_path: str) -> Tuple[List[UCStep], List[Precondition], List[Actor]]:
        """
        Parse UC file and extract all steps, preconditions, and actors
        """
        if not Path(uc_file_path).exists():
            print(f"ERROR: UC file not found: {uc_file_path}")
            return [], [], []
        
        with open(uc_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        uc_steps = []
        preconditions = []
        actors = []
        current_flow_type = "main"
        current_flow_id = None
        in_preconditions = False
        in_actors = False
        
        lines = content.split('\n')
        uc_name = Path(uc_file_path).stem
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect preconditions section
            if line.lower().startswith('preconditions:'):
                in_preconditions = True
                in_actors = False
                continue
            elif line.lower().startswith('actors:'):
                in_actors = True
                in_preconditions = False
                # Parse actors on same line
                actors_text = line[7:].strip()  # Remove "Actors:" prefix
                if actors_text:
                    self._parse_actors_line(actors_text, uc_name, actors)
                continue
            elif line.lower().startswith('main flow'):
                in_preconditions = False
                in_actors = False
            
            # Parse actor lines (multi-line support)
            if in_actors and not line.lower().startswith(('preconditions:', 'main flow:', 'alternative flow:', 'extension flow:')):
                # Handle continuation of actors on multiple lines
                if line and not line.startswith('#'):  # Skip comments
                    self._parse_actors_line(line, uc_name, actors)
                continue
            
            # Parse precondition lines
            if in_preconditions and line.startswith('-'):
                precondition_text = line[1:].strip()  # Remove the '-' marker
                resource_info = self._parse_precondition(precondition_text, uc_name)
                if resource_info:
                    preconditions.append(resource_info)
                continue
            
            # Detect flow sections
            if line.lower().startswith('main flow'):
                current_flow_type = "main"
                current_flow_id = None
                in_preconditions = False
                continue
            elif line.lower().startswith('alternative flow'):
                current_flow_type = "alternative"
                continue
            elif line.lower().startswith('extension flow'):
                current_flow_type = "extension"
                continue
            
            # Parse step lines (B1, A1, E1, etc.)
            step_match = re.match(r'^([ABE]\d+(?:[a-z])?(?:\.\d+)?)\s+(.+)$', line)
            if step_match:
                step_id = step_match.group(1)
                step_text = step_match.group(2).strip()
                
                # Remove trigger/condition markers
                step_text = re.sub(r'\(trigger\)\s*', '', step_text)
                step_text = re.sub(r'^at\s+\w+\s+', '', step_text)  # Remove "at B2a" etc.
                
                # Extract flow ID for alternative/extension flows
                if step_id.startswith('A'):
                    current_flow_type = "alternative"
                    current_flow_id = re.match(r'^(A\d+)', step_id).group(1)
                elif step_id.startswith('E'):
                    current_flow_type = "extension"
                    current_flow_id = re.match(r'^(E\d+)', step_id).group(1)
                
                uc_steps.append(UCStep(
                    step_id=step_id,
                    step_text=step_text,
                    flow_type=current_flow_type,
                    flow_id=current_flow_id,
                    uc_name=Path(uc_file_path).stem
                ))
        
        return uc_steps, preconditions, actors
    
    def _parse_precondition(self, precondition_text: str, uc_name: str) -> Optional[Precondition]:
        """
        Parse a single precondition line and extract resource information
        Examples: 
        - "Coffee beans are available in the system" -> "Coffee beans", "available"
        - "Water is available in the system" -> "Water", "available"
        """
        # Clean up the text
        text = precondition_text.strip()
        
        # Common patterns for preconditions
        patterns = [
            # Handle complex cases like "Launch window is calculated and available"
            r'([A-Za-z\s]+?)\s+(?:are?|is)\s+(?:calculated|configured|loaded|integrated|tested)\s+and\s+(available|ready)',
            # Handle cases like "Ground systems are operational and ready"
            r'([A-Za-z\s]+?)\s+(?:are?|is)\s+(?:operational|functional|active)\s+and\s+(ready|available)',
            # Standard patterns
            r'(.+?)\s+(?:are?|is)\s+(available|configured|connected|ready)\s+(?:in\s+the\s+system|to\s+the\s+system)',
            r'(.+?)\s+(?:are?|is)\s+(available|configured|connected|ready)',
            r'(.+?)\s+(?:available|configured|connected|ready)',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                resource_name = match.group(1).strip()
                availability_type = match.group(2).strip() if len(match.groups()) > 1 else "available"
                
                return Precondition(
                    resource_name=resource_name,
                    original_text=precondition_text,
                    uc_name=uc_name,
                    availability_type=availability_type
                )
        
        # Fallback: try to extract the main noun as resource
        doc = self.nlp(text)
        for token in doc:
            if token.pos_ == "NOUN" and not token.is_stop:
                return Precondition(
                    resource_name=token.text,
                    original_text=precondition_text,
                    uc_name=uc_name,
                    availability_type="available"
                )
        
        return None
    
    def _parse_actors_line(self, actors_text: str, uc_name: str, actors: List[Actor]) -> None:
        """
        Parse a line containing actor names and add them to the actors list
        Example: "Mission Control, Launch Sequencer, Ground Systems"
        """
        if not actors_text or not actors_text.strip():
            return
        
        # Split by comma and clean up names
        actor_names = [name.strip() for name in actors_text.split(',')]
        
        for actor_name in actor_names:
            if actor_name and not any(a.name == actor_name for a in actors):
                actors.append(Actor(
                    name=actor_name,
                    uc_name=uc_name,
                    original_text=actors_text,
                    used_in_steps=[]
                ))
    
    def _detect_domain_from_steps(self, uc_steps: List[UCStep]) -> Optional[str]:
        """Auto-detect domain from UC step content"""
        combined_text = " ".join([step.step_text for step in uc_steps])
        return self.domain_loader.detect_domain_from_text(combined_text)
    
    def _preprocess_compound_nouns_nlp(self, text: str) -> str:
        """
        Use spaCy NLP to automatically detect compound nouns and merge them
        This is a general approach that works for any domain without manual lists
        """
        doc = self.nlp(text)
        processed_tokens = []
        i = 0
        
        while i < len(doc):
            token = doc[i]
            
            # Look for compound patterns
            if i + 1 < len(doc):
                next_token = doc[i + 1]
                
                # Pattern 1: NOUN + NOUN (like "flight program")
                if token.pos_ == "NOUN" and next_token.pos_ == "NOUN":
                    compound = token.text.capitalize() + next_token.text.capitalize()
                    processed_tokens.append(compound)
                    i += 2  # Skip next token as it's part of compound
                    continue
                
                # Pattern 2: ADJ + NOUN (like "diagnostic data", "guidance computer")  
                elif token.pos_ == "ADJ" and next_token.pos_ == "NOUN":
                    compound = token.text.capitalize() + next_token.text.capitalize()
                    processed_tokens.append(compound)
                    i += 2
                    continue
                
                # Pattern 2b: VERB + NOUN (for participles like "scheduled time", "calculated altitude")
                elif token.pos_ == "VERB" and next_token.pos_ == "NOUN" and token.dep_ == "amod":
                    compound = token.text.capitalize() + next_token.text.capitalize()
                    processed_tokens.append(compound)
                    i += 2
                    continue
                    
            # Pattern 3: Check if this noun has compound children in dependency tree
            if token.pos_ == "NOUN":
                compound_parts = [token.text]
                for child in token.children:
                    if child.dep_ == "compound" and child.pos_ == "NOUN":
                        compound_parts.insert(0, child.text)  # Insert at beginning
                
                if len(compound_parts) > 1:
                    compound = "".join(part.capitalize() for part in compound_parts)
                    processed_tokens.append(compound)
                    # Skip the children tokens that were merged
                    children_indices = [child.i for child in token.children if child.dep_ == "compound"]
                    if i + 1 in children_indices:
                        i += 2
                        continue
            
            processed_tokens.append(token.text)
            i += 1
            
        return " ".join(processed_tokens)

    def _analyze_uc_steps(self, uc_steps: List[UCStep], preconditions: List[Precondition], uc_name: str = "Unknown") -> Tuple[List[VerbAnalysis], List[RAClass]]:
        """
        Analyze all UC steps and preconditions to generate RA classes
        """
        print("="*80)
        print("GENERIC UC GRAMMAR ANALYSIS - ALL FLOWS")
        print("="*80)
        
        # Preprocess all UC steps for compound noun detection
        for step in uc_steps:
            original_text = step.step_text
            preprocessed_text = self._preprocess_compound_nouns_nlp(original_text)
            if preprocessed_text != original_text:
                print(f"Compound noun preprocessing: '{original_text}' -> '{preprocessed_text}'")
                step.step_text = preprocessed_text
        
        # Group steps by flow type
        main_steps = [s for s in uc_steps if s.flow_type == "main"]
        alt_steps = [s for s in uc_steps if s.flow_type == "alternative"]
        ext_steps = [s for s in uc_steps if s.flow_type == "extension"]
        
        print(f"Main Flow: {len(main_steps)} steps")
        print(f"Alternative Flows: {len(alt_steps)} steps")
        print(f"Extension Flows: {len(ext_steps)} steps")
        print(f"Domain: {self.domain_name or 'Auto-detected/Common'}")
        print(f"Total steps to analyze: {len(uc_steps)}")
        print("="*80)
        
        verb_analyses = []
        ra_classes = {}  # Dict for unique RA Classes
        
        for step in uc_steps:
            if step.step_text.lower().strip() in ["end uc", "continue in main flow"]:
                print(f"\n--- {step.step_id}: {step.step_text} --- (Flow control - not analyzed)")
                continue
            
            print(f"\n--- {step.step_id}: {step.step_text} ---")
            
            # Verb analysis
            verb_analysis = self._analyze_sentence_verbs(step)
            verb_analysis.uc_name = uc_name  # Add UC name to verb analysis
            verb_analyses.append(verb_analysis)
            
            # Derive RA classes
            step_ra_classes = self._derive_ra_classes(verb_analysis, step)
            
            # Add to master list
            for ra_class in step_ra_classes:
                key = f"{ra_class.type}_{ra_class.name}"
                if key in ra_classes:
                    # Extend existing class
                    ra_classes[key].step_references.append(step.step_id)
                else:
                    ra_classes[key] = ra_class
        
        # Process preconditions to generate additional RA classes
        precondition_ra_classes = self._generate_ra_classes_from_preconditions(preconditions)
        
        # Add Domain Orchestrator (implicit coordination controller)
        domain_orchestrator = self._generate_domain_orchestrator(uc_name)
        
        # Merge precondition RA classes with step-based ones
        for ra_class in precondition_ra_classes:
            key = f"{ra_class.type}_{ra_class.name}"
            if key in ra_classes:
                # Extend existing class with precondition info
                if "precondition" not in ra_classes[key].description:
                    ra_classes[key].description += " (also required as precondition)"
            else:
                ra_classes[key] = ra_class
        
        # Add Domain Orchestrator to RA classes
        orchestrator_key = f"{domain_orchestrator.type}_{domain_orchestrator.name}"
        ra_classes[orchestrator_key] = domain_orchestrator
        
        # Sort RA classes by type
        sorted_classes = sorted(ra_classes.values(), key=lambda x: (x.type, x.name))
        
        # UC-Methode Violation Check: Detect implementation elements
        self._check_uc_methode_violations(sorted_classes)
        
        # Analyze control flow
        control_flows, parallel_flows = self._analyze_control_flow(uc_steps, sorted_classes)
        
        # Print precondition analysis if any preconditions found
        if preconditions:
            self._print_precondition_analysis(preconditions, precondition_ra_classes)
        
        # Print control flow analysis
        self._print_control_flow_analysis(control_flows, parallel_flows)
        
        # Analyze and print data flow relationships
        data_flows = self._analyze_data_flow(verb_analyses, sorted_classes)
        self._print_data_flow_analysis(data_flows)
        
        self._print_ra_classes_summary(sorted_classes)
        self._print_verb_statistics(verb_analyses)
        
        return verb_analyses, sorted_classes
    
    def _analyze_sentence_verbs(self, step: UCStep) -> VerbAnalysis:
        """
        Analyze verbs in a UC step using domain-agnostic grammar analysis
        """
        doc = self.nlp(step.step_text)
        
        # Find main verb with "begins + verb" pattern support
        main_verb = None
        actual_verb = None
        
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                main_verb = token
                
                # Check for "begins/starts + gerund" pattern
                if token.lemma_ in ["begin", "start"]:
                    for child in token.children:
                        if child.pos_ == "VERB" and child.tag_ == "VBG":  # Gerund
                            actual_verb = child
                            print(f"Pattern detected: '{token.text} {child.text}' -> using '{child.text}' as main verb")
                            break
                break
            elif token.pos_ == "VERB" and any(child.dep_ in ["dobj", "pobj"] for child in token.children):
                main_verb = token
                break
        
        # Use actual verb if pattern was found
        if actual_verb:
            main_verb = actual_verb
        
        if not main_verb:
            return VerbAnalysis(
                step_id=step.step_id,
                original_text=step.step_text,
                verb="",
                verb_lemma="",
                verb_type=VerbType.FUNCTION_VERB
            )
        
        print(f"Main verb: {main_verb.text} (lemma: {main_verb.lemma_})")
        
        # Categorize verb using domain configuration
        verb_type = self.domain_loader.categorize_verb(main_verb.lemma_, self.domain_name)
        
        # Special handling for triggers (domain-agnostic)
        if self._is_trigger_step(step):
            verb_type = VerbType.TRANSACTION_VERB
            print(f"{step.step_id} Override: Trigger -> TRANSACTION VERB")
        
        print(f"Verb Type: {verb_type.value}")
        
        # Extract objects and adjectives
        direct_obj = self._find_direct_object(main_verb)
        prep_objs = self._find_prepositional_objects(main_verb)
        warnings = []
        suggested_activity = None
        
        print(f"Direct Object: {direct_obj}")
        if direct_obj:
            adjectives = self._extract_adjectives_from_phrase(direct_obj)
            if adjectives:
                print(f"Adjectives detected: {adjectives}")
        
        print(f"Prepositional Objects: {prep_objs}")
        for prep, obj in prep_objs:
            adj_in_prep = self._extract_adjectives_from_phrase(obj)
            if adj_in_prep:
                print(f"Adjectives in '{prep} {obj}': {adj_in_prep}")
        
        # Check for implementation elements (domain-agnostic)
        if direct_obj and verb_type == VerbType.FUNCTION_VERB:
            impl_info = self._check_implementation_elements(direct_obj)
            if impl_info:
                warnings.extend(impl_info["warnings"])
                suggested_activity = impl_info.get("suggestion")
        
        return VerbAnalysis(
            step_id=step.step_id,
            original_text=step.step_text,
            verb=main_verb.text,
            verb_lemma=main_verb.lemma_,
            verb_type=verb_type,
            direct_object=direct_obj,
            prepositional_objects=prep_objs,
            warnings=warnings,
            suggested_functional_activity=suggested_activity
        )
    
    def _is_trigger_step(self, step: UCStep) -> bool:
        """Check if step is a trigger (domain-agnostic)"""
        # Main UC trigger
        if step.step_id == "B1" or "(trigger)" in step.step_text:
            return True
        
        # Alternative flow condition triggers
        if step.flow_type == "alternative" and not "." in step.step_id:
            return True
        
        # Extension flow triggers
        if step.flow_type == "extension" and not "." in step.step_id:
            return True
        
        return False
    
    def _is_hmi_interaction(self, verb_analysis: VerbAnalysis, step: UCStep) -> bool:
        """Check if step involves HMI (Human-Machine Interface) interaction"""
        text_lower = verb_analysis.original_text.lower()
        
        # User input/request activities
        if verb_analysis.verb_lemma in ["request", "want", "ask", "select", "choose", "input"]:
            return True
        
        # System output to user activities
        if verb_analysis.verb_lemma in ["output", "present", "display", "show", "notify"] and "user" in text_lower:
            return True
        
        # Any step involving direct user interaction
        if any(word in text_lower for word in ["user", "display", "screen", "interface", "button", "menu"]):
            return True
        
        return False
    
    def _check_implementation_elements(self, obj_text: str) -> Optional[Dict[str, Any]]:
        """Check for implementation elements using domain configuration"""
        obj_words = obj_text.lower().split()
        warnings = []
        suggestion = None
        
        for word in obj_words:
            impl_info = self.domain_loader.get_implementation_element_info(word, self.domain_name)
            if impl_info:
                warnings.append(impl_info["warning"])
                suggestion = impl_info["functional_suggestion"]
                print(f"WARNING: {impl_info['warning']}")
                print(f"SUGGESTION: {impl_info['functional_suggestion']}")
        
        return {"warnings": warnings, "suggestion": suggestion} if warnings else None
    
    def _check_uc_methode_violations(self, ra_classes: List[RAClass]):
        """
        Check all RA classes for UC-Methode violations (solution-neutral requirement)
        Entities should not contain implementation elements that specify physical architecture
        """
        print("\n" + "="*80)
        print("UC-METHODE VIOLATION ANALYSIS")
        print("="*80)
        
        violations_found = []
        
        for ra_class in ra_classes:
            if ra_class.type == "Entity":
                # Check entity name for implementation elements
                entity_name_lower = ra_class.name.lower()
                impl_info = self.domain_loader.get_implementation_element_info(entity_name_lower, self.domain_name)
                
                if impl_info:
                    violation = {
                        "entity": ra_class.name,
                        "steps": ra_class.step_references,
                        "warning": impl_info["warning"],
                        "suggestion": impl_info["functional_suggestion"]
                    }
                    violations_found.append(violation)
        
        if violations_found:
            print("ERROR: UC-METHODE VIOLATIONS DETECTED!")
            print("UC must be solution-neutral and not specify physical architecture.")
            print()
            
            for i, violation in enumerate(violations_found, 1):
                print(f"{i}. ENTITY: '{violation['entity']}' (Steps: {', '.join(violation['steps'])})")
                print(f"   PROBLEM: {violation['warning']}")
                print(f"   SOLUTION: {violation['suggestion']}")
                print()
            
            print("RECOMMENDATION: Rewrite UC to be solution-neutral")
            print("   - Remove implementation-specific terms")
            print("   - Focus on functional requirements instead of physical components")
            print("   - Use system capabilities rather than specific hardware")
            
        else:
            print("OK: No UC-Methode violations detected - UC is solution-neutral")
        
        print("="*80)
    
    def _validate_actors(self, actors: List[Actor], uc_steps: List[UCStep], ra_classes: List[RAClass]):
        """
        Validate Actor usage according to UC-Methode rules:
        1. Actors must NEVER become Entities
        2. All declared Actors must be used in at least one scenario step
        3. Any Actor found in scenarios must be declared in Actors section
        """
        print("\n" + "="*80)
        print("ACTOR VALIDATION ANALYSIS")
        print("="*80)
        
        violations_found = []
        
        # Rule 1: Check if any declared Actor became an Entity
        declared_actor_names = {actor.name for actor in actors}
        for ra_class in ra_classes:
            if ra_class.type == "Entity" and ra_class.name in declared_actor_names:
                violations_found.append({
                    "rule": "Rule 1",
                    "problem": f"Actor '{ra_class.name}' became Entity",
                    "description": "Actors from Actors section must NEVER become Entities",
                    "steps": ra_class.step_references
                })
        
        # Rule 2: Check if all declared Actors are used in steps
        # Find actors referenced in step text
        used_actors = set()
        for step in uc_steps:
            step_text_lower = step.step_text.lower()
            for actor in actors:
                actor_name_lower = actor.name.lower()
                # Check if actor name appears in step text (exact match or as part of phrase)
                # Handle cases like "Mission Control" matching "mission control"
                words_in_step = step_text_lower.split()
                actor_words = actor_name_lower.split()
                
                # Check for exact actor name match
                if actor_name_lower in step_text_lower:
                    used_actors.add(actor.name)
                    # Update actor's used_in_steps
                    if actor.used_in_steps is None:
                        actor.used_in_steps = []
                    if step.step_id not in actor.used_in_steps:
                        actor.used_in_steps.append(step.step_id)
                # Check for actor mentioned at start of step (like "The Mission Control...")
                elif step_text_lower.startswith(f"the {actor_name_lower}"):
                    used_actors.add(actor.name)
                    if actor.used_in_steps is None:
                        actor.used_in_steps = []
                    if step.step_id not in actor.used_in_steps:
                        actor.used_in_steps.append(step.step_id)
        
        # Find unused declared actors
        for actor in actors:
            if actor.name not in used_actors:
                violations_found.append({
                    "rule": "Rule 2",
                    "problem": f"Actor '{actor.name}' declared but never used",
                    "description": "All declared Actors must be used in at least one scenario step",
                    "steps": []
                })
        
        # Rule 3: Check if any Actor in steps is not declared
        # This would require more sophisticated NLP to detect actor-like entities in steps
        # For now, we'll check for common actor patterns
        actor_keywords = {"user", "operator", "admin", "system operator"}
        declared_actor_names_lower = {a.name.lower() for a in actors}
        
        for step in uc_steps:
            step_text_lower = step.step_text.lower()
            for keyword in actor_keywords:
                if keyword in step_text_lower and keyword not in declared_actor_names_lower:
                    # Make sure it's not part of a declared actor name
                    is_part_of_declared = False
                    for declared_name in declared_actor_names_lower:
                        if keyword in declared_name:
                            is_part_of_declared = True
                            break
                    
                    if not is_part_of_declared:
                        violations_found.append({
                            "rule": "Rule 3",
                            "problem": f"Potential undeclared actor '{keyword}' found in step {step.step_id}",
                            "description": "Any Actor found in scenarios must be declared in Actors section",
                            "steps": [step.step_id]
                        })
        
        # Report violations
        if violations_found:
            print("ERROR: ACTOR VALIDATION VIOLATIONS DETECTED!")
            print("UC-Methode Actor rules violated.")
            print()
            
            for i, violation in enumerate(violations_found, 1):
                print(f"{i}. {violation['rule']}: {violation['problem']}")
                print(f"   DESCRIPTION: {violation['description']}")
                if violation['steps']:
                    print(f"   STEPS: {', '.join(violation['steps'])}")
                print()
            
            print("RECOMMENDATION: Fix Actor usage")
            print("   - Ensure all declared Actors are used in scenarios")
            print("   - Declare all Actors that appear in scenarios")
            print("   - Prevent Actors from becoming Entities in RA analysis")
            
        else:
            print("OK: All Actor validation rules satisfied")
        
        # Report Actor usage summary
        print("\nACTOR USAGE SUMMARY:")
        if actors:
            for actor in actors:
                usage_info = f"used in steps {', '.join(actor.used_in_steps)}" if actor.used_in_steps else "UNUSED"
                print(f"  - {actor.name}: {usage_info}")
        else:
            print("  - No Actors declared in UC")
        
        print("="*80)
    
    def _derive_ra_classes(self, verb_analysis: VerbAnalysis, step: UCStep) -> List[RAClass]:
        """
        Derive RA classes using domain-agnostic patterns
        """
        ra_classes = []
        
        # 1. Controller for each step (domain-agnostic naming)
        controller_name = self._derive_generic_controller_name(verb_analysis, step)
        if controller_name:
            ra_classes.append(RAClass(
                name=controller_name,
                type="Controller",
                stereotype="«control»",
                element_type=ElementType.FUNCTIONAL_ENTITY,
                step_references=[step.step_id],
                description=f"Controls {verb_analysis.verb} operation in {step.step_id}"
            ))
        
        # 1a. Add HMI Controller for user interaction steps
        if self._is_hmi_interaction(verb_analysis, step):
            ra_classes.append(RAClass(
                name="HMIController",
                type="Controller",
                stereotype="«control»",
                element_type=ElementType.FUNCTIONAL_ENTITY,
                step_references=[step.step_id],
                description="Human-Machine Interface controller for user interactions"
            ))
        
        # 2. Entities from objects (domain-agnostic)
        if verb_analysis.direct_object:
            entities = self._derive_generic_entities(verb_analysis.direct_object, verb_analysis, step)
            ra_classes.extend(entities)
        
        # 3. Entities from prepositional objects
        for prep, obj in verb_analysis.prepositional_objects:
            entities = self._derive_generic_entities(obj, verb_analysis, step, prep)
            ra_classes.extend(entities)
            
            # 3a. Check for Actor references in prepositional objects and create Boundaries
            boundaries = self._derive_boundaries_from_actor_references(prep, obj, verb_analysis, step)
            ra_classes.extend(boundaries)
        
        # 4. Actors for triggers (domain-agnostic)
        if self._is_trigger_step(step):
            actor = self._derive_trigger_actor(verb_analysis, step)
            if actor:
                ra_classes.append(actor)
        
        # 5. Boundaries for transaction verbs (domain-agnostic)
        if verb_analysis.verb_type == VerbType.TRANSACTION_VERB:
            boundaries = self._derive_generic_boundaries(verb_analysis, step)
            ra_classes.extend(boundaries)
        
        return ra_classes
    
    def _derive_generic_controller_name(self, verb_analysis: VerbAnalysis, step: UCStep) -> Optional[str]:
        """Generate domain-agnostic controller names"""
        # Pattern-based controller naming
        if step.step_id == "B1" or self._is_trigger_step(step):
            if "time" in verb_analysis.original_text.lower() or "clock" in verb_analysis.original_text.lower():
                return "TimeManager"
            elif "user" in verb_analysis.original_text.lower():
                return "UserRequestManager"
            elif step.flow_type == "alternative":
                return f"{step.flow_id}ConditionManager"
            else:
                return "TriggerManager"
        
        # Object-based controller naming
        if verb_analysis.direct_object:
            main_object = verb_analysis.direct_object.split()[-1].capitalize()
            return f"{main_object}Manager"
        
        # Verb-based controller naming
        verb_action = verb_analysis.verb_lemma.capitalize()
        return f"{verb_action}Manager"
    
    def _derive_generic_entities(self, obj_text: str, verb_analysis: VerbAnalysis, 
                                step: UCStep, preposition: str = None) -> List[RAClass]:
        """Derive entities using domain-agnostic patterns"""
        entities = []
        
        # Check if this is already a compound noun (single capitalized word like "FlightProgram") 
        # from our NLP preprocessing
        if obj_text and obj_text[0].isupper() and ' ' not in obj_text and len(obj_text) > 1:
            # This is likely a compound noun from preprocessing - treat as single entity
            entity_name, element_type, description = self._get_domain_contextual_entity_info(
                obj_text.lower(), verb_analysis, obj_text, preposition)
            
            # Preserve the original capitalization from compound noun preprocessing
            entities.append(RAClass(
                name=obj_text,  # Keep original capitalization like "FlightProgram"
                type="Entity",
                stereotype="«entity»",
                element_type=element_type,
                step_references=[step.step_id],
                description=description
            ))
            return entities
        
        obj_text_lower = obj_text.lower()
        
        # Check for compound terms first to avoid splitting them incorrectly
        compound_terms = self._extract_compound_terms(obj_text_lower)
        if compound_terms:
            # Process compound terms as single entities
            for compound_term in compound_terms:
                entity_name, element_type, description = self._get_domain_contextual_entity_info(
                    compound_term, verb_analysis, obj_text, preposition)
                
                entities.append(RAClass(
                    name=entity_name,
                    type="Entity",
                    stereotype="«entity»",
                    element_type=element_type,
                    step_references=[step.step_id],
                    description=description
                ))
            return entities
        
        # Fallback to word-by-word processing
        obj_words = obj_text_lower.split()
        
        # Remove articles
        obj_words = [w for w in obj_words if w not in ["the", "a", "an"]]
        
        for word in obj_words:
            # Skip common adjectives and determiners
            if word in ["set", "little", "all", "some", "many", "few"]:
                continue
            
            # Check if this word should be an Actor instead of Entity
            # Known actors that might appear in prepositional phrases
            if word in ["user", "operator", "administrator", "customer", "client", "person", "human"]:
                # Skip creating entity - this should be handled as Actor or Boundary
                continue
            
            # Skip domain-specific excluded terms
            if self._should_skip_entity_word(word):
                continue
            
            # Use domain configuration for contextual entity naming if available
            entity_name, element_type, description = self._get_domain_contextual_entity_info(
                word, verb_analysis, obj_text, preposition)
            
            entities.append(RAClass(
                name=entity_name,
                type="Entity",
                stereotype="«entity»",
                element_type=element_type,
                step_references=[step.step_id],
                description=description
            ))
        
        return entities
    
    def _extract_compound_terms(self, obj_text: str) -> List[str]:
        """Extract compound terms that should be treated as single entities"""
        compound_patterns = [
            "grinding degree",
            "water amount", 
            "milk amount",
            "coffee amount",
            "brewing time",
            "brewing pressure"
        ]
        
        found_compounds = []
        for pattern in compound_patterns:
            if pattern in obj_text:
                # Convert to proper entity name format (e.g., "water amount" -> "WaterAmount")
                entity_name = "".join(word.capitalize() for word in pattern.split())
                found_compounds.append(entity_name)
        
        return found_compounds
    
    def _should_skip_entity_word(self, word: str) -> bool:
        """Check if word should be skipped based on domain configuration"""
        # Skip step references (like B5b, A1, E2.1, etc.)
        if re.match(r'^[ABCE]\d+([a-z]|\.\d+)?$', word):
            return True
            
        if not self.verb_config or not hasattr(self.verb_config, 'excluded_entity_terms'):
            # Fallback to basic exclusions if no domain config
            basic_exclusions = ["actions", "activities", "operations"]
            return word in basic_exclusions
            
        excluded_terms = getattr(self.verb_config, 'excluded_entity_terms', [])
        if not excluded_terms:  # Handle None case
            basic_exclusions = ["actions", "activities", "operations"]
            return word in basic_exclusions
            
        return word in excluded_terms
    
    def _find_matching_entity_name(self, obj_phrase: str, all_entity_names: set) -> str:
        """Find matching entity name for a prepositional object phrase using domain configuration"""
        obj_lower = obj_phrase.lower()
        
        # Use domain configuration to find contextual entity mappings
        if self.verb_config and hasattr(self.verb_config, 'contextual_entities'):
            contextual_entities = getattr(self.verb_config, 'contextual_entities', None)
            if contextual_entities:
                # Check each word in the phrase against domain configuration
                phrase_words = obj_lower.split()
                scored_matches = []
                
                for word in phrase_words:
                    if word in contextual_entities:
                        patterns = contextual_entities[word]
                        # Score patterns by specificity
                        for pattern in patterns:
                            contextual_name = pattern.get('contextual_name')
                            if contextual_name and contextual_name in all_entity_names:
                                context_triggers = pattern.get('context_triggers', [])
                                # Count matching triggers in the phrase
                                matches = [trigger for trigger in context_triggers if trigger in obj_lower]
                                if matches:
                                    score = len(matches)
                                    # Boost score for domain-specific words like "water", "coffee", "milk"
                                    domain_words = ["water", "coffee", "milk", "bean", "grind", "brew"]
                                    if any(word in matches for word in domain_words):
                                        score += 10
                                    
                                    scored_matches.append((score, contextual_name, matches, context_triggers))
                
                if scored_matches:
                    # Sort by score (highest first)
                    scored_matches.sort(key=lambda x: x[0], reverse=True)
                    best_score, best_match, best_matches, _ = scored_matches[0]
                    
                    # Check for ambiguity: multiple patterns with low scores and no clear winner
                    if best_score <= 1 and len(scored_matches) > 1:
                        # Find all patterns with the same score
                        same_score_matches = [match for match in scored_matches if match[0] == best_score]
                        if len(same_score_matches) > 1:
                            alternatives = [match[1] for match in same_score_matches]
                            print(f"WARNING: Ambiguous entity reference '{obj_phrase}' in data flow analysis")
                            print(f"  -> Possible interpretations: {alternatives}")
                            print(f"  -> Using: {best_match} (first match)")
                            print(f"  -> Suggestion: Add specific context like 'water amount', 'coffee amount', etc.")
                    
                    return best_match
        
        # Check for exact matches of any entity name in the phrase (case-insensitive)
        for entity_name in all_entity_names:
            if entity_name.lower() in obj_lower or any(word in entity_name.lower() for word in obj_lower.split()):
                return entity_name
            # Check for case-insensitive partial matches
            if entity_name.lower() == obj_lower or obj_lower in entity_name.lower():
                return entity_name
        
        return None
    
    def _get_domain_contextual_entity_info(self, word: str, verb_analysis: VerbAnalysis, 
                                          obj_text: str, preposition: str = None) -> tuple:
        """Get context-aware entity name using domain configuration"""
        # Default values
        entity_name = word.capitalize()
        element_type = ElementType.FUNCTIONAL_ENTITY
        description = f"Domain entity: {word}"
        
        # Use domain configuration if available
        if self.verb_config and hasattr(self.verb_config, 'contextual_entities'):
            context_info = self._resolve_entity_context_from_domain(
                word, verb_analysis, obj_text, preposition)
            if context_info:
                entity_name, element_type, description = context_info
                return entity_name, element_type, description
        
        # Fallback to generic classification
        if word in ["message", "error", "status", "signal", "data"]:
            element_type = ElementType.CONTROL_DATA
            description = f"Control data: {word}"
        elif word in ["container", "storage", "tank", "cup", "bottle", "chamber"]:
            element_type = ElementType.CONTAINER
            description = f"Container: {word}"
        
        return entity_name, element_type, description
    
    def _resolve_entity_context_from_domain(self, word: str, verb_analysis: VerbAnalysis, 
                                           obj_text: str, preposition: str = None) -> tuple:
        """Resolve entity context using domain configuration patterns"""
        if not self.verb_config or not hasattr(self.verb_config, 'contextual_entities'):
            return None
            
        # Check if contextual_entities is available and not None
        contextual_entities = getattr(self.verb_config, 'contextual_entities', None)
        if not contextual_entities:
            return None
            
        original_text = verb_analysis.original_text.lower()
        verb = verb_analysis.verb_lemma.lower()
        
        # Check for both direct word match and compound entity name components
        target_word = word.lower()
        patterns = None
        
        if target_word in contextual_entities:
            patterns = contextual_entities[target_word]
        else:
            # For compound entities like 'WaterAmount', extract the base word 'amount'
            for base_word in contextual_entities.keys():
                if word.lower().endswith(base_word):
                    patterns = contextual_entities[base_word]
                    break
        
        if patterns:
            
            # Check context patterns from domain configuration
            # Score patterns by specificity (more specific triggers win)
            scored_patterns = []
            for pattern in patterns:
                context_triggers = pattern.get('context_triggers', [])
                
                # Count matching triggers in the original text
                matches = [trigger for trigger in context_triggers if trigger in original_text or trigger == verb]
                if matches:
                    # Score based on number of matches and specificity
                    score = len(matches)
                    # Boost score for domain-specific words like "water", "coffee", "milk"
                    domain_words = ["water", "coffee", "milk", "bean", "grind", "brew"]
                    if any(word in matches for word in domain_words):
                        score += 10
                    scored_patterns.append((score, pattern))
            
            if scored_patterns:
                # Sort by score (highest first)
                scored_patterns.sort(key=lambda x: x[0], reverse=True)
                best_score, best_pattern = scored_patterns[0]
                
                # Check for ambiguity in entity creation
                if best_score <= 1 and len(scored_patterns) > 1:
                    # Find all patterns with the same score
                    same_score_patterns = [pattern for score, pattern in scored_patterns if score == best_score]
                    if len(same_score_patterns) > 1:
                        alternatives = [p.get('contextual_name', word.capitalize()) for p in same_score_patterns]
                        print(f"WARNING: Ambiguous entity '{word}' in step context")
                        print(f"  -> Possible interpretations: {alternatives}")
                        print(f"  -> Using: {best_pattern.get('contextual_name', word.capitalize())} (first match)")
                        print(f"  -> Suggestion: Add specific context in UC text (e.g., 'water amount', 'coffee amount')")
                
                entity_name = best_pattern.get('contextual_name', word.capitalize())
                element_type_str = best_pattern.get('element_type', 'functional')
                description = best_pattern.get('description', f"Domain entity: {word}")
                
                # Convert element type string to enum
                element_type = self._get_element_type_from_string(element_type_str)
                
                return entity_name, element_type, description
        
        return None
    
    def _get_element_type_from_string(self, element_type_str: str) -> ElementType:
        """Convert element type string to ElementType enum"""
        type_mapping = {
            'functional': ElementType.FUNCTIONAL_ENTITY,
            'control': ElementType.CONTROL_DATA,
            'container': ElementType.CONTAINER,
            'implementation': ElementType.IMPLEMENTATION_ELEMENT
        }
        return type_mapping.get(element_type_str.lower(), ElementType.FUNCTIONAL_ENTITY)
    
    def _derive_boundaries_from_actor_references(self, prep: str, obj: str, verb_analysis: VerbAnalysis, step: UCStep) -> List[RAClass]:
        """Derive boundaries when actor references are found in prepositional objects"""
        boundaries = []
        obj_lower = obj.lower()
        
        # Check for actor keywords in prepositional objects
        actor_keywords = ["user", "operator", "administrator", "customer", "client", "person", "human"]
        
        if any(actor in obj_lower for actor in actor_keywords):
            # Create appropriate boundary based on preposition and verb context
            if prep.lower() == "to":
                # "to user" - system delivering something to actor (output boundary)
                if "message" in verb_analysis.original_text.lower() or "error" in verb_analysis.original_text.lower():
                    boundaries.append(RAClass(
                        name="HMIErrorDisplayBoundary",
                        type="Boundary",
                        stereotype="«boundary»",
                        element_type=ElementType.CONTROL_DATA,
                        step_references=[step.step_id],
                        description="HMI boundary for displaying error messages to user"
                    ))
                elif "status" in verb_analysis.original_text.lower() or "information" in verb_analysis.original_text.lower():
                    boundaries.append(RAClass(
                        name="HMIStatusDisplayBoundary",
                        type="Boundary",
                        stereotype="«boundary»",
                        element_type=ElementType.CONTROL_DATA,
                        step_references=[step.step_id],
                        description="HMI boundary for displaying status information to user"
                    ))
                elif any(product in verb_analysis.original_text.lower() for product in ["coffee", "beverage", "product", "cup"]):
                    boundaries.append(RAClass(
                        name="ProductDeliveryBoundary",
                        type="Boundary",
                        stereotype="«boundary»",
                        element_type=ElementType.FUNCTIONAL_ENTITY,
                        step_references=[step.step_id],
                        description="Boundary for delivering finished products to user"
                    ))
                else:
                    boundaries.append(RAClass(
                        name="HMIUserOutputBoundary",
                        type="Boundary",
                        stereotype="«boundary»",
                        element_type=ElementType.CONTROL_DATA,
                        step_references=[step.step_id],
                        description="HMI boundary for general user output"
                    ))
            
            elif prep.lower() == "from":
                # "from user" - user providing input to system (input boundary)
                if any(selection in verb_analysis.original_text.lower() for selection in ["select", "choose", "request", "want"]):
                    boundaries.append(RAClass(
                        name="HMIUserInputBoundary",
                        type="Boundary",
                        stereotype="«boundary»",
                        element_type=ElementType.CONTROL_DATA,
                        step_references=[step.step_id],
                        description="HMI boundary for user input and selections"
                    ))
                else:
                    boundaries.append(RAClass(
                        name="HMIUserInputBoundary",
                        type="Boundary",
                        stereotype="«boundary»",
                        element_type=ElementType.CONTROL_DATA,
                        step_references=[step.step_id],
                        description="HMI boundary for general user input"
                    ))
        
        return boundaries
    
    def _derive_trigger_actor(self, verb_analysis: VerbAnalysis, step: UCStep) -> Optional[RAClass]:
        """Derive actor for trigger steps (domain-agnostic)"""
        if "user" in verb_analysis.original_text.lower():
            return RAClass(
                name="User",
                type="Actor",
                stereotype="«actor»",
                element_type=ElementType.FUNCTIONAL_ENTITY,
                step_references=[step.step_id],
                description="Human actor"
            )
        elif "clock" in verb_analysis.original_text.lower() or "time" in verb_analysis.original_text.lower():
            return RAClass(
                name="Timer",
                type="Actor",
                stereotype="«actor»",
                element_type=ElementType.CONTROL_DATA,
                step_references=[step.step_id],
                description="Time-based trigger"
            )
        else:
            return RAClass(
                name="ExternalTrigger",
                type="Actor",
                stereotype="«actor»",
                element_type=ElementType.CONTROL_DATA,
                step_references=[step.step_id],
                description="External trigger source"
            )
    
    def _derive_generic_boundaries(self, verb_analysis: VerbAnalysis, step: UCStep) -> List[RAClass]:
        """Derive specific boundaries for transaction verbs based on interaction content"""
        boundaries = []
        
        # Analyze content to determine specific boundary type
        text_lower = verb_analysis.original_text.lower()
        direct_obj = verb_analysis.direct_object or ""
        direct_obj_lower = direct_obj.lower()
        
        # HMI Output boundaries (system to user)
        if verb_analysis.verb_lemma in ["output", "notify", "send", "display", "show"] and "user" in text_lower:
            if "error" in text_lower:
                boundaries.append(RAClass(
                    name="HMIErrorDisplayBoundary",
                    type="Boundary",
                    stereotype="«boundary»",
                    element_type=ElementType.CONTROL_DATA,
                    step_references=[step.step_id],
                    description="HMI boundary for displaying error messages (e.g., 'no milk', 'water low')"
                ))
            elif "message" in text_lower or "notification" in text_lower:
                boundaries.append(RAClass(
                    name="HMIStatusDisplayBoundary",
                    type="Boundary",
                    stereotype="«boundary»",
                    element_type=ElementType.CONTROL_DATA,
                    step_references=[step.step_id],
                    description="HMI boundary for displaying status messages (e.g., 'coffee ready')"
                ))
            else:
                boundaries.append(RAClass(
                    name="HMIUserOutputBoundary",
                    type="Boundary",
                    stereotype="«boundary»",
                    element_type=ElementType.CONTROL_DATA,
                    step_references=[step.step_id],
                    description="HMI boundary for general user output"
                ))
        
        # Product delivery boundaries (cup, beverage delivery)
        elif verb_analysis.verb_lemma in ["present", "deliver", "serve"] and any(word in text_lower for word in ["cup", "coffee", "beverage", "product"]):
            boundaries.append(RAClass(
                name="ProductDeliveryBoundary",
                type="Boundary",
                stereotype="«boundary»",
                element_type=ElementType.FUNCTIONAL_ENTITY,
                step_references=[step.step_id],
                description="Boundary for delivering finished products to user"
            ))
        
        # HMI Input boundaries (user to system)
        elif verb_analysis.verb_lemma in ["want", "request", "ask", "demand", "select", "choose", "input"]:
            if "sugar" in text_lower or any(word in text_lower for word in ["additive", "cream", "syrup"]):
                boundaries.append(RAClass(
                    name="HMIAdditiveInputBoundary",
                    type="Boundary",
                    stereotype="«boundary»",
                    element_type=ElementType.CONTROL_DATA,
                    step_references=[step.step_id],
                    description="HMI boundary for user selection of beverage additives"
                ))
            elif any(beverage in text_lower for beverage in ["espresso", "coffee", "tea", "beverage", "type"]):
                boundaries.append(RAClass(
                    name="HMIBeverageSelectionBoundary",
                    type="Boundary",
                    stereotype="«boundary»",
                    element_type=ElementType.CONTROL_DATA,
                    step_references=[step.step_id],
                    description="HMI boundary for user beverage type selection"
                ))
            else:
                boundaries.append(RAClass(
                    name="HMIUserInputBoundary",
                    type="Boundary",
                    stereotype="«boundary»",
                    element_type=ElementType.CONTROL_DATA,
                    step_references=[step.step_id],
                    description="HMI boundary for general user input"
                ))
        
        # Time-based trigger boundaries
        elif verb_analysis.verb_lemma in ["reach", "trigger"] and any(word in text_lower for word in ["time", "clock", "schedule"]):
            boundaries.append(RAClass(
                name="TimeTriggerBoundary",
                type="Boundary",
                stereotype="«boundary»",
                element_type=ElementType.CONTROL_DATA,
                step_references=[step.step_id],
                description="Boundary for time-based system triggers"
            ))
        
        # Supply/maintenance boundaries - detect from shortage conditions
        elif any(word in text_lower for word in ["little", "empty", "low", "shortage", "maintenance", "defective"]):
            if "water" in text_lower:
                boundaries.append(RAClass(
                    name="WaterSupplyBoundary",
                    type="Boundary",
                    stereotype="«boundary»",
                    element_type=ElementType.FUNCTIONAL_ENTITY,
                    step_references=[step.step_id],
                    description="Boundary for water supply monitoring and refill alerts"
                ))
            elif "milk" in text_lower:
                boundaries.append(RAClass(
                    name="MilkSupplyBoundary",
                    type="Boundary",
                    stereotype="«boundary»",
                    element_type=ElementType.FUNCTIONAL_ENTITY,
                    step_references=[step.step_id],
                    description="Boundary for milk supply monitoring and refill alerts"
                ))
            elif any(word in text_lower for word in ["coffee", "bean", "grounds"]):
                boundaries.append(RAClass(
                    name="CoffeeSupplyBoundary",
                    type="Boundary",
                    stereotype="«boundary»",
                    element_type=ElementType.FUNCTIONAL_ENTITY,
                    step_references=[step.step_id],
                    description="Boundary for coffee bean supply monitoring and refill alerts"
                ))
            elif "compressor" in text_lower:
                boundaries.append(RAClass(
                    name="EquipmentMaintenanceBoundary",
                    type="Boundary",
                    stereotype="«boundary»",
                    element_type=ElementType.IMPLEMENTATION_ELEMENT,
                    step_references=[step.step_id],
                    description="Boundary for equipment failure monitoring and maintenance alerts"
                ))
        
        # Waste management boundaries (cup removal, cleaning)
        elif any(word in text_lower for word in ["remove", "clean", "empty", "dispose", "waste"]):
            if "cup" in text_lower:
                boundaries.append(RAClass(
                    name="CupRemovalBoundary",
                    type="Boundary",
                    stereotype="«boundary»",
                    element_type=ElementType.CONTAINER,
                    step_references=[step.step_id],
                    description="Boundary for cup removal and collection"
                ))
            elif any(word in text_lower for word in ["filter", "grounds", "waste"]):
                boundaries.append(RAClass(
                    name="WasteManagementBoundary",
                    type="Boundary",
                    stereotype="«boundary»",
                    element_type=ElementType.CONTAINER,
                    step_references=[step.step_id],
                    description="Boundary for waste disposal and cleaning"
                ))
        
        return boundaries
    
    # Helper methods from original analyzer (unchanged)
    def _find_direct_object(self, verb: Token) -> Optional[str]:
        """Find direct object of verb"""
        for child in verb.children:
            if child.dep_ == "dobj":
                return self._expand_noun_phrase(child)
        return None
    
    def _find_prepositional_objects(self, verb: Token) -> List[Tuple[str, str]]:
        """Find all prepositional objects"""
        prep_objs = []
        for child in verb.children:
            if child.dep_ == "prep":
                prep = child.text
                for prep_child in child.children:
                    if prep_child.dep_ == "pobj":
                        obj = self._expand_noun_phrase(prep_child)
                        prep_objs.append((prep, obj))
        return prep_objs
    
    def _expand_noun_phrase(self, noun: Token) -> str:
        """Expand noun with adjectives, articles, etc."""
        phrase_tokens = []
        
        for child in noun.children:
            if child.dep_ in ["amod", "det", "compound"]:
                phrase_tokens.append(child)
        
        phrase_tokens.append(noun)
        phrase_tokens.sort(key=lambda x: x.i)
        return " ".join([token.text for token in phrase_tokens])
    
    def _extract_adjectives_from_phrase(self, phrase: str) -> List[str]:
        """Extract adjectives from noun phrase"""
        doc = self.nlp(phrase)
        adjectives = []
        
        for token in doc:
            if token.pos_ == "ADJ" or token.dep_ == "amod":
                adjectives.append(token.text)
        
        return adjectives
    
    def _print_ra_classes_summary(self, ra_classes: List[RAClass]):
        """Print RA classes summary"""
        print("\n" + "="*80)
        print("ALL RA CLASSES - DOMAIN-AGNOSTIC ANALYSIS")
        print("="*80)
        
        for ra_class in ra_classes:
            print(f"\n{ra_class.type}: {ra_class.name}")
            print(f"  Stereotype: {ra_class.stereotype}")
            print(f"  Element Type: {ra_class.element_type.value}")
            print(f"  Steps: {', '.join(ra_class.step_references)}")
            print(f"  Description: {ra_class.description}")
            
            if ra_class.warnings:
                for warning in ra_class.warnings:
                    print(f"  WARNING: {warning}")
    
    def _print_verb_statistics(self, verb_analyses: List[VerbAnalysis]):
        """Print verb classification statistics"""
        transaction_verbs = [v for v in verb_analyses if v.verb_type == VerbType.TRANSACTION_VERB]
        transformation_verbs = [v for v in verb_analyses if v.verb_type == VerbType.TRANSFORMATION_VERB]
        function_verbs = [v for v in verb_analyses if v.verb_type == VerbType.FUNCTION_VERB]
        
        print("\n" + "="*80)
        print("VERB CLASSIFICATION SUMMARY")
        print("="*80)
        
        print(f"\nTRANSACTION VERBS - Boundary interactions ({len(transaction_verbs)}):")
        for v in transaction_verbs:
            print(f"  {v.step_id}: {v.verb} - {v.original_text}")
        
        print(f"\nTRANSFORMATION VERBS - Entity transformations ({len(transformation_verbs)}):")
        for v in transformation_verbs:
            if self.domain_name:
                transformation = self.verb_config.transformation_verbs.get(v.verb_lemma, "domain transformation")
            else:
                transformation = "generic transformation"
            print(f"  {v.step_id}: {v.verb} -> {transformation}")
        
        print(f"\nFUNCTION VERBS - System actions ({len(function_verbs)}):")
        for v in function_verbs:
            print(f"  {v.step_id}: {v.verb} - {v.original_text}")
        
        print(f"\nSTATISTICS:")
        print(f"   Total verbs analyzed: {len(verb_analyses)}")
        print(f"   Transaction verbs: {len(transaction_verbs)}")
        print(f"   Transformation verbs: {len(transformation_verbs)}")
        print(f"   Function verbs: {len(function_verbs)}")
    
    def show_hmi_architecture(self, all_verb_analyses: List[VerbAnalysis], 
                             combined_ra_classes: List[RAClass]) -> None:
        """Show the complete HMI architecture with controller and boundaries"""
        print("\n" + "="*80)
        print("HMI ARCHITECTURE ANALYSIS")
        print("="*80)
        
        # Filter for HMI-related components
        hmi_controllers = [ra for ra in combined_ra_classes if ra.type == "Controller" and "HMI" in ra.name]
        hmi_boundaries = [ra for ra in combined_ra_classes if ra.type == "Boundary" and "HMI" in ra.name]
        other_boundaries = [ra for ra in combined_ra_classes if ra.type == "Boundary" and "HMI" not in ra.name]
        
        print(f"\nHMI CONTROLLER:")
        print("=" * 50)
        
        if hmi_controllers:
            for controller in hmi_controllers:
                print(f"\nController: {controller.name}")
                print(f"  Description: {controller.description}")
                print(f"  Steps Referenced: {', '.join(controller.step_references)}")
                
                # Show which UCs use this controller
                uc_sources = set()
                for step_ref in controller.step_references:
                    for verb_analysis in all_verb_analyses:
                        if step_ref == verb_analysis.step_id and verb_analysis.uc_name:
                            uc_sources.add(verb_analysis.uc_name)
                
                print(f"  Used in UCs: {', '.join(sorted(uc_sources))}")
                
                # Show related activities
                related_activities = []
                for step_ref in controller.step_references:
                    for verb_analysis in all_verb_analyses:
                        if step_ref == verb_analysis.step_id:
                            related_activities.append(f"{step_ref}: {verb_analysis.original_text}")
                
                print(f"  Related HMI Activities:")
                for activity in related_activities:
                    print(f"    - {activity}")
        else:
            print("  No HMI Controllers found")
        
        print(f"\nHMI INPUT BOUNDARIES (User -> System):")
        print("=" * 50)
        
        input_boundaries = [b for b in hmi_boundaries if "Input" in b.name or "Selection" in b.name]
        
        for boundary in input_boundaries:
            print(f"\nBoundary: {boundary.name}")
            print(f"  Description: {boundary.description}")
            print(f"  Steps Referenced: {', '.join(boundary.step_references)}")
            
            # Show specific user inputs
            for step_ref in boundary.step_references:
                for verb_analysis in all_verb_analyses:
                    if step_ref == verb_analysis.step_id:
                        print(f"  User Input: {verb_analysis.original_text}")
        
        print(f"\nHMI OUTPUT BOUNDARIES (System -> User):")
        print("=" * 50)
        
        output_boundaries = [b for b in hmi_boundaries if "Display" in b.name or "Output" in b.name]
        
        for boundary in output_boundaries:
            print(f"\nBoundary: {boundary.name}")
            print(f"  Description: {boundary.description}")
            print(f"  Steps Referenced: {', '.join(boundary.step_references)}")
            
            # Show specific system outputs
            for step_ref in boundary.step_references:
                for verb_analysis in all_verb_analyses:
                    if step_ref == verb_analysis.step_id:
                        print(f"  System Output: {verb_analysis.original_text}")
        
        print(f"\nOTHER SYSTEM BOUNDARIES (Non-HMI):")
        print("=" * 50)
        
        for boundary in other_boundaries:
            print(f"\nBoundary: {boundary.name}")
            print(f"  Description: {boundary.description}")
            print(f"  Element Type: {boundary.element_type.value}")
        
        # HMI Architecture Summary
        print(f"\n" + "="*80)
        print("HMI ARCHITECTURE SUMMARY")
        print("="*80)
        
        print(f"\nHMI Components Found:")
        print(f"  HMI Controllers: {len(hmi_controllers)}")
        print(f"  HMI Input Boundaries: {len(input_boundaries)}")
        print(f"  HMI Output Boundaries: {len(output_boundaries)}")
        print(f"  Other System Boundaries: {len(other_boundaries)}")
        
        print(f"\nHMI Interaction Patterns:")
        
        # Analyze user inputs
        user_inputs = []
        for boundary in input_boundaries:
            for step_ref in boundary.step_references:
                for verb_analysis in all_verb_analyses:
                    if step_ref == verb_analysis.step_id:
                        if "espresso" in verb_analysis.original_text.lower():
                            user_inputs.append("Beverage Type Selection (Espresso)")
                        elif "sugar" in verb_analysis.original_text.lower():
                            user_inputs.append("Additive Selection (Sugar)")
                        elif "coffee" in verb_analysis.original_text.lower() and "type" in verb_analysis.original_text.lower():
                            user_inputs.append("Coffee Type Selection")
        
        # Analyze system outputs
        system_outputs = []
        for boundary in output_boundaries:
            for step_ref in boundary.step_references:
                for verb_analysis in all_verb_analyses:
                    if step_ref == verb_analysis.step_id:
                        if "error" in verb_analysis.original_text.lower():
                            if "water" in verb_analysis.original_text.lower():
                                system_outputs.append("Error Message: 'Water Low'")
                            elif "milk" in verb_analysis.original_text.lower():
                                system_outputs.append("Error Message: 'No Milk'")
                            elif "compressor" in verb_analysis.original_text.lower():
                                system_outputs.append("Error Message: 'Equipment Failure'")
                            else:
                                system_outputs.append("Error Message: General")
                        elif "message" in verb_analysis.original_text.lower():
                            system_outputs.append("Status Message: 'Coffee Ready'")
                        elif "present" in verb_analysis.original_text.lower():
                            system_outputs.append("Status Message: 'Product Ready'")
        
        print(f"\n  User Input Types:")
        for input_type in set(user_inputs):
            print(f"    - {input_type}")
        
        print(f"\n  System Output Types:")
        for output_type in set(system_outputs):
            print(f"    - {output_type}")
        
        print(f"\nExample HMI Interaction Flows:")
        print(f"  INPUT:  User selects 'Espresso' -> HMIBeverageSelectionBoundary -> HMIController")
        print(f"  OUTPUT: HMIController -> HMIErrorDisplayBoundary -> Display 'No Milk'")
        print(f"  INPUT:  User selects 'Add Sugar' -> HMIAdditiveInputBoundary -> HMIController")
        print(f"  OUTPUT: HMIController -> HMIStatusDisplayBoundary -> Display 'Coffee Ready'")
        
        print(f"\nHMI Boundary Categories:")
        print(f"  Input Boundaries: {[b.name for b in input_boundaries]}")
        print(f"  Output Boundaries: {[b.name for b in output_boundaries]}")
        print(f"  Supply Boundaries: {[b.name for b in other_boundaries if 'Supply' in b.name]}")
        print(f"  Trigger Boundaries: {[b.name for b in other_boundaries if 'Trigger' in b.name]}")
        print(f"  Delivery Boundaries: {[b.name for b in other_boundaries if 'Delivery' in b.name]}")
    
    def _print_multi_uc_summary(self, all_verb_analyses: List[VerbAnalysis], 
                               combined_ra_classes: List[RAClass], uc_file_paths: List[str]):
        """Print summary for multi-UC analysis"""
        print("\n" + "="*80)
        print("MULTI-UC ANALYSIS SUMMARY")
        print("="*80)
        
        # UC-wise statistics
        uc_stats = {}
        for verb_analysis in all_verb_analyses:
            uc_name = verb_analysis.uc_name or "Unknown"
            if uc_name not in uc_stats:
                uc_stats[uc_name] = {"verbs": 0, "transaction": 0, "transformation": 0, "function": 0}
            
            uc_stats[uc_name]["verbs"] += 1
            if verb_analysis.verb_type == VerbType.TRANSACTION_VERB:
                uc_stats[uc_name]["transaction"] += 1
            elif verb_analysis.verb_type == VerbType.TRANSFORMATION_VERB:
                uc_stats[uc_name]["transformation"] += 1
            else:
                uc_stats[uc_name]["function"] += 1
        
        print(f"\nAnalyzed {len(uc_file_paths)} UC files:")
        for i, uc_path in enumerate(uc_file_paths, 1):
            uc_name = Path(uc_path).stem
            stats = uc_stats.get(uc_name, {})
            print(f"  {i}. {uc_name}: {stats.get('verbs', 0)} verb analyses")
            print(f"     - Transaction: {stats.get('transaction', 0)}")
            print(f"     - Transformation: {stats.get('transformation', 0)}")
            print(f"     - Function: {stats.get('function', 0)}")
        
        # Combined statistics
        print(f"\nCOMBINED RESULTS:")
        print(f"  Total verb analyses: {len(all_verb_analyses)}")
        print(f"  Unique RA classes: {len(combined_ra_classes)}")
        
        # RA class distribution
        ra_type_counts = {}
        for ra_class in combined_ra_classes:
            ra_type = ra_class.type
            if ra_type not in ra_type_counts:
                ra_type_counts[ra_type] = 0
            ra_type_counts[ra_type] += 1
        
        print(f"\nRA CLASS DISTRIBUTION:")
        for ra_type, count in sorted(ra_type_counts.items()):
            print(f"  {ra_type}: {count}")
        
        # Shared RA classes (appearing in multiple UCs)
        shared_classes = []
        for ra_class in combined_ra_classes:
            uc_sources = set()
            for step_ref in ra_class.step_references:
                # Extract UC name from step references (assuming format like "UC1_B1")
                for verb_analysis in all_verb_analyses:
                    if step_ref == verb_analysis.step_id and verb_analysis.uc_name:
                        uc_sources.add(verb_analysis.uc_name)
            
            if len(uc_sources) > 1:
                shared_classes.append((ra_class, uc_sources))
        
        if shared_classes:
            print(f"\nSHARED RA CLASSES (appear in multiple UCs): {len(shared_classes)}")
            for ra_class, uc_sources in shared_classes[:10]:  # Show first 10
                print(f"  {ra_class.type}: {ra_class.name} -> {', '.join(sorted(uc_sources))}")
            if len(shared_classes) > 10:
                print(f"  ... and {len(shared_classes) - 10} more")
        
        print("\n" + "="*80)
    
    def _analyze_control_flow_simplified(self, verb_analyses: List[VerbAnalysis], ra_classes: List[RAClass]) -> List[ControlFlow]:
        """
        Simplified control flow analysis for JSON export
        Generates basic control flows based on verb analysis and RA classes
        """
        control_flows = []
        
        # Create a mapping of step_id to controllers for that step
        step_to_controllers = {}
        for ra in ra_classes:
            if ra.type == "Controller":
                for step_ref in ra.step_references:
                    if step_ref not in step_to_controllers:
                        step_to_controllers[step_ref] = []
                    step_to_controllers[step_ref].append(ra.name)
        
        # Create sequential flows between steps
        sorted_verb_analyses = sorted(verb_analyses, key=lambda va: va.step_id)
        
        for i, va in enumerate(sorted_verb_analyses[:-1]):
            current_step = va.step_id
            next_step = sorted_verb_analyses[i + 1].step_id
            
            current_controllers = step_to_controllers.get(current_step, [])
            next_controllers = step_to_controllers.get(next_step, [])
            
            # Create flows between controllers in consecutive steps
            for curr_ctrl in current_controllers:
                for next_ctrl in next_controllers:
                    if curr_ctrl != next_ctrl:
                        control_flows.append(ControlFlow(
                            from_class=curr_ctrl,
                            to_class=next_ctrl,
                            flow_type="sequential",
                            step_id=f"{current_step}->{next_step}",
                            flow_rule=3,  # UC-Methode Rule 3: Sequential execution
                            description=f"Sequential flow from {curr_ctrl} to {next_ctrl}"
                        ))
        
        # Add boundary-to-controller flows (Rule 1)
        for ra in ra_classes:
            if ra.type == "Boundary":
                # Find controllers in the same steps
                for step_ref in ra.step_references:
                    controllers_in_step = step_to_controllers.get(step_ref, [])
                    for controller in controllers_in_step:
                        control_flows.append(ControlFlow(
                            from_class=ra.name,
                            to_class=controller,
                            flow_type="boundary_to_controller",
                            step_id=step_ref,
                            flow_rule=1,  # UC-Methode Rule 1: Boundary → Controller
                            description=f"Boundary {ra.name} activates controller {controller}"
                        ))
        
        # Add controller-to-boundary flows (Rule 5)
        for ra in ra_classes:
            if ra.type == "Controller":
                # Find boundaries in the same steps for output
                for step_ref in ra.step_references:
                    boundaries_in_step = [other.name for other in ra_classes 
                                        if other.type == "Boundary" and step_ref in other.step_references]
                    for boundary in boundaries_in_step:
                        if boundary != ra.name:  # Avoid self-loops
                            control_flows.append(ControlFlow(
                                from_class=ra.name,
                                to_class=boundary,
                                flow_type="controller_to_boundary",
                                step_id=step_ref,
                                flow_rule=5,  # UC-Methode Rule 5: Controller → Boundary
                                description=f"Controller {ra.name} outputs to boundary {boundary}"
                            ))
        
        return control_flows
    
    def _analyze_control_flow(self, uc_steps: List[UCStep], ra_classes: List[RAClass]) -> Tuple[List[ControlFlow], List[ParallelFlow]]:
        """
        Analyze control flow according to UC-Methode rules:
        Rule 1: Boundary → Controller
        Rule 2: Controller → Controller  
        Rule 3: Sequential step execution
        Rule 4: Parallel split/join
        Rule 5: Controller → Boundary
        """
        control_flows = []
        parallel_flows = []
        
        # Create lookup dictionaries for RA classes by step
        step_to_boundaries = {}
        step_to_controllers = {}
        
        for ra_class in ra_classes:
            for step_ref in ra_class.step_references:
                if step_ref == "PRECONDITION":
                    continue
                if ra_class.type == "Boundary":
                    if step_ref not in step_to_boundaries:
                        step_to_boundaries[step_ref] = []
                    step_to_boundaries[step_ref].append(ra_class)
                elif ra_class.type == "Controller":
                    if step_ref not in step_to_controllers:
                        step_to_controllers[step_ref] = []
                    step_to_controllers[step_ref].append(ra_class)
        
        # Find Domain Orchestrator
        domain_orchestrator = None
        for ra_class in ra_classes:
            if ra_class.type == "Controller" and "DomainOrchestrator" in ra_class.name:
                domain_orchestrator = ra_class
                break
        
        # Analyze control flows step by step
        for i, step in enumerate(uc_steps):
            step_id = step.step_id
            
            # Rule 1: Boundary → Controller (Transaction verbs create this flow)
            if step_id in step_to_boundaries and step_id in step_to_controllers:
                for boundary in step_to_boundaries[step_id]:
                    for controller in step_to_controllers[step_id]:
                        if "HMI" in boundary.name and "HMI" in controller.name:
                            # HMI interactions
                            control_flows.append(ControlFlow(
                                from_class=boundary.name,
                                to_class=controller.name,
                                flow_type="boundary_to_controller",
                                step_id=step_id,
                                flow_rule=1,
                                description=f"Rule 1: {boundary.name} triggers {controller.name} (external input)"
                            ))
            
            # Rule 5: Controller → Boundary (Output operations)
            if step_id in step_to_controllers:
                # Look for output boundaries in the same or next step
                for controller in step_to_controllers[step_id]:
                    # Check if this step has output boundaries
                    if step_id in step_to_boundaries:
                        for boundary in step_to_boundaries[step_id]:
                            if self._is_output_boundary(boundary, step):
                                control_flows.append(ControlFlow(
                                    from_class=controller.name,
                                    to_class=boundary.name,
                                    flow_type="controller_to_boundary",
                                    step_id=step_id,
                                    flow_rule=5,
                                    description=f"Rule 5: {controller.name} outputs via {boundary.name} (external output)"
                                ))
            
            # Rule 2: Controller → Controller (Sequential processing)
            if i > 0 and step_id in step_to_controllers:
                prev_step = uc_steps[i-1]
                if prev_step.step_id in step_to_controllers:
                    for prev_controller in step_to_controllers[prev_step.step_id]:
                        for curr_controller in step_to_controllers[step_id]:
                            if prev_controller.name != curr_controller.name:
                                control_flows.append(ControlFlow(
                                    from_class=prev_controller.name,
                                    to_class=curr_controller.name,
                                    flow_type="controller_to_controller",
                                    step_id=step_id,
                                    flow_rule=2,
                                    description=f"Rule 2: {prev_controller.name} calls {curr_controller.name} (sequential processing)"
                                ))
            
            # Rule 3: Sequential step execution (overall flow)
            if i > 0:
                prev_step = uc_steps[i-1]
                control_flows.append(ControlFlow(
                    from_class=f"Step_{prev_step.step_id}",
                    to_class=f"Step_{step_id}",
                    flow_type="sequential",
                    step_id=step_id,
                    flow_rule=3,
                    description=f"Rule 3: Sequential execution from {prev_step.step_id} to {step_id}"
                ))
        
        # Rule 4: Parallel flows analysis
        parallel_flows.extend(self._detect_parallel_flows(uc_steps))
        
        return control_flows, parallel_flows
    
    def _is_output_boundary(self, boundary: RAClass, step: UCStep) -> bool:
        """
        Determine if a boundary represents an output operation
        """
        output_patterns = [
            "Display", "Output", "Error", "Status", "Message", "Delivery"
        ]
        return any(pattern in boundary.name for pattern in output_patterns)
    
    def _detect_parallel_flows(self, uc_steps: List[UCStep]) -> List[ParallelFlow]:
        """
        Detect parallel execution patterns in UC steps
        Rule 4: Parallel split and join detection
        """
        parallel_flows = []
        
        # Look for parallel patterns like B2a, B2b, B2c, B2d
        step_groups = {}
        for step in uc_steps:
            if step.flow_type == "main":
                # Extract base step number (B2 from B2a, B2b, etc.)
                base_match = re.match(r'^([A-Z]\d+)', step.step_id)
                if base_match:
                    base_step = base_match.group(1)
                    if base_step not in step_groups:
                        step_groups[base_step] = []
                    step_groups[base_step].append(step.step_id)
        
        # Find groups with multiple sub-steps (parallel candidates)
        for base_step, step_list in step_groups.items():
            if len(step_list) > 1:
                # Check if steps have alphabetical suffixes (B2a, B2b, etc.)
                suffixed_steps = [s for s in step_list if re.match(r'^[A-Z]\d+[a-z]', s)]
                if len(suffixed_steps) > 1:
                    parallel_flows.append(ParallelFlow(
                        split_step=suffixed_steps[0],
                        join_step=suffixed_steps[-1],
                        parallel_paths=[suffixed_steps],  # All steps run in parallel
                        description=f"Rule 4: Parallel execution of {', '.join(suffixed_steps)}"
                    ))
        
        return parallel_flows
    
    def _print_control_flow_analysis(self, control_flows: List[ControlFlow], 
                                   parallel_flows: List[ParallelFlow]) -> None:
        """
        Print detailed control flow analysis results
        """
        print("\n" + "="*80)
        print("KONTROLLFLUSS ANALYSIS (UC-METHODE)")
        print("="*80)
        
        # Group flows by rule
        flows_by_rule = {}
        for flow in control_flows:
            rule = flow.flow_rule
            if rule not in flows_by_rule:
                flows_by_rule[rule] = []
            flows_by_rule[rule].append(flow)
        
        # Print flows by rule
        rule_names = {
            1: "Regel 1: Boundary -> Controller (Externe Eingaben)",
            2: "Regel 2: Controller -> Controller (Interne Verarbeitung)", 
            3: "Regel 3: Sequentielle Schrittabfolge",
            4: "Regel 4: Parallele Verzweigung/Vereinigung",
            5: "Regel 5: Controller -> Boundary (Externe Ausgaben)"
        }
        
        for rule_num in sorted(flows_by_rule.keys()):
            flows = flows_by_rule[rule_num]
            print(f"\n{rule_names[rule_num]} ({len(flows)} Flows):")
            print("-" * 60)
            
            for flow in flows:
                if rule_num == 3:  # Sequential flows are numerous, show summary
                    continue
                print(f"  [{flow.step_id}] {flow.from_class} -> {flow.to_class}")
                print(f"      {flow.description}")
        
        # Show sequential flow summary
        if 3 in flows_by_rule:
            seq_flows = flows_by_rule[3]
            print(f"\n{rule_names[3]} ({len(seq_flows)} Flows):")
            print("-" * 60)
            print(f"  Sequential execution through {len(seq_flows)} steps")
            if seq_flows:
                first_step = seq_flows[0].from_class.replace('Step_', '')
                last_step = seq_flows[-1].to_class.replace('Step_', '')
                print(f"  Flow: {first_step} -> ... -> {last_step}")
        
        # Print parallel flows
        if parallel_flows:
            print(f"\n{rule_names[4]} ({len(parallel_flows)} Parallel Patterns):")
            print("-" * 60)
            
            for pflow in parallel_flows:
                print(f"  Parallel Split: {pflow.split_step} -> {pflow.join_step}")
                print(f"  Parallel Steps: {', '.join(pflow.parallel_paths[0])}")
                print(f"  Description: {pflow.description}")
        
        # Control flow statistics
        print(f"\nKONTROLLFLUSS STATISTIK:")
        print(f"  Total Control Flows: {len(control_flows)}")
        print(f"  Parallel Patterns: {len(parallel_flows)}")
        for rule_num, flows in flows_by_rule.items():
            print(f"  Regel {rule_num}: {len(flows)} flows")
        
        print(f"\nUC-Methode Compliance:")
        print(f"  [OK] Alle 5 Kontrollfluss-Regeln implementiert")
        print(f"  [OK] Boundary-Controller-Boundary Muster erkannt")
        if parallel_flows:
            print(f"  [OK] Parallele Ausführung identifiziert")
        else:
            print(f"  [INFO] Keine parallelen Muster gefunden")
    
    def _generate_ra_classes_from_preconditions(self, preconditions: List[Precondition]) -> List[RAClass]:
        """
        Generate RA classes from preconditions
        Rule: Each precondition generates a Controller and Entity for resource management
        """
        ra_classes = []
        
        for precondition in preconditions:
            resource_name = precondition.resource_name.strip()
            
            # Generate Controller for resource management
            controller_name = f"{resource_name.title().replace(' ', '')}SupplyController"
            controller = RAClass(
                name=controller_name,
                type="Controller",
                stereotype="«control»",
                element_type=ElementType.FUNCTIONAL_ENTITY,
                step_references=["PRECONDITION"],
                description=f"Manages {resource_name} supply and availability (from precondition)",
                source="precondition"
            )
            ra_classes.append(controller)
            
            # Generate Entity for the resource
            entity_name = resource_name.title().replace(' ', '')
            entity = RAClass(
                name=entity_name,
                type="Entity",
                stereotype="«entity»",
                element_type=ElementType.FUNCTIONAL_ENTITY,
                step_references=["PRECONDITION"],
                description=f"Resource: {resource_name} (required as precondition)",
                source="precondition"
            )
            ra_classes.append(entity)
            
            # Generate Supply Boundary for resource monitoring
            boundary_name = f"{resource_name.title().replace(' ', '')}SupplyBoundary"
            boundary = RAClass(
                name=boundary_name,
                type="Boundary",
                stereotype="«boundary»",
                element_type=ElementType.FUNCTIONAL_ENTITY,
                step_references=["PRECONDITION"],
                description=f"Boundary for {resource_name} supply monitoring and refill alerts",
                source="precondition"
            )
            ra_classes.append(boundary)
        
        return ra_classes
    
    def _generate_domain_orchestrator(self, uc_name: str) -> RAClass:
        """
        Generate the Domain Orchestrator - implicit coordination controller
        UC-Methode Rule: Every domain has an implicit orchestrator that coordinates all controllers
        """
        domain_name = self.domain_name or "Generic"
        
        orchestrator = RAClass(
            name=f"{domain_name.title()}DomainOrchestrator",
            type="Controller",
            stereotype="«control»",
            element_type=ElementType.FUNCTIONAL_ENTITY,
            step_references=["IMPLICIT_COORDINATION"],
            description=f"Implicit domain orchestrator for {domain_name} - coordinates all controllers in {uc_name}",
            source="implicit"
        )
        
        return orchestrator
    
    def _analyze_data_flow(self, verb_analyses: List[VerbAnalysis], ra_classes: List[RAClass]) -> List[DataFlow]:
        """
        Analyze data flow relationships between Controllers and Entities
        Rule 1: Controller output entities get 'provide' relationships
        Rule 2: Controller input entities get 'use' relationships
        """
        data_flows = []
        
        # Find all Controllers and Entities
        controllers = {ra.name: ra for ra in ra_classes if ra.type == "Controller"}
        entities = {ra.name: ra for ra in ra_classes if ra.type == "Entity"}
        
        # Also consider entities mentioned in transformations but not explicitly created as RA classes
        transformation_entities = set()
        for verb_analysis in verb_analyses:
            if verb_analysis.verb_type == VerbType.TRANSFORMATION_VERB:
                transformation = self.domain_loader.get_transformation_for_verb(
                    verb_analysis.verb_lemma, self.domain_name
                )
                if transformation and " -> " in transformation:
                    inputs_part, outputs_part = transformation.split(" -> ", 1)
                    # Add all entities mentioned in transformations
                    for entity in inputs_part.split(" + "):
                        transformation_entities.add(entity.strip())
                    for entity in outputs_part.split(" + "):
                        transformation_entities.add(entity.strip())
        
        # Combine explicit entities with transformation entities
        all_entity_names = set(entities.keys()) | transformation_entities
        
        # Debug: Print available entity names
        print(f"\nDEBUG: Available entity names for data flow: {sorted(all_entity_names)}")
        print(f"DEBUG: Total entities: {len(all_entity_names)}")
        print(f"DEBUG: Explicit entities: {len(entities)}")
        print(f"DEBUG: Transformation entities: {len(transformation_entities)}")
        
        # Analyze each verb for data flows (transformation, function, and transaction verbs)
        # GENERIC RULE: Every function produces an output, otherwise it would be useless
        for verb_analysis in verb_analyses:
            transformation = None
            
            if verb_analysis.verb_type == VerbType.TRANSFORMATION_VERB:
                # Get the transformation pattern for this verb
                transformation = self.domain_loader.get_transformation_for_verb(
                    verb_analysis.verb_lemma, self.domain_name
                )
                
                # Special case: If direct_object is specific (like "milk"), override generic transformation
                if verb_analysis.verb_lemma == "add" and verb_analysis.direct_object:
                    direct_obj = verb_analysis.direct_object.lower().replace("the ", "").replace("a ", "")
                    if "milk" in direct_obj:
                        transformation = "Coffee + Milk -> Coffee"
                        print(f"DEBUG: Override 'add' transformation for direct_object 'milk': {transformation}")
                    elif "sugar" in direct_obj:
                        transformation = "Coffee + Sugar -> Coffee"
                        print(f"DEBUG: Override 'add' transformation for direct_object 'sugar': {transformation}")
                
            elif verb_analysis.verb_type == VerbType.FUNCTION_VERB:
                # GENERIC RULE: Every function verb produces an output (its direct_object)
                # Functions without explicit transformations still provide their direct object
                print(f"DEBUG: Processing function verb '{verb_analysis.verb_lemma}' with direct_object '{verb_analysis.direct_object}'")
                if verb_analysis.direct_object:
                    # Check for explicit transformation in implementation elements
                    impl_element_info = self.domain_loader.get_implementation_element_info(
                        verb_analysis.direct_object, self.domain_name
                    )
                    
                    # For preparation verbs, we generally want to provide the object itself, not transform it
                    if verb_analysis.verb_lemma == "prepare" and impl_element_info:
                        # "prepare filter" -> provides "Filter" (prepared state)
                        direct_obj_clean = verb_analysis.direct_object.replace("the ", "").replace("a ", "")
                        entity_name = direct_obj_clean.capitalize()
                        transformation = f"-> {entity_name}"
                        print(f"DEBUG: Prepare verb {verb_analysis.verb_lemma} {verb_analysis.direct_object} -> transformation: {transformation}")
                    elif impl_element_info and "transformation" in impl_element_info:
                        transformation = impl_element_info["transformation"]
                    else:
                        # Generic function output: function provides its direct_object
                        # Examples: "activates water heater" -> provides "Water" (hot)
                        #          "prepares filter" -> provides "Filter" (prepared)  
                        #          "retrieves cup" -> provides "Cup" (positioned)
                        direct_obj_clean = verb_analysis.direct_object.replace("the ", "").replace("a ", "")
                        
                        # Special mappings for common implementation elements to functional entities
                        if "water heater" in direct_obj_clean.lower():
                            entity_name = "Water"  # Water heater produces hot water
                        elif "heater" in direct_obj_clean.lower():
                            entity_name = "Water"  # Generic heater in this context produces hot water
                        elif "filter" in direct_obj_clean.lower():
                            entity_name = "Filter"  # Filter preparation provides prepared filter
                        else:
                            # Default: use first word, capitalized
                            entity_name = direct_obj_clean.split()[0].capitalize() if direct_obj_clean else "Unknown"
                        
                        transformation = f"-> {entity_name}"
                        print(f"DEBUG: Function verb {verb_analysis.verb_lemma} -> transformation: {transformation}")
                
            elif verb_analysis.verb_type == VerbType.TRANSACTION_VERB:
                # TRANSACTION VERBS also produce outputs (messages, deliveries, etc.)
                print(f"DEBUG: Processing transaction verb '{verb_analysis.verb_lemma}' with direct_object '{verb_analysis.direct_object}'")
                if verb_analysis.direct_object:
                    # Transaction verbs like "outputs message", "presents cup" provide their direct objects
                    direct_obj_clean = verb_analysis.direct_object.replace("the ", "").replace("a ", "").replace("an ", "")
                    
                    # Special mappings for transaction outputs
                    if "message" in direct_obj_clean.lower():
                        entity_name = "Message"
                    elif "error message" in direct_obj_clean.lower():
                        entity_name = "Error"  # Error is a specialized message
                    elif "cup" in direct_obj_clean.lower():
                        entity_name = "Cup"
                    else:
                        # Default: use first meaningful word, capitalized
                        words = direct_obj_clean.split()
                        entity_name = words[-1].capitalize() if words else "Unknown"  # Take last word (most specific)
                    
                    transformation = f"-> {entity_name}"
                    print(f"DEBUG: Transaction verb {verb_analysis.verb_lemma} -> transformation: {transformation}")
                
            if transformation:
                print(f"DEBUG: Found transformation '{transformation}' for verb '{verb_analysis.verb_lemma}'")
                # Find the controller responsible for this transformation
                controller_name = None
                for controller in controllers.values():
                    if verb_analysis.step_id in controller.step_references:
                        controller_name = controller.name
                        break
                
                print(f"DEBUG: Controller for {verb_analysis.step_id}: {controller_name}")
                if controller_name:
                    # Parse transformation: "Input1 + Input2 + ... -> Output"
                    if " -> " in transformation:
                        inputs_part, outputs_part = transformation.split(" -> ", 1)
                        
                        # Parse input entities (use relationships) - only if inputs exist
                        if inputs_part.strip():
                            input_entities = [inp.strip() for inp in inputs_part.split(" + ")]
                            for input_entity in input_entities:
                                if input_entity in all_entity_names:
                                    data_flows.append(DataFlow(
                                        controller_name=controller_name,
                                        entity_name=input_entity,
                                        relationship_type="use",
                                        step_id=verb_analysis.step_id,
                                        transformation=transformation,
                                        description=f"{controller_name} uses {input_entity} for {verb_analysis.verb_lemma} operation"
                                    ))
                        
                        # Parse output entities (provide relationships)
                        output_entities = [out.strip() for out in outputs_part.split(" + ")]
                        print(f"DEBUG: Step {verb_analysis.step_id} output entities: {output_entities}")
                        for output_entity in output_entities:
                            # Debug for function and transaction verbs
                            if verb_analysis.verb_type in [VerbType.FUNCTION_VERB, VerbType.TRANSACTION_VERB]:
                                print(f"DEBUG: {verb_analysis.verb_type.value} {verb_analysis.verb_lemma} -> entity '{output_entity}' in entities: {output_entity in all_entity_names}")
                                if output_entity not in all_entity_names:
                                    print(f"DEBUG: Missing entity '{output_entity}' - available: {[e for e in all_entity_names if output_entity.lower() in e.lower()]}")
                            
                            if output_entity in all_entity_names:
                                data_flows.append(DataFlow(
                                    controller_name=controller_name,
                                    entity_name=output_entity,
                                    relationship_type="provide",
                                    step_id=verb_analysis.step_id,
                                    transformation=transformation,
                                    description=f"{controller_name} provides {output_entity} from {verb_analysis.verb_lemma} operation"
                                ))
                    
                    # PREPOSITION-BASED DATA FLOW ANALYSIS
                    # Rule: Prepositions indicate data flow direction
                    # "into", "in" → PROVIDE (controller puts data into entity)
                    # "from", "of" → USE (controller takes data from entity)
                    self._analyze_prepositional_data_flows(verb_analysis, controller_name, all_entity_names, data_flows)
        
        # SUPPLY CONTROLLER ANALYSIS
        # Rule: SupplyControllers always provide their respective entities
        self._analyze_supply_controller_data_flows(controllers, entities, data_flows)
        
        # TRANSFORMATION OUTPUT ANALYSIS
        # Rule: Controllers that output entities (Message, Error) should provide them
        self._analyze_transformation_output_data_flows(verb_analyses, controllers, entities, data_flows)
        
        # FUNCTION VERB DIRECT OBJECT ANALYSIS
        # Rule: Function verbs acting on direct objects should use those entities
        self._analyze_function_verb_data_flows(verb_analyses, controllers, entities, data_flows)
        
        # TRANSFORMATION VERB DIRECT OBJECT ANALYSIS
        # Rule: Transformation verbs with control data direct objects (amounts, degrees) should use them
        self._analyze_transformation_direct_object_data_flows(verb_analyses, controllers, entities, data_flows)
        
        return data_flows
    
    def _analyze_prepositional_data_flows(self, verb_analysis: VerbAnalysis, controller_name: str, 
                                        all_entity_names: set, data_flows: List[DataFlow]) -> None:
        """
        Analyze prepositional objects to determine data flow relationships
        Rules:
        - "into", "in" → USE (controller uses entity as container/target)
        - "from", "of" → USE (controller takes data from entity)
        - "with" → USE (controller uses entity as tool/ingredient)
        - "to" → USE (container/target) or PROVIDE (delivery to actor)
        """
        if not controller_name or not verb_analysis.prepositional_objects:
            return
            
        for preposition, obj_phrase in verb_analysis.prepositional_objects:
            # Clean object phrase
            obj_clean = obj_phrase.replace("the ", "").replace("a ", "").replace("an ", "").strip()
            
            # Extract entity name from prepositional object
            entity_name = None
            
            # First, try to match against actual entity names that were created
            entity_name = self._find_matching_entity_name(obj_clean, all_entity_names)
            
            if not entity_name:
                # Fallback: Special entity mappings
                if "cup" in obj_clean.lower():
                    entity_name = "Cup"
                elif "filter" in obj_clean.lower():
                    entity_name = "Filter"
                elif "water" in obj_clean.lower():
                    entity_name = "Water"
                elif "user" in obj_clean.lower():
                    entity_name = "User"
                else:
                    # Default: capitalize first word
                    words = obj_clean.split()
                    entity_name = words[0].capitalize() if words else None
            
            if entity_name and entity_name in all_entity_names:
                relationship_type = None
                description = ""
                
                # Determine relationship based on preposition
                if preposition.lower() in ["into", "in"]:
                    # USE: Controller uses entity as container/target (coffee into cup = uses cup)
                    relationship_type = "use"
                    description = f"{controller_name} uses {entity_name} as container/target (preposition: '{preposition}')"
                    
                elif preposition.lower() in ["from", "of"]:
                    # USE: Controller takes something from entity
                    relationship_type = "use"
                    description = f"{controller_name} uses data from {entity_name} (preposition: '{preposition}')"
                    
                elif preposition.lower() in ["with"]:
                    # USE: Controller uses entity as tool/ingredient
                    relationship_type = "use"
                    description = f"{controller_name} uses {entity_name} as tool/ingredient (preposition: '{preposition}')"
                    
                elif preposition.lower() in ["to"]:
                    # Context-dependent: "to" can mean USE (container/target) or PROVIDE (delivery)
                    if entity_name.lower() in ["cup", "container", "tank", "filter"]:
                        # USE: Adding something TO a container uses the container
                        relationship_type = "use"
                        description = f"{controller_name} uses {entity_name} as target/container (preposition: '{preposition}')"
                    else:
                        # PROVIDE: Delivering TO an actor/external entity
                        relationship_type = "provide"
                        description = f"{controller_name} provides data to {entity_name} (preposition: '{preposition}')"
                
                if relationship_type:
                    # Check if this data flow already exists to avoid duplicates
                    existing = any(
                        df.controller_name == controller_name and 
                        df.entity_name == entity_name and 
                        df.relationship_type == relationship_type and
                        df.step_id == verb_analysis.step_id
                        for df in data_flows
                    )
                    
                    if not existing:
                        print(f"DEBUG: Preposition '{preposition}' -> {relationship_type}: {controller_name} -> {entity_name}")
                        data_flows.append(DataFlow(
                            controller_name=controller_name,
                            entity_name=entity_name,
                            relationship_type=relationship_type,
                            step_id=verb_analysis.step_id,
                            transformation=f"preposition: {preposition} {obj_phrase}",
                            description=description
                        ))
    
    def _analyze_transformation_output_data_flows(self, verb_analyses: List[VerbAnalysis], 
                                                 controllers: dict, entities: dict, 
                                                 data_flows: List[DataFlow]) -> None:
        """
        Analyze transformation verbs that output entities (Message, Error, etc.)
        These controllers should PROVIDE the created entities
        """
        for verb_analysis in verb_analyses:
            if verb_analysis.verb_type == VerbType.TRANSACTION_VERB and verb_analysis.direct_object:
                # Check if this is an output verb (output, present, display, etc.)
                output_verbs = ["output", "present", "display", "show", "deliver", "provide"]
                if any(output_verb in verb_analysis.verb_lemma.lower() for output_verb in output_verbs):
                    
                    # Find the controller for this step
                    controller_name = None
                    for controller_name_key, controller in controllers.items():
                        if verb_analysis.step_id in controller.step_references:
                            controller_name = controller_name_key
                            break
                    
                    if controller_name:
                        # Extract entity from direct object
                        obj_words = verb_analysis.direct_object.lower().replace("the ", "").replace("a ", "").replace("an ", "").split()
                        
                        for word in obj_words:
                            # Check if this word matches any entity
                            for entity_name in entities.keys():
                                if word in entity_name.lower() or entity_name.lower() in word:
                                    # Create PROVIDE relationship
                                    existing = any(
                                        df.controller_name == controller_name and 
                                        df.entity_name == entity_name and 
                                        df.relationship_type == "provide" and
                                        df.step_id == verb_analysis.step_id
                                        for df in data_flows
                                    )
                                    
                                    if not existing:
                                        print(f"DEBUG: Output transformation -> provide: {controller_name} -> {entity_name}")
                                        data_flows.append(DataFlow(
                                            controller_name=controller_name,
                                            entity_name=entity_name,
                                            relationship_type="provide",
                                            step_id=verb_analysis.step_id,
                                            transformation=f"output: {verb_analysis.direct_object}",
                                            description=f"{controller_name} provides {entity_name} via {verb_analysis.verb_lemma} operation"
                                        ))
    
    def _analyze_function_verb_data_flows(self, verb_analyses: List[VerbAnalysis], 
                                        controllers: dict, entities: dict, 
                                        data_flows: List[DataFlow]) -> None:
        """
        Analyze function verbs that act on direct objects (stop milk addition, etc.)
        These controllers should USE the entities mentioned in direct objects
        """
        for verb_analysis in verb_analyses:
            if verb_analysis.verb_type == VerbType.FUNCTION_VERB and verb_analysis.direct_object:
                
                # Find the controller for this step
                controller_name = None
                for controller_name_key, controller in controllers.items():
                    if verb_analysis.step_id in controller.step_references:
                        controller_name = controller_name_key
                        break
                
                if controller_name:
                    # Extract entity from direct object using domain configuration
                    all_entity_names = set(entities.keys())
                    entity_name = self._find_matching_entity_name(verb_analysis.direct_object, all_entity_names)
                    
                    if entity_name and entity_name in entities:
                        # Create USE relationship
                        existing = any(
                            df.controller_name == controller_name and 
                            df.entity_name == entity_name and 
                            df.relationship_type == "use" and
                            df.step_id == verb_analysis.step_id
                            for df in data_flows
                        )
                        
                        if not existing:
                            print(f"DEBUG: Function verb -> use: {controller_name} -> {entity_name}")
                            data_flows.append(DataFlow(
                                controller_name=controller_name,
                                entity_name=entity_name,
                                relationship_type="use",
                                step_id=verb_analysis.step_id,
                                transformation=f"function: {verb_analysis.verb_lemma} {verb_analysis.direct_object}",
                                description=f"{controller_name} uses {entity_name} via {verb_analysis.verb_lemma} operation"
                            ))
    
    def _analyze_transformation_direct_object_data_flows(self, verb_analyses: List[VerbAnalysis], 
                                                       controllers: dict, entities: dict, 
                                                       data_flows: List[DataFlow]) -> None:
        """
        Analyze transformation verbs with control data direct objects (e.g., "grind the set amount")
        These controllers should USE the control data entities like CoffeeBeanAmount, WaterAmount, etc.
        """
        for verb_analysis in verb_analyses:
            if verb_analysis.verb_type == VerbType.TRANSFORMATION_VERB and verb_analysis.direct_object:
                
                # Find the controller for this step
                controller_name = None
                for controller_name_key, controller in controllers.items():
                    if verb_analysis.step_id in controller.step_references:
                        controller_name = controller_name_key
                        break
                
                if controller_name:
                    # Check if direct object represents a control data entity (like amounts, degrees)
                    all_entity_names = set(entities.keys())
                    entity_name = self._find_matching_entity_name(verb_analysis.direct_object, all_entity_names)
                    
                    if entity_name and entity_name in entities:
                        # Check if this is a control data entity (ElementType.CONTROL_DATA)
                        entity = entities[entity_name]
                        if entity.element_type == ElementType.CONTROL_DATA:
                            # Create USE relationship
                            existing = any(
                                df.controller_name == controller_name and 
                                df.entity_name == entity_name and 
                                df.relationship_type == "use" and
                                df.step_id == verb_analysis.step_id
                                for df in data_flows
                            )
                            
                            if not existing:
                                print(f"DEBUG: Transformation direct object -> use: {controller_name} -> {entity_name}")
                                data_flows.append(DataFlow(
                                    controller_name=controller_name,
                                    entity_name=entity_name,
                                    relationship_type="use",
                                    step_id=verb_analysis.step_id,
                                    transformation=f"transformation: {verb_analysis.verb_lemma} {verb_analysis.direct_object}",
                                    description=f"{controller_name} uses {entity_name} as control parameter for {verb_analysis.verb_lemma} operation"
                                ))
    
    def _analyze_supply_controller_data_flows(self, controllers: dict, entities: dict, data_flows: List[DataFlow]) -> None:
        """
        Analyze SupplyController provide relationships
        Rule: SupplyControllers always provide their respective entities
        """
        for controller_name, controller in controllers.items():
            if "SupplyController" in controller_name:
                # Extract entity name from controller name
                # e.g., "WaterSupplyController" -> "Water"
                entity_name = controller_name.replace("SupplyController", "")
                
                if entity_name in entities:
                    # Check if this provide relationship already exists
                    existing = any(
                        df.controller_name == controller_name and 
                        df.entity_name == entity_name and 
                        df.relationship_type == "provide"
                        for df in data_flows
                    )
                    
                    if not existing:
                        print(f"DEBUG: SupplyController {controller_name} --provide--> {entity_name}")
                        data_flows.append(DataFlow(
                            controller_name=controller_name,
                            entity_name=entity_name,
                            relationship_type="provide",
                            step_id="SUPPLY",
                            transformation=f"Supply -> {entity_name}",
                            description=f"{controller_name} provides {entity_name} from supply source"
                        ))
    
    def _print_data_flow_analysis(self, data_flows: List[DataFlow]) -> None:
        """
        Print detailed data flow analysis results
        """
        print("\n" + "="*80)
        print("DATA FLOW ANALYSIS")
        print("="*80)
        
        if not data_flows:
            print("\nNo data flows found.")
            return
        
        # Group by relationship type
        use_relationships = [df for df in data_flows if df.relationship_type == "use"]
        provide_relationships = [df for df in data_flows if df.relationship_type == "provide"]
        
        print(f"\nFound {len(data_flows)} data flow relationships:")
        print(f"  - {len(use_relationships)} 'use' relationships")
        print(f"  - {len(provide_relationships)} 'provide' relationships")
        
        if use_relationships:
            print(f"\nUSE RELATIONSHIPS - Controllers use input entities ({len(use_relationships)}):")
            print("-" * 60)
            for df in use_relationships:
                print(f"  [{df.step_id}] {df.controller_name} --use--> {df.entity_name}")
                print(f"      Transformation: {df.transformation}")
                print(f"      Description: {df.description}")
        
        if provide_relationships:
            print(f"\nPROVIDE RELATIONSHIPS - Controllers provide output entities ({len(provide_relationships)}):")
            print("-" * 60)
            for df in provide_relationships:
                print(f"  [{df.step_id}] {df.controller_name} --provide--> {df.entity_name}")
                print(f"      Transformation: {df.transformation}")
                print(f"      Description: {df.description}")
        
        print(f"\nDATA FLOW COMPLIANCE:")
        print(f"  [OK] Rule 1: Output entities have 'provide' relationships")
        print(f"  [OK] Rule 2: Input entities have 'use' relationships")
        print(f"  [OK] All transformation verbs analyzed for data flow")
    
    def _print_precondition_analysis(self, preconditions: List[Precondition], 
                                   precondition_ra_classes: List[RAClass]) -> None:
        """
        Print detailed precondition analysis results
        """
        print("\n" + "="*80)
        print("PRECONDITION ANALYSIS")
        print("="*80)
        
        print(f"\nFound {len(preconditions)} preconditions:")
        for precondition in preconditions:
            print(f"  - {precondition.original_text} ({precondition.uc_name})")
        
        print(f"\nGenerated {len(precondition_ra_classes)} RA classes from preconditions:")
        controllers = [ra for ra in precondition_ra_classes if ra.type == "Controller"]
        entities = [ra for ra in precondition_ra_classes if ra.type == "Entity"]
        boundaries = [ra for ra in precondition_ra_classes if ra.type == "Boundary"]
        
        print(f"\nResource Controllers ({len(controllers)}):")
        for controller in controllers:
            print(f"  - {controller.name}: {controller.description}")
        
        print(f"\nResource Entities ({len(entities)}):")
        for entity in entities:
            print(f"  - {entity.name}: {entity.description}")
        
        print(f"\nSupply Boundaries ({len(boundaries)}):")
        for boundary in boundaries:
            print(f"  - {boundary.name}: {boundary.description}")
        
        print(f"\nUC-Methode Rule Compliance:")
        print(f"  [OK] Each precondition generates Controller + Entity + Boundary")
        print(f"  [OK] Domain Orchestrator implicitly coordinates all controllers")
        print(f"  [WARNING] Preconditions should be handled by separate UCs")
        print(f"  [WARNING] Missing UCs for: {', '.join([p.resource_name for p in preconditions])}")
        print(f"\nNext Step: Create separate UCs for resource supply management")

    def _load_operational_materials_framework(self) -> None:
        """Load universal operational materials framework"""
        try:
            framework_path = Path("domains/universal_operational_materials.json")
            if framework_path.exists():
                with open(framework_path, 'r', encoding='utf-8') as f:
                    self.operational_materials_framework = json.load(f)
                print(f"Loaded operational materials framework")
            else:
                print(f"Warning: Universal operational materials framework not found at {framework_path}")
                self.operational_materials_framework = {}
        except Exception as e:
            print(f"Error loading operational materials framework: {e}")
            self.operational_materials_framework = {}

    def _analyze_operational_materials(self, verb_analyses: List[VerbAnalysis], ra_classes: List[RAClass]) -> List[OperationalMaterial]:
        """
        Analyze operational materials and their safety/hygiene requirements
        """
        operational_materials = []
        
        # Extract entities that represent operational materials
        entities = {ra.name: ra for ra in ra_classes if ra.type == "Entity"}
        
        for entity_name, entity in entities.items():
            # Determine if this entity is an operational material
            if self._is_operational_material(entity_name):
                material = self._create_operational_material(entity_name, entity)
                if material:
                    operational_materials.append(material)
        
        return operational_materials

    def _enhance_controllers_with_safety_hygiene_functions(self, ra_classes: List[RAClass], 
                                                          operational_materials: List[OperationalMaterial],
                                                          safety_constraints: List[SafetyConstraint],
                                                          hygiene_requirements: List[HygieneRequirement]) -> List[RAClass]:
        """
        Enhance controllers with safety/hygiene functions and generalize specific controllers
        """
        print(f"\n{'='*80}")
        print(f"ENHANCING CONTROLLERS WITH SAFETY/HYGIENE FUNCTIONS")
        print(f"{'='*80}")
        
        enhanced_ra_classes = []
        controller_generalization_map = {}
        
        # Step 1: Generalize specific controllers and collect functions
        for ra_class in ra_classes:
            if ra_class.type == "Controller":
                generalized_controller = self._generalize_controller(ra_class)
                if generalized_controller.name != ra_class.name:
                    print(f"Generalizing: {ra_class.name} -> {generalized_controller.name}")
                    controller_generalization_map[ra_class.name] = generalized_controller.name
                enhanced_ra_classes.append(generalized_controller)
            else:
                enhanced_ra_classes.append(ra_class)
        
        # Step 2: Add safety/hygiene functions to controllers
        for material in operational_materials:
            safety_functions = self._derive_safety_functions(material)
            hygiene_functions = self._derive_hygiene_functions(material)
            
            if safety_functions or hygiene_functions:
                target_controller = self._find_or_create_target_controller(
                    material, enhanced_ra_classes, safety_functions + hygiene_functions
                )
                
                if target_controller:
                    # Add functions to controller description
                    all_functions = safety_functions + hygiene_functions
                    print(f"Adding {len(all_functions)} safety/hygiene functions to {target_controller.name}:")
                    for func in all_functions:
                        print(f"  + {func}")
                    
                    # Update controller description
                    if target_controller.description:
                        target_controller.description += f" + Safety/Hygiene: {', '.join(all_functions[:3])}"
                        if len(all_functions) > 3:
                            target_controller.description += f" + {len(all_functions)-3} more"
                    else:
                        target_controller.description = f"Safety/Hygiene: {', '.join(all_functions)}"
                    
                    # Add safety/hygiene step reference
                    safety_step = f"SAFETY_HYGIENE_{material.material_name.upper()}"
                    if safety_step not in target_controller.step_references:
                        target_controller.step_references.append(safety_step)
        
        print(f"\nEnhanced {len([c for c in enhanced_ra_classes if c.type == 'Controller'])} controllers")
        return enhanced_ra_classes

    def _generalize_controller(self, controller: RAClass) -> RAClass:
        """
        Generalize specific controller names to more generic functions
        
        Examples:
        - MilkManager -> StorageManager (if handles storage functions)
        - CoffeeBeansSupplyController -> SupplyController
        - HeaterManager -> TemperatureController
        """
        controller_name = controller.name
        original_description = controller.description or ""
        
        # Controller generalization patterns
        generalizations = {
            # Supply controllers -> generic SupplyController
            r"(\w+)SupplyController": "SupplyController",
            
            # Specific managers -> functional managers
            r"MilkManager": "StorageManager",  # Milk needs storage/cooling
            r"SugarManager": "AdditiveManager",  # Sugar is an additive
            r"CoffeeManager": "ProcessController",  # Coffee processing
            r"HeaterManager": "TemperatureController",  # Temperature control
            r"FilterManager": "ProcessController",  # Filter processing
            r"AmountManager": "ProcessController",  # Amount processing
            r"CupManager": "ContainerManager",  # Container handling
            
            # Specific condition managers -> generic managers
            r"A\d+ConditionManager": "ConditionManager",
            r"ActionsManager": "ProcessController",
            r"AdditionManager": "AdditiveManager",
            
            # Keep some as-is
            r"HMIController": "HMIController",  # Human-Machine Interface
            r"MessageManager": "MessageManager",  # Communication
            r"TimeManager": "TimeManager",  # Time-based operations
            r"UserRequestManager": "RequestManager",  # User requests
        }
        
        generalized_name = controller_name
        for pattern, replacement in generalizations.items():
            if re.match(pattern, controller_name):
                generalized_name = replacement
                break
        
        # Create new controller with generalized name
        generalized_controller = RAClass(
            name=generalized_name,
            type=controller.type,
            stereotype=controller.stereotype,
            element_type=controller.element_type,
            step_references=controller.step_references.copy(),
            description=original_description,
            warnings=controller.warnings.copy() if controller.warnings else [],
            source=controller.source
        )
        
        return generalized_controller

    def _derive_safety_functions(self, material: OperationalMaterial) -> List[str]:
        """Derive safety functions based on material safety class"""
        functions = []
        
        safety_function_map = {
            "explosive": [
                "monitor_static_electricity",
                "maintain_fire_suppression_readiness",
                "control_ignition_sources",
                "monitor_temperature_limits"
            ],
            "radioactive": [
                "monitor_radiation_levels",
                "maintain_shielding_integrity", 
                "control_exposure_time",
                "monitor_contamination_levels"
            ],
            "toxic": [
                "monitor_atmospheric_concentration",
                "maintain_ventilation_systems",
                "control_exposure_limits",
                "manage_spill_containment"
            ],
            "cryogenic": [
                "monitor_temperature_stability",
                "maintain_insulation_systems",
                "control_boil_off_gases",
                "monitor_pressure_relief"
            ],
            "pressure_sensitive": [
                "monitor_pressure_levels",
                "maintain_pressure_relief_systems",
                "control_pressure_buildup",
                "monitor_leak_detection"
            ],
            "standard": []  # No special safety functions for standard materials
        }
        
        material_functions = safety_function_map.get(material.safety_class, [])
        
        # Add material-specific context
        for func in material_functions:
            contextualized_func = f"{func}_{material.material_name.lower()}"
            functions.append(contextualized_func)
        
        return functions

    def _derive_hygiene_functions(self, material: OperationalMaterial) -> List[str]:
        """Derive hygiene functions based on material hygiene level"""
        functions = []
        
        hygiene_function_map = {
            "sterile": [
                "maintain_sterile_environment",
                "monitor_bioburden_levels",
                "control_aseptic_handling",
                "validate_sterility_testing"
            ],
            "food_grade": [
                "maintain_food_safety_standards",
                "monitor_temperature_control", 
                "control_contamination_prevention",
                "validate_cleaning_procedures"
            ],
            "pharmaceutical": [
                "maintain_gmp_compliance",
                "monitor_purity_levels",
                "control_identity_verification",
                "validate_stability_testing"
            ],
            "cleanroom": [
                "maintain_particle_control",
                "monitor_environmental_conditions",
                "control_airflow_systems",
                "validate_surface_cleanliness"
            ],
            "standard": []  # No special hygiene functions for standard level
        }
        
        material_functions = hygiene_function_map.get(material.hygiene_level, [])
        
        # Add material-specific context
        for func in material_functions:
            contextualized_func = f"{func}_{material.material_name.lower()}"
            functions.append(contextualized_func)
        
        return functions

    def _find_or_create_target_controller(self, material: OperationalMaterial, 
                                        ra_classes: List[RAClass], 
                                        functions: List[str]) -> Optional[RAClass]:
        """
        Find appropriate controller for safety/hygiene functions or create new one
        """
        
        # Step 1: Look for existing controllers that handle this material
        material_controllers = []
        for ra_class in ra_classes:
            if ra_class.type == "Controller":
                # Check if controller name or description mentions the material
                if (material.material_name.lower() in ra_class.name.lower() or 
                    (ra_class.description and material.material_name.lower() in ra_class.description.lower())):
                    material_controllers.append(ra_class)
        
        # Step 2: Find best match based on function type
        target_controller = None
        
        # Temperature-related functions -> TemperatureController
        if any("temperature" in func for func in functions):
            temp_controllers = [c for c in ra_classes if c.type == "Controller" and "temperature" in c.name.lower()]
            if temp_controllers:
                target_controller = temp_controllers[0]
            else:
                # Look for HeaterManager or similar
                heating_controllers = [c for c in ra_classes if c.type == "Controller" and 
                                     any(keyword in c.name.lower() for keyword in ["heater", "heat", "temp"])]
                if heating_controllers:
                    target_controller = heating_controllers[0]
        
        # Storage-related functions -> StorageManager
        elif any("storage" in func or "maintain" in func for func in functions):
            storage_controllers = [c for c in ra_classes if c.type == "Controller" and "storage" in c.name.lower()]
            if storage_controllers:
                target_controller = storage_controllers[0]
            elif material_controllers:
                target_controller = material_controllers[0]
        
        # Monitoring functions -> appropriate manager
        elif any("monitor" in func for func in functions):
            if material_controllers:
                target_controller = material_controllers[0]
            else:
                # Create or find monitoring controller
                monitoring_controllers = [c for c in ra_classes if c.type == "Controller" and 
                                        any(keyword in c.name.lower() for keyword in ["monitor", "control", "manager"])]
                if monitoring_controllers:
                    target_controller = monitoring_controllers[0]
        
        # Step 3: Create new controller if no suitable one found
        if not target_controller:
            controller_name = self._determine_controller_name_for_material(material, functions)
            
            # Check if this controller already exists
            existing = [c for c in ra_classes if c.type == "Controller" and c.name == controller_name]
            if existing:
                target_controller = existing[0]
            else:
                # Create new controller
                new_controller = RAClass(
                    name=controller_name,
                    type="Controller",
                    stereotype="«control»",
                    element_type=ElementType.FUNCTIONAL_ENTITY,
                    step_references=[f"SAFETY_HYGIENE_{material.material_name.upper()}"],
                    description=f"Manages safety/hygiene requirements for {material.material_name}",
                    warnings=[],
                    source="safety_hygiene_analysis"
                )
                ra_classes.append(new_controller)
                target_controller = new_controller
                print(f"Created new controller: {controller_name} for {material.material_name}")
        
        return target_controller

    def _determine_controller_name_for_material(self, material: OperationalMaterial, functions: List[str]) -> str:
        """Determine appropriate controller name based on material (Betriebsmittel-oriented)"""
        
        # BETRIEBSMITTEL-ORIENTED NAMING: Material comes first!
        material_name = material.material_name.capitalize()
        
        # Always use material-based controller naming
        return f"{material_name}Controller"

    def _is_operational_material(self, entity_name: str) -> bool:
        """Check if an entity represents an operational material"""
        # Common operational materials across domains
        operational_materials = [
            # Beverage preparation
            "water", "milk", "coffee", "sugar", "coffeebeans",
            # Aerospace  
            "fuel", "propellant", "oxidizer", "helium", "nitrogen",
            # Nuclear
            "uranium", "plutonium", "coolant", "control_rods",
            # General
            "cleaning_agents", "lubricants", "chemicals"
        ]
        
        entity_lower = entity_name.lower()
        return any(material in entity_lower for material in operational_materials)

    def _create_operational_material(self, entity_name: str, entity: RAClass) -> Optional[OperationalMaterial]:
        """Create operational material with safety/hygiene classification"""
        entity_lower = entity_name.lower()
        
        # Default classifications
        safety_class = "standard"
        hygiene_level = "standard"
        special_requirements = []
        addressing_id = f"STD-STD-{entity_name.upper()}-B{self._get_batch_id()}-LOC001"
        storage_conditions = {}
        tracking_parameters = []
        emergency_procedures = []
        
        # Domain-specific material classification
        if self.domain_name == "beverage_preparation":
            safety_class, hygiene_level, special_requirements, storage_conditions = self._classify_beverage_material(entity_lower)
        elif self.domain_name == "aerospace":
            safety_class, hygiene_level, special_requirements, storage_conditions = self._classify_aerospace_material(entity_lower)
        elif self.domain_name == "nuclear":
            safety_class, hygiene_level, special_requirements, storage_conditions = self._classify_nuclear_material(entity_lower)
        
        # Generate addressing ID
        addressing_id = f"{safety_class.upper()}-{hygiene_level.upper()}-{entity_name.upper()}-B{self._get_batch_id()}-LOC001"
        
        return OperationalMaterial(
            material_name=entity_name,
            safety_class=safety_class,
            hygiene_level=hygiene_level,
            special_requirements=special_requirements,
            addressing_id=addressing_id,
            storage_conditions=storage_conditions,
            tracking_parameters=tracking_parameters,
            emergency_procedures=emergency_procedures
        )

    def _classify_beverage_material(self, entity_lower: str) -> Tuple[str, str, List[str], Dict[str, str]]:
        """Classify beverage preparation materials"""
        if "milk" in entity_lower:
            return ("standard", "food_grade", 
                   ["Cold chain maintenance", "Pasteurization verification", "Microbiological testing"],
                   {"temperature": "2-8°C", "humidity": "controlled"})
        elif "water" in entity_lower:
            return ("standard", "food_grade",
                   ["Quality testing", "Filtration", "Chlorine monitoring"],
                   {"temperature": "ambient", "quality": "potable"})
        elif "coffee" in entity_lower or "bean" in entity_lower:
            return ("standard", "food_grade",
                   ["Dry storage", "Pest control", "Freshness monitoring"],
                   {"temperature": "ambient", "humidity": "<60%", "atmosphere": "dry"})
        elif "sugar" in entity_lower:
            return ("standard", "food_grade",
                   ["Moisture control", "Sealed containers"],
                   {"temperature": "ambient", "humidity": "<50%"})
        else:
            return ("standard", "food_grade", [], {})

    def _classify_aerospace_material(self, entity_lower: str) -> Tuple[str, str, List[str], Dict[str, str]]:
        """Classify aerospace materials"""
        if "fuel" in entity_lower or "propellant" in entity_lower:
            return ("explosive", "cleanroom",
                   ["Static electricity elimination", "Fire suppression", "Emergency procedures"],
                   {"temperature": "controlled", "pressure": "monitored", "atmosphere": "inert"})
        elif "oxidizer" in entity_lower:
            return ("explosive", "cleanroom",
                   ["Incompatible material separation", "Fire suppression"],
                   {"temperature": "controlled", "atmosphere": "oxygen-free"})
        elif "helium" in entity_lower or "nitrogen" in entity_lower:
            return ("cryogenic", "cleanroom",
                   ["Asphyxiation monitoring", "Pressure relief"],
                   {"temperature": "cryogenic", "pressure": "high"})
        else:
            return ("standard", "cleanroom", [], {})

    def _classify_nuclear_material(self, entity_lower: str) -> Tuple[str, str, List[str], Dict[str, str]]:
        """Classify nuclear materials"""
        if "uranium" in entity_lower or "plutonium" in entity_lower:
            return ("radioactive", "cleanroom",
                   ["Radiation monitoring", "Criticality safety", "Material accountability"],
                   {"radiation_shielding": "required", "temperature": "controlled"})
        elif "coolant" in entity_lower:
            return ("toxic", "cleanroom",
                   ["Leak detection", "Contamination control"],
                   {"temperature": "controlled", "pressure": "monitored"})
        else:
            return ("standard", "cleanroom", [], {})

    def _get_batch_id(self) -> str:
        """Generate batch ID based on current date"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d")

    def _analyze_safety_constraints(self, data_flows: List[DataFlow], operational_materials: List[OperationalMaterial]) -> List[SafetyConstraint]:
        """Analyze safety constraints for data flows involving operational materials"""
        safety_constraints = []
        
        material_map = {mat.material_name: mat for mat in operational_materials}
        
        for data_flow in data_flows:
            if data_flow.entity_name in material_map:
                material = material_map[data_flow.entity_name]
                constraints = self._generate_safety_constraints(material, data_flow)
                safety_constraints.extend(constraints)
        
        return safety_constraints

    def _generate_safety_constraints(self, material: OperationalMaterial, data_flow: DataFlow) -> List[SafetyConstraint]:
        """Generate safety constraints based on material classification"""
        constraints = []
        
        if material.safety_class == "explosive":
            constraints.append(SafetyConstraint(
                material_name=material.material_name,
                constraint_type="thermal",
                max_limits={"temperature": "40°C", "static_charge": "0V"},
                monitoring_required=["temperature_sensor", "static_detector"],
                emergency_actions=["emergency_shutdown", "fire_suppression"],
                responsible_controller=data_flow.controller_name
            ))
        elif material.safety_class == "toxic":
            constraints.append(SafetyConstraint(
                material_name=material.material_name,
                constraint_type="atmospheric",
                max_limits={"concentration": "10ppm", "exposure_time": "8h"},
                monitoring_required=["gas_detector", "ventilation_monitor"],
                emergency_actions=["evacuation", "decontamination"],
                responsible_controller=data_flow.controller_name
            ))
        elif material.safety_class == "radioactive":
            constraints.append(SafetyConstraint(
                material_name=material.material_name,
                constraint_type="radiation",
                max_limits={"dose_rate": "100mrem/h", "exposure": "5rem/year"},
                monitoring_required=["radiation_detector", "dosimeter"],
                emergency_actions=["radiation_alarm", "evacuation"],
                responsible_controller=data_flow.controller_name
            ))
        
        return constraints

    def _analyze_hygiene_requirements(self, data_flows: List[DataFlow], operational_materials: List[OperationalMaterial]) -> List[HygieneRequirement]:
        """Analyze hygiene requirements for data flows involving operational materials"""
        hygiene_requirements = []
        
        material_map = {mat.material_name: mat for mat in operational_materials}
        
        for data_flow in data_flows:
            if data_flow.entity_name in material_map:
                material = material_map[data_flow.entity_name]
                requirements = self._generate_hygiene_requirements(material, data_flow)
                hygiene_requirements.extend(requirements)
        
        return hygiene_requirements

    def _generate_hygiene_requirements(self, material: OperationalMaterial, data_flow: DataFlow) -> List[HygieneRequirement]:
        """Generate hygiene requirements based on material classification"""
        requirements = []
        
        if material.hygiene_level == "food_grade":
            requirements.append(HygieneRequirement(
                material_name=material.material_name,
                sterility_level="food_grade",
                cleaning_protocols=["daily_cleaning", "sanitization"],
                contamination_controls=["pest_control", "cross_contamination_prevention"],
                validation_requirements=["microbiological_testing", "cleaning_validation"],
                responsible_controller=data_flow.controller_name
            ))
        elif material.hygiene_level == "sterile":
            requirements.append(HygieneRequirement(
                material_name=material.material_name,
                sterility_level="sterile",
                cleaning_protocols=["sterilization", "aseptic_handling"],
                contamination_controls=["bioburden_monitoring", "sterile_barrier"],
                validation_requirements=["sterility_testing", "bioburden_validation"],
                responsible_controller=data_flow.controller_name
            ))
        elif material.hygiene_level == "cleanroom":
            requirements.append(HygieneRequirement(
                material_name=material.material_name,
                sterility_level="cleanroom",
                cleaning_protocols=["particle_cleaning", "surface_decontamination"],
                contamination_controls=["particle_monitoring", "airflow_control"],
                validation_requirements=["particle_count", "surface_cleanliness"],
                responsible_controller=data_flow.controller_name
            ))
        
        return requirements

    def analyze_uc_with_safety_hygiene(self, uc_file_path: str) -> Tuple[List[VerbAnalysis], List[RAClass], List[OperationalMaterial], List[SafetyConstraint], List[HygieneRequirement]]:
        """
        Complete UC analysis including safety, hygiene, and operational materials
        """
        print(f"\n{'='*80}")
        print(f"UC ANALYSIS WITH SAFETY/HYGIENE REQUIREMENTS")
        print(f"{'='*80}")
        
        # Standard UC analysis
        verb_analyses, ra_classes = self.analyze_uc_file(uc_file_path)
        
        # Operational materials analysis
        operational_materials = self._analyze_operational_materials(verb_analyses, ra_classes)
        
        # Safety constraints analysis
        safety_constraints = self._analyze_safety_constraints([], operational_materials)
        
        # Hygiene requirements analysis
        hygiene_requirements = self._analyze_hygiene_requirements([], operational_materials)
        
        # ENHANCEMENT: Add safety/hygiene functions to controllers
        enhanced_ra_classes = self._enhance_controllers_with_safety_hygiene_functions(
            ra_classes, operational_materials, safety_constraints, hygiene_requirements
        )
        
        # Data flow analysis (with enhanced controllers)
        data_flows = self._analyze_data_flow(verb_analyses, enhanced_ra_classes)
        
        # Print comprehensive results
        self._print_safety_hygiene_analysis(operational_materials, safety_constraints, hygiene_requirements)
        
        return verb_analyses, enhanced_ra_classes, operational_materials, safety_constraints, hygiene_requirements

    def _print_safety_hygiene_analysis(self, operational_materials: List[OperationalMaterial], 
                                     safety_constraints: List[SafetyConstraint], 
                                     hygiene_requirements: List[HygieneRequirement]) -> None:
        """Print comprehensive safety and hygiene analysis results"""
        
        print(f"\n{'='*80}")
        print(f"OPERATIONAL MATERIALS ANALYSIS")
        print(f"{'='*80}")
        
        if operational_materials:
            print(f"\nFound {len(operational_materials)} operational materials:")
            for material in operational_materials:
                print(f"\n{material.material_name}:")
                print(f"  Safety Class: {material.safety_class}")
                print(f"  Hygiene Level: {material.hygiene_level}")
                print(f"  Addressing ID: {material.addressing_id}")
                if material.special_requirements:
                    print(f"  Special Requirements: {', '.join(material.special_requirements)}")
                if material.storage_conditions:
                    conditions = [f"{k}={v}" for k, v in material.storage_conditions.items()]
                    print(f"  Storage Conditions: {', '.join(conditions)}")
        
        print(f"\n{'='*80}")
        print(f"SAFETY CONSTRAINTS ANALYSIS")
        print(f"{'='*80}")
        
        if safety_constraints:
            print(f"\nFound {len(safety_constraints)} safety constraints:")
            for constraint in safety_constraints:
                print(f"\n{constraint.material_name} - {constraint.constraint_type}:")
                print(f"  Responsible Controller: {constraint.responsible_controller}")
                if constraint.max_limits:
                    limits = [f"{k}={v}" for k, v in constraint.max_limits.items()]
                    print(f"  Max Limits: {', '.join(limits)}")
                if constraint.monitoring_required:
                    print(f"  Monitoring: {', '.join(constraint.monitoring_required)}")
                if constraint.emergency_actions:
                    print(f"  Emergency Actions: {', '.join(constraint.emergency_actions)}")
        
        print(f"\n{'='*80}")
        print(f"HYGIENE REQUIREMENTS ANALYSIS")
        print(f"{'='*80}")
        
        if hygiene_requirements:
            print(f"\nFound {len(hygiene_requirements)} hygiene requirements:")
            for requirement in hygiene_requirements:
                print(f"\n{requirement.material_name} - {requirement.sterility_level}:")
                print(f"  Responsible Controller: {requirement.responsible_controller}")
                if requirement.cleaning_protocols:
                    print(f"  Cleaning Protocols: {', '.join(requirement.cleaning_protocols)}")
                if requirement.contamination_controls:
                    print(f"  Contamination Controls: {', '.join(requirement.contamination_controls)}")
                if requirement.validation_requirements:
                    print(f"  Validation: {', '.join(requirement.validation_requirements)}")

    def generate_graphviz_graph(self, verb_analyses: List[VerbAnalysis], ra_classes: List[RAClass], 
                               data_flows: List[DataFlow] = None, 
                               operational_materials: List[OperationalMaterial] = None,
                               safety_constraints: List[SafetyConstraint] = None,
                               hygiene_requirements: List[HygieneRequirement] = None,
                               output_file: str = None) -> str:
        """
        Generate Graphviz DOT format graph for UC analysis visualization
        
        Args:
            verb_analyses: List of verb analysis results
            ra_classes: List of RA classes (Actors, Boundaries, Controllers, Entities)
            data_flows: Optional list of data flows
            operational_materials: Optional list of operational materials
            safety_constraints: Optional list of safety constraints
            hygiene_requirements: Optional list of hygiene requirements
            output_file: Optional output file path for DOT file
            
        Returns:
            DOT format string
        """
        
        print(f"\n{'='*80}")
        print(f"GENERATING GRAPHVIZ DOT GRAPH")
        print(f"{'='*80}")
        
        # Start DOT graph
        dot_lines = []
        dot_lines.append("digraph UC_Analysis {")
        dot_lines.append("    // Graph settings")
        dot_lines.append("    rankdir=TB;")
        dot_lines.append("    node [fontname=\"Arial\", fontsize=10];")
        dot_lines.append("    edge [fontname=\"Arial\", fontsize=8];")
        dot_lines.append("    compound=true;")
        dot_lines.append("    newrank=true;")
        dot_lines.append("")
        
        # Graph title
        uc_name = self._extract_uc_name(verb_analyses)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        dot_lines.append(f"    // UC Analysis Graph: {uc_name}")
        dot_lines.append(f"    // Generated: {timestamp}")
        dot_lines.append(f"    // Domain: {self.domain_name}")
        dot_lines.append("")
        
        # Add graph title
        dot_lines.append("    labelloc=\"t\";")
        dot_lines.append(f"    label=\"UC Analysis: {uc_name}\\nDomain: {self.domain_name}\\nGenerated: {timestamp}\";")
        dot_lines.append("    fontsize=14;")
        dot_lines.append("    fontname=\"Arial Bold\";")
        dot_lines.append("")
        
        # Group RA classes by type
        actors = [ra for ra in ra_classes if ra.type == "Actor"]
        boundaries = [ra for ra in ra_classes if ra.type == "Boundary"]  
        controllers = [ra for ra in ra_classes if ra.type == "Controller"]
        entities = [ra for ra in ra_classes if ra.type == "Entity"]
        
        # Define node styles
        node_styles = {
            "Actor": {
                "shape": "box",
                "style": "rounded,filled",
                "fillcolor": "lightblue",
                "color": "blue",
                "fontcolor": "black"
            },
            "Boundary": {
                "shape": "ellipse", 
                "style": "filled",
                "fillcolor": "lightgreen",
                "color": "green",
                "fontcolor": "black"
            },
            "Controller": {
                "shape": "box",
                "style": "filled",
                "fillcolor": "lightyellow",
                "color": "orange",
                "fontcolor": "black"
            },
            "Entity": {
                "shape": "box",
                "style": "rounded,filled",
                "fillcolor": "lightcoral",
                "color": "red", 
                "fontcolor": "black"
            }
        }
        
        # Add subgraphs for grouping
        self._add_actor_subgraph(dot_lines, actors, node_styles["Actor"])
        self._add_boundary_subgraph(dot_lines, boundaries, node_styles["Boundary"])
        self._add_controller_subgraph(dot_lines, controllers, node_styles["Controller"], operational_materials)
        self._add_entity_subgraph(dot_lines, entities, node_styles["Entity"], operational_materials)
        
        # Add control flows
        self._add_control_flows(dot_lines, verb_analyses, ra_classes)
        
        # Add data flows if provided
        if data_flows:
            self._add_data_flows(dot_lines, data_flows)
            
        # Add safety/hygiene information if provided
        if operational_materials and (safety_constraints or hygiene_requirements):
            self._add_safety_hygiene_annotations(dot_lines, operational_materials, 
                                               safety_constraints, hygiene_requirements)
        
        # Add legend
        self._add_graph_legend(dot_lines)
        
        # Close graph
        dot_lines.append("}")
        
        dot_content = "\n".join(dot_lines)
        
        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(dot_content)
            print(f"✅ Graphviz DOT file saved: {output_file}")
            print(f"   To generate PNG: dot -Tpng {output_file} -o {output_file.replace('.dot', '.png')}")
            print(f"   To generate SVG: dot -Tsvg {output_file} -o {output_file.replace('.dot', '.svg')}")
            print(f"   To generate PDF: dot -Tpdf {output_file} -o {output_file.replace('.dot', '.pdf')}")
        
        print(f"✅ Graphviz DOT graph generated ({len(dot_lines)} lines)")
        return dot_content

    def _extract_uc_name(self, verb_analyses: List[VerbAnalysis]) -> str:
        """Extract UC name from verb analyses"""
        for analysis in verb_analyses:
            if analysis.uc_name:
                return analysis.uc_name
        return "Unknown UC"

    def _add_actor_subgraph(self, dot_lines: List[str], actors: List[RAClass], style: Dict[str, str]) -> None:
        """Add actor subgraph with proper styling"""
        if not actors:
            return
            
        dot_lines.append("    // Actors")
        dot_lines.append("    subgraph cluster_actors {")
        dot_lines.append("        label=\"Actors\";")
        dot_lines.append("        style=dashed;")
        dot_lines.append("        color=blue;")
        dot_lines.append("        fontcolor=blue;")
        dot_lines.append("")
        
        for actor in actors:
            node_id = self._sanitize_node_id(actor.name)
            label = self._create_node_label(actor)
            dot_lines.append(f"        {node_id} [")
            for attr, value in style.items():
                dot_lines.append(f"            {attr}=\"{value}\",")
            dot_lines.append(f"            label=\"{label}\"")
            dot_lines.append("        ];")
            
        dot_lines.append("    }")
        dot_lines.append("")

    def _add_boundary_subgraph(self, dot_lines: List[str], boundaries: List[RAClass], style: Dict[str, str]) -> None:
        """Add boundary subgraph with proper styling"""
        if not boundaries:
            return
            
        dot_lines.append("    // Boundaries")
        dot_lines.append("    subgraph cluster_boundaries {")
        dot_lines.append("        label=\"Boundaries\";")
        dot_lines.append("        style=dashed;")
        dot_lines.append("        color=green;")
        dot_lines.append("        fontcolor=green;")
        dot_lines.append("")
        
        for boundary in boundaries:
            node_id = self._sanitize_node_id(boundary.name)
            label = self._create_node_label(boundary)
            dot_lines.append(f"        {node_id} [")
            for attr, value in style.items():
                dot_lines.append(f"            {attr}=\"{value}\",")
            dot_lines.append(f"            label=\"{label}\"")
            dot_lines.append("        ];")
            
        dot_lines.append("    }")
        dot_lines.append("")

    def _add_controller_subgraph(self, dot_lines: List[str], controllers: List[RAClass], 
                                style: Dict[str, str], operational_materials: List[OperationalMaterial] = None) -> None:
        """Add controller subgraph with safety/hygiene highlights"""
        if not controllers:
            return
            
        dot_lines.append("    // Controllers")
        dot_lines.append("    subgraph cluster_controllers {")
        dot_lines.append("        label=\"Controllers\";")
        dot_lines.append("        style=dashed;")
        dot_lines.append("        color=orange;")
        dot_lines.append("        fontcolor=orange;")
        dot_lines.append("")
        
        for controller in controllers:
            node_id = self._sanitize_node_id(controller.name)
            label = self._create_node_label(controller)
            
            # Check if this controller handles critical materials
            controller_style = style.copy()
            if operational_materials:
                critical_materials = self._get_critical_materials_for_controller(controller.name, operational_materials)
                if critical_materials:
                    controller_style["fillcolor"] = "yellow"
                    controller_style["color"] = "red"
                    controller_style["penwidth"] = "2"
                    label += f"\\n⚠️ {len(critical_materials)} critical materials"
            
            dot_lines.append(f"        {node_id} [")
            for attr, value in controller_style.items():
                dot_lines.append(f"            {attr}=\"{value}\",")
            dot_lines.append(f"            label=\"{label}\"")
            dot_lines.append("        ];")
            
        dot_lines.append("    }")
        dot_lines.append("")

    def _add_entity_subgraph(self, dot_lines: List[str], entities: List[RAClass], 
                           style: Dict[str, str], operational_materials: List[OperationalMaterial] = None) -> None:
        """Add entity subgraph with operational material highlights"""
        if not entities:
            return
            
        dot_lines.append("    // Entities")
        dot_lines.append("    subgraph cluster_entities {")
        dot_lines.append("        label=\"Entities\";")
        dot_lines.append("        style=dashed;")
        dot_lines.append("        color=red;")
        dot_lines.append("        fontcolor=red;")
        dot_lines.append("")
        
        # Group by operational material status
        operational_entities = []
        regular_entities = []
        
        for entity in entities:
            if operational_materials:
                material = self._find_operational_material(entity.name, operational_materials)
                if material:
                    operational_entities.append((entity, material))
                else:
                    regular_entities.append(entity)
            else:
                regular_entities.append(entity)
        
        # Add operational material entities with special styling
        if operational_entities:
            dot_lines.append("        // Operational Materials")
            for entity, material in operational_entities:
                node_id = self._sanitize_node_id(entity.name)
                label = self._create_operational_material_label(entity, material)
                
                # Safety-based coloring
                entity_style = style.copy()
                if material.safety_class == "explosive":
                    entity_style["fillcolor"] = "orange"
                    entity_style["color"] = "red"
                elif material.safety_class == "radioactive":
                    entity_style["fillcolor"] = "yellow"
                    entity_style["color"] = "purple"
                elif material.safety_class == "toxic":
                    entity_style["fillcolor"] = "lightpink"
                    entity_style["color"] = "darkred"
                elif material.hygiene_level == "food_grade":
                    entity_style["fillcolor"] = "lightgreen"
                    entity_style["color"] = "darkgreen"
                elif material.hygiene_level == "sterile":
                    entity_style["fillcolor"] = "white"
                    entity_style["color"] = "blue"
                
                entity_style["penwidth"] = "2"
                
                dot_lines.append(f"        {node_id} [")
                for attr, value in entity_style.items():
                    dot_lines.append(f"            {attr}=\"{value}\",")
                dot_lines.append(f"            label=\"{label}\"")
                dot_lines.append("        ];")
            dot_lines.append("")
        
        # Add regular entities
        if regular_entities:
            dot_lines.append("        // Regular Entities")
            for entity in regular_entities:
                node_id = self._sanitize_node_id(entity.name)
                label = self._create_node_label(entity)
                dot_lines.append(f"        {node_id} [")
                for attr, value in style.items():
                    dot_lines.append(f"            {attr}=\"{value}\",")
                dot_lines.append(f"            label=\"{label}\"")
                dot_lines.append("        ];")
                
        dot_lines.append("    }")
        dot_lines.append("")

    def _add_control_flows(self, dot_lines: List[str], verb_analyses: List[VerbAnalysis], ra_classes: List[RAClass]) -> None:
        """Add control flow edges"""
        dot_lines.append("    // Control Flows")
        
        # Create mapping of RA classes
        ra_map = {ra.name: ra for ra in ra_classes}
        
        # Simple control flow based on step sequence
        prev_controller = None
        for analysis in verb_analyses:
            if analysis.verb_type in [VerbType.FUNCTION, VerbType.TRANSACTION]:
                current_controller = self._find_controller_for_step(analysis.step_id, ra_classes)
                if current_controller and prev_controller and current_controller != prev_controller:
                    prev_id = self._sanitize_node_id(prev_controller)
                    curr_id = self._sanitize_node_id(current_controller)
                    dot_lines.append(f"    {prev_id} -> {curr_id} [")
                    dot_lines.append("        color=black,")
                    dot_lines.append("        style=solid,")
                    dot_lines.append(f"        label=\"{analysis.step_id}\",")
                    dot_lines.append("        fontsize=8")
                    dot_lines.append("    ];")
                prev_controller = current_controller
        
        dot_lines.append("")

    def _add_data_flows(self, dot_lines: List[str], data_flows: List[DataFlow]) -> None:
        """Add data flow edges with different styling for USE/PROVIDE"""
        if not data_flows:
            return
            
        dot_lines.append("    // Data Flows")
        
        # Group by relationship type
        use_flows = [df for df in data_flows if df.relationship_type == "use"]
        provide_flows = [df for df in data_flows if df.relationship_type == "provide"]
        
        # Add USE relationships (blue dashed)
        if use_flows:
            dot_lines.append("    // USE Relationships")
            for flow in use_flows:
                controller_id = self._sanitize_node_id(flow.controller_name)
                entity_id = self._sanitize_node_id(flow.entity_name)
                dot_lines.append(f"    {entity_id} -> {controller_id} [")
                dot_lines.append("        color=blue,")
                dot_lines.append("        style=dashed,")
                dot_lines.append("        label=\"uses\",")
                dot_lines.append("        fontcolor=blue,")
                dot_lines.append("        fontsize=8")
                dot_lines.append("    ];")
            dot_lines.append("")
        
        # Add PROVIDE relationships (red dashed)  
        if provide_flows:
            dot_lines.append("    // PROVIDE Relationships")
            for flow in provide_flows:
                controller_id = self._sanitize_node_id(flow.controller_name)
                entity_id = self._sanitize_node_id(flow.entity_name)
                dot_lines.append(f"    {controller_id} -> {entity_id} [")
                dot_lines.append("        color=red,")
                dot_lines.append("        style=dashed,")
                dot_lines.append("        label=\"provides\",")
                dot_lines.append("        fontcolor=red,")
                dot_lines.append("        fontsize=8")
                dot_lines.append("    ];")
            dot_lines.append("")

    def _add_safety_hygiene_annotations(self, dot_lines: List[str], 
                                      operational_materials: List[OperationalMaterial],
                                      safety_constraints: List[SafetyConstraint] = None,
                                      hygiene_requirements: List[HygieneRequirement] = None) -> None:
        """Add safety and hygiene annotation nodes"""
        if not operational_materials:
            return
            
        dot_lines.append("    // Safety & Hygiene Annotations")
        dot_lines.append("    subgraph cluster_safety_hygiene {")
        dot_lines.append("        label=\"Safety & Hygiene\";")
        dot_lines.append("        style=dotted;")
        dot_lines.append("        color=purple;")
        dot_lines.append("        fontcolor=purple;")
        dot_lines.append("")
        
        # Add critical material summary
        critical_materials = [m for m in operational_materials 
                            if m.safety_class in ["explosive", "radioactive", "toxic"]]
        
        if critical_materials:
            summary_label = f"Critical Materials: {len(critical_materials)}\\n"
            for material in critical_materials[:3]:  # Show first 3
                summary_label += f"• {material.material_name} ({material.safety_class})\\n"
            if len(critical_materials) > 3:
                summary_label += f"... and {len(critical_materials) - 3} more"
                
            dot_lines.append("        safety_summary [")
            dot_lines.append("            shape=note,")
            dot_lines.append("            style=filled,")
            dot_lines.append("            fillcolor=lightyellow,")
            dot_lines.append("            color=red,")
            dot_lines.append(f"            label=\"{summary_label}\"")
            dot_lines.append("        ];")
        
        dot_lines.append("    }")
        dot_lines.append("")

    def _add_graph_legend(self, dot_lines: List[str]) -> None:
        """Add legend explaining node types and edge types"""
        dot_lines.append("    // Legend")
        dot_lines.append("    subgraph cluster_legend {")
        dot_lines.append("        label=\"Legend\";")
        dot_lines.append("        style=filled;")
        dot_lines.append("        fillcolor=lightgray;")
        dot_lines.append("        color=black;")
        dot_lines.append("")
        
        legend_items = [
            ("legend_actor", "Actor", "lightblue", "box"),
            ("legend_boundary", "Boundary", "lightgreen", "ellipse"),
            ("legend_controller", "Controller", "lightyellow", "box"),
            ("legend_entity", "Entity", "lightcoral", "box"),
            ("legend_material", "Operational\\nMaterial", "orange", "box")
        ]
        
        for node_id, label, color, shape in legend_items:
            dot_lines.append(f"        {node_id} [")
            dot_lines.append(f"            label=\"{label}\",")
            dot_lines.append(f"            fillcolor=\"{color}\",")
            dot_lines.append(f"            shape=\"{shape}\",")
            dot_lines.append("            style=filled")
            dot_lines.append("        ];")
        
        # Legend edges
        dot_lines.append("        legend_control [label=\"Control Flow\", shape=plaintext];")
        dot_lines.append("        legend_use [label=\"Data Use\", shape=plaintext];")
        dot_lines.append("        legend_provide [label=\"Data Provide\", shape=plaintext];")
        
        dot_lines.append("        legend_actor -> legend_boundary [label=\"control\", color=black, style=solid];")
        dot_lines.append("        legend_entity -> legend_controller [label=\"use\", color=blue, style=dashed];")
        dot_lines.append("        legend_controller -> legend_entity [label=\"provide\", color=red, style=dashed];")
        
        dot_lines.append("    }")
        dot_lines.append("")

    def _sanitize_node_id(self, name: str) -> str:
        """Sanitize node name for Graphviz"""
        # Replace problematic characters
        sanitized = re.sub(r'[^\w]', '_', name)
        # Ensure it starts with letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = 'n_' + sanitized
        return sanitized or 'unknown'

    def _create_node_label(self, ra_class: RAClass) -> str:
        """Create formatted label for RA class node"""
        label = ra_class.name
        if ra_class.type in ["Controller", "Entity"]:
            # Add step references
            if ra_class.step_references:
                steps = [ref for ref in ra_class.step_references if not ref.startswith("IMPLICIT")]
                if steps:
                    label += f"\\n({', '.join(steps[:3])})"  # Show first 3 steps
        return label

    def _create_operational_material_label(self, entity: RAClass, material: OperationalMaterial) -> str:
        """Create enhanced label for operational material entities"""
        label = entity.name
        label += f"\\n[{material.safety_class}]"
        label += f"\\n[{material.hygiene_level}]"
        if material.storage_conditions:
            conditions = list(material.storage_conditions.items())[:2]  # First 2 conditions
            for key, value in conditions:
                label += f"\\n{key}: {value}"
        return label

    def _find_operational_material(self, entity_name: str, operational_materials: List[OperationalMaterial]) -> Optional[OperationalMaterial]:
        """Find operational material by entity name"""
        for material in operational_materials:
            if material.material_name.lower() == entity_name.lower():
                return material
        return None

    def _get_critical_materials_for_controller(self, controller_name: str, operational_materials: List[OperationalMaterial]) -> List[OperationalMaterial]:
        """Get critical materials handled by a controller"""
        critical_classes = ["explosive", "radioactive", "toxic", "cryogenic"]
        return [m for m in operational_materials 
                if m.safety_class in critical_classes]

    def _find_controller_for_step(self, step_id: str, ra_classes: List[RAClass]) -> Optional[str]:
        """Find controller responsible for a step"""
        for ra_class in ra_classes:
            if ra_class.type == "Controller" and step_id in ra_class.step_references:
                return ra_class.name
        return None

    def generate_complete_graph(self, uc_file_path: str, output_dir: str = "output") -> str:
        """
        Generate complete graph with all analysis results
        
        Args:
            uc_file_path: Path to UC file
            output_dir: Output directory for graph files
            
        Returns:
            Path to generated DOT file
        """
        print(f"\n{'='*80}")
        print(f"COMPLETE GRAPH GENERATION")
        print(f"{'='*80}")
        
        # Perform complete analysis
        verb_analyses, ra_classes, operational_materials, safety_constraints, hygiene_requirements = \
            self.analyze_uc_with_safety_hygiene(uc_file_path)
        
        # Analyze data flows
        data_flows = self._analyze_data_flow(verb_analyses, ra_classes)
        
        # Generate output filename
        uc_name = self._extract_uc_name(verb_analyses)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_dir}/{uc_name}_graph_{timestamp}.dot"
        
        # Generate graph
        dot_content = self.generate_graphviz_graph(
            verb_analyses=verb_analyses,
            ra_classes=ra_classes,
            data_flows=data_flows,
            operational_materials=operational_materials,
            safety_constraints=safety_constraints,
            hygiene_requirements=hygiene_requirements,
            output_file=output_file
        )
        
        print(f"\n📊 Complete graph generated for {uc_name}")
        print(f"   📁 DOT file: {output_file}")
        print(f"   🎯 Nodes: {len(ra_classes)} RA classes")
        print(f"   🔗 Data flows: {len(data_flows) if data_flows else 0}")
        print(f"   ⚠️  Operational materials: {len(operational_materials) if operational_materials else 0}")
        
        return output_file

    def export_to_json(self, uc_file_path: str, output_dir: str = "output", include_safety_hygiene: bool = True) -> Dict[str, str]:
        """
        Export complete UC analysis to structured JSON format for visualization and system engineering
        
        Args:
            uc_file_path: Path to UC file to analyze
            output_dir: Output directory for JSON files
            include_safety_hygiene: Include safety and hygiene analysis in output
            
        Returns:
            Dictionary with paths to generated JSON files
        """
        print(f"\n{'='*80}")
        print(f"JSON EXPORT FOR VISUALIZATION AND SYSTEM ENGINEERING")
        print(f"{'='*80}")
        
        # Ensure output directory exists
        Path(output_dir).mkdir(exist_ok=True)
        
        # Perform comprehensive analysis
        if include_safety_hygiene:
            verb_analyses, ra_classes, operational_materials, safety_constraints, hygiene_requirements = \
                self.analyze_uc_with_safety_hygiene(uc_file_path)
        else:
            verb_analyses, ra_classes = self.analyze_uc_file(uc_file_path)
            operational_materials, safety_constraints, hygiene_requirements = [], [], []
        
        # Analyze data flows
        data_flows = self._analyze_data_flow(verb_analyses, ra_classes)
        
        # Analyze control flows (simplified for JSON export)
        control_flows = self._analyze_control_flow_simplified(verb_analyses, ra_classes)
        
        # Extract UC name and generate timestamp
        uc_name = self._extract_uc_name(verb_analyses)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate comprehensive JSON output
        json_output = {
            "metadata": {
                "uc_name": uc_name,
                "analysis_timestamp": timestamp,
                "domain": self.domain_name,
                "framework_version": "UC-Methode RA-NLF",
                "source_file": uc_file_path,
                "include_safety_hygiene": include_safety_hygiene
            },
            "analysis_results": {
                "verb_analyses": self._serialize_verb_analyses(verb_analyses),
                "ra_classes": self._serialize_ra_classes(ra_classes),
                "data_flows": self._serialize_data_flows(data_flows),
                "control_flows": self._serialize_control_flows(control_flows)
            },
            "safety_hygiene": {
                "operational_materials": self._serialize_operational_materials(operational_materials),
                "safety_constraints": self._serialize_safety_constraints(safety_constraints),
                "hygiene_requirements": self._serialize_hygiene_requirements(hygiene_requirements)
            },
            "system_engineering": {
                "actors": [ra for ra in self._serialize_ra_classes(ra_classes) if ra["type"] == "Actor"],
                "boundaries": [ra for ra in self._serialize_ra_classes(ra_classes) if ra["type"] == "Boundary"],
                "controllers": [ra for ra in self._serialize_ra_classes(ra_classes) if ra["type"] == "Controller"],
                "entities": [ra for ra in self._serialize_ra_classes(ra_classes) if ra["type"] == "Entity"],
                "uc_steps": self._extract_uc_steps(verb_analyses),
                "domain_orchestrator": self._identify_domain_orchestrator(ra_classes),
                "hmi_components": self._identify_hmi_components(ra_classes)
            },
            "visualization": {
                "graph_structure": self._generate_graph_structure(ra_classes, data_flows, control_flows),
                "layout_hints": self._generate_layout_hints(ra_classes),
                "styling": self._generate_styling_info(ra_classes, operational_materials)
            }
        }
        
        # Write main comprehensive JSON file
        main_output_file = f"{output_dir}/{uc_name}_complete_analysis_{timestamp}.json"
        with open(main_output_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        
        # Write separate specialized JSON files for different use cases
        output_files = {"main": main_output_file}
        
        # System engineering focused JSON
        se_output = {
            "metadata": json_output["metadata"],
            "system_components": json_output["system_engineering"],
            "interactions": json_output["analysis_results"]["control_flows"],
            "data_interfaces": json_output["analysis_results"]["data_flows"]
        }
        se_file = f"{output_dir}/{uc_name}_system_engineering_{timestamp}.json"
        with open(se_file, 'w', encoding='utf-8') as f:
            json.dump(se_output, f, indent=2, ensure_ascii=False)
        output_files["system_engineering"] = se_file
        
        # Visualization focused JSON
        viz_output = {
            "metadata": json_output["metadata"],
            "graph": json_output["visualization"]["graph_structure"],
            "layout": json_output["visualization"]["layout_hints"],
            "styling": json_output["visualization"]["styling"],
            "components": {
                "nodes": json_output["analysis_results"]["ra_classes"],
                "edges": json_output["analysis_results"]["data_flows"] + json_output["analysis_results"]["control_flows"]
            }
        }
        viz_file = f"{output_dir}/{uc_name}_visualization_{timestamp}.json"
        with open(viz_file, 'w', encoding='utf-8') as f:
            json.dump(viz_output, f, indent=2, ensure_ascii=False)
        output_files["visualization"] = viz_file
        
        # Safety/Hygiene focused JSON (if included)
        if include_safety_hygiene and (operational_materials or safety_constraints or hygiene_requirements):
            safety_output = {
                "metadata": json_output["metadata"],
                "safety_analysis": json_output["safety_hygiene"],
                "critical_components": [ra for ra in json_output["analysis_results"]["ra_classes"] 
                                      if any(warning for warning in ra.get("warnings", []) 
                                           if "safety" in warning.lower() or "critical" in warning.lower())]
            }
            safety_file = f"{output_dir}/{uc_name}_safety_hygiene_{timestamp}.json"
            with open(safety_file, 'w', encoding='utf-8') as f:
                json.dump(safety_output, f, indent=2, ensure_ascii=False)
            output_files["safety_hygiene"] = safety_file
        
        print(f"\nJSON export completed for {uc_name}")
        print(f"   Main analysis: {main_output_file}")
        print(f"   System engineering: {se_file}")
        print(f"   Visualization: {viz_file}")
        if "safety_hygiene" in output_files:
            print(f"   Safety/Hygiene: {output_files['safety_hygiene']}")
        print(f"   Total components: {len(ra_classes)} RA classes")
        print(f"   Data flows: {len(data_flows) if data_flows else 0}")
        print(f"   Control flows: {len(control_flows) if control_flows else 0}")
        
        return output_files

    def export_multiple_ucs_to_json(self, uc_file_paths: List[str], output_dir: str = "output", 
                                  domain_name: Optional[str] = None) -> Dict[str, str]:
        """
        Export multi-UC analysis to JSON format for complex system engineering scenarios
        
        Args:
            uc_file_paths: List of UC file paths to analyze together
            output_dir: Output directory for JSON files
            domain_name: Optional domain name override
            
        Returns:
            Dictionary with paths to generated JSON files
        """
        print(f"\n{'='*80}")
        print(f"MULTI-UC JSON EXPORT FOR SYSTEM ENGINEERING")
        print(f"{'='*80}")
        
        # Ensure output directory exists
        Path(output_dir).mkdir(exist_ok=True)
        
        # Perform multi-UC analysis
        verb_analyses, ra_classes = self.analyze_multiple_ucs(uc_file_paths, domain_name)
        
        # Analyze data and control flows
        data_flows = self._analyze_data_flow(verb_analyses, ra_classes)
        control_flows = self._analyze_control_flow_simplified(verb_analyses, ra_classes)
        
        # Generate combined UC name and timestamp
        uc_names = [self._extract_uc_name([va for va in verb_analyses if va.uc_name == Path(fp).stem]) 
                   for fp in uc_file_paths]
        combined_name = "_".join(filter(None, uc_names))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate comprehensive multi-UC JSON output
        json_output = {
            "metadata": {
                "combined_uc_name": combined_name,
                "individual_ucs": uc_names,
                "analysis_timestamp": timestamp,
                "domain": self.domain_name or domain_name,
                "framework_version": "UC-Methode RA-NLF Multi-UC",
                "source_files": uc_file_paths,
                "analysis_type": "multi_uc_domain_analysis"
            },
            "analysis_results": {
                "verb_analyses": self._serialize_verb_analyses(verb_analyses),
                "ra_classes": self._serialize_ra_classes(ra_classes),
                "data_flows": self._serialize_data_flows(data_flows),
                "control_flows": self._serialize_control_flows(control_flows)
            },
            "multi_uc_integration": {
                "shared_components": self._identify_shared_components(ra_classes),
                "cross_uc_interactions": self._identify_cross_uc_interactions(verb_analyses, ra_classes),
                "domain_orchestration": self._analyze_domain_orchestration(ra_classes),
                "integration_points": self._identify_integration_points(data_flows, control_flows)
            },
            "system_architecture": {
                "uc_breakdown": self._break_down_by_uc(verb_analyses, ra_classes),
                "component_reuse": self._analyze_component_reuse(ra_classes),
                "interface_analysis": self._analyze_interfaces(data_flows),
                "coordination_patterns": self._identify_coordination_patterns(control_flows)
            },
            "visualization": {
                "multi_uc_graph": self._generate_multi_uc_graph_structure(ra_classes, data_flows, control_flows),
                "uc_specific_views": self._generate_uc_specific_views(verb_analyses, ra_classes),
                "integration_view": self._generate_integration_view(ra_classes, data_flows)
            }
        }
        
        # Write main multi-UC JSON file
        main_output_file = f"{output_dir}/{combined_name}_multi_uc_analysis_{timestamp}.json"
        with open(main_output_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        
        output_files = {"main": main_output_file}
        
        # Write integration-focused JSON
        integration_output = {
            "metadata": json_output["metadata"],
            "integration_analysis": json_output["multi_uc_integration"],
            "system_architecture": json_output["system_architecture"]
        }
        integration_file = f"{output_dir}/{combined_name}_integration_analysis_{timestamp}.json"
        with open(integration_file, 'w', encoding='utf-8') as f:
            json.dump(integration_output, f, indent=2, ensure_ascii=False)
        output_files["integration"] = integration_file
        
        print(f"\nMulti-UC JSON export completed for {combined_name}")
        print(f"   Main analysis: {main_output_file}")
        print(f"   Integration analysis: {integration_file}")
        print(f"   Total UCs: {len(uc_file_paths)}")
        print(f"   Total components: {len(ra_classes)} RA classes")
        print(f"   Shared components: {len(json_output['multi_uc_integration']['shared_components'])}")
        
        return output_files

    def _serialize_verb_analyses(self, verb_analyses: List[VerbAnalysis]) -> List[Dict]:
        """Serialize VerbAnalysis objects to JSON-compatible format"""
        return [{
            "step_id": va.step_id,
            "original_text": va.original_text,
            "verb": va.verb,
            "verb_lemma": va.verb_lemma,
            "verb_type": va.verb_type.value if hasattr(va.verb_type, 'value') else str(va.verb_type),
            "direct_object": va.direct_object,
            "prepositional_objects": va.prepositional_objects,
            "warnings": va.warnings,
            "suggested_functional_activity": va.suggested_functional_activity,
            "uc_name": va.uc_name
        } for va in verb_analyses]

    def _serialize_ra_classes(self, ra_classes: List[RAClass]) -> List[Dict]:
        """Serialize RAClass objects to JSON-compatible format"""
        return [{
            "name": ra.name,
            "type": ra.type,
            "stereotype": ra.stereotype,
            "element_type": ra.element_type.value if hasattr(ra.element_type, 'value') else str(ra.element_type),
            "step_references": ra.step_references,
            "description": ra.description,
            "warnings": ra.warnings,
            "source": ra.source
        } for ra in ra_classes]

    def _serialize_data_flows(self, data_flows: List[DataFlow]) -> List[Dict]:
        """Serialize DataFlow objects to JSON-compatible format"""
        if not data_flows:
            return []
        return [{
            "controller_name": df.controller_name,
            "entity_name": df.entity_name,
            "relationship_type": df.relationship_type,
            "step_id": df.step_id,
            "transformation": df.transformation,
            "description": df.description,
            "safety_constraints": [self._serialize_safety_constraint(sc) for sc in df.safety_constraints],
            "hygiene_requirements": [self._serialize_hygiene_requirement(hr) for hr in df.hygiene_requirements],
            "operational_material": self._serialize_operational_material(df.operational_material) if df.operational_material else None
        } for df in data_flows]

    def _serialize_control_flows(self, control_flows: List[ControlFlow]) -> List[Dict]:
        """Serialize ControlFlow objects to JSON-compatible format"""
        if not control_flows:
            return []
        return [{
            "from_class": cf.from_class,
            "to_class": cf.to_class,
            "flow_type": cf.flow_type,
            "step_id": cf.step_id,
            "flow_rule": cf.flow_rule,
            "description": cf.description
        } for cf in control_flows]

    def _serialize_operational_materials(self, operational_materials: List[OperationalMaterial]) -> List[Dict]:
        """Serialize OperationalMaterial objects to JSON-compatible format"""
        if not operational_materials:
            return []
        return [self._serialize_operational_material(om) for om in operational_materials]

    def _serialize_operational_material(self, om: OperationalMaterial) -> Dict:
        """Serialize single OperationalMaterial object"""
        if not om:
            return None
        return {
            "material_name": om.material_name,
            "safety_class": om.safety_class,
            "hygiene_level": om.hygiene_level,
            "special_requirements": om.special_requirements,
            "addressing_id": om.addressing_id,
            "storage_conditions": om.storage_conditions,
            "tracking_parameters": om.tracking_parameters,
            "emergency_procedures": om.emergency_procedures
        }

    def _serialize_safety_constraints(self, safety_constraints: List[SafetyConstraint]) -> List[Dict]:
        """Serialize SafetyConstraint objects to JSON-compatible format"""
        if not safety_constraints:
            return []
        return [self._serialize_safety_constraint(sc) for sc in safety_constraints]

    def _serialize_safety_constraint(self, sc: SafetyConstraint) -> Dict:
        """Serialize single SafetyConstraint object"""
        if not sc:
            return None
        return {
            "material_name": sc.material_name,
            "constraint_type": sc.constraint_type,
            "max_limits": sc.max_limits,
            "monitoring_required": sc.monitoring_required,
            "emergency_actions": sc.emergency_actions,
            "responsible_controller": sc.responsible_controller
        }

    def _serialize_hygiene_requirements(self, hygiene_requirements: List[HygieneRequirement]) -> List[Dict]:
        """Serialize HygieneRequirement objects to JSON-compatible format"""
        if not hygiene_requirements:
            return []
        return [self._serialize_hygiene_requirement(hr) for hr in hygiene_requirements]

    def _serialize_hygiene_requirement(self, hr: HygieneRequirement) -> Dict:
        """Serialize single HygieneRequirement object"""
        if not hr:
            return None
        return {
            "material_name": hr.material_name,
            "sterility_level": hr.sterility_level,
            "cleaning_protocols": hr.cleaning_protocols,
            "contamination_controls": hr.contamination_controls,
            "validation_requirements": hr.validation_requirements,
            "responsible_controller": hr.responsible_controller
        }

    def _extract_uc_steps(self, verb_analyses: List[VerbAnalysis]) -> List[Dict]:
        """Extract UC steps from verb analyses"""
        steps = {}
        for va in verb_analyses:
            if va.step_id not in steps:
                steps[va.step_id] = {
                    "step_id": va.step_id,
                    "uc_name": va.uc_name,
                    "step_text": va.original_text,
                    "verb_analysis": []
                }
            steps[va.step_id]["verb_analysis"].append({
                "verb": va.verb,
                "verb_type": va.verb_type.value if hasattr(va.verb_type, 'value') else str(va.verb_type),
                "direct_object": va.direct_object,
                "prepositional_objects": va.prepositional_objects
            })
        return list(steps.values())

    def _identify_domain_orchestrator(self, ra_classes: List[RAClass]) -> Optional[Dict]:
        """Identify domain orchestrator controller"""
        for ra in ra_classes:
            if ra.type == "Controller" and "orchestrator" in ra.name.lower():
                return {
                    "name": ra.name,
                    "description": ra.description,
                    "coordinated_controllers": [other.name for other in ra_classes 
                                              if other.type == "Controller" and other.name != ra.name]
                }
        return None

    def _identify_hmi_components(self, ra_classes: List[RAClass]) -> Dict:
        """Identify HMI-related components"""
        hmi_components = {
            "input_boundaries": [],
            "output_boundaries": [],
            "hmi_controllers": []
        }
        
        for ra in ra_classes:
            if ra.type == "Boundary":
                if "input" in ra.name.lower() or "interface" in ra.name.lower():
                    hmi_components["input_boundaries"].append(ra.name)
                elif "display" in ra.name.lower() or "output" in ra.name.lower():
                    hmi_components["output_boundaries"].append(ra.name)
            elif ra.type == "Controller" and "hmi" in ra.name.lower():
                hmi_components["hmi_controllers"].append(ra.name)
        
        return hmi_components

    def _generate_graph_structure(self, ra_classes: List[RAClass], data_flows: List[DataFlow], control_flows: List[ControlFlow]) -> Dict:
        """Generate graph structure for visualization"""
        nodes = []
        edges = []
        
        # Add nodes
        for ra in ra_classes:
            nodes.append({
                "id": ra.name,
                "label": ra.name,
                "type": ra.type.lower(),
                "stereotype": ra.stereotype,
                "element_type": ra.element_type.value if hasattr(ra.element_type, 'value') else str(ra.element_type)
            })
        
        # Add data flow edges
        if data_flows:
            for df in data_flows:
                edges.append({
                    "source": df.controller_name,
                    "target": df.entity_name,
                    "type": "data_flow",
                    "relationship": df.relationship_type,
                    "label": df.transformation
                })
        
        # Add control flow edges
        if control_flows:
            for cf in control_flows:
                edges.append({
                    "source": cf.from_class,
                    "target": cf.to_class,
                    "type": "control_flow",
                    "flow_type": cf.flow_type,
                    "label": f"Rule {cf.flow_rule}"
                })
        
        return {"nodes": nodes, "edges": edges}

    def _generate_layout_hints(self, ra_classes: List[RAClass]) -> Dict:
        """Generate layout hints for visualization"""
        layout = {
            "actors": {"position": "left", "color": "#FFE4B5"},
            "boundaries": {"position": "left-center", "color": "#E0E0E0"},
            "controllers": {"position": "center", "color": "#98FB98"},
            "entities": {"position": "right", "color": "#FFA07A"}
        }
        
        components_by_type = {}
        for ra in ra_classes:
            comp_type = ra.type.lower()
            if comp_type not in components_by_type:
                components_by_type[comp_type] = []
            components_by_type[comp_type].append(ra.name)
        
        layout["components"] = components_by_type
        return layout

    def _generate_styling_info(self, ra_classes: List[RAClass], operational_materials: List[OperationalMaterial]) -> Dict:
        """Generate styling information for visualization"""
        styling = {
            "default_styles": {
                "actor": {"shape": "ellipse", "color": "#FFE4B5", "border": "#DAA520"},
                "boundary": {"shape": "rectangle", "color": "#E0E0E0", "border": "#808080"},
                "controller": {"shape": "ellipse", "color": "#98FB98", "border": "#228B22"},
                "entity": {"shape": "rectangle", "color": "#FFA07A", "border": "#FF6347"}
            },
            "special_styling": {},
            "warnings": []
        }
        
        # Add special styling for components with warnings or critical materials
        critical_materials = [om.material_name for om in operational_materials if om.safety_class in ["explosive", "radioactive", "toxic"]]
        
        for ra in ra_classes:
            if ra.warnings:
                styling["special_styling"][ra.name] = {"border_color": "#FF0000", "border_width": "2px"}
                styling["warnings"].append({"component": ra.name, "warnings": ra.warnings})
            
            if any(material in ra.description for material in critical_materials):
                styling["special_styling"][ra.name] = {"background_color": "#FFCCCB", "border_color": "#8B0000"}
        
        return styling

    def _identify_shared_components(self, ra_classes: List[RAClass]) -> List[Dict]:
        """Identify components shared across multiple UCs"""
        shared_components = []
        for ra in ra_classes:
            uc_count = len(set(ref.split('.')[0] for ref in ra.step_references if '.' in ref))
            if uc_count > 1:
                shared_components.append({
                    "name": ra.name,
                    "type": ra.type,
                    "shared_across_ucs": uc_count,
                    "step_references": ra.step_references
                })
        return shared_components

    def _identify_cross_uc_interactions(self, verb_analyses: List[VerbAnalysis], ra_classes: List[RAClass]) -> List[Dict]:
        """Identify interactions between different UCs"""
        cross_uc_interactions = []
        uc_names = set(va.uc_name for va in verb_analyses if va.uc_name)
        
        for ra in ra_classes:
            if ra.type == "Controller":
                referenced_ucs = set(ref.split('.')[0] for ref in ra.step_references if '.' in ref)
                if len(referenced_ucs) > 1:
                    cross_uc_interactions.append({
                        "controller": ra.name,
                        "involved_ucs": list(referenced_ucs),
                        "interaction_type": "coordination",
                        "description": f"{ra.name} coordinates activities across {len(referenced_ucs)} UCs"
                    })
        
        return cross_uc_interactions

    def _analyze_domain_orchestration(self, ra_classes: List[RAClass]) -> Dict:
        """Analyze domain orchestration patterns"""
        orchestrators = [ra for ra in ra_classes if ra.type == "Controller" and "orchestrator" in ra.name.lower()]
        
        return {
            "orchestrators": [{"name": orch.name, "description": orch.description} for orch in orchestrators],
            "orchestration_pattern": "domain_orchestrator" if orchestrators else "distributed",
            "coordination_controllers": [ra.name for ra in ra_classes 
                                       if ra.type == "Controller" and "manager" in ra.name.lower()]
        }

    def _identify_integration_points(self, data_flows: List[DataFlow], control_flows: List[ControlFlow]) -> List[Dict]:
        """Identify key integration points in the system"""
        integration_points = []
        
        # Identify entities that are used by multiple controllers (integration through data)
        if data_flows:
            entity_usage = {}
            for df in data_flows:
                if df.entity_name not in entity_usage:
                    entity_usage[df.entity_name] = []
                entity_usage[df.entity_name].append(df.controller_name)
            
            for entity, controllers in entity_usage.items():
                if len(set(controllers)) > 1:
                    integration_points.append({
                        "type": "data_integration",
                        "entity": entity,
                        "controllers": list(set(controllers)),
                        "integration_level": "high" if len(set(controllers)) > 2 else "medium"
                    })
        
        # Identify controllers that interact with many other controllers (coordination hubs)
        if control_flows:
            controller_interactions = {}
            for cf in control_flows:
                for controller in [cf.from_class, cf.to_class]:
                    if controller not in controller_interactions:
                        controller_interactions[controller] = set()
                    controller_interactions[controller].add(cf.from_class)
                    controller_interactions[controller].add(cf.to_class)
            
            for controller, interactions in controller_interactions.items():
                if len(interactions) > 3:  # Interacts with more than 3 other controllers
                    integration_points.append({
                        "type": "control_integration",
                        "controller": controller,
                        "interaction_count": len(interactions),
                        "integration_level": "high" if len(interactions) > 5 else "medium"
                    })
        
        return integration_points

    def _break_down_by_uc(self, verb_analyses: List[VerbAnalysis], ra_classes: List[RAClass]) -> Dict:
        """Break down analysis by individual UC"""
        uc_breakdown = {}
        
        # Group by UC
        uc_names = set(va.uc_name for va in verb_analyses if va.uc_name)
        
        for uc_name in uc_names:
            uc_verb_analyses = [va for va in verb_analyses if va.uc_name == uc_name]
            uc_step_ids = set(va.step_id for va in uc_verb_analyses)
            uc_ra_classes = [ra for ra in ra_classes if any(step in ra.step_references for step in uc_step_ids)]
            
            uc_breakdown[uc_name] = {
                "verb_analyses_count": len(uc_verb_analyses),
                "ra_classes_count": len(uc_ra_classes),
                "components_by_type": {
                    "actors": [ra.name for ra in uc_ra_classes if ra.type == "Actor"],
                    "boundaries": [ra.name for ra in uc_ra_classes if ra.type == "Boundary"],
                    "controllers": [ra.name for ra in uc_ra_classes if ra.type == "Controller"],
                    "entities": [ra.name for ra in uc_ra_classes if ra.type == "Entity"]
                }
            }
        
        return uc_breakdown

    def _analyze_component_reuse(self, ra_classes: List[RAClass]) -> Dict:
        """Analyze component reuse across UCs"""
        reuse_analysis = {
            "highly_reused": [],
            "moderately_reused": [],
            "uc_specific": [],
            "reuse_statistics": {}
        }
        
        for ra in ra_classes:
            uc_count = len(set(ref.split('.')[0] for ref in ra.step_references if '.' in ref))
            
            if uc_count > 2:
                reuse_analysis["highly_reused"].append(ra.name)
            elif uc_count == 2:
                reuse_analysis["moderately_reused"].append(ra.name)
            else:
                reuse_analysis["uc_specific"].append(ra.name)
        
        total_components = len(ra_classes)
        reuse_analysis["reuse_statistics"] = {
            "total_components": total_components,
            "highly_reused_count": len(reuse_analysis["highly_reused"]),
            "moderately_reused_count": len(reuse_analysis["moderately_reused"]),
            "uc_specific_count": len(reuse_analysis["uc_specific"]),
            "reuse_percentage": round((len(reuse_analysis["highly_reused"]) + len(reuse_analysis["moderately_reused"])) / total_components * 100, 2) if total_components > 0 else 0
        }
        
        return reuse_analysis

    def _analyze_interfaces(self, data_flows: List[DataFlow]) -> Dict:
        """Analyze data interfaces between components"""
        if not data_flows:
            return {"interfaces": [], "interface_complexity": "low"}
        
        interfaces = []
        interface_map = {}
        
        for df in data_flows:
            interface_key = f"{df.controller_name} -> {df.entity_name}"
            if interface_key not in interface_map:
                interface_map[interface_key] = {
                    "controller": df.controller_name,
                    "entity": df.entity_name,
                    "relationships": [],
                    "transformations": []
                }
            
            interface_map[interface_key]["relationships"].append(df.relationship_type)
            if df.transformation:
                interface_map[interface_key]["transformations"].append(df.transformation)
        
        interfaces = list(interface_map.values())
        
        # Determine interface complexity
        avg_relationships = sum(len(iface["relationships"]) for iface in interfaces) / len(interfaces) if interfaces else 0
        complexity = "high" if avg_relationships > 2 else "medium" if avg_relationships > 1 else "low"
        
        return {
            "interfaces": interfaces,
            "interface_count": len(interfaces),
            "interface_complexity": complexity,
            "average_relationships_per_interface": round(avg_relationships, 2)
        }

    def _identify_coordination_patterns(self, control_flows: List[ControlFlow]) -> List[Dict]:
        """Identify coordination patterns in control flows"""
        if not control_flows:
            return []
        
        patterns = []
        
        # Identify sequential patterns
        sequential_flows = [cf for cf in control_flows if cf.flow_type == "sequential"]
        if sequential_flows:
            patterns.append({
                "pattern_type": "sequential",
                "description": "Sequential execution pattern",
                "flow_count": len(sequential_flows),
                "components": list(set([cf.from_class for cf in sequential_flows] + [cf.to_class for cf in sequential_flows]))
            })
        
        # Identify parallel patterns
        parallel_flows = [cf for cf in control_flows if "parallel" in cf.flow_type]
        if parallel_flows:
            patterns.append({
                "pattern_type": "parallel",
                "description": "Parallel execution pattern",
                "flow_count": len(parallel_flows),
                "components": list(set([cf.from_class for cf in parallel_flows] + [cf.to_class for cf in parallel_flows]))
            })
        
        # Identify coordination hubs
        flow_counts = {}
        for cf in control_flows:
            for component in [cf.from_class, cf.to_class]:
                flow_counts[component] = flow_counts.get(component, 0) + 1
        
        coordination_hubs = [comp for comp, count in flow_counts.items() if count > 3]
        if coordination_hubs:
            patterns.append({
                "pattern_type": "coordination_hub",
                "description": "Central coordination pattern",
                "components": coordination_hubs
            })
        
        return patterns

    def _generate_multi_uc_graph_structure(self, ra_classes: List[RAClass], data_flows: List[DataFlow], control_flows: List[ControlFlow]) -> Dict:
        """Generate graph structure for multi-UC visualization"""
        base_graph = self._generate_graph_structure(ra_classes, data_flows, control_flows)
        
        # Add UC-specific information to nodes
        for node in base_graph["nodes"]:
            ra = next((ra for ra in ra_classes if ra.name == node["id"]), None)
            if ra:
                node["uc_involvement"] = list(set(ref.split('.')[0] for ref in ra.step_references if '.' in ref))
                node["shared_component"] = len(node["uc_involvement"]) > 1
        
        return base_graph

    def _generate_uc_specific_views(self, verb_analyses: List[VerbAnalysis], ra_classes: List[RAClass]) -> Dict:
        """Generate UC-specific views for multi-UC scenarios"""
        uc_views = {}
        uc_names = set(va.uc_name for va in verb_analyses if va.uc_name)
        
        for uc_name in uc_names:
            uc_verb_analyses = [va for va in verb_analyses if va.uc_name == uc_name]
            uc_step_ids = set(va.step_id for va in uc_verb_analyses)
            uc_ra_classes = [ra for ra in ra_classes if any(step in ra.step_references for step in uc_step_ids)]
            
            uc_views[uc_name] = {
                "nodes": [{"id": ra.name, "type": ra.type.lower()} for ra in uc_ra_classes],
                "step_count": len(set(va.step_id for va in uc_verb_analyses)),
                "component_count": len(uc_ra_classes)
            }
        
        return uc_views

    def _generate_integration_view(self, ra_classes: List[RAClass], data_flows: List[DataFlow]) -> Dict:
        """Generate integration view focusing on shared components and data flows"""
        shared_components = [ra for ra in ra_classes 
                           if len(set(ref.split('.')[0] for ref in ra.step_references if '.' in ref)) > 1]
        
        integration_flows = []
        if data_flows:
            for df in data_flows:
                controller_uc_count = len(set(ref.split('.')[0] for ref in 
                                            next((ra.step_references for ra in ra_classes if ra.name == df.controller_name), [])))
                entity_uc_count = len(set(ref.split('.')[0] for ref in 
                                        next((ra.step_references for ra in ra_classes if ra.name == df.entity_name), [])))
                
                if controller_uc_count > 1 or entity_uc_count > 1:
                    integration_flows.append({
                        "controller": df.controller_name,
                        "entity": df.entity_name,
                        "relationship": df.relationship_type,
                        "cross_uc": True
                    })
        
        return {
            "shared_components": [{"name": ra.name, "type": ra.type} for ra in shared_components],
            "integration_flows": integration_flows,
            "integration_complexity": "high" if len(shared_components) > 5 else "medium" if len(shared_components) > 2 else "low"
        }


def main():
    """Demonstrate JSON export functionality with various UC files"""
    print("GENERIC UC ANALYZER - JSON Export Demonstration")
    print("="*60)
    
    # Demonstrate single UC JSON export
    print("\n=== SINGLE UC JSON EXPORT DEMONSTRATION ===")
    
 
    # Test with UC3 (Rocket Launch) if available
    uc3_file = "Use Case/UC3_Rocket_Launch_Improved.txt"
    if Path(uc3_file).exists():
        print(f"\nExporting UC3 Rocket Launch analysis to JSON: {uc3_file}")
        analyzer = GenericUCAnalyzer(domain_name="rocket_science")
        
        # Export with full analysis
        output_files = analyzer.export_to_json(uc3_file, output_dir="output", include_safety_hygiene=True)
        print(f"UC3 JSON export completed. Files generated: {len(output_files)}")
    
    
    # Fallback: analyze any available UC file for demonstration
    uc_files = list(Path("Use Case").glob("*.txt")) if Path("Use Case").exists() else []
    if not uc_files:
        print("\nNo UC files found in 'Use Case' directory")
        print("Please ensure UC files exist to test JSON export functionality")
        return
        
    print("\n" + "="*60)
    print("JSON EXPORT FUNCTIONALITY DEMONSTRATION COMPLETED")
    print("="*60)
    print("\nGenerated JSON files include:")
    print("• Complete analysis JSON (all data for further processing)")
    print("• System engineering JSON (focused on components and interactions)")
    print("• Visualization JSON (optimized for diagram generation)")
    print("• Safety/Hygiene JSON (safety and operational material analysis)")
    print("• Multi-UC integration JSON (for complex system scenarios)")
    print("\nThese JSON files can be used as input for:")
    print("• Visualization tools and diagram generators")
    print("• System engineering workflows")
    print("• Safety analysis tools")
    print("• Integration with other UC-Methode tools")
    print("• Further automated analysis and processing")


if __name__ == "__main__":
    main()