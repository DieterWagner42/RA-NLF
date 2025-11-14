#!/usr/bin/env python3
"""
Structured UC Analyzer - Line-by-Line Processing
Processes UC files systematically from top to bottom with keyword-based analysis
"""

import sys
import json
import os
import re
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import spacy

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import existing components
from domain_verb_loader import DomainVerbLoader, VerbType
from generative_context_manager import GenerativeContextManager, GeneratedContext, ContextType
from material_controller_registry import MaterialControllerRegistry, extract_function_from_verb
# Pure RUP visualizer is imported dynamically in _generate_rup_diagram()

# Configuration Constants
DEFAULT_DOMAIN = "beverage_preparation"  # Default domain if not specified in config

class LineType(Enum):
    """Types of lines in UC files"""
    CAPABILITY = "capability"
    FEATURE = "feature" 
    USE_CASE = "use_case"
    GOAL = "goal"
    PRECONDITION = "precondition"
    ACTOR = "actor"
    MAIN_FLOW = "main_flow"
    BASIC_FLOW = "basic_flow"
    ALT_FLOW = "alt_flow"
    EXT_FLOW = "ext_flow"
    POSTCONDITION = "postcondition"
    STEP = "step"
    END_UC = "end_uc"
    END_USE_CASE = "end_use_case"
    CONTINUE_FLOW = "continue_flow"
    CONTINUE_AT = "continue_at"
    EMPTY = "empty"
    UNKNOWN = "unknown"

class RAType(Enum):
    """RA Class Types"""
    ACTOR = "Actor"
    BOUNDARY = "Boundary" 
    CONTROLLER = "Controller"
    ENTITY = "Entity"

@dataclass
class UCContext:
    """Context information extracted from UC header"""
    capability: str = ""
    feature: str = ""
    use_case_name: str = ""
    goal: str = ""
    domain: str = ""
    actors: List[str] = None
    preconditions: List[str] = None
    
    def __post_init__(self):
        if self.actors is None:
            self.actors = []
        if self.preconditions is None:
            self.preconditions = []

@dataclass
class GrammaticalAnalysis:
    """Results of grammatical analysis for a line"""
    compound_nouns: List[str] = None
    gerunds: List[str] = None
    main_verb: str = ""
    verb_lemma: str = ""
    verb_type: VerbType = None
    weak_verbs: List[str] = None
    direct_object: str = ""
    prepositional_objects: List[Tuple[str, str]] = None
    
    def __post_init__(self):
        if self.compound_nouns is None:
            self.compound_nouns = []
        if self.gerunds is None:
            self.gerunds = []
        if self.weak_verbs is None:
            self.weak_verbs = []
        if self.prepositional_objects is None:
            self.prepositional_objects = []

@dataclass 
class RAClass:
    """Robustness Analysis Class"""
    name: str
    ra_type: RAType
    stereotype: str
    description: str
    step_id: str = ""
    element_type: str = "functional"
    parallel_group: int = 0  # 0 = sequential, 2 = P2 (B2x), 3 = P3 (B3x), etc.

@dataclass
class ControlFlow:
    """Control flow between controllers"""
    source_step: str
    target_step: str
    source_controller: str
    target_controller: str
    flow_type: str
    rule: str
    description: str

@dataclass
class DataFlow:
    """Data flow between controller and entity"""
    step_id: str
    controller: str
    entity: str
    flow_type: str  # "use" or "provide"
    preposition: str = ""
    description: str = ""

@dataclass
class ParallelFlowNode:
    """Node for parallel control flows - distribution and merge"""
    node_id: str
    node_type: str  # "distribution" or "merge"
    step_range: str  # e.g., "B2a-B2b" for parallel steps
    parallel_steps: List[str] = None  # List of parallel step IDs
    description: str = ""
    
    def __post_init__(self):
        if self.parallel_steps is None:
            self.parallel_steps = []

@dataclass
class StepContext:
    """Context information for a UC step with enhanced operational materials support"""
    step_id: str
    step_type: str = "main"  # "trigger", "main", "alternative", "extension"
    domain: str = ""
    phase: str = ""  # "initialization", "validation", "execution", "completion", "error_handling"
    business_context: str = ""  # What business function this step serves
    technical_context: str = ""  # What technical function this step serves
    context_type: str = ""  # Enhanced: generative context type
    global_context: str = ""  # Enhanced: global context from generative analysis
    description: str = ""  # Enhanced: context-aware description
    preconditions: List[str] = None
    actors_involved: List[str] = None
    operational_materials: List[Dict[str, Any]] = None  # Enhanced: operational materials info
    safety_requirements: List[str] = None  # Enhanced: safety requirements
    special_controllers: List[str] = None  # Enhanced: special controllers needed
    
    def __post_init__(self):
        if self.preconditions is None:
            self.preconditions = []
        if self.actors_involved is None:
            self.actors_involved = []
        if self.operational_materials is None:
            self.operational_materials = []
        if self.safety_requirements is None:
            self.safety_requirements = []
        if self.special_controllers is None:
            self.special_controllers = []

@dataclass
class LineAnalysis:
    """Complete analysis result for a single line"""
    line_number: int
    line_text: str
    line_type: LineType
    step_id: str = ""
    step_context: StepContext = None
    grammatical: GrammaticalAnalysis = None
    ra_classes: List[RAClass] = None
    control_flows: List[ControlFlow] = None
    data_flows: List[DataFlow] = None
    parallel_nodes: List[ParallelFlowNode] = None
    errors: List[str] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.grammatical is None:
            self.grammatical = GrammaticalAnalysis()
        if self.ra_classes is None:
            self.ra_classes = []
        if self.control_flows is None:
            self.control_flows = []
        if self.data_flows is None:
            self.data_flows = []
        if self.parallel_nodes is None:
            self.parallel_nodes = []
        if self.errors is None:
            self.errors = []
        if self.suggestions is None:
            self.suggestions = []

class StructuredUCAnalyzer:
    """
    Line-by-line UC Analyzer with structured processing
    """
    
    def __init__(self, domain_name: str = DEFAULT_DOMAIN):
        self.domain_name = domain_name
        self.verb_loader = DomainVerbLoader()
        self.nlp = None
        self._load_spacy()

        # Initialize generative context manager
        self.context_manager = GenerativeContextManager(domain_name)

        # Initialize material controller registry
        self.controller_registry = MaterialControllerRegistry()

        # Load implicit protection functions
        self.protection_functions = self.verb_loader.get_implicit_protection_functions(domain_name)
        print(f"[PROTECTION] Loaded {len(self.protection_functions)} material protection function sets")

        # Analysis state
        self.uc_context = UCContext()
        self.line_analyses: List[LineAnalysis] = []
        self.step_sequence: List[str] = []  # Track step order for control flows
        self.parallel_flow_nodes: List[ParallelFlowNode] = []  # Track parallel nodes
        self.all_lines: List[str] = []  # Store all file lines for look-ahead
        self.parallel_counter = 1  # Counter for P1, P2, P3, etc.
        self.in_parallel_flow = False  # Track if we're currently in a parallel flow
        
        # Entity deduplication tracking
        self.global_entities: Dict[str, RAClass] = {}  # Track all created entities globally
        
        # Global UC context
        self.uc_goal = ""  # Extracted goal from UC text
        self.uc_title = ""  # UC title/name
        
        # Aggregation state tracking
        self.aggregation_warnings: List[str] = []  # Track ambiguity warnings
        
        # Section context tracking
        self.current_section = None  # Track current section (preconditions, actors, etc.)
        
        # Generated contexts tracking
        self.generated_contexts: Dict[str, List[GeneratedContext]] = {}

    def _load_domain_materials(self) -> Dict[str, List[str]]:
        """
        Load domain materials from domain configuration (GENERIC!)

        Returns:
            Dictionary mapping base_name -> list of variants
            Example: {'coffee': ['coffee', 'espresso', 'latte'], ...}
        """
        # Get domain config
        domain_config = self.verb_loader.domain_configs.get(self.domain_name, {})
        domain_materials_config = domain_config.get('domain_materials', {})

        # Build material dictionary
        materials = {}
        for material_key, material_data in domain_materials_config.items():
            base_name = material_data.get('base_name', material_key)
            variants = material_data.get('variants', [material_key])
            materials[base_name] = variants

        return materials

    def _load_common_controllers(self) -> List[dict]:
        """
        Load common controllers from common_domain.json (GENERIC!)

        Returns:
            List of controller definitions sorted by priority
            Example: [{'name': 'HMIManager', 'keywords': [...], 'verbs': [...], 'priority': 100}, ...]
        """
        # Get common domain config
        common_config = self.verb_loader.domain_configs.get('common_domain', {})
        common_controllers_config = common_config.get('common_controllers', {}).get('controllers', {})

        # Build controller list
        controllers = []
        for controller_name, controller_data in common_controllers_config.items():
            controllers.append({
                'name': controller_name,
                'description': controller_data.get('description', ''),
                'keywords': controller_data.get('keywords', []),
                'verbs': controller_data.get('verbs', []),
                'priority': controller_data.get('priority', 0)
            })

        # Sort by priority (highest first)
        controllers.sort(key=lambda x: x['priority'], reverse=True)

        return controllers

    def _is_control_action_verb(self, verb_lemma: str) -> bool:
        """
        Check if verb is a control action verb (stop, switch, pause, etc.)
        These verbs represent internal state changes and do NOT produce entities.

        Args:
            verb_lemma: Lemmatized verb (e.g., "stop", "switch")

        Returns:
            True if verb is a control action verb, False otherwise
        """
        # Get common domain config
        common_config = self.verb_loader.domain_configs.get('common_domain', {})
        control_action_verbs = common_config.get('control_action_verbs', {}).get('verbs', {})

        return verb_lemma in control_action_verbs

    def _is_gerund_action_phrase(self, entity_name: str, original_phrase: str) -> bool:
        """
        Check if entity name is a gerund action phrase (verb-ing + noun).
        These describe actions, not entities.

        Examples:
            "brewing coffee" -> True (action, not entity)
            "heating water" -> True (action, not entity)
            "ground coffee" -> False (entity, not action)

        Args:
            entity_name: Cleaned entity name (e.g., "BrewingCoffee")
            original_phrase: Original phrase (e.g., "brewing coffee")

        Returns:
            True if this is a gerund action phrase, False otherwise
        """
        if not original_phrase or not self.nlp:
            return False

        # Parse the original phrase
        doc = self.nlp(original_phrase.lower())

        # Check if first token is a gerund (VBG) that represents a function/transformation
        if len(doc) >= 2:
            first_token = doc[0]
            # Check if first word is a verb in gerund form (VBG)
            if first_token.tag_ == 'VBG' or (first_token.pos_ == 'VERB' and first_token.text.endswith('ing')):
                # Check if this verb is a known transformation or function verb
                verb_lemma = first_token.lemma_

                # Check if it's a transformation verb
                transformation = self.verb_loader.get_transformation_for_verb(verb_lemma, self.domain_name)
                if transformation:
                    return True  # It's a transformation action, not an entity

                # Check if it's a known function verb
                verb_type = self._classify_verb(verb_lemma)
                if verb_type in [VerbType.TRANSFORMATION_VERB, VerbType.FUNCTION_VERB]:
                    return True  # It's a function action, not an entity

        return False

    def _load_spacy(self):
        """Load spaCy model"""
        try:

            sys.path.append("E:\\spacy")
            self.nlp = spacy.load("en_core_web_lg")
            print("[OK] spaCy model loaded successfully")
        except OSError:
            print("ERROR: spaCy model 'en_core_web_md' not found")
            print("Please install: python -m spacy download en_core_web_md")
            sys.exit(1)

    def _find_triggered_protection_functions(self, material: str, text: str, step_id: str = "") -> List[Tuple[str, str, str]]:
        """
        Find protection functions that should be triggered based on text patterns.

        Args:
            material: Material name (e.g., 'water', 'milk', 'sugar')
            text: UC step text to analyze
            step_id: Step ID for logging (optional)

        Returns:
            List of tuples: (function_name, criticality, constraint)
        """
        triggered = []
        text_lower = text.lower()
        material_lower = material.lower()

        # Map material names to protection function keys
        material_mapping = {
            'water': 'water',
            'milk': 'milk',
            'sugar': 'sugar',
            'coffee': 'coffee_beans',
            'coffee beans': 'coffee_beans',
            'coffeebeans': 'coffee_beans',
            'filter': 'filter'
        }

        protection_key = material_mapping.get(material_lower)
        if not protection_key or protection_key not in self.protection_functions:
            return triggered

        material_protections = self.protection_functions[protection_key]

        # Check all function types (safety_functions, hygiene_functions, quality_functions)
        for func_type in ['safety_functions', 'hygiene_functions', 'quality_functions']:
            if func_type not in material_protections:
                continue

            for protection_func in material_protections[func_type]:
                func_name = protection_func['name']
                trigger_patterns = protection_func.get('trigger_patterns', [])
                criticality = protection_func.get('criticality', 'medium')
                constraint = protection_func.get('constraint', '')

                # Check if any trigger pattern matches the text
                for pattern in trigger_patterns:
                    if pattern.lower() in text_lower:
                        triggered.append((func_name, criticality, constraint))
                        print(f"[PROTECTION] Triggered {func_name} for {material} in {step_id}: '{pattern}' matched")
                        break  # Don't match same function multiple times

        return triggered

    def _add_protection_functions_to_controller(self, controller_name: str, material: str, step_text: str, step_id: str):
        """
        Add implicit protection functions to a material controller.

        Args:
            controller_name: Controller name (e.g., 'WaterLiquidManager')
            material: Material name (e.g., 'water')
            step_text: UC step text
            step_id: Step ID
        """
        triggered_funcs = self._find_triggered_protection_functions(material, step_text, step_id)

        if triggered_funcs:
            controller = self.controller_registry.get_controller_by_name(controller_name)
            if controller:
                for func_name, criticality, constraint in triggered_funcs:
                    # Add as implicit function (marked with prefix)
                    controller.add_function(f"{func_name} [implicit-{criticality}]")
                    print(f"[PROTECTION] Added {func_name} to {controller_name} (criticality: {criticality})")

    def _detect_aggregation_state(self, text_lower: str, verb_lemma: str, material_name: str) -> Optional[Tuple[str, List[str]]]:
        """
        Generic aggregation state detection for any material: solid/liquid/gas
        
        Args:
            text_lower: Lowercase text to analyze
            verb_lemma: Lemmatized verb
            material_name: Name of the material (e.g., 'coffee', 'milk', 'water')
            
        Returns:
            Tuple of (state, warnings) or None if no clear state detected
            state: 'solid', 'liquid', 'gas'
            warnings: List of ambiguity warnings
        """
        domain_config = self.verb_loader.domain_configs[self.domain_name]
        aggregation_states = domain_config.get('aggregation_states', {})
        material_contexts = domain_config.get('material_specific_contexts', {})
        
        warnings = []
        detected_states = []
        
        # Get material-specific context
        material_context = material_contexts.get(material_name, {})
        ambiguous_terms = material_context.get('ambiguous_terms', [])
        
        # Check if text contains ambiguous terms that need context analysis
        has_ambiguous_terms = any(term in text_lower for term in ambiguous_terms)
        
        # Get universal keywords that apply to all states
        universal_keywords = domain_config.get('universal_keywords', {}).get('all_states', [])
        has_universal_keywords = any(keyword in text_lower for keyword in universal_keywords)
        
        # Check each aggregation state
        for state_name, state_config in aggregation_states.items():
            specific_keywords = state_config.get('specific_keywords', [])
            specific_operations = state_config.get('specific_operations', [])
            
            # Check material-specific indicators
            specific_indicators = material_context.get(f'{state_name}_indicators', [])
            
            # Check for specific keyword matches (excluding universal keywords)
            specific_keyword_match = any(keyword in text_lower for keyword in specific_keywords + specific_indicators)
            
            # Check for operation matches
            operation_match = verb_lemma in specific_operations
            
            # Detect state based on specific keywords or operations
            if specific_keyword_match or operation_match:
                detected_states.append((state_name, state_config.get('state_suffix', state_name.title())))
                print(f"[DEBUG AGGREGATION] Detected {material_name} {state_name.upper()} state: specific_match={specific_keyword_match}, operation_match={operation_match}")
        
        # Handle universal keywords (ambiguous cases)
        if has_universal_keywords and len(detected_states) == 0:
            warning = f"UNIVERSAL KEYWORD WARNING: Found universal keywords (measure/amount/etc.) for '{material_name}' without specific state indicators in text: '{text_lower[:50]}...'"
            warnings.append(warning)
            print(f"[WARNING AGGREGATION] {warning}")
        
        # Handle results
        if len(detected_states) == 0:
            if has_ambiguous_terms:
                warning = f"AMBIGUITY WARNING: Material '{material_name}' found in ambiguous context. Cannot determine aggregation state from text: '{text_lower[:50]}...'"
                warnings.append(warning)
                print(f"[WARNING AGGREGATION] {warning}")
                return None
            # No clear indicators, return default based on common physical state
            default_state = self._get_default_material_state(material_name)
            if default_state:
                return (default_state, warnings)
            return None
            
        elif len(detected_states) == 1:
            state_name, state_suffix = detected_states[0]
            return (state_name, warnings)
            
        else:
            # Multiple states detected - this is an ambiguity
            state_names = [state[0] for state in detected_states]
            warning = f"AMBIGUITY WARNING: Multiple aggregation states detected for '{material_name}': {state_names} in text: '{text_lower[:50]}...'"
            warnings.append(warning)
            print(f"[WARNING AGGREGATION] {warning}")
            
            # Return the first detected state but with warning
            state_name, state_suffix = detected_states[0]
            return (state_name, warnings)
    
    def _get_default_material_state(self, material_name: str) -> Optional[str]:
        """Get default physical state for materials from domain JSON configuration"""
        if self.domain_name not in self.verb_loader.domain_configs:
            return None
            
        domain_config = self.verb_loader.domain_configs[self.domain_name]
        defaults = domain_config.get('default_material_states', {})
        return defaults.get(material_name)
    
    def _generate_aggregation_controller_name(self, material_name: str, aggregation_state: str) -> str:
        """Generate controller name based on material and aggregation state"""
        domain_config = self.verb_loader.domain_configs[self.domain_name]
        aggregation_states = domain_config.get('aggregation_states', {})
        
        state_config = aggregation_states.get(aggregation_state, {})
        state_suffix = state_config.get('state_suffix', aggregation_state.title())
        
        material_base = material_name.title()
        return f"{material_base}{state_suffix}Manager"
    
    def _spell_correct_text(self, text: str) -> str:
        """
        Apply GENERIC spell correction using pyspellchecker (NO domain-specific hard-coding!)

        Args:
            text: Text to correct

        Returns:
            Spell-corrected text
        """
        try:
            from spellchecker import SpellChecker

            # Initialize English spell checker (generic!)
            spell = SpellChecker()

            # Split text into words while preserving structure
            import re
            words = re.findall(r'\b\w+\b', text)

            # Check and correct each word (with context awareness)
            corrections_made = {}
            for i, word in enumerate(words):
                # Skip short words, numbers, and capitalized words (likely proper nouns)
                if len(word) < 3 or word.isdigit() or word[0].isupper():
                    continue

                # CONTEXT-AWARE CORRECTION: Check for common patterns
                # Pattern: "coffee beens" should become "coffee beans" (not "coffee been")
                if word.lower() == "beens" and i > 0:
                    # Get previous word
                    prev_word = words[i - 1].lower()
                    # Load materials from domain JSON (GENERIC!)
                    domain_materials = self._load_domain_materials()
                    material_terms = []
                    for material_variants in domain_materials.values():
                        material_terms.extend([v.lower() for v in material_variants])

                    if prev_word in material_terms:
                        # Force correction to "beans" instead of "been"
                        corrections_made[word] = "beans"
                        continue

                # Get standard correction if word is misspelled
                corrected = spell.correction(word.lower())
                if corrected and corrected != word.lower():
                    corrections_made[word] = corrected

            # Apply corrections to text
            corrected_text = text
            for misspelling, correction in corrections_made.items():
                # Use word boundary matching to avoid partial replacements
                pattern = r'\b' + re.escape(misspelling) + r'\b'
                corrected_text = re.sub(pattern, correction, corrected_text, flags=re.IGNORECASE)

            # Log corrections if any were made
            if corrections_made:
                print(f"[SPELL CORRECTION] '{text}' -> '{corrected_text}'")

            return corrected_text

        except ImportError:
            # If pyspellchecker not installed, return text unchanged
            # User can install: pip install pyspellchecker
            return text
        except Exception as e:
            # If spell check fails, return original text
            print(f"[SPELL CHECK ERROR] {e}")
            return text
    
    def _detect_material_state_controller(self, text_lower: str, verb_lemma: str) -> Optional[str]:
        """
        Detect controller based on material and aggregation state using NLP token analysis
        
        Args:
            text_lower: Lowercase text to analyze
            verb_lemma: Lemmatized verb
            
        Returns:
            Controller name or None if no material state detected
        """
        # Apply spell correction first
        corrected_text = self._spell_correct_text(text_lower)
        
        # Use spaCy NLP to analyze the corrected text
        doc = self.nlp(corrected_text)
        
        # Extract all token lemmas and noun chunk roots
        token_lemmas = [token.lemma_ for token in doc]
        noun_roots = [chunk.root.lemma_ for chunk in doc.noun_chunks]
        all_lemmas = set(token_lemmas + noun_roots)
        
        # Get all materials from domain configuration
        domain_config = self.verb_loader.domain_configs[self.domain_name]
        material_contexts = domain_config.get('material_specific_contexts', {})
        
        # Check each material using NLP lemmas
        for material_name, material_context in material_contexts.items():
            # Check if material lemma is found in NLP analysis
            found_material = material_name in all_lemmas
            
            # Also check for compound forms like "coffee" from "coffee beans"
            if not found_material:
                # Check if material name appears in any noun chunk
                for chunk in doc.noun_chunks:
                    if material_name in chunk.text.lower():
                        found_material = True
                        break
            
            if found_material:
                print(f"[DEBUG NLP MATERIAL] Found material '{material_name}' in corrected text")
                
                # Try to detect aggregation state for this material
                state_result = self._detect_aggregation_state(corrected_text, verb_lemma, material_name)
                
                if state_result:
                    aggregation_state, warnings = state_result
                    
                    # Add warnings to global tracking
                    self.aggregation_warnings.extend(warnings)
                    
                    # Generate controller name
                    controller_name = self._generate_aggregation_controller_name(material_name, aggregation_state)
                    
                    # Get requirements and features for debug output
                    requirements = material_context.get('requirements', {}).get(aggregation_state, [])
                    features = material_context.get('variant_features', {}).get(aggregation_state, [])
                    
                    print(f"[DEBUG MATERIAL STATE] Detected {material_name} {aggregation_state.upper()} state: {controller_name} (requirements: {requirements}, features: {features})")
                    
                    return controller_name
        
        return None
    
    def analyze_uc_file(self, uc_file_path: str) -> Tuple[List[LineAnalysis], str]:
        """
        Main analysis method - process UC file line by line
        
        Args:
            uc_file_path: Path to UC file
            
        Returns:
            Tuple of (line_analyses, output_json_path)
        """
        print(f"[ANALYZE] Starting structured analysis of: {uc_file_path}")
        
        # Backup: Move old "new" contents to "old" folder
        self._backup_old_analysis()
        
        # Read and process file line by line
        with open(uc_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Store all lines for look-ahead functionality
        self.all_lines = [line.strip() for line in lines]
        
        # Extract global UC context (goal, title) from the entire UC text
        self._extract_uc_global_context()
        
        # Process each line
        for line_number, line_text in enumerate(lines, 1):
            line_analysis = self._analyze_line(line_number, line_text.strip())
            self.line_analyses.append(line_analysis)
            
            # Update context based on line type
            self._update_context(line_analysis)
            
            # Track step sequence for control flows
            if line_analysis.step_id:
                self.step_sequence.append(line_analysis.step_id)
        
        # Post-processing: Generate control flows
        self._generate_control_flows()

        # Post-processing: Generate Actor-Boundary and Boundary-Controller flows
        self._generate_actor_boundary_flows()

        # Post-processing: Parallel flows are now detected inline during analysis
        # self._detect_parallel_flows()  # No longer needed

        # Post-processing: Error detection and suggestions
        self._detect_errors_and_suggestions()
        
        # Generate outputs
        output_json_path = self._generate_json_output(uc_file_path)
        self._generate_csv_output(uc_file_path)
        
        # Generate RA diagram using official RUP engine
        diagram_path = self._generate_rup_diagram(output_json_path)
        
        return self.line_analyses, output_json_path
    
    def _backup_old_analysis(self):
        """Move old 'new' contents to 'old' folder before new analysis"""
        import os
        import shutil
        from datetime import datetime
        
        # Create timestamp for backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create old directory if it doesn't exist
        os.makedirs("old", exist_ok=True)
        
        # If new directory exists, move its contents to old
        if os.path.exists("new") and os.listdir("new"):
            # Create timestamped subdirectory in old
            backup_dir = f"old/backup_{timestamp}"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Move all files from new to backup directory
            for filename in os.listdir("new"):
                src = os.path.join("new", filename)
                dst = os.path.join(backup_dir, filename)
                if os.path.isfile(src):
                    shutil.move(src, dst)
                    print(f"[BACKUP] Moved {src} -> {dst}")
        
        # Ensure new directory exists for new analysis
        os.makedirs("new", exist_ok=True)
    
    def _analyze_line(self, line_number: int, line_text: str) -> LineAnalysis:
        """Analyze a single line completely with generative context"""
        
        # 0. Apply spell correction to the entire line first
        corrected_line_text = self._spell_correct_text(line_text)
        
        # 1. Determine line type
        line_type = self._classify_line_type(corrected_line_text)
        
        # 2. Extract step ID if applicable  
        step_id = self._extract_step_id(corrected_line_text, line_type)
        
        # Debug line classification for Sugar (use original text for debug output)
        if "sugar" in line_text.lower():
            print(f"[DEBUG SUGAR] Line: '{line_text}' -> Type: {line_type}, Section: {getattr(self, 'current_section', 'None')}")
        
        # 3. Check if this is a meta-line that should only provide context (no RA classes)
        is_meta_line = line_type in [LineType.CAPABILITY, LineType.FEATURE, LineType.USE_CASE, LineType.GOAL, LineType.PRECONDITION]
        
        # 4. Generate contexts using NLP and domain knowledge - SKIP for meta-lines
        generated_contexts = []
        if corrected_line_text.strip() and not is_meta_line:  # Only for non-empty, non-meta lines
            generated_contexts = self.context_manager.generate_contexts_for_text(corrected_line_text, step_id)
            if step_id:
                self.generated_contexts[step_id] = generated_contexts
        
        # 5. Perform grammatical analysis for step lines
        grammatical = GrammaticalAnalysis()
        if line_type == LineType.STEP:
            grammatical = self._perform_grammatical_analysis(corrected_line_text)
        
        # 6. Determine step context using both traditional and generative approaches
        step_context = None
        if step_id:
            step_context = self._determine_step_context_enhanced(step_id, corrected_line_text, line_type, grammatical, generated_contexts)
        
        # 7. Generate RA classes - SKIP for meta-lines (only context, no RA classes)
        ra_classes = []
        if not is_meta_line:
            ra_classes = self._generate_ra_classes_for_line_enhanced(corrected_line_text, line_type, step_id, grammatical, step_context, generated_contexts)
        elif line_type == LineType.PRECONDITION:
            # Special case: Preconditions generate only essential entities and boundaries
            ra_classes = self._generate_precondition_entities_only(corrected_line_text, line_type)
            if "sugar" in line_text.lower():
                print(f"[DEBUG SUGAR RA] Precondition RA classes: {[ra.name for ra in ra_classes]}")
        
        # 8. Generate data flows (will add control flows later in post-processing)
        data_flows = []
        if not is_meta_line:
            data_flows = self._generate_data_flows_for_line(step_id, grammatical, ra_classes)
        
        # 9. Check for parallel flow patterns using simple logic
        parallel_nodes = []
        if step_id and not is_meta_line:
            parallel_nodes = self._create_parallel_nodes_simple(step_id)
        
        return LineAnalysis(
            line_number=line_number,
            line_text=line_text,  # Keep original text for display purposes
            line_type=line_type,
            step_id=step_id,
            step_context=step_context,
            grammatical=grammatical,
            ra_classes=ra_classes,
            data_flows=data_flows,
            parallel_nodes=parallel_nodes
        )
    
    def _classify_line_type(self, line_text: str) -> LineType:
        """Classify the type of UC line with section context tracking"""
        line_lower = line_text.lower().strip()
        
        if not line_text.strip():
            return LineType.EMPTY
        
        # Check for section headers and update current section
        if line_lower.startswith("capability:"):
            self.current_section = "capability"
            return LineType.CAPABILITY
        elif line_lower.startswith("feature:"):
            self.current_section = "feature"
            return LineType.FEATURE
        elif line_lower.startswith("use case"):
            self.current_section = "use_case"
            return LineType.USE_CASE
        elif line_lower.startswith("goal:"):
            self.current_section = "goal"
            return LineType.GOAL
        elif line_lower.startswith("precondition"):
            self.current_section = "preconditions"
            return LineType.PRECONDITION
        elif line_lower.startswith("actors:"):
            self.current_section = "actors"
            return LineType.ACTOR
        elif line_lower.startswith("main flow") or line_lower.startswith("basic flow"):
            self.current_section = "main_flow"
            return LineType.MAIN_FLOW
        elif line_lower.startswith("alternative flow") or line_lower.startswith("alt flow"):
            self.current_section = "alternative_flow"
            return LineType.ALT_FLOW
        elif line_lower.startswith("extension flow") or line_lower.startswith("ext flow"):
            self.current_section = "extension_flow"
            return LineType.EXT_FLOW
        elif line_lower.startswith("postcondition"):
            self.current_section = "postconditions"
            return LineType.POSTCONDITION
        elif self._is_end_statement(line_lower):
            return self._classify_end_statement(line_lower)
        elif self._is_continue_statement(line_lower):
            return self._classify_continue_statement(line_lower)
        elif re.match(r'^[BAEF]\d+[a-z]?(\.\d+)?\s', line_text):  # Step pattern: B1, A1.1, E2, etc.
            return LineType.STEP
        # Context-based classification: lines in precondition section are preconditions
        elif line_text.startswith("-") and self.current_section == "preconditions":
            return LineType.PRECONDITION
        elif line_text.startswith("-") and "precondition" in self._get_recent_context():
            return LineType.PRECONDITION
        else:
            return LineType.UNKNOWN
    
    def _get_recent_context(self) -> str:
        """Get context from recent lines for classification help"""
        if len(self.line_analyses) >= 3:
            recent_lines = [la.line_text.lower() for la in self.line_analyses[-3:]]
            return " ".join(recent_lines)
        return ""
    
    def _is_end_statement(self, line_lower: str) -> bool:
        """Check if line is an end statement"""
        end_patterns = [
            "end uc",
            "end use case",
            r"^[BAEF]\d+[a-z]?(?:\.\d+)?\s+end\s+uc\s*$",
            r"^[BAEF]\d+[a-z]?(?:\.\d+)?\s+end\s+use\s+case\s*$"
        ]
        
        for pattern in end_patterns:
            if re.search(pattern, line_lower):
                return True
        return False
    
    def _classify_end_statement(self, line_lower: str) -> LineType:
        """Classify specific type of end statement"""
        if "end use case" in line_lower:
            return LineType.END_USE_CASE
        else:
            return LineType.END_UC
    
    def _is_continue_statement(self, line_lower: str) -> bool:
        """Check if line is a continue statement"""
        continue_patterns = [
            r"continue\s+(basic|main)\s+flow",
            r"continue\s+at\s+[baef]\d+[a-z]?(?:\.\d+)?",
            r"continue\s+with\s+[baef]\d+[a-z]?(?:\.\d+)?",
            r"continue\s+at\s+step\s+[baef]\d+[a-z]?(?:\.\d+)?",
            r"continue\s+in\s+(basic|main|alternative|extension)\s+flow",
            r"^[baef]\d+[a-z]?(?:\.\d+)?\s+continue\s+",
            r"^[baef]\d+[a-z]?(?:\.\d+)?\s+continue\s+with\s+[baef]\d+[a-z]?(?:\.\d+)?",
            r"^[baef]\d+[a-z]?(?:\.\d+)?\s+continue\s+at\s+[baef]\d+[a-z]?(?:\.\d+)?"
        ]
        
        for pattern in continue_patterns:
            if re.search(pattern, line_lower):
                return True
        return False
    
    def _classify_continue_statement(self, line_lower: str) -> LineType:
        """Classify specific type of continue statement"""
        if re.search(r"continue\s+at\s+[baef]\d+[a-z]?(?:\.\d+)?", line_lower) or \
           re.search(r"continue\s+with\s+[baef]\d+[a-z]?(?:\.\d+)?", line_lower) or \
           re.search(r"continue\s+at\s+step\s+[baef]\d+[a-z]?(?:\.\d+)?", line_lower):
            return LineType.CONTINUE_AT
        else:
            return LineType.CONTINUE_FLOW
    
    def _extract_step_id(self, line_text: str, line_type: LineType) -> str:
        """Extract step ID from step lines, end statements, and continue statements"""
        if line_type == LineType.STEP:
            match = re.match(r'^([BAEF]\d+[a-z]?(?:\.\d+)?)', line_text)
            if match:
                return match.group(1)
        elif line_type in [LineType.END_UC, LineType.END_USE_CASE]:
            # Extract step ID from "B9 End UC" or "A1.4 End UC" or similar
            match = re.match(r'^([BAEF]\d+[a-z]?(?:\.\d+)?)', line_text)
            if match:
                return match.group(1)
        elif line_type in [LineType.CONTINUE_FLOW, LineType.CONTINUE_AT]:
            # Extract step ID from "A1.4 Continue with B3" or similar
            match = re.match(r'^([BAEF]\d+[a-z]?(?:\.\d+)?)', line_text)
            if match:
                return match.group(1)
        return ""
    
    def _update_context(self, line_analysis: LineAnalysis):
        """Update UC context based on line analysis"""
        line_text = line_analysis.line_text
        line_type = line_analysis.line_type
        
        if line_type == LineType.CAPABILITY:
            self.uc_context.capability = line_text.split(":", 1)[1].strip()
        elif line_type == LineType.FEATURE:
            self.uc_context.feature = line_text.split(":", 1)[1].strip()
        elif line_type == LineType.USE_CASE:
            self.uc_context.use_case_name = line_text.split(":", 1)[1].strip()
        elif line_type == LineType.GOAL:
            self.uc_context.goal = line_text.split(":", 1)[1].strip()
        elif line_type == LineType.ACTOR:
            actors_text = line_text.split(":", 1)[1].strip()
            self.uc_context.actors = [a.strip() for a in actors_text.split(",")]
            # System is NOT an actor - actors are external to the system
        elif line_type == LineType.PRECONDITION and line_text.startswith("-"):
            self.uc_context.preconditions.append(line_text[1:].strip())
    
    def _perform_grammatical_analysis(self, line_text: str) -> GrammaticalAnalysis:
        """Perform complete grammatical analysis of a step line"""
        
        # Remove step ID for cleaner analysis
        clean_text = re.sub(r'^[BAEF]\d+[a-z]?\s*', '', line_text)
        clean_text = re.sub(r'^\(.*?\)\s*', '', clean_text)  # Remove (trigger) etc.
        
        # Remove alternative/extension flow trigger patterns
        clean_text = re.sub(r'^at\s+[BAEF]\d+[a-z]?\s*', '', clean_text)  # Remove "at B2a"
        clean_text = re.sub(r'^at\s+any\s+time\s*', '', clean_text)      # Remove "at any time"
        clean_text = re.sub(r'^[BAEF]\d+[a-z]?-[BAEF]\d+[a-z]?\s*', '', clean_text)  # Remove "B3-B5"
        
        # Process with spaCy
        doc = self.nlp(clean_text)
        
        analysis = GrammaticalAnalysis()
        
        # Find compound nouns
        analysis.compound_nouns = self._extract_compound_nouns(doc)
        
        # Find gerunds (-ing forms used as nouns)
        analysis.gerunds = [token.text for token in doc if token.tag_ == "VBG" and token.dep_ in ["nsubj", "dobj"]]
        
        # Find main verb and classify - handle "begin + gerund" pattern
        main_verb_token = self._find_main_verb(doc)
        if main_verb_token:
            # Check for "begin/start + gerund" pattern 
            if main_verb_token.lemma_ in ["begin", "start"]:
                gerund_verb = self._find_gerund_after_begin(main_verb_token, doc)
                if gerund_verb:
                    # Use the gerund as the main verb (e.g., "begins brewing" -> "brew")
                    analysis.main_verb = gerund_verb.lemma_
                    analysis.verb_lemma = gerund_verb.lemma_
                    analysis.verb_type = self._classify_verb(gerund_verb.lemma_)
                else:
                    # No gerund found, use begin/start as main verb
                    analysis.main_verb = main_verb_token.text
                    analysis.verb_lemma = main_verb_token.lemma_
                    analysis.verb_type = self._classify_verb(analysis.verb_lemma)
            else:
                analysis.main_verb = main_verb_token.text
                analysis.verb_lemma = main_verb_token.lemma_
                analysis.verb_type = self._classify_verb(analysis.verb_lemma)
        
        # Find weak verbs (begin, start, continue, etc.)
        weak_verb_patterns = ["begin", "start", "continue", "proceed", "initiate", "commence"]
        analysis.weak_verbs = [token.text for token in doc if token.lemma_ in weak_verb_patterns]
        
        # Find direct object - if we found a gerund after begin/start, use its direct object
        if main_verb_token:
            if main_verb_token.lemma_ in ["begin", "start"]:
                gerund_verb = self._find_gerund_after_begin(main_verb_token, doc)
                if gerund_verb:
                    # Use gerund's direct object
                    analysis.direct_object = self._find_direct_object(gerund_verb)
                else:
                    # No gerund, use begin/start's direct object
                    analysis.direct_object = self._find_direct_object(main_verb_token)
            else:
                analysis.direct_object = self._find_direct_object(main_verb_token)
        else:
            analysis.direct_object = ""
        
        # Find prepositional objects
        # IMPORTANT: Also search in xcomp/ccomp verbs (e.g., "begins brewing")
        analysis.prepositional_objects = []
        if main_verb_token:
            # Get prep objects from main verb
            analysis.prepositional_objects = self._find_prepositional_objects(main_verb_token)

            # Also search in complement verbs (xcomp, ccomp)
            for child in main_verb_token.children:
                if child.dep_ in ['xcomp', 'ccomp'] and child.pos_ == 'VERB':
                    # Get prep objects from complement verb
                    complement_preps = self._find_prepositional_objects(child)
                    # Merge with main verb preps (avoid duplicates)
                    for prep in complement_preps:
                        if prep not in analysis.prepositional_objects:
                            analysis.prepositional_objects.append(prep)
        
        return analysis
    
    def _extract_compound_nouns(self, doc) -> List[str]:
        """Extract compound nouns like 'Launch Window', 'Mission Control' using spaCy's enhanced analysis"""
        compound_nouns = []
        
        # Method 1: Use spaCy's noun chunks (most reliable)
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            words = chunk_text.split()
            
            # Filter out single words and common articles
            if len(words) > 1:
                # Clean up the chunk
                cleaned_words = []
                for word in words:
                    # Skip articles, determiners, and very common words
                    if word.lower() not in {'the', 'a', 'an', 'this', 'that', 'all', 'some', 'any'}:
                        cleaned_words.append(word)
                
                if len(cleaned_words) > 1:
                    compound_nouns.append(" ".join(cleaned_words))
        
        # Method 2: Adjacent proper nouns and nouns with compound patterns
        i = 0
        while i < len(doc):
            if doc[i].pos_ in ["NOUN", "PROPN", "ADJ"] and not doc[i].is_stop:
                compound = [doc[i].text]
                j = i + 1
                
                # Look for adjacent nouns, proper nouns, or relevant adjectives
                while j < len(doc) and doc[j].pos_ in ["NOUN", "PROPN"] and not doc[j].is_stop:
                    # Skip very common words
                    if doc[j].text.lower() not in {'system', 'the', 'a', 'an'}:
                        compound.append(doc[j].text)
                    j += 1
                
                # Only keep if we have 2+ meaningful words
                if len(compound) > 1:
                    compound_text = " ".join(compound)
                    if compound_text not in compound_nouns:
                        compound_nouns.append(compound_text)
                i = j
            else:
                i += 1
        
        # Method 3: Look for technical terms with specific patterns
        technical_patterns = []
        for i, token in enumerate(doc):
            # Pattern: Adjective + Noun (like "optimal trajectory", "user defined time")
            if token.pos_ == "ADJ" and i + 1 < len(doc) and doc[i + 1].pos_ == "NOUN":
                pattern = f"{token.text} {doc[i + 1].text}"
                if pattern.lower() not in {'the system', 'all subsystems'} and len(pattern.split()) == 2:
                    technical_patterns.append(pattern)
            
            # Pattern: [Modifier] + "defined" + Noun (like "user defined", "pre defined", "system defined")
            # IMPORTANT: Check for "of" phrases after the noun (e.g., "amount of coffee beans")
            # Supports: NOUN (user, system, factory), ADJ (pre), ADV (automatically)
            if (i + 2 < len(doc) and
                token.pos_ in ["NOUN", "ADJ", "ADV"] and
                doc[i + 1].lemma_.lower() == "define" and
                doc[i + 2].pos_ == "NOUN"):
                # Build the full phrase including modifiers and "of" phrases
                # Start with "user defined amount"
                modifier = token.text  # "user"
                main_entity = doc[i + 2].text  # "amount"
                phrase_parts = [modifier, doc[i + 1].text, main_entity]

                # Check if there's an "of" phrase following (e.g., "of coffee beans")
                if i + 3 < len(doc) and doc[i + 3].text.lower() == "of" and doc[i + 3].pos_ == "ADP":
                    phrase_parts.append("of")
                    # Add all nouns after "of" (handles compound nouns like "coffee beans")
                    j = i + 4
                    while j < len(doc) and doc[j].pos_ in ['NOUN', 'PROPN']:
                        phrase_parts.append(doc[j].text)
                        j += 1

                compound_with_priority = " ".join(phrase_parts)
                technical_patterns.append(compound_with_priority)
                # Also add just the main entity to ensure it gets priority
                technical_patterns.append(main_entity)
        
        compound_nouns.extend(technical_patterns)
        
        # Clean up and filter
        cleaned_compounds = []
        for compound in compound_nouns:
            # Remove leading/trailing spaces and filter short or common phrases
            compound = compound.strip()
            if (len(compound.split()) >= 2 and 
                len(compound) > 5 and  # Minimum length
                compound.lower() not in {'the system', 'all subsystems', 'any time'}):
                cleaned_compounds.append(compound)
        
        # Remove duplicates and return
        return list(set(cleaned_compounds))
    
    def _find_main_verb(self, doc):
        """Find the main verb in the sentence"""
        # Look for ROOT verb first
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return token
        
        # Fallback: find any verb
        for token in doc:
            if token.pos_ == "VERB":
                return token
        
        return None
    
    def _find_gerund_after_begin(self, begin_token, doc):
        """Find gerund (VBG) that follows begin/start verb"""
        # Look for gerund directly following begin/start
        for i, token in enumerate(doc):
            if token == begin_token:
                # Check next tokens for gerund
                for j in range(i + 1, min(i + 3, len(doc))):  # Look ahead up to 2 tokens
                    next_token = doc[j]
                    if next_token.tag_ == "VBG":  # Gerund (present participle)
                        return next_token
                    elif next_token.pos_ not in ["ADV", "DET", "ADP"]:  # Stop if we hit non-auxiliary words
                        break
        return None
    
    def _classify_verb(self, verb_lemma: str) -> VerbType:
        """Classify verb using domain knowledge"""
        verb_config = self.verb_loader.get_verb_configuration(self.domain_name)
        
        if verb_lemma in verb_config.transaction_verbs:
            return VerbType.TRANSACTION_VERB
        elif verb_lemma in verb_config.transformation_verbs:
            return VerbType.TRANSFORMATION_VERB
        elif verb_lemma in verb_config.function_verbs:
            return VerbType.FUNCTION_VERB
        else:
            # Default classification based on common patterns
            if verb_lemma in ["send", "receive", "transmit", "deliver", "provide", "request"]:
                return VerbType.TRANSACTION_VERB
            elif verb_lemma in ["convert", "transform", "generate", "create", "produce"]:
                return VerbType.TRANSFORMATION_VERB
            else:
                return VerbType.FUNCTION_VERB
    
    def _find_direct_object(self, verb_token) -> str:
        """Find direct object of verb, prioritizing complete noun phrases"""
        if not verb_token:
            return ""
        
        # Collect all direct objects
        direct_objects = []
        for child in verb_token.children:
            if child.dep_ == "dobj":
                expanded = self._expand_noun_phrase(child)
                direct_objects.append((child, expanded))
        
        if not direct_objects:
            return ""
        
        # If only one direct object, return it
        if len(direct_objects) == 1:
            return direct_objects[0][1]
        
        # Multiple direct objects: prioritize based on semantic importance
        # Priority 1: Look for compound phrases containing domain-relevant terms from JSON
        domain_config = self.verb_loader.domain_configs.get(self.domain_name, {})
        priority_terms = domain_config.get('priority_terms', [])
        
        for token, expanded in direct_objects:
            if any(term in expanded.lower() for term in priority_terms):
                # Clean up compound terms: extract the core noun from phrases like "user defined time"
                expanded_lower = expanded.lower()
                for term in priority_terms:
                    if term in expanded_lower:
                        # If we find a priority term, use just that term (e.g., "time" from "user defined time")
                        return term
                return expanded
        
        # Priority 2: Choose the rightmost/last direct object (usually the main object)
        # In "user defined time", "time" comes after "user"
        rightmost_obj = max(direct_objects, key=lambda x: x[0].i)  # x[0].i is token position
        return rightmost_obj[1]
    
    def _find_prepositional_objects(self, verb_token) -> List[Tuple[str, str]]:
        """Find prepositional objects (preposition, object) - searches recursively"""
        if not verb_token:
            return []

        prep_objects = []

        # Helper function for recursive search
        def extract_preps(token, depth=0):
            """Recursively extract prepositional phrases from token and its children"""
            if depth > 3:  # Limit recursion depth
                return

            for child in token.children:
                if child.dep_ == "prep":
                    prep = child.text
                    # Find the object of the preposition
                    for prep_child in child.children:
                        if prep_child.dep_ == "pobj":
                            obj = self._expand_noun_phrase(prep_child)
                            prep_objects.append((prep, obj))
                            # Recursively search in the prepositional object
                            extract_preps(prep_child, depth + 1)

                # Also search in direct/indirect objects for nested prepositions
                elif child.dep_ in ["dobj", "pobj", "attr", "compound"]:
                    extract_preps(child, depth + 1)

        # Start recursive search from verb
        extract_preps(verb_token)

        return prep_objects
    
    def _expand_noun_phrase(self, token) -> str:
        """Expand a token to its full noun phrase, including modifiers and complements"""
        # Strategy 1: Try span-based extraction for prepositional objects and direct objects
        # This handles complex phrases like "the user defined amount of water"

        doc = token.doc

        # FILTER: Skip actor subjects like "system", "user" when they are subjects of the main verb
        # Example: "The system grinds..." -> "system" should NOT be an entity
        if token.dep_ == 'nsubj' and token.head.dep_ == 'ROOT':
            # Check if this is a known actor/system term
            actor_terms = ['system', 'user', 'actor', 'operator', 'administrator', 'developer']
            if token.text.lower() in actor_terms:
                return ""  # Don't expand actor subjects

        # IMPORTANT: First, expand BACKWARDS to capture modifiers BEFORE the noun
        # This captures "user defined" before "amount"
        start_idx = token.i

        # Look backwards for modifiers - continue until we hit a clear boundary
        # IMPORTANT: We want to capture "user defined" before "amount"
        i = token.i - 1

        while i >= 0:
            t = doc[i]

            # Always include: DET, ADJ, NOUN (they're part of noun phrases)
            if t.pos_ in ['DET', 'ADJ', 'NOUN']:
                start_idx = i
                i -= 1
                continue

            # IMPORTANT: Check for ROOT verbs FIRST (before including other verbs)
            # Stop at main clause verbs (ROOT) - these are the main action, not modifiers
            # Example: "grinds" in "grinds the user defined amount" is ROOT - STOP
            if t.pos_ == 'VERB' and t.dep_ == 'ROOT':
                break

            # Include VERBs that act as modifiers (past participles, etc.)
            # This includes "defined" in "user defined amount" (defined is NOT ROOT)
            if t.pos_ == 'VERB':
                start_idx = i
                i -= 1
                continue

            # Stop at clear boundaries: prepositions, punctuation, conjunctions
            if t.pos_ in ['ADP', 'PUNCT', 'CCONJ', 'SCONJ']:
                break

            # Default: stop
            break

        # Check if this token is a prepositional/direct object, or subject (for "of" phrases)
        # IMPORTANT: Include "nsubj" to handle cases like "amount of water" where spaCy
        # misparsed "amount" as subject due to spelling errors in following words
        if token.dep_ in ["pobj", "dobj", "nsubj"]:
            end_idx = token.i + 1   # At least include the token

            # Extend span to include following modifiers/complements
            # IMPORTANT: Include "of" phrases (e.g., "amount of water")
            i = token.i + 1
            while i < len(doc):
                t = doc[i]

                # Special handling for "of" - include the FULL prepositional phrase
                # Example: "amount of coffee beans" should include ALL of "of coffee beans"
                if t.text.lower() == "of" and t.pos_ == "ADP":
                    # Include "of"
                    end_idx = i + 1
                    i += 1

                    # Skip determiners after "of"
                    while i < len(doc) and doc[i].pos_ == 'DET':
                        end_idx = i + 1
                        i += 1

                    # Include ALL compound nouns + final noun
                    # Example: "coffee beans" where "coffee" (compound) + "beans" (pobj)
                    # Keep including NOUNs until we hit a non-noun
                    while i < len(doc) and doc[i].pos_ in ['NOUN', 'PROPN']:
                        end_idx = i + 1
                        i += 1

                    continue

                # Stop at other prepositions (except "of"), punctuation, or main verb
                if t.pos_ in ['ADP', 'PUNCT'] or (t.pos_ == 'VERB' and t.dep_ == 'ROOT'):
                    break
                # Stop at adverbs that typically mark boundaries (directly, then, etc.)
                if t.pos_ == 'ADV' and t.dep_ == 'advmod':
                    break
                # Include if it's part of a noun phrase, adjective, or participle
                if t.pos_ in ['NOUN', 'PROPN', 'ADJ', 'DET', 'VERB']:
                    end_idx = i + 1
                    i += 1
                else:
                    break

            # Extract the span
            span_text = doc[start_idx:end_idx].text
            if span_text and len(span_text) > len(token.text):
                return span_text

        # Strategy 2: Use noun chunks as fallback
        base_phrase = None
        for chunk in token.doc.noun_chunks:
            if token in chunk:
                base_phrase = chunk.text
                break

        if not base_phrase:
            base_phrase = token.text

        # Strategy 3: Dependency-based expansion for complex structures
        collected_tokens = []

        def collect_subtree(t, visited=None):
            """Recursively collect tokens that form a coherent noun phrase"""
            if visited is None:
                visited = set()
            if t.i in visited:
                return
            visited.add(t.i)

            collected_tokens.append((t.i, t.text))

            # Include modifiers: amod (adjectives), compound, nmod, etc.
            for child in t.children:
                if child.dep_ in ['amod', 'compound', 'nmod', 'nummod', 'det', 'poss', 'advmod']:
                    collect_subtree(child, visited)
                # Include complements: "grinding" in "grinding degree"
                elif child.dep_ in ['acomp', 'xcomp', 'acl', 'advcl']:
                    collect_subtree(child, visited)

            # Include head if it's part of the phrase
            if t.head and t.dep_ in ['amod', 'compound', 'nmod', 'poss']:
                if t.head.pos_ in ['NOUN', 'PROPN']:
                    collect_subtree(t.head, visited)

        collect_subtree(token)

        # Sort by position and reconstruct phrase
        if collected_tokens:
            collected_tokens.sort(key=lambda x: x[0])
            expanded = ' '.join([t[1] for t in collected_tokens])
            # Return expanded version if it's longer and meaningful
            if len(expanded) > len(base_phrase):
                return expanded

        return base_phrase
    
    # Placeholder methods - will implement in next steps
    def _generate_ra_classes_for_line(self, line_text: str, line_type: LineType, step_id: str, grammatical: GrammaticalAnalysis, step_context: StepContext = None) -> List[RAClass]:
        """Generate RA classes for this line based on UC-Methode rules"""
        ra_classes = []
        
        # 1. Handle special line types first
        if line_type == LineType.PRECONDITION and line_text.startswith("-"):
            ra_classes.extend(self._generate_precondition_classes(line_text, step_id))
            return ra_classes
        elif line_type == LineType.ACTOR:
            ra_classes.extend(self._generate_actor_classes(line_text))
            return ra_classes
        elif line_type in [LineType.END_UC, LineType.END_USE_CASE]:
            # End statements don't generate RA classes
            return ra_classes
        elif line_type in [LineType.CONTINUE_FLOW, LineType.CONTINUE_AT]:
            # Continue statements are flow control only - no RA classes generated
            return ra_classes
        elif line_type != LineType.STEP:
            return ra_classes  # No RA classes for non-step lines
        
        # 2. Handle step lines - full RA analysis
        if not step_id:
            return ra_classes
        
        # Check if this is a TRIGGER step (B1 or contains "(trigger)")
        is_trigger = step_id == "B1" or "(trigger)" in line_text.lower()

        if is_trigger:
            # TRIGGER: Generate Boundary + Controller based on Actor type
            # Example: Actor "Time" -> SystemControlManager for scheduling
            #          Actor "User" -> HMIManager for user interaction (handled in flow generation)

            # 1. Generate Boundary
            boundaries = self._generate_boundaries_for_step(step_id, grammatical, line_text)
            ra_classes.extend(boundaries)

            # 2. Generate Controller based on Actor/Trigger type
            trigger_controller = self._generate_trigger_controller(step_id, line_text, grammatical)
            if trigger_controller:
                ra_classes.append(trigger_controller)

            # Triggers don't generate entities or data flows
            return ra_classes

        # 3. Generate Controller (only for NON-trigger steps with verbs)
        if grammatical.main_verb:
            controller = self._generate_controller_for_step(step_id, grammatical, step_context, line_text)
            if controller:
                ra_classes.append(controller)

        # 4. Generate Entities from objects and compound nouns
        entities = self._generate_entities_for_step(step_id, grammatical, line_text)
        ra_classes.extend(entities)

        # 5. Generate Boundaries based on Actor + Transaction Verb rule
        boundaries = self._generate_boundaries_for_step(step_id, grammatical, line_text)
        ra_classes.extend(boundaries)
        
        return ra_classes
    
    def _generate_precondition_classes(self, line_text: str, step_id: str) -> List[RAClass]:
        """Generate Entity + Boundary for preconditions"""
        ra_classes = []
        
        # Extract entity from precondition text
        precondition_text = line_text[1:].strip()  # Remove "-"
        
        # Extract main entities using spaCy
        doc = self.nlp(precondition_text)
        entities = []
        
        for chunk in doc.noun_chunks:
            # Clean entity name
            entity_name = self._clean_entity_name(chunk.text)
            if entity_name and entity_name not in entities:
                entities.append(entity_name)
        
        # Generate Entity + Supply Boundary for each precondition
        for entity_name in entities:
            # Entity - use deduplication
            if not self._is_existing_actor(entity_name):
                entity = self._get_or_create_entity(entity_name, "PRECONDITION")
                if entity:
                    # Update description to show it's a precondition entity
                    entity.description = f"Precondition entity: {entity_name.lower()}"
                    ra_classes.append(entity)
            
            # Supply Boundary
            boundary_name = f"{entity_name}SupplyBoundary"
            ra_classes.append(RAClass(
                name=boundary_name,
                ra_type=RAType.BOUNDARY,
                stereotype="<<boundary>>",
                description=f"Boundary for {entity_name} supply monitoring and alerts",
                step_id="PRECONDITION"
            ))
        
        return ra_classes
    
    def _generate_actor_classes(self, line_text: str) -> List[RAClass]:
        """Generate Actor classes from actor line"""
        ra_classes = []
        
        actors_text = line_text.split(":", 1)[1].strip()
        actors = [a.strip() for a in actors_text.split(",")]
        
        for actor_name in actors:
            ra_classes.append(RAClass(
                name=actor_name,
                ra_type=RAType.ACTOR,
                stereotype="<<actor>>",
                description=f"System actor: {actor_name}",
                step_id="ACTOR"
            ))
        
        return ra_classes
    
    def _generate_precondition_entities_only(self, line_text: str, line_type: LineType) -> List[RAClass]:
        """Simple precondition analysis: Zeile lesen, Kontext bestimmen, Betriebsstoff -> Boundary + Entity"""
        ra_classes = []
        
        if not self.nlp:
            return ra_classes
        
        # Simple NLP: Betriebsstoff detection using domain configuration
        domain_config = self.verb_loader.domain_configs.get(self.domain_name, {})
        material_types = domain_config.get('operational_materials_addressing', {}).get('material_types', {})
        
        # Kontext bestimmen: ist es ein Betriebsstoff?
        text_lower = line_text.lower()
        detected_materials = []
        
        for material_key in material_types.keys():
            # Direct match oder flexible matching
            if material_key in text_lower or material_key.replace('_', ' ') in text_lower:
                detected_materials.append(material_key)
        
        # Debug: Show detected materials for preconditions
        if line_type == LineType.PRECONDITION:
            print(f"[DEBUG PRECONDITION] Line: {line_text}")
            print(f"[DEBUG PRECONDITION] Detected materials: {detected_materials}")
            print(f"[DEBUG PRECONDITION] Available material types: {list(material_types.keys())}")
        
        # Für jeden Betriebsstoff: nur Boundary + Entity erzeugen
        for material in detected_materials:
            # NLP-basierte Entity-Name Generierung
            entity_name = ''.join(word.capitalize() for word in material.split('_'))
            
            # Entity erstellen
            entity = RAClass(
                name=entity_name,
                ra_type=RAType.ENTITY,
                stereotype="<<entity>>",
                description=f"Precondition entity: {material}"
            )
            ra_classes.append(entity)
            
            # Boundary erstellen
            boundary_name = f"{entity_name}SupplyBoundary"
            boundary = RAClass(
                name=boundary_name,
                ra_type=RAType.BOUNDARY,
                stereotype="<<boundary>>",
                description=f"Boundary for {entity_name} supply monitoring and alerts"
            )
            ra_classes.append(boundary)

            # Add implicit protection functions for precondition materials
            # E.g., "Milk is available" triggers TemperatureProtection, FreshnessProtection
            material_controller = self.controller_registry.find_controller_by_material(material)
            if material_controller:
                self._add_protection_functions_to_controller(
                    controller_name=material_controller.name,
                    material=material,
                    step_text=line_text,
                    step_id="PRECONDITION"
                )

        return ra_classes
    
    def _generate_controller_for_step(self, step_id: str, grammatical: GrammaticalAnalysis, step_context: StepContext = None, line_text: str = "") -> Optional[RAClass]:
        """Generate Controller using domain-agnostic approach from generic_uc_analyzer.py"""
        if not grammatical.main_verb:
            return None

        # PRIORITY 1: Try Material Controller Registry approach
        # This ensures controllers represent MATERIALS, not verbs/implementation
        registry_controller = self._generate_controller_using_registry(step_id, grammatical, line_text)
        if registry_controller:
            print(f"[REGISTRY] Using material controller: {registry_controller.name} for function {grammatical.verb_lemma}() in {step_id}")
            return registry_controller

        # FALLBACK: Use legacy controller generation logic
        # This will be used for special cases (HMI, System, triggers, etc.)

        # SPECIAL HANDLING: Control action verbs (stop, switch, pause, etc.)
        # GENERIC RULE: Search ENTIRE line text for material references
        # - "switch off water heater" -> contains "water" -> WaterLiquidManager
        # - "stop milk addition" -> contains "milk" -> MilkLiquidManager
        # - "stop all actions" -> contains "all" (system-level) -> SystemControlManager
        # - "switch off itself" -> contains "itself" (system-level) -> SystemControlManager

        if grammatical.verb_lemma and self._is_control_action_verb(grammatical.verb_lemma):
            line_lower = line_text.lower()

            # Step 1: Check if it's system-level (acting on "all", "itself", "everything")
            # NOTE: Do NOT use "system" as keyword - all UC steps say "The system does X"
            # We need to check what the system is acting UPON, not that it's "the system"
            system_level_keywords = ['all', 'itself', 'everything']
            is_system_level = any(keyword in line_lower for keyword in system_level_keywords)

            if not is_system_level:
                # Step 2: GENERIC material search in ENTIRE line text
                # Get all known materials from domain JSON (GENERIC!)
                domain_materials = self._load_domain_materials()
                material_names = []
                for material_variants in domain_materials.values():
                    material_names.extend([v.lower() for v in material_variants])

                # Search for ANY material in the line text
                found_material = None
                for material in material_names:
                    if material in line_lower:
                        found_material = material
                        break  # Use first material found

                if found_material:
                    # Material found -> use material-specific controller
                    # Detect aggregation state from context
                    state_result = self._detect_aggregation_state(
                        line_lower, grammatical.verb_lemma, found_material
                    )

                    if state_result:
                        aggregation_state, warnings = state_result
                        state_suffix = aggregation_state.capitalize()
                        controller_name = f"{found_material.capitalize()}{state_suffix}Manager"
                    else:
                        controller_name = f"{found_material.capitalize()}Manager"

                    description = f"Manages {found_material} control: {grammatical.verb_lemma}() in {step_id}"

                    return RAClass(
                        name=controller_name,
                        ra_type=RAType.CONTROLLER,
                        stereotype="<<controller>>",
                        description=description,
                        step_id=step_id
                    )

            # No specific material found OR system-level action -> SystemControlManager
            controller_name = "SystemControlManager"
            description = f"Manages system control operations: {grammatical.verb_lemma}() in {step_id}"

            return RAClass(
                name=controller_name,
                ra_type=RAType.CONTROLLER,
                stereotype="<<controller>>",
                description=description,
                step_id=step_id
            )

        # Create a step-like object for the generic logic
        step_info = type('StepInfo', (), {
            'step_id': step_id,
            'step_text': line_text,
            'flow_type': self._determine_flow_type(step_id)
        })()

        # Create verb analysis object for the generic logic
        verb_analysis = type('VerbAnalysis', (), {
            'original_text': line_text,
            'verb_lemma': grammatical.main_verb,
            'direct_object': grammatical.direct_object,
            'verb_type': grammatical.verb_type,  # Include verb type for transformation detection
            'suggested_functional_activity': None  # We don't use this in structured analyzer
        })()

        # Use the proven generic controller naming logic
        controller_name = self._derive_generic_controller_name(verb_analysis, step_info)
        
        if not controller_name:
            # Fallback to verb-based naming
            controller_name = f"{grammatical.main_verb.capitalize()}Manager"
        
        # Generate description based on verb type
        # Extract domain object and state from controller name for better description
        controller_base = controller_name.replace('Manager', '')

        # Check if controller has aggregation state suffix (Solid, Liquid, Gas)
        aggregation_suffixes = ['Solid', 'Liquid', 'Gas']
        has_state = any(controller_base.endswith(suffix) for suffix in aggregation_suffixes)

        if has_state:
            # Extract domain object and state
            for suffix in aggregation_suffixes:
                if controller_base.endswith(suffix):
                    domain_obj = controller_base.replace(suffix, '')
                    state = suffix
                    description_base = f"{domain_obj} ({state.lower()} state)"
                    break
        else:
            # No state suffix, use controller base as-is
            description_base = controller_base.lower()

        if grammatical.verb_type == VerbType.TRANSFORMATION_VERB:
            description = f"Manages {description_base} transformations: {grammatical.verb_lemma}() in {step_id}"
        elif grammatical.verb_type == VerbType.TRANSACTION_VERB:
            description = f"Manages {description_base} transactions: {grammatical.verb_lemma}() in {step_id}"
        else:
            # Function verb or unknown
            description = f"Manages {grammatical.verb_lemma} function in {step_id}"
        
        # Determine parallel group from step_id
        parallel_group = self._get_parallel_group_from_step_id(step_id)
        
        return RAClass(
            name=controller_name,
            ra_type=RAType.CONTROLLER,
            stereotype="<<controller>>",
            description=description,
            step_id=step_id,
            parallel_group=parallel_group
        )

    def _generate_controller_using_registry(self, step_id: str, grammatical: GrammaticalAnalysis, line_text: str) -> Optional[RAClass]:
        """
        Generate controller using Material Controller Registry.

        Key principle: Controllers represent MATERIALS, verbs become FUNCTIONS.

        Process:
        1. Detect material from text (water, coffee, milk, etc.)
        2. Extract function from verb (heat, pressurize, grind) - ignore implementation (heater, compressor)
        3. Get material controller from registry
        4. Assign function to controller

        Examples:
            "activates the water heater"
                → Material: water
                → Function: heat (NOT activate!)
                → Controller: WaterLiquidManager.heat()

            "starts water compressor to generate pressure"
                → Material: water
                → Function: pressurize (NOT start!)
                → Controller: WaterLiquidManager.pressurize()

        Args:
            step_id: UC step identifier (e.g., 'B2a', 'B3')
            grammatical: NLP grammatical analysis
            line_text: Full UC step text

        Returns:
            RAClass for the material controller with the function, or None
        """
        if not grammatical.main_verb:
            return None

        # IMPORTANT: For transformation verbs, determine aggregation state from transformation!
        # Example: "grind": CoffeeBeans -> GroundCoffee (solid) -> CoffeeSolidManager
        # Example: "brew": GroundCoffee + HotWater -> Coffee (liquid) -> CoffeeLiquidManager
        # Example: "adds milk": Coffee + Milk -> Coffee (liquid) -> MilkLiquidManager
        is_transformation = hasattr(grammatical, 'verb_type') and grammatical.verb_type == VerbType.TRANSFORMATION_VERB

        if is_transformation:
            # Get transformation info to determine output material and state
            transformation_info = self.verb_loader.get_transformation_for_verb(grammatical.verb_lemma, self.domain_name)

            if transformation_info and '->' in transformation_info:
                parts = transformation_info.split('->')
                if len(parts) >= 2:
                    output_material = parts[-1].strip()
                    if '+' in output_material:
                        output_material = output_material.split('+')[0].strip()

                    # GENERIC: Determine aggregation state from transformation output and keywords
                    aggregation_state = self._determine_aggregation_state_from_output(
                        output_material,
                        grammatical.verb_lemma,
                        line_text
                    )

                    # Select controller with matching material AND aggregation state
                    material_controller = self._select_controller_with_state(
                        output_material,
                        aggregation_state,
                        grammatical.direct_object,
                        line_text
                    )

                    if material_controller:
                        print(f"[REGISTRY TRANSFORM] Using {aggregation_state} state controller: {material_controller.name} for {grammatical.verb_lemma}() -> {output_material} in {step_id}")
                    else:
                        material_controller = self.controller_registry.find_controller_by_text(line_text)
                else:
                    material_controller = self.controller_registry.find_controller_by_text(line_text)
            else:
                material_controller = self.controller_registry.find_controller_by_text(line_text)
        else:
            # Find material controller from text (normal case)
            material_controller = self.controller_registry.find_controller_by_text(line_text)

        if material_controller:
            # Extract the actual function from the verb, ignoring implementation elements
            function_name = extract_function_from_verb(
                verb=grammatical.verb_lemma,
                context=line_text
            )

            # Add function to controller's function set
            material_controller.add_function(function_name)

            # Add implicit protection functions based on triggers
            self._add_protection_functions_to_controller(
                controller_name=material_controller.name,
                material=material_controller.material,
                step_text=line_text,
                step_id=step_id
            )

            # Determine parallel group from step_id
            parallel_group = self._get_parallel_group_from_step_id(step_id)

            # Create controller description with all functions (explicit + implicit)
            description = f"Manages {material_controller.material}"
            if material_controller.aggregation_state:
                description += f" ({material_controller.aggregation_state} state)"

            # List all functions (explicit and implicit)
            all_functions = sorted(material_controller.functions)
            explicit_funcs = [f for f in all_functions if '[implicit-' not in f]
            implicit_funcs = [f for f in all_functions if '[implicit-' in f]

            if explicit_funcs:
                description += f": {', '.join(explicit_funcs)} in {step_id}"
            if implicit_funcs:
                # Clean up implicit function names for display
                implicit_display = []
                for f in implicit_funcs:
                    # Extract function name and criticality
                    func_name = f.split(' [implicit-')[0]
                    criticality = f.split('[implicit-')[1].rstrip(']')
                    implicit_display.append(f"{func_name}[{criticality}]")
                description += f" | Implicit: {', '.join(implicit_display)}"

            return RAClass(
                name=material_controller.name,
                ra_type=RAType.CONTROLLER,
                stereotype="<<controller>>",
                description=description,
                step_id=step_id,
                parallel_group=parallel_group
            )

        return None

    def _determine_aggregation_state_from_output(self, output_material: str, verb: str, line_text: str) -> Optional[str]:
        """
        GENERIC: Determine aggregation state from transformation output material.

        Uses domain JSON aggregation_states configuration:
        - solid: "beans", "ground", "powder", "crystals", etc.
        - liquid: "brew", "brewing", "extraction", "pour", "flow", etc.
        - gas: "steam", "vapor", "pressure", etc.

        Args:
            output_material: Output material name (e.g., "GroundCoffee", "Coffee")
            verb: Verb lemma (e.g., "grind", "brew")
            line_text: Full line text for context

        Returns:
            Aggregation state: "solid", "liquid", "gas", or None
        """
        output_lower = output_material.lower()
        line_lower = line_text.lower()

        # Load aggregation state info from domain JSON
        domain_config = self.verb_loader.domain_configs.get(self.domain_name, {})
        aggregation_states = domain_config.get('aggregation_states', {})
        material_contexts = domain_config.get('material_specific_contexts', {})

        # STEP 1: Check output material name for state keywords
        for state, state_info in aggregation_states.items():
            specific_keywords = state_info.get('specific_keywords', [])
            for keyword in specific_keywords:
                if keyword in output_lower:
                    print(f"[AGGREGATION STATE] Detected {state} from output material '{output_material}' (keyword: {keyword})")
                    return state

        # STEP 2: Check verb operation type
        for state, state_info in aggregation_states.items():
            specific_operations = state_info.get('specific_operations', [])
            if verb in specific_operations:
                print(f"[AGGREGATION STATE] Detected {state} from verb '{verb}' operation")
                return state

        # STEP 3: Check material-specific context (e.g., coffee solid/liquid indicators)
        # Extract base material name (e.g., "Coffee" from "GroundCoffee")
        for material_name, context_info in material_contexts.items():
            if material_name in output_lower:
                # Check solid indicators
                solid_indicators = context_info.get('solid_indicators', [])
                for indicator in solid_indicators:
                    if indicator in line_lower or indicator in output_lower:
                        print(f"[AGGREGATION STATE] Detected solid from material context '{material_name}' (indicator: {indicator})")
                        return 'solid'

                # Check liquid indicators
                liquid_indicators = context_info.get('liquid_indicators', [])
                for indicator in liquid_indicators:
                    if indicator in line_lower or indicator in output_lower:
                        print(f"[AGGREGATION STATE] Detected liquid from material context '{material_name}' (indicator: {indicator})")
                        return 'liquid'

                # Check gas indicators
                gas_indicators = context_info.get('gas_indicators', [])
                for indicator in gas_indicators:
                    if indicator in line_lower or indicator in output_lower:
                        print(f"[AGGREGATION STATE] Detected gas from material context '{material_name}' (indicator: {indicator})")
                        return 'gas'

        # STEP 4: Fallback to default material states
        default_states = domain_config.get('default_material_states', {})
        for material, state in default_states.items():
            if material in output_lower:
                print(f"[AGGREGATION STATE] Using default state {state} for material '{material}'")
                return state

        print(f"[AGGREGATION STATE] Could not determine state for '{output_material}', defaulting to None")
        return None

    def _select_controller_with_state(self, material: str, aggregation_state: Optional[str],
                                     direct_object: Optional[str], line_text: str) -> Optional['MaterialController']:
        """
        GENERIC: Select controller with matching material and aggregation state.

        Priority:
        1. If direct object is a material (e.g., "milk") -> use direct object's controller
        2. Otherwise, use output material with aggregation state

        Args:
            material: Material name (e.g., "Coffee", "GroundCoffee", "HotWater")
            aggregation_state: Aggregation state ("solid", "liquid", "gas")
            direct_object: Direct object from grammatical analysis
            line_text: Full line text

        Returns:
            MaterialController with matching material and state
        """
        # PRIORITY 1: Check if direct object is a material
        # Example: "adds milk" -> direct object "milk" -> MilkLiquidManager
        if direct_object:
            material_controller = self.controller_registry.find_controller_by_text(direct_object)
            if material_controller:
                print(f"[CONTROLLER SELECT] Using direct object material: {material_controller.name}")
                return material_controller

        # PRIORITY 2: Use output material with aggregation state
        # Extract base material from compound names:
        # "GroundCoffee" -> "coffee", "HotWater" -> "water", "SteamedMilk" -> "milk"
        base_material = self._extract_base_material_from_output(material)

        # Example: "grinds coffee beans" -> output "GroundCoffee" (solid) -> base "coffee" -> CoffeeSolidManager
        # Example: "begins brewing coffee" -> output "Coffee" (liquid) -> base "coffee" -> CoffeeLiquidManager
        material_controller = self.controller_registry.get_controller_for_material(base_material, aggregation_state)

        if material_controller:
            print(f"[CONTROLLER SELECT] Using output material with state: {material_controller.name} (base: {base_material}, state: {aggregation_state})")
            return material_controller

        # FALLBACK: Use text-based search (no state matching)
        print(f"[CONTROLLER SELECT] Fallback to text-based search for '{material}'")
        return self.controller_registry.find_controller_by_text(line_text)

    def _extract_base_material_from_output(self, output_material: str) -> str:
        """
        GENERIC: Extract base material name from compound output names.

        Examples:
            "GroundCoffee" -> "coffee"
            "HotWater" -> "water"
            "SteamedMilk" -> "milk"
            "FrothedMilk" -> "milk"
            "Coffee" -> "coffee"

        Args:
            output_material: Output material name (e.g., "GroundCoffee", "HotWater")

        Returns:
            Base material name (e.g., "coffee", "water")
        """
        output_lower = output_material.lower()

        # Load all materials from domain JSON
        domain_materials = self._load_domain_materials()

        # Check each base material to see if it appears in the output name
        for base_name, variants in domain_materials.items():
            # Check base name
            if base_name.lower() in output_lower:
                print(f"[BASE MATERIAL] Extracted '{base_name}' from '{output_material}'")
                return base_name

            # Check variants
            for variant in variants:
                if variant.lower() == output_lower:
                    print(f"[BASE MATERIAL] Found exact match '{base_name}' for '{output_material}'")
                    return base_name

        # If no match found, return the original material name
        print(f"[BASE MATERIAL] No base material found for '{output_material}', using as-is")
        return output_material

    def _determine_flow_type(self, step_id: str) -> str:
        """Determine flow type from step_id"""
        if step_id.startswith('A'):
            return "alternative"
        elif step_id.startswith('E'):
            return "extension"
        else:
            return "main"
    
    def _generate_trigger_controller(self, step_id: str, line_text: str, grammatical: GrammaticalAnalysis) -> Optional[RAClass]:
        """
        Generate Controller for trigger steps based on Actor/Trigger type.
        GENERIC APPROACH: Uses Actor list from UC Context instead of hardcoded keywords.

        Actor-based Controller Mapping:
            - Actor "Time", "Clock", "Timer", "Schedule" -> SystemControlManager (scheduling)
            - Actor "User" + interaction -> HMIManager (handled in flow generation)
            - Actor "Sensor", "Monitor", "System" -> SystemControlManager (monitoring)
            - Any other Actor -> SystemControlManager (generic system function)

        Args:
            step_id: Step ID (B1, E1, A1, etc.)
            line_text: Full line text
            grammatical: Grammatical analysis

        Returns:
            Controller RAClass or None
        """
        line_lower = line_text.lower()

        # GENERIC APPROACH: Check which Actor from UC Context triggers this step
        # Get actors from UC Context (e.g., ["User", "Time"])
        actors = self.uc_context.actors if hasattr(self.uc_context, 'actors') else []

        # Check each actor to see if they appear in the trigger text
        for actor in actors:
            actor_lower = actor.lower()

            # Skip "User" actor - handled by _is_real_user_interaction
            if actor_lower == 'user':
                continue

            # Check if this actor triggers this step
            if actor_lower in line_lower:
                # Actor-based controller generation
                function_name = self._get_function_for_actor(actor_lower, line_lower)

                return RAClass(
                    name='SystemControlManager',
                    ra_type=RAType.CONTROLLER,
                    stereotype='<<controller>>',
                    description=f'Manages system control operations: {function_name}() in {step_id}',
                    step_id=step_id
                )

        # Check for User interaction triggers
        if self._is_real_user_interaction(line_lower, grammatical):
            # SPECIAL CASE: B1 User-Trigger needs HMIManager as controller
            # to create flow: B1 -> HMIManager -> P2_START
            if step_id == "B1":
                return RAClass(
                    name='HMIManager',
                    ra_type=RAType.CONTROLLER,
                    stereotype='<<controller>>',
                    description=f'Manages hmi transactions: request() in {step_id}',
                    step_id=step_id
                )
            # For other User triggers (E1, etc.): HMIManager routing in flow generation
            return None

        # Fallback: Check for system condition keywords from common_domain.json
        # Load keywords dynamically from domain configuration
        common_config = self.verb_loader.domain_configs.get('common_domain', {})
        condition_keywords_config = common_config.get('system_condition_keywords', {})
        condition_keywords = condition_keywords_config.get('keywords', [])

        if condition_keywords and any(keyword in line_lower for keyword in condition_keywords):
            return RAClass(
                name='SystemControlManager',
                ra_type=RAType.CONTROLLER,
                stereotype='<<controller>>',
                description=f'Manages system control operations: monitor() in {step_id}',
                step_id=step_id
            )

        # Default: No controller for this trigger type
        return None

    def _get_function_for_actor(self, actor_lower: str, line_text: str) -> str:
        """
        Determine the function name based on actor type and context.
        FULLY GENERIC APPROACH: Uses common_domain.json common_controllers mapping.

        Args:
            actor_lower: Lowercase actor name
            line_text: Lowercase line text for context

        Returns:
            Function name (schedule, monitor, trigger, etc.)
        """
        # Load common controllers from common_domain.json
        common_config = self.verb_loader.domain_configs.get('common_domain', {})
        common_controllers = common_config.get('common_controllers', {}).get('controllers', {})

        # Check TimeManager keywords for scheduling functions
        time_manager = common_controllers.get('TimeManager', {})
        time_keywords = time_manager.get('keywords', [])
        if any(keyword in actor_lower or keyword in line_text for keyword in time_keywords):
            return 'schedule'

        # Check TriggerManager keywords for triggering functions
        trigger_manager = common_controllers.get('TriggerManager', {})
        trigger_keywords = trigger_manager.get('keywords', [])
        if any(keyword in actor_lower or keyword in line_text for keyword in trigger_keywords):
            return 'trigger'

        # Check ControlManager keywords for monitoring/control functions
        control_manager = common_controllers.get('ControlManager', {})
        control_keywords = control_manager.get('keywords', [])
        if any(keyword in actor_lower or keyword in line_text for keyword in control_keywords):
            return 'monitor'

        # Default: Generic trigger function (when actor is not in common patterns)
        return 'trigger'

    def _is_trigger_step(self, step) -> bool:
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
    
    def _is_human_interaction(self, text: str) -> bool:
        """
        Detect if this step involves interaction with a human actor using NLP
        Returns True if the system interacts with a human, False otherwise
        """
        text_lower = text.lower()
        
        # EXCLUDE ENTITY REFERENCES - not interactions
        entity_patterns = [
            "user defined", "user-defined", "user specified", "user input", 
            "user preference", "user setting", "user configuration"
        ]
        
        # Check for entity patterns first - if found, this is NOT an interaction
        for pattern in entity_patterns:
            if pattern in text_lower:
                return False
        
        # DIRECT HUMAN REFERENCES (only for actual interactions)
        human_keywords = [
            "user", "person", "operator", "customer", "client", "passenger", 
            "driver", "pilot", "technician", "worker", "staff", "employee",
            "human", "people", "individual", "someone", "anybody", "anyone"
        ]
        
        # Check for direct human references (but not entity references)
        for keyword in human_keywords:
            if keyword in text_lower:
                return True
        
        # HUMAN INTERACTION PATTERNS
        # Pattern: "wants" - typically indicates human desire/request
        if "want" in text_lower or "request" in text_lower or "ask" in text_lower:
            return True
            
        # Pattern: Communication TO humans
        communication_patterns = [
            "output.*message", "display.*message", "show.*message", "present.*to",
            "communicate.*with", "inform.*about", "notify.*user", "alert.*user"
        ]
        
        import re
        for pattern in communication_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Pattern: Receiving input FROM humans (using spaCy for deeper analysis)
        doc = self.nlp(text)
        
        # Look for human-like subjects performing actions
        for token in doc:
            # Check if token is likely a human subject
            if (token.dep_ in ["nsubj", "nsubjpass"] and 
                token.pos_ in ["NOUN", "PROPN"] and
                any(human_word in token.text.lower() for human_word in human_keywords)):
                return True
                
            # Check for pronouns that could refer to humans in context
            if (token.dep_ in ["nsubj", "nsubjpass"] and 
                token.pos_ == "PRON" and 
                token.text.lower() in ["he", "she", "they", "who"]):
                return True
        
        return False
    
    def _derive_generic_controller_name(self, verb_analysis, step) -> Optional[str]:
        """Generate domain-agnostic controller names (GENERIC from common_domain.json)"""

        text_lower = verb_analysis.original_text.lower()
        verb_lemma = verb_analysis.verb_lemma

        # PRIORITY 1: Common Controllers (from common_domain.json)
        # Load common controllers sorted by priority
        common_controllers = self._load_common_controllers()

        # Check each common controller
        for controller_def in common_controllers:
            # Check if verb matches
            if verb_lemma in controller_def['verbs']:
                # Additional check for HMIManager: verify human interaction
                if controller_def['name'] == 'HMIManager':
                    if self._is_human_interaction(verb_analysis.original_text):
                        return controller_def['name']
                else:
                    return controller_def['name']

            # Check if keywords match
            keyword_matches = sum(1 for keyword in controller_def['keywords'] if keyword in text_lower)
            if keyword_matches >= 2:  # At least 2 keywords match
                return controller_def['name']

        # PRIORITY 2: Trigger detection (B1 or trigger patterns)
        if step.step_id == "B1" or self._is_trigger_step(step):
            # Try to find TimeManager or TriggerManager from common controllers
            for controller_def in common_controllers:
                if controller_def['name'] in ['TimeManager', 'TriggerManager']:
                    for keyword in controller_def['keywords']:
                        if keyword in text_lower:
                            return controller_def['name']
            # Fallback
            if step.flow_type == "alternative":
                return f"{step.step_id.split('.')[0]}ConditionManager"
            else:
                return "TriggerManager"
        
        # Use functional activity if available (for implementation elements)
        if hasattr(verb_analysis, 'suggested_functional_activity') and verb_analysis.suggested_functional_activity:
            # Generic NLP parsing of functional suggestion
            controller_name = self._derive_abstract_controller_from_nlp(verb_analysis.suggested_functional_activity)
            if controller_name:
                return controller_name
        
        # Priority: Generic material state-based controller detection
        text_lower = verb_analysis.original_text.lower()
        
        # Material state-based controller detection (generic approach)
        material_controller = self._detect_material_state_controller(text_lower, verb_analysis.verb_lemma)
        if material_controller:
            return material_controller

        # PRIORITY: Transformation verb detection (NLP-based, domain-agnostic)
        # Transformation verbs should use verb-based naming, not object-based naming
        if hasattr(verb_analysis, 'verb_type') and verb_analysis.verb_type == VerbType.TRANSFORMATION_VERB:
            # Use NLP to extract context for better naming
            controller_name = self._derive_transformation_controller_name(
                verb_analysis.verb_lemma,
                verb_analysis.direct_object,
                text_lower
            )
            if controller_name:
                return controller_name

        # Object-based controller naming (check for implementation elements first)
        if verb_analysis.direct_object:
            # Check if direct object contains implementation elements
            obj_words = verb_analysis.direct_object.lower().split()
            for word in obj_words:
                impl_info = self.verb_loader.get_implementation_element_info(word, self.domain_name)
                if impl_info:
                    # Use verb-based naming instead for implementation elements
                    verb_action = verb_analysis.verb_lemma.capitalize()
                    # GENERIC: Try to extract the actual target from domain materials
                    domain_materials = self._load_domain_materials()
                    for base_material, variants in domain_materials.items():
                        for variant in variants:
                            if variant in verb_analysis.direct_object.lower():
                                return f"{base_material.capitalize()}{verb_action}ingManager"
                    # Fallback to verb-only naming
                    return f"{verb_action}Manager"
            
            # No implementation elements - use object-based naming
            main_object = verb_analysis.direct_object.split()[-1].capitalize()
            return f"{main_object}Manager"
        
        # Verb-based controller naming
        verb_action = verb_analysis.verb_lemma.capitalize()
        return f"{verb_action}Manager"
    
    def _derive_transformation_controller_name(self, verb_lemma: str, direct_object: str, text: str) -> Optional[str]:
        """
        Derive controller name for transformation verbs using domain configuration and NLP context extraction.

        IMPORTANT: Controller aggregation principle
        - Controller represents a DOMAIN OBJECT (Coffee, Water, Milk), NOT a verb/action
        - Verbs become FUNCTIONS of the controller
        - One controller handles multiple related functions across UC steps and UCs

        Examples:
        - "grinds coffee beans" → CoffeeManager (function: grind())
        - "brews coffee" → CoffeeManager (function: brew())
        - "heats water" → WaterManager (function: heat())
        - "steams milk" → MilkManager (function: steam())

        Strategy:
        1. Get transformation from domain config (e.g., "grind": "CoffeeBeans -> GroundCoffee")
        2. Extract the PRIMARY DOMAIN OBJECT (Coffee, Water, Milk, etc.)
        3. Verify with NLP that the material is mentioned in the text
        4. Build controller name: {DomainObject}Manager (NOT {Object}{Verb}ingManager)

        Args:
            verb_lemma: The lemmatized verb (e.g., "grind")
            direct_object: The direct object from NLP (e.g., "set amount")
            text: The full text for context extraction

        Returns:
            Controller name string (e.g., "CoffeeManager", "WaterManager")
        """
        if not verb_lemma:
            return None

        # Parse text with NLP for semantic understanding
        doc = self.nlp(text)

        # Step 1: Get transformation definition from domain configuration
        transformation_info = self.verb_loader.get_transformation_for_verb(verb_lemma, self.domain_name)

        transformation_input = None
        if transformation_info:
            # Parse transformation string: "CoffeeBeans -> GroundCoffee" or "Water -> HotWater"
            if '->' in transformation_info:
                input_part = transformation_info.split('->')[0].strip()
                # Handle multiple inputs: "GroundCoffee + HotWater + Filter"
                if '+' in input_part:
                    # Take the first (primary) input
                    transformation_input = input_part.split('+')[0].strip()
                else:
                    transformation_input = input_part

        # Step 2: Extract material/target context from text using NLP
        text_materials = []

        # Load materials from domain configuration (GENERIC!)
        priority_materials = self._load_domain_materials()

        # Extract nouns and noun chunks from text
        for chunk in doc.noun_chunks:
            chunk_lower = chunk.text.lower()
            # Check against domain materials
            for base_material, variants in priority_materials.items():
                for variant in variants:
                    if variant in chunk_lower:
                        text_materials.append(base_material)
                        break

        # Remove duplicates while preserving order
        seen = set()
        text_materials = [x for x in text_materials if not (x in seen or seen.add(x))]

        # Step 3: Extract PRIMARY DOMAIN OBJECT from transformation or text
        domain_object = None

        if transformation_input:
            # Extract base material from transformation input
            # Examples:
            # - "CoffeeBeans" -> "Coffee"
            # - "GroundCoffee" -> "Coffee"
            # - "HotWater" -> "Water"
            # - "SteamedMilk" -> "Milk"

            transformation_lower = transformation_input.lower()

            # Check which material is in the transformation
            for material in text_materials:
                if material in transformation_lower:
                    domain_object = material
                    break

            # If no direct match, try to infer from transformation name
            if not domain_object:
                # Extract base material from compound words (GENERIC - from domain materials!)
                domain_materials = self._load_domain_materials()

                for base_name, variants in domain_materials.items():
                    # Check if any variant is in the transformation string
                    for variant in variants:
                        if variant.lower() in transformation_lower:
                            domain_object = base_name
                            break
                    if domain_object:
                        break

        # Step 4: Fallback to NLP-extracted materials if no transformation match
        if not domain_object and text_materials:
            # Use first material found in text as domain object
            domain_object = text_materials[0]

        # Step 5: Detect aggregation state for the domain object
        aggregation_state = None
        if domain_object:
            # Use existing aggregation state detection
            state_result = self._detect_aggregation_state(text.lower(), verb_lemma, domain_object)
            if state_result:
                aggregation_state, warnings = state_result
                # Add warnings to global tracking
                self.aggregation_warnings.extend(warnings)
                print(f"[DEBUG TRANSFORMATION] {domain_object} detected as {aggregation_state} state")

        # Step 6: Build controller name based on DOMAIN OBJECT + AGGREGATION STATE
        if domain_object:
            if aggregation_state:
                # Use aggregation-aware controller naming
                # Examples: CoffeeSolidManager, CoffeeLiquidManager, WaterLiquidManager
                controller_name = self._generate_aggregation_controller_name(domain_object, aggregation_state)
                print(f"[DEBUG TRANSFORMATION] Controller: {controller_name} (domain: {domain_object}, state: {aggregation_state})")
                return controller_name
            else:
                # No aggregation state detected, use simple domain object naming
                domain_object_cap = domain_object.capitalize()
                return f"{domain_object_cap}Manager"
        else:
            # Fallback: Pure verb-based naming as last resort
            verb_gerund = self._convert_to_gerund(verb_lemma)
            return f"{verb_gerund}Manager"

    def _convert_to_gerund(self, verb_lemma: str) -> str:
        """
        Convert verb lemma to gerund form (-ing) with proper English grammar rules.

        Examples:
        - grind -> Grinding
        - brew -> Brewing
        - heat -> Heating
        - add -> Adding
        - run -> Running (consonant doubling)
        """
        verb_lower = verb_lemma.lower()

        # Special cases
        special_cases = {
            'die': 'Dying',
            'lie': 'Lying',
            'tie': 'Tying'
        }

        if verb_lower in special_cases:
            return special_cases[verb_lower]

        # Rule 1: Verbs ending in 'e' (not 'ee', 'oe', 'ye')
        if verb_lower.endswith('e') and not verb_lower.endswith(('ee', 'oe', 'ye')):
            return (verb_lower[:-1] + 'ing').capitalize()

        # Rule 2: Consonant doubling (CVC pattern - Consonant Vowel Consonant)
        # Examples: run -> running, stop -> stopping, begin -> beginning
        if len(verb_lower) >= 3:
            vowels = 'aeiou'
            last_three = verb_lower[-3:]
            if (last_three[0] not in vowels and
                last_three[1] in vowels and
                last_three[2] not in vowels and
                last_three[2] not in 'wxy'):  # Don't double w, x, y
                return (verb_lower + verb_lower[-1] + 'ing').capitalize()

        # Rule 3: Default - just add 'ing'
        return (verb_lower + 'ing').capitalize()

    def _derive_abstract_controller_from_nlp(self, functional_suggestion: str) -> Optional[str]:
        """
        Generic NLP analysis to derive abstract controller names from functional suggestions
        Pattern: "The system <verb>s the <target>" -> "<Target>ProcessingManager"
        """
        if not functional_suggestion:
            return None
            
        # Parse with spaCy for semantic understanding
        doc = self.nlp(functional_suggestion.lower())
        
        # Find main action verb and target object
        main_verb = None
        target_object = None
        
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                main_verb = token
                # Find direct object (target)
                for child in token.children:
                    if child.dep_ == "dobj":
                        target_object = child.text
                        break
                break
        
        if target_object:
            # Create abstract controller name: {Target}ProcessingManager
            target_capitalized = target_object.capitalize()
            return f"{target_capitalized}ProcessingManager"
        elif main_verb:
            # Fallback: Use verb-based naming with "Manager" suffix
            verb_base = main_verb.lemma_.capitalize()
            return f"{verb_base}Manager"
            
        return None
    
    
    def _generate_entities_for_step(self, step_id: str, grammatical: GrammaticalAnalysis, line_text: str) -> List[RAClass]:
        """Generate Entities from direct object, prepositional objects, compound nouns, and transformation outputs"""
        entities = []
        entity_names = set()

        # Check if this is a control action verb (stop, switch, pause, etc.)
        # These verbs represent internal state changes and do NOT produce entities
        is_control_action = self._is_control_action_verb(grammatical.verb_lemma)

        # 1. Direct object (highest priority)
        # IMPORTANT: Skip direct object for control action verbs (e.g., "stop milk addition")
        # IMPORTANT: Skip gerund phrases that describe actions (e.g., "brewing coffee", "heating water")
        if grammatical.direct_object and not is_control_action:
            entity_name = self._clean_entity_name(grammatical.direct_object)

            # FILTER: Skip gerund action phrases (verb-ing + noun)
            # Example: "brewing coffee" -> "brewing" is the action, not an entity
            # Example: "heating water" -> "heating" is the action, not an entity
            if entity_name and self._is_gerund_action_phrase(entity_name, grammatical.direct_object):
                # Skip this - it's an action, not an entity
                pass
            # Check if this name is already defined as an Actor or already exists as Entity
            elif entity_name and entity_name not in entity_names and not self._is_existing_actor(entity_name):
                entity = self._get_or_create_entity(entity_name, step_id)
                if entity:
                    entity_names.add(entity_name)
                    entities.append(entity)

        # 2. Prepositional objects (second priority)
        # IMPORTANT: Also skip for control action verbs
        if not is_control_action:
            for prep, obj in grammatical.prepositional_objects:
                entity_name = self._clean_entity_name(obj)
                # Check if this name is already defined as an Actor or already exists as Entity
                if entity_name and entity_name not in entity_names and not self._is_existing_actor(entity_name):
                    entity = self._get_or_create_entity(entity_name, step_id)
                    if entity:
                        entity_names.add(entity_name)
                        entities.append(entity)

        # 3. Transformation entities (for transformation verbs)
        # IMPORTANT: Extract BOTH input and output entities from domain config
        if hasattr(grammatical, 'verb_type') and grammatical.verb_type == VerbType.TRANSFORMATION_VERB:
            transformation_info = self.verb_loader.get_transformation_for_verb(grammatical.verb_lemma, self.domain_name)
            if transformation_info and '->' in transformation_info:
                parts = transformation_info.split('->')

                # Extract INPUT entities (left side of ->)
                if len(parts) >= 1:
                    input_part = parts[0].strip()
                    # Handle multiple inputs: "GroundCoffee + HotWater + Filter"
                    input_entities = [inp.strip() for inp in input_part.split('+')]
                    for input_entity in input_entities:
                        if input_entity and input_entity not in entity_names:
                            # FILTER: Skip abstract "Additive" if we have concrete material in text
                            # Example: "add milk" has "Milk" in text -> skip abstract "Additive"
                            # Example: "add sugar" has "Sugar" in text -> skip abstract "Additive"
                            if input_entity.lower() == "additive":
                                # Check if we have concrete materials in the text (GENERIC from domain JSON)
                                line_lower = line_text.lower()
                                domain_materials = self._load_domain_materials()
                                concrete_materials = []
                                for material_variants in domain_materials.values():
                                    concrete_materials.extend([v.lower() for v in material_variants])

                                has_concrete_material = any(material in line_lower for material in concrete_materials)
                                if has_concrete_material:
                                    # Skip abstract "Additive", we have concrete material
                                    continue

                            entity = self._get_or_create_entity(input_entity, step_id)
                            if entity:
                                entity_names.add(input_entity)
                                entities.append(entity)

                # Extract OUTPUT entity (right side of ->)
                if len(parts) >= 2:
                    output_part = parts[-1].strip()
                    # Handle multiple outputs: "GroundCoffee + Filter" -> "GroundCoffee"
                    if '+' in output_part:
                        output_part = output_part.split('+')[0].strip()

                    if output_part and output_part not in entity_names:
                        entity = self._get_or_create_entity(output_part, step_id)
                        if entity:
                            entity_names.add(output_part)
                            entities.append(entity)

        # 4. Compound nouns from spaCy (fourth priority - only add if meaningful)
        # IMPORTANT: Also skip for control action verbs (e.g., "milk addition" in "stop milk addition")
        if not is_control_action:
            for compound in grammatical.compound_nouns:
                entity_name = self._clean_entity_name(compound)
                # Only add compound nouns that are not already covered and are meaningful
                if (entity_name and
                    entity_name not in entity_names and
                    len(entity_name) > 3 and  # Minimum length
                    entity_name not in line_text[:10] and  # Skip step IDs
                    not self._is_existing_actor(entity_name) and  # Don't create entities for actors
                    not self._is_compound_redundant(entity_name, entity_names)):
                    entity = self._get_or_create_entity(entity_name, step_id)
                    if entity:
                        entity_names.add(entity_name)
                        entities.append(entity)

        return entities
    
    def _is_existing_actor(self, entity_name: str) -> bool:
        """Check if entity name is already defined as an Actor"""
        return entity_name in self.uc_context.actors
    
    def _get_or_create_entity(self, entity_name: str, step_id: str) -> Optional[RAClass]:
        """Get existing entity or create new one - prevents duplicates"""
        if entity_name in self.global_entities:
            # Entity already exists, return reference to existing one
            return self.global_entities[entity_name]
        else:
            # Check if this is an implementation element that should be filtered out
            if self._is_implementation_element(entity_name):
                print(f"[WARNING] Skipped implementation element '{entity_name}' in {step_id}: Use functional activity instead")
                return None
            
            # Create new entity
            entity = RAClass(
                name=entity_name,
                ra_type=RAType.ENTITY,
                stereotype="<<entity>>",
                description=f"Domain entity: {entity_name.lower()}",
                step_id=step_id
            )
            # Store in global registry
            self.global_entities[entity_name] = entity
            return entity
    
    def _is_implementation_element(self, entity_name: str) -> bool:
        """Check if entity is an implementation element that should be filtered"""
        if not entity_name:
            return False
            
        entity_lower = entity_name.lower()
        
        # Check domain configuration for implementation elements
        if self.domain_name in self.verb_loader.domain_configs:
            domain_config = self.verb_loader.domain_configs[self.domain_name]
        else:
            return False
            
        if domain_config and "implementation_elements" in domain_config:
            impl_elements = domain_config["implementation_elements"].get("elements", {})
            
            # Direct match
            if entity_lower in impl_elements:
                return True
            
            # Check for compound matches (e.g., "radio clock")
            for impl_key in impl_elements.keys():
                if impl_key in entity_lower or entity_lower in impl_key:
                    return True
        
        return False
    
    def _get_functional_equivalent(self, implementation_element: str) -> Optional[str]:
        """Get functional equivalent for implementation element"""
        if not implementation_element:
            return None
            
        impl_lower = implementation_element.lower()
        
        # Get domain configuration
        if self.domain_name in self.verb_loader.domain_configs:
            domain_config = self.verb_loader.domain_configs[self.domain_name]
        else:
            return None
            
        if domain_config and "implementation_elements" in domain_config:
            impl_elements = domain_config["implementation_elements"].get("elements", {})
            
            # Look for transformation or functional suggestion
            for impl_key, impl_data in impl_elements.items():
                if impl_key in impl_lower or impl_lower in impl_key:
                    # Check for transformation (e.g., "Water -> HotWater")
                    if "transformation" in impl_data:
                        transformation = impl_data["transformation"]
                        if " -> " in transformation:
                            # Extract the output from transformation
                            output = transformation.split(" -> ")[1].strip()
                            return output
                    
                    # Check for controller realization (for "water heater" -> use "Water")
                    if "controller_realization" in impl_data:
                        # For heater -> return the material it operates on
                        if "heater" in impl_lower:
                            return "Water"  # Water heater -> Water
                        elif "grinder" in impl_lower:
                            return "CoffeeBeans"  # Grinder -> CoffeeBeans
                    
                    # Fallback: extract functional noun from implementation element
                    if "water" in impl_lower:
                        return "Water"
                    elif "coffee" in impl_lower:
                        return "Coffee"
                    elif "milk" in impl_lower:
                        return "Milk"
        
        return None
    
    def _is_compound_redundant(self, compound_name: str, existing_names: set) -> bool:
        """Check if compound noun is redundant with existing entities"""
        compound_lower = compound_name.lower()
        
        # Check if any existing entity already covers this compound
        for existing in existing_names:
            existing_lower = existing.lower()
            # If the compound is a subset of existing, it's redundant
            if (compound_lower in existing_lower or existing_lower in compound_lower):
                return True
        
        # Common redundant patterns
        redundant_patterns = ['system', 'time', 'control', 'data']
        if any(pattern in compound_lower for pattern in redundant_patterns):
            # Only allow if it's a meaningful technical term
            technical_patterns = ['missioncontrol', 'flighttelemetry', 'guidancecomputer', 'thrustcontrol']
            if not any(tech in compound_lower for tech in technical_patterns):
                return True
        
        return False
    
    def _generate_semantic_controller_name(self, step_id: str, line_text: str, grammatical: GrammaticalAnalysis) -> Tuple[str, str]:
        """Generate controller name based on SEMANTIC DOMAIN ANALYSIS of the complete sentence context"""
        
        # Step 1: Analyze the COMPLETE sentence for domain objects and context
        doc = self.nlp(line_text) if self.nlp else None
        if not doc:
            # Fallback if no spaCy
            return f"{grammatical.verb_lemma.capitalize()}Manager", f"Manages {grammatical.verb_lemma} function in {step_id}"
        
        # Step 2: Extract ALL semantic entities (not just direct object!)
        semantic_entities = []
        for chunk in doc.noun_chunks:
            entity_text = chunk.text.strip().lower()
            if entity_text not in ['system', 'the system']:
                semantic_entities.append(entity_text)
        
        # Step 3: Determine PRIMARY DOMAIN from the complete context
        primary_domain = self._determine_primary_domain_from_context(line_text, semantic_entities, grammatical)
        
        # Step 4: Generate controller name based on PRIMARY DOMAIN (NOT verb!)
        if primary_domain:
            controller_name = f"{primary_domain}Manager"
            description = f"Manages {primary_domain.lower()} operations in {step_id}"
        else:
            # Fallback: use cleaned direct object
            if grammatical.direct_object:
                obj_clean = self._clean_entity_name(grammatical.direct_object)
                controller_name = f"{obj_clean}Manager"
                description = f"Manages {obj_clean.lower()} operations in {step_id}"
            else:
                controller_name = f"{grammatical.verb_lemma.capitalize()}Manager"
                description = f"Manages {grammatical.verb_lemma} function in {step_id}"
        
        return controller_name, description
    
    def _determine_primary_domain_from_context(self, line_text: str, semantic_entities: List[str], grammatical: GrammaticalAnalysis) -> str:
        """Determine primary domain from CORE ACTION CONTEXT using GENERIC domain knowledge from JSON"""
        line_lower = line_text.lower()
        
        if self.domain_name not in self.verb_loader.domain_configs:
            return None
            
        domain_config = self.verb_loader.domain_configs[self.domain_name]
        
        # PRIORITY 0: Technical context mapping - HIGHEST PRIORITY for domain-specific keywords
        technical_mapping = domain_config.get('technical_context_mapping', {})
        if technical_mapping:
            contexts = technical_mapping.get('contexts', {})
            for context_name, keyword_lists in contexts.items():
                for keyword_list_name in keyword_lists:
                    keywords = technical_mapping.get(keyword_list_name, [])
                    for keyword in keywords:
                        if keyword.lower() in line_lower:
                            # Extract domain from context name (e.g., "Time Control" -> "Time")
                            domain_from_context = context_name.split()[0]  # Get first word
                            print(f"[DEBUG DOMAIN] Found technical context '{context_name}' via keyword '{keyword}' -> {domain_from_context}Manager")
                            return domain_from_context
        
        # PRIORITY 1: Domain JSON verb classification
        verb_classification = domain_config.get('verb_classification', {})
        
        # Check transformation verbs (highest priority - they define core business functions)
        transformation_verbs = verb_classification.get('transformation_verbs', {}).get('verbs', {})
        if grammatical.verb_lemma in transformation_verbs:
            transformation_info = transformation_verbs[grammatical.verb_lemma]
            return self._extract_domain_from_transformation(transformation_info)
        
        # Check transaction verbs 
        transaction_verbs = verb_classification.get('transaction_verbs', {}).get('verbs', {})
        if grammatical.verb_lemma in transaction_verbs:
            transaction_info = transaction_verbs[grammatical.verb_lemma]
            return self._extract_domain_from_verb_info(transaction_info, line_text)
        
        # Check function verbs
        function_verbs = verb_classification.get('function_verbs', {}).get('verbs', {})
        if grammatical.verb_lemma in function_verbs:
            function_info = function_verbs[grammatical.verb_lemma]
            return self._extract_domain_from_verb_info(function_info, line_text)
        
        # PRIORITY 2: Material/entity analysis from domain JSON
        operational_materials = domain_config.get('operational_materials_addressing', {}).get('material_types', {})
        for material_name, material_config in operational_materials.items():
            material_keywords = material_config.get('keywords', [])
            if not material_keywords:
                material_keywords = [material_name.replace('_', ' '), material_name.replace('_', '')]
            
            # Check if any material keywords appear in the sentence
            for keyword in material_keywords:
                if keyword.lower() in line_lower:
                    # Return the material as domain (e.g., coffee_beans -> Coffee)
                    return self._normalize_material_to_domain(material_name)
        
        # PRIORITY 3: Semantic entity analysis - focus on PRIMARY BUSINESS OBJECT
        primary_business_entity = self._identify_primary_business_entity(semantic_entities, line_text)
        if primary_business_entity:
            domain_candidate = self._normalize_material_to_domain(primary_business_entity)
            if domain_candidate:
                return domain_candidate
        
        # PRIORITY 4: Fallback semantic entity analysis
        for entity in semantic_entities:
            clean_entity = self._clean_entity_name(entity)
            if clean_entity and len(clean_entity) > 3:
                # Skip implementation details and focus on business objects
                if not self._is_implementation_detail(clean_entity):
                    domain_candidate = self._normalize_material_to_domain(clean_entity)
                    if domain_candidate:
                        return domain_candidate
        
        return None
    
    def _extract_domain_from_transformation(self, transformation_info: str) -> str:
        """Extract domain from transformation pattern (e.g., 'CoffeeBeans -> GroundCoffee' -> 'Coffee')"""
        if ' -> ' in transformation_info:
            source = transformation_info.split(' -> ')[0].strip()
            return self._normalize_material_to_domain(source)
        return None
    
    def _extract_domain_from_verb_info(self, verb_info: str, line_text: str) -> str:
        """Extract domain from verb description using GENERIC domain materials from JSON"""
        verb_lower = verb_info.lower()
        line_lower = line_text.lower()
        
        # Get domain indicators from domain JSON configuration
        if self.domain_name in self.verb_loader.domain_configs:
            domain_config = self.verb_loader.domain_configs[self.domain_name]
            operational_materials = domain_config.get('operational_materials_addressing', {}).get('material_types', {})
            
            # Extract all material names and their keywords from domain JSON
            domain_indicators = []
            for material_name, material_config in operational_materials.items():
                # Add the material name itself
                domain_indicators.append(material_name.replace('_', '').lower())
                # Add keywords if defined
                keywords = material_config.get('keywords', [])
                domain_indicators.extend([kw.lower() for kw in keywords])
            
            # Check for domain indicators in verb description and line text
            for indicator in domain_indicators:
                if indicator in verb_lower or indicator in line_lower:
                    return self._normalize_material_to_domain(indicator)
        
        return None
    
    def _normalize_material_to_domain(self, material_name: str) -> str:
        """GENERIC normalization of material names to domain controllers using domain JSON configuration"""
        if not material_name:
            return None
            
        material_lower = material_name.lower().replace('_', '').replace(' ', '')
        
        # Use domain JSON to find the correct domain controller
        if self.domain_name in self.verb_loader.domain_configs:
            domain_config = self.verb_loader.domain_configs[self.domain_name]
            operational_materials = domain_config.get('operational_materials_addressing', {}).get('material_types', {})
            
            # Check if the material_name matches any known materials in domain JSON
            for domain_material, material_config in operational_materials.items():
                domain_material_clean = domain_material.lower().replace('_', '')
                
                # Direct match
                if material_lower == domain_material_clean:
                    return self._extract_domain_controller_name(domain_material)
                
                # Check keywords
                keywords = material_config.get('keywords', [])
                for keyword in keywords:
                    keyword_clean = keyword.lower().replace('_', '').replace(' ', '')
                    if material_lower == keyword_clean or keyword_clean in material_lower:
                        return self._extract_domain_controller_name(domain_material)
                
                # Partial match (e.g., "coffee" matches "coffee_beans")
                if material_lower in domain_material_clean or domain_material_clean in material_lower:
                    return self._extract_domain_controller_name(domain_material)
        
        # Fallback: Extract meaningful controller name from material name
        return self._extract_domain_controller_name(material_name)
    
    def _extract_domain_controller_name(self, material_name: str) -> str:
        """COMPLETELY GENERIC extraction of controller domain name from material name"""
        if not material_name:
            return None
        
        # Split by underscore and take the PRIMARY domain word (first part)
        parts = material_name.lower().split('_')
        main_domain = parts[0]
        
        # GENERIC rule: Primary material is the controller domain
        # coffee_beans -> Coffee, water -> Water, milk -> Milk, rocket_fuel -> Rocket, etc.
        return main_domain.capitalize()
    
    def _identify_primary_business_entity(self, semantic_entities: List[str], line_text: str) -> str:
        """Identify the PRIMARY BUSINESS OBJECT from sentence context, ignoring implementation details"""
        line_lower = line_text.lower()
        
        # Get domain materials from JSON to identify what are business objects
        business_objects = []
        if self.domain_name in self.verb_loader.domain_configs:
            domain_config = self.verb_loader.domain_configs[self.domain_name]
            operational_materials = domain_config.get('operational_materials_addressing', {}).get('material_types', {})
            
            # Extract known business objects from domain JSON
            for material_name in operational_materials.keys():
                business_objects.append(material_name.replace('_', '').lower())
                # Also add individual words (e.g., "coffee" from "coffee_beans")
                parts = material_name.lower().split('_')
                business_objects.extend(parts)
        
        # Analyze sentence for primary business object pattern
        # Pattern 1: "verb + OBJECT + preposition + implementation" -> OBJECT is primary
        # Example: "retrieves cup from storage container" -> "cup" is primary business object
        
        # Look for business objects in semantic entities, prioritizing by business importance
        primary_candidates = []
        for entity in semantic_entities:
            entity_clean = entity.lower().replace('_', '').replace(' ', '')
            
            # Check if entity is a known business object
            if entity_clean in business_objects:
                primary_candidates.append((entity, self._get_business_priority(entity_clean, line_lower)))
        
        # Return the highest priority business object
        if primary_candidates:
            # Sort by priority (higher is better)
            primary_candidates.sort(key=lambda x: x[1], reverse=True)
            return primary_candidates[0][0]
        
        return None
    
    def _get_business_priority(self, entity: str, line_text: str) -> int:
        """GENERIC business priority score using domain JSON knowledge (higher = more important as primary business object)"""
        priority = 0
        
        # High priority: Direct object patterns (verb + entity)
        direct_object_verbs = ['retrieves', 'presents', 'delivers', 'produces', 'provides', 'creates']
        if any(f"{verb} {entity}" in line_text for verb in direct_object_verbs):
            priority += 10
        
        # Use domain JSON to determine business priority
        if self.domain_name in self.verb_loader.domain_configs:
            domain_config = self.verb_loader.domain_configs[self.domain_name]
            operational_materials = domain_config.get('operational_materials_addressing', {}).get('material_types', {})
            
            # High priority: Primary operational materials (end products)
            for material_name, material_config in operational_materials.items():
                material_clean = material_name.replace('_', '').lower()
                if entity == material_clean:
                    # Check if it's a primary product or customer-facing item
                    safety_class = material_config.get('safety_requirements', {}).get('safety_class', '')
                    if 'food_grade' in safety_class or 'customer_facing' in safety_class:
                        priority += 5
                    else:
                        priority += 3
                    break
            
            # Low priority: Implementation elements
            impl_elements = domain_config.get('implementation_elements', {}).get('elements', {})
            for impl_key in impl_elements.keys():
                impl_clean = impl_key.replace('_', '').lower()
                if entity == impl_clean:
                    priority -= 5
                    break
        
        return priority
    
    def _is_implementation_detail(self, entity_name: str) -> bool:
        """Check if entity is an implementation detail rather than a business object"""
        entity_lower = entity_name.lower()
        
        # Generic implementation detail patterns
        implementation_patterns = [
            'container', 'storage', 'heater', 'system', 'component', 
            'device', 'mechanism', 'apparatus', 'unit', 'module'
        ]
        
        # Check domain-specific implementation elements from JSON
        if self.domain_name in self.verb_loader.domain_configs:
            domain_config = self.verb_loader.domain_configs[self.domain_name]
            impl_elements = domain_config.get('implementation_elements', {}).get('elements', {})
            
            # Add domain-specific implementation elements
            for impl_key in impl_elements.keys():
                implementation_patterns.append(impl_key.replace('_', '').lower())
        
        # Check if entity matches implementation patterns
        return any(pattern in entity_lower for pattern in implementation_patterns)
    
    def _determine_step_context(self, step_id: str, line_text: str, line_type: LineType, grammatical: GrammaticalAnalysis) -> StepContext:
        """Determine the context of a UC step"""
        
        # Determine step type
        step_type = self._classify_step_type(step_id, line_text)
        
        # Determine phase
        phase = self._classify_step_phase(step_id, line_text, grammatical)
        
        # Determine business context
        business_context = self._determine_business_context(step_id, line_text, grammatical)
        
        # Determine technical context
        technical_context = self._determine_technical_context(step_id, line_text, grammatical)
        
        # Find involved actors
        actors_involved = self._find_actors_in_step(line_text)
        
        return StepContext(
            step_id=step_id,
            step_type=step_type,
            domain=self.domain_name,
            phase=phase,
            business_context=business_context,
            technical_context=technical_context,
            preconditions=self.uc_context.preconditions.copy(),
            actors_involved=actors_involved
        )
    
    def _classify_step_type(self, step_id: str, line_text: str) -> str:
        """Classify the type of step"""
        line_lower = line_text.lower()
        
        if "(trigger)" in line_lower:
            return "trigger"
        elif "end uc" in line_lower or "end use case" in line_lower:
            return "end"
        elif "continue" in line_lower:
            return "continue"
        elif step_id.startswith("A"):
            return "alternative"
        elif step_id.startswith("E"):
            return "extension"
        else:
            return "main"
    
    def _classify_step_phase(self, step_id: str, line_text: str, grammatical: GrammaticalAnalysis) -> str:
        """Classify which phase of the process this step represents using domain-specific classification"""
        return self.verb_loader.classify_step_phase(line_text, self.domain_name)
    
    def _determine_business_context(self, step_id: str, line_text: str, grammatical: GrammaticalAnalysis) -> str:
        """Determine what business function this step serves using domain-specific classification"""
        return self.verb_loader.classify_business_context(line_text, self.domain_name)
    
    def _determine_technical_context(self, step_id: str, line_text: str, grammatical: GrammaticalAnalysis) -> str:
        """Determine what technical function this step serves using LLM-enhanced classification"""
        # First try LLM-based analysis for better semantic understanding
        llm_context = self._llm_analyze_technical_context(step_id, line_text)
        if llm_context and llm_context != "Allgemeine Systemsteuerung":
            return llm_context
            
        # Fallback to keyword-based analysis
        line_lower = line_text.lower()
        
        # Load domain configuration for technical context mapping
        if self.domain_name in self.verb_loader.domain_configs:
            domain_config = self.verb_loader.domain_configs[self.domain_name]
            
            if 'technical_context_mapping' in domain_config:
                mapping = domain_config['technical_context_mapping']
                
                # Check each context category
                for context_name, keyword_lists in mapping['contexts'].items():
                    for keyword_list_name in keyword_lists:
                        keywords = mapping.get(keyword_list_name, [])
                        if any(keyword in line_lower for keyword in keywords):
                            return context_name
        
        # Default fallback - use domain JSON or generic English
        return "General System Control"
    
    def _llm_analyze_technical_context(self, step_id: str, line_text: str) -> str:
        """Use spaCy NLP model to analyze technical context semantically"""
        if not self.nlp:
            return "General System Control"
            
        try:
            doc = self.nlp(line_text)
            
            # Analyze trigger types and contexts semantically
            line_lower = line_text.lower()
            
            # Consider global UC context for better classification
            global_context = self.uc_goal.lower() if self.uc_goal else ""
            
            # Use LLM semantic analysis with domain knowledge for ALL steps
            contextual_description = self._generate_contextual_description(line_text, global_context)
            print(f"[LLM CONTEXT] {step_id}: {contextual_description}")

            # Legacy fallbacks using domain JSON contexts
            domain_config = self.verb_loader.domain_configs.get(self.domain_name, {})
            context_mapping = domain_config.get('technical_context_mapping', {}).get('contexts', {})
            
            # Use domain JSON mapping for fallback contexts
            if any(word in line_lower for word in ['lightning', 'storm', 'earthquake', 'flood', 'wind', 'weather']):
                return context_mapping.get("Weather Events", "Weather Events")
            if any(word in line_lower for word in ['user', 'operator', 'request', 'initiates', 'wants', 'asks']):
                return context_mapping.get("User Interface", "User Interface")
            if any(word in line_lower for word in ['time', 'clock', 'scheduled', 'timer', '7:00h', 'window opens']):
                return context_mapping.get("Time Control", "Time Control")
                
        except Exception as e:
            print(f"[DEBUG] LLM analysis failed for {step_id}: {e}")
            
        return "General System Control"
    
    def _determine_step_context_enhanced(self, step_id: str, line_text: str, line_type: LineType, grammatical: GrammaticalAnalysis, generated_contexts: List[GeneratedContext]) -> StepContext:
        """Enhanced step context determination using generative context manager"""
        
        # Start with traditional context
        traditional_context = self._determine_step_context(step_id, line_text, line_type, grammatical)
        
        # Enhance with generated contexts
        if generated_contexts:
            # Merge operational material contexts
            operational_materials = []
            safety_requirements = []
            hygiene_requirements = []
            special_controllers = []
            
            for context in generated_contexts:
                if context.context_type == ContextType.OPERATIONAL_MATERIAL:
                    operational_materials.append({
                        'name': context.context_name,
                        'safety_class': context.safety_class,
                        'hygiene_level': context.hygiene_level,
                        'addressing_format': context.addressing_format
                    })
                
                safety_requirements.extend(context.special_requirements)
                special_controllers.extend(context.controllers)
            
            # Create enhanced context
            enhanced_context = StepContext(
                step_id=step_id,
                context_type=traditional_context.context_type if traditional_context else "Enhanced Context",
                global_context=self._get_enhanced_global_context(generated_contexts),
                description=self._generate_enhanced_description(line_text, generated_contexts),
                operational_materials=operational_materials,
                safety_requirements=list(set(safety_requirements)),
                special_controllers=list(set(special_controllers))
            )
            
            return enhanced_context
        
        return traditional_context if traditional_context else StepContext(step_id=step_id)
    
    def _generate_ra_classes_for_line_enhanced(self, line_text: str, line_type: LineType, step_id: str, grammatical: GrammaticalAnalysis, step_context: StepContext, generated_contexts: List[GeneratedContext]) -> List[RAClass]:
        """Enhanced RA class generation using generative contexts"""
        
        # GUARD LOGIC: Alternative flows (A1, A2, A3) are guards - NO RA classes should be generated
        if step_id and re.match(r'^A\d+$', step_id):
            print(f"[GUARD] Skipping RA class generation for alternative flow guard: {step_id}")
            return []
        
        # Start with traditional RA classes
        traditional_ra_classes = self._generate_ra_classes_for_line(line_text, line_type, step_id, grammatical, step_context)
        
        # Add context-based RA classes
        enhanced_ra_classes = list(traditional_ra_classes)
        
        for context in generated_contexts:
            # Generate operational material entities
            if context.context_type == ContextType.OPERATIONAL_MATERIAL:
                material_entity = RAClass(
                    name=context.context_name,
                    ra_type=RAType.ENTITY,
                    stereotype="<<entity>>",
                    description=f"Operational material: {context.context_name}",
                    step_id=step_id,
                    element_type="operational_material"
                )
                enhanced_ra_classes.append(material_entity)
                
                # Add safety/hygiene controllers if specified
                for controller_name in context.controllers:
                    controller = RAClass(
                        name=controller_name,
                        ra_type=RAType.CONTROLLER,
                        stereotype="<<controller>>",
                        description=f"Safety/hygiene controller for {context.context_name}",
                        step_id=step_id,
                        element_type="safety_hygiene"
                    )
                    enhanced_ra_classes.append(controller)
            
            # Generate safety context controllers
            elif context.context_type == ContextType.SAFETY_CONTEXT:
                safety_controller = RAClass(
                    name=f"{context.context_name}Controller",
                    ra_type=RAType.CONTROLLER,
                    stereotype="<<controller>>",
                    description=f"Safety controller: {context.context_name}",
                    step_id=step_id,
                    element_type="safety"
                )
                enhanced_ra_classes.append(safety_controller)
            
            # Generate hygiene context controllers
            elif context.context_type == ContextType.HYGIENE_CONTEXT:
                hygiene_controller = RAClass(
                    name=f"{context.context_name}Controller",
                    ra_type=RAType.CONTROLLER,
                    stereotype="<<controller>>",
                    description=f"Hygiene controller: {context.context_name}",
                    step_id=step_id,
                    element_type="hygiene"
                )
                enhanced_ra_classes.append(hygiene_controller)
        
        return enhanced_ra_classes
    
    def _get_enhanced_global_context(self, generated_contexts: List[GeneratedContext]) -> str:
        """Generate enhanced global context from generated contexts"""
        if not generated_contexts:
            return "General System Context"
        
        # Prioritize operational materials and safety contexts
        operational_contexts = [c for c in generated_contexts if c.context_type == ContextType.OPERATIONAL_MATERIAL]
        safety_contexts = [c for c in generated_contexts if c.context_type == ContextType.SAFETY_CONTEXT]
        technical_contexts = [c for c in generated_contexts if c.context_type == ContextType.TECHNICAL_CONTEXT]
        
        if operational_contexts:
            return f"Operational Materials Context: {', '.join([c.context_name for c in operational_contexts])}"
        elif safety_contexts:
            return f"Safety Context: {', '.join([c.context_name for c in safety_contexts])}"
        elif technical_contexts:
            return f"Technical Context: {', '.join([c.context_name for c in technical_contexts])}"
        else:
            return f"Generated Context: {generated_contexts[0].context_name}"
    
    def _generate_enhanced_description(self, line_text: str, generated_contexts: List[GeneratedContext]) -> str:
        """Generate enhanced description using generated contexts"""
        if not generated_contexts:
            return line_text
        
        # Create context-aware description
        context_info = []
        for context in generated_contexts:
            if context.context_type == ContextType.OPERATIONAL_MATERIAL:
                context_info.append(f"handles {context.context_name}")
                if context.safety_class:
                    context_info.append(f"with {context.safety_class} safety requirements")
                if context.hygiene_level:
                    context_info.append(f"under {context.hygiene_level} hygiene standards")
        
        if context_info:
            return f"{line_text} - Context: {', '.join(context_info)}"
        else:
            return line_text
    
    def _generate_contextual_description(self, line_text: str, global_context: str) -> str:
        """
        Generate contextual description (GENERIC - no hard-coded materials!)

        Simply returns the original text since semantic analysis with hard-coded
        material names is not domain-independent.
        """
        return line_text
    
    def _extract_main_verb(self, doc):
        """Extract the main action verb from spaCy doc"""
        # IMPORTANT: First try spaCy detection (actual verbs have priority!)
        # Then fallback to semantic detection only if no clear verb found

        # Priority 1: spaCy detection - look for actual VERB tokens
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp"]:
                # Ignore tokens that look like step IDs (B2a, A1, etc.)
                if not re.match(r'^[A-Z]\d+[a-z]?$', token.text):
                    return token.lemma_

        # Priority 2: Semantic verb detection from domain JSON (only if no verb found)
        # This catches cases where verb is implied (e.g., "Milcherhitzung" → "heat")
        domain_config = self.verb_loader.domain_configs.get(self.domain_name, {})
        semantic_verbs = domain_config.get('semantic_verb_detection', {})

        text_lower = doc.text.lower()
        for verb, patterns in semantic_verbs.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return verb
        
        # Last fallback: look for any verb that's not a step ID
        for token in doc:
            if token.pos_ == "VERB":
                if not re.match(r'^[A-Z]\d+[a-z]?$', token.text):
                    return token.lemma_
            
        return None
    
    def _llm_analyze_required_betriebsmittel(self, main_verb: str, entities: list, available_betriebsmittel: dict) -> list:
        """
        GENERIC: Determine required Betriebsstoffe using transformation definitions from domain config

        For transformation verbs: Extract inputs from domain config (e.g., "Input1 + Input2 -> Output")
        For other verbs: Use entities mentioned in text
        """
        if not main_verb:
            return []

        try:
            required_materials = []

            # Check if this verb has a transformation definition in domain config
            transformation_info = self.verb_loader.get_transformation_for_verb(main_verb, self.domain_name)

            if transformation_info and '->' in transformation_info:
                # GENERIC: Parse transformation to extract input materials
                # Format: "Input1 + Input2 + Input3 -> Output"
                parts = transformation_info.split('->')
                if len(parts) >= 1:
                    input_part = parts[0].strip()
                    # Extract all inputs
                    input_materials = [inp.strip().lower() for inp in input_part.split('+')]

                    # Add inputs that are available as Betriebsmittel
                    for material in input_materials:
                        # Check if material exists in available Betriebsmittel (case-insensitive)
                        for available_mat in available_betriebsmittel.keys():
                            if material == available_mat.lower() or material.replace(' ', '_') == available_mat.lower():
                                required_materials.append(available_mat)
                                break
            else:
                # For non-transformation verbs: Use entities from text as required materials
                # Match entities with available Betriebsmittel
                for entity in entities:
                    entity_lower = entity.lower()
                    for available_mat in available_betriebsmittel.keys():
                        if entity_lower in available_mat.lower() or available_mat.lower() in entity_lower:
                            if available_mat not in required_materials:
                                required_materials.append(available_mat)

            return required_materials

        except Exception as e:
            print(f"[DEBUG] Betriebsmittel analysis failed: {e}")
            return []
    
    def _llm_generate_semantic_context(self, main_verb: str, entities: list, purpose: str, required_materials: list) -> str:
        """
        GENERIC: Generate semantic context description from verb + materials (NO hard-coding!)

        Uses verb type classification and primary material to generate context string.
        """
        if not main_verb:
            return "General System Control"

        try:
            # Get verb classification to determine context type
            verb_type = self._classify_verb(main_verb)

            # Determine primary material from required_materials or entities
            primary_material = None
            if required_materials:
                # Use first required material as primary
                primary_material = required_materials[0].replace('_', ' ').title()
            elif entities:
                # Use first entity as primary
                primary_material = entities[0].replace('_', ' ').title()

            # Generate GENERIC context based on verb type and material
            if primary_material:
                if verb_type and 'TRANSFORMATION' in str(verb_type):
                    return f"{primary_material} {main_verb} for {purpose}"
                elif verb_type and 'TRANSACTION' in str(verb_type):
                    return f"{primary_material} {main_verb} for {purpose}"
                else:
                    return f"{primary_material} {main_verb} for {purpose}"
            else:
                # No material identified - use generic description
                return f"{main_verb.capitalize()} operation for {purpose}"

        except Exception as e:
            print(f"[DEBUG] Semantic context generation failed: {e}")
            return "General System Control"
    
    def _extract_uc_global_context(self):
        """Extract goal and title from the entire UC text using LLM analysis"""
        if not self.nlp or not self.all_lines:
            return
            
        try:
            # Join all lines to get full UC text
            full_text = '\n'.join(self.all_lines)
            
            # Extract UC title/name from filename or first line
            self.uc_title = self._extract_uc_title(full_text)
            
            # Extract goal using LLM-based semantic analysis
            self.uc_goal = self._extract_uc_goal_llm(full_text)
            
            print(f"[UC CONTEXT] Title: {self.uc_title}")
            print(f"[UC CONTEXT] Goal: {self.uc_goal}")
            
        except Exception as e:
            print(f"[DEBUG] UC context extraction failed: {e}")
            self.uc_title = "Unknown UC"
            self.uc_goal = "UC goal not extracted"
    
    def _extract_uc_title(self, full_text: str) -> str:
        """Extract UC title from text"""
        lines = full_text.split('\n')
        
        # Look for title in first few lines
        for line in lines[:5]:
            line = line.strip()
            if line and not line.startswith('-') and not line.startswith('Actors:'):
                # Remove common prefixes
                title = line.replace('Use Case:', '').replace('UC:', '').strip()
                if title:
                    return title
        
        return "Unknown Use Case"
    
    def _extract_uc_goal_llm(self, full_text: str) -> str:
        """Extract UC goal - GENERIC approach using Goal: line from UC text"""
        try:
            # PRIORITY 1: Extract from explicit "Goal:" line (most reliable)
            lines = full_text.split('\n')
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.lower().startswith('goal:'):
                    # Extract goal text after "Goal:"
                    goal_text = line_stripped[5:].strip()  # Remove "Goal:"
                    if goal_text:
                        return goal_text

            # PRIORITY 2: Detect domain and create generic goal
            # GENERIC: Check ALL loaded domain configurations for keywords
            for domain_key, domain_config in self.verb_loader.domain_configs.items():
                if domain_key == 'common_domain':
                    continue  # Skip common domain

                domain_keywords = domain_config.get('keywords', [])
                domain_name = domain_config.get('domain_name', domain_key)

                # Check if domain keywords match UC text
                if any(keyword in full_text.lower() for keyword in domain_keywords):
                    return f"Perform {domain_name} operation"

            # PRIORITY 3: Fallback - extract from main flow
            return self._extract_goal_from_main_flow(lines)

        except Exception as e:
            print(f"[DEBUG] Goal extraction failed: {e}")
            return "UC goal not specified"
    
    def _extract_goal_from_main_flow(self, lines: List[str]) -> str:
        """Extract goal from main flow steps as fallback (GENERIC!)"""
        main_flow_materials = []

        # Get all materials from domain JSON (GENERIC!)
        domain_materials = self._load_domain_materials()
        all_material_variants = []
        for material_variants in domain_materials.values():
            all_material_variants.extend([v.lower() for v in material_variants])

        for line in lines:
            line_lower = line.strip().lower()
            # Look for B-steps that indicate the main purpose
            if line_lower.startswith('b1') or line_lower.startswith('b2') or line_lower.startswith('b3'):
                # Find any material mentioned in the line
                for material in all_material_variants:
                    if material in line_lower and material not in main_flow_materials:
                        main_flow_materials.append(material)
                        break  # Only one material per line

        if main_flow_materials:
            return f"System für {', '.join(main_flow_materials[:2])}"

        return "Systemoperation durchführen"
    
    def _find_actors_in_step(self, line_text: str) -> List[str]:
        """Find which actors are involved in this step"""
        involved_actors = []
        line_lower = line_text.lower()
        
        for actor in self.uc_context.actors:
            if actor.lower() in line_lower:
                involved_actors.append(actor)
        
        # System is NOT an actor - only external entities are actors
        # Do not add System as actor
        
        return involved_actors
    
    def _generate_boundaries_for_step(self, step_id: str, grammatical: GrammaticalAnalysis, line_text: str) -> List[RAClass]:
        """Generate Boundaries based on specific functional analysis using common domain JSON"""
        boundaries = []
        line_lower = line_text.lower()
        
        # Get boundary patterns from domain JSON (beverage_preparation has patterns)
        domain_config = self.verb_loader.domain_configs.get(self.domain_name, {})
        boundary_patterns = domain_config.get('boundary_patterns', {})
        boundary_types = boundary_patterns.get('boundary_types', {})
        
        # PRIORITY 1: Analyze trigger patterns for specific boundary types
        if "(trigger)" in line_lower:
            # Determine the type of trigger based on content analysis
            trigger_boundary = self._determine_trigger_boundary_type(line_text, boundary_types)
            if trigger_boundary:
                boundaries.append(trigger_boundary)
            else:
                # Fallback: determine based on context
                trigger_boundary = self._generate_context_based_boundary(step_id, line_text, boundary_types)
                if trigger_boundary:
                    boundaries.append(trigger_boundary)
        
        # PRIORITY 2: Check for human interaction patterns (HMIManager cases)
        elif self._is_human_interaction(line_text):
            # Generate specific functional boundary based on interaction purpose
            boundary = self._generate_functional_boundary_for_interaction(line_text, step_id, grammatical)
            if boundary:
                boundaries.append(boundary)
        
        # PRIORITY 3: Legacy transaction verb logic (keep for compatibility)
        elif grammatical.verb_type == VerbType.TRANSACTION_VERB:
            # Get transaction verb patterns from common domain JSON
            transaction_input_verbs = boundary_patterns.get('transaction_input_verbs', [])
            transaction_output_verbs = boundary_patterns.get('transaction_output_verbs', [])
            
            for actor in self.uc_context.actors:
                if actor.lower() in line_lower:
                    # Generate boundary for Actor + Transaction Verb
                    actor_clean = ''.join(word.capitalize() for word in actor.split())
                    
                    if grammatical.verb_lemma in transaction_output_verbs:
                        boundary_name = f"{actor_clean}OutputBoundary"
                        description = f"Boundary for {actor} to send data/commands to system"
                    elif grammatical.verb_lemma in transaction_input_verbs:
                        boundary_name = f"{actor_clean}InputBoundary"
                        description = f"Boundary for {actor} to receive data/requests from system"
                    else:
                        boundary_name = f"{actor_clean}CommunicationBoundary"
                        description = f"Boundary for {actor} communication with system"
                    
                    boundaries.append(RAClass(
                        name=boundary_name,
                        ra_type=RAType.BOUNDARY,
                        stereotype="<<boundary>>",
                        description=description,
                        step_id=step_id
                    ))
                    break  # Only one boundary per step
        
        return boundaries
    
    def _determine_trigger_boundary_type(self, line_text: str, boundary_types: dict) -> Optional[RAClass]:
        """Determine specific boundary type based on trigger content using generic NLP analysis"""
        line_lower = line_text.lower()
        
        # Get boundary patterns from domain JSON
        domain_config = self.verb_loader.domain_configs.get(self.domain_name, {})
        boundary_patterns = domain_config.get('boundary_patterns', {})
        
        time_control_verbs = boundary_patterns.get('time_control_verbs', [])
        supply_control_verbs = boundary_patterns.get('supply_control_verbs', [])
        
        # Analyze trigger content generically using domain JSON patterns
        for verb in time_control_verbs:
            if verb in line_lower:
                # Time/scheduling trigger - not user input!
                return RAClass(
                    name="TimingBoundary",
                    ra_type=RAType.BOUNDARY,
                    stereotype="<<boundary>>",
                    description="Boundary for time-based system scheduling and triggers",
                    step_id=""
                )
        
        # Check for user interactions using actor analysis
        for actor in self.uc_context.actors:
            if actor.lower() in line_lower:
                # This is a user trigger - generate specific functional boundary
                return self._generate_functional_boundary_for_interaction(line_text, "")  # Use empty step_id for triggers
        
        return None
    
    def _generate_context_based_boundary(self, step_id: str, line_text: str, boundary_types: dict) -> Optional[RAClass]:
        """Generate boundary based on step context when trigger type is unclear"""
        line_lower = line_text.lower()
        
        # Use NLP to identify key domain entities and determine boundary type
        doc = self.nlp(line_text) if self.nlp else None
        if doc:
            for ent in doc.ents:
                if ent.label_ == "TIME":
                    # Time entity detected - scheduling boundary
                    return RAClass(
                        name="TimingBoundary", 
                        ra_type=RAType.BOUNDARY,
                        stereotype="<<boundary>>",
                        description="Boundary for time-based system triggers",
                        step_id=step_id
                    )
        
        # Check for specific material/supply triggers
        if any(word in line_lower for word in ["available", "supply", "level"]):
            return RAClass(
                name="SupplyMonitoringBoundary",
                ra_type=RAType.BOUNDARY, 
                stereotype="<<boundary>>",
                description="Boundary for supply monitoring and alerts",
                step_id=step_id
            )
        
        # Default fallback - external system boundary
        return RAClass(
            name="ExternalSystemBoundary",
            ra_type=RAType.BOUNDARY,
            stereotype="<<boundary>>",
            description="Boundary for external system trigger input",
            step_id=step_id
        )
    
    def _generate_functional_boundary_for_interaction(self, line_text: str, step_id: str, grammatical: Optional['GrammaticalAnalysis'] = None) -> Optional[RAClass]:
        """Generate specific functional boundary names based on interaction purpose - Boundaries are ALWAYS specific!"""
        line_lower = line_text.lower()
        
        # Get boundary patterns from domain JSON
        domain_config = self.verb_loader.domain_configs.get(self.domain_name, {})
        boundary_patterns = domain_config.get('boundary_patterns', {})
        input_verbs = boundary_patterns.get('input_verbs', [])
        output_verbs = boundary_patterns.get('output_verbs', [])
        
        # Analyze the functional purpose of the interaction
        if any(verb in line_lower for verb in ['output', 'display', 'present', 'show']):
            if 'message' in line_lower:
                return RAClass(
                    name="MessageDisplayBoundary",
                    ra_type=RAType.BOUNDARY,
                    stereotype="<<boundary>>",
                    description="Boundary for displaying messages to user",
                    step_id=step_id
                )
            elif 'error' in line_lower:
                return RAClass(
                    name="ErrorDisplayBoundary", 
                    ra_type=RAType.BOUNDARY,
                    stereotype="<<boundary>>",
                    description="Boundary for displaying error information to user",
                    step_id=step_id
                )
            elif any(word in line_lower for word in ['cup', 'product', 'result']):
                return RAClass(
                    name="ProductDeliveryBoundary",
                    ra_type=RAType.BOUNDARY,
                    stereotype="<<boundary>>", 
                    description="Boundary for delivering products to user",
                    step_id=step_id
                )
        elif any(verb in line_lower for verb in ['want', 'request', 'ask', 'input']):
            # BOUNDARIES ARE ALWAYS SPECIFIC!
            # Extract Direct Object from grammatical analysis or fallback to NLP
            direct_object = None

            if grammatical and grammatical.direct_object:
                direct_object = grammatical.direct_object.strip()
            else:
                # Fallback: Extract noun from line_text using NLP
                doc = self.nlp(line_text)
                for token in doc:
                    if token.pos_ == 'NOUN' and token.dep_ in ['dobj', 'obj', 'pobj']:
                        direct_object = token.text
                        break

            # Create specific boundary based on Direct Object
            if direct_object:
                # Capitalize and clean the direct object
                clean_object = self._clean_entity_name(direct_object)
                boundary_name = f"{clean_object}RequestBoundary"
                description = f"Boundary for user {direct_object} requests"

                return RAClass(
                    name=boundary_name,
                    ra_type=RAType.BOUNDARY,
                    stereotype="<<boundary>>",
                    description=description,
                    step_id=step_id
                )
            else:
                # Last resort: Use generic but warn
                print(f"[WARNING] No Direct Object found for request boundary in: {line_text}")
                return RAClass(
                    name="RequestBoundary",
                    ra_type=RAType.BOUNDARY,
                    stereotype="<<boundary>>",
                    description="Boundary for user request (no specific object identified)",
                    step_id=step_id
                )
        
        # Default functional boundary for user interactions
        return RAClass(
            name="HMIBoundary",
            ra_type=RAType.BOUNDARY,
            stereotype="<<boundary>>",
            description="Boundary for human-machine interface interactions",
            step_id=step_id
        )
    
    def _clean_entity_name(self, text: str) -> str:
        """Clean and normalize entity names"""
        if not text:
            return ""
        
        # Remove articles and common words
        words = text.split()
        cleaned_words = []
        
        skip_words = {"the", "a", "an", "this", "that", "with", "by", "for", "to", "from", "at", "in", "on"}
        
        # NEVER allow "system" as entity - it's the UC executor
        system_words = {"system", "systems"}
        
        for word in words:
            if (word.lower() not in skip_words and 
                word.lower() not in system_words and 
                len(word) > 1):
                cleaned_words.append(word.capitalize())
        
        if not cleaned_words:
            return ""
        
        # Join and create PascalCase
        result = "".join(cleaned_words)
        
        # Double-check: never return "System" as entity name
        if result.lower() == "system":
            return ""
            
        return result
    
    def _generate_data_flows_for_line(self, step_id: str, grammatical: GrammaticalAnalysis, ra_classes: List[RAClass]) -> List[DataFlow]:
        """
        Generate data flows based on function semantics.

        IMPORTANT RULE: Each function produces EXACTLY ONE output entity (PROVIDE).
        All other entities are inputs (USE).

        For transformation verbs: Output comes from domain config (e.g., "CoffeeBeans -> GroundCoffee")

        TRIGGERS: Triggers have NO data flows - they are Actor -> Boundary signals only!
        """
        data_flows = []

        if not step_id or not grammatical.main_verb:
            return data_flows

        # TRIGGERS have NO data flows!
        # Check if any RA class is a trigger (only has Boundary, no Controller)
        has_controller = any(ra.ra_type == RAType.CONTROLLER for ra in ra_classes)
        has_boundary = any(ra.ra_type == RAType.BOUNDARY for ra in ra_classes)

        if has_boundary and not has_controller:
            # This is a TRIGGER (only Boundary, no Controller) - NO data flows!
            return data_flows

        # Find controller for this step
        controller_name = None
        for ra_class in ra_classes:
            if ra_class.ra_type == RAType.CONTROLLER:
                controller_name = ra_class.name
                break

        if not controller_name:
            return data_flows

        # Check if this is a control action verb (stop, switch, pause, etc.)
        # These verbs represent internal state changes and do NOT produce data flows
        is_control_action = self._is_control_action_verb(grammatical.verb_lemma)
        if is_control_action:
            return data_flows  # No data flows for control actions

        # Check if this is a transformation verb
        is_transformation = hasattr(grammatical, 'verb_type') and grammatical.verb_type == VerbType.TRANSFORMATION_VERB
        transformation_output = None
        transformation_inputs = []

        if is_transformation:
            # Get transformation info from domain config
            transformation_info = self.verb_loader.get_transformation_for_verb(grammatical.verb_lemma, self.domain_name)
            if transformation_info and '->' in transformation_info:
                parts = transformation_info.split('->')

                # Extract INPUT entities
                if len(parts) >= 1:
                    input_part = parts[0].strip()
                    transformation_inputs = [inp.strip() for inp in input_part.split('+')]

                # Extract OUTPUT entity
                if len(parts) >= 2:
                    output_part = parts[-1].strip()
                    # Handle multiple outputs: "GroundCoffee + Filter" -> "GroundCoffee"
                    if '+' in output_part:
                        output_part = output_part.split('+')[0].strip()
                    transformation_output = output_part

        # STEP 1: Process all prepositional objects
        # For transformation verbs: ALL are USE (parameters/targets)
        # For other verbs: Use preposition semantics (but prefer USE)
        prepositional_entities = set()  # Track prepositional entities
        for prep, obj in grammatical.prepositional_objects:
            entity_name = self._clean_entity_name(obj)
            if entity_name and not self._is_existing_actor(entity_name):
                # FILTER: Skip base materials if transformed version exists in transformation inputs
                # Example: Skip "Water" if "HotWater" is in transformation inputs
                # Example: Skip "CoffeeBeans" if "GroundCoffee" is in transformation inputs
                if is_transformation and transformation_inputs:
                    # Check if a transformed version of this material exists
                    transformed_version = f"{entity_name.capitalize()}"
                    has_transformed_version = False
                    for trans_input in transformation_inputs:
                        # Check for common transformations: Water->HotWater, CoffeeBeans->GroundCoffee
                        if entity_name.lower() in trans_input.lower() and entity_name.lower() != trans_input.lower():
                            has_transformed_version = True
                            break

                    if has_transformed_version:
                        # Skip this base material, use transformed version instead
                        continue

                prepositional_entities.add(entity_name)

                # For transformation verbs: Everything except output is USE
                if is_transformation:
                    flow_type = "use"
                    if prep in ['with', 'using']:
                        description = f"{controller_name} uses {entity_name} as parameter"
                    elif prep in ['into', 'onto', 'to']:
                        description = f"{controller_name} uses {entity_name} as target"
                    elif prep in ['of']:
                        description = f"{controller_name} uses {entity_name} as component"
                    else:
                        description = f"{controller_name} uses {entity_name} as input"
                else:
                    # Non-transformation: Use preposition semantics
                    if prep in ['with', 'from', 'using', 'via', 'through', 'of']:
                        flow_type = "use"
                        description = f"{controller_name} uses {entity_name} as input"
                    elif prep in ['to', 'for', 'into', 'onto']:
                        flow_type = "provide"
                        description = f"{controller_name} provides output to {entity_name}"
                    else:
                        flow_type = "use"
                        description = f"{controller_name} interacts with {entity_name}"

                data_flows.append(DataFlow(
                    step_id=step_id,
                    controller=controller_name,
                    entity=entity_name,
                    flow_type=flow_type,
                    preposition=prep,
                    description=description
                ))

        # STEP 2: Process direct object
        # For transformation verbs: SKIP direct object if it matches the output
        # Output comes from domain config
        if grammatical.direct_object:
            entity_name = self._clean_entity_name(grammatical.direct_object)
            if (entity_name and
                not self._is_existing_actor(entity_name) and
                not self._is_implementation_element(entity_name)):

                if is_transformation:
                    # IMPORTANT: Skip direct object if it's the same as output
                    # Example: "brewing coffee" - "coffee" is output, not input
                    if entity_name.lower() != transformation_output.lower() if transformation_output else True:
                        # Check if it's one of the known inputs from domain config
                        is_known_input = any(entity_name.lower() == inp.lower() for inp in transformation_inputs)

                        # Special case: "add" verbs - the direct object is ALWAYS the material being added
                        # Example: "add milk" -> "milk" is input, even if not in transformation_inputs
                        # Example: "add sugar" -> "sugar" is input
                        is_add_verb = grammatical.verb_lemma.lower() in ['add', 'mix', 'blend', 'combine', 'incorporate']

                        if is_known_input or not transformation_inputs or is_add_verb:
                            # Add as input if:
                            # 1. It's in the transformation inputs OR
                            # 2. No specific inputs defined OR
                            # 3. It's an "add" verb (direct object is what's being added)
                            data_flows.append(DataFlow(
                                step_id=step_id,
                                controller=controller_name,
                                entity=entity_name,
                                flow_type="use",
                                description=f"{controller_name} uses {entity_name} as input"
                            ))
                else:
                    # Check if this is a transaction verb (e.g., present, output, send, deliver)
                    is_transaction = hasattr(grammatical, 'verb_type') and grammatical.verb_type == VerbType.TRANSACTION_VERB

                    if is_transaction:
                        # Transaction verbs: direct object is INPUT (what is being transferred)
                        # The boundary/target comes from prepositional object (to user, to system, etc.)
                        data_flows.append(DataFlow(
                            step_id=step_id,
                            controller=controller_name,
                            entity=entity_name,
                            flow_type="use",
                            description=f"{controller_name} uses {entity_name} for transaction"
                        ))
                    else:
                        # Function verbs: direct object is typically output
                        data_flows.append(DataFlow(
                            step_id=step_id,
                            controller=controller_name,
                            entity=entity_name,
                            flow_type="provide",
                            description=f"{controller_name} provides {entity_name} as result"
                        ))

        # STEP 3: Add transformation INPUT entities from domain config (USE)
        # These are the ACTUAL inputs for the transformation
        # IMPORTANT: Only add if not already in prepositional_entities (avoid duplicates)
        # IMPORTANT: Skip abstract categories (like "Additive") if we have concrete materials
        if is_transformation and transformation_inputs:
            # Check if we already have concrete material entities from text
            # Get all entities already added from text
            existing_entities_from_text = set()
            for flow in data_flows:
                existing_entities_from_text.add(flow.entity.lower())

            for input_entity in transformation_inputs:
                # Skip if already added from prepositional objects
                if input_entity in prepositional_entities:
                    continue

                # Skip abstract categories if we have concrete materials
                # Example: Skip "Additive" if we already have "Milk" or "Sugar"
                if input_entity.lower() == "additive":
                    # Check if we have a concrete material (GENERIC from domain JSON)
                    domain_materials = self._load_domain_materials()
                    concrete_material_names = []
                    for material_variants in domain_materials.values():
                        concrete_material_names.extend([v.lower() for v in material_variants])

                    has_concrete_material = any(
                        material in existing_entities_from_text
                        for material in concrete_material_names
                    )
                    if has_concrete_material:
                        continue  # Skip abstract "Additive", use concrete material instead

                # Create USE data flow for each input
                data_flows.append(DataFlow(
                    step_id=step_id,
                    controller=controller_name,
                    entity=input_entity,
                    flow_type="use",
                    description=f"{controller_name} uses {input_entity} as input"
                ))

        # STEP 4: Add transformation OUTPUT entity (PROVIDE)
        # This is the ONLY PROVIDE entity for transformation verbs!
        if is_transformation and transformation_output:
            data_flows.append(DataFlow(
                step_id=step_id,
                controller=controller_name,
                entity=transformation_output,
                flow_type="provide",
                description=f"{controller_name} produces {transformation_output} as result"
            ))

        # STEP 5: Add ALL remaining entities from ra_classes as USE
        # RULE: ALL entities in the step (EXCEPT output) must have USE data flows!
        # This catches entities extracted by NLP that are not in transformation_inputs
        # Example: "UserDefinedAmountOfCoffee" in B2c
        existing_entity_names = set(df.entity for df in data_flows)

        for ra_class in ra_classes:
            if ra_class.ra_type == RAType.ENTITY:
                entity_name = ra_class.name

                # Skip if already has a data flow
                if entity_name in existing_entity_names:
                    continue

                # Skip if this is the output entity
                if is_transformation and entity_name == transformation_output:
                    continue

                # Add USE data flow for this remaining entity
                data_flows.append(DataFlow(
                    step_id=step_id,
                    controller=controller_name,
                    entity=entity_name,
                    flow_type="use",
                    description=f"{controller_name} uses {entity_name} as input"
                ))

        return data_flows
    
    def _generate_control_flows(self):
        """Generate control flows based on the correct specification logic using existing regex functions"""
        print("[CONTROL] Generating control flows using correct logic with regex functions...")
        
        # Get all step analyses with controllers in order
        step_analyses = [la for la in self.line_analyses if la.step_id and self._get_controller_for_step(la)]
        
        if len(step_analyses) <= 1:
            return
        
        # Parse steps using existing regex functions
        parsed_steps = []
        for step in step_analyses:
            step_id = step.step_id
            controller = self._get_controller_for_step(step)
            
            # Use regex pattern to determine step type directly
            is_parallel = self._has_parallel_pattern(step_id)
            step_number = self._extract_parallel_step_number(step_id) if is_parallel else None
            step_type = 'parallel' if is_parallel else 'serial'
                
            parsed_steps.append({
                'step_id': step_id,
                'type': step_type,
                'step_number': step_number,
                'controller': controller,
                'line_analysis': step
            })
        
        print(f"[CONTROL] Parsed {len(parsed_steps)} steps for control flow analysis")
        for ps in parsed_steps:
            print(f"  - {ps['step_id']}: {ps['type']} (num={ps['step_number']}) -> {ps['controller']}")
        
        # Apply correct control flow logic
        for i in range(len(parsed_steps) - 1):
            current = parsed_steps[i]
            next_step = parsed_steps[i + 1]

            current_type = current['type']
            next_type = next_step['type']
            current_num = current['step_number']
            next_num = next_step['step_number']

            print(f"[CONTROL] Analyzing: {current['step_id']} ({current_type}) -> {next_step['step_id']} ({next_type})")

            # IMPORTANT: Check if current and next_step are in the same flow scope
            # Do NOT create flows between different flow scopes:
            # - Main flow (B1-B6)
            # - Alternative flows (A1.x, A2.x, A3.x, ...)
            # - Extension flows (E1.x, E2.x, ...)
            if not self._are_in_same_flow_scope(current['step_id'], next_step['step_id']):
                print(f"[CONTROL] SKIP: {current['step_id']} and {next_step['step_id']} are in different flow scopes")
                continue

            # Rule 1: serial -> serial
            if current_type == 'serial' and next_type == 'serial':
                print(f"[CONTROL] Rule 1: serial -> serial")
                self._create_control_flow(current, next_step, "sequential", "Rule 1 - Serial to Serial")
                
            # Rule 2: serial -> parallel
            elif current_type == 'serial' and next_type == 'parallel':
                print(f"[CONTROL] Rule 2: serial -> parallel (add P{next_num}_START)")
                px_start = f"P{next_num}_START"
                px_end = f"P{next_num}_END"
                step_range = self._get_parallel_step_range(next_step['step_id'])
                
                # Add parallel distribution node
                self._add_parallel_node(current['line_analysis'], px_start, 'distribution', step_range)
                self._add_parallel_node(current['line_analysis'], px_end, 'merge', step_range)                
                # Connect current to Px_START
                self._create_control_flow_to_parallel_node(current, px_start, "sequential", "Rule 2 - Serial to Parallel Distribution")
                
            # Rule 3: parallel -> parallel (same step number)
            elif (current_type == 'parallel' and next_type == 'parallel' and current_num == next_num):
                print(f"[CONTROL] Rule 3: parallel -> parallel (same step {current_num}) - no direct connection")
                # No direct connection - they're in the same parallel group
                # Ensure px_start/px_end are defined for this parallel group
                if 'px_start' not in locals() or not px_start.startswith(f"P{current_num}_"):
                    px_start = f"P{current_num}_START"
                    px_end = f"P{current_num}_END"
                    step_range = self._get_parallel_step_range(current['step_id'])
                    self._add_parallel_node(current['line_analysis'], px_start, 'distribution', step_range)
                    self._add_parallel_node(current['line_analysis'], px_end, 'merge', step_range)

                 # Connect current to Px_START
                self._create_control_flow_from_parallel_node (px_start ,current , "parallel", "Rule 3: parallel -> parallel (same step number)")
                self._create_control_flow_to_parallel_node(current, px_end, "sequential", "Rule 3: parallel -> parallel (same step number)")
                
                
            # Rule 4: parallel -> parallel (different step number)
            elif (current_type == 'parallel' and next_type == 'parallel' and current_num != next_num):
                # Ensure px_start/px_end are defined for current parallel group
                if 'px_start' not in locals() or not px_start.startswith(f"P{current_num}_"):
                    px_start = f"P{current_num}_START"
                    px_end = f"P{current_num}_END"
                    step_range = self._get_parallel_step_range(current['step_id'])
                    self._add_parallel_node(current['line_analysis'], px_start, 'distribution', step_range)
                    self._add_parallel_node(current['line_analysis'], px_end, 'merge', step_range)

                 # finalize current
                self._create_control_flow_from_parallel_node (px_start ,current , "parallel", "Rule 4: parallel -> parallel (different step number)")
                self._create_control_flow_to_parallel_node(current, px_end, "sequential", "Rule 4: parallel -> parallel (different step number)")
                print(f"[CONTROL] Rule 4: parallel -> parallel (different steps {current_num} -> {next_num})")
                px_endold = px_end
                px_end = f"P{next_num}_END"
                px_start = f"P{next_num}_START"
                
                current_range = self._get_parallel_step_range(current['step_id'])
                next_range = self._get_parallel_step_range(next_step['step_id'])


                # Add parallel nodes
                self._add_parallel_node(current['line_analysis'], px_end, 'merge', current_range)
                self._add_parallel_node(next_step['line_analysis'], px_start, 'distribution', next_range)
                
                # Connect Px_END with Px+1_START
                self._create_control_flow_between_parallel_nodes(px_endold, px_start, "sequential", "Rule 4 - Parallel Group Transition")
                
            # Rule 5: parallel -> serial
            elif current_type == 'parallel' and next_type == 'serial':
                print(f"[CONTROL] Rule 5: parallel -> serial (add P{current_num}_END)")
                current_range = self._get_parallel_step_range(current['step_id'])
                
                # finalize current 
                self._create_control_flow_from_parallel_node (px_start ,current , "parallel", "Rule 5: parallel -> serial")
                self._create_control_flow_to_parallel_node(current, px_end, "sequential", "Rule 5: parallel -> serial")
               # Add parallel merge node
                self._add_parallel_node(current['line_analysis'], px_end, 'merge', current_range)
                
                # Connect Px_END to next serial step
                self._create_control_flow_from_parallel_node(px_end, next_step, "sequential", "Rule 5 - Parallel to Serial Merge")
        
        print(f"[CONTROL] Control flow generation completed")

    def _generate_actor_boundary_flows(self):
        """Generate control flows: Actor -> Boundary -> Controller"""
        print("[ACTOR-BOUNDARY] Generating Actor-Boundary and Boundary-Controller flows...")

        # Collect all actors, boundaries, and controllers from line_analyses
        actors = []
        boundaries = []
        controllers = []

        for line_analysis in self.line_analyses:
            for ra in line_analysis.ra_classes:
                if ra.ra_type == RAType.ACTOR and ra.name not in [a.name for a in actors]:
                    actors.append(ra)
                elif ra.ra_type == RAType.BOUNDARY and ra.name not in [b.name for b in boundaries]:
                    boundaries.append(ra)
                elif ra.ra_type == RAType.CONTROLLER and ra.name not in [c.name for c in controllers]:
                    controllers.append(ra)

        # Step 1: Actor -> Boundary flows for triggers and transaction verbs
        for line_analysis in self.line_analyses:
            if not line_analysis.step_id or not line_analysis.grammatical:
                continue

            step_id = line_analysis.step_id
            grammatical = line_analysis.grammatical
            line_text = line_analysis.line_text.lower()

            # Check if this is a trigger step (B1 or has "(trigger)")
            is_trigger = step_id == "B1" or "(trigger)" in line_text

            # Find relevant boundary for this step
            step_boundaries = [ra for ra in line_analysis.ra_classes if ra.ra_type == RAType.BOUNDARY]

            if step_boundaries and actors:
                # Determine correct actor based on trigger/boundary type
                # - TimingBoundary -> Actor "Time"
                # - User interaction boundaries -> Actor "User"
                actor_name = "User"  # Default

                for boundary in step_boundaries:
                    if boundary.name == "TimingBoundary":
                        # Time-based trigger uses "Time" actor if available
                        time_actor = next((a for a in actors if a.name == "Time"), None)
                        if time_actor:
                            actor_name = time_actor.name
                    elif self._is_real_user_interaction(line_text, grammatical):
                        # User interaction uses "User" actor
                        user_actor = next((a for a in actors if a.name == "User"), None)
                        if user_actor:
                            actor_name = user_actor.name

                for boundary in step_boundaries:
                    boundary_name = boundary.name

                    # Create Actor -> Boundary flow
                    if is_trigger or grammatical.verb_type == VerbType.TRANSACTION_VERB:
                        # Check if this flow already exists in this line_analysis
                        existing = any(
                            cf.source_step == actor_name and cf.target_step == boundary_name
                            for cf in line_analysis.control_flows
                        )

                        if not existing:
                            flow = ControlFlow(
                                source_step=actor_name,
                                target_step=boundary_name,
                                source_controller=actor_name,
                                target_controller=boundary_name,
                                flow_type='signal' if is_trigger else 'transaction',
                                rule='Actor-Boundary interaction',
                                description=f"Actor {actor_name} signals {boundary_name}" if is_trigger else f"Actor {actor_name} interacts with {boundary_name}"
                            )
                            line_analysis.control_flows.append(flow)
                            print(f"[ACTOR-BOUNDARY] Created: {actor_name} -> {boundary_name} ({step_id})")

        # Step 2: Boundary -> Controller flows for each step
        for line_analysis in self.line_analyses:
            if not line_analysis.step_id:
                continue

            step_id = line_analysis.step_id
            line_text = line_analysis.line_text.lower()

            # Find boundaries and controllers for this step
            step_boundaries = [ra for ra in line_analysis.ra_classes if ra.ra_type == RAType.BOUNDARY]
            step_controllers = [ra for ra in line_analysis.ra_classes if ra.ra_type == RAType.CONTROLLER]

            # Check if this is a user interaction step
            # TRUE user interactions:
            #   - "presents cup to user" (to user / from user)
            #   - "User presses button" (User as subject)
            #   - "displays message to user" (output to user)
            # FALSE positives to exclude:
            #   - "user defined time" (user as adjective)
            #   - "user preferences" (user as adjective)
            is_user_interaction = self._is_real_user_interaction(line_text, line_analysis.grammatical)

            # Create Boundary -> Controller flow
            for boundary in step_boundaries:
                if is_user_interaction:
                    # User interaction: Boundary -> HMIManager -> Material-Controller
                    # Step 2a: Create Boundary -> HMIManager flow
                    hmi_controller = next((c for c in controllers if c.name == 'HMIManager'), None)

                    if hmi_controller:
                        # Check if Boundary -> HMIManager flow exists
                        existing_hmi = any(
                            cf.source_step == boundary.name and cf.target_step == 'HMIManager'
                            for cf in line_analysis.control_flows
                        )

                        if not existing_hmi:
                            flow = ControlFlow(
                                source_step=boundary.name,
                                target_step='HMIManager',
                                source_controller=boundary.name,
                                target_controller='HMIManager',
                                flow_type='activation',
                                rule='User-Interaction Boundary to HMI',
                                description=f"User interaction {boundary.name} ({step_id}) processed by HMIManager"
                            )
                            line_analysis.control_flows.append(flow)
                            print(f"[ACTOR-BOUNDARY] Created HMI interaction: {boundary.name} ({step_id}) -> HMIManager")

                    # Step 2b: Create HMIManager -> Material-Controller flows
                    for controller in step_controllers:
                        # Skip if it's HMI itself
                        if controller.name == 'HMIManager':
                            continue

                        # Check if HMIManager -> Controller flow exists
                        existing = any(
                            cf.source_step == 'HMIManager' and cf.target_step == controller.name
                            for cf in line_analysis.control_flows
                        )

                        if not existing:
                            flow = ControlFlow(
                                source_step='HMIManager',
                                target_step=controller.name,
                                source_controller='HMIManager',
                                target_controller=controller.name,
                                flow_type='activation',
                                rule='HMI to Material-Controller',
                                description=f"HMIManager routes user interaction to {controller.name}"
                            )
                            line_analysis.control_flows.append(flow)
                            print(f"[ACTOR-BOUNDARY] Created HMI routing: HMIManager -> {controller.name} ({step_id})")
                else:
                    # Non-user interaction: Direct Boundary -> Controller
                    for controller in step_controllers:
                        # Check if this flow already exists
                        existing = any(
                            cf.source_step == boundary.name and cf.target_step == controller.name
                            for cf in line_analysis.control_flows
                        )

                        if not existing:
                            flow = ControlFlow(
                                source_step=boundary.name,
                                target_step=controller.name,
                                source_controller=boundary.name,
                                target_controller=controller.name,
                                flow_type='activation',
                                rule='Boundary-Controller activation',
                                description=f"Boundary {boundary.name} activates {controller.name}"
                            )
                            line_analysis.control_flows.append(flow)
                            print(f"[ACTOR-BOUNDARY] Created: {boundary.name} -> {controller.name} ({step_id})")

        # Step 3: Special handling for Extension/Alternative Trigger -> Action flows
        # E1 (trigger) has SugarRequestBoundary, E1.1 (action) has SugarSolidManager
        # We need to connect E1's Boundary to E1.1's Controller
        print("[ACTOR-BOUNDARY] Handling Extension/Alternative trigger-to-action flows...")

        for i, line_analysis in enumerate(self.line_analyses):
            if not line_analysis.step_id:
                continue

            step_id = line_analysis.step_id
            line_text = line_analysis.line_text.lower()

            # Check if this is an Extension/Alternative trigger (E1, A1, A2, etc.)
            # Pattern: E1, A1, A2 (no dot, contains "(trigger)")
            is_ext_alt_trigger = (
                (step_id.startswith('E') or step_id.startswith('A')) and
                '.' not in step_id and
                '(trigger)' in line_text
            )

            if is_ext_alt_trigger:
                # Find boundary in this trigger step
                trigger_boundaries = [ra for ra in line_analysis.ra_classes if ra.ra_type == RAType.BOUNDARY]

                if trigger_boundaries:
                    # Check if this is a user transaction (wants, requests, enters, etc.)
                    # Transaction verbs indicate HMI interaction
                    grammatical = line_analysis.grammatical

                    # Check if this is a real user interaction (not just "user" as adjective)
                    # Examples: "User wants sugar", "User requests...", "User presses..."
                    is_user_transaction = self._is_real_user_interaction(line_text, grammatical)

                    if is_user_transaction:
                        # User transaction: Boundary -> HMIManager -> Material-Controller
                        # Step 3a: Create Boundary -> HMIManager flow
                        hmi_controller = next((c for c in controllers if c.name == 'HMIManager'), None)

                        if hmi_controller:
                            for boundary in trigger_boundaries:
                                # Check if Boundary -> HMIManager flow exists
                                existing = any(
                                    cf.source_step == boundary.name and cf.target_step == 'HMIManager'
                                    for cf in line_analysis.control_flows
                                )

                                if not existing:
                                    flow = ControlFlow(
                                        source_step=boundary.name,
                                        target_step='HMIManager',
                                        source_controller=boundary.name,
                                        target_controller='HMIManager',
                                        flow_type='transaction',
                                        rule='User-Transaction Boundary to HMI',
                                        description=f"User transaction {boundary.name} ({step_id}) processed by HMIManager"
                                    )
                                    line_analysis.control_flows.append(flow)
                                    print(f"[ACTOR-BOUNDARY] Created HMI transaction: {boundary.name} ({step_id}) -> HMIManager")

                    # Find the next action step (E1.1, A1.1, etc.)
                    expected_action_step = f"{step_id}.1"

                    # Search for action step in line_analyses
                    for next_line_analysis in self.line_analyses:
                        if next_line_analysis.step_id == expected_action_step:
                            # Find controller in action step
                            action_controllers = [ra for ra in next_line_analysis.ra_classes if ra.ra_type == RAType.CONTROLLER]

                            if action_controllers:
                                if is_user_transaction:
                                    # For user transactions: HMIManager -> Material-Controller
                                    source_controller = 'HMIManager'
                                    source_name = 'HMIManager'
                                else:
                                    # For non-user triggers: Boundary -> Material-Controller
                                    source_controller = trigger_boundaries[0].name
                                    source_name = trigger_boundaries[0].name

                                for controller in action_controllers:
                                    # Check if flow already exists
                                    existing_in_trigger = any(
                                        cf.source_step == source_name and cf.target_step == controller.name
                                        for cf in line_analysis.control_flows
                                    )
                                    existing_in_action = any(
                                        cf.source_step == source_name and cf.target_step == controller.name
                                        for cf in next_line_analysis.control_flows
                                    )

                                    if not existing_in_trigger and not existing_in_action:
                                        flow = ControlFlow(
                                            source_step=source_name,
                                            target_step=controller.name,
                                            source_controller=source_controller,
                                            target_controller=controller.name,
                                            flow_type='activation',
                                            rule='Trigger to Action-Controller',
                                            description=f"{source_name} ({step_id}) activates {controller.name} ({expected_action_step})"
                                        )
                                        # Add to action step's control flows
                                        next_line_analysis.control_flows.append(flow)
                                        print(f"[ACTOR-BOUNDARY] Created trigger-to-action: {source_name} ({step_id}) -> {controller.name} ({expected_action_step})")
                            break

        print(f"[ACTOR-BOUNDARY] Actor-Boundary flow generation completed")

    def _is_real_user_interaction(self, line_text: str, grammatical: 'GrammaticalAnalysis') -> bool:
        """
        Check if this is a real user interaction or just "user" as adjective.

        Real user interactions:
            - "User presses button" - User as subject
            - "presents cup to user" - Output/interaction TO user
            - "receives input from user" - Input FROM user
            - "displays message to user" - Output TO user

        NOT user interactions (false positives):
            - "user defined time" - "user" is adjective modifying "time"
            - "user preferences" - "user" is adjective
            - "user settings" - "user" is adjective

        Args:
            line_text: Lowercase line text
            grammatical: Grammatical analysis object (may be None)

        Returns:
            True if this is a real user interaction, False otherwise
        """
        # Pattern 1: "to user" or "from user" - always a real interaction
        if 'to user' in line_text or 'from user' in line_text:
            return True

        # Pattern 2: Check if "user" appears as adjective before a noun
        # Common adjective patterns: "user defined", "user specified", "user configured"
        adjective_patterns = [
            'user defined',
            'user specified',
            'user configured',
            'user selected',
            'user preferences',
            'user settings',
            'user options'
        ]

        for pattern in adjective_patterns:
            if pattern in line_text:
                return False  # "user" is adjective, not interaction

        # Pattern 3: Check if "User" is at start of sentence (likely subject)
        # Extract the actual UC step text (remove step ID prefix like "B1 (trigger)")
        # Patterns to handle:
        #   "b1 (trigger) user wants sugar" -> "user wants sugar"
        #   "e1 at b3-b5 (trigger) user wants sugar" -> "user wants sugar"
        #   "e1 b4-b5 (trigger) user wants sugar" -> "user wants sugar"  # FIX: without "at"
        #   "b4 the system outputs..." -> "the system outputs..."
        # NOTE: line_text is already lowercase, so pattern must be lowercase too
        import re
        # FIX: Make "at" optional separately from the range pattern
        text_without_prefix = re.sub(r'^[bae]\d+(?:\.\d+)?[a-z]?\s*(?:at\s+)?(?:[bae]\d+-[bae]\d+)?\s*(\([^)]+\))?\s*', '', line_text)

        if text_without_prefix.startswith('user '):
            return True  # User is subject

        # Pattern 4: If none of the above, check if "user" appears at all
        # But be conservative - if it's not clearly an interaction, return False
        return False

    def _is_in_parallel_group(self, step_id: str) -> bool:
        """Check if step is part of a parallel group (has letter suffix)
        Recognizes: B5a, B5b, A1.2a, A1.2b, E3.3a, E3.3b"""
        match = re.match(r'^([BAE]\d+(?:\.\d+)?)([a-z])$', step_id)
        if not match:
            return False
        
        # Check if there are other steps with same base but different letters
        base_step = match.group(1)  # B5, A1.2, E3.3
        parallel_steps = []
        
        for line_analysis in self.line_analyses:
            if line_analysis.step_id:
                step_match = re.match(r'^([BAE]\d+(?:\.\d+)?)([a-z])$', line_analysis.step_id)
                if step_match and step_match.group(1) == base_step:
                    parallel_steps.append(line_analysis.step_id)
        
        # If there are multiple steps with same base, they are parallel
        return len(parallel_steps) > 1

    def _are_in_same_flow_scope(self, step_id1: str, step_id2: str) -> bool:
        """
        Check if two step IDs are in the same flow scope.

        Flow scopes:
        - Main flow: B1, B2a, B3b, B4, B5, B6
        - Alternative flow A1: A1, A1.1, A1.2, A1.3
        - Alternative flow A2: A2, A2.1, A2.2, A2.3
        - Extension flow E1: E1, E1.1, E1.2

        Returns:
            True if both steps are in the same flow scope
            False if they are in different flow scopes
        """
        def get_flow_scope(step_id: str) -> str:
            """Get the flow scope identifier for a step"""
            if step_id.startswith('B'):
                return 'B'  # All B steps are in main flow
            elif step_id.startswith('A'):
                # Extract A1, A2, A3, etc. (number without dot)
                match = re.match(r'^(A\d+)', step_id)
                return match.group(1) if match else step_id
            elif step_id.startswith('E'):
                # Extract E1, E2, E3, etc. (number without dot)
                match = re.match(r'^(E\d+)', step_id)
                return match.group(1) if match else step_id
            else:
                return step_id

        scope1 = get_flow_scope(step_id1)
        scope2 = get_flow_scope(step_id2)

        return scope1 == scope2

    def _is_first_in_parallel_group(self, step_id: str) -> bool:
        """Check if step is first in a parallel group (suffix 'a')"""
        match = re.match(r'^([BAE]\d+(?:\.\d+)?)([a-z])$', step_id)
        return match and match.group(2) == 'a'
    
    def _has_parallel_pattern(self, step_id: str) -> bool:
        """Check if step has parallel pattern (letter suffix) using regex only
        Returns True for: B2a, B3b, A1.2a, E3.3b, etc."""
        return re.match(r'^[BAE]\d+(?:\.\d+)?[a-z]$', step_id) is not None
    
    def _extract_parallel_step_number(self, step_id: str) -> Optional[int]:
        """Extract the parallel step number from step_id using regex
        B2a -> 2, A1.2a -> 12, E3.3b -> 33"""
        match = re.match(r'^[BAE](\d+)(?:\.(\d+))?[a-z]$', step_id)
        if not match:
            return None
        
        main_num = int(match.group(1))
        sub_num = int(match.group(2)) if match.group(2) else 0
        
        # Create unique number: B2a=2, A1.2a=12, E3.3a=33
        return main_num * 10 + sub_num if sub_num > 0 else main_num
    
    def _get_parallel_step_range(self, step_id: str) -> str:
        """Get the range of parallel steps for a given step_id
        B2a -> B2a-B2d, A1.2a -> A1.2a-A1.2d"""
        match = re.match(r'^([BAE]\d+(?:\.\d+)?)([a-z])$', step_id)
        if not match:
            return step_id
        
        base_step = match.group(1)  # B2, A1.2, E3.3
        
        # Find all parallel steps with same base
        parallel_steps = []
        for line_analysis in self.line_analyses:
            if line_analysis.step_id:
                step_match = re.match(r'^([BAE]\d+(?:\.\d+)?)([a-z])$', line_analysis.step_id)
                if step_match and step_match.group(1) == base_step:
                    parallel_steps.append(line_analysis.step_id)
        
        if len(parallel_steps) <= 1:
            return step_id
        
        # Sort and create range
        parallel_steps.sort()
        return f"{parallel_steps[0]}-{parallel_steps[-1]}"
    
    def _is_after_parallel_group(self, step_id: str, current_index: int, step_analyses: List) -> bool:
        """Check if step comes after a parallel group"""
        if current_index == 0:
            return False
        
        # Check if previous step was part of a parallel group
        prev_step = step_analyses[current_index - 1]
        prev_match = re.match(r'^([BAE]\d+(?:\.\d+)?)([a-z])$', prev_step.step_id)
        
        # If previous step has letter suffix and current doesn't continue the group
        if prev_match:
            curr_match = re.match(r'^([BAE]\d+(?:\.\d+)?)([a-z]?)$', step_id)
            if curr_match:
                prev_base = prev_match.group(1)
                curr_base = curr_match.group(1)
                # Different base means we're after the parallel group
                return prev_base != curr_base
        
        return False
    
    def _create_parallel_start_flows(self, current_step, current_index: int, step_analyses: List):
        """Create flows for first step in parallel group: Previous -> Distribution -> Current"""
        if current_index == 0:
            return
        
        current_controller = self._get_controller_for_step(current_step)
        if not current_controller:
            return
        
        # Find step before parallel group
        base_step = re.match(r'^([BAEF]\d+)', current_step.step_id).group(1)
        
        # Find previous step (before parallel group)
        prev_step = None
        for j in range(current_index - 1, -1, -1):
            check_step = step_analyses[j]
            check_base = re.match(r'^([BAEF]\d+)', check_step.step_id).group(1)
            if check_base != base_step:
                prev_step = check_step
                break
        
        if prev_step:
            prev_controller = self._get_controller_for_step(prev_step)
            if prev_controller:
                # Create flow: Previous -> Current (will be modified to use distribution node in diagram)
                control_flow = ControlFlow(
                    source_step=prev_step.step_id,
                    target_step=current_step.step_id,
                    source_controller=prev_controller,
                    target_controller=current_controller,
                    flow_type="parallel_start",
                    rule="Rule 4 - Parallel Flow",
                    description=f"Parallel flow start from {prev_step.step_id} to {current_step.step_id}"
                )
                current_step.control_flows.append(control_flow)
    
    def _create_parallel_step_flows(self, current_step, current_index: int, step_analyses: List):
        """Create flows for parallel steps B2b, B2c, B2d - they are parallel to B2a, not sequential"""
        current_controller = self._get_controller_for_step(current_step)
        if not current_controller:
            return
        
        # Find the base step (e.g., B2 from B2b)
        base_step = re.match(r'^([BAEF]\d+)', current_step.step_id).group(1)
        
        # Find previous step (before entire parallel group, not the previous parallel step)
        prev_step = None
        for j in range(current_index - 1, -1, -1):
            check_step = step_analyses[j]
            check_base = re.match(r'^([BAEF]\d+)', check_step.step_id).group(1)
            if check_base != base_step:
                prev_step = check_step
                break
        
        if prev_step:
            prev_controller = self._get_controller_for_step(prev_step)
            if prev_controller:
                # Create flow: Previous -> Current (parallel step)
                # This indicates the step is parallel, will be routed through distribution node
                control_flow = ControlFlow(
                    source_step=prev_step.step_id,
                    target_step=current_step.step_id,
                    source_controller=prev_controller,
                    target_controller=current_controller,
                    flow_type="parallel_step",  # Different from parallel_start
                    rule="Rule 4 - Parallel Flow",
                    description=f"Parallel step {current_step.step_id} (parallel to other {base_step} steps)"
                )
                current_step.control_flows.append(control_flow)
    
    def _create_parallel_end_flows(self, current_step, current_index: int, step_analyses: List):
        """Create flows for step after parallel group: P_END -> Current (no direct controller connection)"""
        current_controller = self._get_controller_for_step(current_step)
        if not current_controller:
            return
        
        # For steps after parallel groups, the flow should go through P_END node
        # We mark this as a special flow type that should connect via parallel nodes
        # The actual connection B2a/B2b/B2c/B2d -> P1_END -> B3a will be handled
        # by the parallel control flows generation (_generate_parallel_control_flows)
        
        # Create a placeholder flow that indicates this step follows a parallel group
        # This will be used by the JSON generation to create proper P_END connections
        control_flow = ControlFlow(
            source_step="P_END",  # Placeholder - indicates parallel merge
            target_step=current_step.step_id,
            source_controller="PARALLEL_MERGE",  # Special marker
            target_controller=current_controller,
            flow_type="after_parallel",
            rule="Rule 4 - Parallel Flow",
            description=f"Step follows parallel group via merge node"
        )
        current_step.control_flows.append(control_flow)
    
    def _create_sequential_flows(self, current_step, current_index: int, step_analyses: List):
        """Create normal sequential control flows"""
        if current_index == 0:
            return
        
        current_controller = self._get_controller_for_step(current_step)
        if not current_controller:
            return
        
        # Find immediate predecessor
        prev_step = step_analyses[current_index - 1]
        prev_controller = self._get_controller_for_step(prev_step)
        
        if prev_controller:
            control_flow = ControlFlow(
                source_step=prev_step.step_id,
                target_step=current_step.step_id,
                source_controller=prev_controller,
                target_controller=current_controller,
                flow_type="sequential",
                rule="Rule 2",
                description=f"Sequential processing from {prev_step.step_id} to {current_step.step_id}"
            )
            current_step.control_flows.append(control_flow)
    
    def _find_control_flow_predecessors(self, current_step: str, current_index: int, step_analyses: List) -> List[Tuple[str, str]]:
        """Find predecessor steps for control flow"""
        predecessors = []
        
        if current_index == 0:
            return predecessors  # First step has no predecessors
        
        # Check if current step is parallel (same letter, different suffix)
        current_base = re.match(r'^([BAEF]\d+)', current_step).group(1) if re.match(r'^([BAEF]\d+)', current_step) else current_step
        
        # Look for immediate predecessor
        prev_step = step_analyses[current_index - 1]
        prev_base = re.match(r'^([BAEF]\d+)', prev_step.step_id).group(1) if re.match(r'^([BAEF]\d+)', prev_step.step_id) else prev_step.step_id
        
        
        # If current step is parallel (e.g., B2b after B2a), link to step before parallel group
        if current_base == prev_base:  # Same base (B2a, B2b both have base B2)
            # Find step before parallel group
            for j in range(current_index - 1, -1, -1):
                check_step = step_analyses[j]
                check_base = re.match(r'^([BAEF]\d+)', check_step.step_id).group(1) if re.match(r'^([BAEF]\d+)', check_step.step_id) else check_step.step_id
                if check_base != current_base:
                    # Found step before parallel group
                    pred_controller = self._get_controller_for_step(check_step)
                    if pred_controller:
                        predecessors.append((check_step.step_id, pred_controller))
                    break
        else:
            # Check if current step is a substep (e.g., E1.4 after E1.3)
            # For substeps within same flow, create sequential link
            if self._is_substep_sequence(prev_step.step_id, current_step):
                pred_controller = self._get_controller_for_step(prev_step)
                if pred_controller:
                    predecessors.append((prev_step.step_id, pred_controller))
            else:
                # Normal sequential step - link to immediate predecessor
                pred_controller = self._get_controller_for_step(prev_step)
                if pred_controller:
                    predecessors.append((prev_step.step_id, pred_controller))
        
        return predecessors
    
    def _is_substep_sequence(self, prev_step: str, current_step: str) -> bool:
        """Check if current step is sequential substep of previous step"""
        # Extract step prefixes (E1.3 -> E1, E1.4 -> E1)
        prev_match = re.match(r'^([BAEF]\d+)', prev_step)
        curr_match = re.match(r'^([BAEF]\d+)', current_step)
        
        if prev_match and curr_match:
            prev_prefix = prev_match.group(1)
            curr_prefix = curr_match.group(1)
            
            # Same prefix means they're in same flow (E1.3 and E1.4 both have E1)
            if prev_prefix == curr_prefix:
                # Extract substep numbers
                prev_substep = re.match(r'^[BAEF]\d+\.(\d+)', prev_step)
                curr_substep = re.match(r'^[BAEF]\d+\.(\d+)', current_step)
                
                if prev_substep and curr_substep:
                    prev_num = int(prev_substep.group(1))
                    curr_num = int(curr_substep.group(1))
                    # Check if current is next substep (E1.3 -> E1.4)
                    return curr_num == prev_num + 1
        
        return False
    
    def _get_controller_for_step(self, step_analysis) -> Optional[str]:
        """Get controller name for a step analysis"""
        for ra_class in step_analysis.ra_classes:
            if ra_class.ra_type == RAType.CONTROLLER:
                return ra_class.name
        return None
    
    def _create_control_flow(self, current, next_step, flow_type, rule):
        """Create a standard control flow between two steps"""
        control_flow = ControlFlow(
            source_step=current['step_id'],
            target_step=next_step['step_id'],
            source_controller=current['controller'],
            target_controller=next_step['controller'],
            flow_type=flow_type,
            rule=rule,
            description=f"{flow_type.capitalize()} flow from {current['step_id']} to {next_step['step_id']}"
        )
        current['line_analysis'].control_flows.append(control_flow)
        print(f"[CONTROL] Created flow: {current['controller']} -> {next_step['controller']} ({rule})")
    
    def _add_parallel_node(self, line_analysis, node_id, node_type, step_range):
        """Add a parallel flow node"""
        parallel_node = ParallelFlowNode(
            node_id=node_id,
            node_type=node_type,
            step_range=step_range,
            description=f"{node_type.capitalize()} node for parallel steps {step_range}",
            parallel_steps=step_range.split('-') if '-' in step_range else [step_range]
        )
        
        # Add to global parallel nodes if not already present
        if not any(pn.node_id == node_id for pn in self.parallel_flow_nodes):
            self.parallel_flow_nodes.append(parallel_node)
            print(f"[CONTROL] Added parallel node: {node_id} ({node_type}) for {step_range}")
    
    def _create_control_flow_to_parallel_node(self, current, node_id, flow_type, rule):
        """Create a control flow from a step to a parallel node"""
        control_flow = ControlFlow(
            source_step=current['step_id'],
            target_step=node_id,
            source_controller=current['controller'],
            target_controller=node_id,
            flow_type=flow_type,
            rule=rule,
            description=f"Flow from {current['step_id']} to parallel {node_id}"
        )
        current['line_analysis'].control_flows.append(control_flow)
        print(f"[CONTROL] Created flow to parallel node: {current['controller']} -> {node_id} ({rule})")
    
    def _create_control_flow_from_parallel_node(self, node_id, next_step, flow_type, rule):
        """Create a control flow from a parallel node to a step"""
        control_flow = ControlFlow(
            source_step=node_id,
            target_step=next_step['step_id'],
            source_controller=node_id,
            target_controller=next_step['controller'],
            flow_type=flow_type,
            rule=rule,
            description=f"Flow from parallel {node_id} to {next_step['step_id']}"
        )
        next_step['line_analysis'].control_flows.append(control_flow)
        print(f"[CONTROL] Created flow from parallel node: {node_id} -> {next_step['controller']} ({rule})")
    
    def _create_control_flow_between_parallel_nodes(self, source_node, target_node, flow_type, rule):
        """Create a control flow between two parallel nodes"""
        # We need to store this somewhere - let's add it to the first line analysis for now
        if self.line_analyses:
            control_flow = ControlFlow(
                source_step=source_node,
                target_step=target_node,
                source_controller=source_node,
                target_controller=target_node,
                flow_type=flow_type,
                rule=rule,
                description=f"Flow between parallel nodes {source_node} -> {target_node}"
            )
            # Find a suitable line analysis to attach this to
            # Prefer attaching to the first line that has a step_id
            target_line = None
            for la in self.line_analyses:
                if la.step_id:
                    target_line = la
                    break
            
            if target_line:
                target_line.control_flows.append(control_flow)
                print(f"[CONTROL] Created flow between parallel nodes: {source_node} -> {target_node} ({rule})")
    
    
    def _get_parallel_group_from_step_id(self, step_id: str) -> int:
        """
        Determine parallel group directly from step_id
        Returns: 0 for sequential, 2 for P2 (B2x), 3 for P3 (B3x), etc.
        """
        # Check if this step is part of a parallel group
        match = re.match(r'^B(\d+)([a-z])$', step_id)
        if match:
            base_number = int(match.group(1))
            # B2x -> parallel group 2, B3x -> parallel group 3, etc.
            return base_number
        
        # For non-parallel steps (B1, B4, A1, E1, etc.), return 0 (sequential)
        return 0
    
    def _detect_errors_and_suggestions(self):
        """Detect errors and generate suggestions - to be implemented"""
        pass
    
    def _generate_json_output(self, uc_file_path: str) -> str:
        """Generate comprehensive JSON output"""
        
        # Collect all RA classes, control flows, data flows, and parallel nodes
        all_ra_classes = []
        all_control_flows = []
        all_data_flows = []
        
        for line_analysis in self.line_analyses:
            all_ra_classes.extend(line_analysis.ra_classes)
            all_control_flows.extend(line_analysis.control_flows)
            all_data_flows.extend(line_analysis.data_flows)
        
        # Use parallel nodes from analyzer state (avoid duplicates)
        all_parallel_nodes = self.parallel_flow_nodes
        
        # Group by type and deduplicate by name
        actors = self._deduplicate_ra_classes([ra for ra in all_ra_classes if ra.ra_type == RAType.ACTOR])
        controllers = self._deduplicate_ra_classes([ra for ra in all_ra_classes if ra.ra_type == RAType.CONTROLLER])
        boundaries = self._deduplicate_ra_classes([ra for ra in all_ra_classes if ra.ra_type == RAType.BOUNDARY])
        entities = self._deduplicate_ra_classes([ra for ra in all_ra_classes if ra.ra_type == RAType.ENTITY])
        
        # Create JSON structure
        uc_name = Path(uc_file_path).stem
        analysis_json = {
            'meta': {
                'use_case': uc_name,
                'domain': self.domain_name,
                'capability': self.uc_context.capability,
                'feature': self.uc_context.feature,
                'goal': self.uc_context.goal,
                'total_ra_classes': len(all_ra_classes),
                'analysis_engine': 'structured_uc_analyzer.py',
                'generated_at': datetime.now().isoformat(),
                'json_format_version': '2.0'
            },
            'components': {
                'actors': [{'name': ra.name, 'description': ra.description} for ra in actors],
                'controllers': [{'name': ra.name, 'description': ra.description, 'parallel_group': ra.parallel_group} for ra in controllers],
                'boundaries': [{'name': ra.name, 'description': ra.description} for ra in boundaries],
                'entities': [{'name': ra.name, 'description': ra.description} for ra in entities],
                'control_flow_nodes': [{'name': pn.node_id, 'description': pn.description, 'type': pn.node_type} for pn in all_parallel_nodes]
            },
            'layout': {
                'uc_step_order': self._generate_uc_step_layout_order(controllers, all_parallel_nodes)
            },
            'relationships': {
                'control_flows': [
                    {
                        'source': cf.source_controller,
                        'destination': cf.target_controller,
                        'source_step': cf.source_step,
                        'target_step': cf.target_step,
                        'rule': cf.rule,
                        'description': cf.description
                    } for cf in all_control_flows
                ],  # Removed old _generate_parallel_control_flows() - now using correct control flow logic
                'data_flows': [
                    {
                        'controller': df.controller,
                        'entity': df.entity,
                        'step_id': df.step_id,
                        'type': df.flow_type,
                        'preposition': df.preposition,
                        'description': df.description
                    } for df in all_data_flows
                ],
                'parallel_nodes': [
                    {
                        'node_id': pn.node_id,
                        'node_type': pn.node_type,
                        'step_range': pn.step_range,
                        'parallel_steps': pn.parallel_steps,
                        'description': pn.description
                    } for pn in all_parallel_nodes
                ]
            },
            'context': {
                'actors': self.uc_context.actors,
                'preconditions': self.uc_context.preconditions
            },
            'step_contexts': [
                {
                    'step_id': la.step_context.step_id,
                    'step_type': la.step_context.step_type,
                    'domain': la.step_context.domain,
                    'phase': la.step_context.phase,
                    'business_context': la.step_context.business_context,
                    'technical_context': la.step_context.technical_context,
                    'actors_involved': la.step_context.actors_involved,
                    'step_text': la.line_text,
                    'nlp_context': self._extract_nlp_context_for_step(la.step_context.step_id) if la.step_context.step_id else {}
                } for la in self.line_analyses if la.step_context
            ],
            'summary': {
                'total_components': len(all_ra_classes),
                'actors_count': len(actors),
                'controllers_count': len(controllers),
                'boundaries_count': len(boundaries),
                'entities_count': len(entities),
                'control_flows_count': len(all_control_flows),
                'data_flows_count': len(all_data_flows),
                'parallel_nodes_count': len(all_parallel_nodes)
            }
        }
        
        # Save JSON file in new folder
        import os
        os.makedirs("new", exist_ok=True)
        json_file_path = f"new/{uc_name}_Structured_RA_Analysis.json"
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_json, f, indent=2, ensure_ascii=False)
        
        print(f"[JSON] Analysis saved to: {json_file_path}")
        return json_file_path
    
    def _generate_uc_step_layout_order(self, controllers: List[RAClass], parallel_nodes: List[ParallelFlowNode]) -> List[Dict]:
        """Generiere UC-Schritt Layout-Reihenfolge basierend auf bestehenden Analysen"""
        
        # Extrahiere UC-Schritt-Nummern aus Controller-Descriptions
        controller_steps = []
        for controller in controllers:
            step_num = self._extract_step_number_from_description(controller.description)
            controller_steps.append({
                'step_number': step_num,
                'type': 'controller',
                'name': controller.name,
                'parallel_group': controller.parallel_group
            })
        
        # Sortiere nach Schritt-Nummer
        controller_steps.sort(key=lambda x: x['step_number'])
        
        # Erstelle Layout-Order mit Control Flow Nodes
        layout_order = []
        parallel_nodes_by_group = {}
        
        # Gruppiere parallel nodes
        for node in parallel_nodes:
            import re
            match = re.match(r'P(\d+)_(START|END)', node.node_id)
            if match:
                group_num = int(match.group(1))
                node_type = match.group(2)
                if group_num not in parallel_nodes_by_group:
                    parallel_nodes_by_group[group_num] = {}
                parallel_nodes_by_group[group_num][node_type] = node.node_id
        
        # Baue Layout-Reihenfolge auf
        current_parallel_groups = sorted(parallel_nodes_by_group.keys())
        used_parallel_groups = set()
        
        for controller_info in controller_steps:
            step_num = controller_info['step_number']
            parallel_group = controller_info['parallel_group']
            
            # Füge parallel nodes vor entsprechenden Controllers hinzu
            for group_num in current_parallel_groups:
                if group_num in used_parallel_groups:
                    continue
                if group_num == parallel_group:
                    # START node vor parallel group
                    if 'START' in parallel_nodes_by_group[group_num]:
                        layout_order.append({
                            'type': 'control_flow_node',
                            'name': parallel_nodes_by_group[group_num]['START'],
                            'position_before_group': group_num
                        })
                    break
            
            # Füge Controller hinzu
            layout_order.append(controller_info)
            
            # Füge END node nach parallel group hinzu
            if parallel_group > 0 and parallel_group not in used_parallel_groups:
                if 'END' in parallel_nodes_by_group[parallel_group]:
                    layout_order.append({
                        'type': 'control_flow_node', 
                        'name': parallel_nodes_by_group[parallel_group]['END'],
                        'position_after_group': parallel_group
                    })
                used_parallel_groups.add(parallel_group)
        
        return layout_order
    
    def _extract_step_number_from_description(self, description: str) -> int:
        """Extrahiere Schritt-Nummer aus Description für Sortierung"""
        import re
        
        # Suche nach "in B1", "in A1", "in E1" etc.
        match = re.search(r'in ([BAE])(\d+)', description)
        if match:
            step_letter = match.group(1)
            step_number = int(match.group(2))
            
            if step_letter == 'B':
                return step_number
            elif step_letter == 'A':
                return 100 + step_number  # A flows nach B flows
            elif step_letter == 'E':
                return 200 + step_number  # E flows nach A flows
        
        return 999  # Unknown/default
    
    def _extract_nlp_context_for_step(self, step_id: str) -> Dict[str, Any]:
        """Extract NLP context information for debugging purposes"""
        nlp_context = {
            'generated_contexts': [],
            'semantic_features': [],
            'domain_alignment_avg': 0.0,
            'context_types': {},
            'safety_functions': [],
            'detected_functions': []
        }
        
        if step_id not in self.generated_contexts:
            return nlp_context
        
        contexts = self.generated_contexts[step_id]
        if not contexts:
            return nlp_context
        
        # Extract information from generated contexts
        for context in contexts:
            context_data = {
                'context_type': context.context_type.value,
                'context_name': context.context_name,
                'source_text': context.source_text,
                'semantic_features': context.semantic_features,
                'domain_alignment': context.domain_alignment,
                'safety_class': context.safety_class,
                'hygiene_level': context.hygiene_level,
                'special_requirements': context.special_requirements,
                'controllers': context.controllers
            }
            nlp_context['generated_contexts'].append(context_data)
            
            # Aggregate semantic features
            nlp_context['semantic_features'].extend(context.semantic_features)
            
            # Track context types
            context_type = context.context_type.value
            nlp_context['context_types'][context_type] = nlp_context['context_types'].get(context_type, 0) + 1
            
            # Collect safety functions and detected functions
            if context.special_requirements:
                nlp_context['safety_functions'].extend(context.special_requirements)
            
            # Extract detected functions from semantic features
            function_features = [f for f in context.semantic_features if f.startswith('function:') or any(func in f for func in ['Manager', 'Controller', 'heatWater', 'protectOverheating'])]
            nlp_context['detected_functions'].extend(function_features)
        
        # Remove duplicates
        nlp_context['semantic_features'] = list(set(nlp_context['semantic_features']))
        nlp_context['safety_functions'] = list(set(nlp_context['safety_functions']))
        nlp_context['detected_functions'] = list(set(nlp_context['detected_functions']))
        
        # Calculate average domain alignment
        if contexts:
            nlp_context['domain_alignment_avg'] = sum(c.domain_alignment for c in contexts) / len(contexts)
        
        return nlp_context
    
    def _generate_csv_output(self, uc_file_path: str):
        """Generate enhanced CSV output with flow information and parallel nodes in correct positions"""
        
        uc_name = Path(uc_file_path).stem
        import os
        os.makedirs("new", exist_ok=True)
        csv_file_path = f"new/{uc_name}_Structured_UC_Steps_RA_Classes.csv"
        
        # Build mapping using the already detected parallel flow nodes
        parallel_events = {}  # step_id -> list of parallel nodes to insert
        
        # Use the simple detection logic: map P_START and P_END based on detected parallel flows
        parallel_counter = 1
        
        # Group parallel flow nodes by counter number
        start_nodes = [node for node in self.parallel_flow_nodes if node.node_type == "distribution"]
        end_nodes = [node for node in self.parallel_flow_nodes if node.node_type == "merge"]
        
        # Process each parallel flow pair
        for i in range(min(len(start_nodes), len(end_nodes))):
            # Find the corresponding 'a' step for P_START
            for line_analysis in self.line_analyses:
                if (line_analysis.step_id and 
                    re.match(r'^[BAEF]\d+a$', line_analysis.step_id)):
                    
                    step_id = line_analysis.step_id
                    base_step = re.match(r'^([BAEF]\d+)', step_id).group(1)
                    
                    # Check if this is the i-th parallel group we're processing
                    parallel_steps_before = 0
                    for prev_line in self.line_analyses:
                        if (prev_line.step_id and 
                            re.match(r'^[BAEF]\d+a$', prev_line.step_id) and
                            prev_line.step_id < step_id):  # Before current step
                            parallel_steps_before += 1
                    
                    if parallel_steps_before == i:  # This is the i-th parallel group
                        # Add P{n}_START before this step
                        if step_id not in parallel_events:
                            parallel_events[step_id] = []
                        
                        start_node = [
                            f"P{parallel_counter}_START", 
                            f"Parallel flow distribution for {step_id}",
                            f"P{parallel_counter}_START", 
                            "ControlFlowNode", 
                            "<<distribution>>", 
                            f"Distribution node for parallel steps {step_id}",
                            "", "", "", "", "", ""
                        ]
                        parallel_events[step_id].append(('before', start_node))
                        
                        # Find the corresponding end step using logic from parallel flow detection
                        # Based on the simple parallel detection logic, look for the step where this parallel flow ends
                        base_number = int(re.match(r'^[BAEF](\d+)', step_id).group(1))
                        flow_type = step_id[0]  # B, A, E, or F
                        
                        # Look for the next step that would end this parallel flow:
                        # 1. Next step with no letter suffix in same base number range
                        # 2. Or next step with 'a' suffix that starts a new parallel flow
                        end_step_candidates = []
                        
                        for end_line in self.line_analyses:
                            if not end_line.step_id or not end_line.step_id.startswith(flow_type):
                                continue
                            
                            end_match = re.match(rf'^{flow_type}(\d+)([a-z]?)$', end_line.step_id)
                            if end_match:
                                end_number = int(end_match.group(1))
                                end_suffix = end_match.group(2)
                                
                                # Candidate 1: Next step with no suffix (e.g., B3 after B2a)
                                if end_number == base_number + 1 and not end_suffix:
                                    end_step_candidates.append((end_line.step_id, 1))  # Priority 1
                                
                                # Candidate 2: Next step with 'a' suffix that starts new parallel (e.g., B5a after B4a)
                                elif end_number > base_number and end_suffix == 'a':
                                    end_step_candidates.append((end_line.step_id, 2))  # Priority 2
                        
                        # Use the best candidate (lowest priority number)
                        if end_step_candidates:
                            end_step_candidates.sort(key=lambda x: x[1])  # Sort by priority
                            end_step_id = end_step_candidates[0][0]
                            
                            if end_step_id not in parallel_events:
                                parallel_events[end_step_id] = []
                            
                            # Add P{n}_END before this step
                            end_node = [
                                f"P{parallel_counter}_END", 
                                f"Parallel flow merge for {step_id}",
                                f"P{parallel_counter}_END", 
                                "ControlFlowNode", 
                                "<<merge>>", 
                                f"Merge node for parallel steps {step_id}",
                                "", "", "", "", "", ""
                            ]
                            parallel_events[end_step_id].append(('before', end_node))
                        
                        parallel_counter += 1
                        break
        
        # Now generate CSV rows in order, inserting parallel nodes at correct positions
        csv_rows = []
        
        for line_analysis in self.line_analyses:
            if not line_analysis.ra_classes:
                continue
                
            step_id = line_analysis.step_id or ""
            step_text = line_analysis.line_text
            
            # Insert parallel nodes before this step if needed
            if step_id in parallel_events:
                for position, parallel_node_row in parallel_events[step_id]:
                    if position == 'before':
                        csv_rows.append(parallel_node_row)
            
            # Get flows for this step
            step_control_flows = line_analysis.control_flows
            step_data_flows = line_analysis.data_flows
            
            for ra_class in line_analysis.ra_classes:
                # Find relevant flows for this RA class
                relevant_control_flows = [cf for cf in step_control_flows if cf.target_controller == ra_class.name]
                relevant_data_flows = [df for df in step_data_flows if df.controller == ra_class.name]
                
                if relevant_control_flows or relevant_data_flows:
                    # One row per flow relationship
                    for cf in relevant_control_flows:
                        csv_rows.append([
                            step_id, step_text, ra_class.name, ra_class.ra_type.value,
                            ra_class.stereotype, ra_class.description,
                            cf.source_controller, cf.flow_type, cf.rule,
                            "", "", ""  # No data flow info for this row
                        ])
                    
                    for df in relevant_data_flows:
                        csv_rows.append([
                            step_id, step_text, ra_class.name, ra_class.ra_type.value,
                            ra_class.stereotype, ra_class.description,
                            "", "", "",  # No control flow info for this row
                            df.entity, df.flow_type, df.description
                        ])
                    
                    # If no flows, still create one row for the RA class
                    if not relevant_control_flows and not relevant_data_flows:
                        csv_rows.append([
                            step_id, step_text, ra_class.name, ra_class.ra_type.value,
                            ra_class.stereotype, ra_class.description,
                            "", "", "", "", "", ""
                        ])
                else:
                    # No flows - just the RA class
                    csv_rows.append([
                        step_id, step_text, ra_class.name, ra_class.ra_type.value,
                        ra_class.stereotype, ra_class.description,
                        "", "", "", "", "", ""
                    ])
        
        # Write CSV file
        headers = [
            'UC_Schritt', 'Schritt_Text', 'RA_Klasse', 'RA_Typ', 'Stereotype', 'Beschreibung',
            'Control_Flow_Source', 'Control_Flow_Type', 'Control_Flow_Rule',
            'Data_Flow_Entity', 'Data_Flow_Type', 'Data_Flow_Description'
        ]
        
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(headers)
            writer.writerows(csv_rows)
        
        print(f"[CSV] Step analysis saved to: {csv_file_path}")
        return csv_file_path

    def _create_parallel_nodes_simple(self, step_id: str) -> List[ParallelFlowNode]:
        """
        Simple parallel flow detection:
        - Start when step ends with 'a' (e.g., B2a) -> P1_START
        - End when step has no letter suffix (e.g., B3) or starts new 'a' -> P1_END
        """
        parallel_nodes = []
        
        # Check if step ends with 'a' (parallel start)
        if re.match(r'^[BAEF]\d+a$', step_id):
            # Start of parallel flow
            if self.in_parallel_flow:
                # End previous parallel flow first
                end_node = ParallelFlowNode(
                    node_id=f"P{self.parallel_counter}_END",
                    node_type="merge",
                    step_range="",
                    parallel_steps=[],
                    description=f"Merge node for parallel flow P{self.parallel_counter}"
                )
                parallel_nodes.append(end_node)
                self.parallel_flow_nodes.append(end_node)
                print(f"[PARALLEL SIMPLE] Ended P{self.parallel_counter} at {step_id} (new start)")
            
            # Start new parallel flow
            self.parallel_counter += 1
            self.in_parallel_flow = True
            
            start_node = ParallelFlowNode(
                node_id=f"P{self.parallel_counter}_START",
                node_type="distribution",
                step_range="",
                parallel_steps=[],
                description=f"Distribution node for parallel flow P{self.parallel_counter}"
            )
            parallel_nodes.append(start_node)
            self.parallel_flow_nodes.append(start_node)
            
            print(f"[PARALLEL SIMPLE] Started P{self.parallel_counter} at {step_id}")
        
        # Check if step has no letter suffix (parallel end)
        elif re.match(r'^[BAEF]\d+$', step_id) and self.in_parallel_flow:
            # End of parallel flow
            end_node = ParallelFlowNode(
                node_id=f"P{self.parallel_counter}_END",
                node_type="merge",
                step_range="",
                parallel_steps=[],
                description=f"Merge node for parallel flow P{self.parallel_counter}"
            )
            parallel_nodes.append(end_node)
            self.parallel_flow_nodes.append(end_node)
            
            print(f"[PARALLEL SIMPLE] Ended P{self.parallel_counter} at {step_id}")
            self.in_parallel_flow = False
        
        return parallel_nodes
    
    def _deduplicate_ra_classes(self, ra_classes: List[RAClass]) -> List[RAClass]:
        """Remove duplicate RA classes by name, keeping the first occurrence"""
        seen_names = set()
        unique_classes = []
        
        for ra_class in ra_classes:
            if ra_class.name not in seen_names:
                seen_names.add(ra_class.name)
                unique_classes.append(ra_class)
        
        return unique_classes
    
    def _find_parallel_steps_for_counter(self, counter: int) -> List[str]:
        """Find parallel steps for a given counter (P1 -> B2x, P2 -> B3x, etc.)"""
        # Map counter to base step number
        base_step_number = counter + 1  # P1 -> B2, P2 -> B3, etc.
        base_pattern = f"B{base_step_number}"
        
        parallel_steps = []
        for line_analysis in self.line_analyses:
            if line_analysis.step_id and line_analysis.step_id.startswith(base_pattern):
                # Check if it has a letter suffix (B2a, B2b, etc.)
                if re.match(rf'^{re.escape(base_pattern)}[a-z]$', line_analysis.step_id):
                    parallel_steps.append(line_analysis.step_id)
        
        parallel_steps.sort()  # B2a, B2b, B2c, B2d
        return parallel_steps

    def _generate_parallel_control_flows(self) -> List[dict]:
        """
        Generate control flows that include parallel nodes:
        B1 → P1_START → B2a/B2b → P1_END → B3
        """
        parallel_control_flows = []
        
        print(f"[DEBUG] _generate_parallel_control_flows called")
        print(f"[DEBUG] parallel_flow_nodes count: {len(self.parallel_flow_nodes)}")
        for node in self.parallel_flow_nodes:
            print(f"[DEBUG] Node: {node.node_id} ({node.node_type})")
        
        # Process each parallel flow pair
        parallel_groups = {}  # Group by counter: {1: (start_node, end_node), ...}
        
        for node in self.parallel_flow_nodes:
            if node.node_type == "distribution":
                # Extract counter from P1_START -> 1
                counter_match = re.match(r'P(\d+)_START', node.node_id)
                if counter_match:
                    counter = int(counter_match.group(1))
                    if counter not in parallel_groups:
                        parallel_groups[counter] = {'start': None, 'end': None}
                    parallel_groups[counter]['start'] = node
            elif node.node_type == "merge":
                # Extract counter from P1_END -> 1
                counter_match = re.match(r'P(\d+)_END', node.node_id)
                if counter_match:
                    counter = int(counter_match.group(1))
                    if counter not in parallel_groups:
                        parallel_groups[counter] = {'start': None, 'end': None}
                    parallel_groups[counter]['end'] = node
        
        # For each parallel group, create the control flows
        for counter, nodes in parallel_groups.items():
            start_node = nodes['start']
            end_node = nodes['end']
            
            if not start_node or not end_node:
                continue
                
            # Find the steps involved in this parallel flow
            # Dynamically find parallel steps for this counter
            parallel_steps = self._find_parallel_steps_for_counter(counter)
            print(f"[DEBUG] Counter {counter}: Found parallel steps: {parallel_steps}")
            
            if parallel_steps:
                # Find the step before this parallel group
                # Extract base number from first parallel step (B2a -> 2)
                base_number = int(re.match(r'B(\d+)', parallel_steps[0]).group(1))
                prev_step = f"B{base_number - 1}"
                if counter > 1:  # For P2+, the previous step might be an 'a' step
                    prev_step_candidates = [f"B{base_number - 1}", f"B{base_number - 1}a"]
                else:
                    prev_step_candidates = [f"B{base_number - 1}"]
                
                # Find the step after this parallel group
                next_step_candidates = [f"B{base_number + 1}", f"B{base_number + 1}a"]
                
                # Find actual controllers for these steps
                prev_controller = None
                parallel_controllers = []
                next_controller = None
                
                for line_analysis in self.line_analyses:
                    if line_analysis.step_id in prev_step_candidates:
                        for ra_class in line_analysis.ra_classes:
                            if ra_class.ra_type == RAType.CONTROLLER:
                                prev_controller = ra_class.name
                                break
                    
                    if line_analysis.step_id in parallel_steps:
                        # This is a parallel step (B2a, B2b, etc.)
                        for ra_class in line_analysis.ra_classes:
                            if ra_class.ra_type == RAType.CONTROLLER and ra_class.name not in parallel_controllers:
                                parallel_controllers.append(ra_class.name)
                    
                    if line_analysis.step_id in next_step_candidates:
                        for ra_class in line_analysis.ra_classes:
                            if ra_class.ra_type == RAType.CONTROLLER:
                                next_controller = ra_class.name
                                break
                
                # Create control flows: prev → P_START → parallel_steps → P_END → next
                if prev_controller:
                    parallel_control_flows.append({
                        'source': prev_controller,
                        'destination': start_node.node_id,
                        'source_step': prev_step_candidates[0],
                        'target_step': start_node.node_id,
                        'rule': 'Rule 4 - Parallel Flow Distribution',
                        'description': f'{prev_controller} flows to {start_node.node_id}'
                    })
                
                # P_START → each parallel controller
                for parallel_controller in parallel_controllers:
                    parallel_control_flows.append({
                        'source': start_node.node_id,
                        'destination': parallel_controller,
                        'source_step': start_node.node_id,
                        'target_step': f"B{base_number}",
                        'rule': 'Rule 4 - Parallel Flow Distribution',
                        'description': f'{start_node.node_id} distributes to {parallel_controller}'
                    })
                
                # Each parallel controller → P_END
                for parallel_controller in parallel_controllers:
                    parallel_control_flows.append({
                        'source': parallel_controller,
                        'destination': end_node.node_id,
                        'source_step': f"B{base_number}",
                        'target_step': end_node.node_id,
                        'rule': 'Rule 4 - Parallel Flow Merge',
                        'description': f'{parallel_controller} flows to {end_node.node_id}'
                    })
                
                # P_END → next controller
                if next_controller:
                    parallel_control_flows.append({
                        'source': end_node.node_id,
                        'destination': next_controller,
                        'source_step': end_node.node_id,
                        'target_step': next_step_candidates[0],
                        'rule': 'Rule 4 - Parallel Flow Merge',
                        'description': f'{end_node.node_id} flows to {next_controller}'
                    })
        
        return parallel_control_flows

    def _check_parallel_flow_pattern(self, step_id: str, line_number: int) -> List[ParallelFlowNode]:
        """
        Check if current step is part of a parallel flow pattern during line analysis
        
        This method looks ahead in the step sequence to detect parallel patterns
        and creates distribution/merge nodes for the first step of each parallel group.
        """
        parallel_nodes = []
        
        # Extract base step and suffix (e.g., B2a -> base=B2, suffix=a)
        match = re.match(r'^([BAEF]\d+)([a-z]?)$', step_id)
        if not match:
            return parallel_nodes
            
        base_step = match.group(1)
        suffix = match.group(2)
        
        # Only process first step of potential parallel group (suffix 'a')
        if suffix != 'a':
            return parallel_nodes
        
        # Look ahead to find parallel steps by checking already processed steps and upcoming steps
        parallel_steps = self._find_parallel_steps_for_base(base_step, line_number)
        
        if len(parallel_steps) > 1:
            print(f"[PARALLEL INLINE] Found parallel group {base_step}: {parallel_steps}")
            
            # Create distribution node for this parallel group
            dist_node = ParallelFlowNode(
                node_id=f"{base_step}_DIST",
                node_type="distribution", 
                step_range=f"{parallel_steps[0]}-{parallel_steps[-1]}",
                parallel_steps=parallel_steps.copy(),
                description=f"Distribution node for parallel steps {', '.join(parallel_steps)}"
            )
            parallel_nodes.append(dist_node)
            
            # Create merge node for this parallel group
            merge_node = ParallelFlowNode(
                node_id=f"{base_step}_MERGE",
                node_type="merge",
                step_range=f"{parallel_steps[0]}-{parallel_steps[-1]}",
                parallel_steps=parallel_steps.copy(), 
                description=f"Merge node for parallel steps {', '.join(parallel_steps)}"
            )
            parallel_nodes.append(merge_node)
            
            # Store in analyzer state for global access
            self.parallel_flow_nodes.extend(parallel_nodes)
        
        return parallel_nodes
    
    def _find_parallel_steps_for_base(self, base_step: str, current_line_number: int) -> List[str]:
        """
        Find all parallel steps for a given base step by looking ahead in the file
        """
        parallel_steps = []
        
        # First, add the current step itself (e.g., B2a)
        current_step = base_step + "a"
        parallel_steps.append(current_step)
        
        # Look through already processed lines for this base step
        for line_analysis in self.line_analyses:
            if line_analysis.step_id and line_analysis.step_id.startswith(base_step):
                match = re.match(r'^([BAEF]\d+)([a-z])$', line_analysis.step_id)
                if match and line_analysis.step_id not in parallel_steps:
                    parallel_steps.append(line_analysis.step_id)
        
        # Also check the current step sequence we're building
        for step_id in self.step_sequence:
            if step_id.startswith(base_step):
                match = re.match(r'^([BAEF]\d+)([a-z])$', step_id)
                if match:
                    if step_id not in parallel_steps:
                        parallel_steps.append(step_id)
        
        # Look ahead in the file by examining subsequent lines
        for i in range(current_line_number, min(current_line_number + 20, len(self.all_lines))):
            line = self.all_lines[i]
            # Extract step ID from line using simple regex
            step_match = re.search(r'\b([BAEF]\d+[a-z]?(?:\.\d+)?)\b', line)
            if step_match:
                lookahead_step_id = step_match.group(1)
                if lookahead_step_id.startswith(base_step):
                    match = re.match(r'^([BAEF]\d+)([a-z])$', lookahead_step_id)
                    if match and lookahead_step_id not in parallel_steps:
                        parallel_steps.append(lookahead_step_id)
        
        # Sort the found parallel steps
        parallel_steps.sort()
        
        # Verify they are consecutive (a, b, c, etc.)
        if len(parallel_steps) > 1:
            verified_steps = []
            expected_suffix = ord('a')
            for step in parallel_steps:
                match = re.match(r'^([BAEF]\d+)([a-z])$', step)
                if match and ord(match.group(2)) == expected_suffix:
                    verified_steps.append(step)
                    expected_suffix += 1
                else:
                    break
            return verified_steps
        
        return parallel_steps

    def _detect_parallel_flows(self):
        """
        Detect parallel flows in UC steps and create distribution/merge nodes
        
        Parallel flows are detected by:
        1. Sequential steps with same base ID but different suffixes (B2a, B2b)
        2. Create distribution node before first parallel step
        3. Create merge node after last parallel step
        """
        print("[PARALLEL] Detecting parallel flows...")
        
        # Group steps by base ID (e.g., B2a, B2b -> B2)
        parallel_groups = self._group_parallel_steps()
        
        for base_step, parallel_steps in parallel_groups.items():
            if len(parallel_steps) > 1:
                print(f"[PARALLEL] Found parallel group {base_step}: {parallel_steps}")
                
                # Create distribution node
                dist_node = ParallelFlowNode(
                    node_id=f"{base_step}_DIST",
                    node_type="distribution",
                    step_range=f"{parallel_steps[0]}-{parallel_steps[-1]}",
                    parallel_steps=parallel_steps.copy(),
                    description=f"Distribution node for parallel steps {', '.join(parallel_steps)}"
                )
                self.parallel_flow_nodes.append(dist_node)
                
                # Create merge node
                merge_node = ParallelFlowNode(
                    node_id=f"{base_step}_MERGE",
                    node_type="merge",
                    step_range=f"{parallel_steps[0]}-{parallel_steps[-1]}",
                    parallel_steps=parallel_steps.copy(),
                    description=f"Merge node for parallel steps {', '.join(parallel_steps)}"
                )
                self.parallel_flow_nodes.append(merge_node)
                
                # Add parallel nodes to the first parallel step's analysis
                for line_analysis in self.line_analyses:
                    if line_analysis.step_id == parallel_steps[0]:
                        line_analysis.parallel_nodes.extend([dist_node, merge_node])
                        break
        
        print(f"[PARALLEL] Created {len(self.parallel_flow_nodes)} parallel flow nodes")
    
    def _group_parallel_steps(self) -> Dict[str, List[str]]:
        """
        Group parallel steps by their base ID
        
        Returns:
            Dict mapping base step (e.g., "B2") to list of parallel steps (e.g., ["B2a", "B2b"])
        """
        step_groups = {}
        
        for step_id in self.step_sequence:
            # Extract base step ID using regex
            match = re.match(r'^([BAEF]\d+)[a-z]?', step_id)
            if match:
                base_step = match.group(1)
                if base_step not in step_groups:
                    step_groups[base_step] = []
                step_groups[base_step].append(step_id)
        
        # Only return groups with multiple steps (parallel)
        parallel_groups = {base: steps for base, steps in step_groups.items() 
                          if len(steps) > 1 and self._are_steps_parallel(steps)}
        
        return parallel_groups
    
    def _are_steps_parallel(self, steps: List[str]) -> bool:
        """
        Check if steps are truly parallel (consecutive with letter suffixes)
        
        Args:
            steps: List of step IDs to check
            
        Returns:
            True if steps are parallel (e.g., B2a, B2b, B2c)
        """
        if len(steps) < 2:
            return False
        
        # Check if all steps have letter suffixes and are consecutive
        suffixes = []
        for step in steps:
            match = re.match(r'^[BAEF]\d+([a-z])$', step)
            if match:
                suffixes.append(match.group(1))
            else:
                return False  # No letter suffix, not parallel
        
        # Check if suffixes are consecutive letters
        suffixes.sort()
        expected_suffix = ord('a')
        for suffix in suffixes:
            if ord(suffix) != expected_suffix:
                return False
            expected_suffix += 1
        
        return True

    def _generate_rup_diagram(self, json_file_path: str) -> str:
        """
        Generate RUP diagrams (both SVG and PNG) from JSON analysis

        Args:
            json_file_path: Path to the JSON analysis file

        Returns:
            Path to generated diagram file
        """
        diagram_path = ""

        try:
            # Generate SVG diagram using SVGRUPVisualizer
            from svg_rup_visualizer import SVGRUPVisualizer
            svg_visualizer = SVGRUPVisualizer()
            svg_diagram_path = svg_visualizer.generate_svg(json_file_path)
            print(f"[SVG RUP] SVG diagram generated: {svg_diagram_path}")
            diagram_path = svg_diagram_path
        except Exception as e:
            print(f"[ERROR] Failed to generate SVG diagram: {e}")
            import traceback
            traceback.print_exc()

        try:
            # Generate PNG diagram using PureRUPVisualizer
            from pure_rup_visualizer import PureRUPVisualizer

            # Create pure RUP visualizer
            visualizer = PureRUPVisualizer(figure_size=(20, 16))

            # Generate diagram from JSON directly to new directory
            png_diagram_path = visualizer.generate_diagram(json_file_path)
            if not diagram_path:
                diagram_path = png_diagram_path

        except Exception as e:
            print(f"[ERROR] Failed to generate PNG diagram: {e}")
            import traceback
            traceback.print_exc()

        return diagram_path

def main():
    """Test the structured analyzer"""
    if len(sys.argv) > 1:
        uc_file = sys.argv[1]
        # Auto-detect domain based on UC filename or use beverage_preparation as default
        if len(sys.argv) > 2:
            domain = sys.argv[2]
        elif 'UC3_Rocket' in uc_file or 'rocket' in uc_file.lower():
            domain = 'rocket_science'
        elif 'UC4_Nuclear' in uc_file or 'nuclear' in uc_file.lower():
            domain = 'nuclear'
        elif 'UC5_Robot' in uc_file or 'robot' in uc_file.lower():
            domain = 'robotics'
        else:
            domain = DEFAULT_DOMAIN  # Default for UC1, UC2
    else:
        uc_file = 'Use Case/UC3_Rocket_Launch_Improved.txt'
        domain = 'rocket_science'
    
    if not os.path.exists(uc_file):
        print(f"[ERROR] UC file '{uc_file}' not found")
        return
    
    try:
        analyzer = StructuredUCAnalyzer(domain_name=domain)
        line_analyses, json_output = analyzer.analyze_uc_file(uc_file)
        
        # Debug output
        print(f"\n[DEBUG] Sample line analyses:")
        step_lines = [la for la in line_analyses if la.step_id]
        print(f"Found {len(step_lines)} step lines")
        
        # Show B1 specifically first (to answer user's question about B1 context)
        b1_analysis = next((la for la in line_analyses if la.step_id == "B1"), None)
        if b1_analysis:
            print(f"  B1 CONTEXT ANALYSIS:")
            print(f"    Line {b1_analysis.line_number}: {b1_analysis.line_type.value} | Step: {b1_analysis.step_id}")
            print(f"    Text: {b1_analysis.line_text}")
            if b1_analysis.step_context:
                print(f"    CONTEXT:")
                print(f"      Step Type: {b1_analysis.step_context.step_type}")
                print(f"      Phase: {b1_analysis.step_context.phase}")
                print(f"      Business Context: {b1_analysis.step_context.business_context}")
                print(f"      Technical Context: {b1_analysis.step_context.technical_context}")
                if b1_analysis.step_context.actors_involved:
                    print(f"      Actors Involved: {', '.join(b1_analysis.step_context.actors_involved)}")
                else:
                    print(f"      Actors Involved: None (system-triggered event)")
            print()
        
        for i, la in enumerate(line_analyses[17:22]):  # Show B2a-B5 area
            if la.step_id:
                print(f"  Line {la.line_number}: {la.line_type.value} | Step: {la.step_id}")
                print(f"    Text: {la.line_text}")
                print(f"    RA Classes: {len(la.ra_classes)}")
                for ra in la.ra_classes:
                    print(f"      - {ra.ra_type.value}: {ra.name}")
                if la.grammatical.main_verb:
                    print(f"    Verb: {la.grammatical.main_verb} ({la.grammatical.verb_type})")
                    print(f"    Compound Nouns: {la.grammatical.compound_nouns}")
                    print(f"    Prep Objects: {la.grammatical.prepositional_objects}")
                
                # Show step context information
                if la.step_context:
                    print(f"    CONTEXT:")
                    print(f"      Step Type: {la.step_context.step_type}")
                    print(f"      Phase: {la.step_context.phase}")
                    print(f"      Business Context: {la.step_context.business_context}")
                    print(f"      Technical Context: {la.step_context.technical_context}")
                    if la.step_context.actors_involved:
                        print(f"      Actors Involved: {', '.join(la.step_context.actors_involved)}")
                
                print(f"    Data Flows: {len(la.data_flows)}")
                for df in la.data_flows:
                    print(f"      - {df.flow_type}: {df.controller} -> {df.entity} (prep: {df.preposition})")
                print(f"    Control Flows: {len(la.control_flows)}")
                for cf in la.control_flows:
                    print(f"      - {cf.source_step} -> {cf.target_step}: {cf.source_controller} -> {cf.target_controller}")
                print()
        
        print(f"\n[RESULTS] Analysis completed:")
        print(f"   - Lines processed: {len(line_analyses)}")
        print(f"   - Steps found: {len([la for la in line_analyses if la.step_id])}")
        total_ra_classes = sum(len(la.ra_classes) for la in line_analyses)
        print(f"   - Total RA classes: {total_ra_classes}")
        print(f"   - Output saved: {json_output}")
        
    except Exception as e:
        print(f"[ERROR] Error during analysis: {e}")
        import traceback
        traceback.print_exc()


def analyze_from_config(config_file: str = "uc_analysis_config.json") -> Dict[str, Any]:
    """
    Analyze multiple use cases based on configuration file.

    Args:
        config_file: Path to configuration JSON file

    Returns:
        Dictionary with analysis results for all use cases
    """
    print("="*70)
    print("MULTI-UC ANALYSIS FROM CONFIG")
    print("="*70)

    # Load configuration
    if not Path(config_file).exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(f"\nAnalysis: {config.get('analysis_name', 'Unnamed Analysis')}")
    print(f"Description: {config.get('description', 'No description')}")
    print(f"Domain: {config.get('domain', 'unknown')}")

    # Get enabled use cases
    use_cases = [uc for uc in config.get('use_cases', []) if uc.get('enabled', False)]
    print(f"\nEnabled Use Cases: {len(use_cases)}")
    for uc in use_cases:
        print(f"  - {uc['id']}: {uc['name']} ({uc['file']})")

    # Create output directory
    output_dir = config.get('output', {}).get('directory', 'new/multi_uc')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nOutput Directory: {output_dir}")

    # Initialize shared analyzer (shares material controller registry across UCs)
    domain_name = config.get('domain', DEFAULT_DOMAIN)
    analyzer = StructuredUCAnalyzer(domain_name=domain_name)

    # Results storage
    results = {
        'config': config,
        'analyses': {},
        'controller_registry': {},
        'summary': {
            'total_use_cases': len(use_cases),
            'completed': 0,
            'failed': 0
        }
    }

    # Analyze each use case
    for uc_config in use_cases:
        uc_id = uc_config['id']
        uc_file = uc_config['file']
        uc_name = uc_config.get('name', uc_id)

        print(f"\n{'='*70}")
        print(f"ANALYZING {uc_id}: {uc_name}")
        print(f"{'='*70}")

        try:
            # Analyze UC file
            line_analyses, json_output = analyzer.analyze_uc_file(uc_file)

            # Move generated files to output directory
            if json_output:
                # Copy JSON to output directory
                src_json = Path(json_output)
                if src_json.exists():
                    dest_json = Path(output_dir) / f"{uc_id}_Structured_RA_Analysis.json"
                    import shutil
                    shutil.copy(src_json, dest_json)
                    print(f"[MOVED] JSON: {dest_json}")

                    # Copy CSV if exists
                    csv_file = src_json.with_name(src_json.name.replace('_RA_Analysis.json', '_UC_Steps_RA_Classes.csv'))
                    if csv_file.exists():
                        dest_csv = Path(output_dir) / f"{uc_id}_Structured_UC_Steps_RA_Classes.csv"
                        shutil.copy(csv_file, dest_csv)
                        print(f"[MOVED] CSV: {dest_csv}")

                    # Copy SVG if exists
                    svg_file = src_json.with_name(src_json.name.replace('_RA_Analysis.json', '_SVG_RA_Diagram.svg'))
                    if svg_file.exists():
                        dest_svg = Path(output_dir) / f"{uc_id}_Structured_SVG_RA_Diagram.svg"
                        shutil.copy(svg_file, dest_svg)
                        print(f"[MOVED] SVG: {dest_svg}")

                    # Copy PNG if exists
                    png_file = src_json.with_name(src_json.name.replace('_RA_Analysis.json', '_Pure_RA_Diagram.png'))
                    if png_file.exists():
                        dest_png = Path(output_dir) / f"{uc_id}_Structured_Pure_RA_Diagram.png"
                        shutil.copy(png_file, dest_png)
                        print(f"[MOVED] PNG: {dest_png}")

            # Store results
            results['analyses'][uc_id] = {
                'name': uc_name,
                'file': uc_file,
                'status': 'completed',
                'json_output': str(dest_json) if json_output else None,
                'total_lines': len(line_analyses),
                'total_steps': len([la for la in line_analyses if la.step_id])
            }

            results['summary']['completed'] += 1
            print(f"[SUCCESS] {uc_id} analysis completed")

        except Exception as e:
            print(f"[ERROR] Failed to analyze {uc_id}: {e}")
            import traceback
            traceback.print_exc()

            results['analyses'][uc_id] = {
                'name': uc_name,
                'file': uc_file,
                'status': 'failed',
                'error': str(e)
            }
            results['summary']['failed'] += 1

    # Export shared controller registry
    if config.get('options', {}).get('share_controllers', True):
        controllers = analyzer.controller_registry.get_all_controllers()
        results['controller_registry'] = {
            'total_controllers': len(controllers),
            'controllers': [
                {
                    'name': c.name,
                    'material': c.material,
                    'aggregation_state': c.aggregation_state,
                    'functions': sorted(list(c.functions))
                }
                for c in controllers
            ]
        }

    # Save summary report
    summary_file = Path(output_dir) / "multi_uc_analysis_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("MULTI-UC ANALYSIS COMPLETED")
    print(f"{'='*70}")
    print(f"Completed: {results['summary']['completed']}/{results['summary']['total_use_cases']}")
    print(f"Failed: {results['summary']['failed']}/{results['summary']['total_use_cases']}")
    print(f"Summary saved: {summary_file}")

    if results['controller_registry']:
        print(f"\nShared Material Controllers: {results['controller_registry']['total_controllers']}")
        for ctrl in results['controller_registry']['controllers']:
            print(f"  - {ctrl['name']} ({ctrl['material']} {ctrl['aggregation_state'] or 'N/A'})")
            print(f"    Functions: {', '.join(ctrl['functions'])}")

    return results


def main():
    """Main function - supports both config-based and single-file analysis"""
    import argparse

    parser = argparse.ArgumentParser(description='Structured UC Analyzer')
    parser.add_argument('--config', type=str, help='Path to config file for multi-UC analysis')
    parser.add_argument('--uc-file', type=str, help='Single UC file to analyze')
    parser.add_argument('--domain', type=str, default=DEFAULT_DOMAIN, help='Domain name')

    args = parser.parse_args()

    try:
        # Config-based multi-UC analysis
        if args.config:
            results = analyze_from_config(args.config)
            return

        # Single UC analysis
        uc_file = args.uc_file or "Use Case/UC1.txt"
        domain = args.domain

        print("="*70)
        print(f"SINGLE UC ANALYSIS: {uc_file}")
        print("="*70)

        analyzer = StructuredUCAnalyzer(domain_name=domain)
        line_analyses, json_output = analyzer.analyze_uc_file(uc_file)
        print(f"[SUCCESS] Analysis completed. Generated: {json_output}")

        # Generate RA diagrams using both engines
        print("[RUP] Generating RA diagrams...")

        # SVG-basierte Visualisierung mit Wikipedia-Symbolen
        from svg_rup_visualizer import SVGRUPVisualizer
        svg_visualizer = SVGRUPVisualizer()
        svg_diagram_path = svg_visualizer.generate_svg(json_output)
        print(f"[SUCCESS] SVG RA diagram generated: {svg_diagram_path}")

        # Original RUP engine als Backup
        from official_rup_engine import OfficialRUPEngine
        engine = OfficialRUPEngine()
        png_diagram_path = engine.create_official_rup_diagram_from_json(json_output)
        print(f"[SUCCESS] PNG RA diagram generated: {png_diagram_path}")

    except Exception as e:
        print(f"[ERROR] Error during analysis: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()