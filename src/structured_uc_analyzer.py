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
# Pure RUP visualizer is imported dynamically in _generate_rup_diagram()

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
    
    def __init__(self, domain_name: str = "beverage_preparation"):
        self.domain_name = domain_name
        self.verb_loader = DomainVerbLoader()
        self.nlp = None
        self._load_spacy()
        
        # Initialize generative context manager
        self.context_manager = GenerativeContextManager(domain_name)
        
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
        """Get default physical state for materials when context is unclear"""
        defaults = {
            'water': 'liquid',
            'milk': 'liquid', 
            'coffee': 'liquid',  # Default to liquid coffee unless grinding context
            'sugar': 'solid',
            'salt': 'solid',
            'oil': 'liquid',
            'steam': 'gas',
            'air': 'gas'
        }
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
        Apply spell correction to text using domain-specific vocabulary
        
        Args:
            text: Text to correct
            
        Returns:
            Spell-corrected text
        """
        # Get domain-specific vocabulary
        domain_config = self.verb_loader.domain_configs[self.domain_name]
        material_contexts = domain_config.get('material_specific_contexts', {})
        aggregation_states = domain_config.get('aggregation_states', {})
        
        # Build correction vocabulary
        correction_vocabulary = {}
        
        # Add materials and their common variations
        for material_name, context in material_contexts.items():
            # Common misspellings for materials
            if material_name == "coffee":
                correction_vocabulary.update({
                    "coffe": "coffee",
                    "cofee": "coffee", 
                    "cofffe": "coffee",
                    "coffie": "coffee"
                })
            elif material_name == "milk":
                correction_vocabulary.update({
                    "melk": "milk",
                    "milc": "milk"
                })
            elif material_name == "water":
                correction_vocabulary.update({
                    "watter": "water",
                    "watr": "water"
                })
            
            # Add indicators with common misspellings
            solid_indicators = context.get('solid_indicators', [])
            for indicator in solid_indicators:
                if indicator == "beans":
                    correction_vocabulary.update({
                        "beens": "beans",
                        "bens": "beans",
                        "bean": "beans"  # plural correction
                    })
                elif indicator == "grind":
                    correction_vocabulary.update({
                        "grind": "grind",
                        "grinding": "grinding",
                        "grinds": "grind"
                    })
        
        # Add aggregation state keywords
        for state_name, state_config in aggregation_states.items():
            keywords = state_config.get('specific_keywords', [])
            for keyword in keywords:
                if keyword == "powder":
                    correction_vocabulary.update({
                        "powdr": "powder",
                        "poweder": "powder"
                    })
        
        # Apply corrections
        corrected_text = text
        for misspelling, correction in correction_vocabulary.items():
            # Use word boundary matching to avoid partial replacements
            import re
            pattern = r'\b' + re.escape(misspelling) + r'\b'
            corrected_text = re.sub(pattern, correction, corrected_text, flags=re.IGNORECASE)
        
        # Log corrections if any were made
        if corrected_text != text:
            print(f"[SPELL CORRECTION] '{text}' -> '{corrected_text}'")
        
        return corrected_text
    
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
        analysis.prepositional_objects = self._find_prepositional_objects(main_verb_token) if main_verb_token else []
        
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
            
            # Pattern: Noun + "defined" + Noun (like "user defined time")
            # Priority: Use the SECOND noun as the main entity, not the first
            if (i + 2 < len(doc) and 
                token.pos_ == "NOUN" and 
                doc[i + 1].lemma_.lower() == "define" and 
                doc[i + 2].pos_ == "NOUN"):
                # For "user defined time", we want "time" as the main entity
                main_entity = doc[i + 2].text  # "time"
                modifier = token.text  # "user"
                compound_with_priority = f"{modifier} {doc[i + 1].text} {main_entity}"
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
        # Priority 1: Look for compound phrases containing domain-relevant terms
        domain_terms = ["time", "temperature", "pressure", "amount", "level", "degree", "coffee", "water", "milk"]
        for token, expanded in direct_objects:
            if any(term in expanded.lower() for term in domain_terms):
                # Clean up compound terms: extract the core noun from phrases like "user defined time"
                expanded_lower = expanded.lower()
                for term in domain_terms:
                    if term in expanded_lower:
                        # If we find a domain term, use just that term (e.g., "time" from "user defined time")
                        return term
                return expanded
        
        # Priority 2: Choose the rightmost/last direct object (usually the main object)
        # In "user defined time", "time" comes after "user"
        rightmost_obj = max(direct_objects, key=lambda x: x[0].i)  # x[0].i is token position
        return rightmost_obj[1]
    
    def _find_prepositional_objects(self, verb_token) -> List[Tuple[str, str]]:
        """Find prepositional objects (preposition, object)"""
        if not verb_token:
            return []
        
        prep_objects = []
        for child in verb_token.children:
            if child.dep_ == "prep":
                prep = child.text
                # Find the object of the preposition
                for prep_child in child.children:
                    if prep_child.dep_ == "pobj":
                        obj = self._expand_noun_phrase(prep_child)
                        prep_objects.append((prep, obj))
        
        return prep_objects
    
    def _expand_noun_phrase(self, token) -> str:
        """Expand a token to its full noun phrase"""
        # Get the noun chunk that contains this token
        for chunk in token.doc.noun_chunks:
            if token in chunk:
                return chunk.text
        return token.text
    
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
        
        # 3. Generate Controller (always for steps with verbs) - now with context
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
        
        return ra_classes
    
    def _generate_controller_for_step(self, step_id: str, grammatical: GrammaticalAnalysis, step_context: StepContext = None, line_text: str = "") -> Optional[RAClass]:
        """Generate Controller using domain-agnostic approach from generic_uc_analyzer.py"""
        if not grammatical.main_verb:
            return None
        
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
            'suggested_functional_activity': None  # We don't use this in structured analyzer
        })()
        
        # Use the proven generic controller naming logic
        controller_name = self._derive_generic_controller_name(verb_analysis, step_info)
        
        if not controller_name:
            # Fallback to verb-based naming
            controller_name = f"{grammatical.main_verb.capitalize()}Manager"
        
        # Generate description
        description = f"Manages {controller_name.replace('Manager', '').lower()} operations in {step_id}"
        
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
    
    def _determine_flow_type(self, step_id: str) -> str:
        """Determine flow type from step_id"""
        if step_id.startswith('A'):
            return "alternative"
        elif step_id.startswith('E'):
            return "extension"
        else:
            return "main"
    
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
    
    def _derive_generic_controller_name(self, verb_analysis, step) -> Optional[str]:
        """Generate domain-agnostic controller names (from generic_uc_analyzer.py)"""
        # Pattern-based controller naming
        if step.step_id == "B1" or self._is_trigger_step(step):
            if "time" in verb_analysis.original_text.lower() or "clock" in verb_analysis.original_text.lower():
                return "TimeManager"
            elif "user" in verb_analysis.original_text.lower():
                return "UserRequestManager"
            elif step.flow_type == "alternative":
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
        
        # Specific domain mappings for expected controllers 
        
        # Water-related operations
        if ("water" in text_lower or "heater" in text_lower) and ("activate" in text_lower or "heat" in text_lower):
            return "WaterManager"
        
        # Filter-related operations  
        if "filter" in text_lower and ("prepare" in text_lower or "ready" in text_lower):
            return "FilterManager"
            
        # Cup/Container operations
        if "cup" in text_lower and ("retrieve" in text_lower or "place" in text_lower or "present" in text_lower):
            if "present" in text_lower and "user" in text_lower:
                return "UserManager"  # B5: present to user
            return "CupManager"
            
        # Message/Communication operations
        if "message" in text_lower and ("output" in text_lower or "display" in text_lower):
            return "MessageManager"
        
        # Object-based controller naming (check for implementation elements first)
        if verb_analysis.direct_object:
            # Check if direct object contains implementation elements
            obj_words = verb_analysis.direct_object.lower().split()
            for word in obj_words:
                impl_info = self.verb_loader.get_implementation_element_info(word, self.domain_name)
                if impl_info:
                    # Use verb-based naming instead for implementation elements
                    verb_action = verb_analysis.verb_lemma.capitalize()
                    # Try to extract the actual target from context
                    if "water" in verb_analysis.direct_object.lower():
                        return f"Water{verb_action}ingManager"
                    elif "coffee" in verb_analysis.direct_object.lower():
                        return f"Coffee{verb_action}ingManager"
                    else:
                        return f"{verb_action}Manager"
            
            # No implementation elements - use object-based naming
            main_object = verb_analysis.direct_object.split()[-1].capitalize()
            return f"{main_object}Manager"
        
        # Verb-based controller naming
        verb_action = verb_analysis.verb_lemma.capitalize()
        return f"{verb_action}Manager"
    
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
        """Generate Entities from direct object, prepositional objects, and compound nouns (spaCy-based)"""
        entities = []
        entity_names = set()
        
        # 1. Direct object (highest priority)
        if grammatical.direct_object:
            entity_name = self._clean_entity_name(grammatical.direct_object)
            # Check if this name is already defined as an Actor or already exists as Entity
            if entity_name and entity_name not in entity_names and not self._is_existing_actor(entity_name):
                entity = self._get_or_create_entity(entity_name, step_id)
                if entity:
                    entity_names.add(entity_name)
                    entities.append(entity)
        
        # 2. Prepositional objects (second priority)
        for prep, obj in grammatical.prepositional_objects:
            entity_name = self._clean_entity_name(obj)
            # Check if this name is already defined as an Actor or already exists as Entity
            if entity_name and entity_name not in entity_names and not self._is_existing_actor(entity_name):
                entity = self._get_or_create_entity(entity_name, step_id)
                if entity:
                    entity_names.add(entity_name)
                    entities.append(entity)
        
        # 3. Compound nouns from spaCy (third priority - only add if meaningful)
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
            
            # Generate semantic technical context based on domain knowledge
            semantic_context = self._generate_semantic_technical_context(line_text, global_context, step_id)
            
            if semantic_context:
                return semantic_context
                
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
        """Generate contextual description using LLM semantic analysis with domain knowledge"""
        if not self.nlp or not global_context:
            return line_text
            
        try:
            # Parse line with spaCy for semantic understanding
            doc = self.nlp(line_text)
            
            # Extract semantic components
            main_verb = self._extract_main_verb(doc)
            entities = self._extract_entities_from_line(doc)
            purpose = self._extract_purpose_from_global_context(global_context)
            
            # Get domain knowledge
            domain_config = self.verb_loader.domain_configs.get(self.domain_name, {})
            
            # Generate semantic description using domain knowledge
            contextual_desc = self._generate_semantic_description(
                main_verb, entities, purpose, domain_config, line_text
            )
            
            return contextual_desc
            
        except Exception as e:
            print(f"[DEBUG] Contextual description generation failed: {e}")
            return line_text
    
    def _extract_main_verb(self, doc):
        """Extract the main action verb from spaCy doc"""
        # Use semantic verb detection from domain JSON (NO hardcoded rules!)
        domain_config = self.verb_loader.domain_configs.get(self.domain_name, {})
        semantic_verbs = domain_config.get('semantic_verb_detection', {})
        
        text_lower = doc.text.lower()
        for verb, patterns in semantic_verbs.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return verb
        
        # Then fallback to spaCy detection, but filter out step IDs
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp"]:
                # Ignore tokens that look like step IDs (B2a, A1, etc.)
                if not re.match(r'^[A-Z]\d+[a-z]?$', token.text):
                    return token.lemma_
        
        # Last fallback: look for any verb that's not a step ID
        for token in doc:
            if token.pos_ == "VERB":
                if not re.match(r'^[A-Z]\d+[a-z]?$', token.text):
                    return token.lemma_
            
        return None
    
    def _extract_entities_from_line(self, doc):
        """Extract relevant entities/objects from the line using domain knowledge"""
        entities = []
        
        # Standard spaCy entity extraction
        for token in doc:
            if token.dep_ in ["dobj", "pobj"] or token.pos_ in ["NOUN"]:
                # EXCLUDE 'system' - it's NOT an entity (corrected multiple times!)
                if token.lemma_.lower() != "system":
                    entities.append(token.lemma_.lower())
        
        # Enhanced entity detection using domain knowledge (Betriebsstoffe)
        text_lower = doc.text.lower()
        domain_config = self.verb_loader.domain_configs.get(self.domain_name, {})
        betriebsmittel = domain_config.get('operational_materials_addressing', {}).get('material_types', {})
        
        # Check for Betriebsstoffe (operational materials) from domain JSON
        for material_name, material_config in betriebsmittel.items():
            material_keywords = [material_name.replace('_', ' '), material_name.replace('_', '')]
            for keyword in material_keywords:
                if keyword in text_lower:
                    entities.append(material_name)
                    # Also add the base material name
                    base_name = material_name.replace('_', '')
                    if base_name != material_name:
                        entities.append(base_name)
        
        # Additional semantic entity detection for compound terms
        if "water heater" in text_lower:
            entities.extend(["water", "heater"])
        if "coffee beans" in text_lower or "beans" in text_lower:
            entities.extend(["coffee_beans", "beans"])
        if "coffee" in text_lower:
            entities.append("coffee")
        if "filter" in text_lower:
            entities.append("filter")
        if "grind" in text_lower:
            entities.append("grind")
        if "cup" in text_lower:
            entities.append("cup")
        if "milk" in text_lower:
            entities.append("milk")
        if "water" in text_lower:
            entities.append("water")
        if "sugar" in text_lower:
            entities.append("sugar")
            
        # Remove 'system' and duplicates
        entities = [e for e in entities if e != "system"]
        return list(set(entities))
    
    def _extract_purpose_from_global_context(self, global_context: str) -> str:
        """Extract the main purpose/goal from UC context"""
        if not global_context:
            return ""
        
        context_lower = global_context.lower()
        
        # Use spaCy to understand the purpose semantically
        if self.nlp:
            doc = self.nlp(global_context)
            for token in doc:
                if token.pos_ == "NOUN" and token.dep_ in ["ROOT", "dobj"]:
                    return token.lemma_.lower()
        
        # Use spaCy + domain JSON for purpose extraction
        domain_config = self.verb_loader.domain_configs.get(self.domain_name, {})
        
        # Semantic analysis using spaCy and domain knowledge
        if "coffee" in context_lower or "kaffee" in context_lower:
            if "milk" in context_lower or "milch" in context_lower:
                return "milk_coffee"
            return "coffee_preparation"
        elif "espresso" in context_lower:
            return "espresso"
        
        return "beverage_preparation"
    
    def _generate_semantic_description(self, verb: str, entities: list, purpose: str, domain_config: dict, original_text: str) -> str:
        """Generate semantic description using domain knowledge and LLM understanding"""
        if not verb or not entities:
            return original_text
            
        # Get domain-specific knowledge
        term_knowledge = domain_config.get('term_specific_knowledge', {})
        betriebsmittel = domain_config.get('betriebsmittel_tracking', {}).get('material_types', {})
        
        # Semantic mapping based on verb and entities
        description_parts = []
        
        # Determine the functional purpose
        for entity in entities:
            if entity in ['filter', 'filtration']:
                if purpose in ['kaffeezubereitung', 'milchkaffee', 'espresso', 'milk_coffee', 'coffee_preparation']:
                    description_parts.append(f"Filter für {purpose.capitalize()}")
                    break
            elif entity in ['water', 'heater', 'heating']:
                if 'heat' in verb or 'activate' in verb:
                    knowledge = term_knowledge.get('coffee', [])
                    for fact in knowledge:
                        if 'hot water' in fact.lower():
                            description_parts.append(f"Wassererhitzung für {purpose.capitalize()}")
                            break
            elif entity in ['grind', 'grinding', 'beans']:
                if 'grind' in verb:
                    bean_knowledge = betriebsmittel.get('coffee_beans', {})
                    if bean_knowledge:
                        description_parts.append(f"Kaffeebohnen mahlen für {purpose.capitalize()}")
            elif entity in ['milk']:
                if 'add' in verb:
                    milk_knowledge = term_knowledge.get('milk', [])
                    description_parts.append(f"Milch hinzufügen für {purpose.capitalize()}")
        
        # If we found semantic mappings, use them
        if description_parts:
            return description_parts[0]
        
        # Fallback to original text
        return original_text
    
    def _generate_semantic_technical_context(self, line_text: str, global_context: str, step_id: str) -> str:
        """Generate technical context using LLM semantic analysis with domain knowledge"""
        if not self.nlp or not global_context:
            return ""
            
        try:
            # Parse line with spaCy for semantic understanding
            doc = self.nlp(line_text)
            
            # Extract semantic components
            main_verb = self._extract_main_verb(doc)
            entities = self._extract_entities_from_line(doc)
            purpose = self._extract_purpose_from_global_context(global_context)
            
            # Get domain knowledge
            domain_config = self.verb_loader.domain_configs.get(self.domain_name, {})
            term_knowledge = domain_config.get('term_specific_knowledge', {})
            betriebsmittel = domain_config.get('operational_materials_addressing', {}).get('material_types', {})
            
            # Debug output
            print(f"[DEBUG] {step_id}: verb='{main_verb}', entities={entities}, purpose='{purpose}'")
            
            # Generate semantic technical context - Check verb-centric patterns first
            
            # Coffee grinding - check for grinding verbs AND grinding-related entities (including Betriebsstoffe)
            if main_verb in ['grind', 'mill'] and any(e in ['grind', 'grinding', 'beans', 'coffee_beans', 'coffeebeans', 'amount', 'degree', 'set'] for e in entities):
                bean_knowledge = betriebsmittel.get('coffee_beans', {})
                if purpose in ['kaffeezubereitung', 'milchkaffee', 'espresso', 'milk_coffee', 'coffee_preparation']:
                    print(f"[DEBUG] {step_id}: Grinding detected - verb: {main_verb}, entities: {entities}, betriebsmittel: {list(betriebsmittel.keys())}")
                    return f"Kaffeebohnen mahlen für {purpose.capitalize()}"
            
            # Entity-specific checks (including Betriebsstoffe from domain JSON)
            for entity in entities:
                # Water heating for coffee preparation (spaCy + domain analysis)
                if entity in ['water', 'heater'] and main_verb in ['activate', 'heat', 'turn']:
                    if purpose in ['kaffeezubereitung', 'milchkaffee', 'espresso', 'milk_coffee', 'coffee_preparation']:
                        # Use spaCy + domain knowledge for semantic context analysis
                        required_materials = self._llm_analyze_required_betriebsmittel(main_verb, entities, betriebsmittel)
                        context_description = self._llm_generate_semantic_context(main_verb, entities, purpose, required_materials)
                        print(f"[DEBUG] {step_id}: Water heating - required materials: {required_materials}")
                        return context_description
                
                # Filter preparation for coffee
                elif entity in ['filter'] and main_verb in ['prepare', 'ready', 'setup']:
                    if purpose in ['kaffeezubereitung', 'milchkaffee', 'espresso', 'milk_coffee', 'coffee_preparation']:
                        return f"Filter preparation for {purpose}"
                
                # Cup/Container handling
                elif entity in ['cup', 'container'] and main_verb in ['retrieve', 'place', 'position']:
                    if purpose in ['kaffeezubereitung', 'milchkaffee', 'espresso', 'milk_coffee', 'coffee_preparation']:
                        return f"Cup positioning for {purpose}"
                
                # Coffee brewing (LLM semantic analysis with Betriebsstoffe)
                elif entity in ['coffee'] and main_verb in ['brew', 'make', 'prepare']:
                    if purpose in ['kaffeezubereitung', 'milchkaffee', 'espresso', 'milk_coffee', 'coffee_preparation']:
                        # Use LLM to semantically determine required Betriebsstoffe for brewing
                        required_materials = self._llm_analyze_required_betriebsmittel(main_verb, entities, betriebsmittel)
                        context_description = self._llm_generate_semantic_context(main_verb, entities, purpose, required_materials)
                        print(f"[DEBUG] {step_id}: Coffee brewing - required materials: {required_materials}")
                        return context_description
                
                # Milk addition (spaCy + domain analysis)
                elif entity in ['milk'] and main_verb in ['add', 'pour']:
                    if any(term in purpose for term in ['milchkaffee', 'milk_coffee']):
                        # Use spaCy + domain knowledge for semantic context analysis
                        required_materials = self._llm_analyze_required_betriebsmittel(main_verb, entities, betriebsmittel)
                        context_description = self._llm_generate_semantic_context(main_verb, entities, purpose, required_materials)
                        print(f"[DEBUG] {step_id}: Milk addition - required materials: {required_materials}")
                        return context_description
                
                # Message/Communication
                elif entity in ['message'] and main_verb in ['output', 'display', 'show']:
                    return "User Interface"
                
                # Time-based triggers
                elif entity in ['time', 'clock'] and main_verb in ['reach', 'set']:
                    return "Time Control"
            
            return ""
            
        except Exception as e:
            print(f"[DEBUG] Semantic technical context generation failed for {step_id}: {e}")
            return ""
    
    def _llm_analyze_required_betriebsmittel(self, main_verb: str, entities: list, available_betriebsmittel: dict) -> list:
        """Use LLM semantic analysis to determine required Betriebsstoffe for the action"""
        if not self.nlp or not main_verb:
            return []
        
        try:
            # Semantic analysis: What materials does this verb typically require?
            required_materials = []
            
            # LLM semantic understanding of brewing process
            if main_verb in ['brew', 'brewing']:
                # Brewing semantically requires: ground coffee (from coffee_beans), hot water, filter
                if 'coffee_beans' in available_betriebsmittel:
                    required_materials.append('coffee_beans')  # Source for ground coffee
                if 'water' in available_betriebsmittel:
                    required_materials.append('water')  # Hot water
                # Filter is implied in brewing process (from entities or context)
                if any(e in ['filter', 'filtration'] for e in entities):
                    required_materials.append('filter')
            
            # For grinding: coffee beans input
            elif main_verb in ['grind', 'grinding']:
                if 'coffee_beans' in available_betriebsmittel:
                    required_materials.append('coffee_beans')
            
            # For heating/activation: water (thermal operations)
            elif main_verb in ['activate', 'heat', 'turn'] and any(e in ['water', 'heater'] for e in entities):
                if 'water' in available_betriebsmittel:
                    required_materials.append('water')
            
            # For milk addition: milk
            elif main_verb in ['add', 'pour'] and any(e in ['milk'] for e in entities):
                if 'milk' in available_betriebsmittel:
                    required_materials.append('milk')
            
            return required_materials
            
        except Exception as e:
            print(f"[DEBUG] LLM Betriebsmittel analysis failed: {e}")
            return []
    
    def _llm_generate_semantic_context(self, main_verb: str, entities: list, purpose: str, required_materials: list) -> str:
        """Use LLM to generate semantic context description based on verb, entities and required materials"""
        if not self.nlp or not main_verb:
            return "General System Control"
        
        try:
            # LLM semantic context generation
            if main_verb in ['brew', 'brewing'] and 'coffee_beans' in required_materials and 'water' in required_materials:
                # Semantic understanding: brewing with coffee beans and water = coffee preparation
                return f"Coffee brewing process for {purpose}"
            
            elif main_verb in ['grind', 'grinding'] and 'coffee_beans' in required_materials:
                # Semantic understanding: grinding coffee beans = coffee preparation step
                return f"Coffee beans grinding for {purpose}"
            
            elif main_verb in ['add', 'pour'] and 'milk' in required_materials:
                # Semantic understanding: adding milk = milk coffee preparation
                return f"Milk addition for {purpose}"
            
            elif main_verb in ['activate', 'heat', 'turn'] and 'water' in required_materials:
                # Semantic understanding: heating water = thermal preparation
                return f"Water heating for {purpose}"
            
            # Fallback: use verb + main entity
            main_entity = entities[0] if entities else "component"
            return f"{main_verb.capitalize()} {main_entity} for {purpose}"
            
        except Exception as e:
            print(f"[DEBUG] LLM semantic context generation failed: {e}")
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
        """Extract UC goal using LLM-based semantic analysis"""
        try:
            doc = self.nlp(full_text)
            
            # Analyze the main flow to understand the goal
            lines = full_text.lower().split('\n')
            
            # Look for goal indicators in the text
            goal_indicators = []
            
            # Check for coffee/beverage preparation
            if any(word in full_text.lower() for word in ['coffee', 'milk', 'espresso', 'brew', 'drink']):
                if 'milk' in full_text.lower() and 'coffee' in full_text.lower():
                    goal_indicators.append("Milchkaffee automatisch zubereiten")
                elif 'espresso' in full_text.lower():
                    goal_indicators.append("Espresso auf Anfrage zubereiten")
                elif 'coffee' in full_text.lower():
                    goal_indicators.append("Kaffee automatisch zubereiten")
            
            # Check for nuclear/reactor operations
            if any(word in full_text.lower() for word in ['nuclear', 'reactor', 'shutdown', 'emergency']):
                goal_indicators.append("Sicherer Reaktor-Notfall-Shutdown")
            
            # Check for rocket/space operations
            if any(word in full_text.lower() for word in ['rocket', 'launch', 'satellite', 'mission']):
                goal_indicators.append("Erfolgreicher Raketenstart")
            
            # Check for assembly/manufacturing
            if any(word in full_text.lower() for word in ['assembly', 'robot', 'component', 'manufacturing']):
                goal_indicators.append("Präzise Roboter-Montage")
            
            # Check timing information
            timing_context = ""
            if '7:00h' in full_text or '7am' in full_text:
                timing_context = " um 7 Uhr morgens"
            elif 'request' in full_text.lower() or 'user' in full_text.lower():
                timing_context = " auf Benutzeranfrage"
            elif 'emergency' in full_text.lower():
                timing_context = " bei Notfall"
            elif 'operator' in full_text.lower():
                timing_context = " auf Operator-Anweisung"
            
            # Construct goal
            if goal_indicators:
                goal = goal_indicators[0] + timing_context
                return goal
            else:
                # Fallback: extract from main flow
                return self._extract_goal_from_main_flow(lines)
                
        except Exception as e:
            print(f"[DEBUG] LLM goal extraction failed: {e}")
            return "UC-Ziel nicht erkannt"
    
    def _extract_goal_from_main_flow(self, lines: List[str]) -> str:
        """Extract goal from main flow steps as fallback"""
        main_flow_actions = []
        
        for line in lines:
            line = line.strip().lower()
            # Look for B-steps that indicate the main purpose
            if line.startswith('b1') or line.startswith('b2') or line.startswith('b3'):
                if 'activates' in line and 'heater' in line:
                    main_flow_actions.append("Heizung")
                elif 'brew' in line or 'coffee' in line:
                    main_flow_actions.append("Kaffee")
                elif 'milk' in line:
                    main_flow_actions.append("Milch")
                elif 'shutdown' in line:
                    main_flow_actions.append("Shutdown")
                elif 'launch' in line:
                    main_flow_actions.append("Start")
        
        if main_flow_actions:
            return f"System für {', '.join(main_flow_actions[:2])}"
        
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
        """Generate Boundaries based on Actor + Transaction Verb rule"""
        boundaries = []
        
        # Check if this step has an actor and transaction verb
        if grammatical.verb_type != VerbType.TRANSACTION_VERB:
            return boundaries
        
        # Check if any actor appears in the step text
        line_lower = line_text.lower()
        for actor in self.uc_context.actors:
            if actor.lower() in line_lower:
                # Generate boundary for Actor + Transaction Verb
                actor_clean = ''.join(word.capitalize() for word in actor.split())
                
                if grammatical.verb_lemma in ['send', 'transmit', 'deliver', 'provide']:
                    boundary_name = f"{actor_clean}OutputBoundary"
                    description = f"Boundary for {actor} to send data/commands to system"
                elif grammatical.verb_lemma in ['receive', 'request', 'ask', 'input']:
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
        """Generate data flows based on preposition semantics"""
        data_flows = []
        
        if not step_id or not grammatical.main_verb:
            return data_flows
        
        # Find controller for this step
        controller_name = None
        for ra_class in ra_classes:
            if ra_class.ra_type == RAType.CONTROLLER:
                controller_name = ra_class.name
                break
        
        if not controller_name:
            return data_flows
        
        # Rule: Before preposition = USE, After preposition = PROVIDE
        for prep, obj in grammatical.prepositional_objects:
            entity_name = self._clean_entity_name(obj)
            # Only create data flows for actual entities, not actors
            if entity_name and not self._is_existing_actor(entity_name):
                # Determine flow type based on preposition
                if prep in ['with', 'from', 'using', 'via', 'through']:
                    flow_type = "use"
                    description = f"{controller_name} uses {entity_name} as input"
                elif prep in ['to', 'for', 'into', 'onto']:
                    flow_type = "provide"
                    description = f"{controller_name} provides output to {entity_name}"
                else:
                    flow_type = "use"  # Default
                    description = f"{controller_name} interacts with {entity_name}"
                
                data_flows.append(DataFlow(
                    step_id=step_id,
                    controller=controller_name,
                    entity=entity_name,
                    flow_type=flow_type,
                    preposition=prep,
                    description=description
                ))
        
        # Direct object typically = PROVIDE (output)
        if grammatical.direct_object:
            entity_name = self._clean_entity_name(grammatical.direct_object)
            # Only create data flows for actual entities, not actors or implementation elements
            if (entity_name and 
                not self._is_existing_actor(entity_name) and 
                not self._is_implementation_element(entity_name)):
                data_flows.append(DataFlow(
                    step_id=step_id,
                    controller=controller_name,
                    entity=entity_name,
                    flow_type="provide",
                    description=f"{controller_name} provides {entity_name} as result"
                ))
            elif entity_name and self._is_implementation_element(entity_name):
                # For implementation elements, create data flow to functional equivalent
                functional_entity = self._get_functional_equivalent(entity_name)
                if functional_entity:
                    data_flows.append(DataFlow(
                        step_id=step_id,
                        controller=controller_name,
                        entity=functional_entity,
                        flow_type="provide",
                        description=f"{controller_name} provides {functional_entity} as result"
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
                 # Connect current to Px_START
                self._create_control_flow_from_parallel_node (px_start ,current , "parallel", "Rule 3: parallel -> parallel (same step number)")
                self._create_control_flow_to_parallel_node(current, px_end, "sequential", "Rule 3: parallel -> parallel (same step number)")
                
                
            # Rule 4: parallel -> parallel (different step number)
            elif (current_type == 'parallel' and next_type == 'parallel' and current_num != next_num):
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
        Generate RUP diagram from JSON analysis using pure RUP visualizer
        
        Args:
            json_file_path: Path to the JSON analysis file
            
        Returns:
            Path to generated diagram file
        """
        try:
            from pure_rup_visualizer import PureRUPVisualizer
            
            # Create pure RUP visualizer
            visualizer = PureRUPVisualizer(figure_size=(20, 16))
            
            # Generate diagram from JSON directly to new directory
            diagram_path = visualizer.generate_diagram(json_file_path)
            return diagram_path
            
        except Exception as e:
            print(f"[ERROR] Failed to generate RUP diagram: {e}")
            import traceback
            traceback.print_exc()
            return ""

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
            domain = 'beverage_preparation'  # Default for UC1, UC2
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


def main():
    """Test main function"""
    try:
        # Test analysis
        analyzer = StructuredUCAnalyzer(domain_name="beverage_preparation")
        line_analyses, json_output = analyzer.analyze_uc_file("Use Case/UC1.txt")
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