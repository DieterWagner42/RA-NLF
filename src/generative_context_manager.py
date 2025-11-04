#!/usr/bin/env python3
"""
Generative Context Manager - NLP-based Context Generation
Replaces hardcoded contexts with domain-driven generative approach using
universal operational materials framework and domain JSON configurations.
"""

import json
import spacy
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from pathlib import Path
import re
from enum import Enum

class ContextType(Enum):
    """Types of generated contexts"""
    OPERATIONAL_MATERIAL = "operational_material"
    SAFETY_CONTEXT = "safety_context"
    HYGIENE_CONTEXT = "hygiene_context"
    FUNCTIONAL_CONTEXT = "functional_context"
    TECHNICAL_CONTEXT = "technical_context"

@dataclass
class GeneratedContext:
    """A generated context with NLP-derived semantics"""
    context_type: ContextType
    context_name: str
    source_text: str
    semantic_features: List[str]
    domain_alignment: float  # 0.0-1.0 confidence in domain alignment
    safety_class: Optional[str] = None
    hygiene_level: Optional[str] = None
    addressing_format: Optional[str] = None
    special_requirements: List[str] = None
    controllers: List[str] = None
    
    def __post_init__(self):
        if self.special_requirements is None:
            self.special_requirements = []
        if self.controllers is None:
            self.controllers = []

@dataclass
class SemanticPattern:
    """Semantic patterns for context detection"""
    pattern_type: str
    keywords: List[str]
    context_triggers: List[str]
    semantic_weight: float
    domain_specificity: float

class GenerativeContextManager:
    """
    NLP-based context manager that generates contexts dynamically
    using domain knowledge and universal operational materials framework
    """
    
    def __init__(self, domain_name: str = "beverage_preparation"):
        self.domain_name = domain_name
        self.nlp = spacy.load("en_core_web_md")
        
        # Load domain and universal configurations
        self.domain_config = self._load_domain_config()
        self.universal_materials = self._load_universal_materials()
        self.common_domain = self._load_common_domain()
        
        # Initialize semantic patterns from domain data
        self.semantic_patterns = self._build_semantic_patterns()
        
        # Context cache for performance
        self.context_cache: Dict[str, List[GeneratedContext]] = {}
    
    def _load_domain_config(self) -> Dict:
        """Load domain-specific configuration"""
        domain_path = Path(f"domains/{self.domain_name}.json")
        if domain_path.exists():
            with open(domain_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_universal_materials(self) -> Dict:
        """Load universal operational materials framework"""
        materials_path = Path("domains/universal_operational_materials.json")
        if materials_path.exists():
            with open(materials_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_common_domain(self) -> Dict:
        """Load common domain verbs and patterns"""
        common_path = Path("domains/common_domain.json")
        if common_path.exists():
            with open(common_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _build_semantic_patterns(self) -> List[SemanticPattern]:
        """Build semantic patterns from domain and universal materials data"""
        patterns = []
        
        # Safety classification patterns
        if "safety_classifications" in self.universal_materials:
            for hazard_type, hazard_data in self.universal_materials["safety_classifications"]["hazard_categories"].items():
                patterns.append(SemanticPattern(
                    pattern_type="safety",
                    keywords=hazard_data.get("examples", []),
                    context_triggers=hazard_data.get("safety_requirements", []),
                    semantic_weight=0.9,
                    domain_specificity=0.8
                ))
        
        # Hygiene classification patterns
        if "hygiene_classifications" in self.universal_materials:
            for hygiene_level, hygiene_data in self.universal_materials["hygiene_classifications"]["sterility_levels"].items():
                patterns.append(SemanticPattern(
                    pattern_type="hygiene",
                    keywords=hygiene_data.get("examples", []),
                    context_triggers=hygiene_data.get("requirements", []),
                    semantic_weight=0.8,
                    domain_specificity=0.7
                ))
        
        # Domain-specific technical patterns
        if "technical_context_mapping" in self.domain_config:
            for context_name, keyword_groups in self.domain_config["technical_context_mapping"]["contexts"].items():
                all_keywords = []
                for keyword_group in keyword_groups:
                    all_keywords.extend(self.domain_config["technical_context_mapping"].get(keyword_group, []))
                
                patterns.append(SemanticPattern(
                    pattern_type="technical",
                    keywords=all_keywords,
                    context_triggers=[context_name.lower()],
                    semantic_weight=0.7,
                    domain_specificity=1.0
                ))
        
        # Operational materials patterns from domain
        if "operational_materials_addressing" in self.domain_config:
            material_types = self.domain_config["operational_materials_addressing"].get("material_types", {})
            for material_name, material_data in material_types.items():
                patterns.append(SemanticPattern(
                    pattern_type="operational_material",
                    keywords=[material_name],
                    context_triggers=material_data.get("tracking_parameters", []),
                    semantic_weight=0.9,
                    domain_specificity=1.0
                ))
        
        return patterns
    
    def generate_contexts_for_text(self, text: str, step_id: str = "") -> List[GeneratedContext]:
        """
        Generate contexts for given text using NLP and semantic analysis
        """
        # Check cache first
        cache_key = f"{text}#{step_id}"
        if cache_key in self.context_cache:
            return self.context_cache[cache_key]
        
        doc = self.nlp(text)
        contexts = []
        
        # Extract semantic features
        semantic_features = self._extract_semantic_features(doc)
        
        # Generate operational material contexts
        contexts.extend(self._generate_operational_material_contexts(doc, semantic_features, step_id))
        
        # Generate safety contexts
        contexts.extend(self._generate_safety_contexts(doc, semantic_features, step_id))
        
        # Generate hygiene contexts
        contexts.extend(self._generate_hygiene_contexts(doc, semantic_features, step_id))
        
        # Generate technical contexts
        contexts.extend(self._generate_technical_contexts(doc, semantic_features, step_id))
        
        # Generate functional contexts
        contexts.extend(self._generate_functional_contexts(doc, semantic_features, step_id))
        
        # Cache results
        self.context_cache[cache_key] = contexts
        
        return contexts
    
    def _extract_semantic_features(self, doc) -> List[str]:
        """Extract semantic features from spaCy document"""
        features = []
        
        # Extract entities
        for ent in doc.ents:
            features.append(f"entity:{ent.label_}:{ent.text.lower()}")
        
        # Extract verbs and their lemmas
        for token in doc:
            if token.pos_ == "VERB":
                features.append(f"verb:{token.lemma_}")
                features.append(f"verb_form:{token.text.lower()}")
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:
                features.append(f"noun_phrase:{chunk.text.lower()}")
        
        # Extract dependency relationships
        for token in doc:
            if token.dep_ in ["dobj", "nsubj", "pobj"]:
                features.append(f"dependency:{token.dep_}:{token.text.lower()}")
        
        return features
    
    def _generate_operational_material_contexts(self, doc, semantic_features: List[str], step_id: str) -> List[GeneratedContext]:
        """Generate contexts for operational materials (Betriebsstoffe)"""
        contexts = []
        
        # Check for operational materials in domain config
        if "operational_materials_addressing" in self.domain_config:
            material_types = self.domain_config["operational_materials_addressing"].get("material_types", {})
            
            for material_name, material_data in material_types.items():
                # Check if text contains this material
                material_matches = self._find_semantic_matches(doc, [material_name])
                
                if material_matches:
                    # Create abstract Manager controller for this material
                    manager_name = f"{material_name.title()}Manager"
                    
                    # NLP-based function detection: What functions should this manager have?
                    functions = self._detect_material_functions(doc.text, material_name)
                    
                    # Generate Manager context with detected functions
                    manager_context = GeneratedContext(
                        context_type=ContextType.FUNCTIONAL_CONTEXT,
                        context_name=manager_name,
                        source_text=doc.text,
                        semantic_features=semantic_features + functions,  # Add functions as semantic features
                        domain_alignment=0.9,  # High confidence for domain materials
                        safety_class=None,
                        hygiene_level=None,
                        addressing_format="",
                        special_requirements=functions,  # Functions as requirements
                        controllers=[]  # Manager IS the controller
                    )
                    contexts.append(manager_context)
        
        return contexts
    
    def _generate_safety_contexts(self, doc, semantic_features: List[str], step_id: str) -> List[GeneratedContext]:
        """Generate safety-related contexts"""
        contexts = []
        
        safety_classifications = self.universal_materials.get("safety_classifications", {}).get("hazard_categories", {})
        
        for hazard_type, hazard_data in safety_classifications.items():
            # Check for safety-related terms in text
            safety_matches = self._find_semantic_matches(doc, hazard_data.get("examples", []))
            
            if safety_matches:
                context = GeneratedContext(
                    context_type=ContextType.SAFETY_CONTEXT,
                    context_name=f"{hazard_type.title()}SafetyContext",
                    source_text=doc.text,
                    semantic_features=semantic_features,
                    domain_alignment=0.8,
                    safety_class=hazard_type,
                    special_requirements=hazard_data.get("safety_requirements", []),
                    controllers=hazard_data.get("special_controllers", [])
                )
                contexts.append(context)
        
        return contexts
    
    def _generate_hygiene_contexts(self, doc, semantic_features: List[str], step_id: str) -> List[GeneratedContext]:
        """Generate hygiene-related contexts - DISABLED to remove Food_GradeHygieneContextController"""
        # Return empty list to eliminate all hygiene context controllers
        return []
    
    def _generate_technical_contexts(self, doc, semantic_features: List[str], step_id: str) -> List[GeneratedContext]:
        """Generate technical contexts from domain knowledge"""
        contexts = []
        
        if "technical_context_mapping" not in self.domain_config:
            return contexts
        
        tech_contexts = self.domain_config["technical_context_mapping"].get("contexts", {})
        
        for context_name, keyword_groups in tech_contexts.items():
            # Collect all keywords for this context
            all_keywords = []
            for keyword_group in keyword_groups:
                all_keywords.extend(self.domain_config["technical_context_mapping"].get(keyword_group, []))
            
            # Check for matches
            tech_matches = self._find_semantic_matches(doc, all_keywords)
            
            if tech_matches:
                context = GeneratedContext(
                    context_type=ContextType.TECHNICAL_CONTEXT,
                    context_name=f"{context_name.replace(' ', '')}Context",
                    source_text=doc.text,
                    semantic_features=semantic_features,
                    domain_alignment=1.0,  # High confidence for domain-specific technical contexts
                    special_requirements=[f"Technical context: {context_name}"]
                )
                contexts.append(context)
        
        return contexts
    
    def _generate_functional_contexts(self, doc, semantic_features: List[str], step_id: str) -> List[GeneratedContext]:
        """Generate functional contexts from verb analysis"""
        contexts = []
        
        # Analyze verbs for functional context
        for token in doc:
            if token.pos_ == "VERB":
                verb_lemma = token.lemma_
                
                # Check domain verb classifications
                verb_context = self._classify_verb_context(verb_lemma)
                
                if verb_context:
                    context = GeneratedContext(
                        context_type=ContextType.FUNCTIONAL_CONTEXT,
                        context_name=f"{verb_lemma.title()}FunctionalContext",
                        source_text=doc.text,
                        semantic_features=semantic_features,
                        domain_alignment=0.7,
                        special_requirements=[f"Functional activity: {verb_context}"]
                    )
                    contexts.append(context)
        
        return contexts
    
    def _find_semantic_matches(self, doc, keywords: List[str]) -> List[str]:
        """Find semantic matches between document and keywords"""
        matches = []
        doc_text_lower = doc.text.lower()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Direct match
            if keyword_lower in doc_text_lower:
                matches.append(keyword)
                continue
            
            # Semantic similarity using spaCy vectors
            keyword_doc = self.nlp(keyword)
            if doc.has_vector and keyword_doc.has_vector:
                similarity = doc.similarity(keyword_doc)
                if similarity > 0.7:  # High similarity threshold
                    matches.append(keyword)
        
        return matches
    
    def _classify_material_safety_hygiene(self, material_name: str, semantic_features: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """Classify material safety and hygiene levels"""
        safety_class = None
        hygiene_level = None
        
        # Check domain-specific examples
        safety_classifications = self.universal_materials.get("safety_classifications", {}).get("hazard_categories", {})
        for hazard_type, hazard_data in safety_classifications.items():
            if material_name in hazard_data.get("examples", []):
                safety_class = hazard_type
                break
        
        hygiene_classifications = self.universal_materials.get("hygiene_classifications", {}).get("sterility_levels", {})
        for hygiene_type, hygiene_data in hygiene_classifications.items():
            if material_name in hygiene_data.get("examples", []):
                hygiene_level = hygiene_type
                break
        
        return safety_class, hygiene_level
    
    def _get_material_requirements(self, material_name: str, safety_class: Optional[str], hygiene_level: Optional[str]) -> List[str]:
        """Get special requirements for material based on safety and hygiene classifications"""
        requirements = []
        
        if safety_class:
            safety_data = self.universal_materials.get("safety_classifications", {}).get("hazard_categories", {}).get(safety_class, {})
            requirements.extend(safety_data.get("safety_requirements", []))
        
        if hygiene_level:
            hygiene_data = self.universal_materials.get("hygiene_classifications", {}).get("sterility_levels", {}).get(hygiene_level, {})
            requirements.extend(hygiene_data.get("requirements", []))
        
        return requirements
    
    def _get_material_controllers(self, material_name: str, safety_class: Optional[str], hygiene_level: Optional[str]) -> List[str]:
        """Get special controllers for material based on safety and hygiene classifications"""
        controllers = []
        
        if safety_class:
            safety_data = self.universal_materials.get("safety_classifications", {}).get("hazard_categories", {}).get(safety_class, {})
            controllers.extend(safety_data.get("special_controllers", []))
        
        if hygiene_level:
            hygiene_data = self.universal_materials.get("hygiene_classifications", {}).get("sterility_levels", {}).get(hygiene_level, {})
            controllers.extend(hygiene_data.get("special_controllers", []))
        
        return controllers
    
    def _detect_implicit_safety_functions(self, text: str, material_name: str) -> List[str]:
        """NLP-based detection of implicit safety functions from domain JSON configuration"""
        implicit_functions = []
        
        # Get material configuration from domain JSON
        material_types = self.domain_config.get("operational_materials_addressing", {}).get("material_types", {})
        material_config = material_types.get(material_name, {})
        
        # Get implicit safety functions from domain configuration
        implicit_safety_config = material_config.get("implicit_safety_functions", {})
        
        # NLP: Analyze text with spaCy
        doc = self.nlp(text)
        
        # For each configured operation type, check if text semantically matches
        for operation_type, safety_function in implicit_safety_config.items():
            # Get operation patterns from domain config
            operation_patterns = material_config.get("operation_patterns", {}).get(operation_type, [])
            
            # Use NLP semantic similarity to detect operations
            if self._nlp_matches_operation(doc, operation_patterns):
                implicit_functions.append(safety_function)
        
        return implicit_functions
    
    def _nlp_matches_operation(self, doc, operation_patterns: List[str]) -> bool:
        """Use NLP to check if text semantically matches operation patterns"""
        if not operation_patterns:
            return False
            
        # Check for semantic matches using spaCy
        for pattern in operation_patterns:
            pattern_doc = self.nlp(pattern)
            # Semantic similarity check
            if doc.similarity(pattern_doc) > 0.7:  # Threshold for semantic similarity
                return True
            
            # Also check for exact token matches
            for token in doc:
                if token.lemma_.lower() in pattern.lower():
                    return True
        
        return False
    
    def _detect_material_functions(self, text: str, material_name: str) -> List[str]:
        """NLP-based detection of functions that a material manager should have"""
        functions = []
        
        # Get material configuration from domain JSON
        material_types = self.domain_config.get("operational_materials_addressing", {}).get("material_types", {})
        material_config = material_types.get(material_name, {})
        
        # Get function mappings from domain configuration
        function_mappings = material_config.get("function_mappings", {})
        
        # NLP: Analyze text with spaCy
        doc = self.nlp(text)
        
        # For each operation type, check if text semantically matches and add corresponding functions
        for operation_type, function_list in function_mappings.items():
            # Get operation patterns from domain config
            operation_patterns = material_config.get("operation_patterns", {}).get(operation_type, [])
            
            # Use NLP semantic similarity to detect operations
            if self._nlp_matches_operation(doc, operation_patterns):
                functions.extend(function_list)
        
        return list(set(functions))  # Remove duplicates
    
    def _classify_verb_context(self, verb_lemma: str) -> Optional[str]:
        """Classify verb into functional context based on domain knowledge"""
        # Check domain verb classifications
        if "verb_classification" in self.domain_config:
            for verb_type, verb_data in self.domain_config["verb_classification"].items():
                if verb_lemma in verb_data.get("verbs", {}):
                    return verb_data["verbs"][verb_lemma]
        
        # Check common domain verbs
        if "verb_classification" in self.common_domain:
            for verb_type, verb_data in self.common_domain["verb_classification"].items():
                if verb_lemma in verb_data.get("verbs", {}):
                    return verb_data["verbs"][verb_lemma]
        
        return None
    
    def get_material_addressing_format(self, material_name: str, context: GeneratedContext) -> str:
        """Generate addressing format for operational material"""
        if not context.safety_class or not context.hygiene_level:
            return f"{material_name.upper()}-BATCH-{{}}"
        
        # Use universal addressing format
        addressing_structure = self.universal_materials.get("addressing_and_tracking", {}).get("id_structure", {})
        format_template = addressing_structure.get("format", "{SAFETY_CLASS}-{HYGIENE_LEVEL}-{MATERIAL_CODE}-{BATCH_ID}-{LOCATION}")
        
        return format_template.replace("{SAFETY_CLASS}", context.safety_class.upper()).replace("{HYGIENE_LEVEL}", context.hygiene_level.upper()).replace("{MATERIAL_CODE}", material_name.upper())
    
    def get_context_summary(self, contexts: List[GeneratedContext]) -> Dict[str, Any]:
        """Generate summary of all contexts for analysis"""
        summary = {
            "total_contexts": len(contexts),
            "context_types": {},
            "safety_classes": set(),
            "hygiene_levels": set(),
            "special_controllers": set(),
            "domain_alignment_avg": 0.0
        }
        
        if not contexts:
            return summary
        
        for context in contexts:
            context_type = context.context_type.value
            summary["context_types"][context_type] = summary["context_types"].get(context_type, 0) + 1
            
            if context.safety_class:
                summary["safety_classes"].add(context.safety_class)
            if context.hygiene_level:
                summary["hygiene_levels"].add(context.hygiene_level)
            summary["special_controllers"].update(context.controllers)
        
        summary["domain_alignment_avg"] = sum(c.domain_alignment for c in contexts) / len(contexts)
        summary["safety_classes"] = list(summary["safety_classes"])
        summary["hygiene_levels"] = list(summary["hygiene_levels"])
        summary["special_controllers"] = list(summary["special_controllers"])
        
        return summary