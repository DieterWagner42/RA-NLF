"""
UC1 Advanced Verb Analysis
Unterscheidet zwischen Transaktionsverben und Funktionsverben
Erkennt Realisierungselemente und schlägt funktionale Aktivitäten vor
"""

import spacy
from spacy.tokens import Token
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from domain_verb_loader import DomainVerbLoader, VerbType as DomainVerbType

# Use VerbType from domain_verb_loader
VerbType = DomainVerbType

class ElementType(Enum):
    FUNCTIONAL_ENTITY = "functional"    # Water, Coffee (Geschäftsobjekte)
    IMPLEMENTATION_ELEMENT = "implementation"  # Heater, Motor (Realisierungselemente)
    CONTAINER = "container"            # Cup, Filter (Behälter)
    CONTROL_DATA = "control"           # Message, Status (Kontrolldaten)

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
    
    def __post_init__(self):
        if self.prepositional_objects is None:
            self.prepositional_objects = []
        if self.warnings is None:
            self.warnings = []

@dataclass
class RAClass:
    name: str
    type: str  # "Actor", "Boundary", "Controller", "Entity"
    stereotype: str  # "«actor»", "«boundary»", "«control»", "«entity»"
    element_type: ElementType
    step_references: List[str]
    description: str
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class UC1AdvancedVerbAnalyzer:
    """
    Erweiterte Verbanalyse für UC1 mit Transaktions-/Funktionsverben
    und Erkennung von Realisierungselementen
    """
    
    def __init__(self, domain_name: str = "beverage_preparation"):
        self.nlp = spacy.load("en_core_web_md")
        self.domain_name = domain_name
        self.domain_loader = DomainVerbLoader()
        
        # Load domain-specific verb configuration
        self.verb_config = self.domain_loader.get_verb_configuration(domain_name)
        
        # Keep old attributes for backwards compatibility in main()
        self.transformation_verbs = self.verb_config.transformation_verbs
        
        print(f"Loaded verb configuration for domain: {domain_name}")
        print(f"Transaction verbs: {len(self.verb_config.transaction_verbs)}")
        print(f"Transformation verbs: {len(self.verb_config.transformation_verbs)}")
        print(f"Function verbs: {len(self.verb_config.function_verbs)}")
        print(f"Implementation elements: {len(self.verb_config.implementation_elements)}")
        
        # Funktionale Entitäten
        self.functional_entities = {
            "water": ElementType.FUNCTIONAL_ENTITY,
            "coffee": ElementType.FUNCTIONAL_ENTITY,
            "milk": ElementType.FUNCTIONAL_ENTITY,
            "cup": ElementType.CONTAINER,
            "filter": ElementType.CONTAINER,
            "storage": ElementType.CONTAINER,
            "message": ElementType.CONTROL_DATA,
            "time": ElementType.CONTROL_DATA
        }
    
    def analyze_uc1_complete(self) -> Tuple[List[VerbAnalysis], List[RAClass]]:
        """
        Analysiere den kompletten UC1: Hauptablauf + Alternative Flows + Extension Flows
        """
        # Main Flow (B1-B6)
        main_flow_steps = [
            ("B1", "System clock reaches set time of 7:00h (Radio clock)"),
            ("B2a", "The system activates the water heater"),
            ("B2b", "The system prepares filter"),
            ("B2c", "The system grinds the set amount at the set grinding degree directly into the filter"),
            ("B2d", "The system retrieves cup from storage container and places it under the filter"),
            ("B3a", "The system begins brewing coffee with the set water amount into the cup"),
            ("B3b", "The system adds milk to the cup"),
            ("B4", "The system outputs a message to user"),
            ("B5", "The system presents cup to user"),
            ("B6", "End UC")
        ]
        
        # Alternative Flow A1: Water shortage at B2a (Fatal Error)
        alternative_flow_a1 = [
            ("A1", "Water heater has too little water"),  # Condition trigger
            ("A1.1", "The system switches off water heater"),
            ("A1.2", "The system outputs an error message to user"),
            ("A1.3", "End UC")
        ]
        
        # Alternative Flow A2: Milk shortage at B3b (Non-Fatal Error)  
        alternative_flow_a2 = [
            ("A2", "Too little milk"),  # Condition trigger
            ("A2.1", "The system stops milk addition"),
            ("A2.2", "The system outputs an error message to user"), 
            ("A2.3", "End UC")
        ]
        
        # Alternative Flow A3: Overheat detection at any time (Critical Safety Error)
        alternative_flow_a3 = [
            ("A3", "Overheat detected"),  # Safety condition trigger
            ("A3.1", "The system stops all actions"),
            ("A3.2", "The system switch of itself"),
            ("A3.3", "End UC")
        ]
        
        # Extension Flow E1: Sugar addition (Optional Feature)
        # E1 B3-B5 (trigger) User wants sugar in coffee -> analyze the trigger
        extension_flow_e1 = [
            ("E1", "User wants sugar in coffee"),  # The actual trigger to analyze
            ("E1.1", "The system adds sugar to the cup"),
            # E1.2 "Continue in main flow" is just flow control - not system behavior
        ]
        
        # Combine all flows
        uc1_steps = main_flow_steps + alternative_flow_a1 + alternative_flow_a2 + alternative_flow_a3 + extension_flow_e1
        
        print("="*80)
        print("UC1 COMPLETE VERB ANALYSIS - ALL FLOWS")
        print("="*80)
        print(f"Main Flow (B1-B6): {len(main_flow_steps)} steps")
        print(f"Alternative Flow A1 (Water shortage): {len(alternative_flow_a1)} steps") 
        print(f"Alternative Flow A2 (Milk shortage): {len(alternative_flow_a2)} steps")
        print(f"Alternative Flow A3 (Overheat safety): {len(alternative_flow_a3)} steps")
        print(f"Extension Flow E1 (Sugar addition): {len(extension_flow_e1)} steps")
        print(f"Total steps to analyze: {len(uc1_steps)}")
        print("="*80)
        
        verb_analyses = []
        ra_classes = {}  # Dict für einzigartige RA Classes
        
        for step_id, step_text in uc1_steps:
            if step_text == "End UC":
                print(f"\n--- {step_id}: {step_text} --- (Flow control - not analyzed)")
                continue
                
            print(f"\n--- {step_id}: {step_text} ---")
            
            # Verb-Analyse
            verb_analysis = self._analyze_sentence_verbs(step_id, step_text)
            verb_analyses.append(verb_analysis)
            
            # RA Classes ableiten
            step_ra_classes = self._derive_ra_classes(verb_analysis)
            
            # Zu Gesamtliste hinzufügen
            for ra_class in step_ra_classes:
                key = f"{ra_class.type}_{ra_class.name}"
                if key in ra_classes:
                    # Erweitere bestehende Klasse
                    ra_classes[key].step_references.append(step_id)
                else:
                    ra_classes[key] = ra_class
        
        print("\n" + "="*80)
        print("ALLE RA CLASSES - UC1 VOLLSTÄNDIG")
        print("="*80)
        
        # Sortiere RA Classes nach Typ
        sorted_classes = sorted(ra_classes.values(), key=lambda x: (x.type, x.name))
        
        for ra_class in sorted_classes:
            print(f"\n{ra_class.type}: {ra_class.name}")
            print(f"  Stereotype: {ra_class.stereotype}")
            print(f"  Element Type: {ra_class.element_type.value}")
            print(f"  Steps: {', '.join(ra_class.step_references)}")
            print(f"  Description: {ra_class.description}")
            
            if ra_class.warnings:
                for warning in ra_class.warnings:
                    print(f"  WARNING: {warning}")
        
        return verb_analyses, list(sorted_classes)
    
    def _analyze_sentence_verbs(self, step_id: str, sentence: str) -> VerbAnalysis:
        """
        Analysiere Verben in einem UC-Schritt
        """
        doc = self.nlp(sentence)
        
        # Finde Hauptverb - handle "begins + verb" pattern
        main_verb = None
        actual_verb = None
        
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                main_verb = token
                
                # Check for "begins/starts + gerund" pattern (e.g., "begins brewing")
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
        
        # Use actual verb if "begins + verb" pattern was found
        if actual_verb:
            main_verb = actual_verb
        
        if not main_verb:
            return VerbAnalysis(
                step_id=step_id,
                original_text=sentence,
                verb="",
                verb_lemma="",
                verb_type=VerbType.FUNCTION_VERB
            )
        
        print(f"Hauptverb: {main_verb.text} (lemma: {main_verb.lemma_})")
        
        # Verb kategorisieren
        verb_type = self._categorize_verb(main_verb.lemma_)
        
        # Special handling for triggers: UC trigger and extension triggers are always transaction verbs
        if step_id == "B1":
            verb_type = VerbType.TRANSACTION_VERB
            print(f"B1 Override: UC trigger -> TRANSACTION VERB")
        elif step_id == "E1":
            verb_type = VerbType.TRANSACTION_VERB  
            print(f"E1 Override: Extension trigger -> TRANSACTION VERB")
        elif step_id in ["A1", "A2", "A3"]:
            verb_type = VerbType.TRANSACTION_VERB
            print(f"{step_id} Override: Alternative flow condition trigger -> TRANSACTION VERB")
        
        print(f"Verb Type: {verb_type.value}")
        
        # Direct Object
        direct_obj = self._find_direct_object(main_verb)
        print(f"Direct Object: {direct_obj}")
        
        # Adjective analysis for entity qualification
        if direct_obj:
            adjectives = self._extract_adjectives_from_phrase(direct_obj)
            if adjectives:
                print(f"Adjectives detected: {adjectives}")
        
        # Prepositional Objects
        prep_objs = self._find_prepositional_objects(main_verb)
        print(f"Prepositional Objects: {prep_objs}")
        
        # Adjective analysis for prepositional objects
        for prep, obj in prep_objs:
            adj_in_prep = self._extract_adjectives_from_phrase(obj)
            if adj_in_prep:
                print(f"Adjectives in '{prep} {obj}': {adj_in_prep}")
        
        # Warnungen für Realisierungselemente + Transformation-Erkennung
        warnings = []
        suggested_activity = None
        
        # Implementation element detection using domain configuration
        if direct_obj and verb_type == VerbType.FUNCTION_VERB:
            obj_words = direct_obj.lower().split()
            for word in obj_words:
                impl_info = self.domain_loader.get_implementation_element_info(word, self.domain_name)
                if impl_info:
                    warnings.append(impl_info["warning"])
                    suggested_activity = impl_info["functional_suggestion"]
                    print(f"WARNING: {impl_info['warning']}")
                    print(f"SUGGESTION: {impl_info['functional_suggestion']}")
                    
                    # Correct verb type to TRANSFORMATION for implementation elements
                    if word == "heater" and main_verb.lemma_ == "activate":
                        verb_type = VerbType.TRANSFORMATION_VERB
                        print(f"CORRECTION: 'activates heater' -> TRANSFORMATION (Water -> HotWater)")
        
        return VerbAnalysis(
            step_id=step_id,
            original_text=sentence,
            verb=main_verb.text,
            verb_lemma=main_verb.lemma_,
            verb_type=verb_type,
            direct_object=direct_obj,
            prepositional_objects=prep_objs,
            warnings=warnings,
            suggested_functional_activity=suggested_activity
        )
    
    def _categorize_verb(self, verb_lemma: str) -> VerbType:
        """Kategorisiere Verb als Transaction/Transformation/Function using domain configuration"""
        return self.domain_loader.categorize_verb(verb_lemma, self.domain_name)
    
    def _find_direct_object(self, verb: Token) -> Optional[str]:
        """Finde direktes Objekt des Verbs"""
        for child in verb.children:
            if child.dep_ == "dobj":
                return self._expand_noun_phrase(child)
        return None
    
    def _find_prepositional_objects(self, verb: Token) -> List[Tuple[str, str]]:
        """Finde alle Präpositionsobjekte"""
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
        """Erweitere Nomen um Adjektive, Artikel, etc."""
        phrase_tokens = []
        
        # Modifikatoren sammeln
        for child in noun.children:
            if child.dep_ in ["amod", "det", "compound"]:
                phrase_tokens.append(child)
        
        phrase_tokens.append(noun)
        
        # Sortiere nach Position
        phrase_tokens.sort(key=lambda x: x.i)
        return " ".join([token.text for token in phrase_tokens])
    
    def _extract_adjectives_from_phrase(self, phrase: str) -> List[str]:
        """Extract adjectives from a noun phrase"""
        doc = self.nlp(phrase)
        adjectives = []
        
        for token in doc:
            if token.pos_ == "ADJ" or token.dep_ == "amod":
                adjectives.append(token.text)
        
        return adjectives
    
    def _parse_compound_settings(self, verb_analysis: VerbAnalysis) -> List[RAClass]:
        """Parse compound settings into separate entities (e.g., amount + grinding degree)"""
        entities = []
        
        # Special handling for B2c: "grinds the set amount at the set grinding degree"
        if verb_analysis.step_id == "B2c":
            # Extract settings from direct object and prepositional phrases
            settings_found = set()
            
            # Check direct object for "amount"
            if verb_analysis.direct_object and "amount" in verb_analysis.direct_object.lower():
                entities.append(RAClass(
                    name="CoffeeAmount",
                    type="Entity",
                    stereotype="«entity»",
                    element_type=ElementType.CONTROL_DATA,
                    step_references=[verb_analysis.step_id],
                    description="Coffee bean weight/quantity setting"
                ))
                settings_found.add("amount")
                print(f"Setting detected: CoffeeAmount from '{verb_analysis.direct_object}'")
            
            # Check prepositional objects for "grinding degree"
            for prep, obj in verb_analysis.prepositional_objects:
                if "grinding" in obj.lower() and "degree" in obj.lower():
                    entities.append(RAClass(
                        name="GrindingDegree", 
                        type="Entity",
                        stereotype="«entity»",
                        element_type=ElementType.CONTROL_DATA,
                        step_references=[verb_analysis.step_id],
                        description="Coffee grinding fineness setting"
                    ))
                    settings_found.add("grinding_degree")
                    print(f"Setting detected: GrindingDegree from '{prep} {obj}'")
                
                # Check for water amount in B3a
                elif "water" in obj.lower() and "amount" in obj.lower():
                    entities.append(RAClass(
                        name="WaterAmount",
                        type="Entity", 
                        stereotype="«entity»",
                        element_type=ElementType.CONTROL_DATA,
                        step_references=[verb_analysis.step_id],
                        description="Water quantity setting"
                    ))
                    settings_found.add("water_amount")
                    print(f"Setting detected: WaterAmount from '{prep} {obj}'")
        
        # Similar handling for B3a: "with the set water amount"
        elif verb_analysis.step_id == "B3a":
            for prep, obj in verb_analysis.prepositional_objects:
                if "water" in obj.lower() and "amount" in obj.lower():
                    entities.append(RAClass(
                        name="WaterAmount",
                        type="Entity",
                        stereotype="«entity»",
                        element_type=ElementType.CONTROL_DATA,
                        step_references=[verb_analysis.step_id],
                        description="Water quantity setting for brewing"
                    ))
                    print(f"Setting detected: WaterAmount from '{prep} {obj}'")
        
        return entities
    
    def _derive_ra_classes(self, verb_analysis: VerbAnalysis) -> List[RAClass]:
        """
        Leite RA Classes aus Verb-Analyse ab
        """
        ra_classes = []
        
        # 1. Controller für jeden Schritt
        controller_name = self._derive_controller_name(verb_analysis)
        if controller_name:
            ra_classes.append(RAClass(
                name=controller_name,
                type="Controller",
                stereotype="«control»",
                element_type=ElementType.FUNCTIONAL_ENTITY,
                step_references=[verb_analysis.step_id],
                description=f"Controls {verb_analysis.verb} operation in {verb_analysis.step_id}"
            ))
        
        # 2. Entities aus Direct Object
        if verb_analysis.direct_object:
            entities = self._derive_entities_from_object(verb_analysis.direct_object, verb_analysis)
            ra_classes.extend(entities)
        
        # 3. Entities aus Prepositional Objects
        for prep, obj in verb_analysis.prepositional_objects:
            entities = self._derive_entities_from_object(obj, verb_analysis, prep)
            ra_classes.extend(entities)
        
        # 4. Spezielle Actors für B1 (Timer)
        if verb_analysis.step_id == "B1" and "clock" in verb_analysis.original_text.lower():
            ra_classes.append(RAClass(
                name="Timer",
                type="Actor",
                stereotype="«actor»",
                element_type=ElementType.CONTROL_DATA,
                step_references=[verb_analysis.step_id],
                description="Non-human actor that triggers time-based events"
            ))
        
        # 5. Boundaries für Transaktionsverben
        if verb_analysis.verb_type == VerbType.TRANSACTION_VERB:
            boundaries = self._derive_boundaries_from_transaction(verb_analysis)
            ra_classes.extend(boundaries)
        
        # 6. User Actor für Output-Steps and Extension Triggers
        if verb_analysis.step_id in ["B4", "B5"] and "user" in verb_analysis.original_text.lower():
            ra_classes.append(RAClass(
                name="User",
                type="Actor", 
                stereotype="«actor»",
                element_type=ElementType.FUNCTIONAL_ENTITY,
                step_references=[verb_analysis.step_id],
                description="Human actor receiving system output"
            ))
        elif verb_analysis.step_id == "E1" and "user" in verb_analysis.original_text.lower():
            ra_classes.append(RAClass(
                name="User",
                type="Actor",
                stereotype="«actor»", 
                element_type=ElementType.FUNCTIONAL_ENTITY,
                step_references=[verb_analysis.step_id],
                description="Human actor requesting optional features"
            ))
        
        # 7. Parse compound settings into separate entities
        setting_entities = self._parse_compound_settings(verb_analysis)
        ra_classes.extend(setting_entities)
        
        return ra_classes
    
    def _derive_controller_name(self, verb_analysis: VerbAnalysis) -> Optional[str]:
        """Leite Controller-Namen ab"""
        step_id = verb_analysis.step_id
        verb = verb_analysis.verb_lemma
        
        # Generalized Controller-Namen basierend auf Funktion
        if step_id == "B1":
            return "TimeManager"
        elif step_id == "B2a" and "water" in verb_analysis.original_text:
            # Generalized: WaterManager handles all water functions (store, heat, release, pressure)
            return "WaterManager" 
        elif step_id == "B2b" and "filter" in verb_analysis.original_text:
            return "FilterManager"
        elif step_id == "B2c" and "grind" in verb:
            return "CoffeeManager"
        elif step_id == "B2d" and "cup" in verb_analysis.original_text:
            return "ContainerManager"
        elif step_id == "B3a" and "brew" in verb:
            return "CoffeeManager"
        elif step_id == "B3b" and "milk" in verb_analysis.original_text:
            return "MilkManager"
        elif step_id == "B4" and "output" in verb:
            return "UserNotificationManager"
        elif step_id == "B5" and "present" in verb:
            return "UserDeliveryManager"
        
        # Alternative Flow Controllers
        elif step_id == "A1" and "water" in verb_analysis.original_text and "little" in verb_analysis.original_text:
            return "WaterLevelSensorManager"  # Monitors water level
        elif step_id == "A1.1" and "switch" in verb and "water" in verb_analysis.original_text:
            return "WaterManager"  # Same manager handles error conditions
        elif step_id == "A1.2" and "output" in verb and "error" in verb_analysis.original_text:
            return "ErrorNotificationManager"
        elif step_id == "A2" and "milk" in verb_analysis.original_text and "little" in verb_analysis.original_text:
            return "MilkLevelSensorManager"  # Monitors milk level
        elif step_id == "A2.1" and "stop" in verb and "milk" in verb_analysis.original_text:
            return "MilkManager"  # Same manager handles error conditions
        elif step_id == "A2.2" and "output" in verb and "error" in verb_analysis.original_text:
            return "ErrorNotificationManager"
        elif step_id == "A3" and "overheat" in verb_analysis.original_text.lower():
            return "OverheatSensorManager"  # Critical safety monitoring
        elif step_id == "A3.1" and "stop" in verb and "all" in verb_analysis.original_text:
            return "EmergencyShutdownManager"  # Emergency safety controller
        elif step_id == "A3.2" and "switch" in verb and "itself" in verb_analysis.original_text:
            return "SystemPowerManager"  # System power control
        
        # Extension Flow Controllers
        elif step_id == "E1" and "want" in verb and "sugar" in verb_analysis.original_text:
            return "UserRequestManager"  # Handles user requests/preferences
        elif step_id == "E1.1" and "add" in verb and "sugar" in verb_analysis.original_text:
            return "SugarManager"
        
        return None
    
    def _derive_entities_from_object(self, obj_text: str, verb_analysis: VerbAnalysis, 
                                   preposition: str = None) -> List[RAClass]:
        """Leite Entities aus Objekttext ab"""
        entities = []
        obj_words = obj_text.lower().split()
        
        # Entferne Artikel
        obj_words = [w for w in obj_words if w not in ["the", "a", "an"]]
        
        for word in obj_words:
            if word in ["water", "coffee", "milk"]:
                # Funktionale Entität
                entity_name = word.capitalize()
                
                # Spezielle Transformationen
                if word == "water" and verb_analysis.step_id == "B2a":
                    # Water heater -> HotWater
                    entity_name = "HotWater"
                    warnings = ["Water Heater ist Realisierungselement - HotWater Entity verwendet"]
                elif word == "coffee" and verb_analysis.step_id == "B2c":
                    # Ground coffee
                    entity_name = "GroundCoffee"
                    warnings = []
                else:
                    warnings = []
                
                entities.append(RAClass(
                    name=entity_name,
                    type="Entity",
                    stereotype="«entity»",
                    element_type=ElementType.FUNCTIONAL_ENTITY,
                    step_references=[verb_analysis.step_id],
                    description=f"Functional entity representing {word}",
                    warnings=warnings
                ))
            
            elif word in ["cup", "filter", "storage"]:
                # Container Entity
                entities.append(RAClass(
                    name=word.capitalize(),
                    type="Entity",
                    stereotype="«entity»",
                    element_type=ElementType.CONTAINER,
                    step_references=[verb_analysis.step_id],
                    description=f"Container entity for holding/processing"
                ))
            
            elif word in ["message", "error", "sugar"]:
                # Control Data Entity
                entities.append(RAClass(
                    name=word.capitalize(),
                    type="Entity",
                    stereotype="«entity»",
                    element_type=ElementType.CONTROL_DATA,
                    step_references=[verb_analysis.step_id],
                    description=f"Control data entity"
                ))
        
        # Spezielle kombinierte Entities
        if "ground" in obj_text.lower() and "coffee" in obj_text.lower():
            entities.append(RAClass(
                name="GroundCoffee",
                type="Entity",
                stereotype="«entity»",
                element_type=ElementType.FUNCTIONAL_ENTITY,
                step_references=[verb_analysis.step_id],
                description="Ground coffee beans ready for brewing"
            ))
        
        # Transformation-Entities hinzufügen
        if verb_analysis.verb_type == VerbType.TRANSFORMATION_VERB:
            entities.extend(self._add_transformation_entities(verb_analysis))
        
        return entities
    
    def _add_transformation_entities(self, verb_analysis: VerbAnalysis) -> List[RAClass]:
        """Füge Input/Output Entities für Transformationsverben hinzu"""
        entities = []
        verb_lemma = verb_analysis.verb_lemma
        step_id = verb_analysis.step_id
        
        # Transformation-Patterns
        if verb_lemma == "grind" and step_id == "B2c":
            # Input: CoffeeBeans, Output: GroundCoffee
            entities.append(RAClass(
                name="CoffeeBeans",
                type="Entity",
                stereotype="«entity»",
                element_type=ElementType.FUNCTIONAL_ENTITY,
                step_references=[step_id],
                description="Raw coffee beans (Transformation Input)"
            ))
            entities.append(RAClass(
                name="GroundCoffee", 
                type="Entity",
                stereotype="«entity»",
                element_type=ElementType.FUNCTIONAL_ENTITY,
                step_references=[step_id],
                description="Ground coffee beans (Transformation Output)"
            ))
        
        elif (verb_lemma == "activate" and "heater" in verb_analysis.original_text) or \
             (verb_lemma in ["heat", "heating"]):
            # Input: Water, Output: HotWater (bereits durch HotWater Entity abgedeckt)
            entities.append(RAClass(
                name="Water",
                type="Entity", 
                stereotype="«entity»",
                element_type=ElementType.FUNCTIONAL_ENTITY,
                step_references=[step_id],
                description="Cold water (Transformation Input)"
            ))
        
        elif verb_lemma in ["brew", "brewing"] and step_id == "B3a":
            # Input: GroundCoffee + HotWater, Output: Coffee (generalized)
            entities.append(RAClass(
                name="Coffee",
                type="Entity",
                stereotype="«entity»", 
                element_type=ElementType.FUNCTIONAL_ENTITY,
                step_references=[step_id],
                description="Coffee (base beverage)"
            ))
        
        elif verb_lemma == "add" and step_id == "B3b":
            # Generalized approach: Coffee + Milk -> Coffee (with milk additive)
            # No separate MilkCoffee entity - Coffee entity evolves with additives
            print(f"Generalized coffee: Coffee + Milk additive -> Coffee (state: with milk)")
        
        elif verb_lemma == "add" and step_id == "E1.1" and "sugar" in verb_analysis.original_text:
            # Generalized approach: Coffee + Sugar -> Coffee (with sugar additive)
            # No separate SweetenedCoffee entity - Coffee entity evolves with additives
            print(f"Generalized coffee: Coffee + Sugar additive -> Coffee (state: with sugar)")
        
        # Check for generalized coffee additive transformations
        if verb_lemma == "add" and verb_analysis.direct_object:
            additive = verb_analysis.direct_object.lower().split()[-1]  # Get the main additive word
            if self.domain_loader.is_coffee_additive(additive, self.domain_name):
                additive_info = self.domain_loader.get_coffee_additive_info(additive, self.domain_name)
                print(f"Coffee additive detected: {additive_info['transformation']}")
                print(f"Additive type: {additive_info['type']}")
        
        return entities
    
    def _derive_boundaries_from_transaction(self, verb_analysis: VerbAnalysis) -> List[RAClass]:
        """Leite Boundary Objects aus Transaktionsverben ab"""
        boundaries = []
        verb_lemma = verb_analysis.verb_lemma
        step_id = verb_analysis.step_id
        
        if verb_lemma == "output" and step_id == "B4":
            # Message Output Boundary
            boundaries.append(RAClass(
                name="UserNotificationBoundary",
                type="Boundary",
                stereotype="«boundary»",
                element_type=ElementType.CONTROL_DATA,
                step_references=[step_id],
                description="Boundary for outputting messages to user"
            ))
        
        elif verb_lemma == "present" and step_id == "B5":
            # Product Delivery Boundary  
            boundaries.append(RAClass(
                name="UserDeliveryBoundary",
                type="Boundary",
                stereotype="«boundary»",
                element_type=ElementType.FUNCTIONAL_ENTITY,
                step_references=[step_id],
                description="Boundary for presenting final product to user"
            ))
        
        elif verb_lemma == "want" and step_id == "E1":
            # User Request Input Boundary
            boundaries.append(RAClass(
                name="UserRequestBoundary",
                type="Boundary",
                stereotype="«boundary»",
                element_type=ElementType.CONTROL_DATA,
                step_references=[step_id],
                description="Boundary for receiving user requests/preferences"
            ))
        
        return boundaries


def main():
    """Test UC1 Complete Verb Analysis - All Flows"""
    analyzer = UC1AdvancedVerbAnalyzer()
    verb_analyses, ra_classes = analyzer.analyze_uc1_complete()
    
    print("\n" + "="*80)
    print("ZUSAMMENFASSUNG - VERB KATEGORIEN")
    print("="*80)
    
    transaction_verbs = [v for v in verb_analyses if v.verb_type == VerbType.TRANSACTION_VERB]
    transformation_verbs = [v for v in verb_analyses if v.verb_type == VerbType.TRANSFORMATION_VERB]
    function_verbs = [v for v in verb_analyses if v.verb_type == VerbType.FUNCTION_VERB]
    
    print(f"\nTRANSAKTIONSVERBEN - Boundary Identification ({len(transaction_verbs)}):")
    for v in transaction_verbs:
        print(f"  {v.step_id}: {v.verb} - {v.original_text}")
    
    print(f"\nTRANSFORMATIONSVERBEN - Entity rein/Entity raus ({len(transformation_verbs)}):")
    for v in transformation_verbs:
        transformation = analyzer.verb_config.transformation_verbs.get(v.verb_lemma, "unknown transformation")
        print(f"  {v.step_id}: {v.verb} -> {transformation}")
    
    print(f"\nFUNKTIONSVERBEN - Allgemeine Aktionen ({len(function_verbs)}):")
    for v in function_verbs:
        print(f"  {v.step_id}: {v.verb} - {v.original_text}")
    
    # Flow-based statistics
    main_flow_verbs = [v for v in verb_analyses if v.step_id.startswith('B')]
    alt_flow_verbs = [v for v in verb_analyses if v.step_id.startswith('A')]
    ext_flow_verbs = [v for v in verb_analyses if v.step_id.startswith('E')]
    
    print(f"\nFLOW STATISTICS:")
    print(f"   Main Flow (B): {len(main_flow_verbs)} steps")
    print(f"   Alternative Flows (A): {len(alt_flow_verbs)} steps")  
    print(f"   Extension Flows (E): {len(ext_flow_verbs)} steps")
    print(f"   Total Steps Analyzed: {len(verb_analyses)}")
    
    print(f"\nRA CLASSES STATISTICS:")
    print(f"   Actors: {len([c for c in ra_classes if c.type == 'Actor'])}")
    print(f"   Boundaries: {len([c for c in ra_classes if c.type == 'Boundary'])}")
    print(f"   Controllers: {len([c for c in ra_classes if c.type == 'Controller'])}")
    print(f"   Entities: {len([c for c in ra_classes if c.type == 'Entity'])}")
    print(f"   Total RA Classes: {len(ra_classes)}")
    
    print(f"\nVERB CLASSIFICATION:")
    print(f"   Transformation-Verben: {len(transformation_verbs)}")
    print(f"   Transaction-Verben: {len(transaction_verbs)}")
    print(f"   Function-Verben: {len(function_verbs)}")
    print(f"   Warnungen: {len([c for c in ra_classes if c.warnings])}")


if __name__ == "__main__":
    main()