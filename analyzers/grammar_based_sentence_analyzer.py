"""
Grammar-Based Sentence Analyzer für UC-Methode Phase 3
Domain-unabhängige grammatische Analyse von UC-Schritten
"""

import spacy
from spacy.tokens import Token
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    ACTIVATE = "activate"       # system activates X
    PERFORM = "perform"         # system performs X  
    MONITOR = "monitor"         # system monitors X
    PROCESS = "process"         # system processes X
    TRANSFER = "transfer"       # system moves X to Y
    TRANSFORM = "transform"     # system converts X to Y
    CHECK = "check"            # system checks/verifies X
    NOTIFY = "notify"          # system notifies X

class EntityRole(Enum):
    INPUT = "input"            # consumed by action
    OUTPUT = "output"          # produced by action  
    TOOL = "tool"              # used for action
    TARGET = "target"          # receives action
    RESOURCE = "resource"      # provides capability

@dataclass
class GrammaticalAction:
    verb: str
    verb_lemma: str
    action_type: ActionType
    direct_object: Optional[str] = None
    indirect_object: Optional[str] = None
    prepositional_phrases: List[Tuple[str, str]] = None  # (preposition, object)
    subjects: List[str] = None
    
    def __post_init__(self):
        if self.prepositional_phrases is None:
            self.prepositional_phrases = []
        if self.subjects is None:
            self.subjects = []

@dataclass 
class EntityRelation:
    entity: str
    role: EntityRole
    controller: str
    transformation: Optional[str] = None

class GrammarBasedSentenceAnalyzer:
    """
    Domain-unabhängige grammatische Analyse von UC-Schritten
    Verwendet spaCy Dependency Parsing für grammatische Strukturen
    """
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        
        # Grammatische Verb-Kategorien (domain-unabhängig)
        self.verb_categories = {
            ActionType.ACTIVATE: ["activate", "start", "initiate", "begin", "trigger", "enable"],
            ActionType.PERFORM: ["perform", "execute", "run", "carry out", "conduct"],
            ActionType.MONITOR: ["monitor", "watch", "track", "observe", "check", "verify"],
            ActionType.PROCESS: ["process", "handle", "manage", "control"],
            ActionType.TRANSFER: ["move", "transfer", "transport", "deliver", "send"],
            ActionType.TRANSFORM: ["convert", "transform", "change", "produce", "generate", "create"],
            ActionType.CHECK: ["check", "verify", "validate", "confirm", "test"],
            ActionType.NOTIFY: ["notify", "alert", "inform", "report", "signal"]
        }
        
        # Präpositions-Patterns für Entity-Rollen
        self.preposition_roles = {
            "into": EntityRole.TARGET,
            "from": EntityRole.INPUT, 
            "with": EntityRole.TOOL,
            "through": EntityRole.TOOL,
            "at": EntityRole.TARGET,
            "in": EntityRole.TARGET,
            "on": EntityRole.TARGET,
            "to": EntityRole.TARGET
        }
    
    def analyze_sentence(self, sentence: str, step_id: str) -> Tuple[GrammaticalAction, List[EntityRelation]]:
        """
        Grammatische Analyse eines UC-Schritts
        
        Returns:
            GrammaticalAction: Grammatische Struktur des Satzes
            List[EntityRelation]: Gefundene Entity-Controller Beziehungen
        """
        doc = self.nlp(sentence)
        
        print(f"\n=== GRAMMATISCHE ANALYSE: {step_id} ===")
        print(f"Sentence: {sentence}")
        
        # 1. Dependency Parsing anzeigen
        print("\nDependency Parse:")
        for token in doc:
            print(f"  {token.text:12} | {token.pos_:8} | {token.dep_:12} | Head: {token.head.text}")
        
        # 2. Hauptverb finden
        main_verb = self._find_main_verb(doc)
        if not main_verb:
            print("❌ No main verb found!")
            return None, []
        
        print(f"\nMain Verb: {main_verb.text} (lemma: {main_verb.lemma_})")
        
        # 3. Verb kategorisieren
        action_type = self._categorize_verb(main_verb.lemma_)
        print(f"Action Type: {action_type}")
        
        # 4. Grammatische Objekte extrahieren
        direct_obj = self._find_direct_object(main_verb)
        indirect_obj = self._find_indirect_object(main_verb)
        prep_phrases = self._find_prepositional_phrases(main_verb)
        subjects = self._find_subjects(main_verb)
        
        print(f"Direct Object: {direct_obj}")
        print(f"Indirect Object: {indirect_obj}")
        print(f"Prepositional Phrases: {prep_phrases}")
        print(f"Subjects: {subjects}")
        
        # 5. GrammaticalAction erstellen
        action = GrammaticalAction(
            verb=main_verb.text,
            verb_lemma=main_verb.lemma_,
            action_type=action_type,
            direct_object=direct_obj,
            indirect_object=indirect_obj,
            prepositional_phrases=prep_phrases,
            subjects=subjects
        )
        
        # 6. Entity-Controller Beziehungen ableiten
        entity_relations = self._derive_entity_relations(action, step_id)
        
        print(f"\nEntity Relations:")
        for rel in entity_relations:
            print(f"  {rel.entity} ({rel.role.value}) -> {rel.controller}")
        
        return action, entity_relations
    
    def _find_main_verb(self, doc) -> Optional[Token]:
        """Finde das Hauptverb im Satz"""
        # Suche nach ROOT verb oder Hauptverb mit Objekten
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return token
            elif token.pos_ == "VERB" and any(child.dep_ in ["dobj", "pobj"] for child in token.children):
                return token
        return None
    
    def _categorize_verb(self, verb_lemma: str) -> ActionType:
        """Kategorisiere Verb nach Action Type"""
        verb_lemma = verb_lemma.lower()
        
        for action_type, verbs in self.verb_categories.items():
            if verb_lemma in verbs:
                return action_type
        
        # Default basierend auf häufigen Mustern
        if verb_lemma in ["make", "create", "produce", "generate"]:
            return ActionType.TRANSFORM
        elif verb_lemma in ["use", "apply", "employ"]:
            return ActionType.PROCESS
        else:
            return ActionType.PERFORM  # Default
    
    def _find_direct_object(self, verb: Token) -> Optional[str]:
        """Finde direktes Objekt des Verbs"""
        for child in verb.children:
            if child.dep_ == "dobj":
                # Erweitere um Attribute (adjectives etc.)
                obj_phrase = self._expand_noun_phrase(child)
                return obj_phrase
        return None
    
    def _find_indirect_object(self, verb: Token) -> Optional[str]:
        """Finde indirektes Objekt"""
        for child in verb.children:
            if child.dep_ == "iobj":
                return self._expand_noun_phrase(child)
        return None
    
    def _find_prepositional_phrases(self, verb: Token) -> List[Tuple[str, str]]:
        """Finde alle Präpositionsphrasen"""
        prep_phrases = []
        
        for child in verb.children:
            if child.dep_ == "prep":
                prep = child.text
                # Finde Objekt der Präposition
                for prep_child in child.children:
                    if prep_child.dep_ == "pobj":
                        obj = self._expand_noun_phrase(prep_child)
                        prep_phrases.append((prep, obj))
        
        return prep_phrases
    
    def _find_subjects(self, verb: Token) -> List[str]:
        """Finde alle Subjekte"""
        subjects = []
        
        for child in verb.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                subj = self._expand_noun_phrase(child)
                subjects.append(subj)
        
        return subjects
    
    def _expand_noun_phrase(self, noun: Token) -> str:
        """Erweitere Nomen um Adjektive, Artikel, etc."""
        # Sammle alle Tokens der Noun Phrase
        phrase_tokens = []
        
        # Adjektive und Determinanten vor dem Nomen
        for child in noun.children:
            if child.dep_ in ["amod", "det", "compound"]:
                phrase_tokens.append(child)
        
        phrase_tokens.append(noun)
        
        # Nachgestellte Modifikatoren
        for child in noun.children:
            if child.dep_ in ["prep", "relcl"]:
                phrase_tokens.append(child)
        
        # Sortiere nach Position im Satz
        phrase_tokens.sort(key=lambda x: x.i)
        
        return " ".join([token.text for token in phrase_tokens])
    
    def _derive_entity_relations(self, action: GrammaticalAction, step_id: str) -> List[EntityRelation]:
        """Leite Entity-Controller Beziehungen aus grammatischer Struktur ab"""
        relations = []
        
        # Controller Name aus Step ID ableiten (vereinfacht)
        controller_name = f"{step_id}_Controller"
        
        # Direct Object -> meist Input oder Tool
        if action.direct_object:
            entity_name = self._normalize_entity_name(action.direct_object)
            
            if action.action_type == ActionType.ACTIVATE:
                # "activates water heater" -> WaterHeater wird aktiviert (TOOL)
                relations.append(EntityRelation(
                    entity=entity_name,
                    role=EntityRole.TOOL,
                    controller=controller_name
                ))
            elif action.action_type == ActionType.TRANSFORM:
                # "grinds coffee beans" -> Coffee Beans werden verarbeitet (INPUT)
                relations.append(EntityRelation(
                    entity=entity_name,
                    role=EntityRole.INPUT,
                    controller=controller_name
                ))
        
        # Prepositional Phrases -> verschiedene Rollen
        for prep, obj in action.prepositional_phrases:
            entity_name = self._normalize_entity_name(obj)
            role = self.preposition_roles.get(prep, EntityRole.RESOURCE)
            
            relations.append(EntityRelation(
                entity=entity_name,
                role=role,
                controller=controller_name
            ))
        
        return relations
    
    def _normalize_entity_name(self, entity_text: str) -> str:
        """Normalisiere Entity Namen zu CamelCase"""
        # Entferne Artikel und konvertiere zu CamelCase
        words = entity_text.lower().split()
        filtered_words = [w for w in words if w not in ["the", "a", "an"]]
        return "".join([w.capitalize() for w in filtered_words])


def test_grammar_analyzer():
    """Test der grammatischen Analyse mit verschiedenen Domains"""
    
    analyzer = GrammarBasedSentenceAnalyzer()
    
    test_sentences = [
        ("B2a", "The system activates the water heater"),
        ("B2b", "The system prepares filter"),  
        ("B2c", "The system grinds the set amount at the set grinding degree directly into the filter"),
        ("B3a", "The system begins brewing coffee with the set water amount into the cup"),
        
        # UC3 - Rocket
        ("B2a", "The system performs final systems check"),
        ("B2c", "The system initiates engine ignition sequence"),
        ("B3", "The system monitors trajectory and guidance"),
        
        # UC4 - Nuclear  
        ("B2a", "The system immediately inserts control rods"),
        ("B2b", "The system activates emergency cooling"),
        ("B3", "The system monitors radiation levels"),
        
        # UC5 - Robot
        ("B2a", "The system scans component positions"),
        ("B3a", "The system positions component precisely"),
        ("B3b", "The system applies joining process")
    ]
    
    for step_id, sentence in test_sentences:
        action, relations = analyzer.analyze_sentence(sentence, step_id)
        print("\n" + "="*80)


if __name__ == "__main__":
    test_grammar_analyzer()