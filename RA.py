"""
Generische Robustheitsanalyse für RUP Use Cases
Domänenunabhängig - funktioniert für beliebige Anwendungsbereiche

Installation:
pip install spacy pandas networkx matplotlib scikit-learn numpy

python -m spacy download en_core_web_md
"""

import spacy
from spacy.matcher import Matcher, DependencyMatcher
from typing import List, Dict, Set, Tuple
import re
from collections import defaultdict
import json

class GenericRobustnessAnalyzer:
    """
    Domänenunabhängige Robustheitsanalyse
    Nutzt linguistische Patterns statt Domain-spezifischer Keywords
    """
    
    def __init__(self, language='en'):
        """Initialisiere Analyzer"""
        self.nlp = spacy.load('en_core_web_md' if language == 'en' else 'de_core_news_sm')
        self.language = language
        
        # Matcher für generische Patterns
        self.matcher = Matcher(self.nlp.vocab)
        self.dep_matcher = DependencyMatcher(self.nlp.vocab)
        
        # Speicher (inkl. actors)
        self.objects = {
            'boundary': set(),
            'entity': set(),
            'control': set(),
            'actor': set()
        }
        self.relationships = []
        self.feature_map = {}
        
        # Generische Pattern-Definitionen
        self._setup_generic_patterns()
        
        # Lernbare Klassifikation
        self.learned_objects = {
            'boundary': set(),
            'entity': set(),
            'control': set(),
            'actor': set()
        }
        
    def _setup_generic_patterns(self):
        """
        Setup generischer linguistischer Patterns
        Basiert auf grammatikalischen Strukturen, nicht auf Domain-Keywords
        """
        
        # ========== BOUNDARY OBJECT PATTERNS ==========
        
        # Pattern 1: "sends/displays/shows ... to actor"
        # Beispiel: "System sends notification to user"
        boundary_output = [
            {"LEMMA": {"IN": ["send", "display", "show", "output", "present", "provide", "notify"]}},
            {"OP": "*"},
            {"LOWER": "to"},
            {"POS": "NOUN"}
        ]
        self.matcher.add("BOUNDARY_OUTPUT", [boundary_output])
        
        # Pattern 2: "actor provides/enters/inputs ..."
        # Beispiel: "User enters data"
        boundary_input = [
            {"POS": "NOUN", "DEP": "nsubj"},
            {"LEMMA": {"IN": ["enter", "input", "provide", "select", "click", "choose", "request"]}},
            {"POS": "NOUN", "DEP": {"IN": ["dobj", "obj"]}}
        ]
        self.matcher.add("BOUNDARY_INPUT", [boundary_input])
        
        # Pattern 3: Trigger-Patterns
        # Beispiel: "timer reaches ...", "sensor detects ..."
        boundary_trigger = [
            {"POS": "NOUN"},
            {"LEMMA": {"IN": ["reach", "trigger", "detect", "sense", "fire", "activate"]}}
        ]
        self.matcher.add("BOUNDARY_TRIGGER", [boundary_trigger])
        
        # Pattern 4: Configuration/Settings
        # Beispiel: "configured time", "preset value"
        boundary_config = [
            {"LEMMA": {"IN": ["configure", "set", "preset", "define", "specify"]}},
            {"POS": "NOUN"}
        ]
        self.matcher.add("BOUNDARY_CONFIG", [boundary_config])
        
        # ========== ENTITY OBJECT PATTERNS ==========
        
        # Pattern 1: Preconditions "X is available/present/loaded"
        entity_precondition = [
            {"POS": "NOUN"},
            {"LEMMA": "be"},
            {"LEMMA": {"IN": ["available", "present", "loaded", "ready", "exist"]}}
        ]
        self.matcher.add("ENTITY_PRECONDITION", [entity_precondition])
        
        # Pattern 2: Physical manipulation "retrieve/get/place X from/in Y"
        entity_manipulation = [
            {"LEMMA": {"IN": ["retrieve", "get", "place", "put", "store", "load", "unload"]}},
            {"POS": "NOUN"},
            {"LOWER": {"IN": ["from", "in", "into", "to"]}}
        ]
        self.matcher.add("ENTITY_MANIPULATION", [entity_manipulation])
        
        # ========== CONTROL OBJECT PATTERNS ==========
        
        # Pattern 1: "System activates/starts/controls/manages ..."
        control_action = [
            {"LOWER": "system"},
            {"LEMMA": {"IN": ["activate", "start", "initiate", "control", "manage", 
                             "coordinate", "process", "calculate", "validate", "verify",
                             "begin", "stop", "pause", "resume", "execute"]}}
        ]
        self.matcher.add("CONTROL_ACTION", [control_action])
        
    def analyze_use_case(self, use_case: Dict) -> Dict:
        """Analysiere Use Case generisch"""
        print(f"\n{'='*60}")
        print(f"Analysiere: {use_case['name']}")
        print(f"{'='*60}")
        
        # Feature-Mapping
        capability = use_case.get('capability', 'Unknown')
        feature = use_case.get('feature', 'Unknown')
        if capability not in self.feature_map:
            self.feature_map[capability] = {}
        self.feature_map[capability][feature] = use_case['name']
        
        # Extrahiere und speichere Actors
        actors = use_case.get('actors', [])
        for actor in actors:
            if actor.lower() != 'system':  # System ist kein Actor
                self.objects['actor'].add(actor)
        
        # Analysiere Abschnitte
        self._analyze_preconditions_generic(use_case.get('preconditions', []))
        self._analyze_steps_generic(use_case.get('steps', []), use_case['name'], actors)
        self._analyze_alternatives_generic(use_case.get('alternatives', []), use_case['name'])
        self._analyze_extensions_generic(use_case.get('extensions', []), use_case['name'])
        
        return {
            'objects': {k: list(v) for k, v in self.objects.items()},
            'relationships': self.relationships
        }
    
    def _analyze_preconditions_generic(self, preconditions: List[str]):
        """Analysiere Vorbedingungen generisch"""
        print("\n--- Vorbedingungen ---")
        
        for precond in preconditions:
            doc = self.nlp(precond)
            
            # Pattern Matching
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                label = self.nlp.vocab.strings[match_id]
                
                if label == "ENTITY_PRECONDITION":
                    # Extrahiere das Substantiv (Entity)
                    entity = doc[start].text
                    if self._is_valid_entity(entity):
                        self.objects['entity'].add(entity)
                        print(f"  [Entity] {entity}")
            
            # Alle Substantive als potentielle Entities
            for token in doc:
                if token.pos_ == 'NOUN' and not token.is_stop:
                    entity = token.text
                    if self._is_valid_entity(entity) and len(entity) > 2:
                        self.objects['entity'].add(entity)
                        print(f"  [Entity] {entity}")
    
    def _analyze_steps_generic(self, steps: List[Dict], uc_name: str, actors: List[str]):
        """Analysiere Schritte generisch"""
        print("\n--- Ablaufschritte ---")
        
        for step in steps:
            step_id = step.get('id', '')
            step_text = step.get('text', '')
            step_type = step.get('type', 'action')
            
            doc = self.nlp(step_text)
            
            print(f"\n  {step_id}: {step_text[:60]}...")
            
            # 1. BOUNDARY OBJECTS durch Pattern Matching
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                label = self.nlp.vocab.strings[match_id]
                matched_span = doc[start:end]
                
                if label.startswith('BOUNDARY'):
                    boundary = self._extract_boundary_generic(doc, matched_span, label)
                    if boundary:
                        self.objects['boundary'].add(boundary)
                        print(f"    → [Boundary] {boundary}")
                
                elif label.startswith('CONTROL'):
                    control = self._extract_control_generic(doc, matched_span)
                    if control:
                        self.objects['control'].add(control)
                        print(f"    → [Control] {control}")
            
            # 2. ENTITY OBJECTS durch Dependency Parsing
            entities = self._extract_entities_from_dependencies(doc)
            for entity in entities:
                if self._is_valid_entity(entity):
                    self.objects['entity'].add(entity)
                    print(f"    → [Entity] {entity}")
            
            # 3. CONTROL OBJECTS durch Subjekt-Verb Analyse
            if step_text.lower().startswith('system') or step_text.lower().startswith('the system'):
                control = self._extract_control_from_verb(doc)
                if control:
                    self.objects['control'].add(control)
                    print(f"    → [Control] {control}")
            
            # 4. BEZIEHUNGEN extrahieren
            self._extract_relationships_generic(doc, step_text, step_id, uc_name, actors)
    
    def _extract_boundary_generic(self, doc, span, label: str) -> str:
        """Extrahiere Boundary Object generisch"""
        
        # Finde das relevante Substantiv im Span
        nouns = [token for token in span if token.pos_ == 'NOUN']
        
        if not nouns:
            return None
        
        # Baue Boundary-Namen
        if label == "BOUNDARY_OUTPUT":
            # "sends notification to user" → "Notification Interface"
            noun = nouns[0].text
            return f"{noun.capitalize()} Interface"
        
        elif label == "BOUNDARY_INPUT":
            # "user enters data" → "Data Input Interface"
            noun = nouns[-1].text if len(nouns) > 1 else nouns[0].text
            return f"{noun.capitalize()} Input Interface"
        
        elif label == "BOUNDARY_TRIGGER":
            # "timer reaches time" → "Timer Interface"
            noun = nouns[0].text
            return f"{noun.capitalize()} Interface"
        
        elif label == "BOUNDARY_CONFIG":
            # "configured amount" → "Configuration Interface"
            return "Configuration Interface"
        
        return None
    
    def _extract_control_generic(self, doc, span) -> str:
        """Extrahiere Control Object generisch"""
        
        # Finde Verb im Span
        verbs = [token for token in span if token.pos_ == 'VERB']
        if not verbs:
            return None
        
        verb = verbs[0]
        
        # Finde das Objekt des Verbs (was wird kontrolliert?)
        controlled_object = None
        for child in verb.children:
            if child.dep_ in ['dobj', 'obj', 'obl']:
                controlled_object = child.text
                break
        
        if controlled_object:
            return f"{controlled_object.capitalize()} Controller"
        else:
            # Generischer Controller basierend auf Verb
            return f"{verb.lemma_.capitalize()} Controller"
    
    def _extract_control_from_verb(self, doc) -> str:
        """Extrahiere Control basierend auf Hauptverb"""
        
        # Finde Root-Verb
        root_verb = None
        for token in doc:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                root_verb = token
                break
        
        if not root_verb:
            return None
        
        # Finde Objekt des Verbs
        obj = None
        for child in root_verb.children:
            if child.dep_ in ['dobj', 'obj']:
                obj = child
                break
        
        if obj:
            # "System activates heater" → "Heater Controller"
            return f"{obj.text.capitalize()} Controller"
        else:
            # Keine spezifisches Objekt → generischer Name
            return f"{root_verb.lemma_.capitalize()} Controller"
    
    def _extract_entities_from_dependencies(self, doc) -> Set[str]:
        """Extrahiere Entities durch Dependency-Analyse"""
        entities = set()
        
        # Suche nach Objekten von Verben
        for token in doc:
            if token.pos_ == 'VERB':
                for child in token.children:
                    if child.dep_ in ['dobj', 'obj', 'obl', 'pobj'] and child.pos_ == 'NOUN':
                        if self._is_valid_entity(child.text):
                            entities.add(child.text.capitalize())
        
        # Suche nach Substantiven in Präpositionalphrasen (from X, in Y, to Z)
        for token in doc:
            if token.dep_ == 'pobj' and token.pos_ == 'NOUN':
                if self._is_valid_entity(token.text):
                    entities.add(token.text.capitalize())
        
        return entities
    
    def _extract_relationships_generic(self, doc, text: str, step_id: str, uc_name: str, actors: List[str]):
        """Extrahiere Beziehungen generisch"""
        
        # Finde Subject-Verb-Object Tripel
        for token in doc:
            if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                subject = None
                obj = None
                
                # Finde Subject
                for child in token.children:
                    if child.dep_ in ['nsubj', 'nsubjpass']:
                        subject = child.text
                    if child.dep_ in ['dobj', 'obj']:
                        obj = child.text
                
                if subject and obj:
                    # Bestimme Objekttypen
                    subject_type = self._guess_object_type(subject, actors)
                    obj_type = self._guess_object_type(obj, actors)
                    
                    # Mappe zu Objektnamen
                    subject_name = self._map_to_object_name(subject, subject_type)
                    obj_name = self._map_to_object_name(obj, obj_type)
                    
                    if subject_name and obj_name:
                        self.relationships.append({
                            'from': subject_name,
                            'to': obj_name,
                            'type': token.lemma_,
                            'step': step_id,
                            'uc': uc_name
                        })
        
        # Spezielle Patterns für Robustness-Regeln
        
        # Actor → Boundary
        for actor in actors:
            if actor.lower() in text.lower():
                # Suche nach Boundary in diesem Schritt
                for boundary in self.objects['boundary']:
                    if any(word in text.lower() for word in boundary.lower().split()):
                        self.relationships.append({
                            'from': actor,
                            'to': boundary,
                            'type': 'interacts',
                            'step': step_id,
                            'uc': uc_name
                        })
        
        # System → Control (System-Aktionen werden zu Control Objects)
        if text.lower().startswith('system') or 'the system' in text.lower():
            # Finde alle Controls die in diesem Schritt erwähnt werden
            for control in self.objects['control']:
                if any(word in text.lower() for word in control.lower().split()):
                    # Orchestrator koordiniert andere Controller
                    self.relationships.append({
                        'from': 'System Orchestrator',
                        'to': control,
                        'type': 'coordinates',
                        'step': step_id,
                        'uc': uc_name
                    })
    
    def _guess_object_type(self, word: str, actors: List[str]) -> str:
        """Rate Objekttyp basierend auf Wort"""
        word_lower = word.lower()
        
        # Actor?
        if word in actors or word_lower in ['user', 'customer', 'operator', 'admin']:
            return 'actor'
        
        # System?
        if word_lower in ['system', 'application', 'software']:
            return 'control'
        
        # Prüfe gegen bekannte Objekte
        for obj_type in ['boundary', 'entity', 'control']:
            for obj in self.objects[obj_type]:
                if word_lower in obj.lower():
                    return obj_type
        
        # Default: Entity (Daten/Objekte sind meist das Ziel)
        return 'entity'
    
    def _map_to_object_name(self, word: str, obj_type: str) -> str:
        """Mappe Wort auf vollständigen Objektnamen"""
        word_lower = word.lower()
        
        # Actors werden direkt verwendet
        if obj_type == 'actor':
            return word.capitalize()
        
        # Suche in existierenden Objekten
        if obj_type in self.objects:
            for obj in self.objects[obj_type]:
                if word_lower in obj.lower():
                    return obj
        
        # Erstelle neuen Namen basierend auf Typ
        if obj_type == 'boundary':
            return f"{word.capitalize()} Interface"
        elif obj_type == 'control':
            return f"{word.capitalize()} Controller"
        elif obj_type == 'entity':
            return word.capitalize()
        else:
            return word.capitalize()
    
    def _is_valid_entity(self, word: str) -> bool:
        """Prüfe ob Wort ein valides Entity ist"""
        word_lower = word.lower()
        
        # Filter Stopwords und System-Keywords
        invalid = ['system', 'the', 'a', 'an', 'this', 'that', 'these', 'those']
        if word_lower in invalid:
            return False
        
        # Filter zu kurze Wörter
        if len(word) < 3:
            return False
        
        return True
    
    def _analyze_alternatives_generic(self, alternatives: List[Dict], uc_name: str):
        """Analysiere Fehlerszenarien generisch"""
        print("\n--- Alternative Szenarien ---")
        
        for alt in alternatives:
            alt_id = alt.get('id', '')
            condition = alt.get('condition', '')
            is_fatal = alt.get('fatal', False)
            
            print(f"\n  {alt_id}: {condition}")
            
            # Error Handler als Control Object
            error_controller = f"Error Handler {alt_id}"
            self.objects['control'].add(error_controller)
            print(f"    → [Control] {error_controller}")
    
    def _analyze_extensions_generic(self, extensions: List[Dict], uc_name: str):
        """Analysiere Erweiterungen generisch"""
        print("\n--- Erweiterungen ---")
        
        for ext in extensions:
            ext_id = ext.get('id', '')
            feature = ext.get('feature', '')
            text = ext.get('text', '')
            
            print(f"\n  {ext_id}: Feature '{feature}'")
            
            doc = self.nlp(text)
            
            # Extrahiere Objekte aus Extension
            entities = self._extract_entities_from_dependencies(doc)
            for entity in entities:
                if self._is_valid_entity(entity):
                    self.objects['entity'].add(entity)
                    print(f"    → [Entity] {entity} (optional)")
    
    def identify_shared_objects(self, use_cases: List[Dict]) -> Dict:
        """Identifiziere gemeinsame Objekte über UCs"""
        self.objects = {
            'boundary': set(),
            'entity': set(),
            'control': set(),
            'actor': set()
        }
        self.relationships = []
        
        uc_objects = {}
        
        for uc in use_cases:
            temp_objects = {
                'boundary': set(),
                'entity': set(),
                'control': set(),
                'actor': set()
            }
            old_objects = self.objects
            self.objects = temp_objects
            
            self.analyze_use_case(uc)
            
            uc_objects[uc['name']] = {
                'boundary': temp_objects['boundary'].copy(),
                'entity': temp_objects['entity'].copy(),
                'control': temp_objects['control'].copy(),
                'actor': temp_objects['actor'].copy()
            }
            
            for obj_type in ['boundary', 'entity', 'control', 'actor']:
                old_objects[obj_type].update(temp_objects[obj_type])
            
            self.objects = old_objects
        
        # Finde Shared Objects
        shared = {
            'boundary': set(),
            'entity': set(),
            'control': set(),
            'actor': set()
        }
        
        for obj_type in ['boundary', 'entity', 'control', 'actor']:
            for obj in self.objects[obj_type]:
                count = sum(1 for uc_obj in uc_objects.values() if obj in uc_obj[obj_type])
                if count > 1:
                    shared[obj_type].add(obj)
        
        # Füge System Orchestrator hinzu (immer vorhanden)
        self.objects['control'].add('System Orchestrator')
        
        print("\n" + "="*60)
        print("GEMEINSAME OBJEKTE")
        print("="*60)
        for obj_type in ['boundary', 'entity', 'control', 'actor']:
            if shared[obj_type]:
                print(f"\n{obj_type.upper()}:")
                for obj in sorted(shared[obj_type]):
                    print(f"  • {obj}")
        
        return {
            'all_objects': {k: list(v) for k, v in self.objects.items()},
            'shared_objects': {k: list(v) for k, v in shared.items()},
            'uc_objects': {k: {t: list(v[t]) for t in ['boundary', 'entity', 'control', 'actor']} 
                          for k, v in uc_objects.items()},
            'relationships': self.relationships,
            'features': self.feature_map
        }
    
    def export_to_json(self, filename: str = 'robustness_analysis.json'):
        """Exportiere zu JSON"""
        result = {
            'objects': {k: list(v) for k, v in self.objects.items()},
            'relationships': self.relationships,
            'features': self.feature_map
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Exportiert: {filename}")
    
    def generate_statistics(self) -> Dict:
        """Generiere Statistiken"""
        stats = {
            'actor_count': len(self.objects['actor']),
            'boundary_count': len(self.objects['boundary']),
            'entity_count': len(self.objects['entity']),
            'control_count': len(self.objects['control']),
            'relationship_count': len(self.relationships),
            'capability_count': len(self.feature_map)
        }
        
        print("\n" + "="*60)
        print("STATISTIKEN")
        print("="*60)
        for key, value in stats.items():
            print(f"{key:.<40} {value:>3}")
        
        return stats


# ============================================================================
# BEISPIEL
# ============================================================================

if __name__ == "__main__":
    
    # DEINE USE CASES - Kaffeezubereitung
    use_cases = [
        {
            'name': 'UC1: Prepare Latte',
            'capability': 'Coffee Preparation',
            'feature': 'Latte-Feature',
            'goal': 'User can drink their latte every morning at 7am',
            'actors': ['User'],
            'preconditions': [
                'Coffee beans are available in system',
                'Water is available in system',
                'Milk is available in system'
            ],
            'steps': [
                {'id': 'B1', 'type': 'trigger', 'text': 'System timer reaches configured time of 7:00am'},
                {'id': 'B2', 'type': 'action', 'text': 'System activates the water heater'},
                {'id': 'B3', 'type': 'action', 'text': 'System prepares the filter'},
                {'id': 'B4', 'type': 'action', 'text': 'System grinds the configured amount at configured grind level into the filter'},
                {'id': 'B5', 'type': 'action', 'text': 'System retrieves cup from storage container and places it under the filter'},
                {'id': 'B6', 'type': 'action', 'text': 'System heats cold water to brew temperature'},
                {'id': 'B7', 'type': 'action', 'text': 'System begins brewing coffee with configured hot water amount'},
                {'id': 'B8', 'type': 'action', 'text': 'System adds milk to the cup'},
                {'id': 'B9', 'type': 'action', 'text': 'System adds brewed coffee to the cup'},
                {'id': 'B10', 'type': 'action', 'text': 'System sends notification to user'},
                {'id': 'B11', 'type': 'action', 'text': 'User taks cup with the prepared milk coffee'}
            ],
            'alternatives': [
                {'id': 'A1', 'at': 'B2', 'condition': 'Water heater has insufficient water', 'fatal': True},
                {'id': 'A2', 'at': 'B7', 'condition': 'Insufficient milk', 'fatal': True}
            ],
            'extensions': [
                {'id': 'E1', 'at': 'B5-B8', 'feature': 'Sugar-Feature', 'text': 'System adds sugar to cup'}
            ]
        },
        {
            'name': 'UC2: Prepare Espresso',
            'capability': 'Coffee Preparation',
            'feature': 'Espresso-Feature',
            'goal': 'User wants to drink an espresso',
            'actors': ['User'],
            'preconditions': [
                'Coffee beans are available in system',
                'Water is available in system'
            ],
            'steps': [
                {'id': 'B1', 'type': 'trigger', 'text': 'System timer reaches configured time of 7:00am'},
                {'id': 'B2', 'type': 'action', 'text': 'System activates the water heater'},
                {'id': 'B3', 'type': 'action', 'text': 'System prepares the filter'},
                {'id': 'B4', 'type': 'action', 'text': 'System grinds the configured amount at configured grind level into the filter'},
                {'id': 'B5', 'type': 'action', 'text': 'System retrieves cup from storage container and places it under the filter'},
                {'id': 'B6', 'type': 'action', 'text': 'System starts water compressor to generate appropriate water pressure for espressos'},
                {'id': 'B7', 'type': 'action', 'text': 'System begins pressing hot water through the coffee grounds'},
                {'id': 'B8', 'type': 'action', 'text': 'System sends notification to user'}
            ],
            'alternatives': [
                {'id': 'A1', 'at': 'B6', 'condition': 'Compressor defect', 'fatal': True}
            ],
            'extensions': [
                {'id': 'E1', 'at': 'B5-B8', 'feature': 'Sugar-Feature', 'text': 'System adds sugar to cup'}
            ]
        }
    ]
    
    print("="*60)
    print("GENERISCHE ROBUSTNESS ANALYSE")
    print("Deine Kaffee Use Cases")
    print("="*60)
    
    print("\nInitialisiere Generic Robustness Analyzer...")
    analyzer = GenericRobustnessAnalyzer(language='en')
    
    print("\n" + "="*60)
    print("STARTE ANALYSE")
    print("="*60)
    
    results = analyzer.identify_shared_objects(use_cases)
    stats = analyzer.generate_statistics()
    analyzer.export_to_json('robustness_coffee.json')
    
    print("\n" + "="*60)
    print("ERKANNTE OBJEKTE")
    print("="*60)
    
    print("\nACTORS:")
    for actor in sorted(results['all_objects']['actor']):
        print(f"  ◆ {actor}")
    
    print("\nBOUNDARY OBJECTS:")
    for obj in sorted(results['all_objects']['boundary']):
        shared = " (SHARED)" if obj in results['shared_objects']['boundary'] else ""
        print(f"  ⭕ {obj}{shared}")
    
    print("\nENTITY OBJECTS:")
    for obj in sorted(results['all_objects']['entity']):
        shared = " (SHARED)" if obj in results['shared_objects']['entity'] else ""
        print(f"  ■ {obj}{shared}")
    
    print("\nCONTROL OBJECTS:")
    for obj in sorted(results['all_objects']['control']):
        shared = " (SHARED)" if obj in results['shared_objects']['control'] else ""
        print(f"  ▲ {obj}{shared}")
    
    print("\n" + "="*60)
    print("BEZIEHUNGEN (Auszug)")
    print("="*60)
    for rel in results['relationships'][:10]:
        print(f"  {rel['from']} --[{rel['type']}]--> {rel['to']}")
    if len(results['relationships']) > 10:
        print(f"  ... und {len(results['relationships']) - 10} weitere")
    
    print("\n✓ Analyse abgeschlossen!")
    print("\nDatei erstellt: robustness_coffee.json")
    print("\nNächste Schritte:")
    print("  1. Nutze RobustnessGraphVisualizer für Visualisierung")
    print("  2. python -c 'from graph_viz import *; v = RobustnessGraphVisualizer(); v.load_from_json(\"robustness_coffee.json\"); v.build_graph(); v.visualize_hierarchical()'")
    print("  3. Export nach Rhapsody (XMI)")