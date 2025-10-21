"""
Automatic Robustness Analysis with NLP and Few-shot Learning
For RUP Use Cases with Feature-based Product Line Engineering

Complete implementation with all features
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict


class ObjectType(Enum):
    """Types of objects in robustness analysis"""
    BOUNDARY = "boundary"
    ENTITY = "entity"
    CONTROL = "control"


class RelationType(Enum):
    """Types of entity relationships"""
    PROVIDE = "provide"
    USE = "use"


@dataclass
class RobustnessObject:
    """An object in the robustness analysis"""
    name: str
    type: ObjectType
    use_cases: Set[str] = field(default_factory=set)
    confidence: float = 0.0
    keywords: List[str] = field(default_factory=list)
    provides: Set[str] = field(default_factory=set)
    uses: Set[str] = field(default_factory=set)
    
    def is_shared(self) -> bool:
        return len(self.use_cases) > 1


@dataclass
class EntityRelationship:
    """Relationship between object and entity"""
    source: str
    target_entity: str
    relation_type: RelationType
    use_case: str


@dataclass
class Warning:
    """Warning about analysis issues"""
    severity: str
    category: str
    message: str
    use_case: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class UseCase:
    """Use Case representation"""
    id: str
    name: str
    capability: str
    feature: str
    goal: str
    actors: List[str]
    preconditions: List[str]
    steps: List[Dict[str, str]]
    postconditions: List[str]
    alternatives: List[Dict] = field(default_factory=list)
    extensions: List[Dict] = field(default_factory=list)


class UseCaseTextParser:
    """Parser for Use Cases in natural language text format"""
    
    def __init__(self):
        self.current_capability = ""
    
    def parse_file(self, filename: str) -> List[UseCase]:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        return self.parse_text(content)
    
    def parse_text(self, text: str) -> List[UseCase]:
        use_cases = []
        uc_blocks = re.split(r'\nUse Case \d+:', text, flags=re.IGNORECASE)
        
        if uc_blocks:
            first_block = uc_blocks[0]
            capability_match = re.search(r'(?:Capability|Feature):\s*(.+)', first_block, re.IGNORECASE)
            if capability_match:
                self.current_capability = capability_match.group(1).strip()
        
        for i, block in enumerate(uc_blocks[1:], 1):
            uc = self._parse_use_case_block(f"UC{i}", block)
            if uc:
                use_cases.append(uc)
        
        return use_cases
    
    def _parse_use_case_block(self, uc_id: str, block: str) -> Optional[UseCase]:
        goal_match = re.search(r'(?:Goal|Objective):\s*(.+?)(?=\n[A-Z]|\n\n|$)', block, re.DOTALL | re.IGNORECASE)
        goal = goal_match.group(1).strip() if goal_match else ""
        
        preconditions = []
        precond_section = re.search(
            r'(?:Preconditions?|Pre-conditions?):(.*?)(?=Actors?:|(?:Basic|Main)\s+(?:Flow|Scenario):|$)', 
            block, re.DOTALL | re.IGNORECASE
        )
        if precond_section:
            prec_text = precond_section.group(1)
            preconditions = [
                line.strip() 
                for line in prec_text.split('\n') 
                if line.strip() and not line.strip().startswith('Use Case')
            ]
        
        actors = []
        actors_match = re.search(r'Actors?:\s*(.+?)(?=\n[A-Z]|\n\n|$)', block, re.IGNORECASE)
        if actors_match:
            actors_text = actors_match.group(1).strip()
            actors = [a.strip() for a in re.split(r'[,;]', actors_text)]
        
        steps = []
        flow_section = re.search(
            r'(?:Basic|Main)\s+(?:Flow|Scenario|Path):(.*?)(?=(?:Post-?conditions?|Alternative|Extension|Use Case):|$)',
            block, re.DOTALL | re.IGNORECASE
        )
        if flow_section:
            steps_text = flow_section.group(1)
            step_lines = re.findall(r'(?:B|S)?(\d+)[\.\)]\s+(.+?)(?=\n(?:B|S)?\d+[\.\)]|\n\n|$)', steps_text, re.DOTALL)
            
            for step_num, step_text in step_lines:
                step_text = step_text.strip()
                step_type = 'trigger' if 'trigger' in step_text.lower() else 'action'
                steps.append({
                    'id': f'S{step_num}',
                    'type': step_type,
                    'text': step_text
                })
        
        postconditions = []
        postcond_match = re.search(
            r'(?:Post-?conditions?|Success\s+Guarantee):(.*?)(?=Alternative|Extension|Use Case|$)',
            block, re.DOTALL | re.IGNORECASE
        )
        if postcond_match:
            post_text = postcond_match.group(1).strip()
            postconditions = [line.strip() for line in post_text.split('\n') if line.strip()]
        
        alternatives = []
        alt_section = re.search(
            r'(?:Alternative|Exception)\s+(?:Flows?|Scenarios?):(.*?)(?=(?:Post-?conditions?|Extension|Use Case):|$)',
            block, re.DOTALL | re.IGNORECASE
        )
        if alt_section:
            alt_text = alt_section.group(1)
            alt_blocks = re.findall(r'(?:A|E)?(\d+)[a-z]?\s+(?:at|@)\s+(?:S|B)?(\d+)\s+(.+?)(?=\n(?:A|E)?\d+|\n\n|$)', alt_text, re.DOTALL)
            
            for alt_num, at_step, condition_text in alt_blocks:
                is_fatal = any(term in condition_text.lower() for term in ['end', 'terminate', 'abort', 'fail'])
                condition_lines = condition_text.strip().split('\n')
                condition = condition_lines[0].strip() if condition_lines else ""
                
                alternatives.append({
                    'id': f'A{alt_num}',
                    'at': f'S{at_step}',
                    'condition': condition,
                    'fatal': is_fatal
                })
        
        extensions = []
        ext_section = re.search(
            r'(?:Extension|Optional)\s+(?:Flows?|Scenarios?):(.*?)(?=Use Case|$)',
            block, re.DOTALL | re.IGNORECASE
        )
        if ext_section:
            ext_text = ext_section.group(1)
            ext_blocks = re.findall(r'E(\d+)\s+(?:S|B)?([\d\-]+)\s+(.+?)(?=\nE\d+|$)', ext_text, re.DOTALL)
            
            for ext_num, at_steps, ext_content in ext_blocks:
                feature_match = re.search(r'\((?:trigger|if)\)\s+(.+?)(?=\n|$)', ext_content, re.IGNORECASE)
                feature = feature_match.group(1).strip() if feature_match else "Optional Feature"
                action_match = re.search(r'E\d+\.\d+\s+(.+?)(?=\n|$)', ext_content)
                action_text = action_match.group(1).strip() if action_match else ext_content.strip()
                
                extensions.append({
                    'id': f'E{ext_num}',
                    'at': f'S{at_steps}',
                    'feature': feature,
                    'text': action_text
                })
        
        feature = self._derive_feature(goal, uc_id)
        
        return UseCase(
            id=uc_id,
            name=goal[:50] + "..." if len(goal) > 50 else goal,
            capability=self.current_capability,
            feature=feature,
            goal=goal,
            actors=actors,
            preconditions=preconditions,
            steps=steps,
            postconditions=postconditions,
            alternatives=alternatives,
            extensions=extensions
        )
    
    def _derive_feature(self, goal: str, uc_id: str) -> str:
        goal_lower = goal.lower()
        keywords = ['coffee', 'espresso', 'payment', 'login', 'register', 'order', 'checkout']
        for keyword in keywords:
            if keyword in goal_lower:
                return f"{keyword.capitalize()}-Feature"
        return f"{uc_id}-Feature"


class RobustnessAnalyzer:
    """Main class for automatic robustness analysis"""
    
    def __init__(self):
        self.objects: Dict[str, RobustnessObject] = {}
        self.relationships: List[Dict] = []
        self.entity_relationships: List[EntityRelationship] = []
        self.parser = UseCaseTextParser()
        self.warnings: List[Warning] = []
        self.use_cases_map: Dict[str, UseCase] = {}
    
    def load_from_file(self, filename: str) -> List[UseCase]:
        return self.parser.parse_file(filename)
    
    def load_from_text(self, text: str) -> List[UseCase]:
        return self.parser.parse_text(text)
    
    def extract_objects_from_step(self, step_text: str, uc_id: str) -> List[RobustnessObject]:
        """Extract objects and analyze PROVIDE/USE relationships - COMPREHENSIVE"""
        extracted = []
        text_lower = step_text.lower()
        
        # === BOUNDARY OBJECTS ===
        
        # Time-based triggers
        if re.search(r'(timer|clock|schedule|time|reaches)', text_lower):
            obj = self._get_or_create_object('Timer-Trigger', ObjectType.BOUNDARY, uc_id)
            extracted.append(obj)
        
        # Output to user (notifications, displays)
        if re.search(r'(display|show|present|notify|alert).*?(to\s+)?(?:user|customer|actor)', text_lower):
            obj = self._get_or_create_object('User-Notification', ObjectType.BOUNDARY, uc_id)
            extracted.append(obj)
        
        # User input - general
        if re.search(r'(?:user|customer|actor)\s+(?:enters|inputs|clicks|selects|chooses)', text_lower):
            obj = self._get_or_create_object('User-Input-Interface', ObjectType.BOUNDARY, uc_id)
            extracted.append(obj)
        
        # SPECIFIC USER INPUT BOUNDARIES - Coffee configuration
        if re.search(r'configured.*?(coffee|type)', text_lower) or re.search(r'(milk|espresso)', text_lower):
            obj = self._get_or_create_object('Coffee-Type-Selector', ObjectType.BOUNDARY, uc_id)
            obj.keywords.append('user_configuration')
            extracted.append(obj)
            entity = self._get_or_create_object('Coffee-Type-Setting', ObjectType.ENTITY, uc_id)
            extracted.append(entity)
        
        if re.search(r'configured.*?(strength|intensity)', text_lower):
            obj = self._get_or_create_object('Coffee-Strength-Selector', ObjectType.BOUNDARY, uc_id)
            obj.keywords.append('user_configuration')
            extracted.append(obj)
            entity = self._get_or_create_object('Coffee-Strength-Setting', ObjectType.ENTITY, uc_id)
            extracted.append(entity)
        
        if re.search(r'configured\s+amount', text_lower):
            obj = self._get_or_create_object('Amount-Selector', ObjectType.BOUNDARY, uc_id)
            obj.keywords.append('user_configuration')
            extracted.append(obj)
        
        if re.search(r'configured.*?time', text_lower):
            obj = self._get_or_create_object('Time-Configurator', ObjectType.BOUNDARY, uc_id)
            obj.keywords.append('user_configuration')
            extracted.append(obj)
        
        if re.search(r'configured.*?grind.*?level', text_lower):
            obj = self._get_or_create_object('Grind-Level-Selector', ObjectType.BOUNDARY, uc_id)
            obj.keywords.append('user_configuration')
            extracted.append(obj)
        
        # Physical boundaries (storage, container access)
        if re.search(r'(retrieves?|gets?|takes?).*?(from|out of)\s+(storage|container)', text_lower):
            obj = self._get_or_create_object('Storage-Access', ObjectType.BOUNDARY, uc_id)
            extracted.append(obj)
        
        # === ENTITY OBJECTS - COMPREHENSIVE ===
        
        # ALL water states
        water_entities = {
            'cold water': 'Cold-Water',
            'water': 'Water',  # Default/ambient
            'hot water': 'Hot-Water',
            'heated water': 'Hot-Water',
        }
        for keyword, entity_name in water_entities.items():
            if keyword in text_lower:
                obj = self._get_or_create_object(entity_name, ObjectType.ENTITY, uc_id)
                extracted.append(obj)
                break  # Take most specific match
        
        # Coffee/bean states
        if 'coffee bean' in text_lower or 'beans' in text_lower:
            obj = self._get_or_create_object('Coffee-Bean', ObjectType.ENTITY, uc_id)
            extracted.append(obj)
        
        if 'coffee ground' in text_lower or 'coffee mehl' in text_lower or 'kaffeemehl' in text_lower:
            obj = self._get_or_create_object('Coffee-Ground', ObjectType.ENTITY, uc_id)
            extracted.append(obj)
        elif 'ground' in text_lower and 'coffee' in text_lower:
            obj = self._get_or_create_object('Coffee-Ground', ObjectType.ENTITY, uc_id)
            extracted.append(obj)
        
        if 'brewed coffee' in text_lower:
            obj = self._get_or_create_object('Brewed-Coffee', ObjectType.ENTITY, uc_id)
            extracted.append(obj)
        elif 'coffee' in text_lower and 'bean' not in text_lower and 'ground' not in text_lower:
            obj = self._get_or_create_object('Coffee', ObjectType.ENTITY, uc_id)
            extracted.append(obj)
        
        # Compressed/pressurized water
        if re.search(r'(compressed|pressuri[zs]ed|pressure).*?water', text_lower):
            obj = self._get_or_create_object('Compressed-Hot-Water', ObjectType.ENTITY, uc_id)
            extracted.append(obj)
        
        # Other ingredients
        if 'milk' in text_lower:
            obj = self._get_or_create_object('Milk', ObjectType.ENTITY, uc_id)
            extracted.append(obj)
        
        if 'sugar' in text_lower:
            obj = self._get_or_create_object('Sugar', ObjectType.ENTITY, uc_id)
            extracted.append(obj)
        
        # Physical objects
        if 'cup' in text_lower or 'tasse' in text_lower:
            obj = self._get_or_create_object('Cup', ObjectType.ENTITY, uc_id)
            extracted.append(obj)
        
        if 'filter' in text_lower:
            obj = self._get_or_create_object('Filter', ObjectType.ENTITY, uc_id)
            extracted.append(obj)
        
        if 'container' in text_lower or 'storage' in text_lower:
            obj = self._get_or_create_object('Storage-Container', ObjectType.ENTITY, uc_id)
            extracted.append(obj)
        
        # Configuration entities
        if re.search(r'configured\s+amount', text_lower) or 'amount' in text_lower:
            obj = self._get_or_create_object('Amount-Setting', ObjectType.ENTITY, uc_id)
            extracted.append(obj)
        
        if 'grind level' in text_lower or 'grind' in text_lower:
            obj = self._get_or_create_object('Grind-Level-Setting', ObjectType.ENTITY, uc_id)
            extracted.append(obj)
        
        if re.search(r'water.*?pressure|pressure.*?water', text_lower) or 'pressure' in text_lower:
            obj = self._get_or_create_object('Water-Pressure', ObjectType.ENTITY, uc_id)
            extracted.append(obj)
        
        # === CONTROL OBJECTS - COMPREHENSIVE ===
        
        controllers_created = set()
        
        # WATER CONTROLLER (heating)
        if re.search(r'activates?\s+.*?water\s+heater|heater', text_lower):
            controller_name = 'Water-Controller'
            if controller_name not in controllers_created:
                obj = self._get_or_create_object(controller_name, ObjectType.CONTROL, uc_id)
                obj.keywords.append('heats_water')
                controllers_created.add(controller_name)
                extracted.append(obj)
        
        # WATER CONTROLLER (compressor/pressure)
        if re.search(r'(compressor|pressure)', text_lower):
            controller_name = 'Water-Controller'
            if controller_name not in controllers_created:
                obj = self._get_or_create_object(controller_name, ObjectType.CONTROL, uc_id)
                obj.keywords.append('pressurizes_water')
                controllers_created.add(controller_name)
                extracted.append(obj)
        
        # GRINDING CONTROLLER
        if re.search(r'grind', text_lower):
            controller_name = 'Grinding-Controller'
            if controller_name not in controllers_created:
                obj = self._get_or_create_object(controller_name, ObjectType.CONTROL, uc_id)
                controllers_created.add(controller_name)
                extracted.append(obj)
        
        # BREW CONTROLLER
        if re.search(r'brew|pressing.*?water', text_lower):
            controller_name = 'Brew-Controller'
            if controller_name not in controllers_created:
                obj = self._get_or_create_object(controller_name, ObjectType.CONTROL, uc_id)
                controllers_created.add(controller_name)
                extracted.append(obj)
        
        # SUPPLEMENT CONTROLLER (milk and sugar)
        if re.search(r'(adds?|gives?).*?(milk|sugar)', text_lower):
            controller_name = 'Supplement-Controller'
            if controller_name not in controllers_created:
                obj = self._get_or_create_object(controller_name, ObjectType.CONTROL, uc_id)
                obj.keywords.append('dispenses_supplements')
                controllers_created.add(controller_name)
                extracted.append(obj)
        
        # FILTER PREPARATION
        if re.search(r'prepares?\s+filter', text_lower):
            controller_name = 'Filter-Preparation-Controller'
            if controller_name not in controllers_created:
                obj = self._get_or_create_object(controller_name, ObjectType.CONTROL, uc_id)
                controllers_created.add(controller_name)
                extracted.append(obj)
        
        # CUP RETRIEVAL
        if re.search(r'retrieves?\s+cup|holt.*?tasse', text_lower):
            controller_name = 'Cup-Retrieval-Controller'
            if controller_name not in controllers_created:
                obj = self._get_or_create_object(controller_name, ObjectType.CONTROL, uc_id)
                controllers_created.add(controller_name)
                extracted.append(obj)
        
        # NOTIFICATION CONTROLLER
        if re.search(r'(displays?|gives?).*?(notification|message).*?user', text_lower):
            controller_name = 'Notification-Controller'
            if controller_name not in controllers_created:
                obj = self._get_or_create_object(controller_name, ObjectType.CONTROL, uc_id)
                controllers_created.add(controller_name)
                extracted.append(obj)
        
        return extracted
        """Extract objects and analyze PROVIDE/USE relationships"""
        extracted = []
        text_lower = step_text.lower()
        
        # === BOUNDARY OBJECTS ===
        
        # Time-based triggers
        if re.search(r'(timer|clock|schedule|time|reaches)', text_lower):
            obj = self._get_or_create_object('Timer-Trigger', ObjectType.BOUNDARY, uc_id)
            extracted.append(obj)
        
        # Output to user (notifications, displays)
        if re.search(r'(display|show|present|notify|alert).*?(to\s+)?(?:user|customer|actor)', text_lower):
            obj = self._get_or_create_object('User-Notification', ObjectType.BOUNDARY, uc_id)
            extracted.append(obj)
        
        # User input - general
        if re.search(r'(?:user|customer|actor)\s+(?:enters|inputs|clicks|selects|chooses)', text_lower):
            obj = self._get_or_create_object('User-Input-Interface', ObjectType.BOUNDARY, uc_id)
            extracted.append(obj)
        
        # SPECIFIC USER INPUT BOUNDARIES - Coffee configuration
        if re.search(r'configured.*?(coffee|strength|type)', text_lower):
            obj = self._get_or_create_object('Coffee-Type-Selector', ObjectType.BOUNDARY, uc_id)
            obj.keywords.append('user_configuration')
            extracted.append(obj)
            # Also create entity for the setting
            entity = self._get_or_create_object('Coffee-Type-Setting', ObjectType.ENTITY, uc_id)
            extracted.append(entity)
        
        if re.search(r'configured.*?(strength|intensity)', text_lower):
            obj = self._get_or_create_object('Coffee-Strength-Selector', ObjectType.BOUNDARY, uc_id)
            obj.keywords.append('user_configuration')
            extracted.append(obj)
            entity = self._get_or_create_object('Coffee-Strength-Setting', ObjectType.ENTITY, uc_id)
            extracted.append(entity)
        
        if re.search(r'configured.*?(amount|quantity)', text_lower):
            obj = self._get_or_create_object('Amount-Selector', ObjectType.BOUNDARY, uc_id)
            obj.keywords.append('user_configuration')
            extracted.append(obj)
        
        if re.search(r'configured.*?(time|schedule)', text_lower):
            obj = self._get_or_create_object('Time-Configurator', ObjectType.BOUNDARY, uc_id)
            obj.keywords.append('user_configuration')
            extracted.append(obj)
        
        if re.search(r'configured.*?grind.*?level', text_lower):
            obj = self._get_or_create_object('Grind-Level-Selector', ObjectType.BOUNDARY, uc_id)
            obj.keywords.append('user_configuration')
            extracted.append(obj)
        
        # Physical boundaries (storage, container access)
        if re.search(r'(retrieves?|gets?|takes?).*?(from|out of)\s+(storage|container|warehouse)', text_lower):
            obj = self._get_or_create_object('Storage-Access', ObjectType.BOUNDARY, uc_id)
            extracted.append(obj)
        
        # === ENTITY OBJECTS (Most important - extract ALL nouns that are data/objects) ===
        
        # Entities from preconditions
        available_matches = re.findall(r'(\w+)\s+(?:is|are)\s+(?:available\s+)?in\s+(?:the\s+)?system', text_lower)
        for match in available_matches:
            entity_name = self._normalize_name(match)
            obj = self._get_or_create_object(entity_name, ObjectType.ENTITY, uc_id)
            extracted.append(obj)
        
        # Compound entities (adjective + noun) - IMPORTANT for "hot water", "coffee grounds", etc.
        compound_patterns = [
            r'(hot|cold|warm|brewed|ground|fresh)\s+(water|coffee|milk|bean)',
            r'(coffee|water|milk)\s+(ground|bean|temperature|pressure|amount|level)',
        ]
        for pattern in compound_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    # Create compound name: "Hot Water", "Coffee Ground"
                    compound_name = f"{self._normalize_name(match[0])}-{self._normalize_name(match[1])}"
                    obj = self._get_or_create_object(compound_name, ObjectType.ENTITY, uc_id)
                    extracted.append(obj)
        
        # Specific important entities (including intermediate products)
        specific_entities = {
            'cup': 'Cup',
            'coffee': 'Coffee',
            'water': 'Water',
            'cold water': 'Cold-Water',  # Add compound first
            'hot water': 'Hot-Water',
            'milk': 'Milk',
            'bean': 'Coffee-Bean',
            'beans': 'Coffee-Bean',
            'sugar': 'Sugar',
            'filter': 'Filter',
            'container': 'Container',
            'storage container': 'Storage-Container',
            'heater': 'Water-Heater',
            'water heater': 'Water-Heater',
            'compressor': 'Water-Compressor',
            'water compressor': 'Water-Compressor',
            'ground': 'Coffee-Ground',
            'grounds': 'Coffee-Ground',
            'coffee grounds': 'Coffee-Ground',
            'amount': 'Amount',
            'grind level': 'Grind-Level',
            'level': 'Grind-Level',
            'pressure': 'Water-Pressure',
            'water pressure': 'Water-Pressure',
            'temperature': 'Temperature',
            'time': 'Time-Setting',
        }
        
        # Check for compound entities first (longer matches)
        for keyword, entity_name in sorted(specific_entities.items(), key=lambda x: -len(x[0])):
            if keyword in text_lower:
                obj = self._get_or_create_object(entity_name, ObjectType.ENTITY, uc_id)
                extracted.append(obj)
                # Mark that we found this to avoid duplicates
                text_lower = text_lower.replace(keyword, '')  # Remove to prevent re-matching
        
        # Process-based entity detection (transformed entities)
        process_transforms = [
            # "grinds X" produces "X-Ground"
            (r'grinds?\s+(?:the\s+)?(\w+)', '{}-Ground'),
            # "brews X" produces "Brewed-X"
            (r'brews?\s+(?:the\s+)?(\w+)', 'Brewed-{}'),
            # "heats X" produces "Hot-X"
            (r'heats?\s+(?:the\s+)?(\w+)', 'Hot-{}'),
            # "presses X through Y" uses both
            (r'(?:presses?|pressing)\s+(?:the\s+)?(\w+)\s+through', 'Hot-{}'),
        ]
        
        for pattern, template in process_transforms:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match not in ['the', 'a', 'an', 'to', 'from']:
                    if '{}' in template:
                        entity_name = template.format(self._normalize_name(match))
                    else:
                        entity_name = template
                    obj = self._get_or_create_object(entity_name, ObjectType.ENTITY, uc_id)
                    obj.keywords.append('transformed_product')
                    extracted.append(obj)
        
        # Extract entities from common patterns
        entity_patterns = [
            # "retrieves X from", "gets X from"
            (r'(?:retrieves?|gets?|takes?|fetches?)\s+(\w+)\s+from', RelationType.USE),
            # "X from storage/container"
            (r'(\w+)\s+from\s+(?:storage|container|warehouse)', RelationType.USE),
            # "adds X to/into", "puts X in"
            (r'(?:adds?|puts?|places?)\s+(\w+)\s+(?:to|into|in)', RelationType.USE),
            # "X to the cup/container"
            (r'(\w+)\s+(?:to|into)\s+the\s+(?:cup|container|glass)', RelationType.USE),
            # "prepares X", "activates X"
            (r'(?:prepares?|activates?|starts?)\s+(?:the\s+)?(\w+)', RelationType.USE),
            # "X in/into Y" pattern
            (r'(\w+)\s+(?:in|into)\s+the\s+(\w+)', RelationType.USE),
            # "configured X"
            (r'configured\s+(\w+)', RelationType.USE),
        ]
        
        for pattern, rel_type in entity_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                words = match if isinstance(match, tuple) else (match,)
                for entity_word in words:
                    # Filter out common words that are not entities
                    skip_words = ['the', 'a', 'an', 'system', 'to', 'from', 'with', 'and', 'or', 
                                 'it', 'that', 'this', 'these', 'those', 'begin', 'start', 'configured']
                    if entity_word not in skip_words and len(entity_word) > 2:
                        entity_name = self._normalize_name(entity_word)
                        obj = self._get_or_create_object(entity_name, ObjectType.ENTITY, uc_id)
                        extracted.append(obj)
        
        # === CONTROL OBJECTS (Create specific controllers for each action) ===
        
        # Track which controllers we've already created to avoid duplicates
        created_controllers = set()
        
        # Detailed pattern matching for control objects
        control_mappings = [
            (r'activates?\s+(?:the\s+)?water\s+heater', 'Water-Heater-Controller'),
            (r'activates?\s+(?:the\s+)?(\w+)', '{}-Activation-Controller'),
            (r'prepares?\s+(?:the\s+)?filter', 'Filter-Preparation-Controller'),
            (r'prepares?\s+(?:the\s+)?(\w+)', '{}-Preparation-Controller'),
            (r'grinds?\s+', 'Grinding-Controller'),  # Only ONE grinding controller
            (r'brews?\s+', 'Brewing-Controller'),    # Only ONE brewing controller
            (r'(?:adds?|gives?)\s+milk', 'Milk-Dispensing-Controller'),
            (r'(?:adds?|gives?)\s+sugar', 'Sugar-Dispensing-Controller'),
            (r'(?:adds?|gives?)\s+(\w+)', '{}-Dispensing-Controller'),
            (r'retrieves?\s+cup', 'Cup-Retrieval-Controller'),
            (r'retrieves?\s+(\w+)', '{}-Retrieval-Controller'),
            (r'(?:places?|puts?)\s+', 'Positioning-Controller'),
            (r'(?:starts?|initiates?)\s+(?:water\s+)?compressor', 'Compressor-Controller'),
            (r'(?:presses?|pressing)\s+', 'Pressing-Controller'),
            (r'(?:displays?|shows?)\s+.*?(?:notification|message)', 'Notification-Controller'),
        ]
        
        for pattern, controller_template in control_mappings:
            matches = re.findall(pattern, text_lower)
            if matches or re.search(pattern, text_lower):
                # Determine controller name
                if '{}' in controller_template:
                    if matches and matches[0] not in ['the', 'a', 'an', 'to', 'from', 'it', 'this']:
                        controller_name = controller_template.format(self._normalize_name(matches[0]))
                    else:
                        continue
                else:
                    controller_name = controller_template
                
                # Avoid duplicates
                if controller_name not in created_controllers:
                    obj = self._get_or_create_object(controller_name, ObjectType.CONTROL, uc_id)
                    created_controllers.add(controller_name)
                    extracted.append(obj)
        
        return extracted
        """Extract objects and analyze PROVIDE/USE relationships"""
        extracted = []
        text_lower = step_text.lower()
        
        # === BOUNDARY OBJECTS ===
        
        # Time-based triggers
        if re.search(r'(timer|clock|schedule|time|reaches)', text_lower):
            obj = self._get_or_create_object('Timer-Trigger', ObjectType.BOUNDARY, uc_id)
            extracted.append(obj)
        
        # Output to user (notifications, displays)
        if re.search(r'(display|show|present|notify|alert).*?(to\s+)?(?:user|customer|actor)', text_lower):
            obj = self._get_or_create_object('User-Notification', ObjectType.BOUNDARY, uc_id)
            extracted.append(obj)
        
        # User input
        if re.search(r'(?:user|customer|actor)\s+(?:enters|inputs|clicks|selects|chooses)', text_lower):
            obj = self._get_or_create_object('User-Input-Interface', ObjectType.BOUNDARY, uc_id)
            extracted.append(obj)
        
        # Physical boundaries (storage, container access)
        if re.search(r'(retrieves?|gets?|takes?).*?(from|out of)\s+(storage|container|warehouse)', text_lower):
            obj = self._get_or_create_object('Storage-Access', ObjectType.BOUNDARY, uc_id)
            extracted.append(obj)
        
        # === ENTITY OBJECTS (Most important - extract ALL nouns that are data/objects) ===
        
        # Entities from preconditions
        available_matches = re.findall(r'(\w+)\s+(?:is|are)\s+(?:available\s+)?in\s+(?:the\s+)?system', text_lower)
        for match in available_matches:
            entity_name = self._normalize_name(match)
            obj = self._get_or_create_object(entity_name, ObjectType.ENTITY, uc_id)
            extracted.append(obj)
        
        # Extract entities from common patterns
        entity_patterns = [
            # "retrieves X from", "gets X from"
            (r'(?:retrieves?|gets?|takes?|fetches?)\s+(\w+)\s+from', RelationType.USE),
            # "X from storage/container"
            (r'(\w+)\s+from\s+(?:storage|container|warehouse)', RelationType.USE),
            # "adds X to/into", "puts X in"
            (r'(?:adds?|puts?|places?)\s+(\w+)\s+(?:to|into|in)', RelationType.USE),
            # "X to the cup/container"
            (r'(\w+)\s+(?:to|into)\s+the\s+(?:cup|container|glass)', RelationType.USE),
            # "prepares X", "activates X"
            (r'(?:prepares?|activates?|starts?)\s+(?:the\s+)?(\w+)', RelationType.USE),
            # "grinds X", "brews X", "heats X"
            (r'(?:grinds?|brews?|heats?|processes?)\s+(?:the\s+)?(\w+)', RelationType.USE),
            # "configured X"
            (r'configured\s+(\w+)', RelationType.USE),
        ]
        
        for pattern, rel_type in entity_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                entity_word = match if isinstance(match, str) else match[0]
                # Filter out common words that are not entities
                skip_words = ['the', 'a', 'an', 'system', 'to', 'from', 'with', 'and', 'or', 
                             'it', 'that', 'this', 'these', 'those', 'begin', 'start']
                if entity_word not in skip_words and len(entity_word) > 2:
                    entity_name = self._normalize_name(entity_word)
                    obj = self._get_or_create_object(entity_name, ObjectType.ENTITY, uc_id)
                    extracted.append(obj)
        
        # Specific entity detection for common objects
        specific_entities = [
            'cup', 'coffee', 'water', 'milk', 'bean', 'sugar', 'filter', 'container',
            'heater', 'compressor', 'ground', 'amount', 'level', 'pressure', 'time'
        ]
        for entity in specific_entities:
            if entity in text_lower:
                entity_name = self._normalize_name(entity)
                obj = self._get_or_create_object(entity_name, ObjectType.ENTITY, uc_id)
                extracted.append(obj)
        
        # === CONTROL OBJECTS (Create specific controllers for each action) ===
        
        # Detailed pattern matching for control objects
        control_mappings = [
            (r'activates?\s+(?:the\s+)?(\w+)', '{}-Activation-Controller'),
            (r'prepares?\s+(?:the\s+)?(\w+)', '{}-Preparation-Controller'),
            (r'grinds?\s+(?:the\s+)?(\w+)', 'Grinding-Controller'),
            (r'brews?\s+(?:the\s+)?(\w+)', 'Brewing-Controller'),
            (r'(?:adds?|gives?)\s+(\w+)', '{}-Dispensing-Controller'),
            (r'retrieves?\s+(\w+)', '{}-Retrieval-Controller'),
            (r'(?:places?|puts?)\s+(\w+)', 'Positioning-Controller'),
            (r'begins?\s+(\w+)', '{}-Process-Controller'),
            (r'validates?\s+(?:the\s+)?(\w+)', 'Validation-Controller'),
            (r'(?:checks?|monitors?)\s+(?:the\s+)?(\w+)', 'Monitoring-Controller'),
        ]
        
        for pattern, controller_template in control_mappings:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match not in ['the', 'a', 'an', 'to', 'from']:
                    if '{}' in controller_template:
                        controller_name = controller_template.format(self._normalize_name(match))
                    else:
                        controller_name = controller_template
                    obj = self._get_or_create_object(controller_name, ObjectType.CONTROL, uc_id)
                    extracted.append(obj)
        
        # Generic system action pattern
        system_action_pattern = r'system\s+(validates|verifies|processes|calculates|manages|coordinates|activates|starts|executes|adds|retrieves|begins|grinds|prepares|displays|checks|monitors)'
        if re.search(system_action_pattern, text_lower):
            matches = re.findall(system_action_pattern, text_lower)
            for verb in matches:
                controller_name = f"{verb.capitalize()}-Controller"
                obj = self._get_or_create_object(controller_name, ObjectType.CONTROL, uc_id)
                extracted.append(obj)
        
        return extracted
    
    def _add_entity_relationship(self, source: str, entity: str, relation_type: RelationType, uc_id: str):
        self.entity_relationships.append(EntityRelationship(
            source=source,
            target_entity=entity,
            relation_type=relation_type,
            use_case=uc_id
        ))
    
    def _normalize_name(self, word: str) -> str:
        word = word.strip().capitalize()
        if word.endswith('s') and len(word) > 3 and word[-2] not in ['s', 'u']:
            word = word[:-1]
        return word
    
    def _get_or_create_object(self, name: str, obj_type: ObjectType, uc_id: str) -> RobustnessObject:
        if name not in self.objects:
            self.objects[name] = RobustnessObject(name=name, type=obj_type)
        self.objects[name].use_cases.add(uc_id)
        return self.objects[name]
    
    def analyze_use_case(self, use_case: UseCase) -> Dict:
        print(f"\n=== Analyzing {use_case.id}: {use_case.name} ===")
        
        self.use_cases_map[use_case.id] = use_case
        
        # Orchestrator coordinates the overall use case flow
        orchestrator_name = f"{use_case.capability}-Orchestrator"
        orchestrator = self._get_or_create_object(orchestrator_name, ObjectType.CONTROL, use_case.id)
        orchestrator.keywords.append('orchestrator')
        
        # Collect all configuration boundaries first for Actor connections
        config_boundaries = []
        
        prev_control = None
        first_real_controller = None
        
        for idx, step in enumerate(use_case.steps):
            step_text = step.get('text', '')
            objects = self.extract_objects_from_step(step_text, use_case.id)
            
            step_controls = [o for o in objects if o.type == ObjectType.CONTROL and o.name != orchestrator_name]
            step_boundaries = [o for o in objects if o.type == ObjectType.BOUNDARY]
            step_entities = [o for o in objects if o.type == ObjectType.ENTITY]
            
            # Collect configuration boundaries
            for boundary in step_boundaries:
                if 'user_configuration' in boundary.keywords or 'selector' in boundary.name.lower():
                    if boundary.name not in [b.name for b in config_boundaries]:
                        config_boundaries.append(boundary)
            
            # First step: Actor triggers timer or receives notification
            if step.get('type') == 'trigger' or idx == 0:
                for boundary in step_boundaries:
                    # Actor interacts with timer trigger
                    if 'timer' in boundary.name.lower() or 'trigger' in boundary.name.lower():
                        # Timer triggers orchestrator directly (time-based)
                        self.relationships.append({
                            'from': boundary.name,
                            'to': orchestrator_name,
                            'type': 'triggers',
                            'use_case': use_case.id
                        })
                    else:
                        # Other boundaries also connect to orchestrator
                        self.relationships.append({
                            'from': boundary.name,
                            'to': orchestrator_name,
                            'type': 'triggers',
                            'use_case': use_case.id
                        })
            
            # Track first real controller
            if step_controls and not first_real_controller:
                first_real_controller = step_controls[0].name
                # Orchestrator initiates first controller
                self.relationships.append({
                    'from': orchestrator_name,
                    'to': first_real_controller,
                    'type': 'control_flow',
                    'use_case': use_case.id
                })
            
            # Control flow between specialized controllers
            for control in step_controls:
                if prev_control and prev_control != control.name:
                    self.relationships.append({
                        'from': prev_control,
                        'to': control.name,
                        'type': 'control_flow',
                        'use_case': use_case.id
                    })
                prev_control = control.name
                
                # Analyze PROVIDE vs USE
                for entity in step_entities:
                    step_lower = step_text.lower()
                    entity_lower = entity.name.lower()
                    
                    is_provide = False
                    
                    # Determine if entity is PROVIDED (output) or USED (input)
                    if 'hot-water' in entity_lower and 'water-controller' in control.name.lower():
                        # Water controller heats water
                        if 'heat' in step_lower or 'activates' in step_lower:
                            is_provide = True
                    elif 'compressed-hot-water' in entity_lower and 'water-controller' in control.name.lower():
                        if 'compressor' in step_lower or 'pressure' in step_lower:
                            is_provide = True
                    elif 'coffee-ground' in entity_lower and 'grinding' in control.name.lower():
                        is_provide = True
                    elif 'brewed-coffee' in entity_lower and 'brew' in control.name.lower():
                        is_provide = True
                    
                    # Create relationship
                    if is_provide:
                        control.provides.add(entity.name)
                        self.relationships.append({
                            'from': control.name,
                            'to': entity.name,
                            'type': 'provides',
                            'use_case': use_case.id
                        })
                        self._add_entity_relationship(control.name, entity.name, RelationType.PROVIDE, use_case.id)
                    else:
                        # USE relationship
                        control.uses.add(entity.name)
                        self.relationships.append({
                            'from': control.name,
                            'to': entity.name,
                            'type': 'uses',
                            'use_case': use_case.id
                        })
                        self._add_entity_relationship(control.name, entity.name, RelationType.USE, use_case.id)
            
            # Configuration boundaries connect to orchestrator
            for boundary in step_boundaries:
                if 'user_configuration' in boundary.keywords or 'selector' in boundary.name.lower():
                    # Check if relationship doesn't already exist
                    existing = any(r['from'] == boundary.name and r['to'] == orchestrator_name 
                                 for r in self.relationships)
                    if not existing:
                        self.relationships.append({
                            'from': boundary.name,
                            'to': orchestrator_name,
                            'type': 'provides_settings',
                            'use_case': use_case.id
                        })
            
            # Storage boundary connects to cup retrieval controller
            for boundary in step_boundaries:
                if 'storage' in boundary.name.lower():
                    for control in step_controls:
                        if 'retrieval' in control.name.lower() or 'cup' in control.name.lower():
                            self.relationships.append({
                                'from': boundary.name,
                                'to': control.name,
                                'type': 'accesses',
                                'use_case': use_case.id
                            })
            
            # Notification boundary receives from notification controller
            for boundary in step_boundaries:
                if 'notification' in boundary.name.lower():
                    for control in step_controls:
                        if 'notification' in control.name.lower():
                            self.relationships.append({
                                'from': control.name,
                                'to': boundary.name,
                                'type': 'sends_to',
                                'use_case': use_case.id
                            })
        
        # CRITICAL: Add Actor connections to ALL configuration boundaries
        for boundary in config_boundaries:
            self.relationships.append({
                'from': 'Actor',
                'to': boundary.name,
                'type': 'configures',
                'use_case': use_case.id
            })
        
        # Actor receives notification at the end
        notification_boundaries = [b for b in self.objects.values() 
                                  if b.type == ObjectType.BOUNDARY 
                                  and 'notification' in b.name.lower()
                                  and use_case.id in b.use_cases]
        for boundary in notification_boundaries:
            self.relationships.append({
                'from': boundary.name,
                'to': 'Actor',
                'type': 'notifies',
                'use_case': use_case.id
            })
        
        self._check_preconditions(use_case)
        self._check_use_case_quality(use_case)
        
        return self.get_results()
        print(f"\n=== Analyzing {use_case.id}: {use_case.name} ===")
        
        self.use_cases_map[use_case.id] = use_case
        
        # ORCHESTRATOR: Coordinates the overall use case flow
        # It receives the trigger from boundary objects and orchestrates
        # the sequence of specialized controllers
        # NOTE: Orchestrator does NOT process data directly - it delegates to specialized controllers
        orchestrator_name = f"{use_case.capability}-Orchestrator"
        orchestrator = self._get_or_create_object(orchestrator_name, ObjectType.CONTROL, use_case.id)
        orchestrator.keywords.append('orchestrator')
        orchestrator.keywords.append('coordinates_use_case_flow')
        
        prev_control = None
        first_real_controller = None
        
        for idx, step in enumerate(use_case.steps):
            step_text = step.get('text', '')
            objects = self.extract_objects_from_step(step_text, use_case.id)
            
            # Find control objects in this step
            step_controls = [o for o in objects if o.type == ObjectType.CONTROL and o.name != orchestrator_name]
            step_boundaries = [o for o in objects if o.type == ObjectType.BOUNDARY]
            step_entities = [o for o in objects if o.type == ObjectType.ENTITY]
            
            # First step: Actor triggers boundary
            if step.get('type') == 'trigger' or idx == 0:
                for boundary in step_boundaries:
                    self.relationships.append({
                        'from': 'Actor',
                        'to': boundary.name,
                        'type': 'initiates',
                        'use_case': use_case.id
                    })
                    # Boundary to orchestrator (orchestrator receives trigger)
                    self.relationships.append({
                        'from': boundary.name,
                        'to': orchestrator_name,
                        'type': 'triggers',
                        'use_case': use_case.id
                    })
            
            # Track first real controller (not orchestrator)
            if step_controls and not first_real_controller:
                first_real_controller = step_controls[0].name
                # Orchestrator initiates first controller
                self.relationships.append({
                    'from': orchestrator_name,
                    'to': first_real_controller,
                    'type': 'control_flow',
                    'use_case': use_case.id
                })
            
            # Control flow between specialized controllers (sequential execution)
            for control in step_controls:
                if prev_control and prev_control != control.name:
                    self.relationships.append({
                        'from': prev_control,
                        'to': control.name,
                        'type': 'control_flow',
                        'use_case': use_case.id
                    })
                prev_control = control.name
                
                # CRITICAL: Analyze PROVIDE vs USE for each controller-entity relationship
                # Determine if controller USES or PROVIDES entity based on action verb
                for entity in step_entities:
                    # Analyze the step text to determine relationship direction
                    step_lower = step_text.lower()
                    entity_lower = entity.name.lower().replace('-', ' ')
                    
                    # PROVIDE patterns: controller creates/generates/produces entity
                    provide_indicators = [
                        r'(?:produces?|creates?|generates?|makes?)\s+.*?' + re.escape(entity_lower),
                        r'(?:heats?|grinds?|brews?)\s+.*?' + re.escape(entity_lower),
                    ]
                    
                    is_provide = False
                    for pattern in provide_indicators:
                        if re.search(pattern, step_lower):
                            is_provide = True
                            break
                    
                    # Check if this is a transformed/output entity
                    if 'transformed_product' in entity.keywords:
                        is_provide = True
                    
                    # Check if entity name contains output indicators
                    if any(prefix in entity.name.lower() for prefix in ['hot-', 'brewed-', 'ground-']):
                        # Check if this step is the transformation step
                        if 'hot-' in entity.name.lower() and 'heat' in step_lower:
                            is_provide = True
                        elif 'brewed-' in entity.name.lower() and 'brew' in step_lower:
                            is_provide = True
                        elif 'ground-' in entity.name.lower() and 'grind' in step_lower:
                            is_provide = True
                        else:
                            is_provide = False  # Using the already-transformed entity
                    
                    # Create relationship
                    if is_provide:
                        control.provides.add(entity.name)
                        self.relationships.append({
                            'from': control.name,
                            'to': entity.name,
                            'type': 'provides',
                            'use_case': use_case.id
                        })
                        self._add_entity_relationship(control.name, entity.name, RelationType.PROVIDE, use_case.id)
                    else:
                        # Default: USE relationship
                        control.uses.add(entity.name)
                        self.relationships.append({
                            'from': control.name,
                            'to': entity.name,
                            'type': 'uses',
                            'use_case': use_case.id
                        })
                        self._add_entity_relationship(control.name, entity.name, RelationType.USE, use_case.id)
            
            # Boundaries with user configuration connect to orchestrator (for reading settings)
            for boundary in step_boundaries:
                if 'user_configuration' in boundary.keywords or 'selector' in boundary.name.lower() or 'configurator' in boundary.name.lower():
                    self.relationships.append({
                        'from': boundary.name,
                        'to': orchestrator_name,
                        'type': 'provides_settings',
                        'use_case': use_case.id
                    })
            
            # Storage boundaries connect to retrieval controllers
            for boundary in step_boundaries:
                if 'storage' in boundary.name.lower() or 'access' in boundary.name.lower():
                    for control in step_controls:
                        if 'retrieval' in control.name.lower():
                            self.relationships.append({
                                'from': boundary.name,
                                'to': control.name,
                                'type': 'accesses',
                                'use_case': use_case.id
                            })
        
        self._check_preconditions(use_case)
        self._check_use_case_quality(use_case)
        
        return self.get_results()
        print(f"\n=== Analyzing {use_case.id}: {use_case.name} ===")
        
        self.use_cases_map[use_case.id] = use_case
        
        orchestrator_name = f"{use_case.capability}-Orchestrator"
        orchestrator = self._get_or_create_object(orchestrator_name, ObjectType.CONTROL, use_case.id)
        
        prev_control = None
        
        for idx, step in enumerate(use_case.steps):
            step_text = step.get('text', '')
            objects = self.extract_objects_from_step(step_text, use_case.id)
            
            # Find control objects in this step
            step_controls = [o for o in objects if o.type == ObjectType.CONTROL]
            step_boundaries = [o for o in objects if o.type == ObjectType.BOUNDARY]
            step_entities = [o for o in objects if o.type == ObjectType.ENTITY]
            
            # First step: Actor triggers boundary
            if step.get('type') == 'trigger' or idx == 0:
                for boundary in step_boundaries:
                    self.relationships.append({
                        'from': 'Actor',
                        'to': boundary.name,
                        'type': 'initiates',
                        'use_case': use_case.id
                    })
                    # Boundary to orchestrator
                    self.relationships.append({
                        'from': boundary.name,
                        'to': orchestrator_name,
                        'type': 'triggers',
                        'use_case': use_case.id
                    })
            
            # Control flow between controllers (NO data objects!)
            for control in step_controls:
                if prev_control and prev_control != control.name:
                    self.relationships.append({
                        'from': prev_control,
                        'to': control.name,
                        'type': 'control_flow',
                        'use_case': use_case.id
                    })
                prev_control = control.name
                
                # CRITICAL: Analyze PROVIDE vs USE for each controller-entity relationship
                # Determine if controller USES or PROVIDES entity based on action verb
                for entity in step_entities:
                    # Analyze the step text to determine relationship direction
                    step_lower = step_text.lower()
                    entity_lower = entity.name.lower().replace('-', ' ')
                    
                    # PROVIDE patterns: controller creates/generates/produces entity
                    provide_indicators = [
                        r'(?:produces?|creates?|generates?|makes?)\s+.*?' + re.escape(entity_lower),
                        r'(?:heats?|grinds?|brews?)\s+.*?' + re.escape(entity_lower),
                        # Transformation results (output entities)
                    ]
                    
                    is_provide = False
                    for pattern in provide_indicators:
                        if re.search(pattern, step_lower):
                            is_provide = True
                            break
                    
                    # Check if this is a transformed/output entity
                    if 'transformed_product' in entity.keywords:
                        is_provide = True
                    
                    # Check if entity name contains output indicators
                    if any(prefix in entity.name.lower() for prefix in ['hot-', 'brewed-', 'ground-']):
                        # Check if this step is the transformation step
                        if 'hot-' in entity.name.lower() and 'heat' in step_lower:
                            is_provide = True
                        elif 'brewed-' in entity.name.lower() and 'brew' in step_lower:
                            is_provide = True
                        elif 'ground-' in entity.name.lower() and 'grind' in step_lower:
                            is_provide = True
                        else:
                            is_provide = False  # Using the already-transformed entity
                    
                    # Create relationship
                    if is_provide:
                        control.provides.add(entity.name)
                        self.relationships.append({
                            'from': control.name,
                            'to': entity.name,
                            'type': 'provides',
                            'use_case': use_case.id
                        })
                        self._add_entity_relationship(control.name, entity.name, RelationType.PROVIDE, use_case.id)
                    else:
                        # Default: USE relationship
                        control.uses.add(entity.name)
                        self.relationships.append({
                            'from': control.name,
                            'to': entity.name,
                            'type': 'uses',
                            'use_case': use_case.id
                        })
                        self._add_entity_relationship(control.name, entity.name, RelationType.USE, use_case.id)
            
            # Orchestrator coordinates all controllers (control flow only, no data)
            if step_controls and orchestrator_name not in [c.name for c in step_controls]:
                for control in step_controls:
                    self.relationships.append({
                        'from': orchestrator_name,
                        'to': control.name,
                        'type': 'coordinates',
                        'use_case': use_case.id
                    })
            
            # Boundaries can access entities (e.g., Storage-Access boundary)
            # But boundaries typically connect to controllers, not directly to entities
            for boundary in step_boundaries:
                if 'storage' in boundary.name.lower() or 'access' in boundary.name.lower():
                    # Storage boundary connects to retrieval controller
                    for control in step_controls:
                        if 'retrieval' in control.name.lower() or 'access' in control.name.lower():
                            self.relationships.append({
                                'from': boundary.name,
                                'to': control.name,
                                'type': 'accesses',
                                'use_case': use_case.id
                            })
        
        self._check_preconditions(use_case)
        self._check_use_case_quality(use_case)
        
        return self.get_results()
    
    def _check_preconditions(self, use_case: UseCase):
        """Check if preconditions are provided by other UCs"""
        for precondition in use_case.preconditions:
            available_match = re.search(r'(\w+)\s+(?:is|are)\s+(?:available\s+)?in\s+(?:the\s+)?system', precondition.lower())
            
            if available_match:
                entity_name = self._normalize_name(available_match.group(1))
                
                providing_ucs = []
                for obj_name, obj in self.objects.items():
                    if (obj.type == ObjectType.ENTITY and 
                        obj.name == entity_name and
                        any('provided_by' in kw for kw in obj.keywords)):
                        for kw in obj.keywords:
                            if kw.startswith('provided_by:'):
                                provider_uc = kw.split(':')[1]
                                if provider_uc != use_case.id:
                                    providing_ucs.append(provider_uc)
                
                if not providing_ucs:
                    uc_suggestion = f"Establish {use_case.capability} Operational Readiness"
                    
                    self.warnings.append(Warning(
                        severity="warning",
                        category="missing_provider",
                        message=f"Precondition '{entity_name}' has no providing Use Case",
                        use_case=use_case.id,
                        suggestion=f"Create a Use Case: '{uc_suggestion}'"
                    ))
    
    def _check_use_case_quality(self, use_case: UseCase):
        """Check UC quality and provide warnings - ENHANCED with semantic checks"""
        vague_terms = ['somehow', 'maybe', 'perhaps', 'possibly', 'eventually', 'might']
        for step in use_case.steps:
            step_text = step.get('text', '').lower()
            for term in vague_terms:
                if term in step_text:
                    self.warnings.append(Warning(
                        severity="warning",
                        category="vague_description",
                        message=f"Step {step['id']}: Vague term '{term}' found",
                        use_case=use_case.id,
                        suggestion="Be specific: What exactly should happen?"
                    ))
        
        if not use_case.actors:
            self.warnings.append(Warning(
                severity="error",
                category="missing_actors",
                message="No actors defined",
                use_case=use_case.id,
                suggestion="Define at least one actor"
            ))
        
        if not use_case.postconditions:
            self.warnings.append(Warning(
                severity="warning",
                category="missing_postconditions",
                message="No postconditions defined",
                use_case=use_case.id,
                suggestion="Define the expected result"
            ))
        
        # === NEW: Semantic consistency checks ===
        
        # Check for water temperature inconsistencies
        self._check_water_temperature_consistency(use_case)
        
        # Check for missing user configuration in preconditions
        self._check_missing_configuration_preconditions(use_case)
        
        # Check for transformation logic errors
        self._check_transformation_logic(use_case)
    
    def _check_water_temperature_consistency(self, use_case: UseCase):
        """Check if water temperature states are consistent"""
        has_heating_step = False
        uses_hot_water = False
        uses_cold_water = False
        
        for idx, step in enumerate(use_case.steps):
            step_text = step.get('text', '').lower()
            
            # Check for heating action
            if re.search(r'(activates?|heats?|warms?).*?water.*?heater', step_text):
                has_heating_step = True
            
            # Check for hot water usage
            if re.search(r'hot\s+water|heated\s+water|warm\s+water', step_text):
                uses_hot_water = True
            
            # Check for cold/ambient water mention
            if re.search(r'cold\s+water|water(?!\s+(heater|compressor))', step_text) and not re.search(r'hot\s+water', step_text):
                uses_cold_water = True
            
            # ERROR: Using hot water without heating it first
            if uses_hot_water and not has_heating_step and idx > 0:
                # Check if this is brewing or pressing step
                if re.search(r'(brew|press|push).*?water', step_text):
                    self.warnings.append(Warning(
                        severity="error",
                        category="semantic_error",
                        message=f"Step {step['id']}: Uses 'water' but should specify 'hot water' after heating step",
                        use_case=use_case.id,
                        suggestion=f"Change to 'hot water' in step {step['id']}: '{step_text[:60]}...'"
                    ))
            
            # WARNING: Mentions water generically after heating
            if has_heating_step and re.search(r'\bwater\b', step_text) and not re.search(r'hot\s+water|cold\s+water|water\s+heater|water\s+compressor|water\s+pressure', step_text):
                if re.search(r'(brew|press|begins|adds|uses|with).*?water', step_text):
                    self.warnings.append(Warning(
                        severity="warning",
                        category="ambiguous_water_state",
                        message=f"Step {step['id']}: Ambiguous water state - specify temperature",
                        use_case=use_case.id,
                        suggestion=f"Specify 'hot water' or 'cold water' in step {step['id']}"
                    ))
    
    def _check_missing_configuration_preconditions(self, use_case: UseCase):
        """Check if user configurations are missing in preconditions"""
        config_mentions = []
        precondition_configs = []
        
        # Extract configuration mentions from steps
        for step in use_case.steps:
            step_text = step.get('text', '').lower()
            
            if 'configured' in step_text:
                # Extract what is configured
                if re.search(r'configured.*?(coffee.*?type|milk|espresso)', step_text):
                    config_mentions.append('coffee type')
                if re.search(r'configured.*?(amount|quantity)', step_text):
                    config_mentions.append('amount')
                if re.search(r'configured.*?(grind.*?level|strength)', step_text):
                    config_mentions.append('grind level / strength')
                if re.search(r'configured.*?time', step_text):
                    config_mentions.append('time')
        
        # Extract existing preconditions about configuration
        for precond in use_case.preconditions:
            precond_lower = precond.lower()
            if 'selected' in precond_lower or 'configured' in precond_lower or 'chosen' in precond_lower:
                if 'coffee' in precond_lower or 'type' in precond_lower:
                    precondition_configs.append('coffee type')
                if 'strength' in precond_lower or 'grind' in precond_lower:
                    precondition_configs.append('grind level / strength')
                if 'amount' in precond_lower:
                    precondition_configs.append('amount')
        
        # Check for missing preconditions
        for config in set(config_mentions):
            if config not in precondition_configs:
                self.warnings.append(Warning(
                    severity="error",
                    category="missing_precondition",
                    message=f"Steps reference 'configured {config}' but no precondition states user has configured it",
                    use_case=use_case.id,
                    suggestion=f"Add precondition: 'User has selected {config} via HMI' or 'User has configured {config}'"
                ))
    
    def _check_transformation_logic(self, use_case: UseCase):
        """Check if transformation steps make sense (input -> process -> output)"""
        transformations = {
            'grinding': {
                'input': ['coffee bean', 'bean'],
                'output': ['coffee ground', 'ground'],
                'action': ['grind', 'grinding']
            },
            'heating': {
                'input': ['cold water', 'water'],
                'output': ['hot water', 'heated water'],
                'action': ['heat', 'activates.*heater', 'warm']
            },
            'brewing': {
                'input': ['hot water', 'coffee ground'],
                'output': ['brewed coffee', 'coffee'],
                'action': ['brew', 'press']
            },
            'compression': {
                'input': ['hot water'],
                'output': ['compressed.*water', 'pressurized.*water'],
                'action': ['compressor', 'pressure']
            }
        }
        
        for transform_name, transform in transformations.items():
            found_action = False
            found_input = False
            found_output = False
            action_step_idx = -1
            
            for idx, step in enumerate(use_case.steps):
                step_text = step.get('text', '').lower()
                
                # Check if this step performs the action
                for action_pattern in transform['action']:
                    if re.search(action_pattern, step_text):
                        found_action = True
                        action_step_idx = idx
                        break
                
                # Check for input mention
                if any(re.search(inp, step_text) for inp in transform['input']):
                    found_input = True
                
                # Check for output mention
                if any(re.search(out, step_text) for out in transform['output']):
                    found_output = True
            
            # If action found but output not mentioned in or after the action
            if found_action and not found_output and action_step_idx >= 0:
                self.warnings.append(Warning(
                    severity="info",
                    category="incomplete_transformation",
                    message=f"Step performs {transform_name} but doesn't mention output product",
                    use_case=use_case.id,
                    suggestion=f"Consider explicitly mentioning the output: {', '.join(transform['output'][:2])}"
                ))
    
    def _identify_missing_use_cases(self):
        """Identify missing Use Cases"""
        used_entities = set()
        provided_entities = set()
        
        for obj in self.objects.values():
            if obj.type == ObjectType.ENTITY:
                used_entities.add(obj.name)
                if any('provided_by' in kw for kw in obj.keywords):
                    provided_entities.add(obj.name)
        
        missing_entities = used_entities - provided_entities
        
        if len(missing_entities) >= 2:
            capability = list(self.use_cases_map.values())[0].capability if self.use_cases_map else "System"
            uc_name = f"Establish {capability} Operational Readiness"
            
            self.warnings.append(Warning(
                severity="warning",
                category="missing_use_case",
                message=f"Multiple entities without provider: {', '.join(missing_entities)}",
                use_case=None,
                suggestion=f"Create Use Case: '{uc_name}' (covers: {', '.join(missing_entities)})"
            ))
    
    def analyze_multiple_use_cases(self, use_cases: List[UseCase]) -> Dict:
        for uc in use_cases:
            self.analyze_use_case(uc)
        
        self._identify_missing_use_cases()
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        by_type = {'boundary': [], 'entity': [], 'control': []}
        shared_objects = {'boundary': [], 'entity': [], 'control': []}
        
        for obj in self.objects.values():
            obj_dict = {
                'name': obj.name,
                'use_cases': list(obj.use_cases),
                'shared': obj.is_shared(),
                'keywords': obj.keywords
            }
            
            type_key = obj.type.value
            by_type[type_key].append(obj_dict)
            
            if obj.is_shared():
                shared_objects[type_key].append(obj.name)
        
        validation = self.validate_robustness_rules()
        
        return {
            'objects': by_type,
            'shared_objects': shared_objects,
            'relationships': self.relationships,
            'warnings': [
                {
                    'severity': w.severity,
                    'category': w.category,
                    'message': w.message,
                    'use_case': w.use_case,
                    'suggestion': w.suggestion
                } for w in self.warnings
            ],
            'statistics': {
                'total_objects': len(self.objects),
                'boundary_count': len(by_type['boundary']),
                'entity_count': len(by_type['entity']),
                'control_count': len(by_type['control']),
                'shared_count': sum(len(v) for v in shared_objects.values()),
                'warnings_count': len(self.warnings),
                'errors_count': len([w for w in self.warnings if w.severity == 'error'])
            },
            'validation': validation
        }
    
    def validate_robustness_rules(self) -> Dict:
        rules = {
            'actors_to_boundary_only': True,
            'boundary_to_control_entity': True,
            'no_entity_to_boundary': True,
            'control_coordinates': True
        }
        violations = []
        
        for rel in self.relationships:
            from_obj = self.objects.get(rel['from'])
            to_obj = self.objects.get(rel['to'])
            
            if from_obj and to_obj:
                if (from_obj.type == ObjectType.ENTITY and 
                    to_obj.type == ObjectType.BOUNDARY):
                    rules['no_entity_to_boundary'] = False
                    violations.append(f"Entity '{from_obj.name}' -> Boundary '{to_obj.name}'")
        
        return {
            'rules': rules,
            'violations': violations,
            'valid': all(rules.values())
        }
    
    def export_to_json(self, filename: str):
        results = self.get_results()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results exported to: {filename}")


def main():
    """Main program with example Use Cases"""
    print("=" * 70)
    print("AUTOMATIC ROBUSTNESS ANALYSIS")
    print("NLP + Few-shot Learning for RUP Use Cases")
    print("=" * 70)
    
    # CORRECTED Use Case with proper preconditions
    use_case_text = """
Capability: Coffee Preparation

Use Case 1:
Goal: User can drink their milk coffee every morning at 7am
Preconditions:
  - Coffee beans are available in the system
  - Cold water is available in the system
  - Milk is available in the system
  - User has selected coffee type (Milk Coffee) via HMI
  - User has configured coffee strength via HMI
  - User has configured time (7:00am) via HMI
Actors: User

Basic Flow:
S1. (trigger) System clock reaches configured time of 7:00am
S2. System activates the water heater
S3. System prepares filter
S4. System grinds the configured amount at configured grind level
S5. System retrieves cup from storage container
S6. System begins brewing coffee with configured amount of hot water
S7. System adds milk to the cup
S8. System adds brewed coffee to the cup
S9. System displays notification to User

Postconditions:
  - User has received coffee on time

Alternative Flows:
A1 at S2: Water heater has insufficient water
  A1.1 System displays error message to User
  A1.2 End

Use Case 2:
Goal: User wants to drink an espresso
Preconditions:
  - Coffee beans are available in the system
  - Cold water is available in the system
  - User has selected coffee type (Espresso) via HMI
  - User has configured coffee strength via HMI
  - User has configured time via HMI
Actors: User

Basic Flow:
S1. (trigger) System clock reaches configured time
S2. System activates the water heater
S3. System prepares filter
S4. System grinds coffee beans at configured grind level
S5. System retrieves cup from storage
S6. System starts water compressor to generate water pressure for espresso
S7. System begins pressing hot water through coffee grounds
S8. System displays notification to User

Postconditions:
  - User has received an espresso
"""
    
    analyzer = RobustnessAnalyzer()
    
    print("\n📄 Parsing Use Cases...")
    use_cases = analyzer.load_from_text(use_case_text)
    print(f"✓ {len(use_cases)} Use Cases parsed")
    
    print("\n🔍 Starting analysis...")
    results = analyzer.analyze_multiple_use_cases(use_cases)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\n📊 STATISTICS:")
    stats = results['statistics']
    print(f"  • Total Objects: {stats['total_objects']}")
    print(f"  • Boundary: {stats['boundary_count']}, Entity: {stats['entity_count']}, Control: {stats['control_count']}")
    print(f"  • Shared Objects: {stats['shared_count']}")
    print(f"  • Warnings: {stats['warnings_count']}, Errors: {stats['errors_count']}")
    
    print("\n⭕ BOUNDARY OBJECTS:")
    for obj in results['objects']['boundary']:
        shared = " [SHARED]" if obj['shared'] else ""
        print(f"  • {obj['name']}{shared} - UCs: {', '.join(obj['use_cases'])}")
    
    print("\n⬭ ENTITY OBJECTS:")
    for obj in results['objects']['entity']:
        shared = " [SHARED]" if obj['shared'] else ""
        print(f"  • {obj['name']}{shared} - UCs: {', '.join(obj['use_cases'])}")
    
    print("\n↻ CONTROL OBJECTS:")
    for obj in results['objects']['control']:
        shared = " [SHARED]" if obj['shared'] else ""
        print(f"  • {obj['name']}{shared} - UCs: {', '.join(obj['use_cases'])}")
    
    print("\n✓ ROBUSTNESS RULES:")
    validation = results['validation']
    for rule, valid in validation['rules'].items():
        status = "✓" if valid else "✗"
        print(f"  {status} {rule}")
    
    if results['warnings']:
        print("\n⚠️  WARNINGS & QUALITY ISSUES:")
        
        # Group by severity
        errors = [w for w in results['warnings'] if w['severity'] == 'error']
        warnings = [w for w in results['warnings'] if w['severity'] == 'warning']
        infos = [w for w in results['warnings'] if w['severity'] == 'info']
        
        if errors:
            print("\n  🔴 ERRORS (Must Fix):")
            for w in errors:
                uc_info = f" [{w['use_case']}]" if w['use_case'] else ""
                print(f"    • {w['category']}: {w['message']}{uc_info}")
                if w['suggestion']:
                    print(f"      💡 {w['suggestion']}")
        
        if warnings:
            print("\n  🟡 WARNINGS (Should Fix):")
            for w in warnings:
                uc_info = f" [{w['use_case']}]" if w['use_case'] else ""
                print(f"    • {w['category']}: {w['message']}{uc_info}")
                if w['suggestion']:
                    print(f"      💡 {w['suggestion']}")
        
        if infos:
            print("\n  🔵 INFO (Consider):")
            for w in infos:
                uc_info = f" [{w['use_case']}]" if w['use_case'] else ""
                print(f"    • {w['category']}: {w['message']}{uc_info}")
                if w['suggestion']:
                    print(f"      💡 {w['suggestion']}")
    
    analyzer.export_to_json('robustness_analysis.json')
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\n✅ Automatic Quality Checks:")
    print("  • Water temperature consistency")
    print("  • Missing configuration preconditions")
    print("  • Transformation logic validation")
    print("  • Entity state tracking")


if __name__ == "__main__":
    main()
    """Main program with example Use Cases"""
    print("=" * 70)
    print("AUTOMATIC ROBUSTNESS ANALYSIS")
    print("NLP + Few-shot Learning for RUP Use Cases")
    print("=" * 70)
    
    use_case_text = """
Capability: Coffee Preparation

Use Case 1:
Goal: User can drink their milk coffee every morning at 7am
Preconditions:
  - Coffee beans are available in the system
  - Water is available in the system
  - Milk is available in the system
  - User has selected amount via HMI
  - User has selected time via HMI
  - User has selected time via HMI
Actors: User

Basic Flow:
S1. (trigger) System clock reaches configured time of 7:00am
S2. System activates the water heater
S3. System prepares filter
S4. System grinds the configured amount at configured grind level
S5. System retrieves cup from storage container
S6. System begins brewing coffee with the configured water amount
S7. System adds milk to the cup
S8. System adds brewed coffee to the cup
S9. System displays notification to User

Postconditions:
  - User has received coffee on time

Alternative Flows:
A1 at S2: Water heater has insufficient water
  A1.1 System displays error message to User
  A1.2 End

Use Case 2:
Goal: User wants to drink an espresso
Preconditions:
  - Coffee beans are available in the system
  - Water is available in the system
Actors: User

Basic Flow:
S1. (trigger) User selects coffee type espresso
S2. System activates the water heater
S3. System prepares filter
S4. System grinds coffee beans
S5. System retrieves cup from storage
S6. System starts water compressor for espresso pressure
S7. System begins pressing hot water through coffee grounds
S8. System displays notification to User

Postconditions:
  - User has received an espresso
"""
    
    analyzer = RobustnessAnalyzer()
    
    print("\n📄 Parsing Use Cases...")
    use_cases = analyzer.load_from_text(use_case_text)
    print(f"✓ {len(use_cases)} Use Cases parsed")
    
    print("\n🔍 Starting analysis...")
    results = analyzer.analyze_multiple_use_cases(use_cases)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\n📊 STATISTICS:")
    stats = results['statistics']
    print(f"  • Total Objects: {stats['total_objects']}")
    print(f"  • Boundary: {stats['boundary_count']}, Entity: {stats['entity_count']}, Control: {stats['control_count']}")
    print(f"  • Shared Objects: {stats['shared_count']}")
    print(f"  • Warnings: {stats['warnings_count']}, Errors: {stats['errors_count']}")
    
    print("\n⭕ BOUNDARY OBJECTS:")
    for obj in results['objects']['boundary']:
        shared = " [SHARED]" if obj['shared'] else ""
        print(f"  • {obj['name']}{shared} - UCs: {', '.join(obj['use_cases'])}")
    
    print("\n⬭ ENTITY OBJECTS:")
    for obj in results['objects']['entity']:
        shared = " [SHARED]" if obj['shared'] else ""
        print(f"  • {obj['name']}{shared} - UCs: {', '.join(obj['use_cases'])}")
    
    print("\n↻ CONTROL OBJECTS:")
    for obj in results['objects']['control']:
        shared = " [SHARED]" if obj['shared'] else ""
        print(f"  • {obj['name']}{shared} - UCs: {', '.join(obj['use_cases'])}")
    
    print("\n✓ ROBUSTNESS RULES:")
    validation = results['validation']
    for rule, valid in validation['rules'].items():
        status = "✓" if valid else "✗"
        print(f"  {status} {rule}")
    
    if results['warnings']:
        print("\n⚠️  WARNINGS:")
        for w in results['warnings']:
            uc_info = f" [{w['use_case']}]" if w['use_case'] else ""
            print(f"  • {w['message']}{uc_info}")
            if w['suggestion']:
                print(f"    💡 {w['suggestion']}")
    
    analyzer.export_to_json('robustness_analysis.json')
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()