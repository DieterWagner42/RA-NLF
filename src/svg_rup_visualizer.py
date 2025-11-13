#!/usr/bin/env python3
"""
SVG-basierte RUP Visualisierung mit offiziellen Wikipedia-Symbolen
Lädt originale SVG-Symbole direkt von Wikipedia und verwendet sie mit transparentem Hintergrund
"""

import json
import xml.etree.ElementTree as ET
import requests
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import re
import base64

class ComponentType(Enum):
    ACTOR = 'actor'
    BOUNDARY = 'boundary'
    CONTROLLER = 'controller'
    ENTITY = 'entity'
    CONTROL_FLOW_NODE = 'control_flow_node'

@dataclass
class RAComponent:
    """RA Component für SVG-Darstellung"""
    name: str
    component_type: ComponentType
    description: str
    position: Tuple[float, float] = (0, 0)
    parallel_group: int = 0

@dataclass
class ControlFlow:
    """Control Flow für SVG-Darstellung"""
    source: str
    destination: str
    source_step: str
    target_step: str
    rule: str
    description: str

@dataclass
class DataFlow:
    """Data Flow für SVG-Darstellung"""
    controller: str
    entity: str
    flow_type: str
    description: str

class SVGRUPVisualizer:
    """
    SVG-basierte RUP Visualisierung mit offiziellen Wikipedia-Symbolen
    
    Symbole:
    - Actor: https://de.wikipedia.org/wiki/Robustheitsanalyse#/media/Datei:Robustness_Diagram_Actor.svg
    - Boundary: https://upload.wikimedia.org/wikipedia/commons/d/df/Robustness_Diagram_Boundary.svg  
    - Controller: https://de.wikipedia.org/wiki/Robustheitsanalyse#/media/Datei:Robustness_Diagram_Control.svg
    - Entity: https://de.wikipedia.org/wiki/Robustheitsanalyse#/media/Datei:Robustness_Diagram_Entity.svg
    """
    
    def __init__(self, canvas_width=1400, canvas_height=2000):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # Lokale SVG-Dateien für originale RUP-Symbole
        self.local_symbol_files = {
            ComponentType.ACTOR: "svg/Robustness_Diagram_Actor.svg",
            ComponentType.BOUNDARY: "svg/Robustness_Diagram_Boundary.svg", 
            ComponentType.CONTROLLER: "svg/Robustness_Diagram_Control.svg",
            ComponentType.ENTITY: "svg/Robustness_Diagram_Entity.svg"
        }
        
        # Cache für geladene SVG-Symbole
        self.symbol_cache = {}
        
        # Symbol-Größen basierend auf tatsächlichen SVG-Dimensionen
        # Alle Wikipedia RUP-Symbole haben 166x116, skaliert mit 0.8 = ~133x93
        base_width, base_height = 166, 116
        scale_factor = 0.8  # Aus create_svg_component
        scaled_width = int(base_width * scale_factor)
        scaled_height = int(base_height * scale_factor)
        
        self.symbol_sizes = {
            ComponentType.ACTOR: {'width': scaled_width, 'height': scaled_height},
            ComponentType.BOUNDARY: {'width': scaled_width, 'height': scaled_height},
            ComponentType.CONTROLLER: {'width': scaled_width, 'height': scaled_height},
            ComponentType.ENTITY: {'width': scaled_width, 'height': scaled_height},
            ComponentType.CONTROL_FLOW_NODE: {'width': 30, 'height': 30}  # Diamond symbol
        }
    
    def _load_local_symbol(self, component_type: ComponentType) -> str:
        """Lade lokale SVG-Datei und bereite für Embedding vor"""
        if component_type in self.symbol_cache:
            return self.symbol_cache[component_type]
        
        if component_type not in self.local_symbol_files:
            # Fallback für Control Flow Nodes
            return self._create_diamond_symbol()
        
        try:
            file_path = self.local_symbol_files[component_type]
            print(f"[SVG LOAD] Loading {component_type.value} symbol from: {file_path}")
            
            # Prüfe ob Datei existiert
            if not Path(file_path).exists():
                print(f"[SVG LOAD] File not found: {file_path}")
                return self._create_fallback_symbol(component_type)
            
            # Lade SVG-Inhalt
            with open(file_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            
            # SVG-Inhalt bereinigen und für Embedding vorbereiten
            svg_content = self._clean_svg_content(svg_content)
            
            # Cache speichern
            self.symbol_cache[component_type] = svg_content
            
            print(f"[SVG LOAD] Successfully loaded {component_type.value} symbol")
            return svg_content
            
        except Exception as e:
            print(f"[SVG LOAD] Error loading {component_type.value} symbol: {e}")
            # Fallback zu manuell erstellten Symbolen
            return self._create_fallback_symbol(component_type)
    
    def _clean_svg_content(self, svg_content: str) -> str:
        """Bereinige SVG-Inhalt für transparenten Hintergrund und Embedding"""
        
        # Entferne XML-Deklaration
        svg_content = re.sub(r'<\?xml[^>]*\?>', '', svg_content)
        
        # Entferne DOCTYPE
        svg_content = re.sub(r'<!DOCTYPE[^>]*>', '', svg_content)
        
        # Setze transparenten Hintergrund (entferne fill="white" oder ähnliche Hintergründe)
        svg_content = re.sub(r'fill\s*=\s*["\'](?:white|#ffffff|#fff)["\']', 'fill="none"', svg_content, flags=re.IGNORECASE)
        
        # Entferne explizite Hintergrund-Rechtecke
        svg_content = re.sub(r'<rect[^>]*fill\s*=\s*["\'](?:white|#ffffff|#fff)["\'][^>]*>', '', svg_content, flags=re.IGNORECASE)
        
        # Extrahiere nur den Inhalt zwischen <svg> Tags (ohne die svg Tags selbst)
        match = re.search(r'<svg[^>]*>(.*)</svg>', svg_content, re.DOTALL)
        if match:
            inner_content = match.group(1).strip()
            return f'<g class="wikipedia-symbol">{inner_content}</g>'
        
        return svg_content
    
    def _create_fallback_symbol(self, component_type: ComponentType) -> str:
        """Erstelle Fallback-Symbol falls Wikipedia-Download fehlschlägt"""
        if component_type == ComponentType.ACTOR:
            return '''
            <g class="actor-symbol">
                <circle cx="20" cy="12" r="8" fill="none" stroke="black" stroke-width="2"/>
                <line x1="20" y1="20" x2="20" y2="40" stroke="black" stroke-width="2"/>
                <line x1="8" y1="28" x2="32" y2="28" stroke="black" stroke-width="2"/>
                <line x1="20" y1="40" x2="10" y2="55" stroke="black" stroke-width="2"/>
                <line x1="20" y1="40" x2="30" y2="55" stroke="black" stroke-width="2"/>
            </g>
            '''
        elif component_type == ComponentType.BOUNDARY:
            return '''
            <g class="boundary-symbol">
                <rect x="5" y="10" width="70" height="20" rx="10" ry="10" 
                      fill="none" stroke="black" stroke-width="2"/>
            </g>
            '''
        elif component_type == ComponentType.CONTROLLER:
            return '''
            <g class="controller-symbol">
                <circle cx="25" cy="25" r="20" fill="none" stroke="black" stroke-width="2"/>
                <circle cx="25" cy="25" r="6" fill="black"/>
                <path d="M 19 25 L 31 25 M 25 19 L 25 31" stroke="black" stroke-width="2"/>
            </g>
            '''
        elif component_type == ComponentType.ENTITY:
            return '''
            <g class="entity-symbol">
                <circle cx="25" cy="25" r="18" fill="none" stroke="black" stroke-width="2"/>
                <line x1="7" y1="43" x2="43" y2="43" stroke="black" stroke-width="2"/>
            </g>
            '''
        else:
            return self._create_diamond_symbol()
    
    def _create_diamond_symbol(self) -> str:
        """Erstelle Diamant-Symbol für Control Flow Nodes"""
        return '''
        <g class="diamond-symbol">
            <path d="M 15 5 L 25 15 L 15 25 L 5 15 Z" 
                  fill="none" stroke="purple" stroke-width="2"/>
        </g>
        '''
    
    def load_json_data(self, json_file_path: str) -> Dict:
        """JSON-Daten laden"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Speichere Layout-Order für spätere Verwendung
            self._json_layout_order = data.get('layout', {}).get('uc_step_order', [])
            return data
    
    def parse_components(self, json_data: Dict) -> List[RAComponent]:
        """Komponenten aus JSON-Daten parsen"""
        components = []
        
        # Actors
        for actor in json_data.get('components', {}).get('actors', []):
            components.append(RAComponent(
                name=actor['name'],
                component_type=ComponentType.ACTOR,
                description=actor.get('description', ''),
                parallel_group=0
            ))
        
        # Controllers
        for controller in json_data.get('components', {}).get('controllers', []):
            components.append(RAComponent(
                name=controller['name'],
                component_type=ComponentType.CONTROLLER,
                description=controller.get('description', ''),
                parallel_group=controller.get('parallel_group', 0)
            ))
        
        # Boundaries
        for boundary in json_data.get('components', {}).get('boundaries', []):
            components.append(RAComponent(
                name=boundary['name'],
                component_type=ComponentType.BOUNDARY,
                description=boundary.get('description', ''),
                parallel_group=0
            ))
        
        # Entities
        for entity in json_data.get('components', {}).get('entities', []):
            components.append(RAComponent(
                name=entity['name'],
                component_type=ComponentType.ENTITY,
                description=entity.get('description', ''),
                parallel_group=0
            ))
        
        # Control Flow Nodes
        for node in json_data.get('components', {}).get('control_flow_nodes', []):
            components.append(RAComponent(
                name=node['name'],
                component_type=ComponentType.CONTROL_FLOW_NODE,
                description=node.get('description', ''),
                parallel_group=0
            ))
        
        return components
    
    def parse_control_flows(self, json_data: Dict) -> List[ControlFlow]:
        """Control Flows aus JSON-Daten parsen"""
        control_flows = []
        for cf in json_data.get('relationships', {}).get('control_flows', []):
            control_flows.append(ControlFlow(
                source=cf['source'],
                destination=cf['destination'],
                source_step=cf.get('source_step', ''),
                target_step=cf.get('target_step', ''),
                rule=cf.get('rule', ''),
                description=cf.get('description', '')
            ))
        return control_flows
    
    def parse_data_flows(self, json_data: Dict) -> List[DataFlow]:
        """Data Flows aus JSON-Daten parsen"""
        data_flows = []
        for df in json_data.get('relationships', {}).get('data_flows', []):
            data_flows.append(DataFlow(
                controller=df['controller'],
                entity=df['entity'],
                flow_type=df.get('type', df.get('flow_type', 'unknown')),
                description=df.get('description', '')
            ))
        return data_flows
    
    def calculate_layout(self, components: List[RAComponent]) -> List[RAComponent]:
        """Layout berechnen: UC-Schritte von oben nach unten, parallele nebeneinander"""
        
        # Komponenten nach Typ gruppieren
        actors = [c for c in components if c.component_type == ComponentType.ACTOR]
        boundaries = [c for c in components if c.component_type == ComponentType.BOUNDARY]
        controllers = [c for c in components if c.component_type == ComponentType.CONTROLLER]
        entities = [c for c in components if c.component_type == ComponentType.ENTITY]
        control_nodes = [c for c in components if c.component_type == ComponentType.CONTROL_FLOW_NODE]
        
        # Layout-Parameter - gleichmäßige Verteilung über die gesamte Canvas-Breite
        margin = 50
        # Berechne gleichmäßige Abstände: 4 Spalten über 1400px Canvas
        # Actors: ~6%, Boundaries: ~32%, Controllers: ~58%, Entities: ~84%
        actor_x = 84          # Etwas mehr Rand links
        boundary_x = 448      # Gleichmäßiger Abstand (364px von Actors)
        controller_x = 812    # Gleichmäßiger Abstand (364px von Boundaries)
        entity_x = 1176       # Gleichmäßiger Abstand (364px von Controllers)
        
        # Actors links positionieren mit kleinerem Abstand
        if actors:
            self._position_compact_column(actors, actor_x, margin, spacing=70)

        # Boundaries links-mitte positionieren mit größerem Abstand
        if boundaries:
            # Alle Boundaries anzeigen (kein Limit mehr)
            visible_boundaries = boundaries
            # Größerer Startpunkt und größerer Abstand für Boundaries
            boundary_start = margin + 50  # Nicht direkt am Rand starten
            self._position_compact_column(visible_boundaries, boundary_x, boundary_start, spacing=100)
        
        # Controllers und Control Flow Nodes nach UC-Schritten positionieren (datengetrieben)
        if controllers:
            self._position_controllers_with_flow_nodes(controllers, control_nodes, controller_x)
        
        # Entities rechts in einer Spalte mit größerem Abstand positionieren
        if entities:
            # Alle Entities in einer Spalte mit größerem Abstand
            self._position_compact_column(entities, entity_x, margin, spacing=90)
        
        return components
    
    def _position_column(self, components: List[RAComponent], x_position: float, y_start: float, y_end: float):
        """Komponenten in einer Spalte positionieren"""
        if not components:
            return

        if len(components) == 1:
            components[0].position = (x_position, (y_start + y_end) / 2)
        else:
            y_step = (y_end - y_start) / (len(components) - 1)
            for i, component in enumerate(components):
                y_position = y_start + i * y_step
                component.position = (x_position, y_position)

    def _position_compact_column(self, components: List[RAComponent], x_position: float, y_start: float, spacing: int = 70):
        """Komponenten in einer Spalte mit festem Abstand positionieren"""
        if not components:
            return

        # Fester Abstand zwischen Komponenten
        for i, component in enumerate(components):
            y_position = y_start + i * spacing
            component.position = (x_position, y_position)
    
    def _position_controllers_with_flow_nodes(self, controllers: List[RAComponent], control_nodes: List[RAComponent], x_center: float):
        """Controllers und Control Flow Nodes datengetrieben positionieren basierend auf Control Flows"""
        
        # Gruppiere Controllers nach parallel_group
        controller_groups = {}
        for controller in controllers:
            group = controller.parallel_group
            if group not in controller_groups:
                controller_groups[group] = []
            controller_groups[group].append(controller)
        
        # Gruppiere Control Flow Nodes nach Typ (START/END) und extrahiere Group-Nummer
        flow_nodes_by_group = {}
        if control_nodes:
            for node in control_nodes:
                # Extrahiere P-Nummer aus Namen wie "P2_START", "P3_END"
                import re
                match = re.match(r'P(\d+)_(START|END)', node.name)
                if match:
                    group_num = int(match.group(1))
                    node_type = match.group(2)
                    
                    if group_num not in flow_nodes_by_group:
                        flow_nodes_by_group[group_num] = {'START': None, 'END': None}
                    
                    flow_nodes_by_group[group_num][node_type] = node
        
        print(f"[SVG LAYOUT] Controller groups: {controller_groups}")
        print(f"[SVG LAYOUT] Flow node groups: {flow_nodes_by_group}")
        
        # Layout-Parameter
        y_start = 80
        base_y_step = 100
        x_spacing = 100
        current_y = y_start
        
        # Verwende Layout-Reihenfolge aus JSON (falls vorhanden)
        layout_sequence = []
        if hasattr(self, '_json_layout_order') and self._json_layout_order:
            layout_sequence = self._build_layout_from_json_order(controller_groups, flow_nodes_by_group)
        else:
            # Fallback: ursprüngliche Logik
            layout_sequence = self._build_fallback_layout(controller_groups, flow_nodes_by_group)
        
        # Positioniere alle Elemente in der Layout-Sequenz mit korrekter Berücksichtigung der Symbol-Größen
        for item_type, item in layout_sequence:
            if item_type == 'controller':
                # Controller symbol height berücksichtigen
                symbol_height = self.symbol_sizes[ComponentType.CONTROLLER]['height']
                item.position = (x_center, current_y)
                print(f"[SVG LAYOUT] Sequential {item.name} at ({x_center}, {current_y})")
                # Verwende Symbol-Höhe plus Mindestabstand für nächste Position
                current_y += max(base_y_step, symbol_height + 30)
                
            elif item_type == 'flow_node':
                # Control Flow Node symbol height berücksichtigen 
                symbol_height = self.symbol_sizes[ComponentType.CONTROL_FLOW_NODE]['height']
                item.position = (x_center, current_y)
                print(f"[SVG LAYOUT] Flow Node {item.name} at ({x_center}, {current_y})")
                # Verwende Symbol-Höhe plus Mindestabstand für nächste Position
                current_y += max(base_y_step, symbol_height + 30)
                
            elif item_type == 'parallel_group':
                # Parallel controllers - horizontal nebeneinander
                controller_symbol_height = self.symbol_sizes[ComponentType.CONTROLLER]['height']
                
                if len(item) == 1:
                    controller = item[0]
                    controller.position = (x_center, current_y)
                    print(f"[SVG LAYOUT] Single parallel {controller.name} at ({x_center}, {current_y})")
                else:
                    # Bessere symmetrische Verteilung um die Mittellinie
                    if len(item) % 2 == 0:
                        # Gerade Anzahl: Paare um die Mitte
                        half_pairs = len(item) // 2
                        for i, controller in enumerate(item):
                            offset_from_center = (i - half_pairs + 0.5) * x_spacing
                            x_pos = x_center + offset_from_center
                            controller.position = (x_pos, current_y)
                            print(f"[SVG LAYOUT] Parallel {controller.name} at ({x_pos}, {current_y})")
                    else:
                        # Ungerade Anzahl: eines in der Mitte, Rest symmetrisch drum herum
                        middle_index = len(item) // 2
                        for i, controller in enumerate(item):
                            offset_from_center = (i - middle_index) * x_spacing
                            x_pos = x_center + offset_from_center
                            controller.position = (x_pos, current_y)
                            print(f"[SVG LAYOUT] Parallel {controller.name} at ({x_pos}, {current_y})")
                
                # Verwende Controller Symbol-Höhe plus Mindestabstand für nächste Position
                current_y += max(base_y_step, controller_symbol_height + 30)
    
    def _build_layout_from_json_order(self, controller_groups: Dict, flow_nodes_by_group: Dict) -> List[Tuple]:
        """Baue Layout-Sequenz aus JSON Layout-Order"""
        layout_sequence = []
        
        # Erstelle Lookup-Maps
        controller_map = {}
        for group_controllers in controller_groups.values():
            for controller in group_controllers:
                controller_map[controller.name] = controller
        
        # Verarbeite JSON Layout-Order
        for order_item in self._json_layout_order:
            if order_item['type'] == 'controller':
                controller_name = order_item['name']
                if controller_name in controller_map:
                    controller = controller_map[controller_name]
                    parallel_group = order_item.get('parallel_group', 0)
                    
                    if parallel_group == 0:
                        # Sequential controller
                        layout_sequence.append(('controller', controller))
                    else:
                        # Sammle alle Controller der parallel group
                        group_controllers = controller_groups.get(parallel_group, [])
                        if group_controllers and group_controllers not in [item[1] for item in layout_sequence if item[0] == 'parallel_group']:
                            layout_sequence.append(('parallel_group', group_controllers))
            
            elif order_item['type'] == 'control_flow_node':
                node_name = order_item['name']
                # Finde entsprechenden Flow Node
                for group_nodes in flow_nodes_by_group.values():
                    if 'START' in group_nodes and group_nodes['START'].name == node_name:
                        layout_sequence.append(('flow_node', group_nodes['START']))
                        break
                    elif 'END' in group_nodes and group_nodes['END'].name == node_name:
                        layout_sequence.append(('flow_node', group_nodes['END']))
                        break
        
        return layout_sequence
    
    def _build_fallback_layout(self, controller_groups: Dict, flow_nodes_by_group: Dict) -> List[Tuple]:
        """Fallback Layout wenn keine JSON Layout-Order vorhanden ist"""
        layout_sequence = []
        
        # Original Logik: Gruppiere nach parallel_group
        sorted_groups = sorted(controller_groups.keys())
        
        for group_num in sorted_groups:
            group_controllers = controller_groups[group_num]
            
            # Für parallele Gruppen: START node vor Controllers
            if group_num > 0 and group_num in flow_nodes_by_group:
                start_node = flow_nodes_by_group[group_num].get('START')
                if start_node:
                    layout_sequence.append(('flow_node', start_node))
            
            # Controllers
            if group_num == 0:
                # Sequential controllers einzeln
                for controller in group_controllers:
                    layout_sequence.append(('controller', controller))
            else:
                # Parallel group
                layout_sequence.append(('parallel_group', group_controllers))
            
            # Für parallele Gruppen: END node nach Controllers
            if group_num > 0 and group_num in flow_nodes_by_group:
                end_node = flow_nodes_by_group[group_num].get('END')
                if end_node:
                    layout_sequence.append(('flow_node', end_node))
        
        return layout_sequence
    
    def _extract_uc_step_from_description(self, description: str) -> int:
        """Extrahiere UC-Schritt-Nummer aus Controller-Description datengetrieben"""
        import re
        
        # Suche nach Patterns wie "in B1", "in B4", "in A1", "in E1"
        match = re.search(r'in ([BAE])(\d+)', description)
        if match:
            step_letter = match.group(1)
            step_number = int(match.group(2))
            
            # Mappe zu numerischen Werten für Sortierung
            if step_letter == 'B':
                return step_number
            elif step_letter == 'A':
                return 100 + step_number  # A flows nach B flows
            elif step_letter == 'E':
                return 200 + step_number  # E flows nach A flows
        
        # Default für unbekannte Patterns
        return 999
    
    def create_svg_component(self, component: RAComponent) -> str:
        """Erstelle SVG-Element für eine Komponente mit lokalem Symbol"""
        x, y = component.position
        
        # Lade lokales Symbol
        svg_symbol = self._load_local_symbol(component.component_type)
        
        # Symbol-Größe
        symbol_size = self.symbol_sizes[component.component_type]
        
        # Symbol-Gruppe mit Transform und Skalierung
        symbol_group = f'''
        <g transform="translate({x - symbol_size['width']/2}, {y - symbol_size['height']/2}) scale(0.8)">
            {svg_symbol}
        </g>
        '''
        
        # Label näher am Symbol (unter dem Symbol)
        label_y = y + symbol_size['height']/2 + 8
        label = f'''
        <text x="{x}" y="{label_y}" text-anchor="middle"
              font-family="Arial, sans-serif" font-size="12" fill="black">
            {component.name}
        </text>
        '''
        
        return symbol_group + label
    
    def create_svg_control_flow(self, cf: ControlFlow, components: List[RAComponent]) -> str:
        """Erstelle SVG-Element für Control Flow"""
        # Finde Positionen und Typen der verbundenen Komponenten
        comp_map = {comp.name: comp for comp in components}
        
        if cf.source not in comp_map or cf.destination not in comp_map:
            return ""
        
        source_comp = comp_map[cf.source]
        dest_comp = comp_map[cf.destination]
        
        # Berechne Verbindungspunkte an den Symbolrändern, nicht im Zentrum
        source_center_x, source_center_y = source_comp.position
        dest_center_x, dest_center_y = dest_comp.position
        
        # Symbol-Größen für Rand-Berechnung
        source_size = self.symbol_sizes[source_comp.component_type]
        dest_size = self.symbol_sizes[dest_comp.component_type]
        
        # Berechne Richtungsvektor
        dx = dest_center_x - source_center_x
        dy = dest_center_y - source_center_y
        
        # Distanz zwischen Zentren
        distance = (dx**2 + dy**2)**0.5
        
        if distance == 0:
            # Gleiche Position, verwende Zentrum als Fallback
            x1, y1 = source_center_x, source_center_y
            x2, y2 = dest_center_x, dest_center_y
        else:
            # Normiere den Richtungsvektor
            unit_dx = dx / distance
            unit_dy = dy / distance
            
            # Berechne Radius für kreisförmige Symbole (Controller/Entity verwenden Kreise)
            source_radius = min(source_size['width'], source_size['height']) / 2
            dest_radius = min(dest_size['width'], dest_size['height']) / 2
            
            # Start- und Endpunkte an den Symbolrändern
            x1 = source_center_x + unit_dx * source_radius
            y1 = source_center_y + unit_dy * source_radius
            x2 = dest_center_x - unit_dx * dest_radius  
            y2 = dest_center_y - unit_dy * dest_radius
        
        # Pfeil mit Marker
        return f'''
        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" 
              stroke="blue" stroke-width="2" marker-end="url(#arrowhead)"/>
        '''
    
    def create_svg_data_flow(self, df: DataFlow, components: List[RAComponent]) -> str:
        """Erstelle SVG-Element für Data Flow"""
        # Finde Komponenten nach Namen
        comp_map = {comp.name: comp for comp in components}
        
        if df.controller not in comp_map or df.entity not in comp_map:
            return ""
        
        controller_comp = comp_map[df.controller]
        entity_comp = comp_map[df.entity]
        
        # Bestimme Richtung und Farbe basierend auf Data Flow Type
        controller_center_x, controller_center_y = controller_comp.position
        entity_center_x, entity_center_y = entity_comp.position
        
        # Symbol-Größen für Rand-Berechnung
        controller_size = self.symbol_sizes[controller_comp.component_type]
        entity_size = self.symbol_sizes[entity_comp.component_type]
        
        # USE: Entity -> Controller (Input)
        # PROVIDE: Controller -> Entity (Output)
        if df.flow_type.lower() == 'use':
            # USE: Entity ist Quelle, Controller ist Ziel (orange)
            source_x, source_y = entity_center_x, entity_center_y
            dest_x, dest_y = controller_center_x, controller_center_y
            source_radius = min(entity_size['width'], entity_size['height']) / 2
            dest_radius = min(controller_size['width'], controller_size['height']) / 2
            color = "orange"
        else:  # provide
            # PROVIDE: Controller ist Quelle, Entity ist Ziel (grün)
            source_x, source_y = controller_center_x, controller_center_y
            dest_x, dest_y = entity_center_x, entity_center_y
            source_radius = min(controller_size['width'], controller_size['height']) / 2
            dest_radius = min(entity_size['width'], entity_size['height']) / 2
            color = "green"
        
        # Berechne Richtungsvektor
        dx = dest_x - source_x
        dy = dest_y - source_y
        
        # Distanz zwischen Zentren
        distance = (dx**2 + dy**2)**0.5
        
        if distance == 0:
            # Gleiche Position, verwende Zentrum als Fallback
            x1, y1 = source_x, source_y
            x2, y2 = dest_x, dest_y
        else:
            # Normiere den Richtungsvektor
            unit_dx = dx / distance
            unit_dy = dy / distance
            
            # Start- und Endpunkte an den Symbolrändern
            x1 = source_x + unit_dx * source_radius
            y1 = source_y + unit_dy * source_radius
            x2 = dest_x - unit_dx * dest_radius
            y2 = dest_y - unit_dy * dest_radius
        
        # Gestrichelte Linie mit Pfeil-Marker für Richtung
        return f'''
        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" 
              stroke="{color}" stroke-width="1" stroke-dasharray="5,5" marker-end="url(#{color}arrowhead)"/>
        '''
    
    def generate_svg(self, json_file_path: str, output_path: str = None) -> str:
        """Hauptfunktion: JSON lesen, Layout berechnen, SVG generieren"""
        
        # 1. JSON lesen
        json_data = self.load_json_data(json_file_path)
        
        # 2. Komponenten parsen
        components = self.parse_components(json_data)
        control_flows = self.parse_control_flows(json_data)
        data_flows = self.parse_data_flows(json_data)
        
        # 3. Layout berechnen
        positioned_components = self.calculate_layout(components)
        
        # 4. SVG generieren
        svg_content = self._create_full_svg(positioned_components, control_flows, data_flows, json_data)
        
        # 5. Speichern
        if output_path is None:
            base_name = Path(json_file_path).stem.replace('_RA_Analysis', '')
            output_path = f"new/{base_name}_SVG_RA_Diagram.svg"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        print(f"[SVG RUP] SVG diagram generated: {output_path}")
        return output_path
    
    def _create_full_svg(self, components: List[RAComponent], control_flows: List[ControlFlow], 
                        data_flows: List[DataFlow], json_data: Dict) -> str:
        """Erstelle vollständiges SVG-Dokument"""
        
        # SVG Header mit Definitionen und weißem Hintergrund
        svg_header = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{self.canvas_width}" height="{self.canvas_height}" 
     viewBox="0 0 {self.canvas_width} {self.canvas_height}"
     xmlns="http://www.w3.org/2000/svg"
     style="background: white;">

    <!-- Marker für Pfeile -->
    <defs>
        <!-- Control Flow Marker (blau) -->
        <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="blue"/>
        </marker>
        <!-- Data Flow USE Marker (orange: Entity -> Controller) -->
        <marker id="orangearrowhead" markerWidth="10" markerHeight="7" 
                refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="orange"/>
        </marker>
        <!-- Data Flow PROVIDE Marker (grün: Controller -> Entity) -->
        <marker id="greenarrowhead" markerWidth="10" markerHeight="7" 
                refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="green"/>
        </marker>
    </defs>

    <!-- Titel -->
    <text x="{self.canvas_width/2}" y="30" text-anchor="middle" 
          font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="black">
        RUP Robustness Analysis - {json_data.get('meta', {}).get('use_case', 'UC')}
    </text>

'''
        
        # Komponenten hinzufügen
        component_svg = ""
        for component in components:
            component_svg += self.create_svg_component(component) + "\n"
        
        # Control Flows hinzufügen
        control_flow_svg = ""
        for cf in control_flows:
            control_flow_svg += self.create_svg_control_flow(cf, components) + "\n"
        
        # Data Flows hinzufügen
        data_flow_svg = ""
        for df in data_flows:
            data_flow_svg += self.create_svg_data_flow(df, components) + "\n"
        
        # SVG Footer
        svg_footer = "</svg>"
        
        return svg_header + component_svg + control_flow_svg + data_flow_svg + svg_footer


def generate_svg_rup_diagram(json_file_path: str) -> str:
    """Convenience function für externe Nutzung"""
    visualizer = SVGRUPVisualizer()
    return visualizer.generate_svg(json_file_path)


if __name__ == "__main__":
    # Test mit UC1
    json_path = "new/UC1_Structured_RA_Analysis.json"
    generate_svg_rup_diagram(json_path)