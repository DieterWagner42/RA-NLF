#!/usr/bin/env python3
"""
Official RUP/UML Robustness Analysis Diagram Engine
Based on official symbols from https://de.wikipedia.org/wiki/Robustheitsanalyse
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import math
from dataclasses import dataclass, field
from enum import Enum

class ComponentType(Enum):
    ACTOR = 'actor'
    BOUNDARY = 'boundary'
    CONTROLLER = 'controller'
    ENTITY = 'entity'
    PARALLEL_NODE = 'parallel_node'

@dataclass
class RAComponent:
    """RUP/UML compliant RA component"""
    id: str
    label: str
    component_type: ComponentType
    stereotype: str
    position: Tuple[float, float] = (0, 0)
    warnings: List[str] = field(default_factory=list)

class OfficialRUPEngine:
    """
    Official RUP/UML Robustness Analysis engine using Wikipedia-documented symbols:
    - Akteur: Stick figure (Strichmännchen)
    - Boundary-Objekt: Rounded rectangle (Rechteck mit abgerundeten Kanten)  
    - Control-Objekt: Circle with left arrow (Kreis mit Pfeil < nach links)
    - Entity-Objekt: Circle with tangent line at bottom (Kreis mit Linie unten)
    """
    
    def __init__(self, figure_size=(20, 16)):
        self.figure_size = figure_size
        self.dpi = 300
        self.official_styles = {
            ComponentType.ENTITY: {
                'symbol_type': 'circle_with_line',
                'radius': 0.015,  # 50% of 0.03
                'fill_color': '#FFF3E0',
                'border_color': '#F57C00',
                'linewidth': 1,   # 50% of 2
                'text_offset': (0, -0.025),  # 50% of -0.05
                'fontsize': 6,    # 75% of 8 (more readable)
                'fontweight': 'bold'
            },
            ComponentType.CONTROLLER: {
                'symbol_type': 'circle_with_arrow',
                'radius': 0.015,  # 50% of 0.03
                'fill_color': '#F0F8E8',
                'border_color': '#2E7D32',
                'linewidth': 1,   # 50% of 2
                'text_offset': (0, -0.025),  # 50% of -0.05
                'fontsize': 6,    # 75% of 8 (more readable)
                'fontweight': 'bold'
            },
            ComponentType.BOUNDARY: {
                'symbol_type': 'rounded_rectangle',
                'width': 0.04,    # 50% of 0.08
                'height': 0.025,  # 50% of 0.05
                'corner_radius': 0.0075,  # 50% of 0.015
                'fill_color': '#E8F4FD',
                'border_color': '#2E86AB',
                'linewidth': 1,   # 50% of 2
                'text_offset': (0, -0.025),  # 50% of -0.05
                'fontsize': 6,    # 75% of 8 (more readable)
                'fontweight': 'bold'
            },
            ComponentType.ACTOR: {
                'symbol_type': 'stickman',
                'head_radius': 0.0075,  # 50% of 0.015
                'body_height': 0.02,    # 50% of 0.04
                'arm_span': 0.015,      # 50% of 0.03
                'leg_span': 0.0125,     # 50% of 0.025
                'color': '#000000',
                'linewidth': 1,         # 50% of 2
                'text_offset': (0, -0.025),  # 50% of -0.05
                'fontsize': 6,          # 75% of 8 (more readable)
                'fontweight': 'bold'
            },
            ComponentType.PARALLEL_NODE: {
                'symbol_type': 'diamond',
                'size': 0.01,           # Viel kleinere Diamantgröße
                'fill_color': '#1F2937', # Dunkelgrau gefüllt
                'border_color': '#1F2937',
                'linewidth': 1,
                'text_offset': (0, -0.025),
                'fontsize': 4,
                'fontweight': 'bold'
            }
        }

    def draw_official_actor(self, ax, x: float, y: float):
        """Draw official stick figure symbol"""
        style = self.official_styles[ComponentType.ACTOR]
        head_radius = style['head_radius']
        body_height = style['body_height']
        arm_span = style['arm_span']
        leg_span = style['leg_span']
        color = style['color']
        linewidth = style['linewidth']
        
        # Head (circle)
        head = Circle((x, y + body_height * 0.6), head_radius, 
                     facecolor='white', edgecolor=color, linewidth=linewidth)
        ax.add_patch(head)
        
        # Body (vertical line)
        body_top = y + body_height * 0.6 - head_radius
        body_bottom = y - body_height * 0.4
        ax.plot([x, x], [body_top, body_bottom], color=color, linewidth=linewidth)
        
        # Arms (horizontal line)
        arm_y = y + body_height * 0.2
        ax.plot([x - arm_span/2, x + arm_span/2], [arm_y, arm_y], color=color, linewidth=linewidth)
        
        # Legs (two diagonal lines)
        ax.plot([x, x - leg_span/2], [body_bottom, y - body_height], color=color, linewidth=linewidth)
        ax.plot([x, x + leg_span/2], [body_bottom, y - body_height], color=color, linewidth=linewidth)

    def draw_official_boundary(self, ax, x: float, y: float):
        """Draw official boundary symbol: Circle with T symbol"""
        style = self.official_styles[ComponentType.BOUNDARY]
        width = style['width']
        height = style['height']
        color = style['border_color']
        fill_color = style['fill_color']
        linewidth = style['linewidth']
        
        # Draw circle
        radius = 0.03
        circle = Circle((x, y), radius, facecolor=fill_color, edgecolor=color, linewidth=linewidth)
        ax.add_patch(circle)
        
        # Draw T symbol (rotated 90 degrees counter-clockwise)
        t_size = radius * 0.6
        # Vertical line of T
        ax.plot([x - radius, x - radius-t_size], [y , y ], 
               color=color, linewidth=linewidth)
        # Horizontal line of T
        ax.plot([x - radius-t_size,x - radius-t_size], [y + t_size, y - t_size], 
               color=color, linewidth=linewidth)

    def draw_official_controller(self, ax, x: float, y: float):
        """Draw official controller symbol: Circle with left arrow"""
        style = self.official_styles[ComponentType.CONTROLLER]
        radius = style['radius']
        color = style['border_color']
        fill_color = style['fill_color']
        linewidth = style['linewidth']
        
        # Circle
        circle = Circle((x, y), radius, facecolor=fill_color, edgecolor=color, linewidth=linewidth)
        ax.add_patch(circle)
        
        # Arrow < pointing left (centered in circle)
        arrow_size = radius * 0.4
        ax.plot([x + arrow_size, x - arrow_size], [y + arrow_size*0.6+radius, y+radius], 
               color=color, linewidth=linewidth)
        ax.plot([x + arrow_size, x - arrow_size], [y - arrow_size*0.6+radius, y+radius], 
               color=color, linewidth=linewidth)

    def draw_official_entity(self, ax, x: float, y: float):
        """Draw official entity symbol: Circle with tangential line at bottom"""
        style = self.official_styles[ComponentType.ENTITY]
        radius = style['radius']
        color = style['border_color']
        fill_color = style['fill_color']
        linewidth = style['linewidth']
        
        # Circle
        circle = Circle((x, y), radius, facecolor=fill_color, edgecolor=color, linewidth=linewidth)
        ax.add_patch(circle)
        
        # Tangent line at bottom
        line_length = radius * 2.5
        line_y = y - radius
        ax.plot([x - line_length/2, x + line_length/2], [line_y, line_y], 
               color=color, linewidth=linewidth)

    def draw_official_parallel_node(self, ax, x: float, y: float):
        """Draw parallel flow node as filled diamond"""
        style = self.official_styles[ComponentType.PARALLEL_NODE]
        size = style['size']
        color = style['border_color']
        fill_color = style['fill_color']
        linewidth = style['linewidth']
        
        # Create diamond vertices
        diamond_points = np.array([
            [x, y + size],      # Top
            [x + size, y],      # Right
            [x, y - size],      # Bottom
            [x - size, y]       # Left
        ])
        
        # Draw filled diamond
        diamond = Polygon(diamond_points, facecolor=fill_color, edgecolor=color, linewidth=linewidth)
        ax.add_patch(diamond)

    def draw_official_component(self, component: RAComponent, ax):
        """Draw component using official RUP symbols"""
        x, y = component.position
        
        if component.component_type == ComponentType.ACTOR:
            self.draw_official_actor(ax, x, y)
        elif component.component_type == ComponentType.BOUNDARY:
            self.draw_official_boundary(ax, x, y)
        elif component.component_type == ComponentType.CONTROLLER:
            self.draw_official_controller(ax, x, y)
        elif component.component_type == ComponentType.ENTITY:
            self.draw_official_entity(ax, x, y)
        elif component.component_type == ComponentType.PARALLEL_NODE:
            self.draw_official_parallel_node(ax, x, y)

        # Add label
        style = self.official_styles[component.component_type]
        text_x = x + style['text_offset'][0]
        text_y = y + style['text_offset'][1]
        
        # Clean and format label
        clean_label = component.label
        if len(clean_label) > 12:
            words = clean_label.split()
            if len(words) > 1:
                mid_point = len(words) // 2
                line1 = ' '.join(words[:mid_point])
                line2 = ' '.join(words[mid_point:])
                clean_label = f"{line1}\\n{line2}"
        
        ax.text(text_x, text_y, clean_label, ha='center', va='top',
               fontsize=style['fontsize'], fontweight=style['fontweight'],
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85,
                        edgecolor='lightgray', linewidth=0.5))

    def create_official_rup_diagram_from_json(self, json_file_path: str, title: str = None):
        """Create official RUP diagram from JSON analysis file"""
        
        # Load JSON analysis
        with open(json_file_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        # Use JSON metadata for title if not provided
        if title is None:
            use_case = analysis_data['meta']['use_case']
            title = f"{use_case} RA Diagram"
        
        # Convert JSON components to RAComponent objects
        actors = []
        boundaries = []
        controllers = []
        entities = []
        parallel_nodes = []
        
        # Extract actors
        for actor_data in analysis_data['components']['actors']:
            comp = RAComponent(actor_data['name'], actor_data['name'], ComponentType.ACTOR, "<<actor>>")
            actors.append(comp)
        
        # Extract boundaries
        for boundary_data in analysis_data['components']['boundaries']:
            comp = RAComponent(boundary_data['name'], boundary_data['name'], ComponentType.BOUNDARY, "<<boundary>>")
            boundaries.append(comp)
        
        # Extract controllers
        for controller_data in analysis_data['components']['controllers']:
            comp = RAComponent(controller_data['name'], controller_data['name'], ComponentType.CONTROLLER, "<<controller>>")
            controllers.append(comp)
        
        # Extract entities (all entities)
        for entity_data in analysis_data['components']['entities']:
            comp = RAComponent(entity_data['name'], entity_data['name'], ComponentType.ENTITY, "<<entity>>")
            entities.append(comp)
        
        # Extract parallel flow nodes
        if 'control_flow_nodes' in analysis_data['components']:
            for node_data in analysis_data['components']['control_flow_nodes']:
                comp = RAComponent(node_data['name'], node_data['name'], ComponentType.PARALLEL_NODE, f"<<{node_data['type']}>>")
                parallel_nodes.append(comp)
        
        all_components = actors + boundaries + controllers + entities + parallel_nodes
        
        # Store analysis data for layout calculation
        self.analysis_data = analysis_data
        
        # Layout components
        self.calculate_layout(actors, boundaries, controllers, entities, parallel_nodes)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size, dpi=self.dpi)
        fig.patch.set_facecolor('white')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=16, 
               fontweight='bold', transform=ax.transAxes)
        
        # Draw all components
        for component in all_components:
            self.draw_official_component(component, ax)
        
        # Draw relationships if available
        self.draw_relationships(ax, analysis_data['relationships'], all_components)
        
        # Add legend
        self.add_legend(ax)
        
        # Add metadata info
        self.add_metadata_info(ax, analysis_data)
        
        # Save diagram
        uc_name = analysis_data['meta']['use_case']
        output_file = f"{uc_name}_RA_Diagram_Official_RUP.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_file

    def create_official_rup_diagram_from_ra_classes(self, ra_classes: List, title: str = "Official RUP RA Diagram"):
        """Create official RUP diagram directly from RA classes (legacy method)"""
        
        # Convert RA classes to components
        actors = []
        boundaries = []
        controllers = []
        entities = []
        
        for ra_class in ra_classes:
            if ra_class.type == "Actor":
                comp = RAComponent(ra_class.name, ra_class.name, ComponentType.ACTOR, "<<actor>>")
                actors.append(comp)
            elif ra_class.type == "Boundary":
                comp = RAComponent(ra_class.name, ra_class.name, ComponentType.BOUNDARY, "<<boundary>>")
                boundaries.append(comp)
            elif ra_class.type == "Controller":
                comp = RAComponent(ra_class.name, ra_class.name, ComponentType.CONTROLLER, "<<controller>>")
                controllers.append(comp)
            elif ra_class.type == "Entity":
                comp = RAComponent(ra_class.name, ra_class.name, ComponentType.ENTITY, "<<entity>>")
                entities.append(comp)
        
        all_components = actors + boundaries + controllers + entities
        
        # Layout components
        self.calculate_layout(actors, boundaries, controllers, entities)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size, dpi=self.dpi)
        fig.patch.set_facecolor('white')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=16, 
               fontweight='bold', transform=ax.transAxes)
        
        # Draw all components
        for component in all_components:
            self.draw_official_component(component, ax)
        
        # Add legend
        self.add_legend(ax)
        
        # Save diagram
        output_file = f"{title.replace(' ', '_').replace(':', '')}_Official_RUP.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_file

    def calculate_layout(self, actors, boundaries, controllers, entities, parallel_nodes=None):
        """Calculate control flow based layout - controllers top to bottom, parallel controllers side by side"""
        
        # Load control flow information from JSON to determine layout
        if hasattr(self, 'analysis_data') and 'relationships' in self.analysis_data:
            self.calculate_control_flow_layout(actors, boundaries, controllers, entities, parallel_nodes)
        else:
            # Fallback to original layout if no control flow data
            self.calculate_original_layout(actors, boundaries, controllers, entities, parallel_nodes)
    
    def calculate_control_flow_layout(self, actors, boundaries, controllers, entities, parallel_nodes):
        """Layout controllers based on control flow - top to bottom, parallel side by side"""
        
        # Position actors on far left
        if actors:
            self.position_column(actors, 0.05, 0.9, 0.1)
        
        # Position boundaries on left
        if boundaries:
            self.position_column(boundaries[:8], 0.18, 0.9, 0.1)
        
        # Get control flows to determine controller sequence
        control_flows = self.analysis_data['relationships'].get('control_flows', [])
        
        # Build control flow graph
        controller_order = self.build_controller_sequence(controllers, control_flows, parallel_nodes)
        
        # Position controllers based on control flow sequence
        self.position_controllers_by_flow(controller_order, parallel_nodes)
        
        # Position entities on the right
        if entities:
            entity_cols = 2
            entities_per_col = len(entities) // entity_cols + (1 if len(entities) % entity_cols else 0)
            for col in range(entity_cols):
                start_idx = col * entities_per_col
                end_idx = min(start_idx + entities_per_col, len(entities))
                col_entities = entities[start_idx:end_idx]
                if col_entities:
                    x_pos = 0.82 + col * 0.1
                    self.position_column(col_entities, x_pos, 0.9, 0.1)
    
    def calculate_original_layout(self, actors, boundaries, controllers, entities, parallel_nodes):
        """Original layout as fallback"""
        y_start = 0.9
        y_end = 0.1
        
        if actors:
            self.position_column(actors, 0.15, y_start, y_end)
        if boundaries:
            self.position_column(boundaries[:8], 0.35, y_start, y_end)
        if controllers:
            cols = 3
            controllers_per_col = len(controllers) // cols + (1 if len(controllers) % cols else 0)
            for col in range(cols):
                start_idx = col * controllers_per_col
                end_idx = min(start_idx + controllers_per_col, len(controllers))
                col_controllers = controllers[start_idx:end_idx]
                if col_controllers:
                    x_pos = 0.45 + col * 0.1
                    self.position_column(col_controllers, x_pos, y_start, y_end)
        if parallel_nodes:
            self.position_column(parallel_nodes, 0.78, y_start, y_end)
        if entities:
            entity_cols = 3
            entities_per_col = len(entities) // entity_cols + (1 if len(entities) % entity_cols else 0)
            for col in range(entity_cols):
                start_idx = col * entities_per_col
                end_idx = min(start_idx + entities_per_col, len(entities))
                col_entities = entities[start_idx:end_idx]
                if col_entities:
                    x_pos = 0.85 + col * 0.08
                    self.position_column(col_entities, x_pos, y_start, y_end)
    
    def build_controller_sequence(self, controllers, control_flows, parallel_nodes):
        """Build controller sequence based on control flows - identify parallel groups"""
        controller_map = {c.label: c for c in controllers}
        
        # Analyze control flows to identify parallel groups
        sequence = []
        
        # Group controllers by their step patterns to identify parallel flows
        # B2a, B2b, B2c, B2d are parallel (same base number)
        # B3a, B3b are parallel
        parallel_groups = {}
        sequential_controllers = []
        safety_controllers = []
        
        for controller in controllers:
            name = controller.label
            # Filter out safety/hygiene controllers from main flow
            if any(safety in name.lower() for safety in ['foodsafety', 'haccp', 'hygiene']):
                safety_controllers.append(controller)
                continue
                
            # Check if controller name suggests parallel execution
            if any(step in name.lower() for step in ['waterheater', 'filter', 'setamount', 'cup']) and 'manager' in name.lower():
                # These are B2a-B2d parallel controllers
                if 'b2_parallel' not in parallel_groups:
                    parallel_groups['b2_parallel'] = []
                parallel_groups['b2_parallel'].append(controller)
            elif any(step in name.lower() for step in ['coffee', 'milk']) and 'manager' in name.lower():
                # These are B3a-B3b parallel controllers  
                if 'b3_parallel' not in parallel_groups:
                    parallel_groups['b3_parallel'] = []
                parallel_groups['b3_parallel'].append(controller)
            else:
                sequential_controllers.append(controller)
        
        # Find the correct B1 start controller (TimeManager)
        b1_controller = None
        other_controllers = []
        
        for controller in sequential_controllers:
            if 'time' in controller.label.lower():
                b1_controller = controller
            else:
                other_controllers.append(controller)
        
        # Build sequence: start with TimeManager (B1), then parallel groups, then rest
        if b1_controller:
            sequence.append([b1_controller])  # TimeManager (B1)
        
        # Add B2 parallel group
        if 'b2_parallel' in parallel_groups:
            sequence.append(parallel_groups['b2_parallel'])
        
        # Add B3 parallel group  
        if 'b3_parallel' in parallel_groups:
            sequence.append(parallel_groups['b3_parallel'])
        
        # Add remaining sequential controllers (B4, B5, etc.)
        for controller in other_controllers:
            sequence.append([controller])
        
        return sequence
    
    def position_controllers_by_flow(self, controller_order, parallel_nodes):
        """Position controllers based on control flow - top to bottom, parallel side by side"""
        y_start = 0.85
        y_spacing = 0.1  # Vertical spacing between rows
        x_center = 0.5  # Center axis for controllers and diamonds
        x_controller_spacing = 0.12  # Horizontal spacing for parallel controllers
        
        current_y = y_start
        diamond_positions = {}
        
        # Position controllers in sequence
        for level, controller_group in enumerate(controller_order):
            if len(controller_group) == 1:
                # Single controller - center it on center axis
                controller_group[0].position = (x_center, current_y)
            else:
                # Multiple parallel controllers - space them horizontally around center
                total_width = (len(controller_group) - 1) * x_controller_spacing
                start_x = x_center - total_width / 2
                
                for i, controller in enumerate(controller_group):
                    x_pos = start_x + i * x_controller_spacing
                    controller.position = (x_pos, current_y)
                
                # Mark positions for parallel flow diamonds
                if level == 1:  # B2 parallel group
                    diamond_positions['P1_START'] = (x_center, current_y + 0.05)
                    diamond_positions['P1_END'] = (x_center, current_y - 0.05)
                elif level == 2:  # B3 parallel group
                    diamond_positions['P2_START'] = (x_center, current_y + 0.05)
                    diamond_positions['P2_END'] = (x_center, current_y - 0.05)
            
            current_y -= y_spacing
        
        # Position parallel nodes on center axis
        if parallel_nodes:
            for node in parallel_nodes:
                node_name = node.label
                if node_name in diamond_positions:
                    node.position = diamond_positions[node_name]
                else:
                    # Fallback positioning on center axis
                    if "START" in node_name:
                        node.position = (x_center, y_start - 0.03)
                    elif "END" in node_name:
                        node.position = (x_center, y_start - 0.13)

    def position_column(self, components: List[RAComponent], x_position: float, y_start: float, y_end: float):
        """Position components in a vertical column with adequate spacing"""
        if not components:
            return
        
        if len(components) == 1:
            components[0].position = (x_position, (y_start + y_end) / 2)
            return
        
        # Calculate spacing with minimum distance
        min_spacing = 0.08  # Minimum vertical distance between components
        available_height = y_start - y_end
        required_height = (len(components) - 1) * min_spacing
        
        if required_height > available_height:
            # Use available space
            y_positions = np.linspace(y_start, y_end, len(components))
        else:
            # Use minimum spacing, center the column
            total_height = required_height
            start_y = y_start - (available_height - total_height) / 2
            y_positions = [start_y - i * min_spacing for i in range(len(components))]
        
        for component, y_pos in zip(components, y_positions):
            component.position = (x_position, y_pos)

    def draw_relationships(self, ax, relationships: Dict, components: List[RAComponent]):
        """Draw relationship arrows between components"""
        # Create component position lookup
        comp_positions = {comp.label: comp.position for comp in components}
        
        # Draw control flows
        for flow in relationships.get('control_flows', []):
            # Handle both old and new JSON formats
            source = flow.get('source') or flow.get('from')
            dest = flow.get('destination') or flow.get('to')
            
            if source in comp_positions and dest in comp_positions:
                src_pos = comp_positions[source]
                dest_pos = comp_positions[dest]
                
                # Draw arrow from source to destination
                ax.annotate('', xy=dest_pos, xytext=src_pos,
                           arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, alpha=0.7))
        
        # Draw data flows (USE/PROVIDE relationships)
        for flow in relationships.get('data_flows', []):
            # Handle new JSON format with controller/entity keys
            source = flow.get('source') or flow.get('controller')
            dest = flow.get('destination') or flow.get('entity')
            flow_type = flow.get('type', 'use')
            
            if source in comp_positions and dest in comp_positions:
                src_pos = comp_positions[source]
                dest_pos = comp_positions[dest]
                
                # Different styles for USE vs PROVIDE
                if flow_type == 'use':
                    # USE: dashed green line (controller uses entity)
                    ax.annotate('', xy=dest_pos, xytext=src_pos,
                               arrowprops=dict(arrowstyle='->', color='green', lw=1.5, alpha=0.6, linestyle='dashed'))
                else:  # provide
                    # PROVIDE: solid orange line (controller provides entity)
                    ax.annotate('', xy=dest_pos, xytext=src_pos,
                               arrowprops=dict(arrowstyle='->', color='orange', lw=1.5, alpha=0.6))
        
        # Draw actor-boundary connections
        for conn in relationships.get('actor_boundary_connections', []):
            source = conn.get('source') or conn.get('actor')
            dest = conn.get('destination') or conn.get('boundary')
            
            if source in comp_positions and dest in comp_positions:
                src_pos = comp_positions[source]
                dest_pos = comp_positions[dest]
                
                # Draw arrow from actor to boundary
                ax.annotate('', xy=dest_pos, xytext=src_pos,
                           arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.8))

    def add_metadata_info(self, ax, analysis_data: Dict):
        """Add metadata information to diagram"""
        meta = analysis_data['meta']
        summary = analysis_data['summary']
        
        info_text = f"Domain: {meta['domain']} | Components: {summary['total_components']} | " \
                   f"Generated: {meta['generated_at'][:19]}"
        
        ax.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=8,
               transform=ax.transAxes, style='italic', alpha=0.7)

    def add_legend(self, ax):
        """Add RUP symbol legend"""
        legend_x = 0.05
        legend_y = 0.1
        
        ax.text(legend_x, legend_y + 0.06, "RUP/UML Symbole:", fontsize=10, fontweight='bold',
               transform=ax.transAxes)
        
        # Create legend components
        legend_items = [
            ("Actor", ComponentType.ACTOR),
            ("Boundary", ComponentType.BOUNDARY), 
            ("Controller", ComponentType.CONTROLLER),
            ("Entity", ComponentType.ENTITY),
            ("Parallel Node", ComponentType.PARALLEL_NODE)
        ]
        
        for i, (name, comp_type) in enumerate(legend_items):
            y_offset = legend_y - i * 0.02
            
            # Draw mini symbol
            if comp_type == ComponentType.ACTOR:
                self.draw_official_actor(ax, legend_x + 0.02, y_offset - 0.01)
            elif comp_type == ComponentType.BOUNDARY:
                self.draw_official_boundary(ax, legend_x + 0.02, y_offset)
            elif comp_type == ComponentType.CONTROLLER:
                self.draw_official_controller(ax, legend_x + 0.02, y_offset)
            elif comp_type == ComponentType.ENTITY:
                self.draw_official_entity(ax, legend_x + 0.02, y_offset)
            elif comp_type == ComponentType.PARALLEL_NODE:
                self.draw_official_parallel_node(ax, legend_x + 0.02, y_offset)
            
            # Add label
            ax.text(legend_x + 0.05, y_offset, name, fontsize=9, va='center',
                   transform=ax.transAxes)
        
        # Add relationship legend
        rel_y = legend_y - 0.14
        ax.text(legend_x, rel_y, "Beziehungen:", fontsize=9, fontweight='bold',
               transform=ax.transAxes)
        
        # Control flow line
        ax.plot([legend_x + 0.01, legend_x + 0.04], [rel_y - 0.02, rel_y - 0.02], 
               color='blue', lw=1.5, alpha=0.7, transform=ax.transAxes)
        ax.text(legend_x + 0.05, rel_y - 0.02, "Control Flow", fontsize=8, va='center',
               transform=ax.transAxes)
        
        # Data flow USE line (dashed green)
        ax.plot([legend_x + 0.01, legend_x + 0.04], [rel_y - 0.04, rel_y - 0.04], 
               color='green', lw=1.5, alpha=0.6, linestyle='dashed', transform=ax.transAxes)
        ax.text(legend_x + 0.05, rel_y - 0.04, "Data Flow (USE)", fontsize=8, va='center',
               transform=ax.transAxes)
        
        # Data flow PROVIDE line (solid orange)
        ax.plot([legend_x + 0.01, legend_x + 0.04], [rel_y - 0.06, rel_y - 0.06], 
               color='orange', lw=1.5, alpha=0.6, transform=ax.transAxes)
        ax.text(legend_x + 0.05, rel_y - 0.06, "Data Flow (PROVIDE)", fontsize=8, va='center',
               transform=ax.transAxes)
        
        # Actor-boundary line (red)
        ax.plot([legend_x + 0.01, legend_x + 0.04], [rel_y - 0.08, rel_y - 0.08], 
               color='red', lw=2, alpha=0.8, transform=ax.transAxes)
        ax.text(legend_x + 0.05, rel_y - 0.08, "Actor Trigger", fontsize=8, va='center',
               transform=ax.transAxes)


def generate_rup_diagram_from_json(json_file_path: str) -> str:
    """Generate RA diagram from JSON analysis file"""
    engine = OfficialRUPEngine(figure_size=(20, 16))
    output_file = engine.create_official_rup_diagram_from_json(json_file_path)
    return output_file

def generate_uc3_official_rup_diagram(ra_classes: List) -> str:
    """Generate UC3 RA diagram with official RUP symbols (legacy)"""
    engine = OfficialRUPEngine(figure_size=(20, 16))
    output_file = engine.create_official_rup_diagram_from_ra_classes(
        ra_classes=ra_classes,
        title="UC3: Rocket Launch RA Diagram"
    )
    return output_file


def main():
    """Demo of official RUP/UML engine using JSON format"""
    print("Official RUP/UML Robustness Analysis Engine")
    print("Using Wikipedia-documented symbols and JSON format")
    print("=" * 60)
    
    # Try to find and process existing JSON analysis
    json_file = "UC3_Rocket_Launch_Improved_Structured_RA_Analysis.json"
    
    if Path(json_file).exists():
        try:
            print(f"Found JSON analysis file: {json_file}")
            
            # Create engine and generate diagram from JSON
            engine = OfficialRUPEngine(figure_size=(20, 16))
            output_file = engine.create_official_rup_diagram_from_json(json_file)
            
            print(f"Official RUP diagram generated: {output_file}")
            
        except Exception as e:
            print(f"Error creating diagram from JSON: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Fallback to legacy method
        try:
            import sys
            import os
            sys.path.append('src')
            
            from generic_uc_analyzer import GenericUCAnalyzer
            
            # Analyze UC3
            analyzer = GenericUCAnalyzer(domain_name='rocket_science')
            verb_analyses, ra_classes = analyzer.analyze_uc_file('Use Case/UC3_Rocket_Launch_Improved.txt')
            
            print(f"Analyzed UC3: {len(ra_classes)} RA classes found")
            
            # Generate official RUP diagram
            output_file = generate_uc3_official_rup_diagram(ra_classes)
            print(f"Official RUP diagram generated: {output_file}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure UC3_Rocket_Launch_Improved.txt exists and analysis engine is available")


if __name__ == '__main__':
    main()