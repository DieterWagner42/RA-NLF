"""
Enhanced RUP/UML Compliant RA Diagram Engine
Optimized to exactly match the reference diagram style
Based on RA diagrams/RA example.png
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Polygon
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import math
from dataclasses import dataclass, field
from enum import Enum

class ComponentType(Enum):
    ACTOR = "actor"
    BOUNDARY = "boundary" 
    CONTROLLER = "controller"
    ENTITY = "entity"

@dataclass
class RAComponent:
    """Enhanced RUP/UML compliant RA component"""
    id: str
    label: str
    component_type: ComponentType
    stereotype: str
    position: Tuple[float, float] = (0, 0)
    is_main_flow: bool = True  # For layout priority

class EnhancedRUPEngine:
    """
    Enhanced RUP/UML compliant engine that precisely matches the reference style
    """
    
    def __init__(self, figure_size: Tuple[int, int] = (20, 14)):
        self.figure_size = figure_size
        self.dpi = 300
        
        # Precise styling matching reference diagram
        self.reference_styles = {
            ComponentType.ACTOR: {
                "head_size": 0.015,
                "body_height": 0.04,
                "arm_width": 0.025,
                "leg_width": 0.025,
                "color": "#000000",
                "linewidth": 1.5,
                "text_offset": (0, -0.08),
                "fontsize": 9
            },
            
            ComponentType.BOUNDARY: {
                "radius": 0.025,
                "line_length": 0.035,
                "fill_color": "#B0E0E6",  # Powder blue like reference
                "border_color": "#4682B4",
                "linewidth": 1.5,
                "text_offset": (0, -0.06),
                "fontsize": 9
            },
            
            ComponentType.CONTROLLER: {
                "radius": 0.035,  # Larger circles like reference
                "fill_color": "#87CEEB",  # Sky blue like reference
                "border_color": "#4682B4",
                "linewidth": 1.5,
                "text_offset": (0, -0.08),
                "fontsize": 9
            },
            
            ComponentType.ENTITY: {
                "width": 0.06,
                "height": 0.04,
                "fill_color": "#F0F8FF",
                "border_color": "#4682B4",
                "linewidth": 1.5,
                "text_offset": (0, -0.07),
                "fontsize": 9
            }
        }
        
        # Edge styling exactly like reference
        self.reference_edge_style = {
            "color": "#000000",
            "linewidth": 1.2,
            "alpha": 1.0
        }

    def load_and_parse_json(self, json_file_path: str) -> Tuple[List[RAComponent], List[Dict], Dict[str, Any]]:
        """Load and parse JSON data"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        components = self.parse_components_enhanced(json_data)
        edges = self.parse_edges_enhanced(json_data)
        
        return components, edges, json_data

    def parse_components_enhanced(self, json_data: Dict[str, Any]) -> List[RAComponent]:
        """Parse components with enhanced categorization"""
        components = []
        
        # Get nodes from JSON
        nodes = []
        if "graph" in json_data and "nodes" in json_data["graph"]:
            nodes = json_data["graph"]["nodes"]
        elif "components" in json_data and "nodes" in json_data["components"]:
            nodes = json_data["components"]["nodes"]
        
        for node in nodes:
            # Determine if component is in main flow (for layout priority)
            is_main_flow = not any(term in node["label"].lower() 
                                 for term in ["error", "alternative", "extension", "supply"])
            
            component = RAComponent(
                id=node["id"],
                label=node["label"],
                component_type=ComponentType(node["type"]),
                stereotype=node["stereotype"],
                is_main_flow=is_main_flow
            )
            components.append(component)
        
        return components

    def parse_edges_enhanced(self, json_data: Dict[str, Any]) -> List[Dict]:
        """Parse edges for enhanced visualization"""
        edges = []
        
        edge_list = []
        if "graph" in json_data and "edges" in json_data["graph"]:
            edge_list = json_data["graph"]["edges"]
        elif "components" in json_data and "edges" in json_data["components"]:
            edge_list = json_data["components"]["edges"]
        
        for edge in edge_list:
            edges.append({
                "source": edge["source"],
                "target": edge["target"],
                "type": edge.get("type", "control_flow"),
                "label": edge.get("label", ""),
                "relationship": edge.get("relationship", "")
            })
        
        return edges

    def calculate_reference_style_layout(self, components: List[RAComponent], edges: List[Dict]) -> None:
        """
        Calculate layout that matches the organic style of the reference diagram
        """
        
        # Separate by type and importance
        actors = [c for c in components if c.component_type == ComponentType.ACTOR]
        boundaries = [c for c in components if c.component_type == ComponentType.BOUNDARY]
        controllers = [c for c in components if c.component_type == ComponentType.CONTROLLER]
        entities = [c for c in components if c.component_type == ComponentType.ENTITY]
        
        # Build adjacency graph for organic positioning
        adjacency = {comp.id: set() for comp in components}
        for edge in edges:
            if edge["source"] in adjacency and edge["target"] in adjacency:
                adjacency[edge["source"]].add(edge["target"])
                adjacency[edge["target"]].add(edge["source"])
        
        # Position actors around perimeter with some randomization
        self.position_actors_organically(actors)
        
        # Position main controllers first (central importance)
        main_controllers = [c for c in controllers if c.is_main_flow]
        secondary_controllers = [c for c in controllers if not c.is_main_flow]
        
        self.position_main_controllers(main_controllers, adjacency)
        self.position_secondary_controllers(secondary_controllers, main_controllers, adjacency)
        
        # Position boundaries and entities relative to controllers
        self.position_support_components(boundaries + entities, controllers, adjacency)

    def position_actors_organically(self, actors: List[RAComponent]) -> None:
        """Position actors in organic perimeter positions"""
        if not actors:
            return
        
        # Organic positions around perimeter (inspired by reference)
        organic_positions = [
            (0.05, 0.35),   # Left side
            (0.95, 0.45),   # Right side
            (0.88, 0.85),   # Top right
            (0.12, 0.75),   # Left upper
            (0.50, 0.92),   # Top center
            (0.08, 0.15),   # Bottom left
            (0.92, 0.12),   # Bottom right
            (0.75, 0.05),   # Bottom right corner
            (0.25, 0.08),   # Bottom left area
        ]
        
        for i, actor in enumerate(actors):
            if i < len(organic_positions):
                actor.position = organic_positions[i]
            else:
                # Spiral outward for additional actors
                angle = 2 * math.pi * i / len(actors)
                radius = 0.35 + 0.1 * (i // len(organic_positions))
                x = 0.5 + radius * math.cos(angle)
                y = 0.5 + radius * math.sin(angle)
                actor.position = (max(0.05, min(0.95, x)), max(0.05, min(0.95, y)))

    def position_main_controllers(self, main_controllers: List[RAComponent], adjacency: Dict[str, set]) -> None:
        """Position main controllers in organic central layout"""
        if not main_controllers:
            return
        
        # Create organic central positioning using force simulation
        positions = {}
        
        # Initialize positions in organic clusters
        if len(main_controllers) <= 3:
            # Small number - linear arrangement
            for i, controller in enumerate(main_controllers):
                x = 0.3 + (i * 0.4) / max(1, len(main_controllers) - 1)
                y = 0.5 + 0.1 * math.sin(i * math.pi / 2)
                positions[controller.id] = [x, y]
        else:
            # Larger number - organic cluster
            center_x, center_y = 0.5, 0.5
            for i, controller in enumerate(main_controllers):
                angle = 2 * math.pi * i / len(main_controllers)
                radius = 0.15 + 0.05 * (i % 3)  # Varying radius for organic look
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                positions[controller.id] = [x, y]
        
        # Apply force-directed layout for organic positioning
        for iteration in range(30):
            forces = {comp_id: [0, 0] for comp_id in positions}
            
            # Repulsion between controllers
            for comp1_id, pos1 in positions.items():
                for comp2_id, pos2 in positions.items():
                    if comp1_id != comp2_id:
                        dx = pos1[0] - pos2[0]
                        dy = pos1[1] - pos2[1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist > 0.01:
                            repulsion = 0.002 / (dist * dist)
                            forces[comp1_id][0] += repulsion * dx / dist
                            forces[comp1_id][1] += repulsion * dy / dist
            
            # Attraction between connected controllers
            for comp_id, neighbors in adjacency.items():
                if comp_id in positions:
                    pos1 = positions[comp_id]
                    for neighbor_id in neighbors:
                        if neighbor_id in positions:
                            pos2 = positions[neighbor_id]
                            dx = pos2[0] - pos1[0]
                            dy = pos2[1] - pos1[1]
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist > 0:
                                attraction = 0.001 * dist
                                forces[comp_id][0] += attraction * dx / dist
                                forces[comp_id][1] += attraction * dy / dist
            
            # Apply forces with damping
            damping = 0.8
            for comp_id, force in forces.items():
                positions[comp_id][0] += force[0] * damping
                positions[comp_id][1] += force[1] * damping
                # Constrain to central area
                positions[comp_id][0] = max(0.2, min(0.8, positions[comp_id][0]))
                positions[comp_id][1] = max(0.2, min(0.8, positions[comp_id][1]))
        
        # Assign final positions
        for controller in main_controllers:
            if controller.id in positions:
                controller.position = tuple(positions[controller.id])

    def position_secondary_controllers(self, secondary_controllers: List[RAComponent], 
                                     main_controllers: List[RAComponent], 
                                     adjacency: Dict[str, set]) -> None:
        """Position secondary controllers around main ones"""
        for controller in secondary_controllers:
            # Find closest main controller or use adjacency
            connected_main = []
            for neighbor_id in adjacency.get(controller.id, set()):
                main_controller = next((c for c in main_controllers if c.id == neighbor_id), None)
                if main_controller:
                    connected_main.append(main_controller)
            
            if connected_main:
                # Position near connected main controller
                main_pos = connected_main[0].position
                # Add organic offset
                angle = hash(controller.id) % 360 * math.pi / 180
                offset_x = 0.12 * math.cos(angle)
                offset_y = 0.12 * math.sin(angle)
                controller.position = (
                    max(0.1, min(0.9, main_pos[0] + offset_x)),
                    max(0.1, min(0.9, main_pos[1] + offset_y))
                )
            else:
                # Position in secondary area
                controller.position = (0.15, 0.8 - len(secondary_controllers) * 0.1)

    def position_support_components(self, support_components: List[RAComponent], 
                                   controllers: List[RAComponent], 
                                   adjacency: Dict[str, set]) -> None:
        """Position boundaries and entities near their connected controllers"""
        for component in support_components:
            # Find connected controllers
            connected_controllers = []
            for neighbor_id in adjacency.get(component.id, set()):
                controller = next((c for c in controllers if c.id == neighbor_id), None)
                if controller:
                    connected_controllers.append(controller)
            
            if connected_controllers:
                # Position near first connected controller
                controller_pos = connected_controllers[0].position
                # Organic offset based on component type
                if component.component_type == ComponentType.BOUNDARY:
                    offset_factor = 0.08
                else:  # Entity
                    offset_factor = 0.10
                
                # Use component hash for consistent but organic positioning
                angle = (hash(component.id) % 8) * math.pi / 4
                offset_x = offset_factor * math.cos(angle)
                offset_y = offset_factor * math.sin(angle)
                
                component.position = (
                    max(0.05, min(0.95, controller_pos[0] + offset_x)),
                    max(0.05, min(0.95, controller_pos[1] + offset_y))
                )
            else:
                # Default positioning
                if component.component_type == ComponentType.BOUNDARY:
                    component.position = (0.2, 0.8)
                else:
                    component.position = (0.8, 0.2)

    def draw_enhanced_actor(self, ax: plt.Axes, x: float, y: float) -> None:
        """Draw enhanced stick figure exactly like reference"""
        style = self.reference_styles[ComponentType.ACTOR]
        
        head_size = style["head_size"]
        body_height = style["body_height"]
        arm_width = style["arm_width"]
        leg_width = style["leg_width"]
        color = style["color"]
        linewidth = style["linewidth"]
        
        # Head (small circle)
        head = Circle((x, y + body_height * 0.7), head_size, 
                     facecolor='white', edgecolor=color, linewidth=linewidth)
        ax.add_patch(head)
        
        # Body (vertical line)
        body_top = y + body_height * 0.7 - head_size
        body_bottom = y - body_height * 0.3
        ax.plot([x, x], [body_top, body_bottom], color=color, linewidth=linewidth)
        
        # Arms (horizontal line at shoulder level)
        arm_y = y + body_height * 0.3
        ax.plot([x - arm_width/2, x + arm_width/2], [arm_y, arm_y], color=color, linewidth=linewidth)
        
        # Legs (two lines from hip to feet)
        hip_y = body_bottom
        foot_y = y - body_height
        ax.plot([x, x - leg_width/2], [hip_y, foot_y], color=color, linewidth=linewidth)
        ax.plot([x, x + leg_width/2], [hip_y, foot_y], color=color, linewidth=linewidth)

    def draw_enhanced_boundary(self, ax: plt.Axes, x: float, y: float) -> None:
        """Draw enhanced boundary symbol (circle with line)"""
        style = self.reference_styles[ComponentType.BOUNDARY]
        
        radius = style["radius"]
        line_length = style["line_length"]
        fill_color = style["fill_color"]
        border_color = style["border_color"]
        linewidth = style["linewidth"]
        
        # Main circle
        circle = Circle((x, y), radius, facecolor=fill_color, edgecolor=border_color, linewidth=linewidth)
        ax.add_patch(circle)
        
        # Vertical line through center (T-symbol)
        ax.plot([x, x], [y - line_length/2, y + line_length/2], 
               color=border_color, linewidth=linewidth + 0.5)

    def draw_enhanced_controller(self, ax: plt.Axes, x: float, y: float) -> None:
        """Draw enhanced controller symbol (large filled circle)"""
        style = self.reference_styles[ComponentType.CONTROLLER]
        
        radius = style["radius"]
        fill_color = style["fill_color"]
        border_color = style["border_color"]
        linewidth = style["linewidth"]
        
        # Large filled circle exactly like reference
        circle = Circle((x, y), radius, facecolor=fill_color, edgecolor=border_color, linewidth=linewidth)
        ax.add_patch(circle)

    def draw_enhanced_entity(self, ax: plt.Axes, x: float, y: float) -> None:
        """Draw enhanced entity symbol (rectangle)"""
        style = self.reference_styles[ComponentType.ENTITY]
        
        width = style["width"]
        height = style["height"]
        fill_color = style["fill_color"]
        border_color = style["border_color"]
        linewidth = style["linewidth"]
        
        # Rectangle
        rect = Rectangle((x - width/2, y - height/2), width, height, 
                        facecolor=fill_color, edgecolor=border_color, linewidth=linewidth)
        ax.add_patch(rect)

    def draw_enhanced_component(self, component: RAComponent, ax: plt.Axes) -> None:
        """Draw component with enhanced RUP styling"""
        x, y = component.position
        
        # Draw the symbol
        if component.component_type == ComponentType.ACTOR:
            self.draw_enhanced_actor(ax, x, y)
        elif component.component_type == ComponentType.BOUNDARY:
            self.draw_enhanced_boundary(ax, x, y)
        elif component.component_type == ComponentType.CONTROLLER:
            self.draw_enhanced_controller(ax, x, y)
        elif component.component_type == ComponentType.ENTITY:
            self.draw_enhanced_entity(ax, x, y)
        
        # Add label with clean styling
        style = self.reference_styles[component.component_type]
        text_x = x + style["text_offset"][0]
        text_y = y + style["text_offset"][1]
        
        # Clean label (remove angle brackets)
        clean_label = component.label.replace("«", "").replace("»", "")
        
        # Multi-line text wrapping for long labels
        if len(clean_label) > 15:
            words = clean_label.split()
            if len(words) > 1:
                mid = len(words) // 2
                line1 = " ".join(words[:mid])
                line2 = " ".join(words[mid:])
                clean_label = f"{line1}\n{line2}"
        
        ax.text(text_x, text_y, clean_label,
               ha='center', va='top',
               fontsize=style["fontsize"],
               fontweight='normal',
               bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor='white', 
                        alpha=0.9,
                        edgecolor='lightgray',
                        linewidth=0.5))

    def draw_reference_edge(self, edge: Dict, components_dict: Dict[str, RAComponent], ax: plt.Axes) -> None:
        """Draw edge exactly like reference (simple black lines)"""
        if edge["source"] not in components_dict or edge["target"] not in components_dict:
            return
        
        source_comp = components_dict[edge["source"]]
        target_comp = components_dict[edge["target"]]
        
        x1, y1 = source_comp.position
        x2, y2 = target_comp.position
        
        # Simple black line like reference
        style = self.reference_edge_style
        
        ax.plot([x1, x2], [y1, y2], 
               color=style["color"],
               linewidth=style["linewidth"],
               alpha=style["alpha"])
        
        # Add subtle relationship labels for important flows
        if edge.get("relationship") in ["use", "provide"] and edge.get("type") == "data_flow":
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Small offset to avoid line overlap
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                offset_x = -dy / length * 0.03
                offset_y = dx / length * 0.03
            else:
                offset_x, offset_y = 0.02, 0.02
            
            ax.text(mid_x + offset_x, mid_y + offset_y, f"«{edge['relationship']}»",
                   ha='center', va='center',
                   fontsize=7,
                   style='italic',
                   color='darkblue',
                   bbox=dict(boxstyle="round,pad=0.2", 
                           facecolor='lightyellow', 
                           alpha=0.8,
                           edgecolor='gray',
                           linewidth=0.3))

    def create_enhanced_rup_diagram(self, json_file_path: str, output_path: str = None) -> str:
        """Create enhanced RUP diagram that closely matches the reference"""
        
        # Load and parse data
        components, edges, json_data = self.load_and_parse_json(json_file_path)
        
        if not components:
            raise ValueError("No components found in JSON data")
        
        # Calculate reference-style layout
        self.calculate_reference_style_layout(components, edges)
        
        # Create figure with reference styling
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size, dpi=self.dpi)
        fig.patch.set_facecolor('white')
        
        # Set clean axes like reference
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Optional: very light grid like reference (barely visible)
        ax.grid(True, alpha=0.05, linestyle='-', linewidth=0.3, color='gray')
        
        # Create components dictionary
        components_dict = {comp.id: comp for comp in components}
        
        # Draw edges first (behind components)
        for edge in edges:
            self.draw_reference_edge(edge, components_dict, ax)
        
        # Draw components on top
        for component in components:
            self.draw_enhanced_component(component, ax)
        
        # Add title in corner like reference
        metadata = json_data.get("metadata", {})
        uc_name = metadata.get("uc_name", "object RA")
        ax.text(0.02, 0.98, uc_name, transform=ax.transAxes,
               ha='left', va='top', fontsize=11, fontweight='normal')
        
        # Generate output path
        if output_path is None:
            json_path = Path(json_file_path)
            metadata = json_data.get("metadata", {})
            uc_name = metadata.get("uc_name", "Unknown_UC")
            timestamp = metadata.get("analysis_timestamp", "")
            
            output_path = str(json_path.parent / f"{uc_name}_Enhanced_RUP_RA_{timestamp}.png")
        
        # Save with high quality
        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path


def main():
    """Demo of enhanced RUP-compliant engine"""
    print("Enhanced RUP/UML Compliant RA Diagram Engine")
    print("Matching reference diagram style exactly")
    print("=" * 60)
    
    # Create enhanced engine
    engine = EnhancedRUPEngine(figure_size=(20, 14))
    
    # Process visualization files
    output_dir = Path("output")
    viz_files = list(output_dir.glob("*_visualization_*.json"))
    
    if not viz_files:
        print("No visualization JSON files found")
        return
    
    print(f"Found {len(viz_files)} visualization JSON files")
    
    for json_file in viz_files:
        print(f"\nProcessing: {json_file.name}")
        
        try:
            diagram_output = engine.create_enhanced_rup_diagram(str(json_file))
            print(f"  Enhanced RUP diagram: {Path(diagram_output).name}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nEnhanced RUP-compliant diagrams completed!")


if __name__ == "__main__":
    main()