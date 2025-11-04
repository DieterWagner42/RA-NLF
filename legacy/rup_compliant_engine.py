"""
RUP/UML Compliant RA Diagram Engine
Generates Robustness Analysis diagrams according to official RUP/UML standards
Based on the reference diagram: RA diagrams/RA example.png
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path
import math
from dataclasses import dataclass, field
from enum import Enum
import textwrap

class ComponentType(Enum):
    ACTOR = "actor"
    BOUNDARY = "boundary" 
    CONTROLLER = "controller"
    ENTITY = "entity"

class EdgeType(Enum):
    DATA_FLOW = "data_flow"
    CONTROL_FLOW = "control_flow"

@dataclass
class RAComponent:
    """RUP/UML compliant RA component"""
    id: str
    label: str
    component_type: ComponentType
    stereotype: str
    position: Tuple[float, float] = (0, 0)
    warnings: List[str] = field(default_factory=list)
    
class RUPCompliantEngine:
    """
    RUP/UML compliant RA diagram engine matching the reference example
    """
    
    def __init__(self, figure_size: Tuple[int, int] = (16, 12)):
        self.figure_size = figure_size
        self.dpi = 300
        
        # RUP/UML compliant styling based on reference diagram
        self.rup_styles = {
            # Actors: Small stick figures (human icons)
            ComponentType.ACTOR: {
                "symbol": "stickman",
                "size": 0.03,
                "color": "#000000",  # Black
                "text_offset": (0, -0.05),
                "fontsize": 9,
                "fontweight": "normal"
            },
            
            # Boundaries: Circles with vertical line (T-shaped)
            ComponentType.BOUNDARY: {
                "symbol": "boundary_circle",
                "size": 0.04,
                "color": "#87CEEB",  # Light blue like reference
                "border_color": "#4682B4",  # Steel blue border
                "text_offset": (0, -0.06),
                "fontsize": 9,
                "fontweight": "normal"
            },
            
            # Controllers: Large filled circles 
            ComponentType.CONTROLLER: {
                "symbol": "large_circle",
                "size": 0.06,  # Larger than boundaries
                "color": "#ADD8E6",  # Light blue like reference
                "border_color": "#4682B4",  # Steel blue border
                "text_offset": (0, -0.08),
                "fontsize": 9,
                "fontweight": "normal"
            },
            
            # Entities: Rectangles (not visible in reference but standard)
            ComponentType.ENTITY: {
                "symbol": "rectangle",
                "size": 0.05,
                "color": "#F0F8FF",  # Alice blue
                "border_color": "#4682B4",  # Steel blue border
                "text_offset": (0, -0.07),
                "fontsize": 9,
                "fontweight": "normal"
            }
        }
        
        # Edge styling matching reference
        self.edge_styles = {
            "default": {
                "color": "#000000",
                "linewidth": 1.0,
                "alpha": 1.0,
                "style": "-"
            },
            "use_relationship": {
                "color": "#000000", 
                "linewidth": 1.0,
                "alpha": 1.0,
                "style": "-",
                "arrow": True
            }
        }

    def load_and_parse_json(self, json_file_path: str) -> Tuple[List[RAComponent], List[Dict], Dict[str, Any]]:
        """Load and parse JSON for RUP-compliant visualization"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        components = self.parse_components_rup(json_data)
        edges = self.parse_edges_rup(json_data)
        
        return components, edges, json_data

    def parse_components_rup(self, json_data: Dict[str, Any]) -> List[RAComponent]:
        """Parse components for RUP compliance"""
        components = []
        
        # Get components from JSON
        nodes = []
        if "graph" in json_data and "nodes" in json_data["graph"]:
            nodes = json_data["graph"]["nodes"]
        elif "components" in json_data and "nodes" in json_data["components"]:
            nodes = json_data["components"]["nodes"]
        
        for node in nodes:
            component = RAComponent(
                id=node["id"],
                label=node["label"],
                component_type=ComponentType(node["type"]),
                stereotype=node["stereotype"]
            )
            components.append(component)
        
        return components

    def parse_edges_rup(self, json_data: Dict[str, Any]) -> List[Dict]:
        """Parse edges for RUP-compliant styling"""
        edges = []
        
        # Get edges from JSON
        edge_list = []
        if "graph" in json_data and "edges" in json_data["graph"]:
            edge_list = json_data["graph"]["edges"]
        elif "components" in json_data and "edges" in json_data["components"]:
            edge_list = json_data["components"]["edges"]
        
        for edge in edge_list:
            edges.append({
                "source": edge["source"],
                "target": edge["target"],
                "type": edge.get("type", "data_flow"),
                "label": edge.get("label", ""),
                "relationship": edge.get("relationship", "")
            })
        
        return edges

    def calculate_organic_layout(self, components: List[RAComponent], edges: List[Dict]) -> None:
        """
        Calculate organic layout similar to reference diagram
        More free-form positioning like the reference
        """
        
        # Group components by type
        actors = [c for c in components if c.component_type == ComponentType.ACTOR]
        boundaries = [c for c in components if c.component_type == ComponentType.BOUNDARY]
        controllers = [c for c in components if c.component_type == ComponentType.CONTROLLER]
        entities = [c for c in components if c.component_type == ComponentType.ENTITY]
        
        # Create adjacency information for organic positioning
        adjacency = {comp.id: [] for comp in components}
        for edge in edges:
            if edge["source"] in adjacency and edge["target"] in adjacency:
                adjacency[edge["source"]].append(edge["target"])
                adjacency[edge["target"]].append(edge["source"])
        
        # Position actors on the perimeter (like reference)
        self.position_actors_on_perimeter(actors)
        
        # Position controllers in clusters (main components)
        self.position_controllers_organically(controllers, adjacency)
        
        # Position boundaries near controllers they interact with
        self.position_boundaries_near_controllers(boundaries, controllers, adjacency)
        
        # Position entities near controllers
        self.position_entities_near_controllers(entities, controllers, adjacency)

    def position_actors_on_perimeter(self, actors: List[RAComponent]) -> None:
        """Position actors around the perimeter like in reference"""
        if not actors:
            return
        
        # Predefined positions around perimeter (like reference)
        perimeter_positions = [
            (0.1, 0.3),   # Left side
            (0.9, 0.4),   # Right side  
            (0.9, 0.8),   # Top right
            (0.1, 0.6),   # Left upper
            (0.5, 0.9),   # Top center
            (0.1, 0.1),   # Bottom left
            (0.9, 0.1),   # Bottom right
        ]
        
        for i, actor in enumerate(actors):
            if i < len(perimeter_positions):
                actor.position = perimeter_positions[i]
            else:
                # Additional actors in a circle
                angle = 2 * math.pi * i / len(actors)
                x = 0.5 + 0.4 * math.cos(angle)
                y = 0.5 + 0.4 * math.sin(angle)
                actor.position = (max(0.1, min(0.9, x)), max(0.1, min(0.9, y)))

    def position_controllers_organically(self, controllers: List[RAComponent], adjacency: Dict[str, List[str]]) -> None:
        """Position controllers in organic clusters like reference"""
        if not controllers:
            return
        
        # Use force-directed approach for organic positioning
        positions = {}
        
        # Initialize with rough grid
        grid_size = math.ceil(math.sqrt(len(controllers)))
        for i, controller in enumerate(controllers):
            row = i // grid_size
            col = i % grid_size
            x = 0.2 + (col * 0.6) / max(1, grid_size - 1)
            y = 0.2 + (row * 0.6) / max(1, grid_size - 1)
            positions[controller.id] = [x, y]
        
        # Iterative improvement for organic layout
        for iteration in range(50):
            forces = {comp_id: [0, 0] for comp_id in positions}
            
            # Repulsion between all controllers
            for comp1_id, pos1 in positions.items():
                for comp2_id, pos2 in positions.items():
                    if comp1_id != comp2_id:
                        dx = pos1[0] - pos2[0]
                        dy = pos1[1] - pos2[1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist > 0.01:
                            repulsion = 0.001 / (dist * dist)
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
                                attraction = 0.0005 * dist
                                forces[comp_id][0] += attraction * dx / dist
                                forces[comp_id][1] += attraction * dy / dist
            
            # Apply forces
            for comp_id, force in forces.items():
                positions[comp_id][0] += force[0]
                positions[comp_id][1] += force[1]
                # Keep within bounds
                positions[comp_id][0] = max(0.15, min(0.85, positions[comp_id][0]))
                positions[comp_id][1] = max(0.15, min(0.85, positions[comp_id][1]))
        
        # Assign final positions
        for controller in controllers:
            if controller.id in positions:
                controller.position = tuple(positions[controller.id])

    def position_boundaries_near_controllers(self, boundaries: List[RAComponent], 
                                           controllers: List[RAComponent], 
                                           adjacency: Dict[str, List[str]]) -> None:
        """Position boundaries near their connected controllers"""
        for boundary in boundaries:
            # Find connected controllers
            connected_controllers = []
            for neighbor_id in adjacency.get(boundary.id, []):
                controller = next((c for c in controllers if c.id == neighbor_id), None)
                if controller:
                    connected_controllers.append(controller)
            
            if connected_controllers:
                # Position near average of connected controllers
                avg_x = sum(c.position[0] for c in connected_controllers) / len(connected_controllers)
                avg_y = sum(c.position[1] for c in connected_controllers) / len(connected_controllers)
                
                # Add small offset
                offset_x = 0.08 * (len(boundaries) % 2 - 0.5)
                offset_y = 0.08 * ((len(boundaries) // 2) % 2 - 0.5)
                
                boundary.position = (
                    max(0.1, min(0.9, avg_x + offset_x)),
                    max(0.1, min(0.9, avg_y + offset_y))
                )
            else:
                # Default position if no connections
                boundary.position = (0.5, 0.1)

    def position_entities_near_controllers(self, entities: List[RAComponent], 
                                         controllers: List[RAComponent], 
                                         adjacency: Dict[str, List[str]]) -> None:
        """Position entities near their connected controllers"""
        for i, entity in enumerate(entities):
            # Find connected controllers
            connected_controllers = []
            for neighbor_id in adjacency.get(entity.id, []):
                controller = next((c for c in controllers if c.id == neighbor_id), None)
                if controller:
                    connected_controllers.append(controller)
            
            if connected_controllers:
                # Position near first connected controller with offset
                controller = connected_controllers[0]
                offset_angle = (i * 2 * math.pi) / max(1, len(entities))
                offset_x = 0.1 * math.cos(offset_angle)
                offset_y = 0.1 * math.sin(offset_angle)
                
                entity.position = (
                    max(0.1, min(0.9, controller.position[0] + offset_x)),
                    max(0.1, min(0.9, controller.position[1] + offset_y))
                )
            else:
                # Default grid position
                entity.position = (0.8, 0.2 + (i * 0.1))

    def draw_actor_symbol(self, ax: plt.Axes, x: float, y: float, style: Dict[str, Any]) -> None:
        """Draw RUP-compliant actor symbol (stick figure)"""
        size = style["size"]
        color = style["color"]
        
        # Head (small circle)
        head = Circle((x, y + size), size/3, facecolor='white', edgecolor=color, linewidth=1.5)
        ax.add_patch(head)
        
        # Body (vertical line)
        ax.plot([x, x], [y + size - size/3, y - size/2], color=color, linewidth=2)
        
        # Arms (horizontal line)
        ax.plot([x - size/2, x + size/2], [y + size/4, y + size/4], color=color, linewidth=2)
        
        # Legs (two diagonal lines)
        ax.plot([x, x - size/2], [y - size/2, y - size], color=color, linewidth=2)
        ax.plot([x, x + size/2], [y - size/2, y - size], color=color, linewidth=2)

    def draw_boundary_symbol(self, ax: plt.Axes, x: float, y: float, style: Dict[str, Any]) -> None:
        """Draw RUP-compliant boundary symbol (circle with vertical line)"""
        size = style["size"]
        color = style["color"]
        border_color = style["border_color"]
        
        # Main circle
        circle = Circle((x, y), size, facecolor=color, edgecolor=border_color, linewidth=2)
        ax.add_patch(circle)
        
        # Vertical line through circle (T-shape)
        ax.plot([x, x], [y - size, y + size], color=border_color, linewidth=2)

    def draw_controller_symbol(self, ax: plt.Axes, x: float, y: float, style: Dict[str, Any]) -> None:
        """Draw RUP-compliant controller symbol (large filled circle)"""
        size = style["size"]
        color = style["color"]
        border_color = style["border_color"]
        
        # Large filled circle like in reference
        circle = Circle((x, y), size, facecolor=color, edgecolor=border_color, linewidth=2)
        ax.add_patch(circle)

    def draw_entity_symbol(self, ax: plt.Axes, x: float, y: float, style: Dict[str, Any]) -> None:
        """Draw RUP-compliant entity symbol (rectangle)"""
        size = style["size"]
        color = style["color"]
        border_color = style["border_color"]
        
        # Rectangle
        width = size * 1.5
        height = size
        rect = Rectangle((x - width/2, y - height/2), width, height, 
                        facecolor=color, edgecolor=border_color, linewidth=2)
        ax.add_patch(rect)

    def draw_rup_component(self, component: RAComponent, ax: plt.Axes) -> None:
        """Draw component using RUP/UML symbols"""
        x, y = component.position
        style = self.rup_styles[component.component_type]
        
        # Draw the appropriate symbol
        if component.component_type == ComponentType.ACTOR:
            self.draw_actor_symbol(ax, x, y, style)
        elif component.component_type == ComponentType.BOUNDARY:
            self.draw_boundary_symbol(ax, x, y, style)
        elif component.component_type == ComponentType.CONTROLLER:
            self.draw_controller_symbol(ax, x, y, style)
        elif component.component_type == ComponentType.ENTITY:
            self.draw_entity_symbol(ax, x, y, style)
        
        # Add component label
        text_x = x + style["text_offset"][0]
        text_y = y + style["text_offset"][1]
        
        # Clean label (remove stereotype brackets for cleaner look)
        clean_label = component.label.replace("«", "").replace("»", "")
        
        ax.text(text_x, text_y, clean_label,
               ha='center', va='top',
               fontsize=style["fontsize"],
               fontweight=style["fontweight"],
               bbox=dict(boxstyle="round,pad=0.2", 
                        facecolor='white', 
                        alpha=0.8,
                        edgecolor='none'))

    def draw_rup_edge(self, edge: Dict, components_dict: Dict[str, RAComponent], ax: plt.Axes) -> None:
        """Draw edge in RUP style (simple black lines like reference)"""
        if edge["source"] not in components_dict or edge["target"] not in components_dict:
            return
        
        source_comp = components_dict[edge["source"]]
        target_comp = components_dict[edge["target"]]
        
        x1, y1 = source_comp.position
        x2, y2 = target_comp.position
        
        # Simple black line like in reference
        style = self.edge_styles["default"]
        
        # Draw line
        ax.plot([x1, x2], [y1, y2], 
               color=style["color"],
               linewidth=style["linewidth"],
               alpha=style["alpha"],
               linestyle=style["style"])
        
        # Add use/provide labels if present
        if edge.get("relationship") in ["use", "provide"]:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Offset label slightly
            offset = 0.02
            label_x = mid_x + offset
            label_y = mid_y + offset
            
            ax.text(label_x, label_y, f"«{edge['relationship']}»",
                   ha='center', va='center',
                   fontsize=8,
                   style='italic',
                   bbox=dict(boxstyle="round,pad=0.2", 
                           facecolor='white', 
                           alpha=0.9,
                           edgecolor='gray',
                           linewidth=0.5))

    def create_rup_diagram(self, json_file_path: str, output_path: str = None) -> str:
        """Create RUP/UML compliant RA diagram"""
        
        # Load and parse data
        components, edges, json_data = self.load_and_parse_json(json_file_path)
        
        if not components:
            raise ValueError("No components found in JSON data")
        
        # Calculate organic layout
        self.calculate_organic_layout(components, edges)
        
        # Create figure with clean styling
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size, dpi=self.dpi)
        fig.patch.set_facecolor('white')
        
        # Set axis properties for clean diagram
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add light grid background like reference
        ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.5)
        
        # Create components dictionary
        components_dict = {comp.id: comp for comp in components}
        
        # Draw all edges first (behind components)
        for edge in edges:
            self.draw_rup_edge(edge, components_dict, ax)
        
        # Draw all components on top
        for component in components:
            self.draw_rup_component(component, ax)
        
        # Add title in top-left corner like reference
        metadata = json_data.get("metadata", {})
        uc_name = metadata.get("uc_name", "object RA")
        ax.text(0.02, 0.98, uc_name, transform=ax.transAxes,
               ha='left', va='top', fontsize=12, fontweight='bold')
        
        # Generate output path if not provided
        if output_path is None:
            json_path = Path(json_file_path)
            metadata = json_data.get("metadata", {})
            uc_name = metadata.get("uc_name", "Unknown_UC")
            timestamp = metadata.get("analysis_timestamp", "")
            
            output_path = str(json_path.parent / f"{uc_name}_RUP_Compliant_RA_{timestamp}.png")
        
        # Save with high quality
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none',
                   pad_inches=0.1)
        plt.close()
        
        return output_path


def main():
    """Demonstration of RUP-compliant RA diagram engine"""
    print("RUP/UML Compliant RA Diagram Engine")
    print("=" * 50)
    
    # Create engine instance
    engine = RUPCompliantEngine(figure_size=(16, 12))
    
    # Find visualization JSON files
    output_dir = Path("output")
    if not output_dir.exists():
        print("Error: Output directory not found")
        return
    
    viz_files = list(output_dir.glob("*_visualization_*.json"))
    
    if not viz_files:
        print("No visualization JSON files found")
        return
    
    print(f"Found {len(viz_files)} visualization JSON files")
    
    # Process each file with RUP compliance
    for json_file in viz_files:
        print(f"\nProcessing: {json_file.name}")
        
        try:
            diagram_output = engine.create_rup_diagram(str(json_file))
            print(f"  RUP-compliant diagram created: {Path(diagram_output).name}")
            
        except Exception as e:
            print(f"  Error processing {json_file.name}: {e}")
    
    print("\nRUP-compliant RA diagram generation completed!")


if __name__ == "__main__":
    main()