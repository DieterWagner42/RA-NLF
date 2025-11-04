"""
RA Diagram Visualization Engine
Generates UC-Methode compliant Robustness Analysis diagrams from JSON output
WITHOUT using Graphviz - uses matplotlib and networkx instead
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import math
from dataclasses import dataclass
from enum import Enum

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
    """Represents a single RA component for visualization"""
    id: str
    label: str
    component_type: ComponentType
    stereotype: str
    element_type: str
    position: Tuple[float, float] = (0, 0)
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

@dataclass
class RAEdge:
    """Represents a relationship between RA components"""
    source: str
    target: str
    edge_type: EdgeType
    label: str = ""
    style: str = "solid"
    color: str = "black"
    relationship: str = ""

class RADiagramEngine:
    """
    Main visualization engine for generating RA diagrams from JSON
    """
    
    def __init__(self, figure_size: Tuple[int, int] = (20, 14)):
        self.figure_size = figure_size
        self.dpi = 300
        
        # UC-Methode compliant styling
        self.styles = {
            "actor": {
                "shape": "ellipse",
                "facecolor": "#FFE4B5",  # Moccasin
                "edgecolor": "#DAA520",  # Goldenrod
                "linewidth": 2,
                "fontsize": 10,
                "fontweight": "bold"
            },
            "boundary": {
                "shape": "rectangle",
                "facecolor": "#E0E0E0",  # Light gray
                "edgecolor": "#808080",  # Gray
                "linewidth": 2,
                "fontsize": 9,
                "fontweight": "normal"
            },
            "controller": {
                "shape": "ellipse",
                "facecolor": "#98FB98",  # Pale green
                "edgecolor": "#228B22",  # Forest green
                "linewidth": 2,
                "fontsize": 10,
                "fontweight": "bold"
            },
            "entity": {
                "shape": "rectangle",
                "facecolor": "#FFA07A",  # Light salmon
                "edgecolor": "#FF6347",  # Tomato
                "linewidth": 2,
                "fontsize": 9,
                "fontweight": "normal"
            }
        }
        
        # Edge styles
        self.edge_styles = {
            "data_flow": {
                "color": "#0000FF",  # Blue
                "style": "solid",
                "width": 1.5,
                "alpha": 0.7
            },
            "control_flow": {
                "color": "#FF0000",  # Red
                "style": "dashed",
                "width": 1.0,
                "alpha": 0.6
            }
        }
        
        # Layout parameters
        self.layout_config = {
            "actor_x": 0.05,           # Actors on far left
            "boundary_x": 0.25,        # Boundaries left-center
            "controller_x": 0.55,      # Controllers center-right
            "entity_x": 0.85,          # Entities on far right
            "vertical_spacing": 0.08,   # Vertical spacing between components
            "horizontal_spacing": 0.15, # Horizontal spacing between columns
            "margin": 0.05,            # Diagram margins
            "component_width": 0.12,   # Component width
            "component_height": 0.06   # Component height
        }

    def load_json_data(self, json_file_path: str) -> Dict[str, Any]:
        """Load and parse JSON visualization data"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading JSON file {json_file_path}: {e}")

    def parse_components(self, json_data: Dict[str, Any]) -> List[RAComponent]:
        """Parse components from JSON data"""
        components = []
        
        if "graph" in json_data and "nodes" in json_data["graph"]:
            for node in json_data["graph"]["nodes"]:
                component = RAComponent(
                    id=node["id"],
                    label=node["label"],
                    component_type=ComponentType(node["type"]),
                    stereotype=node["stereotype"],
                    element_type=node.get("element_type", "functional")
                )
                components.append(component)
        
        elif "components" in json_data and "nodes" in json_data["components"]:
            # Alternative JSON structure
            for node in json_data["components"]["nodes"]:
                component = RAComponent(
                    id=node["id"],
                    label=node["label"],
                    component_type=ComponentType(node["type"]),
                    stereotype=node["stereotype"],
                    element_type=node.get("element_type", "functional")
                )
                components.append(component)
        
        return components

    def parse_edges(self, json_data: Dict[str, Any]) -> List[RAEdge]:
        """Parse edges/relationships from JSON data"""
        edges = []
        
        # From graph structure
        if "graph" in json_data and "edges" in json_data["graph"]:
            for edge in json_data["graph"]["edges"]:
                ra_edge = RAEdge(
                    source=edge["source"],
                    target=edge["target"],
                    edge_type=EdgeType(edge.get("type", "data_flow")),
                    label=edge.get("label", ""),
                    relationship=edge.get("relationship", "")
                )
                edges.append(ra_edge)
        
        # From components structure
        elif "components" in json_data and "edges" in json_data["components"]:
            for edge in json_data["components"]["edges"]:
                edge_type_str = edge.get("type", "data_flow")
                if edge_type_str not in [e.value for e in EdgeType]:
                    edge_type_str = "data_flow"  # Default fallback
                
                ra_edge = RAEdge(
                    source=edge["source"],
                    target=edge["target"],
                    edge_type=EdgeType(edge_type_str),
                    label=edge.get("label", ""),
                    relationship=edge.get("relationship", "")
                )
                edges.append(ra_edge)
        
        return edges

    def calculate_layout(self, components: List[RAComponent]) -> Dict[str, Tuple[float, float]]:
        """Calculate UC-Methode compliant layout positions"""
        layout = {}
        
        # Group components by type
        grouped = {
            ComponentType.ACTOR: [],
            ComponentType.BOUNDARY: [],
            ComponentType.CONTROLLER: [],
            ComponentType.ENTITY: []
        }
        
        for comp in components:
            grouped[comp.component_type].append(comp)
        
        # Calculate positions for each group
        for comp_type, comp_list in grouped.items():
            if not comp_list:
                continue
            
            # Determine x position based on UC-Methode layout rules
            if comp_type == ComponentType.ACTOR:
                x_pos = self.layout_config["actor_x"]
            elif comp_type == ComponentType.BOUNDARY:
                x_pos = self.layout_config["boundary_x"]
            elif comp_type == ComponentType.CONTROLLER:
                x_pos = self.layout_config["controller_x"]
            else:  # ENTITY
                x_pos = self.layout_config["entity_x"]
            
            # Calculate y positions with even spacing
            count = len(comp_list)
            if count == 1:
                y_positions = [0.5]
            else:
                start_y = self.layout_config["margin"]
                end_y = 1.0 - self.layout_config["margin"]
                y_positions = np.linspace(start_y, end_y, count)
            
            # Assign positions
            for i, comp in enumerate(comp_list):
                layout[comp.id] = (x_pos, y_positions[i])
                comp.position = (x_pos, y_positions[i])
        
        return layout

    def optimize_layout(self, components: List[RAComponent], edges: List[RAEdge]) -> None:
        """Optimize layout to reduce edge crossings and improve readability"""
        
        # Group components by type for optimization
        type_groups = {}
        for comp in components:
            if comp.component_type not in type_groups:
                type_groups[comp.component_type] = []
            type_groups[comp.component_type].append(comp)
        
        # Create adjacency information for ordering optimization
        for comp_type, comp_list in type_groups.items():
            if len(comp_list) <= 1:
                continue
            
            # Count connections for each component
            connections = {}
            for comp in comp_list:
                connections[comp.id] = {
                    'in': [],  # Incoming edges
                    'out': []  # Outgoing edges
                }
            
            for edge in edges:
                if edge.source in connections:
                    connections[edge.source]['out'].append(edge.target)
                if edge.target in connections:
                    connections[edge.target]['in'].append(edge.source)
            
            # Sort components by connection patterns to minimize crossings
            def connection_score(comp):
                comp_connections = connections.get(comp.id, {'in': [], 'out': []})
                # Calculate average y-position of connected components
                connected_y_positions = []
                
                for other_comp in components:
                    if (other_comp.id in comp_connections['in'] or 
                        other_comp.id in comp_connections['out']):
                        connected_y_positions.append(other_comp.position[1])
                
                return np.mean(connected_y_positions) if connected_y_positions else comp.position[1]
            
            # Re-sort components within each type group
            comp_list.sort(key=connection_score)
            
            # Reassign y-positions
            count = len(comp_list)
            if count > 1:
                start_y = self.layout_config["margin"]
                end_y = 1.0 - self.layout_config["margin"]
                y_positions = np.linspace(start_y, end_y, count)
                
                for i, comp in enumerate(comp_list):
                    comp.position = (comp.position[0], y_positions[i])

    def create_component_patch(self, component: RAComponent, ax: plt.Axes) -> patches.Patch:
        """Create matplotlib patch for a component based on UC-Methode styling"""
        x, y = component.position
        style = self.styles[component.component_type.value]
        
        width = self.layout_config["component_width"]
        height = self.layout_config["component_height"]
        
        # Adjust position to center the component
        x_centered = x - width/2
        y_centered = y - height/2
        
        # Create shape based on component type
        if style["shape"] == "ellipse":
            # Actors and Controllers use ellipses
            patch = patches.Ellipse(
                (x, y), width, height,
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                linewidth=style["linewidth"]
            )
        else:
            # Boundaries and Entities use rectangles
            patch = FancyBboxPatch(
                (x_centered, y_centered), width, height,
                boxstyle="round,pad=0.005",
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                linewidth=style["linewidth"]
            )
        
        # Handle special styling (warnings, critical components)
        if component.warnings:
            patch.set_edgecolor("#FF0000")  # Red border for warnings
            patch.set_linewidth(3)
        
        ax.add_patch(patch)
        
        # Add stereotype and label text
        # Stereotype (smaller, above)
        ax.text(x, y + height/3, component.stereotype, 
                ha='center', va='center',
                fontsize=style["fontsize"] - 2,
                style='italic')
        
        # Component name (main text)
        ax.text(x, y, component.label, 
                ha='center', va='center',
                fontsize=style["fontsize"],
                fontweight=style["fontweight"],
                wrap=True)
        
        return patch

    def draw_edge(self, edge: RAEdge, components_dict: Dict[str, RAComponent], ax: plt.Axes) -> None:
        """Draw an edge between two components"""
        if edge.source not in components_dict or edge.target not in components_dict:
            return  # Skip invalid edges
        
        source_comp = components_dict[edge.source]
        target_comp = components_dict[edge.target]
        
        x1, y1 = source_comp.position
        x2, y2 = target_comp.position
        
        # Get edge style
        style = self.edge_styles.get(edge.edge_type.value, self.edge_styles["data_flow"])
        
        # Calculate connection points (edge of components, not center)
        # Simple approach: connect centers, but could be enhanced for edge-to-edge
        
        # Determine line style
        linestyle = '--' if style["style"] == "dashed" else '-'
        
        # Draw arrow
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(
                       arrowstyle='->',
                       color=style["color"],
                       linestyle=linestyle,
                       linewidth=style["width"],
                       alpha=style["alpha"]
                   ))
        
        # Add edge label if present
        if edge.label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Offset label slightly to avoid overlap with line
            offset_y = 0.02 if y1 == y2 else 0
            
            ax.text(mid_x, mid_y + offset_y, edge.label,
                   ha='center', va='bottom',
                   fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor='white', 
                           alpha=0.8,
                           edgecolor='none'))

    def add_legend(self, ax: plt.Axes) -> None:
        """Add UC-Methode compliant legend to the diagram"""
        legend_elements = []
        
        # Component type legend
        for comp_type, style in self.styles.items():
            if style["shape"] == "ellipse":
                element = plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=style["facecolor"],
                                   markeredgecolor=style["edgecolor"],
                                   markersize=10, linewidth=0,
                                   label=f'{comp_type.title()} «{comp_type}»')
            else:
                element = plt.Line2D([0], [0], marker='s', color='w',
                                   markerfacecolor=style["facecolor"],
                                   markeredgecolor=style["edgecolor"],
                                   markersize=10, linewidth=0,
                                   label=f'{comp_type.title()} «{comp_type}»')
            legend_elements.append(element)
        
        # Edge type legend
        for edge_type, style in self.edge_styles.items():
            linestyle = '--' if style["style"] == "dashed" else '-'
            element = plt.Line2D([0], [0], color=style["color"],
                               linestyle=linestyle,
                               linewidth=style["width"],
                               label=edge_type.replace('_', ' ').title())
            legend_elements.append(element)
        
        # Place legend outside plot area
        ax.legend(handles=legend_elements, 
                 loc='center left', 
                 bbox_to_anchor=(1, 0.5),
                 fontsize=9)

    def add_title_and_metadata(self, ax: plt.Axes, json_data: Dict[str, Any]) -> None:
        """Add title and metadata information to the diagram"""
        metadata = json_data.get("metadata", {})
        
        # Main title
        uc_name = metadata.get("uc_name", "Unknown UC")
        domain = metadata.get("domain", "Unknown Domain")
        title = f"Robustness Analysis Diagram: {uc_name}"
        if domain != "Unknown Domain":
            title += f" ({domain.replace('_', ' ').title()})"
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Subtitle with framework info
        framework = metadata.get("framework_version", "UC-Methode RA-NLF")
        timestamp = metadata.get("analysis_timestamp", "")
        subtitle = f"{framework}"
        if timestamp:
            subtitle += f" - Generated: {timestamp[:8]}"
        
        ax.text(0.5, 0.95, subtitle, transform=ax.transAxes,
               ha='center', va='top', fontsize=10, style='italic')

    def create_diagram(self, json_file_path: str, output_path: str = None) -> str:
        """
        Main method to create RA diagram from JSON
        
        Args:
            json_file_path: Path to JSON visualization file
            output_path: Output path for diagram (if None, auto-generated)
            
        Returns:
            Path to generated diagram file
        """
        # Load and parse data
        json_data = self.load_json_data(json_file_path)
        components = self.parse_components(json_data)
        edges = self.parse_edges(json_data)
        
        if not components:
            raise ValueError("No components found in JSON data")
        
        # Calculate layout
        self.calculate_layout(components)
        self.optimize_layout(components, edges)
        
        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size, dpi=self.dpi)
        
        # Set axis properties for diagram
        ax.set_xlim(-0.05, 1.2)  # Extra space for legend
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.axis('off')  # Hide axes for clean diagram
        
        # Create components dictionary for edge drawing
        components_dict = {comp.id: comp for comp in components}
        
        # Draw all components
        for component in components:
            self.create_component_patch(component, ax)
        
        # Draw all edges
        for edge in edges:
            self.draw_edge(edge, components_dict, ax)
        
        # Add title and metadata
        self.add_title_and_metadata(ax, json_data)
        
        # Add legend
        self.add_legend(ax)
        
        # Generate output path if not provided
        if output_path is None:
            json_path = Path(json_file_path)
            metadata = json_data.get("metadata", {})
            uc_name = metadata.get("uc_name", "Unknown_UC")
            timestamp = metadata.get("analysis_timestamp", "")
            
            output_path = str(json_path.parent / f"{uc_name}_RA_Diagram_Engine_{timestamp}.png")
        
        # Save diagram
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path

    def create_svg_diagram(self, json_file_path: str, output_path: str = None) -> str:
        """Create SVG version of the diagram for scalable output"""
        if output_path and not output_path.endswith('.svg'):
            output_path = output_path.replace('.png', '.svg')
        
        # Create PNG version first to get layout
        png_path = self.create_diagram(json_file_path, output_path)
        
        # Create SVG version
        if output_path:
            svg_path = output_path.replace('.png', '.svg') if output_path.endswith('.png') else output_path
        else:
            svg_path = png_path.replace('.png', '.svg')
        
        # Load and parse data again for SVG
        json_data = self.load_json_data(json_file_path)
        components = self.parse_components(json_data)
        edges = self.parse_edges(json_data)
        
        # Calculate layout
        self.calculate_layout(components)
        self.optimize_layout(components, edges)
        
        # Create SVG figure
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size)
        ax.set_xlim(-0.05, 1.2)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Create components dictionary
        components_dict = {comp.id: comp for comp in components}
        
        # Draw all components and edges
        for component in components:
            self.create_component_patch(component, ax)
        
        for edge in edges:
            self.draw_edge(edge, components_dict, ax)
        
        # Add title and legend
        self.add_title_and_metadata(ax, json_data)
        self.add_legend(ax)
        
        # Save as SVG
        plt.tight_layout()
        plt.savefig(svg_path, format='svg', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return svg_path


def main():
    """Demonstration of the RA diagram engine"""
    print("RA Diagram Visualization Engine")
    print("=" * 50)
    
    # Create engine instance
    engine = RADiagramEngine(figure_size=(24, 16))
    
    # Check for available JSON files
    output_dir = Path("output")
    if not output_dir.exists():
        print("Error: Output directory not found")
        return
    
    # Find visualization JSON files
    viz_files = list(output_dir.glob("*_visualization_*.json"))
    
    if not viz_files:
        print("No visualization JSON files found in output directory")
        return
    
    print(f"Found {len(viz_files)} visualization JSON files")
    
    # Process each file
    for json_file in viz_files:
        print(f"\nProcessing: {json_file.name}")
        
        try:
            # Create PNG diagram
            png_output = engine.create_diagram(str(json_file))
            print(f"  PNG diagram created: {png_output}")
            
            # Create SVG diagram
            svg_output = engine.create_svg_diagram(str(json_file))
            print(f"  SVG diagram created: {svg_output}")
            
        except Exception as e:
            print(f"  Error processing {json_file.name}: {e}")
    
    print("\nRA diagram generation completed!")


if __name__ == "__main__":
    main()