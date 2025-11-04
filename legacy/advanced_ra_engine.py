"""
Advanced RA Diagram Visualization Engine
Enhanced version with advanced UC-Methode features and styling
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

class ElementType(Enum):
    FUNCTIONAL = "functional"
    CONTROL = "control"
    CONTAINER = "container"
    IMPLEMENTATION = "implementation"

@dataclass
class RAComponent:
    """Enhanced RA component with advanced styling options"""
    id: str
    label: str
    component_type: ComponentType
    stereotype: str
    element_type: ElementType
    position: Tuple[float, float] = (0, 0)
    warnings: List[str] = field(default_factory=list)
    step_references: List[str] = field(default_factory=list)
    is_shared: bool = False
    is_critical: bool = False
    
@dataclass
class RAEdge:
    """Enhanced edge with UC-Methode rule information"""
    source: str
    target: str
    edge_type: EdgeType
    label: str = ""
    style: str = "solid"
    color: str = "black"
    relationship: str = ""
    flow_rule: Optional[int] = None  # UC-Methode rule number
    step_id: str = ""

class AdvancedRAEngine:
    """
    Advanced RA diagram engine with UC-Methode compliance and enhanced features
    """
    
    def __init__(self, figure_size: Tuple[int, int] = (24, 18)):
        self.figure_size = figure_size
        self.dpi = 300
        
        # Enhanced UC-Methode styling with element type variations
        self.component_styles = {
            ComponentType.ACTOR: {
                ElementType.FUNCTIONAL: {
                    "shape": "ellipse", "facecolor": "#FFE4B5", "edgecolor": "#DAA520",
                    "linewidth": 2, "fontsize": 10, "fontweight": "bold"
                },
                ElementType.CONTROL: {
                    "shape": "ellipse", "facecolor": "#F0E68C", "edgecolor": "#B8860B", 
                    "linewidth": 2, "fontsize": 10, "fontweight": "bold"
                }
            },
            ComponentType.BOUNDARY: {
                ElementType.FUNCTIONAL: {
                    "shape": "rectangle", "facecolor": "#E0E0E0", "edgecolor": "#808080",
                    "linewidth": 2, "fontsize": 9, "fontweight": "normal"
                },
                ElementType.CONTROL: {
                    "shape": "rectangle", "facecolor": "#D3D3D3", "edgecolor": "#696969",
                    "linewidth": 2, "fontsize": 9, "fontweight": "normal"
                }
            },
            ComponentType.CONTROLLER: {
                ElementType.FUNCTIONAL: {
                    "shape": "ellipse", "facecolor": "#98FB98", "edgecolor": "#228B22",
                    "linewidth": 2, "fontsize": 10, "fontweight": "bold"
                },
                ElementType.CONTROL: {
                    "shape": "ellipse", "facecolor": "#90EE90", "edgecolor": "#006400",
                    "linewidth": 2, "fontsize": 10, "fontweight": "bold"
                }
            },
            ComponentType.ENTITY: {
                ElementType.FUNCTIONAL: {
                    "shape": "rectangle", "facecolor": "#FFA07A", "edgecolor": "#FF6347",
                    "linewidth": 2, "fontsize": 9, "fontweight": "normal"
                },
                ElementType.CONTROL: {
                    "shape": "rectangle", "facecolor": "#FFB6C1", "edgecolor": "#DC143C",
                    "linewidth": 2, "fontsize": 9, "fontweight": "normal"
                },
                ElementType.CONTAINER: {
                    "shape": "rectangle", "facecolor": "#DDA0DD", "edgecolor": "#9370DB",
                    "linewidth": 2, "fontsize": 9, "fontweight": "normal"
                },
                ElementType.IMPLEMENTATION: {
                    "shape": "rectangle", "facecolor": "#FFCCCB", "edgecolor": "#8B0000",
                    "linewidth": 3, "fontsize": 9, "fontweight": "bold"  # Warning style
                }
            }
        }
        
        # Enhanced edge styles with UC-Methode rule colors
        self.edge_styles = {
            EdgeType.DATA_FLOW: {
                "use": {"color": "#0000FF", "style": "solid", "width": 1.5, "alpha": 0.7},
                "provide": {"color": "#4169E1", "style": "solid", "width": 1.5, "alpha": 0.7}
            },
            EdgeType.CONTROL_FLOW: {
                1: {"color": "#FF0000", "style": "solid", "width": 2.0, "alpha": 0.8},    # Rule 1: Boundary->Controller
                2: {"color": "#FF4500", "style": "dashed", "width": 1.8, "alpha": 0.7},  # Rule 2: Controller->Controller  
                3: {"color": "#FF6347", "style": "dotted", "width": 1.5, "alpha": 0.6},  # Rule 3: Sequential
                4: {"color": "#FF8C00", "style": "dashdot", "width": 2.0, "alpha": 0.8}, # Rule 4: Parallel
                5: {"color": "#FF1493", "style": "solid", "width": 2.0, "alpha": 0.8}    # Rule 5: Controller->Boundary
            }
        }
        
        # Layout configuration for UC-Methode compliance
        self.layout_config = {
            "actor_x": 0.08,
            "boundary_x": 0.28,
            "controller_x": 0.58,
            "entity_x": 0.88,
            "vertical_spacing": 0.06,
            "horizontal_spacing": 0.15,
            "margin": 0.04,
            "component_width": 0.14,
            "component_height": 0.05,
            "min_vertical_separation": 0.08
        }

    def load_and_parse_json(self, json_file_path: str) -> Tuple[List[RAComponent], List[RAEdge], Dict[str, Any]]:
        """Load JSON and parse into components and edges"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        components = self.parse_components(json_data)
        edges = self.parse_edges(json_data)
        
        return components, edges, json_data

    def parse_components(self, json_data: Dict[str, Any]) -> List[RAComponent]:
        """Parse components with enhanced attributes"""
        components = []
        
        # Get components from different possible JSON structures
        nodes = []
        if "graph" in json_data and "nodes" in json_data["graph"]:
            nodes = json_data["graph"]["nodes"]
        elif "components" in json_data and "nodes" in json_data["components"]:
            nodes = json_data["components"]["nodes"]
        
        # Get styling and warning information
        styling = json_data.get("styling", {})
        warnings_info = styling.get("warnings", [])
        special_styling = styling.get("special_styling", {})
        
        for node in nodes:
            # Determine element type
            element_type_str = node.get("element_type", "functional")
            try:
                element_type = ElementType(element_type_str)
            except ValueError:
                element_type = ElementType.FUNCTIONAL
            
            # Check for warnings and special attributes
            node_warnings = []
            is_critical = False
            is_shared = False
            
            for warning_info in warnings_info:
                if warning_info.get("component") == node["id"]:
                    node_warnings.extend(warning_info.get("warnings", []))
            
            # Check for critical or shared status
            if node["id"] in special_styling:
                special = special_styling[node["id"]]
                if "border_color" in special and special["border_color"] == "#FF0000":
                    is_critical = True
                if "background_color" in special:
                    is_critical = True
            
            # Check if component is shared (appears in multiple UCs)
            if hasattr(node, 'uc_involvement') or 'shared_component' in node:
                is_shared = node.get('shared_component', False)
            
            component = RAComponent(
                id=node["id"],
                label=node["label"],
                component_type=ComponentType(node["type"]),
                stereotype=node["stereotype"],
                element_type=element_type,
                warnings=node_warnings,
                is_critical=is_critical,
                is_shared=is_shared
            )
            components.append(component)
        
        return components

    def parse_edges(self, json_data: Dict[str, Any]) -> List[RAEdge]:
        """Parse edges with UC-Methode rule information"""
        edges = []
        
        # Get edges from different possible JSON structures
        edge_list = []
        if "graph" in json_data and "edges" in json_data["graph"]:
            edge_list = json_data["graph"]["edges"]
        elif "components" in json_data and "edges" in json_data["components"]:
            edge_list = json_data["components"]["edges"]
        
        for edge in edge_list:
            # Determine edge type
            edge_type_str = edge.get("type", "data_flow")
            try:
                edge_type = EdgeType(edge_type_str)
            except ValueError:
                edge_type = EdgeType.DATA_FLOW
            
            # Extract UC-Methode rule information
            flow_rule = None
            if "flow_rule" in edge:
                flow_rule = edge["flow_rule"]
            elif "label" in edge and "Rule" in edge["label"]:
                # Extract rule number from label like "Rule 1", "Rule 2", etc.
                try:
                    rule_text = edge["label"].split("Rule")[1].strip()
                    flow_rule = int(rule_text.split()[0])
                except (ValueError, IndexError):
                    pass
            
            ra_edge = RAEdge(
                source=edge["source"],
                target=edge["target"],
                edge_type=edge_type,
                label=edge.get("label", ""),
                relationship=edge.get("relationship", ""),
                flow_rule=flow_rule,
                step_id=edge.get("step_id", "")
            )
            edges.append(ra_edge)
        
        return edges

    def calculate_advanced_layout(self, components: List[RAComponent], edges: List[RAEdge]) -> None:
        """Advanced layout algorithm with relationship-aware positioning"""
        
        # Group components by type
        grouped = {comp_type: [] for comp_type in ComponentType}
        for comp in components:
            grouped[comp.component_type].append(comp)
        
        # Calculate relationship weights for better positioning
        relationship_weights = self.calculate_relationship_weights(components, edges)
        
        # Position each group with relationship optimization
        for comp_type, comp_list in grouped.items():
            if not comp_list:
                continue
            
            # Determine base x position
            x_pos = self.get_base_x_position(comp_type)
            
            # Sort components within group by relationship weights
            comp_list.sort(key=lambda c: relationship_weights.get(c.id, 0))
            
            # Calculate y positions with proper spacing
            y_positions = self.calculate_y_positions(comp_list, relationship_weights)
            
            # Assign positions
            for comp, y_pos in zip(comp_list, y_positions):
                comp.position = (x_pos, y_pos)

    def calculate_relationship_weights(self, components: List[RAComponent], edges: List[RAEdge]) -> Dict[str, float]:
        """Calculate relationship weights for better component positioning"""
        weights = {comp.id: 0.0 for comp in components}
        
        # Create adjacency information
        connections = {comp.id: {'in': [], 'out': [], 'weights': []} for comp in components}
        
        for edge in edges:
            if edge.source in connections and edge.target in connections:
                connections[edge.source]['out'].append(edge.target)
                connections[edge.target]['in'].append(edge.source)
                
                # Weight based on edge type and UC-Methode rules
                weight = 1.0
                if edge.edge_type == EdgeType.CONTROL_FLOW:
                    weight = 2.0  # Control flows are more important for layout
                if edge.flow_rule:
                    weight *= (1.0 + edge.flow_rule * 0.1)  # Higher rules get slightly more weight
                
                connections[edge.source]['weights'].append(weight)
                connections[edge.target]['weights'].append(weight)
        
        # Calculate average connection position for each component
        component_positions = {comp.id: comp.position[1] if comp.position != (0, 0) else 0.5 
                             for comp in components}
        
        for comp_id, conn_info in connections.items():
            if conn_info['in'] or conn_info['out']:
                connected_positions = []
                for connected_id in conn_info['in'] + conn_info['out']:
                    connected_positions.append(component_positions[connected_id])
                
                if connected_positions:
                    weights[comp_id] = np.mean(connected_positions)
        
        return weights

    def get_base_x_position(self, comp_type: ComponentType) -> float:
        """Get base x position for component type"""
        return {
            ComponentType.ACTOR: self.layout_config["actor_x"],
            ComponentType.BOUNDARY: self.layout_config["boundary_x"],
            ComponentType.CONTROLLER: self.layout_config["controller_x"],
            ComponentType.ENTITY: self.layout_config["entity_x"]
        }[comp_type]

    def calculate_y_positions(self, components: List[RAComponent], weights: Dict[str, float]) -> List[float]:
        """Calculate optimized y positions for a group of components"""
        count = len(components)
        if count == 1:
            return [0.5]
        
        # Base y positions
        start_y = self.layout_config["margin"]
        end_y = 1.0 - self.layout_config["margin"]
        
        if count <= 2:
            return list(np.linspace(start_y, end_y, count))
        
        # More sophisticated spacing for larger groups
        positions = []
        
        # Sort by relationship weights for better flow
        sorted_components = sorted(components, key=lambda c: weights.get(c.id, 0.5))
        
        # Calculate positions with minimum separation
        min_sep = self.layout_config["min_vertical_separation"]
        available_space = end_y - start_y
        needed_space = (count - 1) * min_sep
        
        if needed_space > available_space:
            # Use minimum separation
            positions = [start_y + i * min_sep for i in range(count)]
        else:
            # Use optimized spacing
            positions = list(np.linspace(start_y, end_y, count))
        
        # Ensure we don't exceed bounds
        positions = [max(start_y, min(end_y, pos)) for pos in positions]
        
        return positions

    def get_component_style(self, component: RAComponent) -> Dict[str, Any]:
        """Get enhanced styling for component"""
        base_style = self.component_styles[component.component_type][component.element_type].copy()
        
        # Apply special styling for warnings, critical components, etc.
        if component.is_critical or component.warnings:
            base_style["edgecolor"] = "#FF0000"
            base_style["linewidth"] = 3
        
        if component.is_shared:
            base_style["linewidth"] += 1
            # Add slight color variation for shared components
            if "facecolor" in base_style:
                # Make shared components slightly darker
                color = base_style["facecolor"]
                base_style["facecolor"] = self.darken_color(color, 0.9)
        
        if component.element_type == ElementType.IMPLEMENTATION:
            # Special highlighting for implementation elements (warnings)
            base_style["facecolor"] = "#FFCCCB"
            base_style["edgecolor"] = "#8B0000"
            base_style["linewidth"] = 3
        
        return base_style

    def darken_color(self, color: str, factor: float) -> str:
        """Darken a hex color by a factor"""
        if color.startswith('#'):
            color = color[1:]
        
        r, g, b = [int(color[i:i+2], 16) for i in (0, 2, 4)]
        r, g, b = [int(c * factor) for c in (r, g, b)]
        r, g, b = [max(0, min(255, c)) for c in (r, g, b)]
        
        return f"#{r:02x}{g:02x}{b:02x}"

    def create_enhanced_component_patch(self, component: RAComponent, ax: plt.Axes) -> None:
        """Create enhanced component with UC-Methode styling"""
        x, y = component.position
        style = self.get_component_style(component)
        
        width = self.layout_config["component_width"]
        height = self.layout_config["component_height"]
        
        # Adjust position to center the component
        x_centered = x - width/2
        y_centered = y - height/2
        
        # Create shape based on component type and style
        if style["shape"] == "ellipse":
            patch = patches.Ellipse(
                (x, y), width, height,
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                linewidth=style["linewidth"]
            )
        else:
            # Enhanced rectangle with rounded corners
            patch = FancyBboxPatch(
                (x_centered, y_centered), width, height,
                boxstyle="round,pad=0.008",
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                linewidth=style["linewidth"]
            )
        
        ax.add_patch(patch)
        
        # Add stereotype with proper UC-Methode formatting
        stereotype_y = y + height/2.8
        ax.text(x, stereotype_y, component.stereotype, 
                ha='center', va='center',
                fontsize=style["fontsize"] - 2,
                style='italic',
                weight='normal')
        
        # Component name with intelligent text wrapping
        wrapped_label = self.wrap_text(component.label, 15)
        ax.text(x, y - height/6, wrapped_label, 
                ha='center', va='center',
                fontsize=style["fontsize"],
                fontweight=style["fontweight"])
        
        # Add warning indicators
        if component.warnings:
            self.add_warning_indicator(ax, x + width/2 - 0.01, y + height/2 - 0.01)
        
        # Add shared component indicator
        if component.is_shared:
            self.add_shared_indicator(ax, x - width/2 + 0.01, y + height/2 - 0.01)

    def wrap_text(self, text: str, max_width: int) -> str:
        """Intelligently wrap text for component labels"""
        if len(text) <= max_width:
            return text
        
        # Try to break at natural boundaries
        words = text.split()
        if len(words) > 1:
            # Find good break point
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + len(current_line) <= max_width:
                    current_line.append(word)
                    current_length += len(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            return '\n'.join(lines)
        else:
            # Single long word - use textwrap
            return '\n'.join(textwrap.wrap(text, max_width))

    def add_warning_indicator(self, ax: plt.Axes, x: float, y: float) -> None:
        """Add warning triangle indicator"""
        triangle = patches.Polygon([(x, y), (x-0.01, y-0.015), (x+0.01, y-0.015)],
                                 facecolor='#FF0000', edgecolor='#8B0000', linewidth=1)
        ax.add_patch(triangle)
        ax.text(x, y-0.008, '!', ha='center', va='center', 
               fontsize=8, color='white', fontweight='bold')

    def add_shared_indicator(self, ax: plt.Axes, x: float, y: float) -> None:
        """Add shared component indicator"""
        circle = patches.Circle((x, y), 0.008, facecolor='#0000FF', edgecolor='#000080', linewidth=1)
        ax.add_patch(circle)
        ax.text(x, y, 'S', ha='center', va='center', 
               fontsize=6, color='white', fontweight='bold')

    def draw_enhanced_edge(self, edge: RAEdge, components_dict: Dict[str, RAComponent], ax: plt.Axes) -> None:
        """Draw enhanced edge with UC-Methode rule styling"""
        if edge.source not in components_dict or edge.target not in components_dict:
            return
        
        source_comp = components_dict[edge.source]
        target_comp = components_dict[edge.target]
        
        # Get edge style based on type and rule
        if edge.edge_type == EdgeType.CONTROL_FLOW and edge.flow_rule:
            style = self.edge_styles[EdgeType.CONTROL_FLOW].get(
                edge.flow_rule, 
                self.edge_styles[EdgeType.CONTROL_FLOW][1]
            )
        elif edge.edge_type == EdgeType.DATA_FLOW:
            relationship = edge.relationship or "use"
            style = self.edge_styles[EdgeType.DATA_FLOW].get(
                relationship,
                self.edge_styles[EdgeType.DATA_FLOW]["use"]
            )
        else:
            # Default style
            style = {"color": "#000000", "style": "solid", "width": 1.0, "alpha": 0.7}
        
        # Calculate optimal connection points
        source_point, target_point = self.calculate_connection_points(source_comp, target_comp)
        
        # Convert style to matplotlib parameters
        linestyle = self.convert_line_style(style["style"])
        
        # Draw arrow with enhanced styling
        ax.annotate('', xy=target_point, xytext=source_point,
                   arrowprops=dict(
                       arrowstyle='->',
                       color=style["color"],
                       linestyle=linestyle,
                       linewidth=style["width"],
                       alpha=style["alpha"],
                       shrinkA=10, shrinkB=10
                   ))
        
        # Add enhanced edge label
        if edge.label:
            self.add_edge_label(ax, source_point, target_point, edge.label, style["color"])

    def calculate_connection_points(self, source: RAComponent, target: RAComponent) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate optimal connection points on component edges"""
        x1, y1 = source.position
        x2, y2 = target.position
        
        width = self.layout_config["component_width"]
        height = self.layout_config["component_height"]
        
        # Calculate connection points on component boundaries
        if x2 > x1:  # Target is to the right
            source_point = (x1 + width/2, y1)
            target_point = (x2 - width/2, y2)
        else:  # Target is to the left
            source_point = (x1 - width/2, y1)
            target_point = (x2 + width/2, y2)
        
        return source_point, target_point

    def convert_line_style(self, style_str: str) -> str:
        """Convert style string to matplotlib linestyle"""
        style_map = {
            "solid": "-",
            "dashed": "--",
            "dotted": ":",
            "dashdot": "-."
        }
        return style_map.get(style_str, "-")

    def add_edge_label(self, ax: plt.Axes, source_point: Tuple[float, float], 
                      target_point: Tuple[float, float], label: str, color: str) -> None:
        """Add enhanced edge label with better positioning"""
        mid_x = (source_point[0] + target_point[0]) / 2
        mid_y = (source_point[1] + target_point[1]) / 2
        
        # Calculate offset for label to avoid line overlap
        dx = target_point[0] - source_point[0]
        dy = target_point[1] - source_point[1]
        
        # Perpendicular offset
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            offset_x = -dy / length * 0.02
            offset_y = dx / length * 0.02
        else:
            offset_x, offset_y = 0, 0.02
        
        # Wrap label if too long
        wrapped_label = self.wrap_text(label, 20)
        
        ax.text(mid_x + offset_x, mid_y + offset_y, wrapped_label,
               ha='center', va='center',
               fontsize=8,
               bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor='white', 
                        alpha=0.9,
                        edgecolor=color,
                        linewidth=0.5))

    def add_enhanced_legend(self, ax: plt.Axes, components: List[RAComponent], edges: List[RAEdge]) -> None:
        """Add comprehensive UC-Methode legend"""
        legend_elements = []
        
        # Component types with element type variations
        component_types_seen = set()
        element_types_seen = set()
        
        for comp in components:
            comp_key = (comp.component_type, comp.element_type)
            if comp_key not in component_types_seen:
                component_types_seen.add(comp_key)
                element_types_seen.add(comp.element_type)
                
                style = self.get_component_style(comp)
                marker = 'o' if style["shape"] == "ellipse" else 's'
                
                element_suffix = f" ({comp.element_type.value})" if comp.element_type != ElementType.FUNCTIONAL else ""
                label = f'{comp.component_type.value.title()}{element_suffix}'
                
                element = plt.Line2D([0], [0], marker=marker, color='w',
                                   markerfacecolor=style["facecolor"],
                                   markeredgecolor=style["edgecolor"],
                                   markersize=10, linewidth=0,
                                   label=label)
                legend_elements.append(element)
        
        # Edge types and UC-Methode rules
        edge_rules_seen = set()
        for edge in edges:
            if edge.edge_type == EdgeType.CONTROL_FLOW and edge.flow_rule:
                if edge.flow_rule not in edge_rules_seen:
                    edge_rules_seen.add(edge.flow_rule)
                    style = self.edge_styles[EdgeType.CONTROL_FLOW].get(edge.flow_rule, {})
                    linestyle = self.convert_line_style(style.get("style", "solid"))
                    
                    rule_names = {
                        1: "Rule 1: Boundary→Controller",
                        2: "Rule 2: Controller→Controller", 
                        3: "Rule 3: Sequential Flow",
                        4: "Rule 4: Parallel Flow",
                        5: "Rule 5: Controller→Boundary"
                    }
                    
                    element = plt.Line2D([0], [0], 
                                       color=style.get("color", "#FF0000"),
                                       linestyle=linestyle,
                                       linewidth=style.get("width", 1.0),
                                       label=rule_names.get(edge.flow_rule, f"Control Rule {edge.flow_rule}"))
                    legend_elements.append(element)
            
            elif edge.edge_type == EdgeType.DATA_FLOW:
                relationship = edge.relationship or "use"
                if hasattr(self, '_data_flow_legend_added'):
                    continue
                self._data_flow_legend_added = True
                
                for rel_type, style in self.edge_styles[EdgeType.DATA_FLOW].items():
                    element = plt.Line2D([0], [0],
                                       color=style["color"],
                                       linestyle=self.convert_line_style(style["style"]),
                                       linewidth=style["width"],
                                       label=f"Data Flow ({rel_type})")
                    legend_elements.append(element)
        
        # Special indicators
        if any(comp.warnings for comp in components):
            legend_elements.append(plt.Line2D([0], [0], marker='^', color='#FF0000',
                                            markersize=8, linewidth=0,
                                            label="Warning/Critical"))
        
        if any(comp.is_shared for comp in components):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='#0000FF',
                                            markersize=8, linewidth=0,
                                            label="Shared Component"))
        
        # Create legend with proper positioning
        ax.legend(handles=legend_elements, 
                 loc='center left', 
                 bbox_to_anchor=(1.02, 0.5),
                 fontsize=9,
                 title="UC-Methode Elements",
                 title_fontsize=10,
                 frameon=True,
                 fancybox=True,
                 shadow=True)

    def add_enhanced_title(self, ax: plt.Axes, json_data: Dict[str, Any], components: List[RAComponent], edges: List[RAEdge]) -> None:
        """Add enhanced title with detailed metadata"""
        metadata = json_data.get("metadata", {})
        
        # Main title
        uc_name = metadata.get("uc_name", "Unknown UC")
        domain = metadata.get("domain", "Unknown Domain")
        title = f"Robustness Analysis Diagram: {uc_name}"
        if domain != "Unknown Domain":
            title += f" ({domain.replace('_', ' ').title()})"
        
        ax.set_title(title, fontsize=18, fontweight='bold', pad=25)
        
        # Enhanced subtitle with statistics
        framework = metadata.get("framework_version", "UC-Methode RA-NLF")
        timestamp = metadata.get("analysis_timestamp", "")
        
        # Count components by type
        type_counts = {}
        for comp in components:
            comp_type = comp.component_type.value
            type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
        
        stats = " | ".join([f"{k.title()}: {v}" for k, v in type_counts.items()])
        
        subtitle = f"{framework} | {stats} | Edges: {len(edges)}"
        if timestamp:
            subtitle += f" | Generated: {timestamp[:8]}"
        
        ax.text(0.5, 0.96, subtitle, transform=ax.transAxes,
               ha='center', va='top', fontsize=11, style='italic')

    def create_advanced_diagram(self, json_file_path: str, output_path: str = None) -> str:
        """Create advanced RA diagram with all enhancements"""
        
        # Load and parse data
        components, edges, json_data = self.load_and_parse_json(json_file_path)
        
        if not components:
            raise ValueError("No components found in JSON data")
        
        # Calculate advanced layout
        self.calculate_advanced_layout(components, edges)
        
        # Create figure with enhanced settings
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size, dpi=self.dpi)
        fig.patch.set_facecolor('white')
        
        # Set axis properties for clean diagram
        ax.set_xlim(-0.02, 1.25)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Create components dictionary for edge drawing
        components_dict = {comp.id: comp for comp in components}
        
        # Draw all components with enhanced styling
        for component in components:
            self.create_enhanced_component_patch(component, ax)
        
        # Draw all edges with UC-Methode styling
        for edge in edges:
            self.draw_enhanced_edge(edge, components_dict, ax)
        
        # Add enhanced title and legend
        self.add_enhanced_title(ax, json_data, components, edges)
        self.add_enhanced_legend(ax, components, edges)
        
        # Generate output path if not provided
        if output_path is None:
            json_path = Path(json_file_path)
            metadata = json_data.get("metadata", {})
            uc_name = metadata.get("uc_name", "Unknown_UC")
            timestamp = metadata.get("analysis_timestamp", "")
            
            output_path = str(json_path.parent / f"{uc_name}_Advanced_RA_Diagram_{timestamp}.png")
        
        # Save with high quality
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none',
                   pad_inches=0.1)
        plt.close()
        
        return output_path


def main():
    """Demonstration of the advanced RA diagram engine"""
    print("Advanced RA Diagram Visualization Engine")
    print("=" * 50)
    
    # Create engine instance
    engine = AdvancedRAEngine(figure_size=(28, 20))
    
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
    
    # Process each file
    for json_file in viz_files:
        print(f"\nProcessing: {json_file.name}")
        
        try:
            diagram_output = engine.create_advanced_diagram(str(json_file))
            print(f"  Advanced diagram created: {diagram_output}")
            
        except Exception as e:
            print(f"  Error processing {json_file.name}: {e}")
    
    print("\nAdvanced RA diagram generation completed!")


if __name__ == "__main__":
    main()