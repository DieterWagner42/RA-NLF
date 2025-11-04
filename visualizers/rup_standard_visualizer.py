#!/usr/bin/env python3
"""
RUP Standard RA Diagram Visualizer
==================================

Generates RUP/UML compliant Robustness Analysis diagrams according to Wikipedia standard:
- Actor: Strichmännchen (stick figure)
- Controller: Kreis mit Pfeil < nach links zeigend (circle with arrow < pointing left)
- Boundary: Rechteck mit gebogener Oberkante (rectangle with curved top)
- Entity: Kreis mit Linie unten tangential (circle with tangent line at bottom)

Based on: https://de.wikipedia.org/wiki/Robustheitsanalyse
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class RAComponent:
    name: str
    type: str  # "Actor", "Boundary", "Controller", "Entity"
    x: float
    y: float
    description: str = ""
    steps: List[str] = None

@dataclass
class RAConnection:
    from_component: str
    to_component: str
    type: str  # "control", "data_use", "data_provide"
    label: str = ""

class RUPStandardVisualizer:
    """RUP-standard compliant RA diagram generator"""
    
    def __init__(self, width=16, height=12):
        self.width = width
        self.height = height
        self.fig = None
        self.ax = None
        self.components = {}
        self.connections = []
        
    def create_diagram(self, ra_classes: List, title: str = "RA Diagram") -> str:
        """Create RUP-standard RA diagram from RA classes"""
        
        # Convert RA classes to components
        self._convert_ra_classes(ra_classes)
        
        # Create figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(self.width, self.height))
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 8)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Add title
        self.ax.text(5, 7.5, title, fontsize=16, fontweight='bold', ha='center')
        
        # Position components by type
        self._layout_components()
        
        # Draw components
        self._draw_all_components()
        
        # Add legend
        self._add_legend()
        
        # Save diagram
        output_file = f"{title.replace(' ', '_').replace(':', '')}_RUP_Standard.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _convert_ra_classes(self, ra_classes: List):
        """Convert generic RA classes to components"""
        actors = []
        boundaries = []
        controllers = []
        entities = []
        
        for ra_class in ra_classes:
            if ra_class.type == "Actor":
                actors.append(RAComponent(ra_class.name, "Actor", 0, 0, ra_class.description))
            elif ra_class.type == "Boundary":
                boundaries.append(RAComponent(ra_class.name, "Boundary", 0, 0, ra_class.description))
            elif ra_class.type == "Controller":
                controllers.append(RAComponent(ra_class.name, "Controller", 0, 0, ra_class.description))
            elif ra_class.type == "Entity":
                entities.append(RAComponent(ra_class.name, "Entity", 0, 0, ra_class.description))
        
        # Store components by type
        self.actors = actors
        self.boundaries = boundaries  
        self.controllers = controllers
        self.entities = entities
        
        # Create components dictionary
        for comp in actors + boundaries + controllers + entities:
            self.components[comp.name] = comp
    
    def _layout_components(self):
        """Layout components in RUP-standard arrangement"""
        
        # Actors on the left
        y_start = 6
        y_step = 0.8
        for i, actor in enumerate(self.actors):
            actor.x = 1
            actor.y = y_start - i * y_step
        
        # Boundaries in the middle-left
        for i, boundary in enumerate(self.boundaries[:8]):  # Limit to avoid overlap
            boundary.x = 3
            boundary.y = y_start - i * 0.6
        
        # Controllers in the center
        cols = 3
        for i, controller in enumerate(self.controllers[:12]):  # Limit to avoid overlap
            controller.x = 5 + (i % cols) * 1.5
            controller.y = y_start - (i // cols) * 0.8
        
        # Entities on the right
        for i, entity in enumerate(self.entities[:10]):  # Limit to avoid overlap
            entity.x = 8.5
            entity.y = y_start - i * 0.5
    
    def _draw_all_components(self):
        """Draw all components with RUP-standard symbols"""
        
        # Draw actors (stick figures)
        for actor in self.actors:
            self._draw_actor(actor.x, actor.y, actor.name)
        
        # Draw boundaries (circle with T)
        for boundary in self.boundaries:
            self._draw_boundary(boundary.x, boundary.y, boundary.name)
        
        # Draw controllers (circle with arrow)
        for controller in self.controllers:
            self._draw_controller(controller.x, controller.y, controller.name)
        
        # Draw entities (rectangle with line)
        for entity in self.entities:
            self._draw_entity(entity.x, entity.y, entity.name)
    
    def _draw_actor(self, x: float, y: float, name: str):
        """Draw Actor as Strichmännchen (stick figure)"""
        
        # Head (circle)
        head = Circle((x, y + 0.15), 0.08, fill=False, linewidth=2, color='blue')
        self.ax.add_patch(head)
        
        # Body (vertical line)
        self.ax.plot([x, x], [y + 0.07, y - 0.15], 'b-', linewidth=2)
        
        # Arms (horizontal line)
        self.ax.plot([x - 0.1, x + 0.1], [y, y], 'b-', linewidth=2)
        
        # Legs (two diagonal lines)
        self.ax.plot([x, x - 0.08], [y - 0.15, y - 0.3], 'b-', linewidth=2)
        self.ax.plot([x, x + 0.08], [y - 0.15, y - 0.3], 'b-', linewidth=2)
        
        # Name
        self.ax.text(x, y - 0.45, name, ha='center', va='top', fontsize=8, fontweight='bold')
    
    def _draw_boundary(self, x: float, y: float, name: str):
        """Draw Boundary as Rechteck mit gebogener Oberkante (Wikipedia standard)"""
        
        # Rectangle with curved top (using FancyBboxPatch)
        width = 0.24
        height = 0.16
        
        # Create rectangle with rounded top
        boundary = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor='none',
            edgecolor='green',
            linewidth=2
        )
        self.ax.add_patch(boundary)
        
        # Name
        self.ax.text(x, y - 0.25, name, ha='center', va='top', fontsize=8, fontweight='bold')
    
    def _draw_controller(self, x: float, y: float, name: str):
        """Draw Controller as Kreis mit Pfeil < nach links zeigend (Wikipedia standard)"""
        
        # Circle
        circle = Circle((x, y), 0.12, fill=False, linewidth=2, color='orange')
        self.ax.add_patch(circle)
        
        # Arrow < pointing left (centered in circle, Wikipedia standard)
        self.ax.plot([x - 0.05, x + 0.05], [y + 0.04, y], 'orange', linewidth=2)
        self.ax.plot([x - 0.05, x + 0.05], [y - 0.04, y], 'orange', linewidth=2)
        
        # Name
        self.ax.text(x, y - 0.25, name, ha='center', va='top', fontsize=8, fontweight='bold')
    
    def _draw_entity(self, x: float, y: float, name: str):
        """Draw Entity as Kreis mit Linie unten tangential"""
        
        # Circle
        circle = Circle((x, y), 0.12, fill=False, linewidth=2, color='red')
        self.ax.add_patch(circle)
        
        # Tangent line at bottom of circle
        line_y = y - 0.12
        self.ax.plot([x - 0.15, x + 0.15], [line_y, line_y], 'r-', linewidth=2)
        
        # Name
        self.ax.text(x, y - 0.25, name, ha='center', va='top', fontsize=8, fontweight='bold')
    
    def _add_legend(self):
        """Add RUP symbol legend"""
        
        legend_x = 0.5
        legend_y = 1.5
        
        self.ax.text(legend_x, legend_y + 0.4, "RUP/UML Symbole:", fontsize=10, fontweight='bold')
        
        # Actor legend
        self._draw_actor(legend_x, legend_y, "")
        self.ax.text(legend_x + 0.3, legend_y, "Actor", fontsize=9)
        
        # Boundary legend
        self._draw_boundary(legend_x, legend_y - 0.4, "")
        self.ax.text(legend_x + 0.3, legend_y - 0.4, "Boundary", fontsize=9)
        
        # Controller legend
        self._draw_controller(legend_x, legend_y - 0.8, "")
        self.ax.text(legend_x + 0.3, legend_y - 0.8, "Controller", fontsize=9)
        
        # Entity legend
        self._draw_entity(legend_x, legend_y - 1.2, "")
        self.ax.text(legend_x + 0.3, legend_y - 1.2, "Entity", fontsize=9)

def generate_uc3_rup_diagram(ra_classes: List) -> str:
    """Generate UC3 RA diagram with RUP standard symbols"""
    
    visualizer = RUPStandardVisualizer()
    output_file = visualizer.create_diagram(
        ra_classes=ra_classes,
        title="UC3: Rocket Launch RA Diagram"
    )
    
    return output_file

if __name__ == "__main__":
    # Test with empty classes
    print("RUP Standard Visualizer created successfully!")
    print("Use generate_uc3_rup_diagram(ra_classes) to create diagrams.")