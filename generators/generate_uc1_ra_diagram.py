#!/usr/bin/env python3
"""
Generate a proper RA diagram for UC1 with correct UML symbols
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow, Rectangle
import numpy as np

def draw_actor(ax, x, y, name):
    """Draw actor with stick figure symbol"""
    # Head
    head = Circle((x, y+0.3), 0.05, facecolor='white', edgecolor='black', linewidth=1)
    ax.add_patch(head)
    
    # Body (vertical line)
    ax.plot([x, x], [y+0.25, y-0.1], 'k-', linewidth=1)
    
    # Arms (horizontal line)
    ax.plot([x-0.1, x+0.1], [y+0.1, y+0.1], 'k-', linewidth=1)
    
    # Legs
    ax.plot([x, x-0.08], [y-0.1, y-0.25], 'k-', linewidth=1)  # Left leg
    ax.plot([x, x+0.08], [y-0.1, y-0.25], 'k-', linewidth=1)  # Right leg
    
    # Name
    ax.text(x, y-0.4, name, ha='center', va='center', fontsize=8, weight='bold')

def draw_boundary(ax, x, y, name):
    """Draw boundary with circle and 90° rotated T"""
    # Circle
    circle = Circle((x, y), 0.12, facecolor='lightblue', edgecolor='black', linewidth=1)
    ax.add_patch(circle)
    
    # 90° rotated T (horizontal line with vertical line in middle)
    # Horizontal line
    ax.plot([x-0.08, x+0.08], [y, y], 'k-', linewidth=2)
    # Vertical line (downward)
    ax.plot([x, x], [y, y-0.08], 'k-', linewidth=2)
    
    # Name
    ax.text(x, y-0.25, name, ha='center', va='center', fontsize=8, weight='bold')

def draw_controller(ax, x, y, name):
    """Draw controller with circle and arrow on top"""
    # Special styling for Domain Orchestrator
    if 'DomainOrchestrator' in name:
        circle = Circle((x, y), 0.15, facecolor='gold', edgecolor='black', linewidth=2)
        fontweight = 'bold'
        fontsize = 9
    else:
        circle = Circle((x, y), 0.12, facecolor='lightgreen', edgecolor='black', linewidth=1)
        fontweight = 'bold'
        fontsize = 8
    
    ax.add_patch(circle)
    
    # Arrow pointing up
    arrow = mpatches.FancyArrowPatch((x, y+0.05), (x, y+0.15),
                                   arrowstyle='->', mutation_scale=15, color='black')
    ax.add_patch(arrow)
    
    # Name
    ax.text(x, y-0.25, name, ha='center', va='center', fontsize=fontsize, weight=fontweight)

def draw_entity(ax, x, y, name):
    """Draw entity with circle and tangent line at bottom"""
    # Circle
    circle = Circle((x, y), 0.12, facecolor='lightyellow', edgecolor='black', linewidth=1)
    ax.add_patch(circle)
    
    # Tangent line at bottom
    ax.plot([x-0.08, x+0.08], [y-0.12, y-0.12], 'k-', linewidth=2)
    
    # Name
    ax.text(x, y-0.35, name, ha='center', va='center', fontsize=8, weight='bold')

def draw_control_flow(ax, x1, y1, x2, y2):
    """Draw control flow as simple line"""
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1, alpha=0.7)

def create_uc1_ra_diagram():
    """Create RA diagram for UC1 based on analysis results"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'UC1: Prepare Milk Coffee - RA Diagram (UC-Methode)', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Define positions for all components
    
    # === ACTORS ===
    actor_positions = {
        'Timer': (1, 8),
        'User': (1, 2),
        'ExternalTrigger': (1, 5)
    }
    
    # === BOUNDARIES ===
    boundary_positions = {
        'TimeTriggerBoundary': (3, 8),
        'HMIAdditiveInputBoundary': (3, 2),
        'HMIStatusDisplayBoundary': (11, 6),
        'HMIErrorDisplayBoundary': (11, 4),
        'ProductDeliveryBoundary': (13, 6),
        'WaterSupplyBoundary': (3, 5),
        'CoffeeBeansSupplyBoundary': (3, 3.5),
        'MilkSupplyBoundary': (3, 1)
    }
    
    # === CONTROLLERS ===
    controller_positions = {
        'DomainOrchestrator': (7.5, 6),  # Central coordination controller
        'TimeManager': (5, 8),
        'HMIController': (9, 4),
        'HeaterManager': (5, 7),
        'FilterManager': (6, 7),
        'AmountManager': (7, 7),
        'CupManager': (8, 7),
        'CoffeeManager': (5, 5.5),
        'MilkManager': (6, 5.5),
        'MessageManager': (7, 4),
        'CoffeeBeansSupplyController': (5, 3.5),
        'WaterSupplyController': (5, 5),
        'MilkSupplyController': (5, 1)
    }
    
    # === ENTITIES ===
    entity_positions = {
        'Water': (13, 5),
        'CoffeeBeans': (13, 3.5),
        'Milk': (13, 1),
        'Coffee': (8, 5.5),
        'Cup': (10, 7),
        'Filter': (7, 6),
        'Amount': (8, 6),
        'Message': (9, 3),
        'Error': (10, 3),
        'Sugar': (4, 2)
    }
    
    # Draw all actors
    for name, (x, y) in actor_positions.items():
        draw_actor(ax, x, y, name)
    
    # Draw all boundaries
    for name, (x, y) in boundary_positions.items():
        draw_boundary(ax, x, y, name)
    
    # Draw all controllers
    for name, (x, y) in controller_positions.items():
        draw_controller(ax, x, y, name)
    
    # Draw all entities
    for name, (x, y) in entity_positions.items():
        draw_entity(ax, x, y, name)
    
    # === CONTROL FLOWS (based on UC-Methode analysis) ===
    
    # Rule 1: Boundary -> Controller (External inputs)
    draw_control_flow(ax, 3, 8, 5, 8)        # TimeTriggerBoundary -> TimeManager
    draw_control_flow(ax, 3, 2, 9, 4)        # HMIAdditiveInputBoundary -> HMIController
    draw_control_flow(ax, 3, 5, 5, 5)        # WaterSupplyBoundary -> WaterSupplyController
    
    # Domain Orchestrator coordinates all other controllers
    draw_control_flow(ax, 5, 8, 7.5, 6)      # TimeManager -> DomainOrchestrator
    draw_control_flow(ax, 7.5, 6, 5, 7)      # DomainOrchestrator -> HeaterManager
    draw_control_flow(ax, 7.5, 6, 6, 7)      # DomainOrchestrator -> FilterManager
    draw_control_flow(ax, 7.5, 6, 7, 7)      # DomainOrchestrator -> AmountManager
    draw_control_flow(ax, 7.5, 6, 8, 7)      # DomainOrchestrator -> CupManager
    draw_control_flow(ax, 7.5, 6, 5, 5.5)    # DomainOrchestrator -> CoffeeManager
    draw_control_flow(ax, 7.5, 6, 6, 5.5)    # DomainOrchestrator -> MilkManager
    draw_control_flow(ax, 7.5, 6, 7, 4)      # DomainOrchestrator -> MessageManager
    draw_control_flow(ax, 7.5, 6, 9, 4)      # DomainOrchestrator -> HMIController
    
    # Resource controllers also coordinated by Domain Orchestrator
    draw_control_flow(ax, 7.5, 6, 5, 5)      # DomainOrchestrator -> WaterSupplyController
    draw_control_flow(ax, 7.5, 6, 5, 3.5)    # DomainOrchestrator -> CoffeeBeansSupplyController
    draw_control_flow(ax, 7.5, 6, 5, 1)      # DomainOrchestrator -> MilkSupplyController
    
    # Rule 5: Controller -> Boundary (External outputs)
    draw_control_flow(ax, 9, 4, 11, 6)       # HMIController -> HMIStatusDisplayBoundary
    draw_control_flow(ax, 9, 4, 11, 4)       # HMIController -> HMIErrorDisplayBoundary
    draw_control_flow(ax, 8, 7, 13, 6)       # CupManager -> ProductDeliveryBoundary
    
    # NO DIRECT CONTROL FLOWS TO ENTITIES!
    # Entities are only referenced via associations, not control flows
    # UC-Methode Rule: Controllers must NOT have control flow arrows to Entities
    
    # Add legend
    legend_y = 0.5
    ax.text(0.5, legend_y, 'Legend:', fontsize=10, weight='bold')
    
    # Actor symbol
    draw_actor(ax, 1, legend_y-0.5, 'Actor')
    
    # Boundary symbol  
    draw_boundary(ax, 3, legend_y-0.5, 'Boundary')
    
    # Controller symbol
    draw_controller(ax, 5, legend_y-0.5, 'Controller')
    
    # Entity symbol
    draw_entity(ax, 7, legend_y-0.5, 'Entity')
    
    # Control flow
    ax.plot([9, 10], [legend_y-0.5, legend_y-0.5], 'k-', linewidth=1)
    ax.text(10.5, legend_y-0.5, 'Control Flow', fontsize=8, va='center')
    
    plt.tight_layout()
    plt.savefig('D:\\KI\\RA-NLF\\UC1_RA_Diagram_UC_Methode.png', dpi=300, bbox_inches='tight')
    plt.savefig('D:\\KI\\RA-NLF\\UC1_RA_Diagram_UC_Methode.svg', bbox_inches='tight')
    
    print("RA Diagram for UC1 created successfully!")
    print("Files saved:")
    print("- UC1_RA_Diagram_UC_Methode.png")
    print("- UC1_RA_Diagram_UC_Methode.svg")
    
    plt.close()  # Close instead of show to avoid blocking

if __name__ == "__main__":
    create_uc1_ra_diagram()