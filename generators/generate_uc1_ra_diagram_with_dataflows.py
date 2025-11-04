#!/usr/bin/env python3
"""
Generate RA diagram for UC1 with both control flows and data flows
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
    """Draw control flow as simple black line"""
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1, alpha=0.7)

def draw_data_flow_use(ax, x1, y1, x2, y2):
    """Draw data flow 'use' relationship as dashed blue arrow pointing to Controller"""
    # Arrow points from Entity to Controller (USE direction)
    ax.annotate('', xy=(x1, y1), xytext=(x2, y2),
                arrowprops=dict(arrowstyle='->', color='blue', linestyle='--', 
                               linewidth=1.5, alpha=0.8))

def draw_data_flow_provide(ax, x1, y1, x2, y2):
    """Draw data flow 'provide' relationship as dashed red arrow pointing to Entity"""
    # Arrow points from Controller to Entity (PROVIDE direction)
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='red', linestyle='--', 
                               linewidth=1.5, alpha=0.8))

def create_uc1_ra_diagram_with_dataflows():
    """Create RA diagram for UC1 with control flows and data flows"""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(8, 10.5, 'UC1: Prepare Milk Coffee - RA Diagram with Data Flows (UC-Methode)', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Define positions for all components (optimized for data flow visibility)
    
    # === ACTORS ===
    actor_positions = {
        'Timer': (1, 9),
        'User': (1, 2),
        'ExternalTrigger': (1, 5.5)
    }
    
    # === BOUNDARIES ===
    boundary_positions = {
        'TimeTriggerBoundary': (3, 9),
        'HMIAdditiveInputBoundary': (3, 2),
        'HMIStatusDisplayBoundary': (13, 7),
        'HMIErrorDisplayBoundary': (13, 5),
        'ProductDeliveryBoundary': (15, 7),
        'WaterSupplyBoundary': (3, 5.5),
        'CoffeeBeansSupplyBoundary': (3, 4),
        'MilkSupplyBoundary': (3, 2.5),
        'SugarSupplyBoundary': (3, 1.5)
    }
    
    # === CONTROLLERS ===
    controller_positions = {
        'DomainOrchestrator': (8, 6.5),  # Central coordination controller
        'TimeManager': (5, 9),
        'HMIController': (11, 5.5),
        'HeaterManager': (5, 8),
        'FilterManager': (6, 8),
        'AmountManager': (7, 8),
        'CupManager': (9, 8),
        'CoffeeManager': (6, 6),
        'MilkManager': (7, 5),
        'MessageManager': (9, 4),
        'CoffeeBeansSupplyController': (5, 4),
        'WaterSupplyController': (5, 5.5),
        'MilkSupplyController': (5, 2.5),
        'SugarSupplyController': (5, 1.5),
        'SugarManager': (8, 3)
    }
    
    # === ENTITIES ===
    entity_positions = {
        # Input entities for CoffeeManager
        'GroundCoffee': (4, 6),
        'HotWater': (4, 7),
        'Filter': (4, 8),
        # Output entities
        'Coffee': (10, 6),
        'Cup': (11, 8),
        'Water': (15, 5.5),
        'CoffeeBeans': (15, 4),
        'Milk': (15, 2.5),
        'Amount': (9, 7),
        'Message': (11, 4),
        'Error': (12, 4),
        'Sugar': (15, 1.5)
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
    
    # === CONTROL FLOWS (Black lines) ===
    
    # Actor -> Boundary connections
    draw_control_flow(ax, 1, 9, 3, 9)        # Timer -> TimeTriggerBoundary
    draw_control_flow(ax, 1, 2, 3, 2)        # User -> HMIAdditiveInputBoundary
    draw_control_flow(ax, 1, 5.5, 3, 5.5)    # ExternalTrigger -> WaterSupplyBoundary
    draw_control_flow(ax, 1, 5.5, 3, 4)      # ExternalTrigger -> CoffeeBeansSupplyBoundary
    draw_control_flow(ax, 1, 5.5, 3, 2.5)    # ExternalTrigger -> MilkSupplyBoundary
    draw_control_flow(ax, 1, 5.5, 3, 1.5)    # ExternalTrigger -> SugarSupplyBoundary
    
    # Rule 1: Boundary -> Controller (External inputs)
    draw_control_flow(ax, 3, 9, 5, 9)        # TimeTriggerBoundary -> TimeManager
    draw_control_flow(ax, 3, 2, 11, 5.5)     # HMIAdditiveInputBoundary -> HMIController
    draw_control_flow(ax, 3, 5.5, 5, 5.5)    # WaterSupplyBoundary -> WaterSupplyController
    draw_control_flow(ax, 3, 4, 5, 4)        # CoffeeBeansSupplyBoundary -> CoffeeBeansSupplyController
    draw_control_flow(ax, 3, 2.5, 5, 2.5)    # MilkSupplyBoundary -> MilkSupplyController
    draw_control_flow(ax, 3, 1.5, 5, 1.5)    # SugarSupplyBoundary -> SugarSupplyController
    
    # Domain Orchestrator coordinates all other controllers
    draw_control_flow(ax, 5, 9, 8, 6.5)      # TimeManager -> DomainOrchestrator
    draw_control_flow(ax, 8, 6.5, 5, 8)      # DomainOrchestrator -> HeaterManager
    draw_control_flow(ax, 8, 6.5, 6, 8)      # DomainOrchestrator -> FilterManager
    draw_control_flow(ax, 8, 6.5, 7, 8)      # DomainOrchestrator -> AmountManager
    draw_control_flow(ax, 8, 6.5, 9, 8)      # DomainOrchestrator -> CupManager
    draw_control_flow(ax, 8, 6.5, 6, 6)      # DomainOrchestrator -> CoffeeManager
    draw_control_flow(ax, 8, 6.5, 7, 5)      # DomainOrchestrator -> MilkManager
    draw_control_flow(ax, 8, 6.5, 9, 4)      # DomainOrchestrator -> MessageManager
    draw_control_flow(ax, 8, 6.5, 11, 5.5)   # DomainOrchestrator -> HMIController
    draw_control_flow(ax, 8, 6.5, 5, 5.5)    # DomainOrchestrator -> WaterSupplyController
    draw_control_flow(ax, 8, 6.5, 5, 4)      # DomainOrchestrator -> CoffeeBeansSupplyController
    draw_control_flow(ax, 8, 6.5, 5, 2.5)    # DomainOrchestrator -> MilkSupplyController
    draw_control_flow(ax, 8, 6.5, 5, 1.5)    # DomainOrchestrator -> SugarSupplyController
    draw_control_flow(ax, 8, 6.5, 8, 3)      # DomainOrchestrator -> SugarManager
    
    # Rule 5: Controller -> Boundary (External outputs)
    draw_control_flow(ax, 11, 5.5, 13, 7)    # HMIController -> HMIStatusDisplayBoundary
    draw_control_flow(ax, 11, 5.5, 13, 5)    # HMIController -> HMIErrorDisplayBoundary
    draw_control_flow(ax, 9, 8, 15, 7)       # CupManager -> ProductDeliveryBoundary
    draw_control_flow(ax, 9, 4, 13, 7)       # MessageManager -> HMIStatusDisplayBoundary
    draw_control_flow(ax, 9, 4, 13, 5)       # MessageManager -> HMIErrorDisplayBoundary
    
    # === DATA FLOWS ===
    
    # USE RELATIONSHIPS (Blue dashed lines) - Controller uses Entity
    
    # CoffeeManager uses inputs for brewing 
    draw_data_flow_use(ax, 6, 6, 4, 6)       # CoffeeManager --use--> GroundCoffee
    draw_data_flow_use(ax, 6, 6, 4, 7)       # CoffeeManager --use--> HotWater  
    draw_data_flow_use(ax, 6, 6, 4, 8)       # CoffeeManager --use--> Filter
    draw_data_flow_use(ax, 6, 6, 15, 5.5)    # CoffeeManager --use--> Water (B3a: "with the set water amount")
    draw_data_flow_use(ax, 6, 6, 11, 8)      # CoffeeManager --use--> Cup (B3a: "into the cup" = uses cup as container)
    
    # AmountManager uses CoffeeBeans for grinding
    draw_data_flow_use(ax, 7, 8, 15, 4)      # AmountManager --use--> CoffeeBeans
    
    # MilkManager uses Coffee for adding milk
    draw_data_flow_use(ax, 7, 5, 10, 6)      # MilkManager --use--> Coffee
    
    # MilkManager uses Cup as target (B3b: "to the cup")
    draw_data_flow_use(ax, 7, 5, 11, 8)      # MilkManager --use--> Cup (B3b: "to the cup" = uses cup as target)
    
    # MilkManager uses Milk (B3b: "adds milk")
    draw_data_flow_use(ax, 7, 5, 15, 2.5)    # MilkManager --use--> Milk
    
    # SugarManager uses Coffee and Cup (E1.1: "adds sugar to the cup")
    draw_data_flow_use(ax, 8, 3, 10, 6)      # SugarManager --use--> Coffee
    draw_data_flow_use(ax, 8, 3, 11, 8)      # SugarManager --use--> Cup (E1.1: "to the cup")
    draw_data_flow_use(ax, 8, 3, 15, 1.5)    # SugarManager --use--> Sugar
    
    # HeaterManager uses Water for heating
    draw_data_flow_use(ax, 5, 8, 15, 5.5)    # HeaterManager --use--> Water
    
    # MessageManager uses Message for output
    draw_data_flow_use(ax, 9, 4, 11, 4)      # MessageManager --use--> Message
    
    # PROVIDE RELATIONSHIPS (Red dashed lines) - Controller provides Entity
    
    # HeaterManager provides HotWater (implicit from "activates water heater")
    draw_data_flow_provide(ax, 5, 8, 4, 7)   # HeaterManager --provide--> HotWater
    
    # FilterManager provides Filter (implicit from "prepares filter")
    draw_data_flow_provide(ax, 6, 8, 4, 8)   # FilterManager --provide--> Filter
    
    # AmountManager provides GroundCoffee from grinding
    draw_data_flow_provide(ax, 7, 8, 4, 6)   # AmountManager --provide--> GroundCoffee
    
    # CoffeeManager provides Coffee from brewing
    draw_data_flow_provide(ax, 6, 6, 10, 6)  # CoffeeManager --provide--> Coffee
    
    # MilkManager provides enhanced Coffee (Coffee + Milk)
    draw_data_flow_provide(ax, 7, 5, 10, 6)  # MilkManager --provide--> Coffee (enhanced)
    
    # CupManager provides Cup (from "retrieves cup")
    draw_data_flow_provide(ax, 9, 8, 11, 8)  # CupManager --provide--> Cup
    
    # AmountManager uses Filter as container (B2c: "into the filter") - PREPOSITION-BASED  
    draw_data_flow_use(ax, 7, 8, 4, 8)       # AmountManager --use--> Filter (B2c: "into the filter" = uses filter as container)
    
    # MessageManager provides Message (from "outputs message")
    draw_data_flow_provide(ax, 9, 4, 11, 4)  # MessageManager --provide--> Message
    
    # MessageManager provides Error (from "outputs error message")
    draw_data_flow_provide(ax, 9, 4, 12, 4)  # MessageManager --provide--> Error
    
    # HMIController provides Message (user interaction outputs)
    draw_data_flow_provide(ax, 11, 5.5, 11, 4)  # HMIController --provide--> Message
    
    # SugarManager provides enhanced Coffee (E1.1: "adds sugar")
    draw_data_flow_provide(ax, 8, 3, 10, 6)     # SugarManager --provide--> Coffee
    
    # SupplyControllers provide their respective entities
    draw_data_flow_provide(ax, 5, 5.5, 15, 5.5) # WaterSupplyController --provide--> Water
    draw_data_flow_provide(ax, 5, 4, 15, 4)     # CoffeeBeansSupplyController --provide--> CoffeeBeans
    draw_data_flow_provide(ax, 5, 2.5, 15, 2.5) # MilkSupplyController --provide--> Milk
    draw_data_flow_provide(ax, 5, 1.5, 15, 1.5) # SugarSupplyController --provide--> Sugar
    
    # Add comprehensive legend
    legend_y = 1
    ax.text(0.5, legend_y+0.5, 'Legend:', fontsize=12, weight='bold')
    
    # UML Symbols
    draw_actor(ax, 1, legend_y, 'Actor')
    draw_boundary(ax, 3, legend_y, 'Boundary')
    draw_controller(ax, 5, legend_y, 'Controller')
    draw_entity(ax, 7, legend_y, 'Entity')
    
    # Flow types
    ax.plot([9, 10], [legend_y, legend_y], 'k-', linewidth=1)
    ax.text(10.5, legend_y, 'Control Flow', fontsize=9, va='center')
    
    ax.plot([9, 10], [legend_y-0.3, legend_y-0.3], 'b--', linewidth=1.5)
    ax.text(10.5, legend_y-0.3, 'Data Flow: Use', fontsize=9, va='center', color='blue')
    
    ax.plot([9, 10], [legend_y-0.6, legend_y-0.6], 'r--', linewidth=1.5)
    ax.text(10.5, legend_y-0.6, 'Data Flow: Provide', fontsize=9, va='center', color='red')
    
    # Add transformation annotations with corrected preposition semantics
    ax.text(5.5, 6.8, 'GroundCoffee +\nHotWater + Filter +\nWater (with) + Cup (into)\n→ Coffee', 
            ha='center', va='center', fontsize=7, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat', alpha=0.7))
    
    ax.text(7.5, 7.2, 'CoffeeBeans\n→ GroundCoffee\n(into Filter)', 
            ha='center', va='center', fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('D:\\KI\\RA-NLF\\UC1_RA_Diagram_With_DataFlows.png', dpi=300, bbox_inches='tight')
    plt.savefig('D:\\KI\\RA-NLF\\UC1_RA_Diagram_With_DataFlows.svg', bbox_inches='tight')
    
    print("RA Diagram with Data Flows for UC1 created successfully!")
    print("Files saved:")
    print("- UC1_RA_Diagram_With_DataFlows.png")
    print("- UC1_RA_Diagram_With_DataFlows.svg")
    
    print("\nData Flow Legend:")
    print("Blue dashed lines (--use-->): Controller uses Entity as input")
    print("Red dashed lines (--provide-->): Controller provides Entity as output")
    print("Black solid lines: Control flow (UC-Methode rules)")
    
    plt.close()  # Close instead of show to avoid blocking

if __name__ == "__main__":
    create_uc1_ra_diagram_with_dataflows()