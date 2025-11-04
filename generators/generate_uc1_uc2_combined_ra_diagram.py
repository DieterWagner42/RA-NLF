#!/usr/bin/env python3
"""
Generate combined RA diagram for UC1 (Prepare Milk Coffee) + UC2 (Prepare Espresso) 
with both control flows and data flows, showing shared and unique components
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
    # UC2-specific controllers (compression-related)
    elif name in ['CompressorManager', 'PressureManager']:
        circle = Circle((x, y), 0.12, facecolor='lightcoral', edgecolor='black', linewidth=2)
        fontweight = 'bold'
        fontsize = 8
    # UC1-specific controllers
    elif name in ['MilkManager', 'SugarManager', 'TimeManager', 'AmountManager']:
        circle = Circle((x, y), 0.12, facecolor='lightsteelblue', edgecolor='black', linewidth=1)
        fontweight = 'bold'
        fontsize = 8
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
    # Color coding for entity types
    if name in ['Pressure', 'Compressor', 'HotWater']:  # UC2-specific
        facecolor = 'mistyrose'
        edgecolor = 'red'
        linewidth = 2
    elif name in ['Milk', 'Sugar']:  # UC1-specific
        facecolor = 'lightcyan'
        edgecolor = 'blue'
        linewidth = 2
    else:  # Shared entities
        facecolor = 'lightyellow'
        edgecolor = 'black'
        linewidth = 1
    
    # Circle
    circle = Circle((x, y), 0.12, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(circle)
    
    # Tangent line at bottom
    ax.plot([x-0.08, x+0.08], [y-0.12, y-0.12], color=edgecolor, linewidth=2)
    
    # Name
    ax.text(x, y-0.35, name, ha='center', va='center', fontsize=8, weight='bold')

def draw_control_flow(ax, x1, y1, x2, y2):
    """Draw control flow as simple black line"""
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1, alpha=0.7)

def draw_data_flow_use(ax, x1, y1, x2, y2):
    """Draw data flow 'use' relationship as dashed blue line"""
    ax.plot([x1, x2], [y1, y2], 'b--', linewidth=1.5, alpha=0.8)

def draw_data_flow_provide(ax, x1, y1, x2, y2):
    """Draw data flow 'provide' relationship as dashed red line"""
    ax.plot([x1, x2], [y1, y2], 'r--', linewidth=1.5, alpha=0.8)

def create_uc1_uc2_combined_ra_diagram():
    """Create combined RA diagram for UC1 + UC2 with control flows and data flows"""
    
    fig, ax = plt.subplots(1, 1, figsize=(22, 16))
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1, 13)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(10, 12.5, 'UC1 + UC2: Milk Coffee & Espresso Preparation - Combined RA Diagram (UC-Methode)', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Section headers
    ax.text(3, 11.5, 'UC1: Milk Coffee', ha='center', va='center', fontsize=12, weight='bold', color='blue')
    ax.text(17, 11.5, 'UC2: Espresso', ha='center', va='center', fontsize=12, weight='bold', color='red')
    ax.text(10, 11.5, 'SHARED', ha='center', va='center', fontsize=12, weight='bold', color='green')
    
    # Define positions for all components
    
    # === ACTORS ===
    actor_positions = {
        'Timer': (1, 10),
        'User': (1, 6),
        'ExternalTrigger': (1, 3),
    }
    
    # === BOUNDARIES ===
    boundary_positions = {
        # UC1-specific boundaries
        'TimeTriggerBoundary': (3, 10),
        'HMIAdditiveInputBoundary': (3, 1.5),
        'MilkSupplyBoundary': (3, 2),
        'SugarSupplyBoundary': (3, 1),
        
        # Shared boundaries
        'HMIStatusDisplayBoundary': (10, 9),
        'HMIErrorDisplayBoundary': (10, 8),
        'ProductDeliveryBoundary': (10, 7),
        'WaterSupplyBoundary': (3, 4),
        'CoffeeBeansSupplyBoundary': (3, 3),
        
        # UC2-specific boundaries
        'HMIBeverageSelectionBoundary': (17, 10),
        'EquipmentMaintenanceBoundary': (17, 8),
    }
    
    # === CONTROLLERS ===
    controller_positions = {
        # Central Domain Orchestrator
        'Beverage_PreparationDomainOrchestrator': (10, 6),
        
        # UC1-specific controllers
        'TimeManager': (5, 10),
        'AmountManager': (5, 9),
        'MilkManager': (5, 2),
        'SugarManager': (5, 1),
        'MilkSupplyController': (4, 2),
        'SugarSupplyController': (4, 1),
        
        # Shared controllers
        'HMIController': (8, 8),
        'HeaterManager': (7, 9),
        'FilterManager': (8, 9),
        'CupManager': (9, 9),
        'CoffeeManager': (8, 6),
        'MessageManager': (12, 8),
        'CoffeeBeansSupplyController': (4, 3),
        'WaterSupplyController': (4, 4),
        
        # UC2-specific controllers
        'CompressorManager': (15, 6),  # THE COMPRESSION CONTROLLER
        'PressureManager': (15, 5),
    }
    
    # === ENTITIES ===
    entity_positions = {
        # UC1-specific entities
        'Milk': (19, 2),
        'Sugar': (19, 1),
        'Amount': (6, 8),
        
        # Shared entities
        'Water': (19, 4),
        'CoffeeBeans': (19, 3),
        'GroundCoffee': (6, 6),
        'Filter': (6, 9),
        'Coffee': (12, 6),
        'Cup': (11, 9),
        'Message': (14, 8),
        'Error': (14, 7),
        
        # UC2-specific entities
        'Pressure': (17, 5),
        'Compressor': (17, 6),
        'HotWater': (13, 5),
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
    draw_control_flow(ax, 1, 10, 3, 10)      # Timer -> TimeTriggerBoundary
    draw_control_flow(ax, 1, 6, 3, 1.5)      # User -> HMIAdditiveInputBoundary (UC1)
    draw_control_flow(ax, 1, 6, 17, 10)      # User -> HMIBeverageSelectionBoundary (UC2)
    draw_control_flow(ax, 1, 3, 3, 4)        # ExternalTrigger -> WaterSupplyBoundary
    draw_control_flow(ax, 1, 3, 3, 3)        # ExternalTrigger -> CoffeeBeansSupplyBoundary
    draw_control_flow(ax, 1, 3, 3, 2)        # ExternalTrigger -> MilkSupplyBoundary
    draw_control_flow(ax, 1, 3, 3, 1)        # ExternalTrigger -> SugarSupplyBoundary
    
    # Boundary -> Controller connections
    draw_control_flow(ax, 3, 10, 5, 10)      # TimeTriggerBoundary -> TimeManager
    draw_control_flow(ax, 3, 1.5, 8, 8)      # HMIAdditiveInputBoundary -> HMIController
    draw_control_flow(ax, 17, 10, 8, 8)      # HMIBeverageSelectionBoundary -> HMIController
    draw_control_flow(ax, 3, 4, 4, 4)        # WaterSupplyBoundary -> WaterSupplyController
    draw_control_flow(ax, 3, 3, 4, 3)        # CoffeeBeansSupplyBoundary -> CoffeeBeansSupplyController
    draw_control_flow(ax, 3, 2, 4, 2)        # MilkSupplyBoundary -> MilkSupplyController
    draw_control_flow(ax, 3, 1, 4, 1)        # SugarSupplyBoundary -> SugarSupplyController
    
    # Domain Orchestrator coordinates all controllers
    draw_control_flow(ax, 5, 10, 10, 6)      # TimeManager -> DomainOrchestrator
    draw_control_flow(ax, 8, 8, 10, 6)       # HMIController -> DomainOrchestrator
    draw_control_flow(ax, 10, 6, 7, 9)       # DomainOrchestrator -> HeaterManager
    draw_control_flow(ax, 10, 6, 8, 9)       # DomainOrchestrator -> FilterManager
    draw_control_flow(ax, 10, 6, 5, 9)       # DomainOrchestrator -> AmountManager
    draw_control_flow(ax, 10, 6, 9, 9)       # DomainOrchestrator -> CupManager
    draw_control_flow(ax, 10, 6, 8, 6)       # DomainOrchestrator -> CoffeeManager
    draw_control_flow(ax, 10, 6, 5, 2)       # DomainOrchestrator -> MilkManager (UC1)
    draw_control_flow(ax, 10, 6, 5, 1)       # DomainOrchestrator -> SugarManager (UC1)
    draw_control_flow(ax, 10, 6, 15, 6)      # DomainOrchestrator -> CompressorManager (UC2)
    draw_control_flow(ax, 10, 6, 15, 5)      # DomainOrchestrator -> PressureManager (UC2)
    draw_control_flow(ax, 10, 6, 12, 8)      # DomainOrchestrator -> MessageManager
    draw_control_flow(ax, 10, 6, 4, 4)       # DomainOrchestrator -> WaterSupplyController
    draw_control_flow(ax, 10, 6, 4, 3)       # DomainOrchestrator -> CoffeeBeansSupplyController
    draw_control_flow(ax, 10, 6, 4, 2)       # DomainOrchestrator -> MilkSupplyController
    draw_control_flow(ax, 10, 6, 4, 1)       # DomainOrchestrator -> SugarSupplyController
    
    # Controller -> Boundary (outputs)
    draw_control_flow(ax, 8, 8, 10, 9)       # HMIController -> HMIStatusDisplayBoundary
    draw_control_flow(ax, 8, 8, 10, 8)       # HMIController -> HMIErrorDisplayBoundary
    draw_control_flow(ax, 9, 9, 10, 7)       # CupManager -> ProductDeliveryBoundary
    draw_control_flow(ax, 12, 8, 10, 9)      # MessageManager -> HMIStatusDisplayBoundary
    draw_control_flow(ax, 12, 8, 10, 8)      # MessageManager -> HMIErrorDisplayBoundary
    draw_control_flow(ax, 15, 6, 17, 8)      # CompressorManager -> EquipmentMaintenanceBoundary
    
    # === DATA FLOWS ===
    
    # USE RELATIONSHIPS (Blue dashed lines) - Controller uses Entity
    
    # Shared data flows
    draw_data_flow_use(ax, 8, 6, 6, 6)       # CoffeeManager --use--> GroundCoffee
    draw_data_flow_use(ax, 8, 6, 13, 5)      # CoffeeManager --use--> HotWater
    draw_data_flow_use(ax, 8, 6, 6, 9)       # CoffeeManager --use--> Filter
    draw_data_flow_use(ax, 8, 6, 19, 4)      # CoffeeManager --use--> Water
    draw_data_flow_use(ax, 8, 6, 11, 9)      # CoffeeManager --use--> Cup
    draw_data_flow_use(ax, 5, 9, 19, 3)      # AmountManager --use--> CoffeeBeans
    draw_data_flow_use(ax, 5, 9, 6, 9)       # AmountManager --use--> Filter
    draw_data_flow_use(ax, 7, 9, 19, 4)      # HeaterManager --use--> Water
    draw_data_flow_use(ax, 12, 8, 14, 8)     # MessageManager --use--> Message
    
    # UC1-specific data flows
    draw_data_flow_use(ax, 5, 2, 12, 6)      # MilkManager --use--> Coffee
    draw_data_flow_use(ax, 5, 2, 19, 2)      # MilkManager --use--> Milk
    draw_data_flow_use(ax, 5, 2, 11, 9)      # MilkManager --use--> Cup
    draw_data_flow_use(ax, 5, 1, 12, 6)      # SugarManager --use--> Coffee
    draw_data_flow_use(ax, 5, 1, 19, 1)      # SugarManager --use--> Sugar
    draw_data_flow_use(ax, 5, 1, 11, 9)      # SugarManager --use--> Cup
    
    # UC2-specific data flows (COMPRESSION FUNCTION)
    draw_data_flow_use(ax, 15, 6, 19, 4)     # CompressorManager --use--> Water
    draw_data_flow_use(ax, 15, 5, 17, 5)     # PressureManager --use--> Pressure
    draw_data_flow_use(ax, 15, 5, 17, 6)     # PressureManager --use--> Compressor
    
    # PROVIDE RELATIONSHIPS (Red dashed lines) - Controller provides Entity
    
    # Shared provide flows
    draw_data_flow_provide(ax, 7, 9, 13, 5)  # HeaterManager --provide--> HotWater
    draw_data_flow_provide(ax, 8, 9, 6, 9)   # FilterManager --provide--> Filter
    draw_data_flow_provide(ax, 5, 9, 6, 6)   # AmountManager --provide--> GroundCoffee
    draw_data_flow_provide(ax, 8, 6, 12, 6)  # CoffeeManager --provide--> Coffee
    draw_data_flow_provide(ax, 9, 9, 11, 9)  # CupManager --provide--> Cup
    draw_data_flow_provide(ax, 12, 8, 14, 8) # MessageManager --provide--> Message
    draw_data_flow_provide(ax, 12, 8, 14, 7) # MessageManager --provide--> Error
    
    # Supply controllers provide their entities
    draw_data_flow_provide(ax, 4, 4, 19, 4)  # WaterSupplyController --provide--> Water
    draw_data_flow_provide(ax, 4, 3, 19, 3)  # CoffeeBeansSupplyController --provide--> CoffeeBeans
    draw_data_flow_provide(ax, 4, 2, 19, 2)  # MilkSupplyController --provide--> Milk
    draw_data_flow_provide(ax, 4, 1, 19, 1)  # SugarSupplyController --provide--> Sugar
    
    # UC1-specific provide flows
    draw_data_flow_provide(ax, 5, 2, 12, 6)  # MilkManager --provide--> Coffee (enhanced)
    draw_data_flow_provide(ax, 5, 1, 12, 6)  # SugarManager --provide--> Coffee (enhanced)
    
    # UC2-specific provide flows (COMPRESSION FUNCTION)
    draw_data_flow_provide(ax, 15, 6, 17, 6) # CompressorManager --provide--> Compressor
    draw_data_flow_provide(ax, 15, 6, 17, 5) # CompressorManager --provide--> Pressure
    draw_data_flow_provide(ax, 15, 5, 13, 5) # PressureManager --provide--> HotWater (pressurized)
    
    # Add comprehensive legend
    legend_y = 1.5
    ax.text(0.5, legend_y+1, 'Legend:', fontsize=12, weight='bold')
    
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
    
    # Color coding legend
    ax.text(13, legend_y+0.5, 'Color Coding:', fontsize=10, weight='bold')
    # UC1 controller
    circle1 = Circle((13.5, legend_y), 0.08, facecolor='lightsteelblue', edgecolor='black')
    ax.add_patch(circle1)
    ax.text(14, legend_y, 'UC1 Controller', fontsize=8, va='center')
    # UC2 controller
    circle2 = Circle((13.5, legend_y-0.3), 0.08, facecolor='lightcoral', edgecolor='black')
    ax.add_patch(circle2)
    ax.text(14, legend_y-0.3, 'UC2 Controller', fontsize=8, va='center')
    # Shared controller
    circle3 = Circle((13.5, legend_y-0.6), 0.08, facecolor='lightgreen', edgecolor='black')
    ax.add_patch(circle3)
    ax.text(14, legend_y-0.6, 'Shared Controller', fontsize=8, va='center')
    
    # Add transformation annotations
    ax.text(7, 5.5, 'UC1: Coffee + Milk + Sugar\n-> Enhanced Coffee', 
            ha='center', va='center', fontsize=8, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.7))
    
    ax.text(16, 4, 'UC2: Water + Pressure\n-> Pressurized HotWater\n-> Espresso', 
            ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='mistyrose', alpha=0.7))
    
    # Highlight compression function
    ax.text(15, 7, 'COMPRESSION\nFUNCTION', ha='center', va='center', fontsize=10, weight='bold', color='red',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('D:\\KI\\RA-NLF\\UC1_UC2_Combined_RA_Diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('D:\\KI\\RA-NLF\\UC1_UC2_Combined_RA_Diagram.svg', bbox_inches='tight')
    
    print("Combined RA Diagram for UC1 + UC2 created successfully!")
    print("Files saved:")
    print("- UC1_UC2_Combined_RA_Diagram.png")
    print("- UC1_UC2_Combined_RA_Diagram.svg")
    
    print("\nDiagram Features:")
    print("- UC1-specific components: Light blue controllers, cyan entities")
    print("- UC2-specific components: Light coral controllers, pink entities")
    print("- Shared components: Light green controllers, yellow entities")
    print("- Compression function highlighted in UC2 section")
    print("- Complete control flows and data flows for both use cases")
    
    plt.close()

if __name__ == "__main__":
    create_uc1_uc2_combined_ra_diagram()