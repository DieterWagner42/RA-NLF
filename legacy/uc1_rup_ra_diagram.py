"""
UC1 RUP Analysis Class Diagram Generator
Creates proper RUP Analysis Class Diagram with standard UML symbols
Following the RA example.png style with correct Boundary, Control, Entity symbols
"""

import json
import xml.etree.ElementTree as ET

def generate_uc1_rup_diagram():
    """Generate proper RUP Analysis Class Diagram as SVG"""
    
    # Load analysis results
    with open("Zwischenprodukte/UC1_Coffee_phase2_analysis.json", "r", encoding="utf-8") as f:
        phase2_data = json.load(f)
    
    with open("Zwischenprodukte/UC1_Coffee_phase3_analysis.json", "r", encoding="utf-8") as f:
        phase3_data = json.load(f)
    
    # SVG dimensions
    width = 1200
    height = 800
    
    # Create SVG root
    svg = ET.Element("svg", {
        "width": str(width),
        "height": str(height),
        "xmlns": "http://www.w3.org/2000/svg",
        "viewBox": f"0 0 {width} {height}"
    })
    
    # Add styles
    style = ET.SubElement(svg, "style")
    style.text = """
        .actor { fill: none; stroke: #333; stroke-width: 2; }
        .actor-text { font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }
        .boundary { fill: #E6F3FF; stroke: #0066CC; stroke-width: 2; }
        .control { fill: #FFF2E6; stroke: #FF6600; stroke-width: 2; }
        .entity { fill: #F0FFF0; stroke: #009900; stroke-width: 2; }
        .class-text { font-family: Arial, sans-serif; font-size: 11px; text-anchor: middle; }
        .association { stroke: #333; stroke-width: 1; fill: none; }
        .title { font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-anchor: middle; }
    """
    
    # Title
    title = ET.SubElement(svg, "text", {
        "x": str(width//2), "y": "25", "class": "title"
    })
    title.text = "UC1: Prepare Milk Coffee - RUP Analysis Class Diagram"
    
    # Helper function to create actor (stick figure)
    def create_actor(x, y, name):
        g = ET.SubElement(svg, "g")
        # Head
        ET.SubElement(g, "circle", {
            "cx": str(x), "cy": str(y), "r": "8", "class": "actor"
        })
        # Body
        ET.SubElement(g, "line", {
            "x1": str(x), "y1": str(y+8), "x2": str(x), "y2": str(y+30), "class": "actor"
        })
        # Arms
        ET.SubElement(g, "line", {
            "x1": str(x-10), "y1": str(y+18), "x2": str(x+10), "y2": str(y+18), "class": "actor"
        })
        # Legs
        ET.SubElement(g, "line", {
            "x1": str(x), "y1": str(y+30), "x2": str(x-8), "y2": str(y+45), "class": "actor"
        })
        ET.SubElement(g, "line", {
            "x1": str(x), "y1": str(y+30), "x2": str(x+8), "y2": str(y+45), "class": "actor"
        })
        # Name
        ET.SubElement(g, "text", {
            "x": str(x), "y": str(y+60), "class": "actor-text"
        }).text = name
        return g
    
    # Helper function to create boundary class (circle)
    def create_boundary(x, y, name):
        g = ET.SubElement(svg, "g")
        # Circle
        ET.SubElement(g, "circle", {
            "cx": str(x), "cy": str(y), "r": "25", "class": "boundary"
        })
        # Name
        ET.SubElement(g, "text", {
            "x": str(x), "y": str(y+35), "class": "class-text"
        }).text = name
        return g
    
    # Helper function to create control class (circle with arrow)
    def create_control(x, y, name):
        g = ET.SubElement(svg, "g")
        # Circle
        ET.SubElement(g, "circle", {
            "cx": str(x), "cy": str(y), "r": "25", "class": "control"
        })
        # Arrow inside circle
        ET.SubElement(g, "polygon", {
            "points": f"{x-8},{y+5} {x+8},{y+5} {x},{y-8}",
            "fill": "#FF6600"
        })
        # Name
        ET.SubElement(g, "text", {
            "x": str(x), "y": str(y+35), "class": "class-text"
        }).text = name
        return g
    
    # Helper function to create entity class (circle with underline)
    def create_entity(x, y, name):
        g = ET.SubElement(svg, "g")
        # Circle
        ET.SubElement(g, "circle", {
            "cx": str(x), "cy": str(y), "r": "25", "class": "entity"
        })
        # Underline inside circle
        ET.SubElement(g, "line", {
            "x1": str(x-15), "y1": str(y+8), "x2": str(x+15), "y2": str(y+8),
            "stroke": "#009900", "stroke-width": "3"
        })
        # Name
        ET.SubElement(g, "text", {
            "x": str(x), "y": str(y+35), "class": "class-text"
        }).text = name
        return g
    
    # Helper function to create association line
    def create_association(x1, y1, x2, y2, label=""):
        g = ET.SubElement(svg, "g")
        ET.SubElement(g, "line", {
            "x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2), "class": "association"
        })
        if label:
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            ET.SubElement(g, "text", {
                "x": str(mid_x), "y": str(mid_y-5), "class": "class-text", "font-size": "10"
            }).text = label
        return g
    
    # Actor positions
    actors = [
        (100, 80, "User"),
        (100, 200, "Timer")
    ]
    
    # Create actors
    for x, y, name in actors:
        create_actor(x, y, name)
    
    # Boundary positions (Input)
    boundaries_input = [
        (300, 100, "CoffeeBeans\nInput"),
        (300, 200, "Water\nInput"), 
        (300, 300, "Milk\nInput")
    ]
    
    # Create input boundaries
    for x, y, name in boundaries_input:
        create_boundary(x, y, name)
    
    # Control positions (Manager Controllers)
    controls = [
        (500, 120, "CoffeeBeansManager"),
        (500, 200, "WaterManager"),
        (500, 280, "MilkManager"),
        (500, 60, "GetraenkeOrchestrator")
    ]
    
    # Create control classes
    for x, y, name in controls:
        create_control(x, y, name)
    
    # Entity positions
    entities = [
        (700, 120, "Kaffeebohnen"),
        (850, 120, "Kaffeemehl"),
        (700, 200, "Wasser"),
        (700, 280, "Milch")
    ]
    
    # Create entity classes
    for x, y, name in entities:
        create_entity(x, y, name)
    
    # Output boundaries
    boundaries_output = [
        (950, 100, "CoffeeBeans\nWaste Output"),
        (950, 300, "User\nPresentation")
    ]
    
    # Create output boundaries
    for x, y, name in boundaries_output:
        create_boundary(x, y, name)
    
    # Create associations
    associations = [
        # Actor to Input Boundaries
        (140, 95, 275, 100, "trigger"),
        (140, 215, 275, 200, "B1: 7:00h"),
        
        # Input Boundaries to Controllers
        (325, 100, 475, 120, ""),
        (325, 200, 475, 200, ""), 
        (325, 300, 475, 280, ""),
        
        # Orchestrator to Managers
        (525, 85, 525, 95, "coordinates"),
        
        # Controllers to Entities
        (525, 120, 675, 120, "uses"),
        (525, 200, 675, 200, "uses"),
        (525, 280, 675, 280, "uses"),
        
        # Entity transformations
        (725, 120, 825, 120, "grinding"),
        
        # Controllers to Output Boundaries
        (525, 120, 925, 100, "waste"),
        (525, 280, 925, 300, "B4,B5"),
        
        # Output to Actor
        (925, 315, 140, 110, "presents")
    ]
    
    # Create association lines
    for x1, y1, x2, y2, label in associations:
        create_association(x1, y1, x2, y2, label)
    
    # Add UC Step annotations
    annotations = [
        (400, 50, "B2a-d: Parallel Steps", "font-size: 10; fill: #666;"),
        (650, 50, "B3a,B3b: Entity Usage", "font-size: 10; fill: #666;"),
        (850, 50, "Transformations", "font-size: 10; fill: #666;")
    ]
    
    for x, y, text, style in annotations:
        ET.SubElement(svg, "text", {
            "x": str(x), "y": str(y), "style": style
        }).text = text
    
    # Add legend
    legend_y = height - 120
    ET.SubElement(svg, "text", {
        "x": "50", "y": str(legend_y), "class": "class-text", "font-weight": "bold"
    }).text = "RUP Analysis Class Stereotypes:"
    
    # Legend items
    legend_items = [
        (80, legend_y + 20, "boundary", "Boundary Classes (Interface)"),
        (80, legend_y + 40, "control", "Control Classes (Workflow)"),
        (80, legend_y + 60, "entity", "Entity Classes (Data)")
    ]
    
    for x, y, class_type, description in legend_items:
        if class_type == "boundary":
            ET.SubElement(svg, "circle", {"cx": str(x), "cy": str(y), "r": "8", "class": "boundary"})
        elif class_type == "control":
            g = ET.SubElement(svg, "g")
            ET.SubElement(g, "circle", {"cx": str(x), "cy": str(y), "r": "8", "class": "control"})
            ET.SubElement(g, "polygon", {
                "points": f"{x-3},{y+2} {x+3},{y+2} {x},{y-3}",
                "fill": "#FF6600"
            })
        else:  # entity
            g = ET.SubElement(svg, "g")
            ET.SubElement(g, "circle", {"cx": str(x), "cy": str(y), "r": "8", "class": "entity"})
            ET.SubElement(g, "line", {
                "x1": str(x-5), "y1": str(y+3), "x2": str(x+5), "y2": str(y+3),
                "stroke": "#009900", "stroke-width": "2"
            })
        
        ET.SubElement(svg, "text", {
            "x": str(x+20), "y": str(y+4), "class": "class-text"
        }).text = description
    
    # Add phase information
    phase_info = [
        f"Phase 1: {phase3_data['phase1_summary'][:80]}...",
        f"Phase 2: {phase3_data['phase2_summary'][:80]}...", 
        f"Phase 3: {phase3_data['phase3_summary'][:80]}..."
    ]
    
    for i, info in enumerate(phase_info):
        ET.SubElement(svg, "text", {
            "x": "400", "y": str(legend_y + 20 + i*15), "class": "class-text", "font-size": "9"
        }).text = info
    
    # Save SVG
    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ", level=0)
    
    output_file = "Zwischenprodukte/UC1_Coffee_RUP_Analysis_Diagram.svg"
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    
    print(f"RUP Analysis Class Diagram generated successfully!")
    print(f"File saved to: {output_file}")
    print(f"This diagram uses proper RUP/UML symbols:")
    print(f"  ○ Circle = Boundary Classes")
    print(f"  ○▲ Circle with Arrow = Control Classes") 
    print(f"  ○_ Circle with Underline = Entity Classes")
    print(f"  Stick figures = Actors")
    
    return output_file

if __name__ == "__main__":
    generate_uc1_rup_diagram()