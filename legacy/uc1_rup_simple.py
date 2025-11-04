"""
UC1 RUP Analysis Class Diagram Generator (Simplified)
Creates proper RUP Analysis Class Diagram with standard UML symbols
"""

def generate_uc1_rup_svg():
    """Generate proper RUP Analysis Class Diagram as SVG"""
    
    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="1000" height="600" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600">
  <style>
    .actor { fill: none; stroke: #333; stroke-width: 2; }
    .actor-text { font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }
    .boundary { fill: #E6F3FF; stroke: #0066CC; stroke-width: 2; }
    .control { fill: #FFF2E6; stroke: #FF6600; stroke-width: 2; }
    .entity { fill: #F0FFF0; stroke: #009900; stroke-width: 2; }
    .class-text { font-family: Arial, sans-serif; font-size: 10px; text-anchor: middle; }
    .association { stroke: #333; stroke-width: 1; fill: none; }
    .title { font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-anchor: middle; }
  </style>
  
  <!-- Title -->
  <text x="500" y="25" class="title">UC1: Prepare Milk Coffee - RUP Analysis Class Diagram</text>
  
  <!-- Actors (Stick Figures) -->
  <!-- User Actor -->
  <g>
    <circle cx="80" cy="70" r="8" class="actor"/>
    <line x1="80" y1="78" x2="80" y2="100" class="actor"/>
    <line x1="70" y1="88" x2="90" y2="88" class="actor"/>
    <line x1="80" y1="100" x2="72" y2="115" class="actor"/>
    <line x1="80" y1="100" x2="88" y2="115" class="actor"/>
    <text x="80" y="130" class="actor-text">User</text>
  </g>
  
  <!-- Timer Actor -->
  <g>
    <circle cx="80" cy="180" r="8" class="actor"/>
    <line x1="80" y1="188" x2="80" y2="210" class="actor"/>
    <line x1="70" y1="198" x2="90" y2="198" class="actor"/>
    <line x1="80" y1="210" x2="72" y2="225" class="actor"/>
    <line x1="80" y1="210" x2="88" y2="225" class="actor"/>
    <text x="80" y="240" class="actor-text">Timer</text>
  </g>
  
  <!-- Input Boundary Classes (Circles) -->
  <!-- CoffeeBeans Input -->
  <circle cx="250" cy="100" r="25" class="boundary"/>
  <text x="250" y="135" class="class-text">CoffeeBeans Input</text>
  
  <!-- Water Input -->
  <circle cx="250" cy="180" r="25" class="boundary"/>
  <text x="250" y="215" class="class-text">Water Input</text>
  
  <!-- Milk Input -->
  <circle cx="250" cy="260" r="25" class="boundary"/>
  <text x="250" y="295" class="class-text">Milk Input</text>
  
  <!-- Control Classes (Circles with Arrow) -->
  <!-- GetraenkeOrchestrator -->
  <circle cx="450" cy="60" r="25" class="control"/>
  <polygon points="442,65 458,65 450,52" fill="#FF6600"/>
  <text x="450" y="95" class="class-text">GetraenkeOrchestrator</text>
  
  <!-- CoffeeBeansManager -->
  <circle cx="450" cy="120" r="25" class="control"/>
  <polygon points="442,125 458,125 450,112" fill="#FF6600"/>
  <text x="450" y="155" class="class-text">CoffeeBeansManager</text>
  
  <!-- WaterManager -->
  <circle cx="450" cy="180" r="25" class="control"/>
  <polygon points="442,185 458,185 450,172" fill="#FF6600"/>
  <text x="450" y="215" class="class-text">WaterManager</text>
  
  <!-- MilkManager -->
  <circle cx="450" cy="240" r="25" class="control"/>
  <polygon points="442,245 458,245 450,232" fill="#FF6600"/>
  <text x="450" y="275" class="class-text">MilkManager</text>
  
  <!-- Entity Classes (Circles with Underline) -->
  <!-- Kaffeebohnen -->
  <circle cx="650" cy="120" r="25" class="entity"/>
  <line x1="635" y1="128" x2="665" y2="128" stroke="#009900" stroke-width="3"/>
  <text x="650" y="155" class="class-text">Kaffeebohnen</text>
  
  <!-- Kaffeemehl -->
  <circle cx="750" cy="120" r="25" class="entity"/>
  <line x1="735" y1="128" x2="765" y2="128" stroke="#009900" stroke-width="3"/>
  <text x="750" y="155" class="class-text">Kaffeemehl</text>
  
  <!-- Wasser -->
  <circle cx="650" cy="180" r="25" class="entity"/>
  <line x1="635" y1="188" x2="665" y2="188" stroke="#009900" stroke-width="3"/>
  <text x="650" y="215" class="class-text">Wasser</text>
  
  <!-- Milch -->
  <circle cx="650" cy="240" r="25" class="entity"/>
  <line x1="635" y1="248" x2="665" y2="248" stroke="#009900" stroke-width="3"/>
  <text x="650" y="275" class="class-text">Milch</text>
  
  <!-- Output Boundary Classes -->
  <!-- Waste Output -->
  <circle cx="850" cy="100" r="25" class="boundary"/>
  <text x="850" y="135" class="class-text">Waste Output</text>
  
  <!-- User Presentation -->
  <circle cx="850" cy="260" r="25" class="boundary"/>
  <text x="850" y="295" class="class-text">User Presentation</text>
  
  <!-- Associations -->
  <!-- Actor to Boundaries -->
  <line x1="105" y1="85" x2="225" y2="100" class="association"/>
  <text x="165" y="90" class="class-text" font-size="9">B5</text>
  
  <line x1="105" y1="195" x2="225" y2="180" class="association"/>
  <text x="165" y="185" class="class-text" font-size="9">B1: 7:00h</text>
  
  <!-- Boundaries to Controls -->
  <line x1="275" y1="100" x2="425" y2="120" class="association"/>
  <line x1="275" y1="180" x2="425" y2="180" class="association"/>
  <line x1="275" y1="260" x2="425" y2="240" class="association"/>
  
  <!-- Orchestrator to Managers -->
  <line x1="450" y1="85" x2="450" y2="95" class="association"/>
  
  <!-- Controls to Entities -->
  <line x1="475" y1="120" x2="625" y2="120" class="association"/>
  <text x="550" y="115" class="class-text" font-size="9">uses</text>
  
  <line x1="475" y1="180" x2="625" y2="180" class="association"/>
  <text x="550" y="175" class="class-text" font-size="9">uses</text>
  
  <line x1="475" y1="240" x2="625" y2="240" class="association"/>
  <text x="550" y="235" class="class-text" font-size="9">uses</text>
  
  <!-- Entity Transformation -->
  <line x1="675" y1="120" x2="725" y2="120" class="association"/>
  <text x="700" y="115" class="class-text" font-size="9">grinding</text>
  
  <!-- Controls to Output -->
  <line x1="475" y1="120" x2="825" y2="100" class="association"/>
  <text x="650" y="105" class="class-text" font-size="9">waste</text>
  
  <line x1="475" y1="240" x2="825" y2="260" class="association"/>
  <text x="650" y="255" class="class-text" font-size="9">B4,B5</text>
  
  <!-- Output to Actor -->
  <line x1="825" y1="275" x2="105" y2="100" class="association"/>
  <text x="465" y="190" class="class-text" font-size="9">presents</text>
  
  <!-- UC Step Annotations -->
  <text x="350" y="50" class="class-text" font-size="11" fill="#666">B2a-d: Parallel Steps</text>
  <text x="550" y="50" class="class-text" font-size="11" fill="#666">B3a,B3b: Entity Usage</text>
  <text x="750" y="50" class="class-text" font-size="11" fill="#666">Transformations</text>
  
  <!-- Legend -->
  <text x="50" y="380" class="class-text" font-weight="bold" font-size="12">RUP Analysis Class Stereotypes:</text>
  
  <!-- Boundary Legend -->
  <circle cx="80" cy="400" r="8" class="boundary"/>
  <text x="100" y="405" class="class-text">Boundary Classes (Interface)</text>
  
  <!-- Control Legend -->
  <circle cx="80" cy="420" r="8" class="control"/>
  <polygon points="76,422 84,422 80,418" fill="#FF6600"/>
  <text x="100" y="425" class="class-text">Control Classes (Workflow)</text>
  
  <!-- Entity Legend -->
  <circle cx="80" cy="440" r="8" class="entity"/>
  <line x1="75" y1="443" x2="85" y2="443" stroke="#009900" stroke-width="2"/>
  <text x="100" y="445" class="class-text">Entity Classes (Data)</text>
  
  <!-- Actor Legend -->
  <g>
    <circle cx="80" cy="465" r="5" class="actor"/>
    <line x1="80" y1="470" x2="80" y2="478" class="actor"/>
    <line x1="76" y1="474" x2="84" y2="474" class="actor"/>
    <line x1="80" y1="478" x2="77" y2="483" class="actor"/>
    <line x1="80" y1="478" x2="83" y2="483" class="actor"/>
  </g>
  <text x="100" y="470" class="class-text">Actors (Users/External Systems)</text>
  
  <!-- Phase Summary -->
  <text x="400" y="400" class="class-text" font-size="10">Phase 1: Domain analysis, Actor classification, Temporal requirements</text>
  <text x="400" y="415" class="class-text" font-size="10">Phase 2: Resource analysis, Manager Controllers, Boundary Objects</text>
  <text x="400" y="430" class="class-text" font-size="10">Phase 3: Interaction analysis, UC flow mapping, Orchestration patterns</text>
  
  <!-- UC Flow Mapping -->
  <text x="50" y="520" class="class-text" font-weight="bold" font-size="12">UC Flow Mapping:</text>
  <text x="50" y="540" class="class-text" font-size="10">B1: Timer → ZeitManager (Trigger)</text>
  <text x="250" y="540" class="class-text" font-size="10">B2a-d: Parallel execution</text>
  <text x="400" y="540" class="class-text" font-size="10">B3a,B3b: Entity usage</text>
  <text x="550" y="540" class="class-text" font-size="10">B4,B5: User presentation</text>
  
</svg>'''
    
    # Save SVG
    output_file = "Zwischenprodukte/UC1_Coffee_RUP_Analysis_Diagram.svg"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"RUP Analysis Class Diagram generated successfully!")
    print(f"File saved to: {output_file}")
    print(f"This diagram uses proper RUP/UML symbols:")
    print(f"  ○ Circle = Boundary Classes")
    print(f"  ○▲ Circle with Arrow = Control Classes") 
    print(f"  ○_ Circle with Underline = Entity Classes")
    print(f"  Stick figures = Actors")
    
    return output_file

if __name__ == "__main__":
    generate_uc1_rup_svg()