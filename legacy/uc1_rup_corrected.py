"""
UC1 RUP Analysis Class Diagram Generator (CORRECTED)
Fixes Timer connection and implements correct coordination rules:
- Timer -> Zeit Boundary -> ZeitManager -> GeträenkeOrchestrator 
- GeträenkeOrchestrator is the main coordinator (not ZeitManager)
- Triggers like Timer are NOT coordinators
"""

def generate_uc1_rup_corrected():
    """Generate corrected RUP Analysis Class Diagram as SVG"""
    
    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="1100" height="650" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 650">
  <style>
    .actor { fill: none; stroke: #333; stroke-width: 2; }
    .actor-text { font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }
    .boundary { fill: #E6F3FF; stroke: #0066CC; stroke-width: 2; }
    .control { fill: #FFF2E6; stroke: #FF6600; stroke-width: 2; }
    .entity { fill: #F0FFF0; stroke: #009900; stroke-width: 2; }
    .class-text { font-family: Arial, sans-serif; font-size: 10px; text-anchor: middle; }
    .association { stroke: #333; stroke-width: 1; fill: none; }
    .coordination { stroke: #FF6600; stroke-width: 2; fill: none; }
    .title { font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-anchor: middle; }
    .rule-text { font-family: Arial, sans-serif; font-size: 11px; fill: #CC0000; font-weight: bold; }
  </style>
  
  <!-- Title -->
  <text x="550" y="25" class="title">UC1: Prepare Milk Coffee - RUP Analysis Class Diagram (CORRECTED)</text>
  
  <!-- Rule Annotation -->
  <text x="550" y="45" class="rule-text">Rule: Timer -> Zeit Boundary -> ZeitManager -> GeträenkeOrchestrator (Coordinator)</text>
  
  <!-- Actors (Stick Figures) -->
  <!-- User Actor -->
  <g>
    <circle cx="80" cy="150" r="8" class="actor"/>
    <line x1="80" y1="158" x2="80" y2="180" class="actor"/>
    <line x1="70" y1="168" x2="90" y2="168" class="actor"/>
    <line x1="80" y1="180" x2="72" y2="195" class="actor"/>
    <line x1="80" y1="180" x2="88" y2="195" class="actor"/>
    <text x="80" y="210" class="actor-text">User</text>
  </g>
  
  <!-- Timer Actor -->
  <g>
    <circle cx="80" cy="80" r="8" class="actor"/>
    <line x1="80" y1="88" x2="80" y2="110" class="actor"/>
    <line x1="70" y1="98" x2="90" y2="98" class="actor"/>
    <line x1="80" y1="110" x2="72" y2="125" class="actor"/>
    <line x1="80" y1="110" x2="88" y2="125" class="actor"/>
    <text x="80" y="140" class="actor-text">Timer</text>
  </g>
  
  <!-- CORRECTED: Zeit Boundary for Timer -->
  <circle cx="200" cy="80" r="25" class="boundary"/>
  <text x="200" y="115" class="class-text">Zeit Boundary</text>
  
  <!-- Input Boundary Classes (Circles) -->
  <!-- CoffeeBeans Input -->
  <circle cx="200" cy="160" r="25" class="boundary"/>
  <text x="200" y="195" class="class-text">CoffeeBeans Input</text>
  
  <!-- Water Input -->
  <circle cx="200" cy="220" r="25" class="boundary"/>
  <text x="200" y="255" class="class-text">Water Input</text>
  
  <!-- Milk Input -->
  <circle cx="200" cy="280" r="25" class="boundary"/>
  <text x="200" y="315" class="class-text">Milk Input</text>
  
  <!-- Control Classes (Circles with Arrow) -->
  <!-- CORRECTED: ZeitManager (NOT coordinator, just time handler) -->
  <circle cx="350" cy="80" r="25" class="control"/>
  <polygon points="342,85 358,85 350,72" fill="#FF6600"/>
  <text x="350" y="115" class="class-text">ZeitManager</text>
  
  <!-- GetraenkeOrchestrator (MAIN COORDINATOR) -->
  <circle cx="500" cy="80" r="30" class="control"/>
  <polygon points="490,87 510,87 500,70" fill="#FF6600"/>
  <text x="500" y="120" class="class-text">GeträenkeOrchestrator</text>
  <text x="500" y="135" class="rule-text">(COORDINATOR)</text>
  
  <!-- CoffeeBeansManager -->
  <circle cx="400" cy="180" r="25" class="control"/>
  <polygon points="392,185 408,185 400,172" fill="#FF6600"/>
  <text x="400" y="215" class="class-text">CoffeeBeansManager</text>
  
  <!-- WaterManager -->
  <circle cx="500" cy="180" r="25" class="control"/>
  <polygon points="492,185 508,185 500,172" fill="#FF6600"/>
  <text x="500" y="215" class="class-text">WaterManager</text>
  
  <!-- MilkManager -->
  <circle cx="600" cy="180" r="25" class="control"/>
  <polygon points="592,185 608,185 600,172" fill="#FF6600"/>
  <text x="600" y="215" class="class-text">MilkManager</text>
  
  <!-- Entity Classes (Circles with Underline) -->
  <!-- Kaffeebohnen -->
  <circle cx="400" cy="300" r="25" class="entity"/>
  <line x1="385" y1="308" x2="415" y2="308" stroke="#009900" stroke-width="3"/>
  <text x="400" y="335" class="class-text">Kaffeebohnen</text>
  
  <!-- Kaffeemehl (transformation) -->
  <circle cx="550" cy="300" r="25" class="entity"/>
  <line x1="535" y1="308" x2="565" y2="308" stroke="#009900" stroke-width="3"/>
  <text x="550" y="335" class="class-text">Kaffeemehl</text>
  
  <!-- Wasser -->
  <circle cx="500" cy="360" r="25" class="entity"/>
  <line x1="485" y1="368" x2="515" y2="368" stroke="#009900" stroke-width="3"/>
  <text x="500" y="395" class="class-text">Wasser</text>
  
  <!-- Milch -->
  <circle cx="600" cy="360" r="25" class="entity"/>
  <line x1="585" y1="368" x2="615" y2="368" stroke="#009900" stroke-width="3"/>
  <text x="600" y="395" class="class-text">Milch</text>
  
  <!-- Output Boundary Classes -->
  <!-- Waste Output -->
  <circle cx="750" cy="160" r="25" class="boundary"/>
  <text x="750" y="195" class="class-text">Waste Output</text>
  
  <!-- User Presentation -->
  <circle cx="750" cy="240" r="25" class="boundary"/>
  <text x="750" y="275" class="class-text">User Presentation</text>
  
  <!-- CORRECTED Associations -->
  <!-- CORRECT: Timer -> Zeit Boundary -->
  <line x1="105" y1="85" x2="175" y2="80" class="association"/>
  <text x="140" y="80" class="class-text" font-size="9">B1: 7:00h</text>
  
  <!-- CORRECT: Zeit Boundary -> ZeitManager -->
  <line x1="225" y1="80" x2="325" y2="80" class="association"/>
  <text x="275" y="75" class="class-text" font-size="9">time trigger</text>
  
  <!-- CORRECT: ZeitManager -> GeträenkeOrchestrator -->
  <line x1="375" y1="80" x2="470" y2="80" class="coordination"/>
  <text x="422" y="75" class="class-text" font-size="9">activates coordinator</text>
  
  <!-- User to User Presentation -->
  <line x1="105" y1="165" x2="725" y2="240" class="association"/>
  <text x="415" y="210" class="class-text" font-size="9">B5: receives cup</text>
  
  <!-- Input Boundaries to respective Controllers -->
  <line x1="225" y1="160" x2="375" y2="180" class="association"/>
  <line x1="225" y1="220" x2="475" y2="180" class="association"/>
  <line x1="225" y1="280" x2="575" y2="180" class="association"/>
  
  <!-- MAIN COORDINATION: GeträenkeOrchestrator -> All Managers -->
  <line x1="480" y1="100" x2="410" y2="160" class="coordination"/>
  <text x="440" y="135" class="class-text" font-size="9">coordinates</text>
  
  <line x1="500" y1="110" x2="500" y2="155" class="coordination"/>
  <text x="510" y="135" class="class-text" font-size="9">coordinates</text>
  
  <line x1="520" y1="100" x2="590" y2="160" class="coordination"/>
  <text x="560" y="135" class="class-text" font-size="9">coordinates</text>
  
  <!-- Controls to Entities -->
  <line x1="400" y1="205" x2="400" y2="275" class="association"/>
  <text x="410" y="240" class="class-text" font-size="9">uses</text>
  
  <line x1="500" y1="205" x2="500" y2="335" class="association"/>
  <text x="510" y="270" class="class-text" font-size="9">uses</text>
  
  <line x1="600" y1="205" x2="600" y2="335" class="association"/>
  <text x="610" y="270" class="class-text" font-size="9">uses</text>
  
  <!-- Entity Transformation -->
  <line x1="425" y1="300" x2="525" y2="300" class="association"/>
  <text x="475" y="295" class="class-text" font-size="9">B2c: grinding</text>
  
  <!-- Controls to Output -->
  <line x1="425" y1="180" x2="725" y2="160" class="association"/>
  <text x="575" y="165" class="class-text" font-size="9">waste</text>
  
  <line x1="580" y1="180" x2="725" y2="240" class="association"/>
  <text x="652" y="215" class="class-text" font-size="9">B4,B5</text>
  
  <!-- Legend -->
  <text x="50" y="450" class="class-text" font-weight="bold" font-size="12">RUP Analysis Class Stereotypes:</text>
  
  <!-- Boundary Legend -->
  <circle cx="80" cy="470" r="8" class="boundary"/>
  <text x="100" y="475" class="class-text">Boundary Classes (Interface)</text>
  
  <!-- Control Legend -->
  <circle cx="80" cy="490" r="8" class="control"/>
  <polygon points="76,492 84,492 80,488" fill="#FF6600"/>
  <text x="100" y="495" class="class-text">Control Classes (Workflow)</text>
  
  <!-- Entity Legend -->
  <circle cx="80" cy="510" r="8" class="entity"/>
  <line x1="75" y1="513" x2="85" y2="513" stroke="#009900" stroke-width="2"/>
  <text x="100" y="515" class="class-text">Entity Classes (Data)</text>
  
  <!-- Coordination Legend -->
  <line x1="80" y1="530" x2="95" y2="530" class="coordination"/>
  <text x="100" y="535" class="class-text">Coordination (Main Control Flow)</text>
  
  <!-- Rules Box -->
  <rect x="400" y="440" width="600" height="120" fill="#FFF8DC" stroke="#FF6600" stroke-width="2"/>
  <text x="410" y="460" class="rule-text" font-size="12">CORRECTED RUP Rules Applied:</text>
  <text x="410" y="480" class="class-text" font-size="10">1. Timer (Actor) -> Zeit Boundary -> ZeitManager (Control) -> GeträenkeOrchestrator (Coordinator)</text>
  <text x="410" y="495" class="class-text" font-size="10">2. GeträenkeOrchestrator is the MAIN COORDINATOR (orange lines)</text>
  <text x="410" y="510" class="class-text" font-size="10">3. ZeitManager is NOT a coordinator, just handles time trigger</text>
  <text x="410" y="525" class="class-text" font-size="10">4. Rule: Triggers like Timer are NOT coordinators</text>
  <text x="410" y="540" class="class-text" font-size="10">5. UC Steps: B1(Timer) -> B2a-d(Parallel) -> B3a-b(Sequential) -> B4-B5(Output)</text>
  
</svg>'''
    
    # Save SVG
    output_file = "Zwischenprodukte/UC1_Coffee_RUP_CORRECTED.svg"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"CORRECTED RUP Analysis Class Diagram generated!")
    print(f"File saved to: {output_file}")
    print(f"")
    print(f"CORRECTIONS APPLIED:")
    print(f"✓ Timer -> Zeit Boundary -> ZeitManager -> GetraenkeOrchestrator")
    print(f"✓ GetraenkeOrchestrator is the MAIN COORDINATOR")
    print(f"✓ ZeitManager is NOT a coordinator, just handles time")
    print(f"✓ Rule: Triggers like Timer are NOT coordinators")
    print(f"✓ Orange lines show coordination relationships")
    
    return output_file

if __name__ == "__main__":
    generate_uc1_rup_corrected()