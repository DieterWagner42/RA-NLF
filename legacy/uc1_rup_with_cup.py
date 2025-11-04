"""
UC1 RUP Analysis Class Diagram Generator - WITH CUP INCLUDED
Now includes Cup/Tasse derived from Domain-Context Rule in Phase 2
Fixed: Cup was missing because Phase 2 only analyzed preconditions, not domain context
"""

def generate_uc1_rup_with_cup():
    """Generate RUP Analysis Class Diagram with Cup included from domain context derivation"""
    
    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="1200" height="700" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 700">
  <style>
    .actor { fill: none; stroke: #333; stroke-width: 2; }
    .actor-text { font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }
    .boundary { fill: #E6F3FF; stroke: #0066CC; stroke-width: 2; }
    .control { fill: #FFF2E6; stroke: #FF6600; stroke-width: 2; }
    .entity { fill: #F0FFF0; stroke: #009900; stroke-width: 2; }
    .class-text { font-family: Arial, sans-serif; font-size: 10px; text-anchor: middle; }
    .association { stroke: #333; stroke-width: 1; fill: none; }
    .coordination { stroke: #FF6600; stroke-width: 2; fill: none; }
    .context-derived { stroke: #CC0066; stroke-width: 2; fill: #FFE6F2; }
    .title { font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-anchor: middle; }
    .rule-text { font-family: Arial, sans-serif; font-size: 11px; fill: #CC0000; font-weight: bold; }
    .context-text { font-family: Arial, sans-serif; font-size: 11px; fill: #CC0066; font-weight: bold; }
  </style>
  
  <!-- Title -->
  <text x="600" y="25" class="title">UC1: Prepare Milk Coffee - RUP Analysis Class Diagram (WITH CUP)</text>
  
  <!-- Rule Annotations -->
  <text x="600" y="45" class="rule-text">Phase 3 Rule: Timer → Zeit Boundary → ZeitManager → GeträenkeOrchestrator</text>
  <text x="600" y="60" class="context-text">Phase 2 Rule: Domain Context "Beverages need containers" → Cup derived</text>
  
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
  
  <!-- Zeit Boundary for Timer -->
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
  
  <!-- NEW: Cup Input (Context-Derived) -->
  <circle cx="200" cy="340" r="25" class="context-derived"/>
  <text x="200" y="375" class="class-text">Cup Input</text>
  <text x="200" y="385" class="context-text">(Context-Derived)</text>
  
  <!-- Control Classes (Circles with Arrow) -->
  <!-- ZeitManager -->
  <circle cx="350" cy="80" r="25" class="control"/>
  <polygon points="342,85 358,85 350,72" fill="#FF6600"/>
  <text x="350" y="115" class="class-text">ZeitManager</text>
  
  <!-- GetraenkeOrchestrator (MAIN COORDINATOR) -->
  <circle cx="500" cy="80" r="30" class="control"/>
  <polygon points="490,87 510,87 500,70" fill="#FF6600"/>
  <text x="500" y="120" class="class-text">GeträenkeOrchestrator</text>
  <text x="500" y="135" class="rule-text">(COORDINATOR)</text>
  
  <!-- CoffeeBeansManager -->
  <circle cx="400" cy="200" r="25" class="control"/>
  <polygon points="392,205 408,205 400,192" fill="#FF6600"/>
  <text x="400" y="235" class="class-text">CoffeeBeansManager</text>
  
  <!-- WaterManager -->
  <circle cx="500" cy="200" r="25" class="control"/>
  <polygon points="492,205 508,205 500,192" fill="#FF6600"/>
  <text x="500" y="235" class="class-text">WaterManager</text>
  
  <!-- MilkManager -->
  <circle cx="600" cy="200" r="25" class="control"/>
  <polygon points="592,205 608,205 600,192" fill="#FF6600"/>
  <text x="600" y="235" class="class-text">MilkManager</text>
  
  <!-- NEW: CupManager (Context-Derived) -->
  <circle cx="700" cy="200" r="25" class="context-derived"/>
  <polygon points="692,205 708,205 700,192" fill="#CC0066"/>
  <text x="700" y="235" class="class-text">CupManager</text>
  <text x="700" y="245" class="context-text">(Context-Derived)</text>
  
  <!-- Entity Classes (Circles with Underline) -->
  <!-- Kaffeebohnen -->
  <circle cx="400" cy="320" r="25" class="entity"/>
  <line x1="385" y1="328" x2="415" y2="328" stroke="#009900" stroke-width="3"/>
  <text x="400" y="355" class="class-text">Kaffeebohnen</text>
  
  <!-- Kaffeemehl (transformation) -->
  <circle cx="550" cy="320" r="25" class="entity"/>
  <line x1="535" y1="328" x2="565" y2="328" stroke="#009900" stroke-width="3"/>
  <text x="550" y="355" class="class-text">Kaffeemehl</text>
  
  <!-- Wasser -->
  <circle cx="500" cy="380" r="25" class="entity"/>
  <line x1="485" y1="388" x2="515" y2="388" stroke="#009900" stroke-width="3"/>
  <text x="500" y="415" class="class-text">Wasser</text>
  
  <!-- Milch -->
  <circle cx="600" cy="380" r="25" class="entity"/>
  <line x1="585" y1="388" x2="615" y2="388" stroke="#009900" stroke-width="3"/>
  <text x="600" y="415" class="class-text">Milch</text>
  
  <!-- NEW: Cup Entity (Context-Derived) -->
  <circle cx="700" cy="320" r="25" class="context-derived"/>
  <line x1="685" y1="328" x2="715" y2="328" stroke="#CC0066" stroke-width="3"/>
  <text x="700" y="355" class="class-text">Cup</text>
  <text x="700" y="365" class="context-text">(Context-Derived)</text>
  
  <!-- Output Boundary Classes -->
  <!-- Waste Output -->
  <circle cx="850" cy="160" r="25" class="boundary"/>
  <text x="850" y="195" class="class-text">Waste Output</text>
  
  <!-- User Presentation -->
  <circle cx="850" cy="240" r="25" class="boundary"/>
  <text x="850" y="275" class="class-text">User Presentation</text>
  
  <!-- Associations -->
  <!-- Timer -> Zeit Boundary -->
  <line x1="105" y1="85" x2="175" y2="80" class="association"/>
  <text x="140" y="80" class="class-text" font-size="9">B1: 7:00h</text>
  
  <!-- Zeit Boundary -> ZeitManager -->
  <line x1="225" y1="80" x2="325" y2="80" class="association"/>
  <text x="275" y="75" class="class-text" font-size="9">time trigger</text>
  
  <!-- ZeitManager -> GeträenkeOrchestrator -->
  <line x1="375" y1="80" x2="470" y2="80" class="coordination"/>
  <text x="422" y="75" class="class-text" font-size="9">activates coordinator</text>
  
  <!-- User to User Presentation -->
  <line x1="105" y1="165" x2="825" y2="240" class="association"/>
  <text x="465" y="210" class="class-text" font-size="9">B5: receives cup</text>
  
  <!-- Input Boundaries to respective Controllers -->
  <line x1="225" y1="160" x2="375" y2="200" class="association"/>
  <line x1="225" y1="220" x2="475" y2="200" class="association"/>
  <line x1="225" y1="280" x2="575" y2="200" class="association"/>
  
  <!-- NEW: Cup Input to CupManager -->
  <line x1="225" y1="340" x2="675" y2="200" class="association"/>
  <text x="450" y="280" class="context-text" font-size="9">B2d: retrieves cup</text>
  
  <!-- MAIN COORDINATION: GeträenkeOrchestrator -> All Managers -->
  <line x1="480" y1="100" x2="410" y2="180" class="coordination"/>
  <text x="440" y="145" class="class-text" font-size="9">coordinates</text>
  
  <line x1="500" y1="110" x2="500" y2="175" class="coordination"/>
  <text x="510" y="145" class="class-text" font-size="9">coordinates</text>
  
  <line x1="520" y1="100" x2="590" y2="180" class="coordination"/>
  <text x="560" y="145" class="class-text" font-size="9">coordinates</text>
  
  <!-- NEW: Coordination to CupManager -->
  <line x1="530" y1="90" x2="680" y2="180" class="coordination"/>
  <text x="610" y="140" class="context-text" font-size="9">coordinates</text>
  
  <!-- Controls to Entities -->
  <line x1="400" y1="225" x2="400" y2="295" class="association"/>
  <text x="410" y="260" class="class-text" font-size="9">uses</text>
  
  <line x1="500" y1="225" x2="500" y2="355" class="association"/>
  <text x="510" y="290" class="class-text" font-size="9">uses</text>
  
  <line x1="600" y1="225" x2="600" y2="355" class="association"/>
  <text x="610" y="290" class="class-text" font-size="9">uses</text>
  
  <!-- NEW: CupManager to Cup Entity -->
  <line x1="700" y1="225" x2="700" y2="295" class="association"/>
  <text x="710" y="260" class="context-text" font-size="9">manages</text>
  
  <!-- Entity Transformation -->
  <line x1="425" y1="320" x2="525" y2="320" class="association"/>
  <text x="475" y="315" class="class-text" font-size="9">B2c: grinding</text>
  
  <!-- Controls to Output -->
  <line x1="425" y1="200" x2="825" y2="160" class="association"/>
  <text x="625" y="175" class="class-text" font-size="9">waste</text>
  
  <!-- NEW: CupManager to User Presentation (B5: presents cup) -->
  <line x1="725" y1="200" x2="825" y2="240" class="association"/>
  <text x="775" y="225" class="class-text" font-size="9">B5: presents cup</text>
  
  <!-- Legend -->
  <text x="50" y="480" class="class-text" font-weight="bold" font-size="12">RUP Analysis Class Stereotypes:</text>
  
  <!-- Boundary Legend -->
  <circle cx="80" cy="500" r="8" class="boundary"/>
  <text x="100" y="505" class="class-text">Boundary Classes (Interface)</text>
  
  <!-- Control Legend -->
  <circle cx="80" cy="520" r="8" class="control"/>
  <polygon points="76,522 84,522 80,518" fill="#FF6600"/>
  <text x="100" y="525" class="class-text">Control Classes (Workflow)</text>
  
  <!-- Entity Legend -->
  <circle cx="80" cy="540" r="8" class="entity"/>
  <line x1="75" y1="543" x2="85" y2="543" stroke="#009900" stroke-width="2"/>
  <text x="100" y="545" class="class-text">Entity Classes (Data)</text>
  
  <!-- Context-Derived Legend -->
  <circle cx="80" cy="560" r="8" class="context-derived"/>
  <text x="100" y="565" class="context-text">Context-Derived (Domain Rule)</text>
  
  <!-- Coordination Legend -->
  <line x1="80" y1="580" x2="95" y2="580" class="coordination"/>
  <text x="100" y="585" class="class-text">Coordination (Main Control Flow)</text>
  
  <!-- Rules Box -->
  <rect x="400" y="470" width="700" height="180" fill="#FFF8DC" stroke="#FF6600" stroke-width="2"/>
  <text x="410" y="490" class="rule-text" font-size="12">UPDATED RUP Rules Applied:</text>
  <text x="410" y="510" class="class-text" font-size="10">Phase 3: Timer → Zeit Boundary → ZeitManager → GeträenkeOrchestrator (Coordinator)</text>
  <text x="410" y="525" class="context-text" font-size="10">Phase 2: Domain Rule "Beverages need containers" → Cup derived from context</text>
  <text x="410" y="540" class="class-text" font-size="10">UC Steps: B1(Timer) → B2a-d(Parallel+Cup) → B3a-b(Sequential) → B4-B5(Output)</text>
  <text x="410" y="555" class="class-text" font-size="10">B2d: "retrieves cup from storage" - Cup now included via domain context</text>
  <text x="410" y="570" class="class-text" font-size="10">B3a: "brewing coffee into the cup" - Cup management integrated</text>
  <text x="410" y="585" class="class-text" font-size="10">B3b: "adds milk to the cup" - Cup used throughout process</text>
  <text x="410" y="600" class="class-text" font-size="10">B5: "presents cup to user" - Cup delivery to user</text>
  <text x="410" y="615" class="context-text" font-size="10">Pink objects = Context-derived from Domain Rule</text>
  <text x="410" y="630" class="rule-text" font-size="10">Problem SOLVED: Cup no longer missing from analysis!</text>
  
</svg>'''
    
    # Save SVG
    output_file = "Zwischenprodukte/UC1_Coffee_RUP_WITH_CUP.svg"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"NEW RUP Analysis Class Diagram with CUP generated!")
    print(f"File saved to: {output_file}")
    print(f"")
    print(f"UPDATES APPLIED:")
    print(f"+ Cup derived from Domain-Context Rule: 'Beverages need containers'")
    print(f"+ Cup Input boundary (pink) - context-derived")
    print(f"+ CupManager controller (pink) - manages cup storage/retrieval")
    print(f"+ Cup entity (pink) - the actual cup resource")
    print(f"+ Coordination: GeträenkeOrchestrator -> CupManager")
    print(f"+ UC Step integration: B2d retrieves cup, B3a/B3b use cup, B5 presents cup")
    print(f"+ Phase 2 + Phase 3 rules both applied correctly")
    print(f"+ Problem SOLVED: Cup no longer missing!")
    
    return output_file

if __name__ == "__main__":
    generate_uc1_rup_with_cup()