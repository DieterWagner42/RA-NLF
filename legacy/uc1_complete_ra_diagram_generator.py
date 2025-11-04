"""
UC1 Vollständiges RUP Analysis Class Diagram Generator
Basierend auf der neuen Phase 3 vollständigen Analyse
ALLE gefundenen Controller und Entities werden dargestellt
"""

def generate_uc1_complete_ra_diagram():
    """Generate complete RUP diagram with all discovered controllers and entities from Phase 3"""
    
    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="1400" height="900" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 900">
  <style>
    .actor { fill: none; stroke: #333; stroke-width: 2; }
    .actor-text { font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }
    .boundary { fill: #E6F3FF; stroke: #0066CC; stroke-width: 2; }
    .control { fill: #FFF2E6; stroke: #FF6600; stroke-width: 2; }
    .entity { fill: #F0FFF0; stroke: #009900; stroke-width: 2; }
    .context-derived { stroke: #CC0066; stroke-width: 2; fill: #FFE6F2; }
    .class-text { font-family: Arial, sans-serif; font-size: 10px; text-anchor: middle; }
    .association { stroke: #333; stroke-width: 1; fill: none; }
    .use-relation { stroke: #0066CC; stroke-width: 2; fill: none; stroke-dasharray: 5,5; }
    .provide-relation { stroke: #009900; stroke-width: 2; fill: none; }
    .coordination { stroke: #FF6600; stroke-width: 2; fill: none; }
    .title { font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-anchor: middle; }
    .success-text { font-family: Arial, sans-serif; font-size: 11px; fill: #009900; font-weight: bold; }
    .step-text { font-family: Arial, sans-serif; font-size: 9px; fill: #666; }
  </style>
  
  <!-- Title -->
  <text x="700" y="25" class="title">UC1: Vollständiges RUP Analysis Class Diagram</text>
  <text x="700" y="45" class="success-text">ALLE UC-Schritte analysiert - Phase 3 Vollständig</text>
  
  <!-- Actors -->
  <!-- User -->
  <g>
    <circle cx="80" cy="120" r="8" class="actor"/>
    <line x1="80" y1="128" x2="80" y2="150" class="actor"/>
    <line x1="70" y1="138" x2="90" y2="138" class="actor"/>
    <line x1="80" y1="150" x2="72" y2="165" class="actor"/>
    <line x1="80" y1="150" x2="88" y2="165" class="actor"/>
    <text x="80" y="180" class="actor-text">User</text>
  </g>
  
  <!-- Timer -->
  <g>
    <circle cx="80" cy="80" r="8" class="actor"/>
    <line x1="80" y1="88" x2="80" y2="110" class="actor"/>
    <line x1="70" y1="98" x2="90" y2="98" class="actor"/>
    <line x1="80" y1="110" x2="72" y2="125" class="actor"/>
    <line x1="80" y1="110" x2="88" y2="125" class="actor"/>
    <text x="80" y="40" class="actor-text">Timer</text>
  </g>
  
  <!-- Boundaries -->
  <!-- Zeit Boundary -->
  <rect x="180" y="70" width="80" height="40" class="boundary"/>
  <text x="220" y="85" class="class-text">«boundary»</text>
  <text x="220" y="100" class="class-text">Zeit</text>
  
  <!-- User Input Boundary -->
  <rect x="180" y="130" width="80" height="40" class="boundary"/>
  <text x="220" y="145" class="class-text">«boundary»</text>
  <text x="220" y="160" class="class-text">User Input</text>
  
  <!-- User Output Boundary -->
  <rect x="1180" y="200" width="80" height="40" class="boundary"/>
  <text x="1220" y="215" class="class-text">«boundary»</text>
  <text x="1220" y="230" class="class-text">User Output</text>
  
  <!-- Controllers - ALLE VON PHASE 3 GEFUNDENEN -->
  <!-- B1: ZeitManager -->
  <circle cx="320" cy="90" r="15" class="control"/>
  <polygon points="315,93 325,93 320,87" fill="#FF6600"/>
  <text x="320" y="115" class="class-text">ZeitManager</text>
  <text x="320" y="127" class="step-text">B1</text>
  
  <!-- B2a: WaterIsManager -->
  <circle cx="450" cy="150" r="15" class="control"/>
  <polygon points="445,153 455,153 450,147" fill="#FF6600"/>
  <text x="450" y="175" class="class-text">WaterIsManager</text>
  <text x="450" y="187" class="step-text">B2a</text>
  
  <!-- B2b: FilterManager (JETZT GEFUNDEN!) -->
  <circle cx="600" cy="150" r="15" class="control"/>
  <polygon points="595,153 605,153 600,147" fill="#FF6600"/>
  <text x="600" y="175" class="class-text">FilterManager</text>
  <text x="600" y="187" class="step-text">B2b</text>
  
  <!-- B2c: CoffeeBeansAreManager -->
  <circle cx="750" cy="150" r="15" class="control"/>
  <polygon points="745,153 755,153 750,147" fill="#FF6600"/>
  <text x="750" y="175" class="class-text">CoffeeBeansAreManager</text>
  <text x="750" y="187" class="step-text">B2c</text>
  
  <!-- B2d: CupManager -->
  <circle cx="900" cy="150" r="15" class="control"/>
  <polygon points="895,153 905,153 900,147" fill="#FF6600"/>
  <text x="900" y="175" class="class-text">CupManager</text>
  <text x="900" y="187" class="step-text">B2d</text>
  
  <!-- B3b: MilkIsManager -->
  <circle cx="750" cy="300" r="15" class="control"/>
  <polygon points="745,303 755,303 750,297" fill="#FF6600"/>
  <text x="750" y="325" class="class-text">MilkIsManager</text>
  <text x="750" y="337" class="step-text">B3b</text>
  
  <!-- B4: UserNotificationManager -->
  <circle cx="1050" cy="200" r="15" class="control"/>
  <polygon points="1045,203 1055,203 1050,197" fill="#FF6600"/>
  <text x="1050" y="225" class="class-text">UserNotificationManager</text>
  <text x="1050" y="237" class="step-text">B4</text>
  
  <!-- B5: UserDeliveryManager -->
  <circle cx="1050" cy="300" r="15" class="control"/>
  <polygon points="1045,303 1055,303 1050,297" fill="#FF6600"/>
  <text x="1050" y="325" class="class-text">UserDeliveryManager</text>
  <text x="1050" y="337" class="step-text">B5</text>
  
  <!-- GetränkeOrchestrator (Coordination) -->
  <circle cx="600" cy="250" r="20" class="control"/>
  <polygon points="593,255 607,255 600,245" fill="#FF6600"/>
  <text x="600" y="280" class="class-text">GetränkeOrchestrator</text>
  <text x="600" y="292" class="step-text">Coordination</text>
  
  <!-- Entities - ALLE VON PHASE 3 GEFUNDENEN -->
  <!-- B1: Zeit Entity -->
  <rect x="300" y="400" width="80" height="40" class="entity"/>
  <text x="340" y="415" class="class-text">«entity»</text>
  <text x="340" y="430" class="class-text">Zeit</text>
  
  <!-- B2a: Water + Heater -->
  <rect x="420" y="400" width="80" height="40" class="entity"/>
  <text x="460" y="415" class="class-text">«entity»</text>
  <text x="460" y="430" class="class-text">Water</text>
  
  <rect x="420" y="460" width="80" height="40" class="entity"/>
  <text x="460" y="475" class="class-text">«entity»</text>
  <text x="460" y="490" class="class-text">Heater</text>
  
  <!-- B2b: Filter (JETZT GEFUNDEN!) -->
  <rect x="580" y="400" width="80" height="40" class="entity"/>
  <text x="620" y="415" class="class-text">«entity»</text>
  <text x="620" y="430" class="class-text">Filter</text>
  
  <!-- B2c: CoffeeBeans + GroundCoffee -->
  <rect x="720" y="400" width="80" height="40" class="entity"/>
  <text x="760" y="415" class="class-text">«entity»</text>
  <text x="760" y="430" class="class-text">CoffeeBeans</text>
  
  <rect x="720" y="460" width="80" height="40" class="entity"/>
  <text x="760" y="475" class="class-text">«entity»</text>
  <text x="760" y="490" class="class-text">GroundCoffee</text>
  
  <!-- B2d: Cup + Storage -->
  <rect x="880" y="400" width="80" height="40" class="entity"/>
  <text x="920" y="415" class="class-text">«entity»</text>
  <text x="920" y="430" class="class-text">Cup</text>
  
  <rect x="880" y="460" width="80" height="40" class="entity"/>
  <text x="920" y="475" class="class-text">«entity»</text>
  <text x="920" y="490" class="class-text">Storage</text>
  
  <!-- B3a: Coffee -->
  <rect x="600" y="520" width="80" height="40" class="entity"/>
  <text x="640" y="535" class="class-text">«entity»</text>
  <text x="640" y="550" class="class-text">Coffee</text>
  
  <!-- B3b: Milk -->
  <rect x="1020" y="400" width="80" height="40" class="entity"/>
  <text x="1060" y="415" class="class-text">«entity»</text>
  <text x="1060" y="430" class="class-text">Milk</text>
  
  <!-- RUP Compliant Relations: Controller -> use/provide -> Entity -->
  
  <!-- ZeitManager -> use -> Zeit -->
  <line x1="320" y1="105" x2="340" y2="400" class="use-relation"/>
  <text x="330" y="250" class="class-text" font-size="9">«use»</text>
  
  <!-- WaterIsManager -> use -> Water -->
  <line x1="450" y1="165" x2="460" y2="400" class="use-relation"/>
  <text x="455" y="280" class="class-text" font-size="9">«use»</text>
  
  <!-- WaterIsManager -> use -> Heater -->
  <line x1="450" y1="165" x2="460" y2="460" class="use-relation"/>
  <text x="455" y="310" class="class-text" font-size="9">«use»</text>
  
  <!-- FilterManager -> use -> Filter -->
  <line x1="600" y1="165" x2="620" y2="400" class="use-relation"/>
  <text x="610" y="280" class="class-text" font-size="9">«use»</text>
  
  <!-- CoffeeBeansAreManager -> use -> CoffeeBeans -->
  <line x1="750" y1="165" x2="760" y2="400" class="use-relation"/>
  <text x="755" y="280" class="class-text" font-size="9">«use»</text>
  
  <!-- CoffeeBeansAreManager -> provide -> GroundCoffee -->
  <line x1="750" y1="165" x2="760" y2="460" class="provide-relation"/>
  <text x="755" y="310" class="success-text" font-size="9">«provide»</text>
  
  <!-- CupManager -> use -> Cup -->
  <line x1="900" y1="165" x2="920" y2="400" class="use-relation"/>
  <text x="910" y="280" class="class-text" font-size="9">«use»</text>
  
  <!-- CupManager -> use -> Storage -->
  <line x1="900" y1="165" x2="920" y2="460" class="use-relation"/>
  <text x="910" y="310" class="class-text" font-size="9">«use»</text>
  
  <!-- MilkIsManager -> use -> Milk -->
  <line x1="750" y1="315" x2="1020" y2="420" class="use-relation"/>
  <text x="880" y="370" class="class-text" font-size="9">«use»</text>
  
  <!-- Multiple Controllers -> provide -> Coffee (B3a brewing result) -->
  <line x1="750" y1="180" x2="640" y2="520" class="provide-relation"/>
  <text x="690" y="350" class="success-text" font-size="9">«provide»</text>
  <line x1="450" y1="180" x2="620" y2="520" class="provide-relation"/>
  <line x1="900" y1="180" x2="660" y2="520" class="provide-relation"/>
  
  <!-- Actor-Boundary Relations -->
  <line x1="88" y1="80" x2="180" y2="90" class="association"/>
  <line x1="88" y1="120" x2="180" y2="150" class="association"/>
  
  <!-- Boundary-Controller Relations -->
  <line x1="260" y1="90" x2="305" y2="90" class="association"/>
  <line x1="260" y1="150" x2="300" y2="150" class="association"/>
  
  <!-- Controller Control Flow (Sequential) -->
  <line x1="335" y1="90" x2="435" y2="150" class="coordination"/>
  <line x1="465" y1="150" x2="585" y2="150" class="coordination"/>
  <line x1="615" y1="150" x2="735" y2="150" class="coordination"/>
  <line x1="765" y1="150" x2="885" y2="150" class="coordination"/>
  
  <!-- Orchestration to Managed Controllers -->
  <line x1="580" y1="250" x2="450" y2="165" class="coordination"/>
  <line x1="600" y1="235" x2="600" y2="165" class="coordination"/>
  <line x1="620" y1="250" x2="750" y2="165" class="coordination"/>
  <line x1="620" y1="250" x2="900" y2="165" class="coordination"/>
  <line x1="600" y1="270" x2="750" y2="285" class="coordination"/>
  
  <!-- Output Controllers to Boundary -->
  <line x1="1065" y1="200" x2="1180" y2="220" class="provide-relation"/>
  <text x="1120" y="210" class="success-text" font-size="9">«provide»</text>
  <line x1="1065" y1="300" x2="1200" y2="240" class="provide-relation"/>
  
  <!-- Success Labels -->
  <text x="700" y="680" class="success-text">✓ B1: ZeitManager → Zeit</text>
  <text x="700" y="700" class="success-text">✓ B2a: WaterIsManager → Water, Heater</text>
  <text x="700" y="720" class="success-text">✓ B2b: FilterManager → Filter (JETZT GEFUNDEN!)</text>
  <text x="700" y="740" class="success-text">✓ B2c: CoffeeBeansAreManager → CoffeeBeans, GroundCoffee</text>
  <text x="700" y="760" class="success-text">✓ B2d: CupManager → Cup, Storage</text>
  <text x="700" y="780" class="success-text">✓ B3a: Multi-Controller → Coffee (brewing)</text>
  <text x="700" y="800" class="success-text">✓ B3b: MilkIsManager → Milk</text>
  <text x="700" y="820" class="success-text">✓ B4: UserNotificationManager, B5: UserDeliveryManager</text>
  
  <text x="700" y="850" class="error-text">KEINE Entity-Entity Beziehungen! Nur Controller → Entity!</text>
  
</svg>'''
    
    return svg_content

if __name__ == "__main__":
    svg_content = generate_uc1_complete_ra_diagram()
    
    # Save to file
    with open("UC1_Complete_RA_Diagram.html", "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>UC1 Vollständiges RUP Analysis Class Diagram</title>
    <meta charset="UTF-8">
</head>
<body style="margin: 0; padding: 20px; background-color: #f5f5f5;">
    <h1>UC1: Vollständiges RUP Analysis Class Diagram</h1>
    <p><strong>Phase 3 Vollständige Analyse:</strong> ALLE UC-Schritte analysiert und RA-Klassen gefunden!</p>
    <div style="border: 2px solid #333; background-color: white; display: inline-block;">
        {svg_content}
    </div>
    <br><br>
    <div style="background-color: #e8f5e8; padding: 15px; border-left: 5px solid #4CAF50;">
        <h3>Verbesserungen in Phase 3:</h3>
        <ul>
            <li><strong>B2b:</strong> FilterManager + Filter Entity (wurde vorher ignoriert!)</li>
            <li><strong>B2c:</strong> CoffeeBeansAreManager + CoffeeBeans/GroundCoffee Entities</li>
            <li><strong>Alle UC-Schritte:</strong> Vollständige Systemverhalten-Analyse</li>
            <li><strong>RUP-konform:</strong> Nur Controller→Entity Beziehungen</li>
        </ul>
    </div>
</body>
</html>""")
    
    print("UC1 Vollstaendiges RA-Diagramm generiert: UC1_Complete_RA_Diagram.html")
    print("ALLE UC-Schritte analysiert - Phase 3 Vollstaendig!")