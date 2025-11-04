"""
UC1 Korrekte RUP Analysis Class Diagram Generator
KORREKTUR: Keine Beziehungen zwischen Entities!
RUP-Regel: Controller -> use -> Entity; Controller -> provide -> Entity
"""

def generate_uc1_correct_rup_diagram():
    """Generate corrected RUP diagram without Entity-Entity relationships"""
    
    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="1200" height="750" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 750">
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
    .error-text { font-family: Arial, sans-serif; font-size: 11px; fill: #CC0000; font-weight: bold; }
  </style>
  
  <!-- Title -->
  <text x="600" y="25" class="title">UC1: Korrekte RUP Analysis Class Diagram</text>
  <text x="600" y="45" class="success-text">KEINE Entity-Entity Beziehungen! Nur Controller-Entity!</text>
  
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
    <text x="80" y="140" class="actor-text">Timer</text>
  </g>
  
  <!-- Boundaries -->
  <circle cx="180" cy="80" r="20" class="boundary"/>
  <text x="180" y="110" class="class-text">Zeit</text>
  
  <circle cx="180" cy="150" r="20" class="boundary"/>
  <text x="180" y="180" class="class-text">CoffeeBeans Input</text>
  
  <circle cx="180" cy="200" r="20" class="boundary"/>
  <text x="180" y="230" class="class-text">Water Input</text>
  
  <circle cx="180" cy="250" r="20" class="boundary"/>
  <text x="180" y="280" class="class-text">Milk Input</text>
  
  <circle cx="180" cy="300" r="20" class="context-derived"/>
  <text x="180" y="330" class="class-text">Cup Input</text>
  <text x="180" y="340" class="error-text">(Context)</text>
  
  <circle cx="950" cy="200" r="20" class="boundary"/>
  <text x="950" y="230" class="class-text">User Output</text>
  
  <!-- Controllers -->
  <!-- ZeitManager -->
  <circle cx="300" cy="80" r="20" class="control"/>
  <polygon points="295,83 305,83 300,77" fill="#FF6600"/>
  <text x="300" y="110" class="class-text">ZeitManager</text>
  
  <!-- GetraenkeOrchestrator -->
  <circle cx="500" cy="80" r="25" class="control"/>
  <polygon points="492,85 508,85 500,72" fill="#FF6600"/>
  <text x="500" y="115" class="class-text">GetraenkeOrchestrator</text>
  <text x="500" y="125" class="success-text">(Coordinator)</text>
  
  <!-- Manager Controllers -->
  <circle cx="400" cy="150" r="20" class="control"/>
  <polygon points="395,153 405,153 400,147" fill="#FF6600"/>
  <text x="400" y="180" class="class-text">CoffeeBeansManager</text>
  
  <circle cx="400" cy="200" r="20" class="control"/>
  <polygon points="395,203 405,203 400,197" fill="#FF6600"/>
  <text x="400" y="230" class="class-text">WaterManager</text>
  
  <circle cx="400" cy="250" r="20" class="control"/>
  <polygon points="395,253 405,253 400,247" fill="#FF6600"/>
  <text x="400" y="280" class="class-text">MilkManager</text>
  
  <circle cx="400" cy="300" r="20" class="context-derived"/>
  <polygon points="395,303 405,303 400,297" fill="#CC0066"/>
  <text x="400" y="330" class="class-text">CupManager</text>
  
  <!-- Input Entities -->
  <circle cx="600" cy="150" r="15" class="entity"/>
  <line x1="590" y1="155" x2="610" y2="155" stroke="#009900" stroke-width="2"/>
  <text x="600" y="180" class="class-text">Coffee Beans</text>
  
  <circle cx="600" cy="200" r="15" class="entity"/>
  <line x1="590" y1="205" x2="610" y2="205" stroke="#009900" stroke-width="2"/>
  <text x="600" y="230" class="class-text">Water</text>
  
  <circle cx="600" cy="250" r="15" class="entity"/>
  <line x1="590" y1="255" x2="610" y2="255" stroke="#009900" stroke-width="2"/>
  <text x="600" y="280" class="class-text">Milk</text>
  
  <circle cx="600" cy="300" r="15" class="context-derived"/>
  <line x1="590" y1="305" x2="610" y2="305" stroke="#CC0066" stroke-width="2"/>
  <text x="600" y="330" class="class-text">Cup</text>
  
  <!-- Output Entities -->
  <circle cx="750" cy="150" r="15" class="entity"/>
  <line x1="740" y1="155" x2="760" y2="155" stroke="#009900" stroke-width="2"/>
  <text x="750" y="180" class="class-text">Ground Coffee</text>
  
  <circle cx="750" cy="200" r="15" class="entity"/>
  <line x1="740" y1="205" x2="760" y2="205" stroke="#009900" stroke-width="2"/>
  <text x="750" y="230" class="class-text">Hot Water</text>
  
  <circle cx="750" cy="250" r="15" class="entity"/>
  <line x1="740" y1="255" x2="760" y2="255" stroke="#009900" stroke-width="2"/>
  <text x="750" y="280" class="class-text">Steamed Milk</text>
  
  <!-- Final Product -->
  <circle cx="850" cy="200" r="20" class="entity"/>
  <line x1="835" y1="207" x2="865" y2="207" stroke="#009900" stroke-width="3"/>
  <text x="850" y="235" class="class-text">Milk Coffee</text>
  
  <!-- Actor-Boundary connections -->
  <line x1="100" y1="85" x2="160" y2="80" class="association"/>
  <line x1="100" y1="125" x2="160" y2="150" class="association"/>
  
  <!-- Boundary-Controller connections -->
  <line x1="200" y1="80" x2="280" y2="80" class="association"/>
  
  <!-- ERGÄNZUNG: Boundary → Manager Controller Kontrollfluss -->
  <line x1="200" y1="150" x2="380" y2="150" class="association"/>
  <text x="290" y="145" class="class-text" font-size="9">control</text>
  
  <line x1="200" y1="200" x2="380" y2="200" class="association"/>
  <text x="290" y="195" class="class-text" font-size="9">control</text>
  
  <line x1="200" y1="250" x2="380" y2="250" class="association"/>
  <text x="290" y="245" class="class-text" font-size="9">control</text>
  
  <line x1="200" y1="300" x2="380" y2="300" class="association"/>
  <text x="290" y="295" class="class-text" font-size="9">control</text>
  
  <!-- Timer coordination -->
  <line x1="320" y1="80" x2="475" y2="80" class="coordination"/>
  <text x="395" y="75" class="success-text" font-size="9">activates</text>
  
  <!-- Orchestrator coordination -->
  <line x1="485" y1="95" x2="415" y2="135" class="coordination"/>
  <line x1="490" y1="95" x2="415" y2="185" class="coordination"/>
  <line x1="495" y1="95" x2="415" y2="235" class="coordination"/>
  <line x1="500" y1="95" x2="415" y2="285" class="coordination"/>
  <text x="450" y="125" class="success-text" font-size="9">coordinates</text>
  
  <!-- KORREKTE RUP BEZIEHUNGEN: Controller -> use -> Entity -->
  <line x1="420" y1="150" x2="585" y2="150" class="use-relation"/>
  <text x="500" y="145" class="class-text" font-size="9">«use»</text>
  
  <line x1="420" y1="200" x2="585" y2="200" class="use-relation"/>
  <text x="500" y="195" class="class-text" font-size="9">«use»</text>
  
  <line x1="420" y1="250" x2="585" y2="250" class="use-relation"/>
  <text x="500" y="245" class="class-text" font-size="9">«use»</text>
  
  <line x1="420" y1="300" x2="585" y2="300" class="use-relation"/>
  <text x="500" y="295" class="class-text" font-size="9">«use»</text>
  
  <!-- KORREKTE RUP BEZIEHUNGEN: Controller -> provide -> Entity -->
  <line x1="420" y1="155" x2="735" y2="155" class="provide-relation"/>
  <text x="575" y="150" class="success-text" font-size="9">«provide»</text>
  
  <line x1="420" y1="205" x2="735" y2="205" class="provide-relation"/>
  <text x="575" y="200" class="success-text" font-size="9">«provide»</text>
  
  <line x1="420" y1="255" x2="735" y2="255" class="provide-relation"/>
  <text x="575" y="250" class="success-text" font-size="9">«provide»</text>
  
  <!-- GetraenkeOrchestrator -> use -> Output Entities -->
  <line x1="525" y1="85" x2="735" y2="155" class="use-relation"/>
  <line x1="525" y1="85" x2="735" y2="205" class="use-relation"/>
  <line x1="525" y1="85" x2="735" y2="255" class="use-relation"/>
  <text x="630" y="130" class="class-text" font-size="9">«use»</text>
  
  <!-- GetraenkeOrchestrator -> provide -> Final Product -->
  <line x1="525" y1="85" x2="835" y2="185" class="provide-relation"/>
  <text x="680" y="130" class="success-text" font-size="9">«provide»</text>
  
  <!-- KORREKTUR: Kein direkter Entity-Boundary Fluss! -->
  <!-- Muss über Controller gehen: GetraenkeOrchestrator -> provide -> Entity -->
  <!-- Dann separater Controller für Output: OutputManager -> use -> Entity -> provide -> Boundary -->
  
  <!-- OutputManager Controller -->
  <circle cx="900" cy="120" r="15" class="control"/>
  <polygon points="895,123 905,123 900,117" fill="#FF6600"/>
  <text x="900" y="145" class="class-text">OutputManager</text>
  
  <!-- OutputManager -> use -> Final Product -->
  <line x1="900" y1="135" x2="850" y2="185" class="use-relation"/>
  <text x="875" y="155" class="class-text" font-size="9">«use»</text>
  
  <!-- OutputManager -> provide -> User Output Boundary -->
  <line x1="915" y1="120" x2="930" y2="200" class="provide-relation"/>
  <text x="922" y="160" class="success-text" font-size="9">«provide»</text>
  
  <!-- User Output Boundary -> User (final delivery) -->
  <line x1="970" y1="200" x2="1050" y2="125" class="association"/>
  <text x="1010" y="160" class="class-text" font-size="9">B5: delivery</text>
  
  <!-- RUP Rules Legend -->
  <text x="50" y="400" class="success-text" font-weight="bold" font-size="14">KORREKTE RUP REGELN:</text>
  
  <text x="60" y="420" class="success-text" font-size="12">1. Actor -> Boundary (Interface)</text>
  <line x1="60" y1="430" x2="120" y2="430" class="association"/>
  
  <text x="60" y="450" class="success-text" font-size="12">2. Boundary -> Controller (Control)</text>
  <line x1="60" y1="460" x2="120" y2="460" class="association"/>
  <text x="90" y="455" class="class-text" font-size="8">control</text>
  
  <text x="60" y="480" class="success-text" font-size="12">3. Controller -> use -> Entity (Input)</text>
  <line x1="60" y1="490" x2="120" y2="490" class="use-relation"/>
  <text x="90" y="485" class="class-text" font-size="8">«use»</text>
  
  <text x="60" y="510" class="success-text" font-size="12">4. Controller -> provide -> Entity (Output)</text>
  <line x1="60" y1="520" x2="120" y2="520" class="provide-relation"/>
  <text x="90" y="515" class="success-text" font-size="8">«provide»</text>
  
  <text x="60" y="540" class="error-text" font-size="12">5. NIEMALS: Entity -> Entity</text>
  <line x1="60" y1="550" x2="120" y2="550" stroke="#CC0000" stroke-width="3"/>
  <text x="90" y="545" class="error-text" font-size="8">VERBOTEN!</text>
  
  <text x="60" y="570" class="error-text" font-size="12">6. NIEMALS: Entity -> Boundary</text>
  <line x1="60" y1="580" x2="120" y2="580" stroke="#CC0000" stroke-width="3"/>
  <text x="90" y="575" class="error-text" font-size="8">VERBOTEN!</text>
  
  <!-- Kontrollfluss Details -->
  <text x="300" y="450" class="success-text" font-weight="bold" font-size="14">KOMPLETTER RUP KONTROLLFLUSS:</text>
  
  <text x="310" y="470" class="class-text" font-size="11">Beispiel CoffeeBeans:</text>
  <text x="320" y="485" class="class-text" font-size="10">1. User -> CoffeeBeans Input (Interface)</text>
  <text x="320" y="500" class="class-text" font-size="10">2. CoffeeBeans Input -> CoffeeBeansManager (Control)</text>
  <text x="320" y="515" class="class-text" font-size="10">3. CoffeeBeansManager -> use -> Coffee Beans</text>
  <text x="320" y="530" class="class-text" font-size="10">4. CoffeeBeansManager perform grinding</text>
  <text x="320" y="545" class="success-text" font-size="10">5. CoffeeBeansManager -> provide -> Ground Coffee</text>
  
  <text x="650" y="470" class="class-text" font-size="11">Output Kontrollfluss (KORRIGIERT):</text>
  <text x="660" y="485" class="class-text" font-size="10">• GetraenkeOrchestrator -> provide -> Milk Coffee</text>
  <text x="660" y="500" class="success-text" font-size="10">• OutputManager -> use -> Milk Coffee</text>
  <text x="660" y="515" class="success-text" font-size="10">• OutputManager -> provide -> User Output Boundary</text>
  <text x="660" y="530" class="class-text" font-size="10">• User Output Boundary -> User (Interface)</text>
  <text x="660" y="545" class="error-text" font-size="10">NIEMALS: Milk Coffee -> User Output direkt!</text>
  
  <!-- Success Summary -->
  <rect x="50" y="580" width="1100" height="80" fill="#E6FFE6" stroke="#009900" stroke-width="2"/>
  <text x="60" y="600" class="success-text" font-size="12">VOLLSTÄNDIGE RUP-KORREKTUR IMPLEMENTIERT!</text>
  <text x="60" y="615" class="success-text" font-size="11">Actor -> Boundary -> Controller -> Entity Muster komplett</text>
  <text x="60" y="630" class="success-text" font-size="11">KEINE Entity-Entity oder Entity-Boundary Beziehungen!</text>
  <text x="60" y="645" class="success-text" font-size="11">Alle Flüsse gehen über Controller (RUP-konform)</text>
  
</svg>'''
    
    # Save SVG
    output_file = "Zwischenprodukte/UC1_Coffee_RUP_KORREKT_KEINE_ENTITY_BEZIEHUNGEN.svg"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"Korrekte RUP Analysis Class Diagram generated!")
    print(f"File saved to: {output_file}")
    print(f"")
    print(f"KORREKTUR ANGEWENDET:")
    print(f"- Entfernt: Alle Entity-Entity Beziehungen")
    print(f"- Ergänzt: Boundary -> Controller Kontrollfluss")
    print(f"- Korrekt: Actor -> Boundary -> Controller -> Entity")
    print(f"- Korrekt: Controller -> use -> Entity (Input)")
    print(f"- Korrekt: Controller -> provide -> Entity (Output)")
    print(f"- Korrekt: Controller perform Transformationen")
    print(f"")
    print(f"VOLLSTÄNDIGER RUP KONTROLLFLUSS:")
    print(f"1. User -> CoffeeBeans Input (Interface)")
    print(f"2. CoffeeBeans Input -> CoffeeBeansManager (Control)")
    print(f"3. CoffeeBeansManager -> use -> Coffee Beans")
    print(f"4. CoffeeBeansManager perform grinding")
    print(f"5. CoffeeBeansManager -> provide -> Ground Coffee")
    print(f"")
    print(f"OUTPUT KORREKTUR (Entity-Boundary Verbindung entfernt):")
    print(f"6. GetraenkeOrchestrator -> provide -> Milk Coffee")
    print(f"7. OutputManager -> use -> Milk Coffee") 
    print(f"8. OutputManager -> provide -> User Output Boundary")
    print(f"9. User Output Boundary -> User (Interface)")
    print(f"")
    print(f"NIEMALS: Entity -> Boundary direkt! Immer über Controller!")
    
    return output_file

if __name__ == "__main__":
    generate_uc1_correct_rup_diagram()