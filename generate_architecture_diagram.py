#!/usr/bin/env python3
"""
Generate Architecture Diagram for RA-NLF System
Creates a visual representation of the system architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

def create_architecture_diagram():
    """Create comprehensive architecture diagram"""

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # Title
    ax.text(8, 19.5, 'RA-NLF Software Architecture',
            ha='center', va='top', fontsize=20, fontweight='bold')
    ax.text(8, 19, 'Robustness Analysis Framework with Natural Language Foundation',
            ha='center', va='top', fontsize=12, style='italic')

    # Color scheme
    colors = {
        'presentation': '#E3F2FD',   # Light blue
        'application': '#FFF9C4',     # Light yellow
        'domain': '#C8E6C9',          # Light green
        'infrastructure': '#FFCCBC',  # Light orange
        'data': '#F3E5F5'             # Light purple
    }

    # Layer Y positions
    y_presentation = 17.5
    y_application = 14.5
    y_domain = 11.5
    y_infrastructure = 8.5
    y_data = 5.5

    # ===== PRESENTATION LAYER =====
    layer_height = 2.0

    # Layer box
    rect = FancyBboxPatch((0.5, y_presentation - layer_height), 15, layer_height,
                          boxstyle="round,pad=0.1",
                          linewidth=2, edgecolor='#1976D2',
                          facecolor=colors['presentation'])
    ax.add_patch(rect)

    ax.text(1, y_presentation - 0.2, 'PRESENTATION LAYER',
            fontsize=11, fontweight='bold', color='#1976D2')

    # Components
    components = [
        ('SVG\nVisualizer', 2.5, 0.8),
        ('PNG\nVisualizer', 4.5, 0.8),
        ('JSON\nExporter', 6.5, 0.8),
        ('CSV\nExporter', 8.5, 0.8),
        ('Combined\nDiagram', 10.5, 0.8)
    ]

    for name, x, width in components:
        comp_rect = FancyBboxPatch((x, y_presentation - 1.5), width, 0.8,
                                  boxstyle="round,pad=0.05",
                                  linewidth=1, edgecolor='#0D47A1',
                                  facecolor='white')
        ax.add_patch(comp_rect)
        ax.text(x + width/2, y_presentation - 1.1, name,
                ha='center', va='center', fontsize=8)

    # ===== APPLICATION LAYER =====
    rect = FancyBboxPatch((0.5, y_application - layer_height), 15, layer_height,
                          boxstyle="round,pad=0.1",
                          linewidth=2, edgecolor='#F57F17',
                          facecolor=colors['application'])
    ax.add_patch(rect)

    ax.text(1, y_application - 0.2, 'APPLICATION LAYER',
            fontsize=11, fontweight='bold', color='#F57F17')

    # Main component - Structured UC Analyzer
    main_rect = FancyBboxPatch((2, y_application - 1.7), 5, 1.2,
                              boxstyle="round,pad=0.05",
                              linewidth=2, edgecolor='#F57F17',
                              facecolor='white')
    ax.add_patch(main_rect)
    ax.text(4.5, y_application - 1.1, 'StructuredUCAnalyzer',
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(4.5, y_application - 1.4, '(6057 lines - Core Engine)',
            ha='center', va='center', fontsize=7, style='italic')

    # Multi-UC Manager
    multi_rect = FancyBboxPatch((8, y_application - 1.7), 3, 1.2,
                               boxstyle="round,pad=0.05",
                               linewidth=1, edgecolor='#F57F17',
                               facecolor='white')
    ax.add_patch(multi_rect)
    ax.text(9.5, y_application - 1.1, 'Multi-UC\nManager',
            ha='center', va='center', fontsize=8)

    # Config Manager
    config_rect = FancyBboxPatch((12, y_application - 1.7), 2.5, 1.2,
                                boxstyle="round,pad=0.05",
                                linewidth=1, edgecolor='#F57F17',
                                facecolor='white')
    ax.add_patch(config_rect)
    ax.text(13.25, y_application - 1.1, 'Config\nManager',
            ha='center', va='center', fontsize=8)

    # ===== DOMAIN LAYER =====
    rect = FancyBboxPatch((0.5, y_domain - layer_height), 15, layer_height,
                          boxstyle="round,pad=0.1",
                          linewidth=2, edgecolor='#388E3C',
                          facecolor=colors['domain'])
    ax.add_patch(rect)

    ax.text(1, y_domain - 0.2, 'DOMAIN LAYER',
            fontsize=11, fontweight='bold', color='#388E3C')

    # Domain components
    domain_comps = [
        ('Material Controller\nRegistry', 2, 2.5),
        ('Control Flow\nEngine', 5, 2.2),
        ('Data Flow\nAnalyzer', 7.5, 2.2),
        ('RA Class\nGenerator', 10, 2.2),
        ('UC-Methode\nRules', 12.5, 2.2)
    ]

    for name, x, width in domain_comps:
        comp_rect = FancyBboxPatch((x, y_domain - 1.6), width, 1.0,
                                  boxstyle="round,pad=0.05",
                                  linewidth=1, edgecolor='#1B5E20',
                                  facecolor='white')
        ax.add_patch(comp_rect)
        ax.text(x + width/2, y_domain - 1.1, name,
                ha='center', va='center', fontsize=7)

    # ===== INFRASTRUCTURE LAYER =====
    rect = FancyBboxPatch((0.5, y_infrastructure - layer_height), 15, layer_height,
                          boxstyle="round,pad=0.1",
                          linewidth=2, edgecolor='#E64A19',
                          facecolor=colors['infrastructure'])
    ax.add_patch(rect)

    ax.text(1, y_infrastructure - 0.2, 'INFRASTRUCTURE LAYER',
            fontsize=11, fontweight='bold', color='#E64A19')

    # Infrastructure components
    infra_comps = [
        ('Domain Verb\nLoader', 2, 2.5),
        ('Generative\nContext Mgr', 5, 2.5),
        ('NLP Engine\n(spaCy)', 8, 2.2),
        ('Control Flow\nCorrector', 10.5, 2.2),
        ('RUP Symbol\nLoader', 13, 2.0)
    ]

    for name, x, width in infra_comps:
        comp_rect = FancyBboxPatch((x, y_infrastructure - 1.6), width, 1.0,
                                  boxstyle="round,pad=0.05",
                                  linewidth=1, edgecolor='#BF360C',
                                  facecolor='white')
        ax.add_patch(comp_rect)
        ax.text(x + width/2, y_infrastructure - 1.1, name,
                ha='center', va='center', fontsize=7)

    # ===== DATA LAYER =====
    rect = FancyBboxPatch((0.5, y_data - layer_height), 15, layer_height,
                          boxstyle="round,pad=0.1",
                          linewidth=2, edgecolor='#7B1FA2',
                          facecolor=colors['data'])
    ax.add_patch(rect)

    ax.text(1, y_data - 0.2, 'DATA LAYER',
            fontsize=11, fontweight='bold', color='#7B1FA2')

    # Data components
    data_comps = [
        ('Domain\nJSON', 2, 1.8),
        ('UC Files\n(.txt)', 4, 1.8),
        ('spaCy\nModel', 6, 1.8),
        ('RUP\nSymbols', 8, 1.8),
        ('Analysis\nConfig', 10, 1.8),
        ('Output\nFiles', 12, 1.8)
    ]

    for name, x, width in data_comps:
        comp_rect = FancyBboxPatch((x, y_data - 1.6), width, 1.0,
                                  boxstyle="round,pad=0.05",
                                  linewidth=1, edgecolor='#4A148C',
                                  facecolor='white')
        ax.add_patch(comp_rect)
        ax.text(x + width/2, y_data - 1.1, name,
                ha='center', va='center', fontsize=7)

    # ===== ARROWS SHOWING DATA FLOW =====
    arrow_style = "Simple,tail_width=0.5,head_width=4,head_length=8"

    # Application -> Domain
    arrow1 = FancyArrowPatch((4.5, y_application - layer_height),
                            (4.5, y_domain),
                            arrowstyle=arrow_style, color='gray',
                            linewidth=1.5, alpha=0.7)
    ax.add_patch(arrow1)

    # Domain -> Infrastructure
    arrow2 = FancyArrowPatch((6, y_domain - layer_height),
                            (6, y_infrastructure),
                            arrowstyle=arrow_style, color='gray',
                            linewidth=1.5, alpha=0.7)
    ax.add_patch(arrow2)

    # Infrastructure -> Data
    arrow3 = FancyArrowPatch((7, y_infrastructure - layer_height),
                            (7, y_data),
                            arrowstyle=arrow_style, color='gray',
                            linewidth=1.5, alpha=0.7)
    ax.add_patch(arrow3)

    # Application -> Presentation
    arrow4 = FancyArrowPatch((4.5, y_application),
                            (4.5, y_presentation - layer_height),
                            arrowstyle=arrow_style, color='gray',
                            linewidth=1.5, alpha=0.7)
    ax.add_patch(arrow4)

    # ===== KEY FEATURES BOX =====
    features_y = 3.5
    feature_rect = FancyBboxPatch((0.5, features_y - 2.8), 7, 2.5,
                                 boxstyle="round,pad=0.1",
                                 linewidth=2, edgecolor='#455A64',
                                 facecolor='#ECEFF1')
    ax.add_patch(feature_rect)

    ax.text(1, features_y - 0.3, 'Key Features',
            fontsize=10, fontweight='bold', color='#263238')

    features = [
        '✓ Domain-Independent NLP Analysis',
        '✓ Material-Based Controllers (Solid/Liquid/Gas)',
        '✓ Multi-UC Shared Component Registry',
        '✓ UC-Methode Rules 1-5 Compliance',
        '✓ Config-Driven Batch Processing',
        '✓ RUP Symbol Library (Wikipedia Standard)'
    ]

    y_pos = features_y - 0.7
    for feature in features:
        ax.text(1.2, y_pos, feature, fontsize=7, color='#37474F')
        y_pos -= 0.3

    # ===== TECHNOLOGY STACK BOX =====
    tech_rect = FancyBboxPatch((8.5, features_y - 2.8), 7, 2.5,
                              boxstyle="round,pad=0.1",
                              linewidth=2, edgecolor='#455A64',
                              facecolor='#ECEFF1')
    ax.add_patch(tech_rect)

    ax.text(9, features_y - 0.3, 'Technology Stack',
            fontsize=10, fontweight='bold', color='#263238')

    tech_items = [
        '• Python 3.x',
        '• spaCy (en_core_web_md) - NLP Engine',
        '• Matplotlib - Pure RUP Diagrams',
        '• SVG Generation - Scalable Graphics',
        '• JSON - Domain Configurations',
        '• Git - Version Control'
    ]

    y_pos = features_y - 0.7
    for item in tech_items:
        ax.text(9.2, y_pos, item, fontsize=7, color='#37474F')
        y_pos -= 0.3

    # ===== METADATA =====
    ax.text(8, 0.5, 'RA-NLF Architecture Diagram | Version 2.0 | 2025-11-13',
            ha='center', va='bottom', fontsize=8, style='italic', color='gray')
    ax.text(8, 0.2, 'Repository: https://github.com/DieterWagner42/RA-NLF',
            ha='center', va='bottom', fontsize=7, color='gray')

    # Save
    plt.tight_layout()
    plt.savefig('ARCHITECTURE_DIAGRAM.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print('[OK] Architecture diagram saved: ARCHITECTURE_DIAGRAM.png')

    plt.close()


def create_component_diagram():
    """Create detailed component interaction diagram"""

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(8, 11.5, 'RA-NLF Component Interaction Diagram',
            ha='center', va='top', fontsize=18, fontweight='bold')

    # Color scheme for component types
    colors = {
        'analyzer': '#BBDEFB',
        'loader': '#FFE0B2',
        'manager': '#C5E1A5',
        'visualizer': '#F8BBD0',
        'registry': '#D1C4E9'
    }

    # ===== MAIN COMPONENTS =====

    # StructuredUCAnalyzer (center)
    main_box = FancyBboxPatch((6, 7), 4, 2,
                             boxstyle="round,pad=0.1",
                             linewidth=3, edgecolor='#1976D2',
                             facecolor=colors['analyzer'])
    ax.add_patch(main_box)
    ax.text(8, 8.3, 'StructuredUCAnalyzer', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(8, 7.9, 'Main Analysis Engine', ha='center', va='center',
            fontsize=8, style='italic')
    ax.text(8, 7.5, '6057 lines', ha='center', va='center',
            fontsize=7, color='gray')

    # DomainVerbLoader (left top)
    loader_box = FancyBboxPatch((1, 8.5), 3, 1.5,
                               boxstyle="round,pad=0.05",
                               linewidth=2, edgecolor='#FF6F00',
                               facecolor=colors['loader'])
    ax.add_patch(loader_box)
    ax.text(2.5, 9.5, 'DomainVerbLoader', ha='center', va='center',
            fontsize=9, fontweight='bold')
    ax.text(2.5, 9.1, '641 lines', ha='center', va='center',
            fontsize=7, color='gray')

    # MaterialControllerRegistry (left bottom)
    registry_box = FancyBboxPatch((1, 5.5), 3, 1.5,
                                 boxstyle="round,pad=0.05",
                                 linewidth=2, edgecolor='#6A1B9A',
                                 facecolor=colors['registry'])
    ax.add_patch(registry_box)
    ax.text(2.5, 6.5, 'MaterialController\nRegistry', ha='center', va='center',
            fontsize=9, fontweight='bold')
    ax.text(2.5, 6, '284 lines', ha='center', va='center',
            fontsize=7, color='gray')

    # GenerativeContextManager (right top)
    context_box = FancyBboxPatch((12, 8.5), 3, 1.5,
                                boxstyle="round,pad=0.05",
                                linewidth=2, edgecolor='#558B2F',
                                facecolor=colors['manager'])
    ax.add_patch(context_box)
    ax.text(13.5, 9.5, 'GenerativeContext\nManager', ha='center', va='center',
            fontsize=9, fontweight='bold')
    ax.text(13.5, 8.9, '530 lines', ha='center', va='center',
            fontsize=7, color='gray')

    # SVG Visualizer (bottom left)
    svg_box = FancyBboxPatch((2, 2.5), 3, 1.5,
                            boxstyle="round,pad=0.05",
                            linewidth=2, edgecolor='#C2185B',
                            facecolor=colors['visualizer'])
    ax.add_patch(svg_box)
    ax.text(3.5, 3.5, 'SVGRUPVisualizer', ha='center', va='center',
            fontsize=9, fontweight='bold')
    ax.text(3.5, 3, '771 lines', ha='center', va='center',
            fontsize=7, color='gray')

    # Pure RUP Visualizer (bottom center)
    pure_box = FancyBboxPatch((6, 2.5), 3, 1.5,
                             boxstyle="round,pad=0.05",
                             linewidth=2, edgecolor='#C2185B',
                             facecolor=colors['visualizer'])
    ax.add_patch(pure_box)
    ax.text(7.5, 3.5, 'PureRUPVisualizer', ha='center', va='center',
            fontsize=9, fontweight='bold')
    ax.text(7.5, 3, '459 lines', ha='center', va='center',
            fontsize=7, color='gray')

    # Multi-UC Analyzer (bottom right)
    multi_box = FancyBboxPatch((10, 2.5), 3, 1.5,
                              boxstyle="round,pad=0.05",
                              linewidth=2, edgecolor='#1976D2',
                              facecolor=colors['analyzer'])
    ax.add_patch(multi_box)
    ax.text(11.5, 3.5, 'MultiUCAnalyzer', ha='center', va='center',
            fontsize=9, fontweight='bold')
    ax.text(11.5, 3, '481 lines', ha='center', va='center',
            fontsize=7, color='gray')

    # ===== ARROWS =====
    arrow_style = "Simple,tail_width=0.3,head_width=3,head_length=6"

    # Loader -> Analyzer
    ax.annotate('', xy=(6, 8.5), xytext=(4, 9),
               arrowprops=dict(arrowstyle='->', lw=2, color='#FF6F00'))
    ax.text(5, 8.8, 'Load verbs', fontsize=7, style='italic', color='#FF6F00')

    # Registry -> Analyzer
    ax.annotate('', xy=(6, 7.5), xytext=(4, 6.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='#6A1B9A'))
    ax.text(5, 7, 'Get controllers', fontsize=7, style='italic', color='#6A1B9A')

    # Context Manager -> Analyzer
    ax.annotate('', xy=(10, 8.5), xytext=(12, 9),
               arrowprops=dict(arrowstyle='->', lw=2, color='#558B2F'))
    ax.text(11, 8.8, 'Generate context', fontsize=7, style='italic', color='#558B2F')

    # Analyzer -> Visualizers
    ax.annotate('', xy=(3.5, 4), xytext=(7, 7),
               arrowprops=dict(arrowstyle='->', lw=2, color='#C2185B'))
    ax.text(5, 5.5, 'JSON data', fontsize=7, style='italic', color='#C2185B')

    ax.annotate('', xy=(7.5, 4), xytext=(8, 7),
               arrowprops=dict(arrowstyle='->', lw=2, color='#C2185B'))

    ax.annotate('', xy=(11.5, 4), xytext=(9, 7),
               arrowprops=dict(arrowstyle='->', lw=2, color='#1976D2'))

    # ===== DATA STORES =====

    # Domain JSON
    data_box1 = FancyBboxPatch((0.5, 0.5), 2.5, 0.8,
                              boxstyle="round,pad=0.05",
                              linewidth=1, edgecolor='gray',
                              facecolor='#EEEEEE')
    ax.add_patch(data_box1)
    ax.text(1.75, 0.9, 'domains/*.json', ha='center', va='center',
            fontsize=7, family='monospace')

    # UC Files
    data_box2 = FancyBboxPatch((3.5, 0.5), 2.5, 0.8,
                              boxstyle="round,pad=0.05",
                              linewidth=1, edgecolor='gray',
                              facecolor='#EEEEEE')
    ax.add_patch(data_box2)
    ax.text(4.75, 0.9, 'Use Case/*.txt', ha='center', va='center',
            fontsize=7, family='monospace')

    # spaCy Model
    data_box3 = FancyBboxPatch((6.5, 0.5), 2.5, 0.8,
                              boxstyle="round,pad=0.05",
                              linewidth=1, edgecolor='gray',
                              facecolor='#EEEEEE')
    ax.add_patch(data_box3)
    ax.text(7.75, 0.9, 'en_core_web_md', ha='center', va='center',
            fontsize=7, family='monospace')

    # Config File
    data_box4 = FancyBboxPatch((9.5, 0.5), 3, 0.8,
                              boxstyle="round,pad=0.05",
                              linewidth=1, edgecolor='gray',
                              facecolor='#EEEEEE')
    ax.add_patch(data_box4)
    ax.text(11, 0.9, 'uc_analysis_config.json', ha='center', va='center',
            fontsize=7, family='monospace')

    # RUP Symbols
    data_box5 = FancyBboxPatch((13, 0.5), 2.5, 0.8,
                              boxstyle="round,pad=0.05",
                              linewidth=1, edgecolor='gray',
                              facecolor='#EEEEEE')
    ax.add_patch(data_box5)
    ax.text(14.25, 0.9, 'svg/*.svg', ha='center', va='center',
            fontsize=7, family='monospace')

    # Arrows from data stores
    for x in [1.75, 4.75, 7.75, 11, 14.25]:
        ax.annotate('', xy=(x, 2.5), xytext=(x, 1.3),
                   arrowprops=dict(arrowstyle='->', lw=1, color='gray', alpha=0.5))

    # Legend
    legend_y = 11
    ax.text(1, legend_y, 'Component Types:', fontsize=8, fontweight='bold')

    legend_items = [
        ('Analyzer', colors['analyzer']),
        ('Loader', colors['loader']),
        ('Manager', colors['manager']),
        ('Visualizer', colors['visualizer']),
        ('Registry', colors['registry'])
    ]

    x_legend = 1
    for name, color in legend_items:
        legend_box = FancyBboxPatch((x_legend, legend_y - 0.5), 1, 0.3,
                                   boxstyle="round,pad=0.02",
                                   linewidth=1, edgecolor='gray',
                                   facecolor=color)
        ax.add_patch(legend_box)
        ax.text(x_legend + 0.5, legend_y - 0.35, name, ha='center', va='center',
                fontsize=6)
        x_legend += 1.2

    # Save
    plt.tight_layout()
    plt.savefig('ARCHITECTURE_COMPONENTS.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print('[OK] Component diagram saved: ARCHITECTURE_COMPONENTS.png')

    plt.close()


def create_data_flow_diagram():
    """Create data flow diagram showing analysis pipeline"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 16)
    ax.axis('off')

    # Title
    ax.text(7, 15.5, 'RA-NLF Analysis Pipeline',
            ha='center', va='top', fontsize=18, fontweight='bold')
    ax.text(7, 15, 'From Use Case Text to RUP Diagram',
            ha='center', va='top', fontsize=12, style='italic')

    # Pipeline stages
    stages = [
        (14, 'INPUT', 'Use Case File\n(UC1.txt)', '#E3F2FD'),
        (12.5, 'PHASE 1', 'Context Analysis\n• Capability\n• Goal\n• Domain Detection', '#FFF9C4'),
        (11, 'PHASE 2', 'Resource Analysis\n• Preconditions\n• Operational Materials\n• Safety/Hygiene', '#C8E6C9'),
        (9.5, 'PHASE 3', 'Interaction Analysis\n• Transaction Verbs → Boundaries\n• Transformation Verbs → Controllers\n• Function Verbs → Material Functions', '#FFCCBC'),
        (8, 'PHASE 4', 'Control Flow Generation\n• UC-Methode Rules 1-5\n• Parallel Flow Nodes\n• Actor-Boundary Flows', '#F3E5F5'),
        (6.5, 'PHASE 5', 'Data Flow Analysis\n• USE Relationships (with/from)\n• PROVIDE Relationships (to/for)\n• Transformation Patterns', '#C5E1A5'),
        (5, 'PHASE 6', 'RA Classification\n• Assign Stereotypes\n• Generate Components\n• Validate Rules', '#BBDEFB'),
        (3.5, 'OUTPUT', 'JSON Analysis\n+ CSV Steps\n+ SVG Diagram\n+ PNG Diagram', '#D7CCC8')
    ]

    for y, phase, description, color in stages:
        # Phase box
        box = FancyBboxPatch((2, y - 1.2), 10, 1,
                           boxstyle="round,pad=0.1",
                           linewidth=2, edgecolor='gray',
                           facecolor=color)
        ax.add_patch(box)

        # Phase label
        ax.text(2.5, y - 0.3, phase, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

        # Description
        ax.text(7, y - 0.7, description, ha='center', va='center',
                fontsize=7, multialignment='center')

        # Arrow to next phase (except last)
        if y > 3.5:
            arrow = FancyArrowPatch((7, y - 1.2), (7, y - 1.5),
                                   arrowstyle='->', lw=2, color='#455A64',
                                   mutation_scale=20)
            ax.add_patch(arrow)

    # Side annotations
    annotations = [
        (12.5, 'Extracts UC metadata and identifies domain'),
        (11, 'Identifies required materials and their properties'),
        (9.5, 'Creates Boundaries, Controllers, and Entities from verbs'),
        (8, 'Connects components following UC-Methode patterns'),
        (6.5, 'Determines entity usage and production by controllers'),
        (5, 'Finalizes RA class assignments and validates'),
        (3.5, 'Generates multiple output formats for visualization')
    ]

    for y, text in annotations:
        ax.text(12.5, y - 0.7, text, fontsize=6, style='italic',
                color='#546E7A', ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECEFF1',
                         edgecolor='none', alpha=0.8))

    # Key algorithms box
    algo_y = 1.5
    algo_box = FancyBboxPatch((1, algo_y - 1.2), 12, 1,
                             boxstyle="round,pad=0.1",
                             linewidth=2, edgecolor='#1565C0',
                             facecolor='#E1F5FE')
    ax.add_patch(algo_box)

    ax.text(1.5, algo_y - 0.2, 'Key Algorithms:', fontsize=9, fontweight='bold',
            color='#01579B')

    algorithms = [
        'NLP: spaCy (en_core_web_md)',
        'Controller Selection: Material + Aggregation State',
        'Parallel Detection: Step ID Pattern Matching',
        'Flow Generation: UC-Methode Rules 1-5'
    ]

    x_algo = 1.5
    for algo in algorithms:
        ax.text(x_algo, algo_y - 0.7, f'• {algo}', fontsize=7, color='#0277BD')
        x_algo += 3

    # Save
    plt.tight_layout()
    plt.savefig('ARCHITECTURE_DATAFLOW.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print('[OK] Data flow diagram saved: ARCHITECTURE_DATAFLOW.png')

    plt.close()


if __name__ == '__main__':
    print("="*70)
    print("GENERATING RA-NLF ARCHITECTURE DIAGRAMS")
    print("="*70)

    create_architecture_diagram()
    create_component_diagram()
    create_data_flow_diagram()

    print("\n" + "="*70)
    print("SUCCESS: All architecture diagrams generated")
    print("="*70)
    print("\nFiles created:")
    print("  - ARCHITECTURE_DIAGRAM.png (Layered Architecture)")
    print("  - ARCHITECTURE_COMPONENTS.png (Component Interactions)")
    print("  - ARCHITECTURE_DATAFLOW.png (Analysis Pipeline)")
    print("\nSee ARCHITECTURE.md for detailed documentation")
