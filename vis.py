"""
Graph-Visualisierung für Robustheitsanalyse
Visualisiert Boundary, Entity und Control Objects sowie ihre Beziehungen

Installation:
pip install networkx matplotlib pygraphviz (optional)
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import json
from typing import Dict, List, Tuple
import math

class RobustnessGraphVisualizer:
    """
    Visualisiert Robustness Analysis Graphen
    """
    
    def __init__(self, analysis_results: Dict = None):
        """
        Initialisiere Visualizer
        
        Args:
            analysis_results: Ergebnis von RobustnessAnalyzer
        """
        self.results = analysis_results
        self.graph = nx.DiGraph()
        
        # Farben für verschiedene Object-Typen
        self.colors = {
            'boundary': '#9b59b6',  # Lila
            'entity': '#27ae60',    # Grün
            'control': '#3498db',   # Blau
            'actor': '#e74c3c'      # Rot
        }
        
        # Shapes für verschiedene Object-Typen (UML Notation)
        self.shapes = {
            'boundary': 'circle',    # ⭕ Kreis
            'entity': 'square',      # ⬭ Rechteck
            'control': 'triangle',   # ↻ Pfeil/Dreieck
            'actor': 'diamond'       # ◆ Raute
        }
        
    def load_from_json(self, filename: str):
        """Lade Analyse-Ergebnisse aus JSON"""
        with open(filename, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        print(f"✓ Lade Ergebnisse aus {filename}")
    
    def build_graph(self):
        """Baue NetworkX Graph aus Analyse-Ergebnissen"""
        if not self.results:
            print("Fehler: Keine Analyse-Ergebnisse vorhanden")
            return
        
        print("\n" + "="*60)
        print("BAUE ROBUSTNESS GRAPH")
        print("="*60)
        
        # Füge Objekte als Knoten hinzu
        objects = self.results.get('all_objects', self.results.get('objects', {}))
        shared = self.results.get('shared_objects', {})
        
        for obj_type in ['boundary', 'entity', 'control']:
            obj_list = objects.get(obj_type, [])
            shared_list = shared.get(obj_type, [])
            
            for obj in obj_list:
                is_shared = obj in shared_list
                self.graph.add_node(
                    obj,
                    type=obj_type,
                    shared=is_shared,
                    color=self.colors[obj_type],
                    shape=self.shapes[obj_type]
                )
        
        # Füge Beziehungen als Kanten hinzu
        relationships = self.results.get('relationships', [])
        for rel in relationships:
            from_node = rel.get('from')
            to_node = rel.get('to')
            rel_type = rel.get('type', 'relates')
            
            if from_node in self.graph and to_node in self.graph:
                self.graph.add_edge(
                    from_node,
                    to_node,
                    type=rel_type,
                    uc=rel.get('uc', ''),
                    step=rel.get('step', '')
                )
        
        print(f"\n✓ Graph erstellt:")
        print(f"  Knoten: {self.graph.number_of_nodes()}")
        print(f"  Kanten: {self.graph.number_of_edges()}")
    
    def visualize_hierarchical(self, filename: str = 'robustness_hierarchical.png', 
                               figsize: Tuple = (16, 10)):
        """
        Visualisiere Graph in hierarchischem Layout
        Boundary oben → Control Mitte → Entity unten
        """
        if not self.graph:
            print("Fehler: Graph ist leer. Führe zuerst build_graph() aus.")
            return
        
        print("\n--- Erstelle hierarchische Visualisierung ---")
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='#2c3e50')
        ax.set_facecolor('#34495e')
        
        # Erstelle hierarchisches Layout
        pos = self._hierarchical_layout()
        
        # Zeichne Knoten nach Typ
        for obj_type in ['boundary', 'entity', 'control']:
            nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == obj_type]
            
            if not nodes:
                continue
            
            # Bestimme Node-Größen (größer für shared objects)
            node_sizes = [
                3000 if self.graph.nodes[n].get('shared', False) else 2000
                for n in nodes
            ]
            
            # Zeichne Knoten
            if obj_type == 'boundary':
                # Kreise für Boundary
                nx.draw_networkx_nodes(
                    self.graph, pos,
                    nodelist=nodes,
                    node_color=self.colors[obj_type],
                    node_size=node_sizes,
                    node_shape='o',
                    edgecolors='white',
                    linewidths=2,
                    ax=ax,
                    alpha=0.9
                )
            elif obj_type == 'entity':
                # Quadrate für Entity
                nx.draw_networkx_nodes(
                    self.graph, pos,
                    nodelist=nodes,
                    node_color=self.colors[obj_type],
                    node_size=node_sizes,
                    node_shape='s',
                    edgecolors='white',
                    linewidths=2,
                    ax=ax,
                    alpha=0.9
                )
            elif obj_type == 'control':
                # Dreiecke für Control
                nx.draw_networkx_nodes(
                    self.graph, pos,
                    nodelist=nodes,
                    node_color=self.colors[obj_type],
                    node_size=node_sizes,
                    node_shape='^',
                    edgecolors='white',
                    linewidths=2,
                    ax=ax,
                    alpha=0.9
                )
        
        # Zeichne Kanten
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color='#95a5a6',
            arrows=True,
            arrowsize=15,
            arrowstyle='->',
            width=2,
            alpha=0.6,
            ax=ax,
            connectionstyle='arc3,rad=0.1'
        )
        
        # Labels
        labels = {node: self._wrap_label(node, 15) for node in self.graph.nodes()}
        nx.draw_networkx_labels(
            self.graph, pos,
            labels,
            font_size=8,
            font_color='white',
            font_weight='bold',
            ax=ax
        )
        
        # Titel und Legende
        ax.set_title('Robustness Analysis - Hierarchical View', 
                    fontsize=18, fontweight='bold', color='white', pad=20)
        
        self._add_legend(ax)
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, facecolor='#2c3e50', bbox_inches='tight')
        print(f"✓ Gespeichert: {filename}")
        plt.show()
    
    def visualize_by_usecase(self, filename: str = 'robustness_by_uc.png',
                            figsize: Tuple = (18, 12)):
        """
        Visualisiere Graph gruppiert nach Use Cases
        """
        if not self.results:
            print("Fehler: Keine Ergebnisse vorhanden")
            return
        
        print("\n--- Erstelle Use-Case-basierte Visualisierung ---")
        
        uc_objects = self.results.get('uc_objects', {})
        
        if not uc_objects:
            print("Warnung: Keine UC-spezifischen Objekte gefunden")
            return
        
        n_ucs = len(uc_objects)
        n_cols = min(2, n_ucs)
        n_rows = math.ceil(n_ucs / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, facecolor='#2c3e50')
        if n_ucs == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (uc_name, uc_obj) in enumerate(uc_objects.items()):
            ax = axes[idx]
            ax.set_facecolor('#34495e')
            
            # Erstelle Subgraph für diesen UC
            nodes = []
            for obj_type in ['boundary', 'entity', 'control']:
                nodes.extend(uc_obj.get(obj_type, []))
            
            subgraph = self.graph.subgraph(nodes).copy()
            
            # Layout
            pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
            
            # Zeichne nach Typ
            for obj_type in ['boundary', 'entity', 'control']:
                type_nodes = [n for n in subgraph.nodes() 
                             if n in uc_obj.get(obj_type, [])]
                
                if type_nodes:
                    shape = 'o' if obj_type == 'boundary' else ('s' if obj_type == 'entity' else '^')
                    nx.draw_networkx_nodes(
                        subgraph, pos,
                        nodelist=type_nodes,
                        node_color=self.colors[obj_type],
                        node_size=1500,
                        node_shape=shape,
                        edgecolors='white',
                        linewidths=2,
                        ax=ax,
                        alpha=0.9
                    )
            
            # Kanten
            nx.draw_networkx_edges(
                subgraph, pos,
                edge_color='#95a5a6',
                arrows=True,
                arrowsize=10,
                width=1.5,
                alpha=0.5,
                ax=ax
            )
            
            # Labels
            labels = {node: self._wrap_label(node, 12) for node in subgraph.nodes()}
            nx.draw_networkx_labels(
                subgraph, pos,
                labels,
                font_size=7,
                font_color='white',
                ax=ax
            )
            
            ax.set_title(uc_name, fontsize=12, fontweight='bold', color='white')
            ax.axis('off')
        
        # Verstecke ungenutzte Subplots
        for idx in range(n_ucs, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Robustness Analysis - By Use Case', 
                    fontsize=16, fontweight='bold', color='white', y=0.98)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, facecolor='#2c3e50', bbox_inches='tight')
        print(f"✓ Gespeichert: {filename}")
        plt.show()
    
    def visualize_feature_tree(self, filename: str = 'robustness_features.png',
                              figsize: Tuple = (14, 10)):
        """
        Visualisiere Feature-Hierarchie mit zugeordneten Use Cases
        """
        if not self.results:
            print("Fehler: Keine Ergebnisse vorhanden")
            return
        
        features = self.results.get('features', {})
        if not features:
            print("Warnung: Keine Feature-Informationen gefunden")
            return
        
        print("\n--- Erstelle Feature-Tree Visualisierung ---")
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='#2c3e50')
        ax.set_facecolor('#34495e')
        
        # Erstelle Feature-Tree Graph
        tree = nx.DiGraph()
        
        # Root Node
        tree.add_node('Root', type='root', level=0)
        
        # Capabilities
        for cap_idx, (capability, feature_dict) in enumerate(features.items()):
            cap_node = f"CAP: {capability}"
            tree.add_node(cap_node, type='capability', level=1)
            tree.add_edge('Root', cap_node)
            
            # Features
            for feat_idx, (feature, uc) in enumerate(feature_dict.items()):
                feat_node = f"FEAT: {feature}"
                tree.add_node(feat_node, type='feature', level=2)
                tree.add_edge(cap_node, feat_node)
                
                # Use Case
                uc_node = uc
                tree.add_node(uc_node, type='usecase', level=3)
                tree.add_edge(feat_node, uc_node)
        
        # Hierarchical Layout
        pos = nx.nx_agraph.graphviz_layout(tree, prog='dot') if self._has_graphviz() else self._manual_tree_layout(tree)
        
        # Farben nach Typ
        node_colors = []
        for node in tree.nodes():
            node_type = tree.nodes[node].get('type', 'default')
            if node_type == 'root':
                node_colors.append('#e74c3c')
            elif node_type == 'capability':
                node_colors.append('#f39c12')
            elif node_type == 'feature':
                node_colors.append('#3498db')
            elif node_type == 'usecase':
                node_colors.append('#27ae60')
            else:
                node_colors.append('#95a5a6')
        
        # Zeichne Tree
        nx.draw_networkx_nodes(
            tree, pos,
            node_color=node_colors,
            node_size=2000,
            edgecolors='white',
            linewidths=2,
            ax=ax,
            alpha=0.9
        )
        
        nx.draw_networkx_edges(
            tree, pos,
            edge_color='#95a5a6',
            arrows=True,
            arrowsize=15,
            width=2,
            alpha=0.6,
            ax=ax
        )
        
        # Labels
        labels = {node: self._wrap_label(node, 20) for node in tree.nodes()}
        nx.draw_networkx_labels(
            tree, pos,
            labels,
            font_size=8,
            font_color='white',
            font_weight='bold',
            ax=ax
        )
        
        ax.set_title('Feature Hierarchy & Use Case Mapping', 
                    fontsize=16, fontweight='bold', color='white', pad=20)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, facecolor='#2c3e50', bbox_inches='tight')
        print(f"✓ Gespeichert: {filename}")
        plt.show()
    
    def export_graph_formats(self, prefix: str = 'robustness'):
        """
        Exportiere Graph in verschiedenen Formaten
        """
        print("\n--- Exportiere Graph-Formate ---")
        
        # GraphML (für yEd, Gephi)
        nx.write_graphml(self.graph, f"{prefix}.graphml")
        print(f"✓ GraphML: {prefix}.graphml")
        
        # GML
        nx.write_gml(self.graph, f"{prefix}.gml")
        print(f"✓ GML: {prefix}.gml")
        
        # Adjacency List
        nx.write_adjlist(self.graph, f"{prefix}.adjlist")
        print(f"✓ Adjacency List: {prefix}.adjlist")
        
        # Edge List
        nx.write_edgelist(self.graph, f"{prefix}.edgelist")
        print(f"✓ Edge List: {prefix}.edgelist")
    
    def _hierarchical_layout(self) -> Dict:
        """Erstelle hierarchisches Layout: Boundary → Control → Entity"""
        pos = {}
        
        # Gruppiere Knoten nach Typ
        boundary_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'boundary']
        control_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'control']
        entity_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'entity']
        
        # Y-Positionen für Ebenen
        y_positions = {
            'boundary': 2.0,
            'control': 0.0,
            'entity': -2.0
        }
        
        # Verteile Knoten horizontal
        for node_list, y in [(boundary_nodes, y_positions['boundary']),
                             (control_nodes, y_positions['control']),
                             (entity_nodes, y_positions['entity'])]:
            if not node_list:
                continue
            
            n = len(node_list)
            x_positions = [i * 3.0 - (n - 1) * 1.5 for i in range(n)]
            
            for node, x in zip(node_list, x_positions):
                pos[node] = (x, y)
        
        return pos
    
    def _manual_tree_layout(self, tree: nx.DiGraph) -> Dict:
        """Manuelles Tree-Layout falls Graphviz nicht verfügbar"""
        pos = {}
        levels = nx.get_node_attributes(tree, 'level')
        
        # Gruppiere nach Level
        level_nodes = {}
        for node, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)
        
        # Positioniere
        for level, nodes in level_nodes.items():
            n = len(nodes)
            for idx, node in enumerate(nodes):
                x = idx * 3.0 - (n - 1) * 1.5
                y = -level * 2.0
                pos[node] = (x, y)
        
        return pos
    
    def _wrap_label(self, text: str, width: int) -> str:
        """Umbreche lange Labels"""
        if len(text) <= width:
            return text
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def _add_legend(self, ax):
        """Füge Legende hinzu"""
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Boundary Objects',
                   markerfacecolor=self.colors['boundary'], markersize=12),
            Line2D([0], [0], marker='s', color='w', label='Entity Objects',
                   markerfacecolor=self.colors['entity'], markersize=12),
            Line2D([0], [0], marker='^', color='w', label='Control Objects',
                   markerfacecolor=self.colors['control'], markersize=12),
            Line2D([0], [0], marker='o', color='w', label='Shared (Cross-UC)',
                   markerfacecolor='white', markersize=15, markeredgewidth=2,
                   markeredgecolor=self.colors['control'])
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 fontsize=10, framealpha=0.9, facecolor='#34495e',
                 edgecolor='white', labelcolor='white')
    
    def _has_graphviz(self) -> bool:
        """Prüfe ob Graphviz verfügbar ist"""
        try:
            import pygraphviz
            return True
        except:
            return False


# ============================================================================
# BEISPIEL-VERWENDUNG
# ============================================================================

if __name__ == "__main__":
    
    # Option 1: Lade aus JSON
    visualizer = RobustnessGraphVisualizer()
    
    try:
        visualizer.load_from_json('robustness_coffee.json')
    except FileNotFoundError:
        print("Erstelle Demo-Daten...")
        # Demo-Daten für Visualisierung
        demo_results = {
            'all_objects': {
                'boundary': ['Timer/Clock', 'User Notification', 'Settings Interface'],
                'entity': ['Water', 'Milk', 'Coffee Beans', 'Sugar', 'Cup', 'Filter'],
                'control': ['Water Heater Controller', 'Grinder Controller', 
                           'Brew Controller', 'Milk Controller', 'Orchestrator']
            },
            'shared_objects': {
                'boundary': ['Timer/Clock', 'User Notification'],
                'entity': ['Water', 'Coffee Beans', 'Cup', 'Filter'],
                'control': ['Orchestrator', 'Water Heater Controller', 'Grinder Controller']
            },
            'relationships': [
                {'from': 'Timer/Clock', 'to': 'Orchestrator', 'type': 'triggers', 'uc': 'UC1'},
                {'from': 'Orchestrator', 'to': 'Water Heater Controller', 'type': 'controls', 'uc': 'UC1'},
                {'from': 'Water Heater Controller', 'to': 'Water', 'type': 'manipulates', 'uc': 'UC1'},
                {'from': 'Orchestrator', 'to': 'Grinder Controller', 'type': 'controls', 'uc': 'UC1'},
                {'from': 'Grinder Controller', 'to': 'Coffee Beans', 'type': 'manipulates', 'uc': 'UC1'},
                {'from': 'Orchestrator', 'to': 'Brew Controller', 'type': 'controls', 'uc': 'UC1'},
                {'from': 'Brew Controller', 'to': 'Cup', 'type': 'uses', 'uc': 'UC1'},
                {'from': 'Orchestrator', 'to': 'User Notification', 'type': 'sends', 'uc': 'UC1'},
            ],
            'features': {
                'Coffee Preparation': {
                    'Latte-Feature': 'UC1: Prepare Latte',
                    'Espresso-Feature': 'UC2: Prepare Espresso'
                }
            },
            'uc_objects': {
                'UC1: Prepare Latte': {
                    'boundary': ['Timer/Clock', 'User Notification', 'Settings Interface'],
                    'entity': ['Water', 'Milk', 'Coffee Beans', 'Cup', 'Filter'],
                    'control': ['Water Heater Controller', 'Grinder Controller', 
                               'Brew Controller', 'Milk Controller', 'Orchestrator']
                },
                'UC2: Prepare Espresso': {
                    'boundary': ['Timer/Clock', 'User Notification'],
                    'entity': ['Water', 'Coffee Beans', 'Cup', 'Filter'],
                    'control': ['Water Heater Controller', 'Grinder Controller', 
                               'Brew Controller', 'Orchestrator']
                }
            }
        }
        visualizer.results = demo_results
    
    # Baue Graph
    visualizer.build_graph()
    
    # Erstelle Visualisierungen
    print("\n" + "="*60)
    print("ERSTELLE VISUALISIERUNGEN")
    print("="*60)
    
    # 1. Hierarchische Ansicht
    visualizer.visualize_hierarchical('robustness_hierarchical.png')
    
    # 2. Use-Case Ansicht
    visualizer.visualize_by_usecase('robustness_by_uc.png')
    
    # 3. Feature Tree
    visualizer.visualize_feature_tree('robustness_features.png')
    
    # 4. Exportiere in verschiedenen Formaten
    visualizer.export_graph_formats('robustness_graph')
    
    print("\n✓ Alle Visualisierungen erstellt!")
    print("\nErstelle Dateien:")
    print("  • robustness_hierarchical.png - Hierarchische Ansicht")
    print("  • robustness_by_uc.png - Use-Case gruppiert")
    print("  • robustness_features.png - Feature-Hierarchie")
    print("  • robustness_graph.graphml - Für yEd/Gephi")
    print("  • robustness_graph.gml - Graph Markup Language")