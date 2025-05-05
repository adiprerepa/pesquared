import networkx as nx
import matplotlib.pyplot as plt

class DAGish(nx.DiGraph):
    """
    A directed graph that ensures every node has a label and can be plotted
    with nodes positioned horizontally based on their level in the SCC condensation.
    """
    
    def add_node(self, node_for_adding, **attr):
        """Add a node to the graph, ensuring it has a label."""
        if 'label' not in attr:
            raise ValueError(f"Node {node_for_adding} must have a label")
        super().add_node(node_for_adding, **attr)
    
    def add_nodes_from(self, nodes_for_adding, **attr):
        """Add multiple nodes to the graph, ensuring each has a label."""
        for n in nodes_for_adding:
            if isinstance(n, tuple) and len(n) > 1:
                # n is (node, attr_dict) tuple
                node, node_attr = n
                merged_attr = attr.copy()
                merged_attr.update(node_attr)
                if 'label' not in merged_attr:
                    raise ValueError(f"Node {node} must have a label")
            else:
                # n is just a node identifier
                if 'label' not in attr:
                    raise ValueError(f"Node {n} must have a label")
        
        super().add_nodes_from(nodes_for_adding, **attr)
    
    def label_nodes_with_levels(self):
        """
        Label nodes with their levels in the SCC condensation.
        """
        # Find strongly connected components
        components = list(nx.strongly_connected_components(self))
        component_map = {}
        for i, component in enumerate(components):
            for node in component:
                component_map[node] = i
        
        # Create condensation graph
        condensation = nx.condensation(self)
        
        # Use topological generations to get levels
        levels = {}
        try:
            for level, generation in enumerate(nx.topological_generations(condensation)):
                for scc_id in generation:
                    levels[scc_id] = level
        except nx.NetworkXUnfeasible:
            # Graph has cycles between SCCs (should not happen after condensation)
            raise ValueError("Cannot determine levels: condensation graph has cycles")
        
        # Assign levels to nodes
        for node in self.nodes():
            scc_id = component_map[node]
            level = levels[scc_id]
            self.nodes[node]['level'] = level

    
    def plot(self, figsize=(10, 6), node_size=500, font_size=10, vertical_spacing=1.0, title=None, **kwargs):
        """
        Plot the graph with nodes positioned horizontally based on their level in the SCC condensation.
        
        Parameters:
        -----------
        figsize : tuple
            Size of the figure (width, height)
        node_size : int
            Size of the nodes in the plot
        font_size : int
            Size of the font for node labels
        vertical_spacing : float
            Spacing between nodes in the same level
        title : str, optional
            Title for the plot
        **kwargs : dict
            Additional arguments passed to nx.draw
        """
        # Find strongly connected components
        components = list(nx.strongly_connected_components(self))
        component_map = {}
        for i, component in enumerate(components):
            for node in component:
                component_map[node] = i
        
        # Create condensation graph
        condensation = nx.condensation(self)
        
        # Use topological generations to get levels
        levels = {}
        try:
            for level, generation in enumerate(nx.topological_generations(condensation)):
                for scc_id in generation:
                    levels[scc_id] = level
        except nx.NetworkXUnfeasible:
            # Graph has cycles between SCCs (should not happen after condensation)
            raise ValueError("Cannot determine levels: condensation graph has cycles")
        
        # Assign positions to nodes
        pos = {}
        nodes_at_level = {}  # Track nodes at each level for vertical spacing
        
        for node in self.nodes():
            scc_id = component_map[node]
            level = levels[scc_id]
            
            if level not in nodes_at_level:
                nodes_at_level[level] = 0
            else:
                nodes_at_level[level] += 1
            
            pos[node] = (level, -nodes_at_level[level] * vertical_spacing)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Draw the graph
        nx.draw(
            self, 
            pos=pos, 
            with_labels=True,
            labels={n: data.get('label', str(n)) for n, data in self.nodes(data=True)},
            node_size=node_size,
            font_size=font_size,
            **kwargs
        )
        
        # Add title if provided
        if title:
            plt.title(title)
        
        plt.tight_layout()
        plt.show()