import networkx as nx

G = nx.path_graph(4)

centrality = nx.eigenvector_centrality_numpy(G)

for node, value in centrality.items():
    print(f"Node {node}: {value:.4f}")
