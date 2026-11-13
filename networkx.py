import networkx as nx

G = nx.cycle_graph(4)

centrality = nx.eigenvector_centrality_numpy(G)

print(centrality)
