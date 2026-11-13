import networkx as nx

G = nx.Graph()

G.add_edges_from([
    ("Alice", "Bob"),
    ("Alice", "Charlie"),
    ("Bob", "Diana"),
    ("Charlie", "Eve"),
    ("Diana", "Eve"),
])

print("All people:", list(G.nodes()))
print("Alice's friends:", list(G.neighbors("Alice")))

path = nx.shortest_path(G, source="Alice", target="Eve")
print("Shortest path from Alice to Eve:", " -> ".join(path))
print("Number of connections:", len(path) - 1)
