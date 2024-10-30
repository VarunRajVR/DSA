import heapq

node_map = { 'A': 0, 'B': 1, 'C': 2, 'D': 3 , 'E': 4, 'F':5} # for floyyd warshall algo
# Example graph 1
weighted_graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('A', 4), ('C', 5), ('D', 10)],
    'C': [('A', 2), ('B', 5), ('E', 3)],
    'D': [('B', 10), ('E', 11)],
    'E': [('C', 3), ('D', 11), ('F', 1)],
    'F': [('E', 1)]
}

# Example graph 2- for kruskal
vertices = ['A', 'B', 'C', 'D', 'E', 'F']
edges = [
    (1, 'A', 'B'),
    (4, 'A', 'C'),
    (3, 'A', 'D'),
    (2, 'B', 'E'),
    (4, 'B', 'C'),
    (5, 'C', 'F'),
    (2, 'D', 'C'),
    (7, 'D', 'F'),
    (3, 'E', 'F'),
]

def dijkstra(weighted_graph, src):
    shortest_path = {}
    minHeap = [[0, src]]
    
    while minHeap:
        w1, v1 = heapq.heappop(minHeap)
        
        if v1 in shortest_path:
            continue
        
        shortest_path[v1] = w1
        
        for v2, w2 in weighted_graph[v1]:
            if v2 not in shortest_path:
                heapq.heappush(minHeap, [w1 + w2, v2])
    
    # After processing all nodes, ensure that every node is in shortest_path
    n = len(weighted_graph)
    for node in weighted_graph:
        if node not in shortest_path:
            shortest_path[node] = float('inf')  # Use infinity to denote unreachable nodes
    
    return shortest_path

def bellman_ford(weighted_graph, src):
    relaxed = False
    # Step 1: Initialize distances from src to all other vertices as infinity
    distance = {node: float('inf') for node in weighted_graph}
    distance[src] = 0

    # Step 2: Relax all edges |V| - 1 times
    for _ in range(len(weighted_graph) - 1):
        for node in weighted_graph:# for each node
            for neighbor, weight in weighted_graph[node]: # for each neighbor
                if distance[node] + weight < distance[neighbor]: # relaxation condition
                    distance[neighbor] = distance[node] + weight # relaxation
                    relaxed = True
        if not relaxed:
            break


    # Step 3: Check for negative-weight cycles
    for node in weighted_graph:
        for neighbor, weight in weighted_graph[node]:
            if distance[node] + weight < distance[neighbor]:
                print("Graph contains a negative-weight cycle")
                return None

    return distance



def floyd_warshall_from_adj_list(adj_list):
    V = len(weighted_graph)
    #  Initialize the distance matrix with infinity
    dist = [[float('inf')] * V for _ in range(V)]
    
    # Set the diagonal to 0 (distance from a node to itself)
    for i in range(V):
        dist[i][i] = 0
    
    #  Convert adjacency list to distance matrix
    for node in adj_list:
        node_index = node_map[node]  # Convert node label to index
        for neighbor, weight in adj_list[node]:
            neighbor_index = node_map[neighbor]  # Convert neighbor label to index
            dist[node_index][neighbor_index] = weight

    #  Update the distance matrix using the Floyd-Warshall algorithm
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    #  Check for negative-weight cycles
    for i in range(V):
        if dist[i][i] < 0:
            print("Graph contains a negative-weight cycle")
            return None

    return dist


# shortest_paths = floyd_warshall_from_adj_list(weighted_graph)
# if shortest_paths:
#     for row in shortest_paths:
#         print(row)

# shortest_paths = bellman_ford(weighted_graph, 'A')
# print(f"Bellman solution:{shortest_paths}")


# shortest_paths = dijkstra(weighted_graph, 'A')
# print(shortest_paths)


def prims_algorithm(graph, start_node):
    # Initialize the minimum spanning tree (MST) and a priority queue
    mst = []
    visited = set()
    min_heap = [(0, start_node, None)]  # (weight, current_node, previous_node)

    total_weight = 0

    while min_heap:
        weight, current_node, prev_node = heapq.heappop(min_heap)

        # Skip this node if it has already been visited
        if current_node in visited:
            continue

        # Mark this node as visited
        visited.add(current_node)

        # If prev_node is not None, this means it's part of the MST
        if prev_node is not None:
            mst.append((prev_node, current_node, weight))
            total_weight += weight

        # Explore the neighbors of the current node
        for neighbor, edge_weight in graph[current_node]:
            if neighbor not in visited:
                heapq.heappush(min_heap, (edge_weight, neighbor, current_node))

    return mst, total_weight


# mst, total_weight = prims_algorithm(weighted_graph, 'A')
# print("Minimum Spanning Tree:", mst)
# print("Total Weight:", total_weight)

#krsukal
class DisjointSet:
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices} # incremented only when rank collision occurs. 

    def find(self, item):
        if self.parent[item] != item: #only root will have the parent as its own. 
            self.parent[item] = self.find(self.parent[item])  # Path compression
        return self.parent[item]

    def union(self, set1, set2):
        root1 = self.find(set1)
        root2 = self.find(set2)
        # ensuring no cycles are formed. 
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1


def kruskal_algorithm(vertices, edges):
    # Step 1: Sort all edges by weight nlogn time
    edges.sort()

    # Step 2: Initialize the disjoint set
    disjoint_set = DisjointSet(vertices)

    mst = []
    total_weight = 0

    # Step 3: Iterate over the sorted edges and build the MST
    for weight, u, v in edges:
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            mst.append((u, v, weight))
            total_weight += weight

    return mst, total_weight

mst, total_weight = kruskal_algorithm(vertices, edges)
print("Minimum Spanning Tree:", mst)
print("Total Weight:", total_weight)

