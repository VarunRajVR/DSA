import heapq
graph = {
    'a': ['b', 'c'],
    'b': ['d'],
    'c': ['e'],
    'd': ['f'],
    'e': [],
    'f': [],
    'g': []

}
matrix = [
    [1, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 1],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 0, 0, 1, 1, 1]
]


def dfs (graph, start):
    stack = [start]
    while stack:
        current = stack.pop()
        print(current)
        for i in graph[current]:
            stack.append(i)

def bfs (graph, start):
    queue = [start]
    while queue:
        current = queue.pop(0)
        print(current)
        for i in graph[current]:
            queue.append(i)

def dfs_recur(graph, start):
    print(start)
    for i in graph[start]:
        dfs_recur(graph, i)

def bfs_recur(graph, queue, visited=None):
    if visited is None:
        visited = set()
    if not queue:
        return

    vertex = queue.pop(0)
    # Visit the vertex if it hasn't been visited yet
    if vertex not in visited:
        print(vertex, end=" ")
        visited.add(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                queue.append(neighbor)

    # Recursive call to process the next level
    bfs_recur(graph, queue, visited)

def has_path(graph, src, dest):
    if src ==dest: return True
    for i in graph[src]:
        if has_path(graph, i, dest) ==True:
            return True
    
    return False

def connected_components(graph):
    visited=set()
    count = 0
    for i in graph:
        if explore(graph, i, visited) == True:
            count+=1
    return count
#for no of comp
def explore(graph, node, visited):
    if node in visited: return False
    visited.add(node)
    for i in graph[node]:
        explore(graph, i, visited)
    return True

#for largest comp
def explore_size(grah, node, visited):
    if node in visited: return False
    visited.add(node)
    count =1
    for i in graph[node]:
        count+= explore_size(graph, i, visited)
    return count


def largest_component(graph):
    visited=set()
    largest=0
    for i in graph:
        size = explore_size(graph, i, visited)
        largest= max(size, largest)
    return largest

def egde_to_adj(edge_list):
    graph ={[] for i in len(edge_list)}
    for i, j in edge_list:
        graph[i].append(j)
        graph[j].append(i)
    return graph
    

def shortest_path(graph, src, dest):
    queue = [(src, 0)]
    visited=set(src)

    while queue:
        [node,distance]= queue.pop(0)
        if node == dest: return distance
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append([neighbor, distance+1])
        
    return -1

def island_count(matrix):
    visited =set()
    count=0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if explore_grid(matrix, i, j, visited):
                count+=1
    return count 

def explore_grid(matrix,i , j, visited):
    #check if the neighbour is in bounds
    row_inbound = 0 <= i < len(matrix)
    col_inbound = 0 <= j < len(matrix[0])
    if not col_inbound or not row_inbound: return False
    #check for water
    if matrix[i][j] == 0: return False
    # # for consistency in the sets, we stringify it. 
    pos = str(i)+ ','+ str(j)
    if pos in visited: return False
    visited.add(pos)
    # explore_grid(matrix, i-1, j, visited)
    # explore_grid(matrix, i+1, j, visited)
    # explore_grid(matrix, i, j-1, visited)
    # explore_grid(matrix, i, j+1, visited)

    #size of island 
    size =1
    size+= explore_grid(matrix, i-1, j, visited)
    size+= explore_grid(matrix, i+1, j, visited)
    size+= explore_grid(matrix, i, j-1, visited)
    size+= explore_grid(matrix, i, j+1, visited)

    return size

def smallest_island(matrix):
    visited =set()
    explore_size = float('inf')
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            island_size = explore_grid(matrix, i, j, visited)
            if island_size > 0:
                explore_size = min(island_size, explore_size)      
    return explore_size if explore_size != float('inf') else 0

 

# Note: use a set to avoid infinite loops with cycle.
# Recursion
# bfs_recur(graph, ['a'])
# dfs_recur(graph,'a')

#Recursion
# dfs(graph, 'a')
# # bfs(graph, 'a')

#has path
# print(has_path(graph,'f', 'e'))
# print(has_path(graph,'b', 'f'))

# #connected components. 
# print(connected_components(graph))

# #largest component
# print(largest_component(graph))

# #shortest distance:
# print(shortest_path(graph, 'a', 'f'))

# #island problems:
# print(island_count(matrix))
# print(smallest_island(matrix))



