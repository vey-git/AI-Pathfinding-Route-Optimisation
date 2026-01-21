import math
import osmnx as ox
import networkx as nx
import time
import tracemalloc
import matplotlib.pyplot as plt
import heapq


def DetermineNetworkType():
    choice = input("Are you walking or driving?: ").lower()
    if choice == "walking":
        network_type = 'walk'
    else:
        network_type = 'drive'
    print(f"...Evaluating for '{network_type}' network type.")
    return network_type


def locationArea():
    distance = input("\nAre you looking to look for a wide or small distance?: ")
    print("\nEvaluating best map location for your input")
    if distance == "wide":
        return "Oxfordshire, UK"
    else:
        return "Oxford, UK"


locations = [
    "Radcliffe Camera, Oxford",
    "Ashmolean Museum, Oxford",
    "Oxford Railway Station",
    "Bicester Village, Bicester",
    "Reading Station, Reading, UK",
    "Milton Keynes Central railway station",
    "Buckingham Palace, London",
    "Carfax Tower, Oxford",
    "Magdalen Bridge, Oxford",
    "Cowley Road, Oxford"
]

nxMap = ox.graph_from_place(locationArea(), network_type=DetermineNetworkType())

location_nodes = {}
for loc in locations:
    try:
        coords = ox.geocode(loc)
        lat, lon = coords
        node_id = ox.nearest_nodes(nxMap, X=lon, Y=lat)
        location_nodes[loc] = node_id
    except Exception as e:
        print(f"Could not process location {loc}: {e}")


node_to_location_name = {node_id: name for name, node_id in location_nodes.items()}

for u, v, k, data in nxMap.edges(data=True, keys=True):
    if 'highway' in data and data['highway'] in ['motorway', 'trunk']:
        data['constraint_highway'] = True

    road_name = data.get('name')
    if road_name == 'Cowley Road':
        data['constraint_unsafe'] = True
    elif road_name in ['Dangerous Road Name', 'Risky Alley']:
        data['constraint_unsafe'] = True

    unsafe_node_1 = location_nodes["Cowley Road, Oxford"]
    nxMap.nodes[unsafe_node_1]['is_near_unsafe_area'] = True
    print(f"Tagged node {unsafe_node_1} (Cowley Road) as 'is_near_unsafe_area'")



def BFS(graph, start, end):
    print("\n****USING BREADTH-FIRST-SEARCH****")
    queue = [[start]]
    explored = {start}
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == end:
            return path, explored
        for neighbourID in graph.neighbors(node):
            if neighbourID not in explored:
                explored.add(neighbourID)
                new_path = list(path)
                new_path.append(neighbourID)
                queue.append(new_path)
    return None, explored


def DFS(graph, start, end):
    print("\n\n****USING DEPTH-FIRST-SEARCH****")
    frontier = [[start]]
    explored = {start}
    while frontier:
        path = frontier.pop(-1)
        node = path[-1]
        if node == end:
            return path, explored
        for neighbourID in graph.neighbors(node):
            if neighbourID not in explored:
                explored.add(neighbourID)
                new_path = list(path)
                new_path.append(neighbourID)
                frontier.append(new_path)
    return None, explored


def heuristic(graph, node1_id, node2_id):
    if node1_id not in graph.nodes or node2_id not in graph.nodes: return float('inf')
    n1 = graph.nodes[node1_id]
    n2 = graph.nodes[node2_id]
    if 'y' not in n1 or 'x' not in n1 or 'y' not in n2 or 'x' not in n2: return float('inf')
    return math.sqrt((n1['y'] - n2['y']) ** 2 + (n1['x'] - n2['x']) ** 2)


def logic_aware_heuristic(graph, node1_id, node2_id):
    h_cost = heuristic(graph, node1_id, node2_id)
    if node1_id in graph.nodes:
        node_data = graph.nodes[node1_id]
        if node_data.get('is_near_unsafe_area', False):
            h_cost += 10000
    return h_cost


def a_star(graph, start, end):
    print("\n\n****USING A* SEARCH****")
    frontier = [(0, 0, [start])]
    explored = set()

    while frontier:
        try:
            f_cost, g_cost, path = heapq.heappop(frontier)
        except IndexError: return None, explored
        node = path[-1]

        if node in explored: continue
        explored.add(node)

        if node == end:
            print(f"Path has been found! Total distance: {g_cost / 1000:.2f} km")
            return path, explored

        for neighbour_id in graph.neighbors(node):
            if neighbour_id not in explored:
                try: edge_data = graph.get_edge_data(node, neighbour_id)[0]
                except (KeyError, IndexError): continue
                distanceToNeighbour = edge_data.get('length', 0)
                if distanceToNeighbour <= 0: continue
                new_g_cost = g_cost + distanceToNeighbour
                h_cost = heuristic(graph, neighbour_id, end)
                if h_cost == float('inf'): continue
                new_f_cost = new_g_cost + h_cost
                new_path = path + [neighbour_id]
                heapq.heappush(frontier, (new_f_cost, new_g_cost, new_path))
    return None, explored


def a_star_with_logic(graph, start, end):
    print("\n\n****USING A* SEARCH with Logic****")
    frontier = [(0, 0, [start])]
    explored = set()

    while frontier:
        try: f_cost, g_cost, path = heapq.heappop(frontier)
        except IndexError: return None, explored
        node = path[-1]

        if node in explored: continue
        explored.add(node)

        if node == end:
            print(f"Path has been found! Total distance: {g_cost / 1000:.2f} km")
            return path, explored

        for neighbour_id in graph.neighbors(node):
            if neighbour_id not in explored:
                try: edge_data = graph.get_edge_data(node, neighbour_id)[0]
                except (KeyError, IndexError): continue
                if edge_data.get('constraint_highway', False): continue
                if edge_data.get('constraint_unsafe', False): continue
                distanceToNeighbour = edge_data.get('length', 0)
                if distanceToNeighbour <= 0: continue
                new_g_cost = g_cost + distanceToNeighbour
                h_cost = logic_aware_heuristic(graph, neighbour_id, end)
                if h_cost == float('inf'): continue
                new_f_cost = new_g_cost + h_cost
                new_path = path + [neighbour_id]
                heapq.heappush(frontier, (new_f_cost, new_g_cost, new_path))
    return None, explored


def calculatMM_Priority(g_cost, h_cost):
    f_cost = g_cost + h_cost
    if h_cost == float('inf'): return float('inf')
    priority = max(f_cost, 2 * g_cost)
    return priority


def bidirectionalMM(graph, start, end):
    print("\n\n****USING BIDIRECTIONAL MM SEARCH****")
    open_forward = [(0, 0, start, [start])]
    open_backward = [(0, 0, end, [end])]

    g_costs_forward = {start: 0}
    g_costs_backward = {end: 0}

    paths_forward = {start: [start]}
    paths_backward = {end: [end]}

    best_path_cost = float('inf')
    best_path = None

    nodes_visited_count = 0

    while open_forward and open_backward:
        if not open_forward or not open_backward: break
        min_priority_f = open_forward[0][0]
        min_priority_b = open_backward[0][0]
        if min_priority_f == float('inf') and min_priority_b == float('inf'): break

        if (min_priority_f + min_priority_b) >= best_path_cost:
            if best_path:
                print(f"Optimal path found! Cost: {best_path_cost / 1000:.2f} km")
                return best_path, nodes_visited_count
            else: break

        if min_priority_f <= min_priority_b:
            try: _, g_cost, current_node, path = heapq.heappop(open_forward)
            except IndexError: break
            nodes_visited_count += 1

            if current_node in g_costs_backward:
                path_cost = g_cost + g_costs_backward[current_node]
                if path_cost < best_path_cost:
                    best_path_cost = path_cost
                    best_path = path + paths_backward[current_node][::-1][1:]

            for neighbor_id in graph.neighbors(current_node):
                try: edge_data = graph.get_edge_data(current_node, neighbor_id)[0]
                except (KeyError, IndexError): continue
                distanceToNeighbour = edge_data.get('length', 0)
                if distanceToNeighbour <= 0: continue
                new_g_cost = g_cost + distanceToNeighbour

                if new_g_cost < g_costs_forward.get(neighbor_id, float('inf')):
                    g_costs_forward[neighbor_id] = new_g_cost
                    new_path = path + [neighbor_id]
                    paths_forward[neighbor_id] = new_path
                    h_cost = heuristic(graph, neighbor_id, end)
                    if h_cost == float('inf'): continue
                    mm_priority = calculatMM_Priority(new_g_cost, h_cost)
                    if mm_priority != float('inf'): heapq.heappush(open_forward, (mm_priority, new_g_cost, neighbor_id, new_path))

        else:
            try: _, g_cost, current_node, path = heapq.heappop(open_backward)
            except IndexError: break
            nodes_visited_count += 1

            if current_node in g_costs_forward:
                path_cost = g_cost + g_costs_forward[current_node]
                if path_cost < best_path_cost:
                    best_path_cost = path_cost
                    best_path = paths_forward[current_node] + path[::-1][1:]

            for neighbor_id in graph.predecessors(current_node):
                try: edge_data = graph.get_edge_data(neighbor_id, current_node)[0]
                except (KeyError, IndexError): continue
                distanceToNeighbour = edge_data.get('length', 0)
                if distanceToNeighbour <= 0: continue
                new_g_cost = g_cost + distanceToNeighbour

                if new_g_cost < g_costs_backward.get(neighbor_id, float('inf')):
                    g_costs_backward[neighbor_id] = new_g_cost
                    new_path = path + [neighbor_id]
                    paths_backward[neighbor_id] = new_path
                    h_cost = heuristic(graph, neighbor_id, start)
                    if h_cost == float('inf'): continue
                    mm_priority = calculatMM_Priority(new_g_cost, h_cost)
                    if mm_priority != float('inf'): heapq.heappush(open_backward, (mm_priority, new_g_cost, neighbor_id, new_path))

    return best_path, nodes_visited_count

startLocation = "Radcliffe Camera, Oxford"
endLocation = "Ashmolean Museum, Oxford"

start_node_id = location_nodes.get(startLocation)
end_node_id = location_nodes.get(endLocation)

if start_node_id is None or end_node_id is None:
    print(f"\nError: Could not find nodes for start ('{startLocation}') or end ('{endLocation}') locations.")
    exit()
else:
    print(f"\nRunning searches between {startLocation} ({start_node_id}) and {endLocation} ({end_node_id})...")
    a_star_path, _ = a_star(nxMap, start_node_id, end_node_id)
    if a_star_path:
        print("\nVisualizing the optimal A* path...")
        try: ox.plot_graph_route(nxMap, a_star_path, route_color='cyan', node_size=0)
        except Exception as e: print(f"Could not plot A* path: {e}")
    dfs_path, _ = DFS(nxMap, start_node_id, end_node_id)
    if dfs_path: print("\n--- DFS Route ---")
    else: print("DFS path not found.")
    bfs_path, _ = BFS(nxMap, start_node_id, end_node_id)
    if bfs_path: print("\n--- BFS Route ---")
    else: print("BFS path not found.")

num_nodes = len(nxMap.nodes)
num_edges = len(nxMap.edges)
print(f"\nThe graph has {num_nodes} nodes and {num_edges} edges.")

all_metrics = {
    'A* Search': {},
    'BFS': {},
    'DFS': {},
    'Bidirectional MM': {},
    'A* Search with Logic': {}
}

print("\n" + "=" * 30)
print("PERFORMANCE METRICS")
print("=" * 30)

tracemalloc.start()
startTime = time.time()
a_star_path, a_star_explored = a_star(nxMap, start_node_id, end_node_id)
endTime = time.time()
current, peak_memory = tracemalloc.get_traced_memory()
tracemalloc.stop()

exec_time = endTime - startTime
nodes_visited = len(a_star_explored) if a_star_explored else 0
path_length_km = 0
if a_star_path:
    try:
        path_length_meters = nx.path_weight(nxMap, a_star_path, weight='length')
        path_length_km = path_length_meters / 1000.0
    except (nx.NetworkXError, KeyError, TypeError) as e: path_length_km = -1
peak_memory_mb = peak_memory / 1024 ** 2

all_metrics['A* Search']['time'] = exec_time
all_metrics['A* Search']['nodes'] = nodes_visited
all_metrics['A* Search']['length'] = path_length_km
all_metrics['A* Search']['memory'] = peak_memory_mb

print(f"\n--- A* Results ---")
print(f"Execution Time: {exec_time:.4f} seconds")
print(f"Nodes Visited: {nodes_visited}")
print(f"Path Length: {path_length_km:.2f} km")
print(f"Peak Memory: {peak_memory_mb:.2f} MB")

tracemalloc.start()
startTime = time.time()
a_star_logic_path, a_star_logic_explored = a_star_with_logic(nxMap, start_node_id, end_node_id)
endTime = time.time()
current, peak_memory = tracemalloc.get_traced_memory()
tracemalloc.stop()

exec_time = endTime - startTime
nodes_visited = len(a_star_logic_explored) if a_star_logic_explored else 0
path_length_km = 0
if a_star_logic_path:
    try:
        path_length_meters = nx.path_weight(nxMap, a_star_logic_path, weight='length')
        path_length_km = path_length_meters / 1000.0
    except (nx.NetworkXError, KeyError, TypeError) as e: path_length_km = -1
peak_memory_mb = peak_memory / 1024 ** 2

all_metrics['A* Search with Logic']['time'] = exec_time
all_metrics['A* Search with Logic']['nodes'] = nodes_visited
all_metrics['A* Search with Logic']['length'] = path_length_km
all_metrics['A* Search with Logic']['memory'] = peak_memory_mb

print(f"\n--- A* with Logic Results ---")
print(f"Execution Time: {exec_time:.4f} seconds")
print(f"Nodes Visited: {nodes_visited}")
print(f"Path Length: {path_length_km:.2f} km")
print(f"Peak Memory: {peak_memory_mb:.2f} MB")

tracemalloc.start()
startTime = time.time()
bfs_path, bfs_explored = BFS(nxMap, start_node_id, end_node_id)
endTime = time.time()
current, peak_memory = tracemalloc.get_traced_memory()
tracemalloc.stop()

exec_time = endTime - startTime
nodes_visited = len(bfs_explored) if bfs_explored else 0
path_length_km = 0
if bfs_path:
    try:
        path_length_meters = nx.path_weight(nxMap, bfs_path, weight='length')
        path_length_km = path_length_meters / 1000.0
    except (nx.NetworkXError, KeyError, TypeError) as e: path_length_km = -1
peak_memory_mb = peak_memory / 1024 ** 2

all_metrics['BFS']['time'] = exec_time
all_metrics['BFS']['nodes'] = nodes_visited
all_metrics['BFS']['length'] = path_length_km
all_metrics['BFS']['memory'] = peak_memory_mb

print(f"\n--- BFS Results ---")
print(f"Execution Time: {exec_time:.4f} seconds")
print(f"Nodes Visited: {nodes_visited}")
print(f"Path Length: {path_length_km:.2f} km")
print(f"Peak Memory: {peak_memory_mb:.2f} MB")

tracemalloc.start()
startTime = time.time()
dfs_path, dfs_explored = DFS(nxMap, start_node_id, end_node_id)
endTime = time.time()
current, peak_memory = tracemalloc.get_traced_memory()
tracemalloc.stop()

exec_time = endTime - startTime
nodes_visited = len(dfs_explored) if dfs_explored else 0
path_length_km = 0
if dfs_path:
    try:
        path_length_meters = nx.path_weight(nxMap, dfs_path, weight='length')
        path_length_km = path_length_meters / 1000.0
    except (nx.NetworkXError, KeyError, TypeError) as e: path_length_km = -1
peak_memory_mb = peak_memory / 1024 ** 2

all_metrics['DFS']['time'] = exec_time
all_metrics['DFS']['nodes'] = nodes_visited
all_metrics['DFS']['length'] = path_length_km
all_metrics['DFS']['memory'] = peak_memory_mb

print(f"\n--- DFS Results ---")
print(f"Execution Time: {exec_time:.4f} seconds")
print(f"Nodes Visited: {nodes_visited}")
print(f"Path Length: {path_length_km:.2f} km")
print(f"Peak Memory: {peak_memory_mb:.2f} MB")

tracemalloc.start()
startTime = time.time()
mm_path, mm_nodes_visited = bidirectionalMM(nxMap, start_node_id, end_node_id)
endTime = time.time()
current, peak_memory = tracemalloc.get_traced_memory()
tracemalloc.stop()

exec_time = endTime - startTime
nodes_visited = mm_nodes_visited
path_length_km = 0
if mm_path:
    try:
        path_length_meters = nx.path_weight(nxMap, mm_path, weight='length')
        path_length_km = path_length_meters / 1000.0
    except (nx.NetworkXError, KeyError, TypeError) as e: path_length_km = -1
peak_memory_mb = peak_memory / 1024 ** 2

all_metrics['Bidirectional MM']['time'] = exec_time
all_metrics['Bidirectional MM']['nodes'] = nodes_visited
all_metrics['Bidirectional MM']['length'] = path_length_km
all_metrics['Bidirectional MM']['memory'] = peak_memory_mb

print(f"\n--- Bidirectional MM Results ---")
print(f"Execution Time: {exec_time:.4f} seconds")
print(f"Nodes Visited: {nodes_visited}")
print(f"Path Length: {path_length_km:.2f} km")
print(f"Peak Memory: {peak_memory_mb:.2f} MB")

valid_algorithms = [algo for algo, metrics in all_metrics.items() if 'time' in metrics] # Basic check
if not valid_algorithms:
    print("\nNo algorithm results found. Skipping plots.")
else:
    print("\nGenerating plots...")
    execution_times = [all_metrics[algo]['time'] for algo in valid_algorithms]
    nodes_visited = [all_metrics[algo]['nodes'] for algo in valid_algorithms]
    path_lengths = [all_metrics[algo]['length'] for algo in valid_algorithms if all_metrics[algo]['length'] != -1]
    valid_algo_for_length = [algo for algo in valid_algorithms if all_metrics[algo]['length'] != -1]
    memory_usage = [all_metrics[algo]['memory'] for algo in valid_algorithms]

    bar_colors = ['blue', 'purple', 'orange', 'green', 'red'][:len(valid_algorithms)]

    if execution_times:
        plt.figure(figsize=(10, 6))
        plt.bar(valid_algorithms, execution_times, color=bar_colors)
        plt.title('Algorithm Performance: Execution Time', fontsize=16)
        plt.ylabel('Execution Time (seconds)', fontsize=12)
        plt.xlabel('Algorithm', fontsize=12)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.show()

    if nodes_visited:
        plt.figure(figsize=(10, 6))
        plt.bar(valid_algorithms, nodes_visited, color=bar_colors)
        plt.title('Algorithm Performance: Nodes Visited', fontsize=16)
        plt.ylabel('Number of Nodes Visited', fontsize=12)
        plt.xlabel('Algorithm', fontsize=12)
        plt.yscale('log')
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.show()

    if path_lengths:
        plt.figure(figsize=(10, 6))
        plt.bar(valid_algo_for_length, path_lengths, color=bar_colors[:len(valid_algo_for_length)])
        plt.title('Algorithm Performance: Path Length (km)', fontsize=16)
        plt.ylabel('Path Length (km)', fontsize=12)
        plt.xlabel('Algorithm', fontsize=12)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.show()

    if memory_usage:
        plt.figure(figsize=(10, 6))
        plt.bar(valid_algorithms, memory_usage, color=bar_colors)
        plt.title('Algorithm Performance: Peak Memory Usage (MB)', fontsize=16)
        plt.ylabel('Peak Memory (MB)', fontsize=12)
        plt.xlabel('Algorithm', fontsize=12)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.show()

