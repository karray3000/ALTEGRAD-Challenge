import pickle
import networkx as nx

def get_shortest_paths():
    try:
        shortest_paths = pickle.load(open('./data/shortest_paths.pkl', 'rb'))
    except:
        G = nx.read_weighted_edgelist('./data/edgelist.txt', create_using=nx.DiGraph())
        shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G))
        pickle.dump(shortest_paths, open('./data/shortest_paths.pkl', 'wb'))
    return shortest_paths