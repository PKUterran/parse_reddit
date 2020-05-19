import numpy as np


class Graph:
    def __init__(self, node_features: np.ndarray, link_features: np.ndarray, links: np.ndarray, labels: np.ndarray):
        self.N_NODES = node_features.shape[0]
        self.node_features = node_features
        self.link_features = link_features
        self.links = links
        self.labels = labels
        self.OFFSET = self.N_NODES

        self.neighbor_sets = [set() for _ in range(self.N_NODES)]
        # self.nodes_link_map = {}
        for i, (s, t) in enumerate(links):
            self.neighbor_sets[s].add(t)
            self.neighbor_sets[t].add(s)
            # self.nodes_link_map[s * self.OFFSET + t] = i
            # self.nodes_link_map[t * self.OFFSET + s] = i

        self.DEGREE = np.average(list(map(lambda x: len(x), self.neighbor_sets)))
        print('Average degree:', self.DEGREE)

    # def get_link_id_by_nodes(self, s, t):
    #     return self.nodes_link_map[s * self.OFFSET + t]
