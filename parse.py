import numpy as np
import random
from queue import deque

from load import load
from Graph import Graph

DATA_DIR = 'data'


def sample(graph: Graph, tag: str, max_nodes=100, dfs=0.5, dense_drop=0., sparse_drop=0., seed=123):
    random.seed = seed
    np.random.seed(seed)
    node_set = set(range(graph.N_NODES))
    picked = set()
    q = deque()

    while len(picked) < max_nodes and len(node_set) > 0:
        if len(q) == 0:
            q.extend(random.sample(node_set, 1))
        n = q.popleft()
        if n not in node_set:
            continue
        neighbors = graph.neighbor_sets[n]
        node_set.remove(n)
        if len(neighbors) <= graph.DEGREE and random.random() < sparse_drop:
            continue
        if len(neighbors) > graph.DEGREE and random.random() < dense_drop:
            continue

        picked.add(n)
        vs = neighbors & node_set
        for v in vs:
            if random.random() < dfs:
                q.appendleft(v)
            else:
                q.append(v)

    raw_nodes = list(picked)
    with open('{}/{}.content'.format(DATA_DIR, tag), 'w+') as o:
        for i, n in enumerate(raw_nodes):
            o.write('{} {} {}\n'.format(i,
                                        ' '.join(['{:.3f}'.format(i) for i in graph.node_features[n]]),
                                        graph.labels[n]))

    raw_links = []
    for u, v in graph.links:
        if u in picked and v in picked:
            raw_links.append((u, v))

    raw_new_node_map = {n: i for i, n in enumerate(raw_nodes)}
    with open('{}/{}.cites'.format(DATA_DIR, tag), 'w+') as o:
        for u, v in raw_links:
            o.write('{} {}\n'.format(raw_new_node_map[u], raw_new_node_map[v]))

    print('For {}, {} nodes and {} links with degree {:.1f}.'.format(tag,
                                                                     len(raw_nodes),
                                                                     len(raw_links),
                                                                     len(raw_links) / len(raw_nodes)))


if __name__ == '__main__':
    graph = load()
    sample(graph, 'reddit', max_nodes=50000)
    sample(graph, 'reddit_dfs', dfs=1., max_nodes=25000)
    sample(graph, 'reddit_bfs', dfs=0., max_nodes=25000)
    sample(graph, 'reddit_dense', sparse_drop=0.5, max_nodes=30000)
    sample(graph, 'reddit_sparse', dense_drop=0.8, max_nodes=35000)
