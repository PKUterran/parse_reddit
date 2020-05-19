import numpy as np
import json

from Graph import Graph

REDDIT_G = 'reddit/reddit-G.json'
REDDIT_LABEL_MAP = 'reddit/reddit-label-map.json'
REDDIT_LINK_FEATS = 'reddit/reddit-link-feats.npy'
REDDIT_LINK_ID_MAP = 'reddit/reddit-link-id-map.json'
REDDIT_NODE_FEATS = 'reddit/reddit-node-feats.npy'
REDDIT_NODE_ID_MAP = 'reddit/reddit-node-id-map.json'


def load() -> Graph:
    g = json.load(open(REDDIT_G))
    nodes = g['nodes']                                  # list(dict['id'(int), 'state'(str)])
    links = g['links']                                  # list(dict['id'(int), 'source'(int), 'target'(int)])
    label_map = json.load(open(REDDIT_LABEL_MAP))       # dict[str(int)]
    link_features = np.load(REDDIT_LINK_FEATS)          # np.ndarray with shape (1222411, 42)
    link_id_map = json.load(open(REDDIT_LINK_ID_MAP))   # dict[str(int)]
    node_features = np.load(REDDIT_NODE_FEATS)          # np.ndarray with shape (61836, 602)
    node_id_map = json.load(open(REDDIT_NODE_ID_MAP))   # dict[str(int)]

    lks = np.zeros([link_features.shape[0], 2], dtype=np.int)
    labels = np.zeros([node_features.shape[0]], dtype=np.int)
    for link in links:
        i = link['id']
        source = link['source']
        target = link['target']
        lks[i, 0] = source
        lks[i, 1] = target

    for i, label in label_map.items():
        labels[node_id_map[i]] = label

    graph = Graph(node_features, link_features, lks, labels)
    return graph


if __name__ == '__main__':
    g = load()
    print(g.node_features)
    print(g.link_features)
    print(g.links)
    print(g.labels)
