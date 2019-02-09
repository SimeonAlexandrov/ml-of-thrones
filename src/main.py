import json
import itertools
import networkx as nx
import matplotlib.pyplot as plt

from data_ingestion import get_characters
from data_ingestion import get_scenes


def graph_creation(scenes):
    g = nx.Graph()
    # g.add_edge(1,2)
    # print(g.has_edge(1,2))
    # print(g.has_edge(3,5))
    # scenes = scenes[2000:3000]

    for scene in scenes:
        character_names = [ ch['name'] for ch in scene['characters']]

        if len(character_names) > 1:
            for interaction in itertools.combinations(character_names, 2):
                if g.has_edge(interaction[0], interaction[1]):
                    g[interaction[0]][interaction[1]]['weight'] += 1
                else:
                    g.add_edge(interaction[0], interaction[1], weight=1)
    return g
    


def main():
    scenes = get_scenes()

    g = graph_creation(scenes)
    print('Vertexes: %s' % g.number_of_nodes())
    print('Edges num: %s' % g.number_of_edges())
    edges = g.edges.data('weight')
    max_connection = sorted(edges, key=lambda x: x[2], reverse=True)[:10]
    print('Connections:')
    print(max_connection)
    print('Degree:')
    print(sorted(g.degree(), key=lambda x: x[1], reverse=True)[:10])
    print('Weighted degree:')
    print(sorted(g.degree(weight='weight'), key=lambda x: x[1], reverse=True)[:10])
    # print('Betweeness centrality:')
    # betweeness_centrlity = nx.betweenness_centrality(g)
    # # print(betweeness_centrlity)
    # pagerank = nx.pagerank(g)
    # # print(pagerank)

    # clustering = nx.clustering(g, weight='weight')
    # print(clustering)

    # eigen_centrality = nx.eigenvector_centrality(g)
    # print(eigen_centrality)

    closeness_centrality = nx.closeness_centrality(g)
    print(closeness_centrality)

    # nodes
    nx.draw(g)

    # edges

    plt.axis('off')
    # plt.show() # display

    # for node in g.nodes():
    #     print(node)
    # living = []
    # dead = []
    # for ch in characters:
    #     if ch.get('killedBy'):
    #         dead.append(ch)
    #     else:
    #         living.append(ch)
    # print(len(scenes))
    # print('Living: %s   Dead: %s' % (len(living), len(dead)))
if __name__ == "__main__":
    main()