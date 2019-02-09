import json
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from data_ingestion import get_characters
from data_ingestion import get_scenes
def draw_graph(g):
     # nodes
    nx.draw(g)

    # edges

    plt.axis('off')
    plt.show() # display

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
    
def add_attribute(dataset, new_attribute):
    for k,v in new_attribute.items():
        dataset[k] = dataset[k] + [v]
    return dataset

def prepare_dataset(g):
    dataset = {}

    characters = get_characters()
    dataset = {}
    degree = g.degree() 
    for deg in degree:
        dataset[deg[0]] = [deg[1]]

    weighted_degree = g.degree(weight='weight')
    for wd in weighted_degree:
        dataset[wd[0]] = dataset[wd[0]] + [wd[1]]
    # Node metrics of the graph
    betweeness_centrlity = nx.betweenness_centrality(g)
    pagerank = nx.pagerank(g)
    clustering = nx.clustering(g, weight='weight')
    eigen_centrality = nx.eigenvector_centrality(g)
    closeness_centrality = nx.closeness_centrality(g)

    dataset = add_attribute(dataset, betweeness_centrlity)
    dataset = add_attribute(dataset, pagerank)
    dataset = add_attribute(dataset, clustering)
    dataset = add_attribute(dataset, eigen_centrality)
    dataset = add_attribute(dataset, closeness_centrality)
    
    for ch in characters:
        try:
            if ch.get('killedBy'):
                if ch['characterName'] == 'Jon Snow' or ch['characterName'] == 'Benjen Stark':
                    # Corner cases
                    dataset[ch['characterName']] = dataset[ch['characterName']] + [True]     
                else:
                    dataset[ch['characterName']] = dataset[ch['characterName']] + [False]
            else:
                dataset[ch['characterName']] = dataset[ch['characterName']] + [True]
        except KeyError:
            # Flashback characteers, not relevant etc.
            pass
    
    to_remove = []
    for k,v in dataset.items():
        if len(v) != 8:
            # Unknown fate, therefore not useful for model
            to_remove.append(k)
    for k in to_remove:
        dataset.pop(k)


    return dataset

def main():
    scenes = get_scenes()
    g = graph_creation(scenes)

    dataset = prepare_dataset(g)


#    draw_graph(g)

if __name__ == "__main__":
    main()