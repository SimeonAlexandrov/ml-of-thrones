import json
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from data_ingestion import get_characters
from data_ingestion import get_scenes
def draw_graph(g):
     # nodes
    nx.draw(g)
    plt.axis('off')
    plt.show() # display

def graph_creation(scenes):
    g = nx.Graph()
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

    # Panda's dataframes transformation
    df = pd.DataFrame.from_dict(dataset).T
    df.columns = [ 'degree', 'wdegree', 'betweeness_c', 'pagerank', 'clustering', 'eigen_centrality', 'closeness_c', 'is_alive']
    
    degree = df[['degree']].values.astype(float)
    min_max_scaler = MinMaxScaler()
    degree_scaled = min_max_scaler.fit_transform(degree)
    df['degree'] = degree_scaled

    wdegree = df[['wdegree']].values.astype(float)
    min_max_scaler = MinMaxScaler()
    wdegree_scaled = min_max_scaler.fit_transform(wdegree)
    df['wdegree'] = wdegree_scaled
 
    
    labels = df['is_alive']
    features = df.drop(['is_alive'], axis=1)
    return features, labels, df

def main():
    scenes = get_scenes()
    g = graph_creation(scenes)

    data, labels, raw = prepare_dataset(g)
    print('Dataset is ready.')

    prediction_res = []
    total_accuracy = []

    for i in range(3000):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
        clf=RandomForestClassifier(n_estimators=50)
        # print(X_test)
        clf.fit(X_train,y_train.to_list())
        y_pred=clf.predict(X_test)
        
        accuracy = metrics.accuracy_score(y_test.to_list(), y_pred)

        print("Accuracy: %s" % accuracy)
        total_accuracy.append(accuracy)
        zipped_res = zip(list(y_test.index),y_pred.tolist())
        prediction_res.append(list(zipped_res))


    main_alive_characters = raw.query('is_alive') \
                                .query('degree > 0.1') \
                                .query('wdegree > 0.1')

    result = []
    names = list(main_alive_characters.index)
    for name in names:
        predictions_for_name = []
        for prediction in prediction_res:
            for subprediction in prediction:
                if subprediction[0] == name:
                    predictions_for_name += [subprediction[1]]
        
        if len(predictions_for_name) > 0:
            predictions_to_die = [pr for pr in predictions_for_name if not pr]
            probability_for_name = float(len(predictions_to_die)) / float(len(predictions_for_name))
            result.append([probability_for_name, name])
            print('\n')
            print('Name %s' % name)
            print('participated in %s tests' % len(predictions_for_name))
            print('Voted for not alive %s' % len(predictions_to_die))
            print('Probability: %s' % probability_for_name)
        else:
            print('No tests for %s' % name)

    print('\n')
    accuracy_sum = float(sum(total_accuracy))
    accuracy_count = float(len(total_accuracy))
    average_accuracy = accuracy_sum / accuracy_count
    print('Average accuracy: %s' % average_accuracy)

    print('\n')
    final_res = sorted(result, key=lambda x: x[0], reverse=True)
    print('Character name   Probability')
    for entry in final_res:
        print('%s   %s' % (entry[1], entry[0]))
#    draw_graph(g)

if __name__ == "__main__":
    main()