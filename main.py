import nltk
from nltk.corpus import brown
import string
from typing import List
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import math
import numpy as np
from networkx.algorithms import community
import networkx as nx
from itertools import chain, combinations
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import random


PUNCT = [c for c in string.punctuation] + [(c+c) for c in string.punctuation] + [(c+c+c) for c in string.punctuation]
STOP = set(stopwords.words('english'))
WINDOW = list(range(1, 8))


# create indirected graph (0.5 points)
# create directed grpah (0.5 points)
# create function which can work with different window size (from 1 to 7) (1 point)


def create_lm_coocurance(sentences:List[List[str]], window_size: int = 2):
    if window_size not in WINDOW:
        raise ValueError("The window size can only be an integer from 1 to 7")
    pos_sentences = [preprocess(s) for s in sentences]
    pairs = []
    all_words = set()

    for pos_sent in pos_sentences:
        # collecting collocations on the right
        pairs += generate_pairs(pos_sent,window_size)
        # collecting collocations on the left
        pos_sent.reverse()
        pairs += generate_pairs(pos_sent, window_size)

    graph = nx.Graph()
    graph.add_edges_from(pairs)

    return graph, pairs


def preprocess(sentence:List[str]):
    sentence = [w for w in sentence if w not in PUNCT]
    sentence = nltk.pos_tag(sentence)
    for i, t in enumerate(sentence):
        if t[0][0].isupper() and t[1] != "NNP":
            sentence[i] = (t[0].lower(),t[1])

    # МОГУ ДОБАВИТЬ ЛЕММАТИЗАЦИЮ ВОТ ТУТ, СМ ФУНКЦИЮ
    sentence = lemmatize(sentence)

    # МОГУ ДОБАВИТЬ УБИРАНИЕ СТОП СЛОВ ТУТ
    sentence = [w for w in sentence if w[0] not in STOP]
    return sentence


def lemmatize(sentence):
    lemmer = WordNetLemmatizer()
    for i, t in enumerate(sentence):
        if t[1].startswith('J'):
            sentence[i] = (lemmer.lemmatize(t[0], wordnet.ADJ),'J')
        elif t[1].startswith('V'):
            sentence[i] = (lemmer.lemmatize(t[0], wordnet.VERB),'V')
        elif t[1].startswith('N'):
            sentence[i] = (lemmer.lemmatize(t[0], wordnet.NOUN),'N')
        elif t[1].startswith('R'):
            sentence[i] = (lemmer.lemmatize(t[0], wordnet.ADV),'R')
        return sentence


def generate_pairs(pos_sentence, window):
    pairs = []
    for i, pos_word in enumerate(pos_sentence):
        if i != len(pos_sentence) - 1:
            pairs += form_word_pairs(pos_word,i+1,min(len(pos_sentence)-1,i+window),pos_sentence)
    return pairs


def form_word_pairs(word, cur_i, last_i, sentence):
    pairs = []
    for i in range(cur_i, last_i+1):
        pairs.append((word,sentence[i]))
    return pairs


def generate_colors(nodes, tag):
    colors = []
    cur, color = 0, 1
    for n in nodes:
        if n[1][tag] == cur:
            colors.append(color)
        elif n[1][tag] > cur:
            cur = n[1][tag]
            color += 1
            colors.append(color)
        else:
            raise ValueError("Nodes must be sorted in ascending order!")
    return colors


# my function to get a merge height so that it is unique (probably not that efficient)
def get_merge_height(sub, node_labels, subset_rank_dict):
    sub_tuple = tuple(sorted([node_labels[i] for i in sub]))
    n = len(sub_tuple)
    other_same_len_merges = {k: v for k, v in subset_rank_dict.items() if len(k) == n}
    min_rank, max_rank = min(other_same_len_merges.values()), max(other_same_len_merges.values())
    range = (max_rank-min_rank) if max_rank > min_rank else 1
    return float(len(sub)) + 0.8 * (subset_rank_dict[sub_tuple] - min_rank) / range


def plot_dendrogram(G, method_to_find_community=community.girvan_newman):
    communities = list(method_to_find_community(G))
    # building initial dict of node_id to each possible subset:
    node_id = 0
    init_node2community_dict = {node_id: communities[0][0].union(communities[0][1])}
    for comm in communities:
        for subset in list(comm):
            if subset not in init_node2community_dict.values():
                node_id += 1
                init_node2community_dict[node_id] = subset

    # turning this dictionary to the desired format in @mdml's answer
    node_id_to_children = {e: [] for e in init_node2community_dict.keys()}
    for node_id1, node_id2 in combinations(init_node2community_dict.keys(), 2):
        for node_id_parent, group in init_node2community_dict.items():
            if len(init_node2community_dict[node_id1].intersection(init_node2community_dict[node_id2])) == 0 and group == init_node2community_dict[node_id1].union(init_node2community_dict[node_id2]):
                node_id_to_children[node_id_parent].append(node_id1)
                node_id_to_children[node_id_parent].append(node_id2)

    # also recording node_labels dict for the correct label for dendrogram leaves
    node_labels = dict()
    for node_id, group in init_node2community_dict.items():
        if len(group) == 1:
            node_labels[node_id] = list(group)[0]
        else:
            node_labels[node_id] = ''

    # also needing a subset to rank dict to later know within all k-length merges which came first
    subset_rank_dict = dict()
    rank = 0
    for e in communities[::-1]:
        for p in list(e):
            if tuple(p) not in subset_rank_dict:
                subset_rank_dict[tuple(sorted(p))] = rank
                rank += 1
    subset_rank_dict[tuple(sorted(chain.from_iterable(communities[-1])))] = rank

    # finally using @mdml's magic, slightly modified:
    G = nx.DiGraph(node_id_to_children)
    nodes = G.nodes()
    leaves = set([n for n in nodes if G.out_degree(n) == 0])
    inner_nodes = [n for n in nodes if G.out_degree(n) > 0]

    # Compute the size of each subtree
    subtree = dict( (n, [n]) for n in leaves )
    for u in inner_nodes:
        children = set()
        node_list = list(node_id_to_children[u])
        while len(node_list) > 0:
            v = node_list.pop(0)
            children.add( v )
            node_list += node_id_to_children[v]
        subtree[u] = sorted(children & leaves)

    inner_nodes.sort(key=lambda n: len(subtree[n])) # <-- order inner nodes ascending by subtree size, root is last

    # Construct the linkage matrix
    leaves = sorted(leaves)
    index  = dict( (tuple([n]), i) for i, n in enumerate(leaves) )
    Z = []
    k = len(leaves)
    for i, n in enumerate(inner_nodes):
        children = node_id_to_children[n]
        x = children[0]
        for y in children[1:]:
            z = tuple(sorted(subtree[x] + subtree[y]))
            i, j = index[tuple(sorted(subtree[x]))], index[tuple(sorted(subtree[y]))]
            Z.append([i, j, get_merge_height(subtree[n], node_labels, subset_rank_dict), len(z)]) # <-- float is required by the dendrogram function
            index[z] = k
            subtree[z] = list(z)
            x = z
            k += 1

    plt.figure(figsize=(15, 15))
    dendrogram(Z, labels=np.ndarray([node_labels[node_id] for node_id in leaves]))
    plt.xticks(fontsize=12)
    plt.show()


def show_attr_graph(g, pos, attr):
    nodes = sorted(g.nodes(data=True), key=lambda x: x[1][attr])
    gcc = g.subgraph(sorted(nx.connected_components(g), key=len, reverse=True)[0])

    colors = generate_colors(nodes, attr)
    plt.figure(figsize=(15, 15))
    nx.draw(
        gcc,
        pos,
        nodelist=[n[0] for n in nodes],
        node_size=80,
        node_color=colors,
        labels=dict([(n, n[0]) for n in g.nodes()]),
        cmap=plt.cm.cool,
        with_labels=True
    )
    plt.show()


def draw_communities(g, res):
    subset_color = [
        "gold",
        "violet",
        "limegreen",
        "darkorange",
    ]
    if len(res.keys()) > len(subset_color):
        subset_color = subset_color * (len(res.keys()) // len(subset_color) + 1)

    pos = nx.spring_layout(g)
    color = []
    for v in g.nodes():
        for i, words in res.items():
            if v in words:
                color.append(subset_color[i])
    plt.figure(figsize=(20, 10))
    nx.draw(g, pos, node_size=80, node_color=color, with_labels=False)
    plt.show()


def main(show_flag, centralities, communities):
    # nltk.download('brown')
    # brown.words()
    # print(brown.raw()[:100])
    # print(brown.sents()[:1])
    # nltk.download('stopwords')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('wordnet')
    g, ps = create_lm_coocurance(brown.sents()[:100])
    pos = nx.spring_layout(g)
    # figsize=(8, 8) чем больше цифры здесь, тем больше картинка, все потому что отрисовка в networkx завязаы
    if show_flag:
        plt.figure(figsize=(10, 10))
        nx.draw(g, pos, labels=dict([(n, n[0]) for n in g.nodes()]))
        plt.show()
    # число вершин
    print(g.number_of_nodes())

    # число ребер
    print(g.number_of_edges())

    # число связных компонент
    print(nx.number_connected_components(g))
    g = g.subgraph(max(nx.connected_components(g), key=len))

    # плотность графа
    print(nx.density(g))
    # cредняя степень вершины
    print(sum([g.degree[n] for n in g.nodes]) / g.number_of_nodes())

    # распределение степеней внутри графа и график
    degrees = dict(g.degree())
    degree_values = sorted(set(degrees.values()))
    histogram = [list(degrees.values()).count(i) / float(nx.number_of_nodes(g)) \
                 for i in degree_values]


    if show_flag:
        fig, ax = plt.subplots()
        plt.title("Bar graph of degree distribution")
        plt.bar(degree_values, histogram, width=9)
        plt.xlabel('Degree')
        plt.ylabel('Fraction of Nodes')
        ax.set_xticks(degree_values)
        ax.set_xticklabels(degree_values)
        plt.show()

    if show_flag:
        plt.title("Histogram of degree distribution")
        plt.hist(list(degrees.values()), bins=len(degree_values))
        plt.xticks(np.arange(0, max(degree_values), max(degree_values) * 0.1))
        plt.show()

    if show_flag:
        plt.title("Log-log plot of degree distribution")
        plt.loglog(degree_values, histogram)
        plt.show()

    if show_flag:
        plt.title("Scatter of degree distribution")
        plt.scatter(degree_values, histogram)
        plt.show()

    degree_logs = [math.log(i) for i in degree_values]
    frequency_logs = [math.log(i) for i in histogram]

    if show_flag:
        plt.title("Log-log of degree distribution")
        plt.scatter(degree_logs, frequency_logs)
        plt.show()

    x = np.array(degree_logs)
    y = np.array(frequency_logs)
    z = np.polyfit(x, y, 1)
    m, b = z[0], z[1]
    if show_flag:
        plt.scatter(x, y)
        plt.plot(x, m * x + b)
        plt.title("Log-log of degree distribution with linear regression")
        plt.show()

    degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
    gcc = g.subgraph(sorted(nx.connected_components(g), key=len, reverse=True)[0])

    if show_flag:
        plt.loglog(degree_sequence, "b-", marker="o")
        plt.title("Degree rank plot")
        plt.ylabel("degree")
        plt.xlabel("rank")

        # draw graph in inset
        plt.axes([0.45, 0.45, 0.45, 0.45])

        pos = nx.spring_layout(gcc)
        plt.axis("off")
        nx.draw_networkx_nodes(gcc, pos, node_size=20)
        nx.draw_networkx_edges(gcc, pos, alpha=0.4)

        plt.show()

    """Используйте ваши знания о центральностях и об их применимости к тем или иным типам графов.
    Рассчитайте их (возможно на сабграфе), сделайте отображение графа,
    на котором размер вершины будет зависить от показателя центральности,
    ну или цветом, как здесь:
    https://networkx.org/documentation/stable/auto_examples/drawing/plot_random_geometric_graph.html#sphx-glr-auto-examples-drawing-plot-random-geometric-graph-py
    (3 points)
    Не забудь проанализировать, ведь визуализация (иногда) ключ к пониманию
    """
    # выделение или невыделение подграфа (субграфа)
    gs = g.copy()

    # Центральности. Если граф ненаправленный, то все виды центральности
    # (degree, betweenness, closeness, eigenvector)

    ## calculate degree centrality,
    if centralities:
        dg_centr = nx.degree_centrality(gs)
        bt_centr = nx.betweenness_centrality(gs)
        cs_centr = nx.closeness_centrality(gs)
        eg_centr = nx.eigenvector_centrality(gs)


        ## set degree centrality metrics on each node,
        # if show_flag:
        nx.set_node_attributes(gs, dg_centr, 'dg')
        nx.set_node_attributes(gs, bt_centr, 'bt')
        nx.set_node_attributes(gs, cs_centr, 'cs')
        nx.set_node_attributes(gs, eg_centr, 'eg')

        show_attr_graph(gs, pos, 'dg')
        show_attr_graph(gs, pos, 'bt')
        show_attr_graph(gs, pos, 'cs')
        show_attr_graph(gs, pos, 'eg')

    """Ваша задача: применить методы поиска сообществ и попытаться интерпретировать выдачу (3 points)"""
    if communities:
        # бинарное разбиение исходного графа на сообщества
        comp = community.girvan_newman(gs)
        res_gm = {i: words for i, words in enumerate(tuple(sorted(c) for c in next(comp)))}

        # алгоритм Kernighan–Lin
        comp = community.kernighan_lin_bisection(gs)
        res_kl = {i: list(words) for i, words in enumerate(comp)}

        # comp = community.greedy_modularity_communities(gs)
        # res_gmc = {i: list(words) for i, words in enumerate(comp)}
        # print(res_gmc.keys())

    draw_communities(gs, res_gm)
    draw_communities(gs, res_kl)


    # k = 2
    # # Посмотрим чуть глубже: 2 означает еще две итерации деления на сообщества
    # for communities in itertools.islice(comp, k):
    #     print({indx: words for indx, words in enumerate(tuple(sorted(c) for c in communities))})
    #
    # plot_dendrogram(g)


if __name__ == '__main__':
    main(False, True, True)
