import numpy as np
from scipy import spatial
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
import json
import random
import os
import pickle
# import graphviz
from tqdm import tqdm
import json 
from datasets import load_dataset

merged_relations = [
    'antonym',
    'at_location',
    'capable_of',
    'causes',
    'created_by',
    'is_a',
    'desires',
    'has_subevent',
    'part_of',
    'has_context',
    'has_property',
    'made_of',
    'not_capable_of',
    'not_desires',
    'receives_action',
    'related_to',
    'used_for',
]

concept2id = None
id2concept = None
relation2id = None
id2relation = None
cpnet = None
cpnet_simple = None

concept_embs = None
relation_embs = None

node_description = ["List all nodes of the graph shown in the image.",
                    "Provide the names of all nodes displayed in the graph image.",
                    "Can you name all the nodes shown in the graph image?",
                    "Identify all the vertices in the diagram of the graph provided.",
                    "Detail all the vertices from the graph depicted in the image."]

highest_node_degree = ["Name one of the node with the highest degree in the graph. And what is its degree?",
                "Identify one of the node that has the most connections in the graph and specify its degree.",
                "Can you tell me which node (name one) has the highest degree in this graph and what that degree is?",
                "Provide the name and degree of the node with the most connections in the graph."
                "Which node in the graph has the greatest number of connections, and what is that total?"]

node_degree_question = ["What is the degree of the node with the name \"{node}\"?",
                "What is the degree of the node labeled \"{node}\"?",
                "Can you tell me the degree of the node named \"{node}\"?",
                "What is the total number of connections that the node \"{node\"} has?", 
                "How many connections does the node \"{node}\" have?"]

node_number_detection = ["How many nodes are there in the graph?",
                "What is the total number of nodes in the graph?",
                "Can you tell me how many nodes are in the graph?",
                "What is the total number of vertices in the graph?",
                "How many vertices are there in the graph?"]

edge_number_detection = ["How many edges are there in the graph?",
                "What is the total number of edges in the graph?",
                "Can you tell me how many edges are in the graph?",
                "What is the total number of connections in the graph?",
                "How many connections are there in the graph?"]

triple_listing = ["List all the triples in the graph.",
                "Provide all the triples in the graph.",
                "Can you list all the triples in the graph?",
                "Detail all the triples in the graph.",
                "Enumerate all the triples in the graph."]

def load_resources(cpnet_vocab_path='data/cpnet/concept.txt'):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}


def load_cpnet(cpnet_graph_path='data/cpnet/conceptnet.en.pruned.graph'):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def load_csqa_graph(graph_path='data/obqa/graph/train.graph.jsonl'):
    with open(graph_path, 'r') as f:
        graph = [json.loads(line) for line in f]

    return graph


def get_edge(src_concept, tgt_concept):
    global cpnet
    rel_list = cpnet[src_concept][tgt_concept]  # list of dicts
    seen = set()
    res = [r['rel'] for r in rel_list.values() if r['rel'] not in seen and (seen.add(r['rel']) or True)]  # get unique values from rel_list
    # convert everyhing larger than 16 to 15 and make sure no duplicates
    res = [r for r in res if r <= 16]
    if len(res) > 1 and 15 in res:
        res.remove(15)
    return res


# main function 
def main():
    load_resources()
    load_cpnet()
    graph = load_csqa_graph()

    for idx in tqdm(range(len(graph)//4)):
        subgraph = set()
        nodes = set()
        node_degree = {}

        for g in graph[idx*4:idx*4+4]:
        # for g in graph[5:10]:
            node2name = {}
            node2id = {}
            for node in g['nodes']:
                node2name[node['id']] = id2concept[node['cid']]
                node2id[node['id']] = node['cid']
                # graph_image.node(id2concept[node['cid']])
                node_name = id2concept[node['cid']].replace('_', ' ')
                nodes.add(node_name)
                if node_name not in node_degree:
                    node_degree[node_name] = 0

            for link in g['links']:
                source = node2name[link['source']]
                target = node2name[link['target']]
                node_degree[source.replace('_', ' ')] += 1
                node_degree[target.replace('_', ' ')] += 1
                try:
                    edges = [id2relation[x] for x in get_edge(node2id[link['source']], node2id[link['target']])]
                    if len(edges) != 0:
                        subgraph.add((source.replace('_',' '), edges[0].replace('_',' '), target.replace('_',' ')))
                except:
                    continue
        
        if len(subgraph) == 0:
            continue

        # node description task
        question1 = "<image>\n" + random.choice(node_description)
        answer1 = f"The image depicts the following nodes: {', '.join(nodes)}"
        data1 = {'conversations': [{'from':'human', 'value':question1},
                                    {'from': 'gpt', 'value': answer1}],
                'id': str(idx)+'0', 
                'image': f'csqa_{idx}.gv.png'}

        # highest node degree task
        highest_degree_node = max(node_degree, key=node_degree.get)
        question2 = "<image>\n" + random.choice(highest_node_degree)
        answer2 = f"One node with the highest degree is \"{highest_degree_node}\" with a degree of {node_degree[highest_degree_node]}."
        data2 = {'conversations': [{'from':'human', 'value':question2},
                                    {'from': 'gpt', 'value': answer2}],
                'id': str(idx)+'1', 
                'image': f'csqa_{idx}.gv.png'}

        # node degree task
        node = random.choice(list(node_degree.keys()))
        question3 = "<image>\n" + random.choice(node_degree_question).replace('{node}', node)
        answer3 = f"The degree of the node \"{node}\" is {node_degree[node]}."
        data3 = {'conversations': [{'from':'human', 'value':question3},
                                    {'from': 'gpt', 'value': answer3}],
                'id': str(idx)+'2', 
                'image': f'csqa_{idx}.gv.png'}

        # node number detection task
        question4 = "<image>\n" + random.choice(node_number_detection)
        answer4 = f"There are {len(nodes)} nodes in the graph."
        data4 = {'conversations': [{'from':'human', 'value':question4},
                                    {'from': 'gpt', 'value': answer4}],
                'id': str(idx)+'3', 
                'image': f'csqa_{idx}.gv.png'}

        # edge number detection task
        question5 = "<image>\n" + random.choice(edge_number_detection)
        answer5 = f"There are {len(subgraph)} edges in the graph."
        data5 = {'conversations': [{'from':'human', 'value':question5},
                                    {'from': 'gpt', 'value': answer5}],
                'id': str(idx)+'4', 
                'image': f'csqa_{idx}.gv.png'}
        
        # triple listing task
        question6 = "<image>\n" + random.choice(triple_listing)
        answer6 = f"The triples in the graph are listed as: {', '.join([f'({x[0]}, {x[1]}, {x[2]})' for x in subgraph])}"
        data6 = {'conversations': [{'from':'human', 'value':question6},
                                    {'from': 'gpt', 'value': answer6}],
                'id': str(idx)+'5', 
                'image': f'csqa_{idx}.gv.png'}  
                
        triples = ', '.join([f'({x[0]}, {x[1]}, {x[2]})' for x in subgraph])
        with open('kg_train_data.jsonl', 'a') as f:
            f.write(json.dumps(data1) + '\n')
            f.write(json.dumps(data2) + '\n')
            f.write(json.dumps(data3) + '\n')
            f.write(json.dumps(data4) + '\n')
            f.write(json.dumps(data5) + '\n')
            f.write(json.dumps(data6) + '\n')


if __name__ == '__main__':
    main()