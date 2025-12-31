import os
from tqdm import tqdm
import json
import yaml

import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.preprocessing import RobustScaler
import networkx as nx
import pandas as pd
from src.model import GFormer
from src.utils import create_edge_index, PLIDataset
from src.shapley import shapG, coalition_degree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Working on device: ", device)

with open("parameters.yml", 'r', encoding='utf-8') as paramFile:
    args = yaml.load(paramFile, Loader=yaml.FullLoader)

DATA_PATH = args["explainer"]["DATA_PATH"]
SAVE_FOLDER = args["explainer"]["SAVE_FOLDER"]
MODEL_PATH = args["explainer"]["MODEL_PATH"]
MODEL_NAME = args["explainer"]["GNN_MODEL"]
EDGE_WEIGHT = args["explainer"]["EDGE_WEIGHT"]
SCALING = args["explainer"]["SCALING"]
BATCH_SIZE = args["explainer"]["BATCH_SIZE"]
LEARNING_RATE = float(args["explainer"]["LEARNING_RATE"])
WEIGHT_DECAY = float(args["explainer"]["WEIGHT_DECAY"])
SEED = args["explainer"]["SEED"]
NUM_CLASSES = 1
HIDDEN_CHANNELS = args["explainer"]["HIDDEN_CHANNELS"]
EPOCHS = args["explainer"]["EPOCHS"]
NODE_FEATURES = args["explainer"]["NODE_FEATURES"]
AFFINITY_SET = args["explainer"]["AFFINITY_SET"]
SAMPLES_TO_EXPLAIN = args["explainer"]["SAMPLES_TO_EXPLAIN"]
MEAN_LOWER_BOUND = args["explainer"]["MEAN_LOWER_BOUND"]
MEAN_UPPER_BOUND = args["explainer"]["MEAN_UPPER_BOUND"]
LOW_BOUND = args["explainer"]["LOW_BOUND"]
CLEAN_DATA = args["explainer"]["CLEAN_DATA"]
MIN_AFFINITY = args["explainer"]["MIN_AFFINITY"]
MAX_AFFINITY = args["explainer"]["MAX_AFFINITY"]
HIGH_BOUND = args["explainer"]["HIGH_BOUND"]

assert(AFFINITY_SET == "low" or AFFINITY_SET == "high" or AFFINITY_SET == "medium")

print("Explaining affinity set: ", AFFINITY_SET)

interaction_affinities = None

with open(DATA_PATH + '/interaction_affinities.json', 'r') as fp:
    interaction_affinities = json.load(fp)

affinities_df = pd.DataFrame.from_dict(interaction_affinities, orient='index', columns=['affinity'])

if CLEAN_DATA == True:
    affinities_df = affinities_df[affinities_df['affinity'] >= MIN_AFFINITY]
    affinities_df = affinities_df[affinities_df['affinity'] <= MAX_AFFINITY]

vals_cleaned = list(affinities_df['affinity'])
mean_interaction_affinity_no_outliers = np.mean(vals_cleaned)

affinities_df = affinities_df.sort_values(by = "affinity", ascending=True)

interaction_affinities = affinities_df.to_dict(orient='index')

descriptors_interaction_dict = None
num_node_features = 0
if NODE_FEATURES:
    descriptors_interaction_dict = {}
    descriptors_interaction_dict["CA"] = [1, 0, 0, 0, 0, 0, 0, 0]
    descriptors_interaction_dict["NZ"] = [0, 1, 0, 0, 0, 0, 0, 0]
    descriptors_interaction_dict["N"] = [0, 0, 1, 0, 0, 0, 0, 0]
    descriptors_interaction_dict["OG"] = [0, 0, 0, 1, 0, 0, 0, 0]
    descriptors_interaction_dict["O"] = [0, 0, 0, 0, 1, 0, 0, 0]
    descriptors_interaction_dict["CZ"] = [0, 0, 0, 0, 0, 1, 0, 0]
    descriptors_interaction_dict["OD1"] = [0, 0, 0, 0, 0, 0, 1, 0]
    descriptors_interaction_dict["ZN"] = [0, 0, 0, 0, 0, 0, 0, 1]
    num_node_features = len(descriptors_interaction_dict["CA"])

# 数据加载
def generate_pli_dataset_dict(data_path):
    directory = os.fsencode(data_path)

    dataset_dict = {}
    dirs = os.listdir(directory)
    dirs = sorted(dirs, key=str)

    for file in tqdm(dirs):
        interaction_name = os.fsdecode(file)

        if interaction_name in interaction_affinities:
            if os.path.isdir(data_path + interaction_name):
                dataset_dict[interaction_name] = {}
                G = None
                with open(data_path + interaction_name + "/" + interaction_name + "_interaction_graph.json", 'r') as f:
                    data = json.load(f)
                    G = nx.Graph()

                    for node in data['nodes']:
                        G.add_node(node["id"], atom_type=node["attype"], origin=node["pl"])

                    for edge in data['edges']:
                        if edge["id1"] != None and edge["id2"] != None:
                            G.add_edge(edge["id1"], edge["id2"], weight=float(edge["length"]))

                    for node in data['nodes']:
                        nx.set_node_attributes(G, {node["id"]: node["attype"]}, "atom_type")
                        nx.set_node_attributes(G, {node["id"]: node["pl"]}, "origin")

                dataset_dict[interaction_name]["networkx_graph"] = G
                edge_index, edge_weight = create_edge_index(G, weighted=True)

                dataset_dict[interaction_name]["edge_index"] = edge_index
                dataset_dict[interaction_name]["edge_weight"] = edge_weight
                # ——— 新增：根据 origin 给边贴标签 ———
                edge_index_np = edge_index.cpu().numpy()
                edge_type = []
                for src, dst in zip(edge_index_np[0], edge_index_np[1]):
                    origin_src = G.nodes[int(src)]['origin']
                    origin_dst = G.nodes[int(dst)]['origin']
                    edge_type.append(1 if origin_src != origin_dst else 0)
                dataset_dict[interaction_name]['edge_type'] = torch.LongTensor(edge_type)

                num_nodes = G.number_of_nodes()

                if not NODE_FEATURES:
                    dataset_dict[interaction_name]["x"] = torch.full((num_nodes, 1), 1.0,
                                                                     dtype=torch.float)  # dummy feature
                else:
                    dataset_dict[interaction_name]["x"] = torch.zeros((num_nodes, num_node_features), dtype=torch.float)
                    for node in G.nodes:
                        dataset_dict[interaction_name]["x"][node] = torch.tensor(
                            descriptors_interaction_dict[G.nodes[node]["atom_type"]], dtype=torch.float)
                dataset_dict[interaction_name]["y"] = torch.FloatTensor(
                    [interaction_affinities[interaction_name]["affinity"]])

    return dataset_dict

pli_dataset_dict = generate_pli_dataset_dict(DATA_PATH + "/dataset/")

if SCALING:
    first_level = [pli_dataset_dict[key]["edge_weight"] for key in pli_dataset_dict]
    second_level = [item for sublist in first_level for item in sublist]
    transformer = RobustScaler().fit(np.array(second_level).reshape(-1, 1))
    for key in tqdm(pli_dataset_dict):
        scaled_weights = transformer.transform(np.array(pli_dataset_dict[key]["edge_weight"]).reshape(-1, 1))
        scaled_weights = [x[0] for x in scaled_weights]
        pli_dataset_dict[key]["edge_weight"] = torch.FloatTensor(scaled_weights)

data_list = [Data(x=sample["x"], edge_index=sample["edge_index"], edge_weight=sample["edge_weight"], edge_type=sample['edge_type'], y=sample["y"], networkx_graph=sample["networkx_graph"], interaction_name=name) for name, sample in pli_dataset_dict.items()]
dataset = PLIDataset(".", data_list = data_list)

hold_out_interactions = []
core_set_2016_interactions = []
with open(DATA_PATH + "pdb_ids/hold_out_set.csv", 'r') as f:
    hold_out_interactions = f.readlines()

hold_out_interactions = [interaction.strip() for interaction in hold_out_interactions]

with open(DATA_PATH + "pdb_ids/core_set.csv", 'r', encoding='utf-8') as f:
    core_set_2016_interactions = f.readlines()

core_set_2016_interactions = [interaction.strip() for interaction in core_set_2016_interactions]

hold_out_data = [dataset[i] for i in range(len(dataset)) if dataset[i].interaction_name in hold_out_interactions]
core_set_2016_data = [dataset[i] for i in range(len(dataset)) if dataset[i].interaction_name in core_set_2016_interactions]

rng = np.random.default_rng(seed = SEED)
rng.shuffle(hold_out_data)
rng.shuffle(core_set_2016_data)

hold_out_loader = DataLoader(hold_out_data, batch_size=BATCH_SIZE)
core_set_2016_loader = DataLoader(core_set_2016_data, batch_size=BATCH_SIZE)

# 加载模型
GNN = GFormer if MODEL_NAME == "GFormer" else None
model = GNN(node_features_dim=dataset[0].x.shape[1], num_classes=NUM_CLASSES, hidden_channels=HIDDEN_CHANNELS).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 解释阶段
def compute_shapley_values_for_edges(G):
    shapley_values = shapG(G, f=coalition_degree, depth=1, m=15, approximate_by_ratio=True, scale=True, verbose=False)
    return shapley_values
num_all_test_interactions = len(hold_out_data)
rng = np.random.default_rng(seed=SEED)
all_test_interaction_indices = np.array(range(num_all_test_interactions))
rng.shuffle(all_test_interaction_indices)

test_interaction_indices = []
num_test_interactions = SAMPLES_TO_EXPLAIN
test_interaction_names = []
test_interactions_affinities = []
test_interaction_names_affinities_dict = {}

for test_interaction_index in all_test_interaction_indices:
    model.eval()
    test_interaction = hold_out_data[test_interaction_index]

    edge_weight_to_pass = None
    if EDGE_WEIGHT:
        edge_weight_to_pass = test_interaction.edge_weight.to(device)
    batch = torch.zeros(test_interaction.x.shape[0], dtype=int).to(device)

    test_affinity_value = test_interaction.y.item()
    if AFFINITY_SET == "medium":
        if test_affinity_value < MEAN_LOWER_BOUND or test_affinity_value > MEAN_UPPER_BOUND:
            continue
    elif AFFINITY_SET == "low":
        if test_affinity_value >= LOW_BOUND:
            continue
    else:
        if test_affinity_value <= HIGH_BOUND:
            continue

    out = model(test_interaction.x.to(device), test_interaction.edge_index.to(device), batch=batch,
                edge_weight=edge_weight_to_pass)
    pred = out.item()
    if AFFINITY_SET == "low":
        if pred >= MEAN_LOWER_BOUND:
            continue
    elif AFFINITY_SET == "high":
        if pred <= MEAN_UPPER_BOUND:
            continue
    else:
        if pred < MEAN_LOWER_BOUND or pred > MEAN_UPPER_BOUND:
            continue

    test_interaction_indices.append(test_interaction_index)
    test_interaction_names.append(test_interaction.interaction_name)
    test_interactions_affinities.append(test_affinity_value)
    test_interaction_names_affinities_dict[test_interaction.interaction_name] = test_affinity_value

    if len(test_interaction_indices) == num_test_interactions:
        break
print(f"总共选择了 {len(test_interaction_indices)} 个用于解释的样本")


for index in tqdm(test_interaction_indices):
    model.eval()
    test_interaction = hold_out_data[index]
    print("\nInteraction: " + test_interaction.interaction_name)

    edge_weight_to_pass = test_interaction.edge_weight.to(device)
    batch = torch.zeros(test_interaction.x.shape[0], dtype=int, device=device)
    test_affinity_value = test_interaction.y.item()
    out = model(test_interaction.x.to(device), test_interaction.edge_index.to(device), batch=batch,
                        edge_weight=edge_weight_to_pass)
    print(f'out = {out}')
    shapley_values = compute_shapley_values_for_edges(
        G=test_interaction.networkx_graph,
        edge_index=test_interaction.edge_index,
        edge_weight=edge_weight_to_pass,
        edge_type=test_interaction.edge_type,
        model=model,
        device=device
    )

    SAVE_PATH = SAVE_FOLDER + "/" + MODEL_NAME + "/" + AFFINITY_SET + " affinity" + "/" + test_interaction.interaction_name + "/"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    with open(SAVE_PATH + test_interaction.interaction_name + "_shapley_values.txt", "w+") as f:
        f.write("Interaction name: " + test_interaction.interaction_name + "\n\n")
        f.write("Affinity: " + str(test_interaction.y.item()) + "\n")
        f.write("Predicted value: " + str(out.item()) + "\n\n")
        f.write("Shapley values for edges: \n")

        G = test_interaction.networkx_graph
        edge_list = list(G.edges())

        edge_type_tensor = test_interaction.edge_type.cpu()
        edge_index_np = test_interaction.edge_index.cpu().numpy()
        edge_type_dict = {}
        for i in range(edge_index_np.shape[1]):
            u, v = edge_index_np[0, i], edge_index_np[1, i]
            edge_type_dict[(u, v)] = int(edge_type_tensor[i])

        shapley_values_with_edges = {edge_list[int(edge)]: value for edge, value in shapley_values.items()}

        inter_shapley_total = 0.0
        intra_shapley_total = 0.0
        inter_count = 0
        intra_count = 0

        for edge, value in shapley_values_with_edges.items():
            u, v = edge
            etype = edge_type_dict.get((u, v), edge_type_dict.get((v, u), -1))
            tag = "[INTERACTION]" if etype == 1 else ""
            f.write(f"{edge}: {value:.16f} {tag}\n")
            if etype == 1:
                inter_shapley_total += value
                inter_count += 1
            elif etype == 0:
                intra_shapley_total += value
                intra_count += 1

        f.write("\n\nSummary:\n")
        total_edges = inter_count + intra_count
        if total_edges > 0:
            f.write(f"Interaction edge count: {inter_count} / {total_edges}\n")
            f.write(f"Intra edge count: {intra_count} / {total_edges}\n")
            f.write(f"Interaction edge average Shapley: {inter_shapley_total / inter_count if inter_count > 0 else 0:.6f}\n")
            f.write(f"Intra edge average Shapley: {intra_shapley_total / intra_count if intra_count > 0 else 0:.6f}\n")
            f.write(f"Interaction edge Shapley ratio: {(inter_shapley_total / (inter_shapley_total + intra_shapley_total)) if (inter_shapley_total + intra_shapley_total) > 0 else 0:.6f}\n")
