import torch
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time
import yaml
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import json
import networkx as nx
import pandas as pd
from src.model import GFormer
from src.utils import create_edge_index, PLIDataset, set_all_seeds, save_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import LinearLR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Working on device: ", device)

if __name__ == "__main__":
    with open("parameters.yml", encoding='utf-8') as paramFile:
        args = yaml.load(paramFile, Loader=yaml.FullLoader)

    DATA_PATH = args["trainer"]["DATA_PATH"]

    CLEAN_DATA = args["trainer"]["CLEAN_DATA"]
    MIN_AFFINITY = args["trainer"]["MIN_AFFINITY"]
    MAX_AFFINITY = args["trainer"]["MAX_AFFINITY"]
    NUM_CLASSES = 1 #set it up to 1 since we are facing a regression problem

    MEAN_LOWER_BOUND = args["trainer"]["MEAN_LOWER_BOUND"]
    MEAN_UPPER_BOUND = args["trainer"]["MEAN_UPPER_BOUND"]
    LOW_BOUND = args["trainer"]["LOW_BOUND"]
    HIGH_BOUND = args["trainer"]["HIGH_BOUND"]

    MODEL_NAME = args["trainer"]["GNN_MODEL"]
    print(">>> Using model:", MODEL_NAME)

    GNN =GFormer if MODEL_NAME == "GFormer" else None

    SAVE_BEST_MODEL = args["trainer"]["SAVE_BEST_MODEL"]
    MODEL_SAVE_FOLDER = args["trainer"]["MODEL_SAVE_FOLDER"]

    EDGE_WEIGHT = args["trainer"]["EDGE_WEIGHT"]
    SCALING = args["trainer"]["SCALING"]

    SEED = args["trainer"]["SEED"]
    HIDDEN_CHANNELS = args["trainer"]["HIDDEN_CHANNELS"]
    EPOCHS = args["trainer"]["EPOCHS"]
    NODE_FEATURES = args["trainer"]["NODE_FEATURES"]
    BATCH_SIZE = args["trainer"]["BATCH_SIZE"]
    LEARNING_RATE = float(args["trainer"]["LEARNING_RATE"])
    WEIGHT_DECAY = float(args["trainer"]["WEIGHT_DECAY"])

    # 定义运行次数和种子列表
    # num_runs = 5  # 运行次数，可根据需要调整
    # seeds = [42, 123, 456, 789, 1000]  # 随机种子列表
    seed = 42
    print(f"seed = {seed}")
    set_all_seeds(seed)

    interaction_affinities = None

    with open(DATA_PATH + '/interaction_affinities.json', 'r', encoding='utf-8') as fp:
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

    def generate_pli_dataset_dict(data_path):

        directory = os.fsencode(data_path)

        dataset_dict = {}
        dirs = os.listdir(directory)
        for file in tqdm(dirs):
            interaction_name = os.fsdecode(file)

            if interaction_name in interaction_affinities:
                if os.path.isdir(data_path + interaction_name):
                    dataset_dict[interaction_name] = {}
                    G = None
                    with open(data_path + interaction_name + "/" + interaction_name + "_interaction_graph.json", 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        G = nx.Graph()

                        for node in data['nodes']:
                            G.add_node(node["id"], atom_type=node["attype"], origin=node["pl"])

                        for edge in data['edges']:
                            if edge["id1"] != None and edge["id2"] != None:
                                G.add_edge(edge["id1"], edge["id2"], weight= float(edge["length"]))


                        for node in data['nodes']:
                            nx.set_node_attributes(G, {node["id"]: node["attype"]}, "atom_type")
                            nx.set_node_attributes(G, {node["id"]: node["pl"]}, "origin")

                    dataset_dict[interaction_name]["networkx_graph"] = G
                    edge_index, edge_weight = create_edge_index(G, weighted=True)

                    # Invert weights (distance -> affinity)
                    edge_weight = 1.0 / (edge_weight + 1e-5)

                    dataset_dict[interaction_name]["edge_index"] = edge_index
                    dataset_dict[interaction_name]["edge_weight"] = edge_weight
                    edge_index_np = edge_index.cpu().numpy()
                    edge_type = []
                    for src, dst in zip(edge_index_np[0], edge_index_np[1]):
                        origin_src = G.nodes[int(src)]['origin']
                        origin_dst = G.nodes[int(dst)]['origin']
                        edge_type.append(1 if origin_src != origin_dst else 0) #1为相互作用边
                    dataset_dict[interaction_name]['edge_type'] = torch.LongTensor(edge_type)


                    num_nodes = G.number_of_nodes()

                    if not NODE_FEATURES:
                        dataset_dict[interaction_name]["x"] = torch.full((num_nodes, 1), 1.0, dtype=torch.float) #dummy feature
                    else:
                        dataset_dict[interaction_name]["x"] = torch.zeros((num_nodes, num_node_features),dtype=torch.float)
                        for node in G.nodes:

                            dataset_dict[interaction_name]["x"][node] = torch.tensor(descriptors_interaction_dict[G.nodes[node]["atom_type"]], dtype=torch.float)

                    ## gather label
                    dataset_dict[interaction_name]["y"] = torch.FloatTensor([interaction_affinities[interaction_name]["affinity"]])


        return dataset_dict

    pli_dataset_dict = generate_pli_dataset_dict(DATA_PATH + "dataset/")

    if SCALING:
        first_level = [pli_dataset_dict[key]["edge_weight"] for key in pli_dataset_dict]
        second_level = [item for sublist in first_level for item in sublist]
        if MODEL_NAME == "GCN" or MODEL_NAME == "GFormer":
            transformer = MinMaxScaler().fit(np.array(second_level).reshape(-1, 1))
        else:
            transformer = RobustScaler().fit(np.array(second_level).reshape(-1, 1))
        for key in tqdm(pli_dataset_dict):
            scaled_weights = transformer.transform(np.array(pli_dataset_dict[key]["edge_weight"]).reshape(-1, 1))
            scaled_weights = [x[0] for x in scaled_weights]
            pli_dataset_dict[key]["edge_weight"] = torch.FloatTensor(scaled_weights)

    data_list = []
    for name, item in tqdm(pli_dataset_dict.items()):
        data = Data(
            x=item['x'],
            edge_index=item['edge_index'],
            edge_weight=item['edge_weight'],
            edge_type=item['edge_type'],
            y=item['y'],
            interaction_name=name
        )
        data_list.append(data)
    dataset = PLIDataset(".", data_list = data_list)


    train_interactions = []
    val_interactions = []
    core_set_2013_interactions = []
    core_set_2016_interactions = []
    hold_out_interactions = []

    with open(DATA_PATH + "pdb_ids/training_set.csv", 'r', encoding='utf-8') as f:
        train_interactions = f.readlines()

    train_interactions = [interaction.strip() for interaction in train_interactions]

    with open(DATA_PATH + "pdb_ids/validation_set.csv", 'r', encoding='utf-8') as f:
        val_interactions = f.readlines()

    val_interactions = [interaction.strip() for interaction in val_interactions]

    with open(DATA_PATH + "pdb_ids/core2013.csv", 'r', encoding='utf-8') as f:
        core_set_2013_interactions = f.readlines()

    core_set_2013_interactions = [interaction.strip() for interaction in core_set_2013_interactions]

    with open(DATA_PATH + "pdb_ids/core_set.csv", 'r', encoding='utf-8') as f:
        core_set_2016_interactions = f.readlines()

    core_set_2016_interactions = [interaction.strip() for interaction in core_set_2016_interactions]

    with open(DATA_PATH + "pdb_ids/hold_out_set.csv", 'r', encoding='utf-8') as f:
        hold_out_interactions = f.readlines()

    hold_out_interactions = [interaction.strip() for interaction in hold_out_interactions]

    train_data = [dataset[i] for i in range(len(dataset)) if dataset[i].interaction_name in train_interactions]
    val_data = [dataset[i] for i in range(len(dataset)) if dataset[i].interaction_name in val_interactions]
    core_set_2013_data = [dataset[i] for i in range(len(dataset)) if dataset[i].interaction_name in core_set_2013_interactions]
    core_set_2016_data = [dataset[i] for i in range(len(dataset)) if dataset[i].interaction_name in core_set_2016_interactions]
    hold_out_data = [dataset[i] for i in range(len(dataset)) if dataset[i].interaction_name in hold_out_interactions]

    rng = np.random.default_rng(seed = SEED)
    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(core_set_2013_data)
    rng.shuffle(core_set_2016_data)
    rng.shuffle(hold_out_data)


    print("Number of samples after outlier removal: ", len(dataset))
    print("Number of training samples: ", len(train_data))
    print("Number of validation samples: ", len(val_data))
    print("Number of 2013 core set samples: ", len(core_set_2013_data))
    print("Number of 2013 core set low affinity samples: ", len([sample for sample in core_set_2013_data if sample.y < LOW_BOUND]))
    print("Number of 2013 core set medium affinity samples: ", len([sample for sample in core_set_2013_data if sample.y >= MEAN_LOWER_BOUND and sample.y <= MEAN_UPPER_BOUND]))
    print("Number of 2013 core set high affinity samples: ", len([sample for sample in core_set_2013_data if sample.y > HIGH_BOUND]))
    print("Number of 2016 core set samples: ", len(core_set_2016_data))
    print("Number of 2016 core set low affinity samples: ", len([sample for sample in core_set_2016_data if sample.y < LOW_BOUND]))
    print("Number of 2016 core set medium affinity samples: ", len([sample for sample in core_set_2016_data if sample.y >= MEAN_LOWER_BOUND and sample.y <= MEAN_UPPER_BOUND]))
    print("Number of 2016 core set high affinity samples: ", len([sample for sample in core_set_2016_data if sample.y > HIGH_BOUND]))
    print("Number of hold out samples: ", len(hold_out_data))
    print("Number of hold out low affinity samples: ", len([sample for sample in hold_out_data if sample.y < LOW_BOUND]))
    print("Number of hold out medium affinity samples: ", len([sample for sample in hold_out_data if sample.y >= MEAN_LOWER_BOUND and sample.y <= MEAN_UPPER_BOUND]))
    print("Number of hold out high affinity samples: ", len([sample for sample in hold_out_data if sample.y > HIGH_BOUND]))

    # core_set_hold_out_interactions = core_set_2013_interactions + core_set_2016_interactions + hold_out_interactions
    core_set_interactions = core_set_2013_interactions + core_set_2016_interactions
    core_set_data = [dataset[i] for i in range(len(dataset)) if dataset[i].interaction_name in core_set_interactions]

    # print("Number of test (core + hold out) samples: ", len(core_set_hold_out_data))
    print("Number of test (core 2013+ core 2016) samples: ", len(core_set_data))
    print("Number of test low affinity samples: ", len([sample for sample in core_set_data if sample.y < LOW_BOUND]))
    print("Number of test medium affinity samples: ", len([sample for sample in core_set_data if sample.y >= MEAN_LOWER_BOUND and sample.y <= MEAN_UPPER_BOUND]))
    print("Number of test high affinity samples: ", len([sample for sample in core_set_data if sample.y > HIGH_BOUND]))


    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    core_set_2013_loader = DataLoader(core_set_2013_data, batch_size = BATCH_SIZE)
    core_set_2016_loader = DataLoader(core_set_2016_data, batch_size=BATCH_SIZE)
    hold_out_loader = DataLoader(hold_out_data, batch_size=BATCH_SIZE)

    from sklearn.preprocessing import StandardScaler

    #只用训练集估计参数
    y_train = np.array([d.y.item() for d in train_data])
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

    def scale_subset(subset):
        for d in subset:
            d.y = torch.FloatTensor(y_scaler.transform(d.y.reshape(1, -1))).squeeze()


    scale_subset(train_data)
    scale_subset(val_data)
    scale_subset(core_set_2013_data)
    scale_subset(core_set_2016_data)
    scale_subset(hold_out_data)


    # ### Train the network
    model = GNN(node_features_dim = dataset[0].x.shape[1], num_classes = NUM_CLASSES, hidden_channels=HIDDEN_CHANNELS).to(device)

    lr = LEARNING_RATE

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    main_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    epochs = EPOCHS


    def huber_loss(y_pred, y_true, delta=1.0):
        diff = y_true - y_pred
        abs_diff = torch.abs(diff)
        quadratic = torch.minimum(abs_diff, torch.tensor(delta))
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic ** 2 + delta * linear
        return torch.mean(loss)

    def pcc_loss(y_pred, y_true):
        pred_mean = torch.mean(y_pred)
        true_mean = torch.mean(y_true)
        cov = torch.mean((y_pred - pred_mean) * (y_true - true_mean))
        std_pred = torch.std(y_pred)
        std_true = torch.std(y_true)
        return -cov / (std_pred * std_true)

    class CombinedLoss(nn.Module):
        def __init__(self, alpha=1.0, beta=1.0, delta=1.0):
            super(CombinedLoss, self).__init__()
            self.alpha = alpha
            self.beta = beta
            self.delta = delta

        def forward(self, y_pred, y_true):
            huber = huber_loss(y_pred, y_true, delta=self.delta)
            pcc = pcc_loss(y_pred, y_true)
            return self.alpha * huber + self.beta * pcc

    # 示例用法
    criterion = CombinedLoss(alpha=0.3, beta=0.7, delta=1.0)

    def train():
        model.train()

        for data in train_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch, edge_weight = data.edge_weight, edge_type=data.edge_type)
            # regularization = (0.01) * sum(torch.sum(param ** 2) for param in model.parameters())
            loss = criterion(torch.squeeze(out), data.y) # +regularization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()


    def test(loader, y_scaler=None):
        model.eval()
        all_true = []
        all_pred = []

        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(
                    data.x,
                    data.edge_index,
                    data.batch,
                    edge_weight=data.edge_weight,
                    edge_type=data.edge_type
                )
                out = out.squeeze()
                y_true = data.y
                y_pred = out

                all_true.append(y_true.view(-1).cpu().numpy())
                all_pred.append(y_pred.detach().view(-1).cpu().numpy())
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        # >>> 逆变换
        if y_scaler is not None:
            all_true = y_scaler.inverse_transform(all_true.reshape(-1, 1)).ravel()
            all_pred = y_scaler.inverse_transform(all_pred.reshape(-1, 1)).ravel()
        # >>> NaN/Inf 清理
        mask = ~np.isnan(all_true) & ~np.isnan(all_pred) & \
               ~np.isinf(all_true) & ~np.isinf(all_pred)
        all_true = all_true[mask]
        all_pred = all_pred[mask]

        # 计算指标（基于原始亲和力）
        avg_mse = mean_squared_error(all_true, all_pred)
        avg_rmse = np.sqrt(avg_mse)
        avg_mae = mean_absolute_error(all_true, all_pred)
        sd = np.std(all_true - all_pred)

        return avg_rmse, avg_mae, sd, all_true, all_pred

    patience = 15
    best_epoch = 0
    no_improve_count = 0
    MIN_LR = 1e-5
    early_stop = False

    best_val_loss = float("inf")
    for epoch in range(epochs):
        if early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
        train()
        train_rmse, train_mae, train_SD, _, _= test(train_loader,y_scaler)
        val_rmse, val_mae, val_SD, _, _= test(val_loader, y_scaler)
        warmup_scheduler.step()
        if epoch >= 10:
            main_scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < MIN_LR:
            print(f"Learning rate too low ({current_lr:.2e}), resetting to initial LR")
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE * 0.5
        if val_rmse < best_val_loss:  # 仅比较 RMSE
            best_val_loss = val_rmse
            no_improve_count = 0
            best_epoch = epoch
            if SAVE_BEST_MODEL:
                save_model(model, MODEL_SAVE_FOLDER, model_name = MODEL_NAME + "_best")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                early_stop = True
                print(f"No improvement for {patience} epochs, triggering early stop")
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: LR = {current_lr:.2e}, Val RMSE = {val_rmse:.4f}")
        print(f'Epoch: {epoch:03d}, Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, Train SD: {train_SD:.4f}, Val SD: {val_SD:.4f}')
    if early_stop:
        print(f"Training stopped early at epoch {epoch}")
    else:
        print(f"Training completed all {epochs} epochs")

    core_set_2013_rmse, core_set_2013_mae, core_set_2013_SD, _, _ = test(
        core_set_2013_loader, y_scaler)
    core_set_2016_rmse, core_set_2016_mae, core_set_2016_SD, _, _ = test(
        core_set_2016_loader, y_scaler)
    hold_out_set_rmse, hold_out_set_mae, hold_out_set_SD, _, _ = test(
        hold_out_loader, y_scaler)

    if not SAVE_BEST_MODEL:
        print(f'Core set 2013 RMSE with latest model: {core_set_2013_rmse:.4f}, MSE: {core_set_2013_mae:.4f}, SD: {core_set_2013_SD:.4f}')
        print(f'Core set 2016 RMSE with latest model: {core_set_2016_rmse:.4f}, MSE: {core_set_2016_mae:.4f}, SD: {core_set_2016_SD:.4f}')
        print(f'Hold out set 2019 RMSE with latest model: {hold_out_set_rmse:.4f}, MSE: {hold_out_set_mae:.4f}, SD: {hold_out_set_SD:.4f}')
        save_model(model, MODEL_SAVE_FOLDER, model_name=MODEL_NAME + "_latest", timestamp=True)
    print(f'Best model at epoch: {best_epoch:03d}')
    print("Best val loss: ", best_val_loss)

    if SAVE_BEST_MODEL:
        model = GNN(node_features_dim = dataset[0].x.shape[1], num_classes = NUM_CLASSES, hidden_channels=HIDDEN_CHANNELS).to(device)
        model.load_state_dict(torch.load("models/model_" + MODEL_NAME + "_best.ckpt"))
        model.to(device)

        core_set_2013_rmse, core_set_2013_mae, core_set_2013_SD, _ ,_ = test(core_set_2013_loader, y_scaler)
        print(f'Core set 2013 RMSE: {core_set_2013_rmse:.4f}, MAE: {core_set_2013_mae:.4f}, SD: {core_set_2013_SD:.4f}')

        core_set_2016_rmse, core_set_2016_mae , core_set_2016_SD,_ ,_ = test(core_set_2016_loader, y_scaler)
        print(f'Core set 2016 RMSE: {core_set_2016_rmse:.4f}, MAE: {core_set_2016_mae:.4f}, SD: {core_set_2016_SD:.4f}')

        hold_out_set_rmse, hold_out_set_mae, hold_out_set_SD ,_ ,_= test(hold_out_loader, y_scaler)
        print(f'Hold out set 2019 RMSE: {hold_out_set_rmse:.4f}, MAE: {hold_out_set_mae:.4f}, SD: {hold_out_set_SD:.4f}')

        source_file = f"models/model_{MODEL_NAME}_best.ckpt"
        target_file = f"models/model_{MODEL_NAME}_best_{str(best_epoch)}_{int(time.time())}.ckpt"

        os.rename(source_file, target_file)
    