import torch
import os

settings = [(32, 0.5, 0.0, 0.4), (32, -0.5, 0.0, 0.4), (32, -0.4, 0.1, 0.4), (32, -0.7, 0.3, 0.4), (32, 0.9, 0.05, 0.4)]

for set in settings:
    N = set[0]
    J = set[1]
    Jb = set[2]
    temp = set[3]
    
    data = torch.load(f'PATH/ising/data/ising_data_{N}_{J}_{Jb}_{temp}.pt', weights_only=False)[4]

    newpath = f'PATH/GMNN/semisupervised/data/ising/{N}_{J}_{Jb}_{temp}' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    with open(f'{newpath}/feature.txt', 'w') as f:
        for node_id, feature_row in enumerate(data.x):
            line = f"{node_id}\t0:{feature_row[0].item()}\n"
            # line = f"{node_id}\t0:{0.0}\n"
            f.write(line)

    # with open(f'{newpath}/feature.txt', "w") as f:
    #     for node_id, y in enumerate(data.y):
    #         f.write(f"{node_id}\t0:{y.item()}\n")

    
    with open(f'{newpath}/net.txt', "w") as f:
        for src, dst in data.edge_index.t():
            f.write(f"{src.item()}\t{dst.item()}\t1\n")

    with open(f'{newpath}/label.txt', "w") as f:
        for node_id, y in enumerate(data.y):
            f.write(f"{node_id}\t{y.item()}\n")

    with open(f'{newpath}/train.txt', "w") as f:
        for node_id, x in enumerate(data.x):
            if (data['train_mask'][node_id]):
                f.write(f"{node_id}\n")
    
    with open(f'{newpath}/dev.txt', "w") as f:
        for node_id, x in enumerate(data.x):
            if (data['val_mask'][node_id]):
                f.write(f"{node_id}\n")
    
    with open(f'{newpath}/test.txt', "w") as f:
        for node_id, x in enumerate(data.x):
            if (data['test_mask'][node_id]):
                f.write(f"{node_id}\n")