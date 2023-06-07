import torch
from tqdm import tqdm
from data_loading import StructDataset
from equiformer.graph_attention_transformer import graph_attention_transformer_nonlinear_l2


def main():
    max_epochs = 10000
    max_num_neighbors = 256
    train_batch_size = 16
    radius = 5.0
    node_filter = 'CA'
    
    device = torch.device('cuda')
    
    test_set = StructDataset(path='./StructuredDatasets/test2_dataset.pkl', batch_size=32, random=False, node_filter=node_filter)
    task_mean, task_std = 53.14, 11.78
    
    network = graph_attention_transformer_nonlinear_l2(
        irreps_in='5x0e', 
        radius=radius, 
        num_basis=128, 
        out_channels=1, 
        task_mean=task_mean, 
        task_std=task_std, 
        atomref=None, 
        drop_path=0.0
    )
    network.load_state_dict(torch.load('./checkpoints/epoch_1.pth'))
    network = network.to(device)
    # network = torch.nn.DataParallel(network)
    n_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f'{n_parameters} parameters')
    
    test_loss = []
    network.eval()
    with torch.no_grad():
        for i in range(len(test_set)):
            data = test_set[i]
            pos = data['pos'].to(device)
            batch = data['batch'].to(device)
            node_atom = data['node_atom'].to(device)
            pred = network(f_in=None, pos=pos, batch=batch, node_atom=node_atom, max_num_neighbors=max_num_neighbors)
            loss = torch.nn.L1Loss()(data["Tm"].to(device), pred * task_std + task_mean)
            test_loss.append(loss.item())
    print(f'test avg MAE={sum(test_loss) / len(test_loss)}')


if __name__ == '__main__':
    main()
