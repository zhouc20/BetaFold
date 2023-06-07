import torch
from tqdm import tqdm
from data_loading import StructDataset
from equiformer.graph_attention_transformer import graph_attention_transformer_nonlinear_l2


def main():
    max_epochs = 10000
    device = torch.device('cuda')
    
    train_set = StructDataset(path='./StructuredDatasets/train_dataset.pkl', batch_size=16, random=True)
    test_set = StructDataset(path='./StructuredDatasets/test2_dataset.pkl', batch_size=1, random=False)
    task_mean, task_std = 53.14, 11.78
    
    network = graph_attention_transformer_nonlinear_l2(
        irreps_in='5x0e', 
        radius=5.0, 
        num_basis=128, 
        out_channels=1, 
        task_mean=task_mean, 
        task_std=task_std, 
        atomref=None, 
        drop_path=0.0
    ).to(device)
    n_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f'{n_parameters} parameters')
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, network.parameters()), lr=2e-4, weight_decay=1e-8)
    criterion = torch.nn.L1Loss()
    for epoch in range(max_epochs):
        with tqdm(desc=f'Epoch {epoch}/{max_epochs}', total=len(train_set.structures), unit='proteins') as pbar:
            for i in range(len(train_set)):
                data = train_set[i]
                pos = data['pos'].to(device)
                batch = data['batch'].to(device)
                node_atom = data['node_atom'].to(device)
                pred = network(f_in=None, pos=pos, batch=batch, node_atom=node_atom)
                loss = criterion(pred, (data['Tm'].to(device) - task_mean) / task_std)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    pbar.set_postfix({'loss': f'{criterion(data["Tm"].to(device), pred * task_std + task_mean).item():.2f}'})
                pbar.update(train_set.batch_size)
                
                
        


if __name__ == '__main__':
    main()
