# main.py

from utils import *
from models import PathVGAE
import warnings
warnings.filterwarnings("ignore")


def train_model(mode='betweenness', epochs=200, learning_rate=0.005, print_kt=True):
    if mode == 'betweenness':
        dataset = torch.load('Data/training_data_betweenness.pth') # contains partitioned LA dataset for batch training
    elif mode == 'closeness':
        dataset = torch.load('Data/training_data_closeness.pth')  # contains partitioned LA dataset for batch training
    else:
        raise ValueError(f"Unsupported centrality type: '{mode}'.")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = PathVGAE(ninput=dataset.model_size, nhid=16, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_kendall_tau = 0

        for batch in dataloader:
            adj_pad = batch['adj']
            adj_pad_t = adj_pad.transpose(1, 2)
            target = batch['target']
            current_size = target.shape[1]
            optimizer.zero_grad()
            y_pred = model(adj_pad, adj_pad_t)
            loss = rank_loss(y_pred, target, current_size)
            kendall_tau = calculate_kendall_tau(y_pred[:current_size], target[:current_size])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_kendall_tau += kendall_tau

        avg_loss = total_loss / len(dataloader)
        avg_kendall_tau = total_kendall_tau / len(dataloader)
        if epoch % 10 == 0:
            if print_kt:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Training KT Score: {avg_kendall_tau}')

    return model, mode


def evaluate_model(model, mode):
    if mode == 'betweenness':
        test_dataset = torch.load('Data/testing_data_betweenness.pth')  # contains samples in this order: Santa Clara, DC, Maricopa, Hennepin, Yellowstone
    elif mode == 'closeness':
        test_dataset = torch.load('Data/testing_data_closeness.pth') # contains samples in this order: Santa Clara, DC, Maricopa, Hennepin, Yellowstone
    else:
        raise ValueError(f"Unsupported centrality type: '{mode}'.")

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    network_names = ['Santa Clara', 'Washington D.C.', 'Maricopa', 'Hennepin', 'Yellowstone']
    model.eval()
    total_test_kendall_tau = 0
    
    print('-------------------------------------------------------------------------------')
    with torch.no_grad():
        for batch, network_name in zip(test_dataloader, network_names):
            adj_pad_test = batch['adj']
            adj_pad_t_test = adj_pad_test.transpose(1, 2)
            target_test = batch['target']
            current_size = target_test.shape[1]

            y_test_pred = model(adj_pad_test, adj_pad_t_test)
            test_kendall_tau = calculate_kendall_tau(y_test_pred[:current_size], target_test[:current_size])
            print(f"{network_name} KT Score: {test_kendall_tau}")

            total_test_kendall_tau += test_kendall_tau

    avg_test_kendall_tau = total_test_kendall_tau / len(test_dataloader)
    print(f"Average Kendall Tau: {avg_test_kendall_tau}")



######## Train and evaluate model ###########
if __name__ == '__main__':
    model, mode = train_model(mode='betweenness')
    evaluate_model(model, mode)
