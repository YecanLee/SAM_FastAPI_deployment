import argparse
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from model import Niconets  
from data import load_data 

from torch.utils.data import DataLoader, TensorDataset

def train_model(model, train_path, lr):
    model.train()

    images, labels = load_data(train_path)

    images_tensor = torch.tensor(images, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    train_dataset = TensorDataset(images_tensor, labels_tensor)
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epochs = 5
    losses = []
    for epoch in range(epochs):
        running_loss = 0
        batch_count = 0
        for batch_images, batch_labels in trainloader:
            optimizer.zero_grad()
            log_probs = model(batch_images)
            loss = criterion(log_probs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1

        average_epoch_loss = running_loss / batch_count
        losses.append(average_epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {average_epoch_loss:.4f}")
    
    torch.save(model.state_dict(), 'model.pth')

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()
    

def evaluate_model(model, test_data_path):
    model.eval()

    images, labels = load_data(test_data_path)

    images_tensor = torch.tensor(images, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    test_dataset = TensorDataset(images_tensor, labels_tensor)
    testloader = DataLoader(test_dataset, batch_size=32)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_images, batch_labels in testloader:
            log_probs = model(batch_images)
            _, predicted = torch.max(log_probs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy*100:.2f}%")



def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Training parser
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--data_path", type=str, required=True)
    parser_train.add_argument("--lr", type=float, default=1e-4)

    # Evaluation parser
    parser_eval = subparsers.add_parser("evaluate")
    parser_eval.add_argument("--data_path", type=str, required=True)
    parser_eval.add_argument("model_path", type=str)

    args = parser.parse_args()

    if args.command == "train":
        model = Niconets()
        train_model(model, args.data_path, args.lr)
    elif args.command == "evaluate":
        model = Niconets()
        model.load_state_dict(torch.load(args.model_path))
        evaluate_model(model, args.data_path)


if __name__ == "__main__":
    main()

# for running in terminal
# python main.py train --data_path r"C:\Users\A\Desktop\final exercise1\train_0.npz" --lr 1e-4
# python main.py evaluate --data_path r"C:\Users\A\Desktop\final exercise1\test.npz" model.pth
