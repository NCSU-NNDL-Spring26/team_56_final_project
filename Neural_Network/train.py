import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import load_multiple_records, print_class_distribution 
from dataset import ECGBeatDataset 
from model import ECGCNN 
from utils import set_seed


def main():
    print("Starting training script...")
    set_seed(42)

    
    record_ids = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209",
    "210", "212", "213", "214", "215", "217", "219", "220",
    "221", "222", "223", "228", "230", "231", "232", "233", "234"]

    record_paths = [f"../../data/data/MITBIHRawData/{rid}" for rid in record_ids]

    X, y = load_multiple_records(record_paths, window_size=200, channel=0)

    print(f"\nCombined data shape: X = {X.shape}, y = {y.shape}")
    print_class_distribution(y)

    dataset = ECGBeatDataset(X, y)
    print(f"\nDataset size: {len(dataset)}")

    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    print(f"Train/Val/Test sizes: {n_train}, {n_val}, {n_test}")

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ECGCNN(input_length=400, num_classes=5).to(device)

    #criterion = nn.CrossEntropyLoss()          #Produced too high of accuracy while excluding minority classes almost entirely. 
                                                #implemented class weights to address this issue and not ignore minority classes.
    import numpy as np

    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))

    num_classes = 5
    weights = []

    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weights.append(1.0/count)

    weights = np.array(weights)
    weights = weights / weights.sum() * num_classes

    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    class_weights[3] *= 0.25

    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                preds = outputs.argmax(dim=1)

                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")

    print("Training complete.")

    from sklearn.metrics import classification_report, confusion_matrix

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
