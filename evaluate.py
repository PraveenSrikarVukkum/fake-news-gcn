from models.gcn_model import GCNClassifier
from utils.preprocess import load_data, tfidf_features
import torch

df = load_data("data/fake_news.csv")
features, labels = tfidf_features(df)

model = GCNClassifier(input_dim=features.shape[1], hidden_dim=64, output_dim=2)
model.load_state_dict(torch.load("gcn_model.pth"))
model.eval()

with torch.no_grad():
    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t().contiguous()
    out = model(Data(x=x, edge_index=edge_index))
    preds = out.argmax(dim=1)
    acc = (preds == torch.tensor(labels)).sum().item() / len(labels)
    print(f"Test Accuracy: {acc:.4f}")
