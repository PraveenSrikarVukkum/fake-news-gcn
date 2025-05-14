from utils.preprocess import load_data, tfidf_features, split_data
from models.gcn_model import GCNClassifier
from torch_geometric.data import Data
import torch
from sklearn.metrics import accuracy_score
import numpy as np

# Load and preprocess
df = load_data("data/fake_news.csv")
features, labels = tfidf_features(df)
X_train, X_test, y_train, y_test = split_data(features, labels)

# Graph simulation (use sentence-level graph in production)
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
edge_index = edge_index.t().contiguous()

data = Data(
    x=torch.tensor(X_train, dtype=torch.float),
    edge_index=edge_index,
    y=torch.tensor(y_train, dtype=torch.long)
)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCNClassifier(input_dim=X_train.shape[1], hidden_dim=64, output_dim=2).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = torch.nn.functional.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "gcn_model.pth")
