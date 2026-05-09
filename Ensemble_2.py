# ============================================================
# Stacking Ensemble với các base models đã lưu dạng .pt
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# ------------------------------------------------------------
# 1. Cấu hình
# ------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 2
INPUT_DIM = 30
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001


# ------------------------------------------------------------
# 2. Định nghĩa kiến trúc base model
# ------------------------------------------------------------
# Lưu ý:
# Khi load state_dict từ .pt, kiến trúc model phải giống
# kiến trúc đã dùng lúc train base model.

class BaseMLP(nn.Module):
    def __init__(self, input_dim=30, num_classes=2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ------------------------------------------------------------
# 3. Định nghĩa meta-model
# ------------------------------------------------------------
# Nếu có 3 base models, mỗi model trả về xác suất 2 class,
# thì input của meta-model là:
#
# 3 * 2 = 6 features

class MetaModel(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ------------------------------------------------------------
# 4. Load dataset demo
# ------------------------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ------------------------------------------------------------
# 5. Load các base models từ file .pt
# ------------------------------------------------------------

base_model_paths = [
    "models/base_model_1.pt",
    "models/base_model_2.pt",
    "models/base_model_3.pt"
]

base_models = []

for path in base_model_paths:
    model = BaseMLP(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(path, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    base_models.append(model)

print(f"Loaded {len(base_models)} base models.")


# ------------------------------------------------------------
# 6. Hàm tạo stacking features
# ------------------------------------------------------------
# Hàm này lấy output của từng base model.
# Ta dùng softmax để chuyển logits thành xác suất.
#
# Ví dụ:
# base_model_1 output: [0.2, 0.8]
# base_model_2 output: [0.4, 0.6]
# base_model_3 output: [0.1, 0.9]
#
# stacking feature sẽ là:
# [0.2, 0.8, 0.4, 0.6, 0.1, 0.9]

def create_stacking_features(base_models, dataloader):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(DEVICE)

            batch_predictions = []

            for model in base_models:
                logits = model(X_batch)

                probabilities = torch.softmax(logits, dim=1)

                batch_predictions.append(probabilities)

            stacking_features = torch.cat(batch_predictions, dim=1)

            all_features.append(stacking_features.cpu())
            all_labels.append(y_batch)

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_features, all_labels


# ------------------------------------------------------------
# 7. Tạo dữ liệu mới cho meta-model
# ------------------------------------------------------------

X_meta_train, y_meta_train = create_stacking_features(
    base_models,
    train_loader
)

X_meta_test, y_meta_test = create_stacking_features(
    base_models,
    test_loader
)

print("Meta train shape:", X_meta_train.shape)
print("Meta test shape:", X_meta_test.shape)


# ------------------------------------------------------------
# 8. Huấn luyện meta-model
# ------------------------------------------------------------

meta_input_dim = len(base_models) * NUM_CLASSES

meta_model = MetaModel(
    input_dim=meta_input_dim,
    num_classes=NUM_CLASSES
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    meta_model.parameters(),
    lr=LEARNING_RATE
)

meta_train_dataset = TensorDataset(
    X_meta_train,
    y_meta_train
)

meta_train_loader = DataLoader(
    meta_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

for epoch in range(EPOCHS):
    meta_model.train()

    total_loss = 0.0

    for X_batch, y_batch in meta_train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = meta_model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(meta_train_loader)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")


# ------------------------------------------------------------
# 9. Đánh giá meta-model
# ------------------------------------------------------------

meta_model.eval()

with torch.no_grad():
    X_meta_test = X_meta_test.to(DEVICE)

    test_logits = meta_model(X_meta_test)

    y_pred = torch.argmax(test_logits, dim=1).cpu().numpy()

y_true = y_meta_test.numpy()

accuracy = accuracy_score(y_true, y_pred)

print("\n===== STACKING ENSEMBLE RESULT =====")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=data.target_names
))


# ------------------------------------------------------------
# 10. Lưu meta-model
# ------------------------------------------------------------

torch.save(
    meta_model.state_dict(),
    "models/meta_model.pt"
)

print("\nSaved meta-model to models/meta_model.pt")
