import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, root):
        self.data = []
        self.labels = []
        self.label2idx = {}
        for i, folder in enumerate(os.listdir(root)):
            self.label2idx[folder] = i
            path = os.path.join(root, folder)
            for file in os.listdir(path):
                if file.endswith(".wav") or file.endswith(".mp3"):
                    self.data.append(os.path.join(path, file))
                    self.labels.append(i)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.data[idx])
        mfcc = torchaudio.transforms.MFCC(n_mfcc=40)(waveform).squeeze(0)
        return mfcc, self.labels[idx]

    def __len__(self):
        return len(self.data)

dataset = AudioDataset("dataset/voice")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32*9*9, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # add channel
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = AudioCNN(len(dataset.label2idx))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for mfcc, labels in loader:
        optimizer.zero_grad()
        outputs = model(mfcc)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

torch.save(model, "models/voice_model.pth")
print("Voice model saved!")
