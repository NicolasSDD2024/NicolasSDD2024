# Définition du modèle avec Dropout
class RandomDropoutModel(nn.Module):
    def __init__(self):
        super(RandomDropoutModel, self).__init__()
        self.dropout1 = nn.Dropout(p=0.25) # Dropout avec une probabilité de 25%
        self.fc1 = nn.Linear(784, 120)
        self.dropout2 = nn.Dropout(p=0.4)  # Dropout avec une probabilité de 40%
        self.fc2 = nn.Linear(120, 60)
        self.dropout3 = nn.Dropout(p=0.4)  # Dropout avec une probabilité de 40%
        self.fc3 = nn.Linear(60, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.dropout1(x)  # Appliquer le premier dropout
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)  # Appliquer le deuxième dropout
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)  # Appliquer le troisième dropout
        x = self.fc3(x)
        return x