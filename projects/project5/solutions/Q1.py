# Définition de la classe du modèle avec une couche de Dropout
class ModelWithSingleDropout(nn.Module):
    def __init__(self):
        super(ModelWithSingleDropout, self).__init__()
        self.fc1 = nn.Linear(784, 120)
        self.dropout = nn.Dropout(p=0.5)  # créer la couche de dropout avec p=0.5
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Appliquer le dropout après la première couche
        x = self.fc2(x)
        return x