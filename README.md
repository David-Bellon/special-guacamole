# special-guacamole
Okay let's try to detect fraud in transactions using an autoencoder. Simply we train an autoencoder using the data of real transactions and then we pass new data we have, if the output is way different from the input we may say that the transaction is fake otherwise we assume it is true.  
## The Code

```python
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from arch import Autoencoder
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device("cuda")

df = pd.read_csv("creditcard.csv")
df["Amount"] = (df["Amount"] - df["Amount"].mean())/(df["Amount"].std())
df = df.drop(columns=["Time"])
real = df.loc[df["Class"] == 0]
false = df.loc[df["Class"] != 0]

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

print(len(df_train))

class MyClass(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        aux = self.data.iloc[index]
        data = aux.drop("Class")
        label = aux["Class"]
        data = torch.tensor(data, dtype=torch.float).to(device)
        label = torch.tensor(label, dtype=torch.float).to(device)

        return data, label


train_set = DataLoader(
    MyClass(df_train),
    batch_size=128,
    shuffle=True
)

test_set = DataLoader(
    MyClass(df_test),
    batch_size=100
)

fake_test = DataLoader(
    MyClass(false),
    batch_size=100
)

model = Autoencoder(29, 8).to(device)
loss = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)


def train(input):
    optim.zero_grad()

    out = model(input)
    lost = loss(out, input)

    lost.backward()
    optim.step()

    return lost

epochs = 20
for epoch in range(epochs):
    epoch_loss = 0.0
    for i, (data, label) in tqdm(enumerate(train_set)):
        epoch_loss += train(data)

    print(f"Epoch: {epoch} ---- Autoencoder Loss: {epoch_loss.item()/i}")

torch.save(model, "autoencoder.pt")


```

I don't know. It's code.  
## The test
Now we get the part of the data we reserve for test. First we use real transactions so we can see how good it performs under real conditions:  
```python

model = torch.load("autoencoder.pt")
ok = 0
not_ok = 0
with torch.no_grad():
    for (data, label) in fake_test: # Here you put test_set to do it with real transactions
        out = model(data)
        for i, value in enumerate(out):
            if loss(value, data[i]) >= 0.75:
                not_ok += 1
            else:
                ok += 1
print(len(fake_test))
print(ok)
print(not_ok)

plt.bar(x=["Right", "Wrong"], height=[ok, not_ok])
plt.show()
```
And if we plot it we get the following:  
![rights](https://user-images.githubusercontent.com/91338053/232322861-76f9ef7a-67b1-48ed-a550-977c851cff3d.png)  
As you can see it performs ok so not bad.  

Now lets see how it does with the fake transactions.  
![wrongs](https://user-images.githubusercontent.com/91338053/232322896-1da9b6ea-1614-4e8a-870f-5de83a8d1885.png)  
As you can see not bad as well. Hey it's just an autoencoder an the criteria it is just a margin error we decide so not bad.

## Ending
I'm dying ok? so not too much time. It works somehow, not incredible but works. Can be more precise if you put the threshold error lower but you may get more error on real ones. Guess you can't have everything.  
