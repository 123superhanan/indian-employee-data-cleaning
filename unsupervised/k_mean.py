import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#sample data
data = {
    'Customer': ['ali','zuban','neha','surha','lela','neon'],
    'Age':['32','45','32','23','67','21'],
    'Spending':['132','245','432','123','67','121']
}
df = pd.DataFrame(data)
print(df)
X = df[['Age','Spending']]
model = KMeans(n_clusters=2,random_state=42,n_init=10)