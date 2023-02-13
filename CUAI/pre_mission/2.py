import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./data.csv")
plt.plot(data.x, data.y)
plt.show()