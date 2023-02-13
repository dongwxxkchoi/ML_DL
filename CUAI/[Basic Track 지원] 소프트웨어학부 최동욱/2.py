import pandas as pd
import matplotlib.pyplot as plt

# read data.csv and save as dataframe
data = pd.read_csv("./data.csv")
# graph whose x-axis represents data.x, y-axis represents data.y
plt.plot(data.x, data.y)
# show the graph on the monitor
plt.show()

