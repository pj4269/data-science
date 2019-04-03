import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("countries of the world.csv")

print df.head()


plt.scatter( df['GDP ($ per capita)'], df['Pop. Density (per sq. mi.)'],color='red')
plt.show()

N = 500
x = np.random.rand(N)
y = np.random.rand(N)
colors = (0,0,0)
area = np.pi*3
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
