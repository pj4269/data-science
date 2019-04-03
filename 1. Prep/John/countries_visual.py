import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("countries of the world.csv")

print (df[['GDP ($ per capita)', 'Pop. Density (per sq. mi.)']].head())
print (df.dtypes)

df['Pop. Density (per sq. mi.)'] = df['Pop. Density (per sq. mi.)'].str.replace(',','.')

print (df[['GDP ($ per capita)', 'Pop. Density (per sq. mi.)']].head())

df['Pop. Density (per sq. mi.)'] = df['Pop. Density (per sq. mi.)'].apply(float)

plt.scatter( df['GDP ($ per capita)'], df['Pop. Density (per sq. mi.)'],color='red', alpha=0.5)
plt.title('Scatter Plot')
plt.xlabel('GDP per Capita')
plt.ylabel('Pop Density')
plt.show()

