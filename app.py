import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# carregando dataset
df = pd.read_csv("DBcarros.csv")

# matriz de correlação ja com os dados tratados e somente com as variaveis necessarias
plt.rcParams["figure.figsize"] = (18, 9)
ax = sns.heatmap(df.corr(), annot=True)
plt.show()

# coeficies x e y
y = df['horsepower']
x = df['acceleration']

# convertendo meu x para uma matriz bidimensional
x = np.array(x).reshape(-1, 1)

modelo = LinearRegression()
modelo.fit(x, y)

coef_angular = modelo.coef_
coef_linear = modelo.intercept_

# y = a+bx -> gerando a reta da regressao linear
plt.scatter(x, y)
plt.plot(x, coef_linear + coef_angular*x, color='red')
plt.show()
