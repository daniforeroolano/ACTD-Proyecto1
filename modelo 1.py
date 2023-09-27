# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 00:33:44 2023

@author: 57314
"""

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy . estimators import BayesianEstimator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
#importo los datos
dataset = fetch_ucirepo(id=697)
#Separo para que no me quede una tabla
data = dataset['data']
targets = dataset['data']['targets']

# Convierto a dataframe
df_data = pd.DataFrame(data['features'])

# meto el target al data frame
df = pd.concat([df_data, pd.DataFrame({'Target': targets['Target']})], axis=1)

# Especificar las variables que queremos discretizar
variables_a_discretizar = ['Previous qualification (grade)', 'Admission grade','Unemployment rate','Inflation rate','GDP','Curricular units 2nd sem (grade)','Curricular units 1st sem (grade)']

# Especificicar el número de intervalos deseados para cada variable
num_bins = 4

# Crear un nuevo DataFrame para almacenar las variables originales y discretizadas
df_nuevo = pd.DataFrame()

# Discretizar las variables seleccionadas y añadirlas al nuevo DataFrame
for col in variables_a_discretizar:
    df[col+'_bin'] = pd.cut(df[col], bins=num_bins, labels=False)
    df_nuevo[col+'_bin'] = df[col+'_bin']

# Añadir las variables que no se han discretizado al nuevo DataFrame
for col in df.columns:
    if col not in variables_a_discretizar and 'bin' not in col:
        df_nuevo[col] = df[col]

# Obtener los límites de los intervalos para cada variable
bin_limits = {col: pd.cut(df[col], bins=num_bins).unique() for col in variables_a_discretizar}

# Imprimir el nuevo DataFrame con las variables originales y discretizadas
print(df_nuevo)
print("Bin limits:")
print(bin_limits)

df=df_nuevo
print(df.head())
# Cambiar los encabezados
nuevos_encabezados = ['G','M','AH','AI','AJ','AF','Z',"A", "B", "C", "D", "E", "F", "H", "I", "J", "K", "L", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "AA", "AB", "AC", "AD", "AE", "AG", "Target"]
df.columns = nuevos_encabezados
print(df.head())
modelo1=BayesianNetwork([("G", "W"), ("G", "V"), ("V", "Z"), ("W", "Z"), ("AC", "AF"), ("AB", "AF"), ("Z", "Target"), ("AF", "Target"), ("S", "Target"),("D", "Target")])
X = df[["G","V","W","AC","AB","Z","AF","S","D","Target"]]
X_train, X_test = train_test_split(X,test_size=0.2, random_state=43)
#modelo1.fit(data=X_train, estimator = MaximumLikelihoodEstimator) 
estimador_bayesiano = BayesianEstimator(model=modelo1, data=X_train)
# Estimar las CPDs (tablas de probabilidades condicionales) para todas las variables
cpd_G = estimador_bayesiano.estimate_cpd('G')
cpd_V = estimador_bayesiano.estimate_cpd('V')
cpd_W = estimador_bayesiano.estimate_cpd('W')
cpd_AC = estimador_bayesiano.estimate_cpd('AC')
cpd_AB = estimador_bayesiano.estimate_cpd('AB')
cpd_Z = estimador_bayesiano.estimate_cpd('Z')
cpd_AF = estimador_bayesiano.estimate_cpd('AF')
cpd_S = estimador_bayesiano.estimate_cpd('S')
cpd_D = estimador_bayesiano.estimate_cpd('D')
cpd_Target = estimador_bayesiano.estimate_cpd('Target')

# Añadir las CPDs estimadas al modelo
modelo1.add_cpds(cpd_G, cpd_V, cpd_W, cpd_AC, cpd_AB, cpd_Z, cpd_AF, cpd_S, cpd_D, cpd_Target)

# Verificar si el modelo es válido
valido = modelo1.check_model()

for i in modelo1.nodes():
    print(modelo1.get_cpds(i)) 
print(X.columns)
new_column_values = []

# Calcular la frecuencia de valores en la columna "Target"
frecuencia = X_test['Target'].value_counts()

# Mostrar la frecuencia
print(frecuencia)

for index, row in X_test.iterrows():  
    G_value = row["G"]
    V_value = row["V"]
    W_value = row["W"]
    AC_value = row["AC"]
    AB_value = row["AB"]
    Z_value = row["Z"]
    AF_value = row["AF"]
    S_value = row["S"]
    D_value = row["D"]
    infer = VariableElimination(modelo1)
    result= infer.query(["Target"], evidence={"G": G_value, "V": V_value, "W": W_value, "AC": AC_value, "AB": AB_value, "Z": Z_value, "AF": AF_value, "S": S_value, "D": D_value})
    result_values = result.values
    max_prob_index = np.argmax(result.values)
    max_prob_option = modelo1.get_cpds('Target').state_names['Target'][max_prob_index]
              
    new_column_values.append(max_prob_option)
 
X_test["Estimado Target"] = new_column_values
print(X_test["Estimado Target"])


df2=X_test[["Target","Estimado Target"]]
print(df2.head())

# Agrupar por las columnas y contar las combinaciones únicas
combination_counts = df2.groupby(["Target", "Estimado Target"]).size().reset_index(name="count")

print(combination_counts)

print(len(X_test))