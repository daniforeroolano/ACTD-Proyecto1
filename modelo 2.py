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
modelo1=BayesianNetwork([ ("Z", "Target"), ("AF", "Target"),("P", "Target"),("Q", "Target"),("D", "Target"), ("D","Z"),("D","AF"),("Z","AF"),("R","AF"),("R","Z"),("G","AF"),("G","Z"),("T","AF"),("T","Z"),("S","AF"),("S","Z"),("W","Y"),("Y","Z"),("AC","AE"),("AE","AF")])
X = df[["Z","Target","AF","D","R","G","T","S","W","Y","AC","AE","P","Q"]]
#V
#AB
#-R
#-T
#-Y
#-AE
#-I
#-J
X_train, X_test = train_test_split(X,test_size=0.2, random_state=43)
modelo1.fit(data=X_train, estimator = MaximumLikelihoodEstimator) 
#estimador_bayesiano = BayesianEstimator(model=modelo1, data=X_train)
# Estimar las CPDs (tablas de probabilidades condicionales) para todas las variables
#cpd_Z = estimador_bayesiano.estimate_cpd('Z')
#cpd_Target = estimador_bayesiano.estimate_cpd('Target')
#cpd_AF = estimador_bayesiano.estimate_cpd('AF')
#cpd_D = estimador_bayesiano.estimate_cpd('D')
#cpd_R = estimador_bayesiano.estimate_cpd('R')
#cpd_G = estimador_bayesiano.estimate_cpd('G')
#cpd_T = estimador_bayesiano.estimate_cpd('T')
#cpd_S = estimador_bayesiano.estimate_cpd('S')
#cpd_W = estimador_bayesiano.estimate_cpd('W')
#cpd_Y = estimador_bayesiano.estimate_cpd('Y')
#cpd_AC = estimador_bayesiano.estimate_cpd('AC')
#cpd_AE = estimador_bayesiano.estimate_cpd('AE')
#cpd_P = estimador_bayesiano.estimate_cpd('P')
#cpd_Q = estimador_bayesiano.estimate_cpd('Q')

# Añadir las CPDs estimadas al modelo
#modelo1.add_cpds(cpd_Z, cpd_Target, cpd_AF, cpd_D, cpd_R, cpd_G, cpd_T, cpd_S, cpd_W,cpd_Y,cpd_AC,cpd_AE,cpd_P,cpd_Q)

# Verificar si el modelo es válido
#valido = modelo1.check_model()
#print(valido)

for i in modelo1.nodes():
    print(modelo1.get_cpds(i)) 
print(X.columns)
new_column_values = []



for index, row in X_test.iterrows():  
    Z_value = row["Z"]
    AF_value = row["AF"]
    D_value = row["D"]
    R_value = row["R"]
    G_value = row["G"]
    T_value = row["T"]
    S_value = row["S"]
    W_value = row["W"]
    Y_value = row["Y"]
    AC_value = row["AC"]
    AE_value = row["AE"]
    P_value = row["P"]
    Q_value = row["Q"]
    
    infer = VariableElimination(modelo1)
    result= infer.query(["Target"], evidence={"Z": Z_value, "AF": AF_value, "D": D_value, "R": R_value, "G": G_value, "T": T_value, "S": S_value,
                                              "W":W_value,"Y":Y_value,"AC":AC_value,"AE":AE_value,"P":P_value,"Q":Q_value})
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