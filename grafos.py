# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 00:17:45 2023

@author: 57314
"""


from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
direccion="C:/Users/57314/Documents/universidad/Analitica/proyecto/dataDIC.csv"
df=pd.read_csv(direccion)
print(df.head())
modelo1=BayesianNetwork([("G", "W"), ("G", "V"), ("F", "V"), ("F", "Z"), ("V", "Z"), ("W", "Z"), ("AC", "AF"), ("AB", "AF"), ("Z", "AF"), ("Z", "Target"), ("AH", "Target"), ("T", "Target"), ("AF", "Target"), ("I", "Target"), ("J", "Target"), ("S", "Target")])
X = df[["G","F","V","W","AC","AB","Z","AH","T","AF","I","J","S","Target"]]
X_train, X_test = train_test_split(X,test_size=0.2, random_state=42)
modelo1.fit(data=X_train, estimator = MaximumLikelihoodEstimator) 

for i in modelo1.nodes():
    print(modelo1.get_cpds(i)) 

new_column_values = []
for index, row in X_test.iterrows():
    G_value = row["G"]
    F_value = row["F"]
    V_value = row["V"]
    W_value = row["W"]
    AC_value = row["AC"]
    AB_value = row["AB"]
    Z_value = row["Z"]
    AH_value = row["AH"]
    T_value = row["T"]
    AF_value = row["AF"]
    I_value = row["I"]
    J_value = row["J"]
    S_value = row["S"]
    infer = VariableElimination(modelo1)
    result= infer.query(["Target"], evidence={"G": G_value, "F": F_value, "V": V_value, "W": W_value, "AC": AC_value, "AB": AB_value, "Z": Z_value, "AH": AH_value, "T": T_value, "AF": AF_value, "I": I_value, "J": J_value, "S": S_value})
    result_values = result.values
    indices_100_percent = [i for i, value in enumerate(result_values) if value > 0.5]
    possible_results = ["Dropout", "Graduate","Enrolled"]  
    results_with_100_percent = [possible_results[i] for i in indices_100_percent]
    new_column_values.append(results_with_100_percent)

import networkx as nx
import matplotlib.pyplot as plt

# Crear un objeto de grafo dirigido
Graf = nx.DiGraph()

# Agregar nodos al grafo (variables)
Graf.add_nodes_from(["G", "W", "V", "F", "Z", "AC", "AB", "AH", "T", "AF", "I", "J", "S", "Target"])

# Agregar arcos (relaciones) entre nodos
edges = [("G", "W"), ("G", "V"), ("F", "V"), ("F", "Z"), ("V", "Z"), ("W", "Z"), 
         ("AC", "AF"), ("AB", "AF"), ("Z", "AF"), ("Z", "Target"), ("AH", "Target"), 
         ("T", "Target"), ("AF", "Target"), ("I", "Target"), ("J", "Target"), ("S", "Target")]

Graf.add_edges_from(edges)

# Dibujar la red bayesiana
pos = nx.spring_layout(Graf, seed=42)  # Posiciona los nodos
nx.draw(Graf, pos, with_labels=True, node_size=800, node_color='skyblue', font_size=8, font_color='black', font_weight='bold', arrowsize=20)

# Mostrar el grafo
plt.title("Red Bayesiana")
plt.show()


#Modelo 2
modelo2=BayesianNetwork([("S", "V"), ("S", "Z"), ("S", "AF"), ("S", "AB"), ("T", "V"), ("T", "Z"), ("T", "AF"), ("T", "AB"), ("R", "V"), ("R", "Z"), ("R", "AF"), ("R", "AB"), ("M", "V"), ("M", "Z"), ("M", "AF"), ("M", "AB"), ("F", "V"), ("F", "Z"), ("F", "AF"), ("F", "AB"), ("G", "V"), ("G", "Z"), ("G", "AF"), ("G", "AB"), ("W", "V"), ("W", "Z"), ("AC", "AF"), ("AC", "AB"), ("V", "Target"), ("Z", "Target"), ("AF", "Target"), ("AB", "Target")])
X = df[["S", "T", "R", "M", "F", "G", "W", "AC", "V", "Z", "AF", "AB", "Target"]]
X_train, X_test = train_test_split(X,test_size=0.2, random_state=42)
modelo2.fit(data=X_train, estimator = MaximumLikelihoodEstimator) 

for i in modelo2.nodes():
    print(modelo2.get_cpds(i)) 

new_column_values = []
for index, row in X_test.iterrows():
    S_value = row["S"]
    T_value = row["T"]
    R_value = row["R"]
    M_value = row["M"]
    F_value = row["F"]
    G_value = row["G"]
    W_value = row["W"]
    AC_value = row["AC"]
    V_value = row["V"]
    Z_value = row["Z"]
    AF_value = row["AF"]
    AB_value = row["AB"]
    infer = VariableElimination(modelo2)
    result= infer.query(["Target"], evidence={"S": S_value, "T": T_value, "R": R_value, "M": M_value, "F": F_value, "G": G_value, "W": W_value, "AC": AC_value, "V": V_value, "Z": Z_value, "AF": AF_value, "AB": AB_value})
    result_values = result.values
    indices_100_percent = [i for i, value in enumerate(result_values) if value > 0.5]
    possible_results = ["Dropout", "Graduate","Enrolled"]  
    results_with_100_percent = [possible_results[i] for i in indices_100_percent]
    new_column_values.append(results_with_100_percent)

import networkx as nx
import matplotlib.pyplot as plt

# Crear un objeto de grafo dirigido
Graf = nx.DiGraph()

# Agregar nodos al grafo (variables)
Graf.add_nodes_from(["S", "T", "R", "M", "F", "G", "W", "AC", "V", "Z", "AF", "AB", "Target"])

# Agregar arcos (relaciones) entre nodos
edges = [("S", "V"), ("S", "Z"), ("S", "AF"), ("S", "AB"), ("T", "V"), ("T", "Z"), ("T", "AF"), ("T", "AB"), ("R", "V"), ("R", "Z"), ("R", "AF"), ("R", "AB"), ("M", "V"), ("M", "Z"), ("M", "AF"), ("M", "AB"), ("F", "V"), ("F", "Z"), ("F", "AF"), ("F", "AB"), ("G", "V"), ("G", "Z"), ("G", "AF"), ("G", "AB"), ("W", "V"), ("W", "Z"), ("AC", "AF"), ("AC", "AB"), ("V", "Target"), ("Z", "Target"), ("AF", "Target"), ("AB", "Target")]

Graf.add_edges_from(edges)

# Dibujar la red bayesiana
pos = nx.spring_layout(Graf, seed=42)  # Posiciona los nodos
nx.draw(Graf, pos, with_labels=True, node_size=800, node_color='skyblue', font_size=8, font_color='black', font_weight='bold', arrowsize=20)

# Mostrar el grafo
plt.title("Red Bayesiana")
plt.show()

#Modelo 3
modelo3=BayesianNetwork([("M", "Target"), ("Z", "Target"), ("AF", "Target"), ("W", "Target"), ("AC", "Target"), ("F", "Target"), ("Q", "Target"), ("P", "Target"), ("R", "Target"), ("E", "Target"), ("S", "Target"), ("T", "Target"), ("O", "Target"), ("N", "Target"), ("AI", "Target"), ("AH", "Target")])
X = df[["M", "Z", "AF", "W", "AC", "F", "Q", "P", "R", "E", "S", "T", "O", "N", "AI", "AH", "Target"]]
X_train, X_test = train_test_split(X,test_size=0.2, random_state=42)
modelo2.fit(data=X_train, estimator = MaximumLikelihoodEstimator) 

for i in modelo3.nodes():
    print(modelo3.get_cpds(i)) 

new_column_values = []
for index, row in X_test.iterrows():
    M_value = row["M"]
    Z_value = row["Z"]
    AF_value = row["AF"]
    W_value = row["W"]
    AC_value = row["AC"]
    F_value = row["F"]
    Q_value = row["Q"]
    P_value = row["P"]
    R_value = row["R"]
    E_value = row["E"]
    S_value = row["S"]
    T_value = row["T"]
    O_value = row["O"]
    N_value = row["N"]
    AI_value = row["AI"]
    AH_value = row["AH"]
    infer = VariableElimination(modelo3)
    result= infer.query(["Target"], evidence={"M": M_value, "Z": Z_value, "AF": AF_value, "W": W_value, "AC": AC_value, "F": F_value, "Q": Q_value, "P": P_value, "R": R_value, "E": E_value, "S": S_value, "T": T_value, "O": O_value, "N": N_value, "AI": AI_value, "AH": AH_value})
    result_values = result.values
    indices_100_percent = [i for i, value in enumerate(result_values) if value > 0.5]
    possible_results = ["Dropout", "Graduate","Enrolled"]  
    results_with_100_percent = [possible_results[i] for i in indices_100_percent]
    new_column_values.append(results_with_100_percent)

import networkx as nx
import matplotlib.pyplot as plt

# Crear un objeto de grafo dirigido
Graf = nx.DiGraph()

# Agregar nodos al grafo (variables)
Graf.add_nodes_from(["M", "Z", "AF", "W", "AC", "F", "Q", "P", "R", "E", "S", "T", "O", "N", "AI", "AH", "Target"])

# Agregar arcos (relaciones) entre nodos
edges = [("M", "Target"), ("Z", "Target"), ("AF", "Target"), ("W", "Target"), ("AC", "Target"), ("F", "Target"), ("Q", "Target"), ("P", "Target"), ("R", "Target"), ("E", "Target"), ("S", "Target"), ("T", "Target"), ("O", "Target"), ("N", "Target"), ("AI", "Target"), ("AH", "Target")]

Graf.add_edges_from(edges)

# Dibujar la red bayesiana
pos = nx.spring_layout(Graf, seed=42)  # Posiciona los nodos
nx.draw(Graf, pos, with_labels=True, node_size=800, node_color='skyblue', font_size=8, font_color='black', font_weight='bold', arrowsize=20)

# Mostrar el grafo
plt.title("Red Bayesiana")
plt.show()