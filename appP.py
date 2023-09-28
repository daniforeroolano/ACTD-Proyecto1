# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:31:55 2023

@author: 57314
"""

import dash
from dash import dcc  # dash core components
from dash import html # dash html components
import plotly.express as px
import pandas as pd
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
import dash
from dash import dcc  # dash core components
from dash import html # dash html components
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State

#Modelo
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
df=df_nuevo
nuevos_encabezados = ['G','M','AH','AI','AJ','AF','Z',"A", "B", "C", "D", "E", "F", "H", "I", "J", "K", "L", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "AA", "AB", "AC", "AD", "AE", "AG", "Target"]
df.columns = nuevos_encabezados
modelo1=BayesianNetwork([ ("Z", "Target"), ("AF", "Target"),("P", "Target"),("Q", "Target"),("D", "Target"), ("D","Z"),("D","AF"),("Z","AF"),("R","AF"),("R","Z"),("G","AF"),("G","Z"),("T","AF"),("T","Z"),("S","AF"),("S","Z"),("W","Y"),("Y","Z"),("AC","AE"),("AE","AF")])
X = df[["Z","Target","AF","D","R","G","T","S","W","Y","AC","AE","P","Q"]]
X_train, X_test = train_test_split(X,test_size=0.2, random_state=43)

estimador_bayesiano = BayesianEstimator(model=modelo1, data=X_train)
# Estimar las CPDs (tablas de probabilidades condicionales) para todas las variables
cpd_Z = estimador_bayesiano.estimate_cpd('Z')
cpd_Target = estimador_bayesiano.estimate_cpd('Target')
cpd_AF = estimador_bayesiano.estimate_cpd('AF')
cpd_D = estimador_bayesiano.estimate_cpd('D')
cpd_R = estimador_bayesiano.estimate_cpd('R')
cpd_G = estimador_bayesiano.estimate_cpd('G')
cpd_T = estimador_bayesiano.estimate_cpd('T')
cpd_S = estimador_bayesiano.estimate_cpd('S')
cpd_W = estimador_bayesiano.estimate_cpd('W')
cpd_Y = estimador_bayesiano.estimate_cpd('Y')
cpd_AC = estimador_bayesiano.estimate_cpd('AC')
cpd_AE = estimador_bayesiano.estimate_cpd('AE')
cpd_P = estimador_bayesiano.estimate_cpd('P')
cpd_Q = estimador_bayesiano.estimate_cpd('Q')

# Añadir las CPDs estimadas al modelo
modelo1.add_cpds(cpd_Z, cpd_Target, cpd_AF, cpd_D, cpd_R, cpd_G, cpd_T, cpd_S, cpd_W,cpd_Y,cpd_AC,cpd_AE,cpd_P,cpd_Q)


#Codigo para la respuesta
infer = VariableElimination(modelo1)

result= infer.query(["Target"], evidence={"Z": 3, "AF": 3, "D": 9254, "R": 1, "G": 2, "T": 19, "S": 0,
                                              "W":6,"Y":6,"AC":6,"AE":6,"P":0,"Q":0})
result_values = result.values


#dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
panel_1_content = html.Div([
    html.H2("Instructions"),
    html.P("This aplication is a tool that predicts the academic success of a student, based on personal, economic and academic atributes."),
    html.P("Follow these instructions:"),
    html.P("1. fill up all the information required"),
    html.P("2. click on the execution boton"),
    html.P("3. then the results will show up"),
])
panel_2_content = html.Div([
    html.H2("Data record"),
    html.P("This application is a tool that predicts the academic success of a student, based on personal, economic, and academic attributes."),
    html.Div([  # Agregamos un contenedor para organizar los elementos
        html.H6("Scholarship holder:", style={'display': 'inline-block', 'margin-right': '10px'}),  # Establecemos el estilo inline-block
        dcc.Dropdown(
            id='S',  # Identificador único para la lista desplegable
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0},
            ],
            multi=False,  # Cambiado a False para permitir una sola selección
            value='1',  # Opción preseleccionada
            style={'display': 'inline-block'}  # Establecemos el estilo inline-block
        ),
        html.H6("    Age at enrollment:", style={'display': 'inline-block', 'margin-right': '10px'}),  # Establecemos el estilo inline-block
        dcc.Dropdown(
            id='T',  # Identificador único para la lista desplegable
            options=[
                {'label': 17, 'value': 17},
                {'label': 18, 'value': 18},
                {'label': 19, 'value': 19},
                {'label': 20, 'value': 20},
                {'label': 21, 'value': 21},
                {'label': 22, 'value': 22},
                {'label': 23, 'value': 23},
                {'label': 24, 'value': 24},
                {'label': 25, 'value': 25},
                {'label': 26, 'value': 26},
                {'label': 27, 'value': 27},
                {'label': 28, 'value': 28},
                {'label': 29, 'value': 29},
                {'label': 30, 'value': 30},
                {'label': 31, 'value': 31},
                {'label': 32, 'value': 32},
                {'label': 33, 'value': 33},
                {'label': 34, 'value': 34},
                {'label': 35, 'value': 35},
                {'label': 36, 'value': 36},
                {'label': 37, 'value': 37},
                {'label': 38, 'value': 38},
                {'label': 39, 'value': 39},
                {'label': 40, 'value': 40},
                {'label': 41, 'value': 41},
                {'label': 42, 'value': 42},
                {'label': 43, 'value': 43},
                {'label': 44, 'value': 44},
                {'label': 45, 'value': 45},
                {'label': 46, 'value': 46},
                {'label': 47, 'value': 47},
                {'label': 48, 'value': 48},
                {'label': 49, 'value': 49},
                {'label': 50, 'value': 50},
                {'label': 51, 'value': 51},
                {'label': 52, 'value': 52},
                {'label': 53, 'value': 53},
                {'label': 54, 'value': 54},
                {'label': 55, 'value': 55},
                {'label': 57, 'value': 57},
                {'label': 58, 'value': 58},
                {'label': 59, 'value': 59},
                {'label': 60, 'value': 60},
                {'label': 61, 'value': 61},
                {'label': 62, 'value': 62},
                {'label': 70, 'value': 70},
            ],
            multi=False,  # Cambiado a False para permitir una sola selección
            value='37',  # Opción preseleccionada
            style={'display': 'inline-block','width': '50px'}  # Establecemos el estilo inline-block
        ),
        html.H6("    Gender:", style={'display': 'inline-block', 'margin-right': '10px'}),  # Establecemos el estilo inline-block
        dcc.Dropdown(
            id='R',  # Identificador único para la lista desplegable
            options=[
                {'label': 'Male', 'value': 1},
                {'label': 'Female', 'value': 0},
            ],
            multi=False,  # Cambiado a False para permitir una sola selección
            value='1',  # Opción preseleccionada
            style={'display': 'inline-block','width': '80px'}  # Establecemos el estilo inline-block
        ),
        html.H6("    Previous qualification (grade):", style={'display': 'inline-block', 'margin-right': '10px'}),  # Establecemos el estilo inline-block
        dcc.Dropdown(
            id='G',  # Identificador único para la lista desplegable
            options=[
                {'label': 'less than 118', 'value': 0},
                {'label': 'Between 118 and 142', 'value': 1},
                {'label': 'Between 142 and 166', 'value': 2},
                {'label': 'more than 166', 'value': 3},
            ],
            multi=False,  # Cambiado a False para permitir una sola selección
            value='1',  # Opción preseleccionada
            style={'display': 'inline-block','width': '200px'}  # Establecemos el estilo inline-block
        ),
        html.H6("Curricular units 2nd sem (enrolled):", style={'display': 'inline-block', 'margin-right': '10px'}),  # Establecemos el estilo inline-block
        dcc.Dropdown(
            id='AC',  # Identificador único para la lista desplegable
            options=[
                {'label': 0, 'value': 0},
                {'label': 1, 'value': 1},
                {'label': 2, 'value': 2},
                {'label': 3, 'value': 3},
                {'label': 4, 'value': 4},
                {'label': 5, 'value': 5},
                {'label': 6, 'value': 6},
                {'label': 7, 'value': 7},
                {'label': 8, 'value': 8},
                {'label': 9, 'value': 9},
                {'label': 10, 'value': 10},
                {'label': 11, 'value': 11},
                {'label': 12, 'value': 12},
                {'label': 13, 'value': 13},
                {'label': 14, 'value': 14},
                {'label': 15, 'value': 15},
                {'label': 16, 'value': 16},
                {'label': 17, 'value': 17},
                {'label': 18, 'value': 18},
                {'label': 19, 'value': 19},
                {'label': 21, 'value': 21},
                {'label': 23, 'value': 23},
            ],
            multi=False,  # Cambiado a False para permitir una sola selección
            value='6',  # Opción preseleccionada
            style={'display': 'inline-block','width': '50px'}  # Establecemos el estilo inline-block
        ),
        html.H6("Tuition fees up to date:", style={'display': 'inline-block', 'margin-right': '10px'}),  # Establecemos el estilo inline-block
        dcc.Dropdown(
            id='Q',  # Identificador único para la lista desplegable
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0},
            ],
            multi=False,  # Cambiado a False para permitir una sola selección
            value=1,  # Opción preseleccionada
            style={'display': 'inline-block'}  # Establecemos el estilo inline-block
        ),
        html.H6("Debtor:", style={'display': 'inline-block', 'margin-right': '10px'}),  # Establecemos el estilo inline-block
        dcc.Dropdown(
            id='P',  # Identificador único para la lista desplegable
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0},
            ],
            multi=False,  # Cambiado a False para permitir una sola selección
            value=0,  # Opción preseleccionada
            style={'display': 'inline-block'}  # Establecemos el estilo inline-block
        ),
        html.H6("Course:", style={'display': 'inline-block', 'margin-right': '10px'}),  # Establecemos el estilo inline-block
        dcc.Dropdown(
            id='D',  # Identificador único para la lista desplegable
            options=[
                {'label':'Biofuel Production Technologies','value':33},
                {'label':'Animation and Multimedia Design','value':171},
                {'label':'Social Service (evening attendance)','value':8014},
                {'label':'Agronomy','value':9003},
                {'label':'Communication Design','value':9070},
                {'label':'Veterinary Nursing','value':9085},
                {'label':'Informatics Engineering','value':9119},
                {'label':'Equinculture','value':9130},
                {'label':'Management','value':9147},
                {'label':'Social Service','value':9238},
                {'label':'Tourism','value':9254},
                {'label':'Nursing','value':9500},
                {'label':'Oral Hygiene','value':9556},
                {'label':'Advertising and Marketing Management','value':9670},
                {'label':'Journalism and Communication','value':9773},
                {'label':'Basic Education','value':9853},
                {'label':'Management (evening attendance)','value':9991},
            ],
            multi=False,  # Cambiado a False para permitir una sola selección
            value=8014,  # Opción preseleccionada
            style={'display': 'inline-block','width': '300px'}  # Establecemos el estilo inline-block
        ),
        html.H6("Curricular units 1st sem (grade):", style={'display': 'inline-block', 'margin-right': '10px'}),  # Establecemos el estilo inline-block
        dcc.Dropdown(
            id='Z',  # Identificador único para la lista desplegable
            options=[
                {'label':'less than 4','value':0},
                {'label':'Between 4 and 9','value':1},
                {'label':'Between 9 and 14','value':2},
                {'label':'More than 14','value':3},
            ],
            multi=False,  # Cambiado a False para permitir una sola selección
            value=2,  # Opción preseleccionada
            style={'display': 'inline-block','width': '200px'}  # Establecemos el estilo inline-block
        ),
        html.H6("Curricular units 2nd sem (grade):", style={'display': 'inline-block', 'margin-right': '10px'}),  # Establecemos el estilo inline-block
        dcc.Dropdown(
            id='AF',  # Identificador único para la lista desplegable
            options=[
                {'label':'less than 4','value':0},
                {'label':'Between 4 and 9','value':1},
                {'label':'Between 9 and 13','value':2},
                {'label':'More than 13','value':3},
            ],
            multi=False,  # Cambiado a False para permitir una sola selección
            value=2,  # Opción preseleccionada
            style={'display': 'inline-block','width': '200px'}  # Establecemos el estilo inline-block
        ),
        html.H6("Curricular units 1st sem (approved):", style={'display': 'inline-block', 'margin-right': '10px'}),  # Establecemos el estilo inline-block
        dcc.Dropdown(
            id='Y',  # Identificador único para la lista desplegable
            options=[
                {'label':'0','value':0},
                {'label':'1','value':1},
                {'label':'2','value':2},
                {'label':'3','value':3},
                {'label':'4','value':4},
                {'label':'5','value':5},
                {'label':'6','value':6},
                {'label':'7','value':7},
                {'label':'8','value':8},
                {'label':'9','value':9},
                {'label':'10','value':10},
                {'label':'11','value':11},
                {'label':'12','value':12},
                {'label':'13','value':13},
                {'label':'14','value':14},
                {'label':'15','value':15},
                {'label':'16','value':16},
                {'label':'17','value':17},
                {'label':'18','value':18},
                {'label':'19','value':19},
                {'label':'20','value':20},
                {'label':'21','value':21},
                {'label':'26','value':26},
            ],
            multi=False,  # Cambiado a False para permitir una sola selección
            value=6,  # Opción preseleccionada
            style={'display': 'inline-block'}  # Establecemos el estilo inline-block
        ),
        html.H6("Curricular units 2nd sem (approved):", style={'display': 'inline-block', 'margin-right': '10px'}),  # Establecemos el estilo inline-block
        dcc.Dropdown(
            id='AE',  # Identificador único para la lista desplegable
            options=[
                {'label':'0','value':0},
                {'label':'1','value':1},
                {'label':'2','value':2},
                {'label':'3','value':3},
                {'label':'4','value':4},
                {'label':'5','value':5},
                {'label':'6','value':6},
                {'label':'7','value':7},
                {'label':'8','value':8},
                {'label':'9','value':9},
                {'label':'10','value':10},
                {'label':'11','value':11},
                {'label':'12','value':12},
                {'label':'13','value':13},
                {'label':'14','value':14},
                {'label':'16','value':16},
                {'label':'17','value':17},
                {'label':'18','value':18},
                {'label':'19','value':19},
                {'label':'20','value':20},
            ],
            multi=False,  # Cambiado a False para permitir una sola selección
            value=5,  # Opción preseleccionada
            style={'display': 'inline-block'}  # Establecemos el estilo inline-block
        ),
        html.H6("Curricular units 1st sem (enrolled):", style={'display': 'inline-block', 'margin-right': '10px'}),  # Establecemos el estilo inline-block
        dcc.Dropdown(
            id='W',  # Identificador único para la lista desplegable
            options=[
                {'label':'0','value':0},
                {'label':'1','value':1},
                {'label':'2','value':2},
                {'label':'3','value':3},
                {'label':'4','value':4},
                {'label':'5','value':5},
                {'label':'6','value':6},
                {'label':'7','value':7},
                {'label':'8','value':8},
                {'label':'9','value':9},
                {'label':'10','value':10},
                {'label':'11','value':11},
                {'label':'12','value':12},
                {'label':'13','value':13},
                {'label':'14','value':14},
                {'label':'15','value':15},
                {'label':'16','value':16},
                {'label':'17','value':17},
                {'label':'18','value':18},
                {'label':'19','value':19},
                {'label':'21','value':21},
                {'label':'23','value':23},
                {'label':'26','value':26},
            ],
            multi=False,  # Cambiado a False para permitir una sola selección
            value=7,  # Opción preseleccionada
            style={'display': 'inline-block'}  # Establecemos el estilo inline-block
        ),
    ], className='panel'),
])
panel_3_content = html.Div([
    html.H2("Resultados"),
    html.P("This aplication is a tool that predicts the academic success of a student, based on personal, economic and academic atributes."),
])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
# Diseño de la aplicación con los paneles
app.layout = html.Div([
    html.H1(children='Academic success predictor'),  # Encabezado principal

    # Contenedor de Paneles
    html.Div([
        panel_1_content,  # Agrega aquí otros paneles si es necesario
        panel_2_content,
    ], className="panel-container"),
    # Agrega tus componentes aquí, como el botón y cualquier otra interfaz que desees
    html.Button('Realizar inferencia', id='inferencia-button'),
    # Agrega un espacio para mostrar el resultado
    html.Div([
        html.H2("Results"),
        html.P(id='resultado-inferencia'),
    ])
    
])
@app.callback(
    Output('resultado-inferencia', 'children'),
    [Input('inferencia-button', 'n_clicks')],
    [State('S', 'value'),
     State('T', 'value'),
     State('R', 'value'),
     State('G', 'value'),
     State('AC', 'value'),
     State('W', 'value'),
     State('AE', 'value'),
     State('Y', 'value'),
     State('AF', 'value'),
     State('Z', 'value'),
     State('D', 'value'),
     State('P', 'value'),
     State('Q', 'value')]
)
def realizar_inferencia(n_clicks, S, T, R, G, AC, W, AE, Y, AF, Z, D, P, Q):
    if n_clicks is None:
        return "Esperando a que se haga clic en el botón..."

    try:
        result = infer.query(
            variables=["Target"],
            evidence={"S": int(S), "T": int(T), "R": int(R), "G": int(G), "AC": int(AC), "W": int(W), "AE": int(AE), "Y": int(Y), "AF": int(AF), "Z": int(Z), "D": int(D), "P": int(P), "Q": int(Q)}
        )
        result_values = result.values
        max_prob_index = np.argmax(result.values)
        max_prob_option = modelo1.get_cpds('Target').state_names['Target'][max_prob_index]
        
        # Crear la gráfica de barras
        opciones = modelo1.get_cpds('Target').state_names['Target']
        probabilidades = result_values.tolist()
        
        # Creating the bar chart
        grafico = {
            'data': [{'x':opciones , 'y': probabilidades, 'type': 'bar'}],
            'layout': {'title': 'Probabilidad de cada opción', 'xaxis': {'title': 'Opciones'}, 'yaxis': {'title': 'Probabilidad'}}
        }
        
        return [
            html.Div(f"La opción más probable es: {opciones[np.argmax(probabilidades)]}"),
            dcc.Graph(figure=grafico),
        ]

    except Exception as e:
        return f"Error durante la inferencia: {str(e)}"

if __name__ == '__main__':
    app.run_server(debug=True)