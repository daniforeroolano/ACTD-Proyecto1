import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv('C:/Users/oem/Documents/universidad de los andes/octavo/Analitica computacional para la toma de decisiones/Proyecto 1/data.csv', sep=';')

print("columns")
print(df.columns) 
print("size")
print(df.size) 
print("info")
print(df.info())
print("shape")
print(df.shape)
print("describe")
print(df.describe)
print("nulos")
print(df.isnull().sum())
print("unicos")
for columna in df.columns:
    print(f'Valores unicos en la columna: {columna}: ')
    print(df[columna].unique())
    print('\n')

    
print("Grafico del target")
# Calcular la frecuencia de cada valor en la columna "target"
frecuencia_target = df['Target'].value_counts()

# Crear el gráfico de frecuencia
plt.bar(frecuencia_target.index, frecuencia_target.values)

# Personalizar el gráfico
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Frecuencia de la columna "target"')
plt.xticks(rotation=45)  # Rotar las etiquetas del eje x para mayor legibilidad

# Mostrar el gráfico
plt.show()