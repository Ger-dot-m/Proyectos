import pandas as pd
class Similaridad():
    def __init__(self, data: pd.DataFrame, labels: tuple) -> None:
        """
        Inicializa una instancia de la clase Similaridad.
        
        Calcula elementos similares en un conjunto de datos basado en similitud de texto
        utilizando la medida de similitud del coseno y el modelo TF-IDF.

        Parámetros:
        - data (DataFrame): El DataFrame que contiene los datos para el cálculo de similitud.
        - labels (list): Una lista que contiene tres elementos en el siguiente orden:
            1. ID: La columna de identificación de elementos.
            2. Target: La columna que contiene los nombres de los elementos a comparar.
            3. Scores: La columna que contiene los puntajes de similitud entre elementos.
        """
        pd.options.mode.chained_assignment = None 
        self.DataFrame = data
        self.labels = labels  # lista: id, target, scores
        self.cosine_similarity = None
    
    def train(self, pdv: str):
        """
        Entrena el modelo de similitud utilizando TF-IDF y similitud coseno.

        Parámetros:
        - pdv (str): El punto de venta (PDV) para el cual se entrena el modelo.
        """
        self.pdv = pdv
        import numpy as np
        np.int = int
        np.float = float
        from sklearn.metrics.pairwise import linear_kernel
        from sklearn.feature_extraction.text import TfidfVectorizer
        _vectorizer = TfidfVectorizer()
        try:
            data = self.DataFrame.loc[self.pdv][self.labels[1]]
            _tfidf = _vectorizer.fit_transform(data)
            self.cosine_similarity = linear_kernel(_tfidf)
            df_temp = self.DataFrame.loc[self.pdv]
            df_temp.reset_index(inplace=True)
            self.items = df_temp
            self.names = self.items[self.labels[1]]
        except Exception as e:
            self.cosine_similarity = None
            print(f'Error en entrenamiento de modelo: {e}.')
    
    def similarity(self, title: str, number: int, by='default'):
        """
        Calcula y devuelve elementos similares a uno dado.

        Parámetros:
        - title (str): El nombre del elemento a comparar.
        - number (int): El número de elementos similares a mostrar.
        - by (str): Opciones para seleccionar la columna de scores:
            - "default": Muestra solo el puntaje de similitud.
            - "target": Muestra el producto de la columna scores y target.

        Retorna:
        - list: Una lista de elementos similares con sus nombres, puntajes y objetivos (si aplicable).
        """
        if self.cosine_similarity is None:
            print('No se ha entrenado el modelo.')
            return 0
        if title not in self.names.values:
            print(f'{title} no se encuentra en los datos.')
            return 0
        self.title = title
        producto_id = self.items[self.items[self.labels[1]] == title].index.values[0]
        try:
            scores = list(enumerate(self.cosine_similarity[producto_id]))
        except Exception as e:
            print(e)
            return 0
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        sorted_scores = sorted_scores[1:]
        if number >= len(sorted_scores):
            number = len(sorted_scores)
        data = list()
        for item in sorted_scores[:number]:
            name = self.items[self.items.index == item[0]][self.labels[1]].values[0]
            if by == 'default':
                target = self.items[self.items.index == item[0]][self.labels[2]].values[0]
            elif by == 'target':
                target = self.items[self.items.index == item[0]][self.labels[2]].values[0]
                target *= item[1]
            data.append({
                'name': name,
                'score': item[1],
                'target': round(target, 2)
            })
        self.output = data
        return data

    def show(self):
        """
        Muestra los elementos similares en la salida estándar.
        """
        if self.cosine_similarity is None:
            print("No se ha entrenado el modelo.")
            return 0
        print(f'Similares a {self.title} en {self.pdv}.')
        for x in self.output:
            a, b, c = x['name'], x['target'], x['score']
            print(f'{a}, target: {b}, score: {c}')
    
    def exportar(self, name='datos', ascendente=False):
        """
        Exporta los elementos similares a un archivo CSV.

        Parámetros:
        - name (str): El nombre del archivo a exportar.
        - ascendente (bool): El orden de los datos en el archivo CSV.
        """
        try:
            self.pd.DataFrame(self.output
                ).sort_values('target', ascending=ascendente
                ).to_csv(f'{name}.csv', index=False)
        except Exception as e:
            print(f'Error en la exportación: {e}')
