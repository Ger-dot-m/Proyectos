from sklearn.pipeline import Pipeline
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np

class NormalizeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return self.__normaliza__(X)
    
    def __normaliza__(self, X) -> None:
        X_serie = X['Importe']
        X_serie.index = X.iloc[:, 0]
        X = X_serie
        parametro = np.median(X) 
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        aux = (X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))
        X[aux] = parametro
        return X

class ResampleTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        try:
            return X.resample('2W').sum()
        except: print(X.resample('2W').sum())

class Forecast():
    def __init__(self, order, seasonal_order) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.data = None

    def fit(self, data, extra):
        self.data = data
        if self.seasonal_order[-1] > 1:
            try:
                model = SARIMAX(endog = self.data,
                                        order=self.order,
                                        seasonal_order=self.seasonal_order,
                                        simple_differencing=False)
                self.model = model.fit(disp=False)
            except Exception as e:
                print(f'Error en modelo SARIMAX: {e}')
                print(self.order)
                print(self.seasonal_order)
                return 0
        elif self.seasonal_order[-1] == 1:
            self.data -= seasonal_decompose(self.data).seasonal
            try:
                self.data -= seasonal_decompose(self.data).seasonal
                model = ARIMA(endog = self.data, order=self.order)
                self.model = model.fit()
            except Exception as e:
                print(f'Error en el modelo: {e}')
                return 0

        elif self.seasonal_order[-1] == 0:
            try:
                self.data -= seasonal_decompose(self.data).seasonal
                model = ARIMA(endog = self.data, order=self.order)
                self.model = model.fit()
            except Exception as e:
                print(f'Error en el modelo: {e}')
                return 0
    
    def predict(self, X, steps):
        self.forecast = self.model.get_forecast(steps)
        pred = self.forecast.predicted_mean
        return pred

class Proceso(Pipeline):
    def predict(self, X, steps=2):
        return super().predict(X, steps=steps)