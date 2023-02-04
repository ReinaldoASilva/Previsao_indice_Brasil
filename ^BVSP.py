import yfinance as yf 
import pandas as pd 
from pandas_profiling import ProfileReport
import plotly_express as px 
from pycaret.regression import *

# Fechamento
BVSP = yf.Ticker('^BVSP')
BVSP = BVSP.history(start='2000-01-01', end='2023-02-03')
#BVSP = BVSP.history(period='20y')
BVSP = pd.DataFrame(BVSP)
BVSP = BVSP.drop(['Dividends', 'Stock Splits'],axis=1)

BVSP['MM20d'] = BVSP['Close'].rolling(window=20).mean().round(2)
BVSP['MM200d']= BVSP['Close'].rolling(window=200).mean().round(2)

profile = ProfileReport(BVSP, title='Previsão do preço de fechamento do BVSP', html={'style':{'full_width':True}})
profile.to_notebook_iframe()
profile.to_file(output_file='Relatório- Previsão BVSP.html')

BVSP_prever = BVSP.tail(365)

BVSP.drop(BVSP.tail(365).index, inplace=True)

BVSP['Close'] = BVSP['Close'].shift(-1)

BVSP.dropna(inplace=True)

BVSP.reset_index(drop=True, inplace=True)
BVSP_prever.reset_index(drop=True, inplace=True)

setup(data=BVSP, target='Close', session_id=123, remove_perfect_collinearity=False)

top3 = compare_models(n_select=3)

lar = create_model('lar', fold=10)
ridge = create_model('ridge', fold=10)

ridge_params = {'alpha': [0.02, 0.024, 0.025, 0.026, 0.03]}
tune_ridge = tune_model(ridge, n_iter=1000, optimize = 'RMSE', custom_grid=ridge_params)
tune_lar = tune_model(lar, n_iter=1000, optimize = 'RMSE')

plot_model(tune_ridge, plot='error')
plot_model(tune_ridge, plot='feature')

predict_model(tune_ridge)

final_tune_ridge = finalize_model(tune_ridge)

prev_close = predict_model(final_tune_ridge, data=BVSP_prever)

fig = px.line(round(prev[['Close','Label']],2),
                x = round(prev[['Close', 'Label']],2).index,
                y = ['Close', 'Label'],
                title = 'Preço fechamento x preço previsto de BVSP ',
                width = 1500, height = 1000)
fig.show()

# Maxima

BVSP = yf.Ticker('^BVSP')
BVSP = BVSP.history(start='2000-01-01', end='2023-02-03')
#BVSP = BVSP.history(period='20y')
BVSP = pd.DataFrame(BVSP)
BVSP = BVSP.drop(['Dividends', 'Stock Splits'],axis=1)

BVSP['MM20d'] = BVSP['Close'].rolling(window=20).mean().round(2)
BVSP['MM200d']= BVSP['Close'].rolling(window=200).mean().round(2)

profile = ProfileReport(BVSP, title='Previsão do preço de fechamento do BVSP', html={'style':{'full_width':True}})
profile.to_notebook_iframe()
profile.to_file(output_file='Relatório- Previsão BVSP.html')

BVSP_prever = BVSP.tail(365)

BVSP.drop(BVSP.tail(365).index, inplace=True)

BVSP['High'] = BVSP['High'].shift(-1)

BVSP.dropna(inplace=True)

BVSP.reset_index(drop=True, inplace=True)
BVSP_prever.reset_index(drop=True, inplace=True)

setup(data=BVSP, target='High', session_id=123, remove_perfect_collinearity=False)

top3 = compare_models(n_select=3)

lar = create_model('lar', fold=10)
ridge = create_model('ridge', fold=10)

ridge_params = {'alpha': [0.02, 0.024, 0.025, 0.026, 0.03]}
tune_ridge = tune_model(ridge, n_iter=1000, optimize = 'RMSE', custom_grid=ridge_params)
tune_lar = tune_model(lar, n_iter=1000, optimize = 'RMSE')

plot_model(tune_ridge, plot='error')
plot_model(tune_ridge, plot='feature')

predict_model(tune_ridge)

final_tune_ridge = finalize_model(tune_ridge)

prev_max = predict_model(final_tune_ridge, data=BVSP_prever)

fig = px.line(round(prev_max[['Close','Label']],2),
                x = round(prev[['Close', 'Label']],2).index,
                y = ['Close', 'Label'],
                title = 'Preço fechamento x preço previsto de BVSP ',
                width = 1500, height = 1000)
fig.show()

# Minima


BVSP = yf.Ticker('^BVSP')
#BVSP = BVSP.history(start='2000-01-01', end='2023-02-02')
BVSP = BVSP.history(period='20y')
BVSP = pd.DataFrame(BVSP)
BVSP = BVSP.drop(['Dividends', 'Stock Splits'],axis=1)

BVSP['MM7d'] = BVSP['Low'].rolling(window=7).mean().round(2)
BVSP['MM30d']= BVSP['Low'].rolling(window=7).mean().round(2)

profile = ProfileReport(BVSP, title='Previsão do preço de fechamento do BVSP', html={'style':{'full_width':True}})
profile.to_notebook_iframe()
profile.to_file(output_file='Relatório- Previsão BVSP.html')

BVSP_prever = BVSP.tail(365)

BVSP.drop(BVSP.tail(365).index, inplace=True)

BVSP['Low'] = BVSP['Low'].shift(-1)

BVSP.dropna(inplace=True)

BVSP.reset_index(drop=True, inplace=True)
BVSP_prever.reset_index(drop=True, inplace=True)

setup(data=BVSP, target='Low', session_id=123, remove_perfect_collinearity=False)

top3 = compare_models(n_select=3)

lar = create_model('lar', fold=10)
ridge = create_model('ridge', fold=10)

ridge_params = {'alpha': [0.02, 0.024, 0.025, 0.026, 0.03]}
tune_ridge = tune_model(ridge, n_iter=1000, optimize = 'RMSE', custom_grid=ridge_params)
tune_lar = tune_model(lar, n_iter=1000, optimize = 'RMSE')

plot_model(tune_ridge, plot='error')
plot_model(tune_ridge, plot='feature')

predict_model(tune_ridge)

final_tune_ridge = finalize_model(tune_ridge)

prev_min = predict_model(final_tune_ridge, data=BVSP_prever)

fig = px.line(round(prev[['Close','Label']],2),
                x = round(prev[['Close', 'Label']],2).index,
                y = ['Close', 'Label'],
                title = 'Preço fechamento x preço previsto de BVSP ',
                width = 1500, height = 1000)
fig.show()

# Abertura

BVSP = yf.Ticker('^BVSP')
#BVSP = BVSP.history(start='2000-01-01', end='2023-02-02')
BVSP = BVSP.history(period='20y')
BVSP = pd.DataFrame(BVSP)
BVSP = BVSP.drop(['Dividends', 'Stock Splits'],axis=1)

BVSP['MM7d'] = BVSP['Open'].rolling(window=7).mean().round(2)
BVSP['MM30d']= BVSP['Open'].rolling(window=7).mean().round(2)

profile = ProfileReport(BVSP, title='Previsão do preço de fechamento do BVSP', html={'style':{'full_width':True}})
profile.to_notebook_iframe()
profile.to_file(output_file='Relatório- Previsão BVSP.html')

BVSP_prever = BVSP.tail(365)

BVSP.drop(BVSP.tail(365).index, inplace=True)

BVSP['Open'] = BVSP['Open'].shift(-1)

BVSP.dropna(inplace=True)

BVSP.reset_index(drop=True, inplace=True)
BVSP_prever.reset_index(drop=True, inplace=True)

setup(data=BVSP, target='Open', session_id=123, remove_perfect_collinearity=False)

top3 = compare_models(n_select=3)

lar = create_model('lar', fold=10)
ridge = create_model('ridge', fold=10)

ridge_params = {'alpha': [0.02, 0.024, 0.025, 0.026, 0.03]}
tune_ridge = tune_model(ridge, n_iter=1000, optimize = 'RMSE', custom_grid=ridge_params)
tune_lar = tune_model(lar, n_iter=1000, optimize = 'RMSE')

plot_model(tune_ridge, plot='error')
plot_model(tune_ridge, plot='feature')

predict_model(tune_ridge)

final_tune_ridge = finalize_model(tune_ridge)

prev_open = predict_model(final_tune_ridge, data=BVSP_prever)

fig = px.line(round(prev_max[['Close','Label']],2),
                x = round(prev[['Close', 'Label']],2).index,
                y = ['Close', 'Label'],
                title = 'Preço fechamento x preço previsto de BVSP ',
                width = 1500, height = 1000)
fig.show()

