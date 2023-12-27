import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
# import cdata.amazons3 as mod

import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

main_df = pd.read_csv('history_final.csv')

main_df['day_of_week'] = pd.to_datetime(main_df['Date']).dt.dayofweek
main_df['month'] = pd.to_datetime(main_df['Date']).dt.month
main_df['Sector'] = main_df['Sector'].map(
    {'Industrials': 0, 'Health Care': 1, 'Information Technology': 2, 'Consumer Discretionary': 3, 'Financials': 4,
     'Materials': 5, 'Real Estate': 6, 'Consumer Staples': 7, 'Energy': 8, 'Utilities': 9,
     'Telecommunication Services': 10})
encoded_df = pd.read_csv('encoded_df.csv')

y = main_df['Label']

names = main_df['Symbol'].unique()

X_train, X_test, y_train, y_test = train_test_split(encoded_df, y, test_size=0.2, random_state=123)
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, y_train)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Next Day Stock Movement Prediction", className='text-center text-primary'))
    ]),
    dbc.Row([
        dbc.Col([dcc.Dropdown(id='slct-symbol',
                              options=names,
                              multi=False,
                              value='GOOG',
                              style={'width': '40%'}
                              ),
                 html.Div(id='output_container', children=[]),
                 html.Br(),
                 dcc.Graph(id='stock', figure={})
                 ])
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(id='ml1', children=[]),
            #dcc.Graph(id='ml1'),
            html.Br(),
            html.Br()
        ]),
        dbc.Col([
            #dcc.Graph(id='roc'),
            html.Br()
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.Img(src='assets/img_2.png', className='mx-auto d-block', style={'width': '80%'}),
            html.Br()
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.Img(src='assets/img_4.png', className='mx-auto d-block', style={'width': '80%', 'height': '80%'}),
            html.Br()
        ])
    ])
], fluid=False)


# app.run(jupyter_mode="external")


# server = app.server

@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='stock', component_property='figure')],
    [Input(component_id='slct-symbol', component_property='value')]
)
def update_graph(option_slctd):
    print(option_slctd)
    print(type(option_slctd))

    container = "Stock chosen is: {}".format(option_slctd)

    dff = main_df.copy()
    dff = dff[dff['Symbol'] == option_slctd]

    dff['diff'] = dff['Close'] - dff['Open']
    dff.loc[dff['diff'] >= 0, 'color'] = 'green'
    dff.loc[dff['diff'] < 0, 'color'] = 'red'

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(x=dff['Date'],
                                 open=dff['Open'],
                                 high=dff['High'],
                                 low=dff['Low'],
                                 close=dff['Close'],
                                 name='Price'))

    fig.update_layout(xaxis_rangeslider_visible=True)  # show range slider
    fig.update_layout(title={'text': option_slctd, 'x': 0.5})
    fig.update_xaxes(rangebreaks=[
        dict(bounds=['sat', 'mon']),  # hide weekends
    ])
    fig.layout.template = 'ggplot2'

    return container, fig


# @app.callback(
#     [Output(component_id='ml1', component_property='figure'),
#      Output(component_id='roc', component_property='figure')],
#     [Input(component_id='slct-symbol', component_property='value')]
# )
@app.callback(
    Output(component_id='ml1', component_property='children'),
    [Input(component_id='slct-symbol', component_property='value')]
)
def update_figure(option_slctd):
    print(option_slctd)
    print(type(option_slctd))

    # container = "Model chosen is: {}".format(option_slctd)

    test_df = main_df[main_df['Symbol'] == option_slctd]
    test_df = test_df.sort_values('Date').groupby('Symbol').tail(1)
    testy_df = test_df.drop(
        ['Label', 'Date', 'Open', 'Close', 'CloseDiff', 'Symbol', 'VolumeScaledNormalized', 'Low', 'High'], axis=1)

    tester = test_df['Label']

    pred = xgb_model.predict(testy_df)
    print(pred)
    print(type(pred))

    result = int(pred[0])
    print(result)
    print(type(result))

    result = "The model predicts todays stock will move (0 for down, 1 for up) : {}".format(result)

    # conf_matrix = confusion_matrix(tester, pred)
    #
    # labels = ['Down', 'Up']
    # fig = ff.create_annotated_heatmap(
    #     z=conf_matrix,
    #     x=labels,
    #     y=labels,
    #     colorscale='Blues',
    # )
    # fig.update_layout(
    #     title='Confusion Matrix - Stock Price Movement Prediction',
    #     xaxis=dict(title='Predicted', side='bottom'),
    #     yaxis=dict(title='Actual'),
    # )
    #
    # fpr, tpr, _ = roc_curve(tester, pred)
    #
    # fig1 = go.Figure()
    #
    # fig1.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='blue', width=2), name='ROC Curve'))
    #
    # fig1.add_shape(type='line', line=dict(color='gray', width=1, dash='dash'),
    #                x0=0, y0=0, x1=1, y1=1)
    #
    # fig1.update_layout(
    #     xaxis_title='False Positive Rate',
    #     yaxis_title='True Positive Rate',
    #     title='ROC Curve',
    #     showlegend=True,
    #     xaxis=dict(range=[0, 1]),
    #     yaxis=dict(range=[0, 1]),
    #     width=600,
    #     height=400,
    #     margin=dict(l=50, r=50, t=50, b=50),
    # )
    #
    # return fig, fig1
    return result

if __name__ == '__main__':
    app.run_server(debug=True)
# %%
