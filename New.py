
'''
This is the main function of Portfolio Optimizer

Created by YishiLiu and edited by Chaoyi Ye and YishiLiu

'''
import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from plotly.tools import mpl_to_plotly
import pylab as plt
import pandas as pd
import plotly.graph_objs as go
import dash_bootstrap_components as dbc

import DataPreparation as DP
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.value_at_risk import CVAROpt
from pypfopt.hierarchical_risk_parity import HRPOpt
from pandas import DataFrame
from Optimizer import Optimization, All_frontier,Min_Max_drawdown,Max_Sortino_Ratio,Max_Omega_Ratio,coskew,cokurt,tailriskparity,co_drawdown,Infor_Ratio,metrics
from Resampled import Resampling_EF
import BL_optimizer as BLO
import DataPreparation_BL as DPB
from numpy import  dot, transpose
from numpy.linalg import inv
import pylab
import math
import pandas_datareader as pdr

external_stylesheets = ["https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"]

portfolio=pd.read_csv('https://raw.githubusercontent.com/peaceLYS/Portfolio-Optimizer/master/portfolio.csv',encoding='utf-8')

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Portfolio Optimizer'
app.config['suppress_callback_exceptions'] = True



# Dropdown choises specifying
months = ['1','2','3','4','5','6','7','8','9','10','11','12']

Optimization_Goal = ['1.Maximize Sharpe Ratio','2.Minimize Variance','3.Minimize Conditional Value-at-Risk',
'4.Risk Parity','5.Maximize Information Ratio','6.Minimize Volatility subject to...',
'7.Maximize Return subject to...','8.Minimize Maximum Drawdown subject to...',
'9.Maximize Omega Ratio subject to...','10.Maximize Sortino Ratio subject to...',
'11.Tail Risk Parity','12.Co-drawdown','13.Co-skew','14.Co-kurtosis']

Compared_Allocation = ['None','Equal Weighted','Maximum Sharpe Ratio Weights',
'Inverse Volatility Weighted','Risk Parity Weighted']

#Number_Of_Assets = list(range(1,200))# is 200 enough?

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

page_1_layout=html.Div([
    html.Div([

        dbc.NavbarSimple(
                children=[
                    dbc.NavItem(dbc.NavLink("Home", href="#")),
                    dbc.DropdownMenu(
                        children=[

                            dbc.DropdownMenuItem("FAQ", header=True),
                            #dcc.Location(id='url', refresh=False),
                            dcc.Link("Black-Litterman", href="/page-Black-Litterman"),
                            dbc.DropdownMenuItem("Example", href="#"),
                        ],
                        nav=True,
                        in_navbar=True,
                        label="More",
                    ),
                ],
                brand="Portfolio Optimization",
                brand_href="#",
                color="#A3D1BE",
                dark=True,
               style={'width':'100%'},
            )],className='container-fluid',style={'max-weight':'50000','padding-left':'0px','padding-right':'0px'}),
    dbc.Jumbotron(
    [
        html.H1("Portfolio Optimization", className="display-3"),
        html.P("This portfolio optimizer tool supports the following portfolio optimization strategies: ",
            className="lead",
        ),
        html.Hr(className="my-2"),
        html.P( "Mean Variance Optimization - Find the optimal risk adjusted portfolio that lies on the efficient frontier "),
        html.P( "Minimize Conditional Value-at-Risk - Optimize the portfolio to minimize the expected tail loss "),
        html.P( "Maximize Information Ratio - Find the portfolio that maximizes the information ratio against the selected benchmark "),
        html.P( "Risk Parity - Find the portfolio that equalizes the risk contribution of portfolio assets "),
        html.P( "Maximize Sortino Ratio - Find the portfolio that maximizes the Sortino ratio for the given minimum acceptable return "),
        html.P( "Maximize Omega Ratio - Find the portfolio that maximizes the Omega ratio for the given minimum acceptable return "),
        html.P( "Minimize Maximum Drawdown - Find the portfolio with the minimum worst case drawdown with optional minimum acceptable return"),
        html.Hr(className="my-2"),
        html.P("The optimization is based on the monthly return statistics of the selected portfolio assets for"
              "the given time period. The optimization result does not predict what allocation would perform"
              "best outside the given time period, and the actual performance of portfolios constructed"
              "using the optimized asset weights may vary from the given performance goal. "),
        html.P("The required inputs for the optimization include the time range and the portfolio assets."
              "Portfolio asset weights and constraints are optional. You can also use the Black-Litterman"
              "model based portfolio optimization, which allows the benchmark portfolio asset weights to be optimized based on investor's views. "),
    ]
),
    html.Div([
        html.Div([
            html.Strong('Portfolio Type'),
            html.I(id='type-button', n_clicks=0, className='fa fa-info-circle'),
            dbc.Tooltip(
            "Select portfolio type",
            target="type-button",
        ),
        ],className='col-3',style={ 'margin': '5px 5px 5px'}),

        html.Div([
            dcc.Dropdown(
                options=[
            {'label': 'Tickers', 'value': 'TC'},
            {'label': 'Asset Class', 'value': 'AC'},
        ],
        value='TC',style={'backgroundColor':'#EDF0EE'}
    ),
        ],className='col-6',style={ 'margin': '5px 5px 5px'}),
        html.Div([],className='col-3')


    ],className='row',style={ 'margin-left': '10px'}),

    html.Div([
            html.Div([
                html.Strong('Time Period'),
                html.I(id='time-button', n_clicks=0, className='fa fa-info-circle'),
                dbc.Tooltip(
            "Specify time period in years or in months",
            target="time-button",
        ),
            ],className='col-3',style={ 'margin': '5px 5px 5px'}),
            html.Div(
                [dcc.Dropdown(id='Period',
                     options=[
            {'label': 'Month-to-Month', 'value': 'MTM'},
            {'label': 'Year-to-Year', 'value': 'YTY'}
        ],
            value='YTY',style={'backgroundColor':'#EDF0EE'}
        ),
            ],className='col-6',style={ 'margin': '5px 5px 5px'}),
            html.Div([],className='col-3'),


        ],className='row',style={ 'margin-left': '10px'}),
    html.Div([
                html.Div([
                    html.Strong('Start Year'),
                    html.I(id='start-button', n_clicks=0, className='fa fa-info-circle'),
                    dbc.Tooltip(
            "Start year for portfolio optimization period",
            target="start-button",
        ),
                ],className='col-3',style={ 'margin': '5px 5px 5px'}),
                html.Div([
                    dcc.Dropdown(id='beginY',
                         options=[
            {'label': i+1985, 'value': i+1985}
            for i in range(datetime.datetime.now().year-1985)
        ],
        value=1985,style={'backgroundColor':'#EDF0EE'}
            ),
                ],className='col-6',style={ 'margin': '5px 5px 5px'}),
                html.Div([],className='col-3')
            ],className='row',style={ 'margin-left': '10px'}),
   html.Div([
    html.Div([
        html.Strong('Start Month'),
                html.I(id='start-month', n_clicks=0, className='fa fa-info-circle'),
                dbc.Tooltip(
            "Specify start period in years or in months",
            target="start-month",
        ),
    ],className='col-3',style={'margin': '5px 5px 5px'}),
    html.Div([
        dcc.Dropdown(id = 'StartM',options=[{'label': i, 'value' : i} for i in range(1,13)],value=1,style={'backgroundColor':'#EDF0EE'})
    ],className='col-6',style={'margin': '5px 5px 5px'}),
    html.Div([],className='col-3',style={'margin': '5px 5px 5px'}),
],className='row',style={ 'margin-left': '10px'})
,
    html.Div([
                html.Div([
                    html.Strong('End Year'),
                    html.I(id='end-button', n_clicks=0, className='fa fa-info-circle'),
                    dbc.Tooltip(
            "End year for portfolio optimization period",
            target="end-button",
        ),
                ],className='col-3',style={ 'margin': '5px 5px 5px'}),
                html.Div([
                    dcc.Dropdown(id='endY',
                         options=[
            {'label': i+1985, 'value': i+1985}
            for i in range(datetime.datetime.now().year-1984)
        ],
        value=2019,style={'backgroundColor':'#EDF0EE'}
            ),
                ],className='col-6',style={ 'margin': '5px 5px 5px'}),
                html.Div([],className='col-3')
            ],className='row',style={ 'margin-left': '10px'}),
    html.Div([
    html.Div([
        html.Strong('End Month'),
                html.I(id='end-month', n_clicks=0, className='fa fa-info-circle'),
                dbc.Tooltip(
            "Specify start period in years or in months",
            target="end-month",
        ),
    ],className='col-3',style={'margin': '5px 5px 5px'}),
    html.Div([
        dcc.Dropdown(id='EndM',options=[{'label': i, 'value' : i} for i in range(1,13)],value=1,style={'backgroundColor':'#EDF0EE'})
    ],className='col-6',style={'margin': '5px 5px 5px'}),
    html.Div([],className='col-3',style={'margin': '5px 5px 5px'}),
],className='row',style={ 'margin-left': '10px'}),
    html.Div([
                html.Div([
                    html.Strong('Optimization Goal'),
                    html.I(id='optimization-button', n_clicks=0, className='fa fa-info-circle'),
                    dbc.Tooltip(
            "Select performance goal for optimization",
            target="optimization-button",
        ),
                ],className='col-3',style={ 'margin': '5px 5px 5px'}),
                html.Div([
                    dcc.Dropdown(id='opt_goal',
                         options=[{'label': i, 'value': i} for i in Optimization_Goal],value='1.Maximize Sharpe Ratio',style={'backgroundColor':'#EDF0EE'}
            ),
                ],className='col-6',style={ 'margin': '5px 5px 5px'}),
                html.Div([],className='col-3')
            ],className='row',style={ 'margin-left': '10px'}),
    html.Div([
    html.Div([html.Strong('Target Annual Return'),
                html.I(id='target-ann', n_clicks=0, className='fa fa-info-circle'),
                dbc.Tooltip(
            "Target Annual Return",
            target="target-ann",
        ),],className='col-3',style={'margin': '5px 5px 5px'}),
    html.Div([
        dbc.InputGroup(
            [   dbc.Input(id='target-ret-input',value="0.00",style={'z-index':'0','position':'relative'}),
                dbc.InputGroupAddon("%", addon_type="append"),])
    ],className='col-6',style={'margin': '5px 5px 5px'}),
    html.Div([],className='col-3',style={'margin': '5px 5px 5px'}),

],className='row',style={ 'margin-left': '10px'}),
    html.Div([
                html.Div([
                    html.Strong(['Use Historical Volitility']),
                    html.I(id='vol-button', n_clicks=0, className='fa fa-info-circle'),
                    dbc.Tooltip(
            "Use historical volitility or specify expected volitility of assets",
            target="vol-button",
        ),
                ],className='col-3',style={ 'margin': '5px 5px 5px'}),
                html.Div([
                    dcc.Dropdown(id='dropdown-vol',
                         options=[
                {'label': 'Yes', 'value': 'Y'},
                {'label': 'No', 'value': 'NO'}
            ],
                value='Y',style={'backgroundColor':'#EDF0EE'}
            ),
                ],className='col-6',style={ 'margin': '5px 5px 5px','data-container':"body"}),
                html.Div([],className='col-3')


            ],className='row',style={ 'margin-left': '10px'}),
html.Div([
    html.Div([html.Strong('Target Annual Volatility'),
                html.I(id='target-ann-vol', n_clicks=0, className='fa fa-info-circle'),
                dbc.Tooltip(
            "Target Annual Return",
            target="target-ann-vol",
        ),],className='col-3',style={'margin': '5px 5px 5px'}),
    html.Div([
        dbc.InputGroup(
            [   dbc.Input(id='target-vol-input',value="0.00",style={'z-index':'0','position':'relative'}),
                dbc.InputGroupAddon("%", addon_type="append"),])
    ],className='col-6',style={'margin': '5px 5px 5px'}),
    html.Div([],className='col-3',style={'margin': '5px 5px 5px'}),
    
],className='row',style={ 'margin-left': '10px'}),
    html.Div([
            html.Div([
                html.Strong('Asset Constraint'),
                html.I(id='asset-button', n_clicks=0, className='fa fa-info-circle'),
                dbc.Tooltip(
            "Specify allocation constraint for assets",
            target="asset-button",
        ),
            ],className='col-3',style={ 'margin': '5px 5px 5px'}),
            html.Div([
                dcc.Dropdown(id='dropdown-ass',
                     options=[
            {'label': 'Yes', 'value': 'Y'},
            {'label': 'No', 'value': 'NO'}
        ],
            value='Y',style={'backgroundColor':'#EDF0EE'}
        ),
            ],className='col-6',style={ 'margin': '5px 5px 5px'}),
            html.Div([],className='col-3')


        ],className='row',style={ 'margin-left': '10px'}),
     html.Div([
                html.Div([
                    html.Strong('Group Constraint'),
                    html.I(id='group-button', n_clicks=0, className='fa fa-info-circle'),
            dbc.Tooltip(
            "Specify asset class level allocation constraints",
            target="group-button",
        ),
                ],className='col-3',style={ 'margin': '5px 5px 5px'}),
                html.Div([
                    dcc.Dropdown(id='dropdown-gro',
                         options=[
                {'label': 'Yes', 'value': 'Y'},
                {'label': 'No', 'value': 'NO'}
            ],
                value='NO',style={'backgroundColor':'#EDF0EE'}
            ),
                ],className='col-6',style={ 'margin': '5px 5px 5px'}),
                html.Div([],className='col-3')


            ],className='row',style={ 'margin-left': '10px'}),
    html.Div([
                html.Div([
                    html.Strong('Compared Allocation'),
                    html.I(id='compare-button', n_clicks=0, className='fa fa-info-circle'),
                    dbc.Tooltip(
            "Select allocation for comparison",
            target="compare-button",
        ),
                ],className='col-3',style={ 'margin': '5px 5px 5px'}),
                html.Div([
                    dcc.Dropdown(
                         options=[{'label': i, 'value': i} for i in Compared_Allocation ],
        value='None',style={'backgroundColor':'#EDF0EE'}
            ),
                ],className='col-6',style={ 'margin': '5px 5px 5px'}),
                html.Div([],className='col-3')


            ],className='row',style={ 'margin-left': '10px'}),
    html.Div([
    html.Div([html.Strong('Benchmark Ticker'),
                html.I(id='ben-tic', n_clicks=0, className='fa fa-info-circle'),
                dbc.Tooltip(
            "Ticker symbol for the benchmark asset",
            target="ben-tic",
        ),],className='col-3',style={'margin': '5px 5px 5px'}),
    html.Div([
            dbc.InputGroup([
            dbc.Input(id="benchmark-t", value='None'),
            dbc.InputGroupAddon(
                    dbc.Button([html.Span(className='fa fa-search')], id="input-benchmark-t",color='light',style={
'border-Radius':"0px"}),
                    addon_type="prepend" ),
            dbc.Modal([
            dbc.ModalHeader("Find ETF,Mutual Funds or Stock Symbol"),
            dbc.ModalBody([
                html.Div([
                    html.Div([
                        html.Strong(['Type'],className='col-4'),
                        dcc.Dropdown(id='dropdown-menu-bench',
                         options=[
                {'label': 'ETF', 'value': 'ETF'},
                {'label': 'Mutual Fund', 'value': 'MU_F'},
                {'label': 'Stock', 'value': 'stock'},
                {'label': 'Asset', 'value': 'Asset'},
                {'label': 'Cash', 'value': 'Cash'}
            ],
                value='ETF',className='col-md-8',style={'min-width':'200px'}
            ),
                        ],className='row'),
                    html.Div([
                            html.Strong(['Name'],className='col-md-4'),
                            dbc.Input(id='Name-benchmark',placeholder='Enter partial name(3 or more characters)',className='col-md-8')
                        ],className='row'),
                    ])
                ]),
             dbc.ModalFooter(
                    children=[
                    dbc.Button("Close", id="close-benchmark", className="ml-auto"),
                    dbc.Button("Select",id="select-benchmark")]
                ),
            ],id='modal-benchmark'),
    ])
    ],className='col-6',style={'margin': '5px 5px 5px'}),
    html.Div([],className='col-3',style={'margin': '5px 5px 5px'}),

],className='row',style={ 'margin-left': '10px','z-index':'0','position':'relative'}),
        html.Label('Portfolio Assets'),
        #give an example
        html.Div(html.A('(Download example format)', download='test.csv', href='/test.csv'),
                        style={'fontSize':'15px','float':'right'}),
    html.Div([
        html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Your Portfolio Files')
            ]),
            style={
                'width': '80%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        )],className='col-md-9'),
        html.Div([html.Button(id='submit_button', n_clicks=0, children='Submit',
        style={'background-color': '#045FB4','color':'white','float': 'right'},)],className='col-md-3',style={'margin-up':'20px'}
        )
        ],className='row'),
        html.Hr(),
        html.Div(id='output-data-upload'),
html.Div([
        html.H1(["Portfolio Optimization Result"]),
        dbc.ButtonGroup([
                                dbc.Button([html.Span(className='fa fa-external-link')], id="link-por",color='link'),
                                dbc.Button("Link", id='link-po',color="link"),]),
         dbc.ButtonGroup([
                                dbc.Button([html.Span(className='fa fa-file-pdf-o')], id="pdf-por",color='link'),
                                dbc.Button("PDF", id='pdf-po',color="link"),]),
         dbc.ButtonGroup([
                                dbc.Button([html.Span(className='fa fa-file-excel-o')], id="excel-por",color='link'),
                                dbc.Button("Excel", id='excel-po',color="link"),])
    ],className='row',style={'margin-left':'20px'}),

    html.Div([
        dbc.CardHeader(
            dbc.Tabs([
                dbc.Tab(label="Summary", tab_id="summary"),
                dbc.Tab(label="Metrics", tab_id="metrics"),
                dbc.Tab(label="Efficient Frontier",tab_id="Ef_F"),
                dbc.Tab(label="Annual Return", tab_id="An_R"),
                dbc.Tab(label="Monthly Return", tab_id="Mo_R"),
                dbc.Tab(label="Drawdowns",tab_id="drawdowns"),
                dbc.Tab(label="Assets", tab_id="assets"),
                dbc.Tab(label="Resampling", tab_id="Resample"),
            ],id='card-tabs',active_tab='summary',card=True
            )
        ),
        dbc.CardBody(html.P(id="card-content", className="card-text")),]),
        #END

], className='container-fluid',style={'backgroundColor':'#DAE6DB','max-weight':'50000','padding-left':'0px','padding-right':'0px'})


'''This function is for loading csv file and upload SUMMARY TAB'''
@app.callback(Output('card-content', 'children'),
              [Input("submit_button","n_clicks"),
               Input("card-tabs", "active_tab")],
              [State('upload-data', 'contents'),
               State('upload-data', 'filename'),
               State('upload-data', 'last_modified'),
               State('opt_goal', 'value'),
               State('beginY','value'),
               State('endY','value'),
               State('StartM','value'),
               State('EndM','value'),
               State('target-ret-input','value'),
               State('target-vol-input','value'),
               State('dropdown-ass','value'),
               State('benchmark-t','value'),
               ])
def parse_contents(clicks,tabs,contents, filename, date, goal, beginY,endY, startM, endM,tar_r,tar_v,constraint,benchmark):
    if clicks:
        if contents is not None:
            contentsS = ','.join(contents)
            #date = int(''.join(date))
            filename = ''.join(filename) #filename is a list
            goal = ''.join(goal)
            content_type, content_string = contentsS.split(',')

            decoded = base64.b64decode(content_string)
            try:
                if 'csv' in filename:
                    # Assume that the user uploaded a CSV file
                    df2 = pd.read_csv(
                        io.StringIO(decoded.decode('utf-8')))
                elif 'xls' in filename:
                    # Assume that the user uploaded an excel file
                    df2 = pd.read_excel(io.BytesIO(decoded))
            except Exception as e:
                print(e)
                return html.Div([
                    'There was an error processing this file.'
                ])
            rf =DP.GetRF()
            price,Benchmark = DP.Yahoo(df2,beginY, endY, startM, endM,benchmark)
            (mu,S,returns) = DP.GetFromPrice(price)
            #get default weight bounds in appropriate number
            #define bonunds
            weight_bounds=[]
            if constraint == 'Y': #if chose yes
                weight_bounds = DP.GetBoundry(df2)
            else :
                #get default weight bounds in appropriate number
                for i in range(len(df2)):
                    weight_bounds.append((0,1))

            vr =CVAROpt(returns,weight_bounds)
            rp=HRPOpt(returns)
            ef = EfficientFrontier(mu, S, weight_bounds)
            tar_r=float(tar_r)
            tar_v=float(tar_v)
            if goal=='8.Minimize Maximum Drawdown subject to...':
                weight=Min_Max_drawdown(price,tar_r,len(df2),period='M').x
                weight2 = [ '%.2f' % (elem) for elem in weight ] #round results
            elif goal=='10.Maximize Sortino Ratio subject to...':
                weight=Max_Sortino_Ratio(price,tar_r/12,len(df2),weight_bounds,period='M').x
                weight2 = [ '%.2f' % (elem) for elem in weight ] #round results
            elif goal=='9.Maximize Omega Ratio subject to...':
                weight=Max_Omega_Ratio(price,tar_r/12,rf,len(df2),weight_bounds,period='M').x
                weight2 = [ '%.2f' % (elem) for elem in weight ] #round results
            elif goal=='13.Co-skew':
                weight=coskew(price,len(df2),weight_bounds,period='M').x
                weight2 = [ '%.2f' % (elem) for elem in weight ] #round results
            elif goal=='14.Co-kurtosis':
                weight=cokurt(price,len(df2),weight_bounds,period='M').x
                weight2 = [ '%.2f' % (elem) for elem in weight ] #round results
            elif goal=='11.Tail Risk Parity':
                weight=tailriskparity(price,len(df2),weight_bounds).x
                weight2 = [ '%.2f' % (elem) for elem in weight ] #round results
            elif goal=='12.Co-drawdown':
                weight=co_drawdown(price,len(df2),weight_bounds).x
                weight2 = [ '%.2f' % (elem) for elem in weight ] #round results   
            elif goal=='5.Maximize Information Ratio':
                start_ben = datetime.date(beginY, startM, 1)#resolve if only input year later
                end_ben = datetime.date(endY,endM, 27)
                ben_data=pdr.get_data_yahoo(benchmark, start_ben, end_ben)
                ben_data.dropna(axis=0,how='any',inplace=True)
                ben_price=ben_data["Close"]
                print(ben_price)
                weight=Infor_Ratio(price,ben_price,len(df2),weight_bounds,period='M').x
                weight2 = [ '%.2f' % (elem) for elem in weight ] #round results   
                
            else:
                weights = Optimization(goal, ef, vr,rp, rf,tar_r,tar_v)
                print("This is our goal *********")
                print(goal)
                weight = list(weights.values())
                weight2 = [ '%.2f' % (elem) for elem in weight ] #round results
            
             #Provided Portfolio
            yourAllo = DP.GetAllo(df2)
            df_Allo = DataFrame.from_dict(yourAllo)

            if tabs =='summary':
                df = DataFrame(weight2, columns=['Optimized Weights'],index=list(df2.iloc[:,0]))
                dfc = df.copy()
                dfc.insert(0,"Assets",list(df2.iloc[:,0]),True)
                #dfc.append({'Weight Bounds':weight_bounds}, ignore_index=True)
                return html.Div([
                        html.Div([
                                html.H5('Optimization Results',style={'margin-right':'20px'}),
                                html.H6(filename,style={'margin-right':'20px'}),
                                #html.H6(datetime.datetime.fromtimestamp(date),style={'margin-right':'20px'})
                                ]),
                        html.Div([
                                html.Div([
                        dash_table.DataTable(
                            data=dfc.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in dfc.columns]
                        ),],className='col-md-7'),
                        html.Div([
                            dcc.Graph(id='wePie',
                                       figure=go.Figure(
                                           data=[go.Pie(labels=list(df2.iloc[:,0]),
                                                        values=list(df.iloc[:,0]))],
                                           layout=go.Layout(
                                               title='Optimized weights')
                                               ))],className='col-md-5'),
                        ],className='row'),
                        html.Div([
                                html.Div([
                        dash_table.DataTable(
                            data=df_Allo.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in df_Allo.columns]
                        ),],className='col-md-7'),
                        html.Div([
                            dcc.Graph(id='pPie',
                                       figure=go.Figure(
                                           data=[go.Pie(labels=yourAllo['Assets'],
                                                        values=yourAllo['Provided Allocation'])],
                                           layout=go.Layout(
                                               title='Provided Weights')
                                               ))],className='col-md-5'),
                        ],className='row'),
                        ])
                        #])])
            elif tabs =='metrics':
                name,value = metrics(weight,price,Benchmark)
                dfm=pd.DataFrame()
                dfm['Names']=name
                dfm['{}'.format(goal)]=value
                wP = list(yourAllo['Provided Allocation'])
                nameP,valueP = metrics(wP,price,Benchmark)
                dfm['Provided Portfolio'] = valueP
                return html.Div([
                        html.Div([
                                html.H5('Optimization Metrics',style={'margin-right':'20px'})
                                ]),
                        html.Div([
                                html.Div([
                        dash_table.DataTable(
                            data=dfm.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in dfm.columns]
                        ),],className='col-md-8'),
                        ])
                        ])
            elif tabs =='An_R':
                returnsA = DP.GetAnnualR(price)
                weight2 = [ float(elem) for elem in weight ] #round results, a list
                #dot product returns and weights and *100
                pf_returns = returnsA.mul(weight2).sum(axis = 1).div(0.01)
                dates = returnsA.index.tolist()
                pf_returns = [ '%.2f' % elem for elem in pf_returns ]

                old_returns = returnsA.mul(yourAllo['Provided Allocation']).sum(axis = 1).div(0.01)
                old_returns = [ '%.2f' % elem for elem in old_returns ]
                dic = {'Dates':dates,'Optimized Portfolio Returns(%)':pf_returns,
                'Provided Profolio Returns(%)':old_returns}
                df_MR = DataFrame.from_dict(dic)
                return html.Div([
                html.Div([
                    dash_table.DataTable(
                        data=df_MR.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in df_MR.columns]
                    ),],className='col-md-5'),
                    html.Div([
                    dcc.Graph(id='MRGraph',
                                 figure={
                                    'data':[
                                            go.Scatter(
                                                        x = dates,
                                                        y = pf_returns,
                                                        mode = 'lines+markers',
                                                        name = 'Optimized Annually Return'),
                                            go.Scatter(
                                                        x = dates,
                                                        y = old_returns,
                                                        mode = 'lines+markers',
                                                        name = 'Provided Annually Return')
                                                ],
                                     'layout': go.Layout(yaxis=go.layout.YAxis(title='Portfolio Historical Returns (%)'),
                                    xaxis=go.layout.XAxis(title='Dates'))
                                 })
                    ],className='col-md-7')
                    ],className='row')
            elif tabs =='Mo_R':
                weight2 = [ float(elem) for elem in weight ] #round results, a list
                #dot product returns and weights and *100
                pf_returns = returns.mul(weight2).sum(axis = 1).div(0.01)
                dates = returns.index.tolist()
                pf_returns = [ '%.2f' % elem for elem in pf_returns ]

                old_returns = returns.mul(yourAllo['Provided Allocation']).sum(axis = 1).div(0.01)
                old_returns = [ '%.2f' % elem for elem in old_returns ]
                dic = {'Dates':dates,'Optimized Portfolio Returns(%)':pf_returns,
                'Provided Profolio Returns(%)':old_returns}
                df_MR = DataFrame.from_dict(dic)
                return html.Div([
                html.Div([
                    dash_table.DataTable(
                        data=df_MR.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in df_MR.columns]
                    ),],className='col-md-5'),
                    html.Div([
                    dcc.Graph(id='MRGraph',
                                 figure={
                                    'data':[
                                            go.Scatter(
                                                        x = dates,
                                                        y = pf_returns,
                                                        mode = 'lines+markers',
                                                        name = 'Optimized Monthly Return'),
                                            go.Scatter(
                                                        x = dates,
                                                        y = old_returns,
                                                        mode = 'lines+markers',
                                                        name = 'Provided Monthly Return')
                                                ],
                                     'layout': go.Layout(yaxis=go.layout.YAxis(title='Portfolio Historical Returns (%)'),
                                    xaxis=go.layout.XAxis(title='Dates'))
                                 })
                    ],className='col-md-7')
                    ],className='row')
            elif tabs =='drawdowns':
                weight2 = [ float(elem) for elem in weight ] #round results, a list
                #dot product returns and weights and *100
                pf_returns = returns.mul(weight2).sum(axis = 1).div(0.01)
                dates = returns.index.tolist()
                pf_returns = [ '%.2f' % elem for elem in pf_returns ]
                dateP = []
                for d in dates:
                    dateP.append(d)
                old_returns = returns.mul(yourAllo['Provided Allocation']).sum(axis = 1).div(0.01)
                old_returns = [ '%.2f' % elem for elem in old_returns ]
                dicO = {'Dates':dates,'Optimized Portfolio Returns(%)':pf_returns}
                dicP = {'Dates':dateP, 'Provided Portfolio Returns(%)':old_returns}
                #Delete positive returns in the list in the dic
                for i in range(len(dates)-1,0,-1):
                    if float(dicO['Optimized Portfolio Returns(%)'][i]) > 0 :
                        dicO['Optimized Portfolio Returns(%)'].pop(i)
                        dicO['Dates'].pop(i)
                for i in range(len(dateP)-1,0,-1):
                    if float(dicP['Provided Portfolio Returns(%)'][i]) > 0 :
                        dicP['Provided Portfolio Returns(%)'].pop(i)
                        dicP['Dates'].pop(i)
                return html.Div([
                    html.Div([
                    dcc.Graph(id='MRGraph',
                                 figure={
                                    'data':[
                                            go.Scatter(
                                                        x = dicO['Dates'],
                                                        y = dicO['Optimized Portfolio Returns(%)'],
                                                        mode = 'lines+markers',
                                                        name = 'Drawdowns_Optimized'),
                                            go.Scatter(
                                                        x = dicP['Dates'],
                                                        y = dicP['Provided Portfolio Returns(%)'],
                                                        mode = 'lines+markers',
                                                        name = 'Drawdowns_Provided')
                                                ],
                                     'layout': go.Layout(yaxis=go.layout.YAxis(title='Drawdowns (%)'),
                                    xaxis=go.layout.XAxis(title='Dates'))
                                 })
                    ],className='col-md-7')
                    ])
            elif tabs == 'Ef_F' :
                (pf_weights, pf_mu, pf_sigma) = All_frontier(mu,S,ef,rf)
                Er = weight @ mu
                vol = math.sqrt(weight @ S @ weight)
                weight_P = list(yourAllo.values())[1]
                Er_P = weight_P @ mu
                vol_P = math.sqrt(weight_P @ S @ weight_P)
                return html.Div([
                    dcc.Graph(id='EFGraph',
                                 figure={
                                    'data':[
                                            go.Scatter(
                                                        x = pf_sigma,
                                                        y = pf_mu,
                                                        mode = 'lines+markers',
                                                        name = 'MV Efficient Frontier'),
                                            go.Scatter(
                                                        x = [vol],
                                                        y = [Er],
                                                        mode = 'markers',
                                                        marker={
                                                            'size': 15,
                                                            'line': {'width': 0.5, 'color': 'white'}
                                                            },
                                                        name = '{}'.format(goal)),
                                            go.Scatter(
                                                        x = [vol_P],
                                                        y = [Er_P],
                                                        mode = 'markers',
                                                        marker={
                                                            'size': 15,
                                                            'line': {'width': 0.5, 'color': 'white'}
                                                            },
                                                        name = 'Provided Portfolio')
                                                ],
                                     'layout': go.Layout(yaxis=go.layout.YAxis(title='Means'),
                                    xaxis=go.layout.XAxis(title='Standard Deviation'))
                                 })
                ],className='col-md-7')
            elif tabs == 'Resample':
                (pf_weights, pf_mu, pf_sigma) = All_frontier(mu,S,ef,rf)
                (muR,sR) = Resampling_EF(mu, S,weight_bounds,rf)
                return html.Div([
                    html.H6("""The Resampling procedure is based on Richard Michaud's Theory"""),
                    html.H6("""The procedure has U.S. Patent #6,003,018 by Michaud et al., December 19, 1999."""),
                    dcc.Graph(id='ResampledEF',
                                 figure={
                                    'data':[
                                            go.Scatter(
                                                        x = sR,
                                                        y = muR,
                                                        mode = 'lines+markers',
                                                        name = 'Resampled Efficient Frontier'),
                                            go.Scatter(
                                                        x = pf_sigma,
                                                        y = pf_mu,
                                                        mode = 'lines+markers',
                                                        name = 'MV Efficient Frontier')

                                                ],
                                     'layout': go.Layout(yaxis=go.layout.YAxis(title='Means'),
                                    xaxis=go.layout.XAxis(title='Standard Deviation'))
                                 })
                ],className='col-md-7')
                #*****************Assets tab*********************************
            elif tabs =='assets':
                tickers = list(df2.iloc[:,0])
                Asset_mu = mu.tolist()
                Asset_muR = [ '%.2f' % (elem*100) for elem in Asset_mu ]
                std_monthly = returns.std().tolist()
                a = list(map(lambda x: x - rf, Asset_mu)) #-rf to each number in list
                Asset_std = list(map(lambda x: x * math.sqrt(12), std_monthly))
                Asset_stdR = [ '%.2f' % (elem*100) for elem in Asset_std ]
                Asset_Sharpe = [x/y for x, y in zip(a, Asset_std)]
                Asset_SharpeR = [ '%.2f' % elem for elem in Asset_Sharpe ]
                returnsA = DP.GetAnnualR(price)
                minimum = returnsA.min().tolist() #monthly
                print(minimum)
                max_drawdown = [ '%.2f' % (elem*100) for elem in minimum ]
                dic = {'Tickers':tickers,'Mean Historical Return':Asset_muR,
                'Standard Dev':Asset_stdR,'Max. Drawdown':max_drawdown,'Sharpe Ratio':Asset_SharpeR}
                df_assets = DataFrame.from_dict(dic)
                #for graph
                dates = returnsA.index.tolist()
                return html.Div([
                html.Div([html.H6('All are Annuallized Data in %')]),
                html.Div([
                    dash_table.DataTable(
                        data=df_assets.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in df_assets.columns]
                    ),]),
                    html.Div([
                        html.A('Annual Returns of Assets in Portfolio'),
                        dcc.Graph(
                            id='life-exp-vs-gdp',
                            figure={
                                'data': [
                                    go.Bar(
                                        x=dates,
                                        y=returnsA['{}'.format(t)].tolist(),
                                        #text=df[df['continent'] == i]['country'],
                                        #mode='markers',
                                        name=t
                                    ) for t in tickers
                                ],
                                'layout': go.Layout(yaxis=go.layout.YAxis(title='Annual Return'),
                               xaxis=go.layout.XAxis(title='Years'))
                               })
                    ])
                    ])


'''***********************************************************************************************
'''
page_2_layout = html.Div([html.Div([
    dcc.Store(id='memory-output'),
]),
html.Div([
    html.Div([
        dbc.NavbarSimple(
                children=[
                    dbc.NavItem(dbc.NavLink("Home", href="#")),
                    dbc.DropdownMenu(
                        children=[
                            dbc.DropdownMenuItem("FAQ", header=True),
                            dcc.Link("Portfolio Optimization", href="/page-Main"),
                            dbc.DropdownMenuItem("Example", href="#"),
                        ],
                        nav=True,
                        in_navbar=True,
                        label="More",
                    ),
                ],
                brand="Black Litterman",
                brand_href="#",
                color="#A3D1BE",
                dark=True,
               style={'width':'100%'},
            )],className='container-fluid',style={'max-weight':'50000','padding-left':'0px','padding-right':'0px'}),
    dbc.Jumbotron(
    [
        html.H1("Black-Litterman Asset Allocation Model", className="display-3"),
        html.P("This portfolio optimizer tool implements the Black-Litterman asset allocation model.",
            className="lead",
        ),
        html.Hr(className="my-2"),
        html.P( "The Black-Litterman asset allocation model combines ideas from the Capital Asset Pricing Model (CAPM) and the Markowitz’s mean-variance optimization model to provide a a method to calculate the optimal portfolio weights based on the given inputs. "),
        html.P( "The model first calculates the implied market equilibrium returns based on the given benchmark asset allocation weights, and then allows the investor to adjust these expected returns based on the investor's views. The opinion adjusted returns are then passed to the mean variance optimizer to derive the optimal asset allocation weights. "),
        
    ]
),
    html.H1(' Benchmark Portfolio'),
    
   
       html.Div([
                html.Div([
                    html.Strong('Start Year'),
                    html.I(id='bl-start-button', n_clicks=0, className='fa fa-info-circle'),
                    dbc.Tooltip(
            "Start year for portfolio optimization period",
            target="bl-start-button",
        ),
                ],className='col-3',style={ 'margin': '5px 5px 5px'}),
                html.Div([
                    dcc.Dropdown(id='bl-beginY',
                         options=[
            {'label': i+1985, 'value': i+1985}
            for i in range(datetime.datetime.now().year-1985)
        ],
        value=1985,style={'backgroundColor':'#EDF0EE'}
            ),
                ],className='col-6',style={ 'margin': '5px 5px 5px'}),
                html.Div([],className='col-3')
            ],className='row',style={ 'margin-left': '10px'}), 
   html.Div([
    html.Div([
        html.Strong('Start Month'),
                html.I(id='bl-start-month', n_clicks=0, className='fa fa-info-circle'),
                dbc.Tooltip(
            "Specify start period in years or in months",
            target="bl-start-month",
        ),
    ],className='col-3',style={'margin': '5px 5px 5px'}),
    html.Div([
        dcc.Dropdown(id='bl-StartM',options=[{'label': i, 'value' : i} for i in range(1,13)],value=1,style={'backgroundColor':'#EDF0EE'})
    ],className='col-6',style={'margin': '5px 5px 5px'}),
    html.Div([],className='col-3',style={'margin': '5px 5px 5px'}),
],className='row',style={ 'margin-left': '10px'})
,
    html.Div([
                html.Div([
                    html.Strong('End Year'),
                    html.I(id='bl-end-button', n_clicks=0, className='fa fa-info-circle'),
                    dbc.Tooltip(
            "End year for portfolio optimization period",
            target="bl-end-button",
        ),
                ],className='col-3',style={ 'margin': '5px 5px 5px'}),
                html.Div([
                    dcc.Dropdown(id='bl-endY',
                         options=[
            {'label': i+1985, 'value': i+1985}
            for i in range(datetime.datetime.now().year-1984)
        ],
        value=2019,style={'backgroundColor':'#EDF0EE'}
            ),
                ],className='col-6',style={ 'margin': '5px 5px 5px'}),
                html.Div([],className='col-3')
            ],className='row',style={ 'margin-left': '10px'}), 
    html.Div([
    html.Div([
        html.Strong('End Month'),
                html.I(id='bl-end-month', n_clicks=0, className='fa fa-info-circle'),
                dbc.Tooltip(
            "Specify start period in years or in months",
            target="bl-end-month",
        ),
    ],className='col-3',style={'margin': '5px 5px 5px'}),
    html.Div([
        dcc.Dropdown(id='bl-EndM',options=[{'label': i, 'value' : i} for i in range(1,13)],value=1,style={'backgroundColor':'#EDF0EE'})
    ],className='col-6',style={'margin': '5px 5px 5px'}),
    html.Div([],className='col-3',style={'margin': '5px 5px 5px'}),
],className='row',style={ 'margin-left': '10px'}),
html.A('(This is an example format of the CSV file you should input)', style={'fontSize':'15px','float':'right'},download='BL_test.csv', href='/BL_test.csv'),
 html.Div([
        html.Div([
        dcc.Upload(
            id='bl-upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Your Portfolio Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        )],className='col-md-10'),
        ],className='row'),
    html.Div([
        html.Hr(),
        html.Div(id='bl-output-data-upload')],className='row',style={ 'margin-left': '10px'}),
    
    html.Div([
        dbc.Button("Next", color="info",id='bl-button-step-2', className="mr-1"),
        html.Div(dbc.Button("Equal", color="info",id='bl-button-step-3', className="mr-1")),
    ],className='row',style={ 'margin-left': '10px'}),
    html.Div([html.H1(' Results')],className='row',style={ 'margin-left': '10px'}),
    html.Div(id='bl-table-3'),
    html.Div(id='bl-table'),
    html.Div(id='bl-optimization'),
    html.Div([
    
      dbc.ButtonGroup([
                                dbc.Button([html.Span(className='fa fa-file-pdf-o')], id="pdf-por",color='link'),
                                dbc.Button("PDF", id='pdf-po',color="link"),]),
    
]),
]) 
                      ],id='bl-step-3',className='container-fluid',style={'backgroundColor':'#DAE6DB','max-weight':'50000','padding-left':'0px','padding-right':'0px'})
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-Black-Litterman':
        return page_2_layout
    elif pathname=='/page-Main':
        return page_1_layout
    else: return page_1_layout



def bl_parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df2 = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df2 = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([

        html.Div([
                html.H3('Optimization Results',style={'margin-left':'20px'}),
                ],className='row'),
        html.Div([
                html.Div([
        dash_table.DataTable(
            data=df2.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df2.columns],
            style_as_list_view=True,
    style_cell={'padding': '5px','backgroundColor':'#EFF2EF'},
    style_header={
        'backgroundColor': '#C0C4C0',
        'fontWeight': 'bold'
    },
        ),],className='col-md-7'),
    html.Div([
        dcc.Graph(#id='wePie',
                   figure=go.Figure(
                       data=[go.Pie(labels=list(df2['tickers']),
                                    values=list(df2['weights']))],
                       layout=go.Layout(
                           title='Optimized weights')
                           ))],className='col-md-5'),
    ],className='row',style={ 'margin-left': '10px'})
    ])

@app.callback(Output('bl-optimization','children'),
              [Input("bl-button-step-2","n_clicks")],
              [State('bl-upload-data', 'contents'),
              State('bl-upload-data', 'filename'),
               State('bl-upload-data', 'last_modified'),
               State('bl-beginY','value'),
               State('bl-endY','value'),
               State('bl-StartM','value'),
               State('bl-EndM','value'),
               ])

def bl_update_output(clicks,contents, filename, date,beginY,endY,beginM,endM):
     if clicks:
        if contents is not None:
            contentsS = ','.join(contents)
            #date = int(''.join(date))
            filename = ''.join(filename) #filename is a list
            content_type, content_string = contentsS.split(',')

            decoded = base64.b64decode(content_string)
            try:
                if 'csv' in filename:
                    # Assume that the user uploaded a CSV file
                    df_bl = pd.read_csv(
                        io.StringIO(decoded.decode('utf-8')))
                elif 'xls' in filename:
                    # Assume that the user uploaded an excel file
                    df_bl = pd.read_excel(io.BytesIO(decoded))
            except Exception as e:
                print(e)
                return html.Div([
                    'There was an error processing this file.'
                ])
            start = datetime.date(beginY, beginM, 1)
            end = datetime.date(endY,endM, 1)
            df_bl.dropna(axis=0,how='any')
            Names = df_bl.iloc[:,0]
            tickerList = Names.tolist()
            cap = DPB.YahooMC(df_bl)
            names, prices, caps = BLO.load_data_net(tickerList, start, end, cap)
            n = len(names)
            names, W, R, C = BLO.assets_meanvar(names, prices, caps)
            rfmonthly = DPB.GetRF() # get monthly risk-free rate
            rf = math.exp((math.log(1+rfmonthly))/30)-1 # get monthly risk-free rate
            his_weight,his_corr=BLO.print_assets(names, W, R, C)
            mean1, var,f_mean, f_var, f_weights,opt_his_ret,opt_his_corr=BLO.optimize_and_display(names, R, C, rf)
            figure1=figureplt(C,R,n,names,var,mean1,f_var,f_mean)
            lmb = (mean1 - rf) / var                         # Calculate return/risk trade-off
            Pi = dot(dot(lmb, C), W)                        # Calculate equilibrium excess returns
            mean2, var2,f_mean2, f_var2, f_weights2,opt_equ_ret,opt_equ_corr=BLO.optimize_and_display(names, Pi+rf, C, rf)
            figure2=figureplt(C,R,n,names,var2,mean2,f_var2,f_mean2)
            views = []
            for i in range(0,len(df_bl.dropna().iloc[:,2])):
                views.append((df_bl.iloc[i,2], df_bl.iloc[i,3], df_bl.iloc[i,4], df_bl.iloc[i,5]))
            Q, P = BLO.prepare_views_and_link_matrix(names, views)
            tau = .025 # scaling factor
            # Calculate omega - uncertainty matrix about views
            omega = dot(dot(dot(tau, P), C), transpose(P)) # 0.025 * P * C * transpose(P)
            # Calculate equilibrium excess returns with views incorporated
            sub_a = inv(dot(tau, C))
            sub_b = dot(dot(transpose(P), inv(omega)), P)
            sub_c = dot(inv(dot(tau, C)), Pi)
            sub_d = dot(dot(transpose(P), inv(omega)), Q)
            Pi = dot(inv(sub_a + sub_b), (sub_c + sub_d))
            mean3, var3,f_mean3, f_var3, f_weights3,opt_equ_adj_ret,opt_equ_adj_corr=BLO.optimize_and_display(names, Pi+rf, C, rf)
            figure3=figureplt(C,R,n,names,var3,mean3,f_var3,f_mean3)
            layout_bl=html.Div([
            html.H3('Historical Weights',style={'margin-left':'20px'}),
            html.Div([
                html.Div([
        dash_table.DataTable(
            data=his_weight.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in his_weight.columns],
            style_as_list_view=True,
    style_cell={'padding': '5px','backgroundColor':'#EFF2EF'},
    style_header={
        'backgroundColor': '#C0C4C0',
        'fontWeight': 'bold'
    },
        ),],className='col-md-4',style={ 'margin-left': '10px'}),
    html.Div([
       dash_table.DataTable(
            data=his_corr.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in his_corr.columns],
            style_as_list_view=True,
    style_cell={'padding': '5px','backgroundColor':'#EFF2EF'},
    style_header={
        'backgroundColor': '#C0C4C0',
        'fontWeight': 'bold'
    },
        )],className='col-md-7', style={ 'margin-left': '10px'})
    ],className='row',style={ 'margin-up': '20px'}),
   
    html.H3('Optimization based on Historical returns',style={'margin-right':'20px'}),
    html.Div([
                html.Div([
        dash_table.DataTable(
            data=opt_his_ret.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in opt_his_ret.columns],
            style_as_list_view=True,
    style_cell={'padding': '5px','backgroundColor':'#EFF2EF'},
    style_header={
        'backgroundColor': '#C0C4C0',
        'fontWeight': 'bold'
    },
        ),],className='col-md-4',style={ 'margin-left': '10px'}),
    html.Div([
       dash_table.DataTable(
            data=opt_his_corr.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in opt_his_corr.columns],
            style_as_list_view=True,
    style_cell={'padding': '5px','backgroundColor':'#EFF2EF'},
    style_header={
        'backgroundColor': '#C0C4C0',
        'fontWeight': 'bold'
    },
        )],className='col-md-7', style={ 'margin-left': '10px'})
    ],className='row',style={ 'margin-up': '20px'}),
     html.Div([
                    dcc.Graph(
                               figure=figure1
                                 )
                ],className='row'),
    
    html.H3('Optimization based on Equilibrium returns',style={'margin-right':'20px'}),
    html.Div([
                html.Div([
        dash_table.DataTable(
            data=opt_his_ret.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in opt_his_ret.columns],
            style_as_list_view=True,
    style_cell={'padding': '5px','backgroundColor':'#EFF2EF'},
    style_header={
        'backgroundColor': '#C0C4C0',
        'fontWeight': 'bold'
    },
        ),],className='col-md-4',style={ 'margin-left': '10px'}),
    html.Div([
       dash_table.DataTable(
            data=opt_his_corr.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in opt_his_corr.columns],
            style_as_list_view=True,
    style_cell={'padding': '5px','backgroundColor':'#EFF2EF'},
    style_header={
        'backgroundColor': '#C0C4C0',
        'fontWeight': 'bold'
    },
        )],className='col-md-7', style={ 'margin-left': '10px'})
    ],className='row',style={ 'margin-up': '20px'}),
                                        html.Div([
                    dcc.Graph(
                               figure=figure2
                                 )
                ],className='row'),
    html.H3('Optimization based on Equilibrium returns with adjusted views',style={'margin-left':'20px'}),
            html.Div([
                html.Div([
        dash_table.DataTable(
            data=opt_equ_adj_ret.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in opt_equ_adj_ret.columns],
            style_as_list_view=True,
    style_cell={'padding': '5px','backgroundColor':'#EFF2EF'},
    style_header={
        'backgroundColor': '#C0C4C0',
        'fontWeight': 'bold'
    },
        ),],className='col-md-4',style={ 'margin-left': '10px'}),
    html.Div([
       dash_table.DataTable(
            data=opt_equ_adj_corr.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in opt_equ_adj_corr.columns],
            style_as_list_view=True,
    style_cell={'padding': '5px','backgroundColor':'#EFF2EF'},
    style_header={
        'backgroundColor': '#C0C4C0',
        'fontWeight': 'bold'
    },
        )],className='col-md-7', style={ 'margin-left': '10px'})
    ],className='row',style={ 'margin-up': '20px'}),
        html.Div([
                    dcc.Graph(
                               figure=figure3
                                 )
                ],className='row'),
    ] )  
            return layout_bl

            
def figureplt(C,R,n,names,var,mean1,f_var,f_mean):
    figure_his, ax = plt.subplots()
    ax.scatter([C[i,i]**.5 for i in range(n)], R, marker='x')  # draw assets
            
    for i in range(n):                                                                              # draw labels
        ax.text(C[i,i]**.5, R[i], '  %s'%names[i], verticalalignment='center')
    ax.scatter(var**.5, mean1, marker='o')                 # draw tangency portfolio
    ax.plot(f_var**.5, f_mean)                                    # draw min-var frontier
    ax.grid(True)
    plotly_fig = mpl_to_plotly(figure_his)
    return plotly_fig


@app.callback(Output('bl-table-3', 'children'),
              [Input("bl-button-step-3","n_clicks")],
              [State('bl-upload-data', 'contents'),
              State('bl-upload-data', 'filename'),
               State('bl-upload-data', 'last_modified')])
def bl_update_output_2(clicks,list_of_contents, list_of_names, list_of_dates):
    if clicks:
        if list_of_contents is not None:
            children = [
                    bl_parse_contents(c, n, d) for c, n, d in
                    zip(list_of_contents, list_of_names, list_of_dates)]
            return children



external_css=["https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css",
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css']
for css in external_css:
    app.css.append_css({"external_url":css})

external_js=["https://code.jquery.com/jquery-3.3.1.slim.min.js",
            'https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js',
            'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js',
            'https://cdn.staticfile.org/twitter-bootstrap/3.3.7/js/bootstrap.min.js',
            'https://rawgit.com/lwileczek/dash/master/undo_redo5.css']
for js in external_js:
    app.scripts.append_script({"external_url":js})


if __name__ == '__main__':
    app.run_server(debug=True,port=10440)
