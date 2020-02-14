import pandas as pd
import numpy as np
import base64
import datetime
import io
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from statistics import mean

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# static functions ====
def bin_recode(value_list:list, threshold:int, recoded_list):
    '''recoding probability values (-1: 1) to binary values (spam, ham etc.)
     according to given threshold'''
    new_list = []
    for val in value_list:
        if np.isnan(val):
            new_val = 'NA'
        elif val > threshold:
            new_val = recoded_list[0]
        else:
            new_val = recoded_list[1]
        new_list.append(new_val)
    return new_list


def conf_matrix(correct_values:list, test_values:list, m_name:str, levels:list, count_na=False):
    '''construct confusion matrix, correct_values and test_values should be the same type
    if count_na = True NA in data will be counted as errors, if False they are ignored.
     m_name is the name of ML model used for prediction'''
    values = levels
    if len(values) == 1:
        value_1 = values[0]
        if count_na is False:
            # remove values from correct where test is NA
            na_indices = [i for i, x in enumerate(test_values) if x == 'NA']
            start_len = len(correct_values)
            new_cor = []
            new_test = []
            for i in range(start_len):
                if i not in na_indices:
                    new_cor.append(correct_values[i])
                    new_test.append(test_values[i])
            correct_values = new_cor
            test_values = new_test
        total_value_1 = correct_values.count(value_1)
        values_1_correct = 0
        for i, val in enumerate(correct_values):
            if val == test_values[i] and val == value_1:
                values_1_correct += 1
        name = [value_1]
        correct = [values_1_correct]
        errors = [total_value_1-values_1_correct]
        total = [total_value_1]
    elif len(values) == 2:
        values.sort()
        value_1 = values[0]
        value_2 = values[1]
        if count_na is False:
            # remove values from correct where test is NA
            na_indices = [i for i, x in enumerate(test_values) if x == 'NA']
            start_len = len(correct_values)
            new_cor = []
            new_test = []
            for i in range(start_len):
                if i not in na_indices:
                    new_cor.append(correct_values[i])
                    new_test.append(test_values[i])
            correct_values = new_cor
            test_values = new_test
        total_value_1 = correct_values.count(value_1)
        total_value_2 = correct_values.count(value_2)
        values_1_correct = 0
        values_2_correct = 0
        for i, val in enumerate(correct_values):
            if val == test_values[i] and val == value_1:
                values_1_correct += 1
            elif val == test_values[i] and val == value_2:
                values_2_correct += 1
        name = [value_1, value_2, 'total']
        correct = [values_1_correct, values_2_correct, values_1_correct + values_2_correct]
        errors = [total_value_1-values_1_correct, total_value_2-values_2_correct, total_value_1-values_1_correct + total_value_2-values_2_correct]
        total = [total_value_1, total_value_2, total_value_1 + total_value_2]
    elif len(values) == 0:
        print('No values found!')
    else:
        print(f'too many correct values {values} should be 1 or 2')
    cm = pd.DataFrame({m_name:name, 'Correct':correct, 'Errors':errors, 'Total':total})
    cm['%'] = cm['Correct']/cm['Total']*100
    return cm


def values_table(cm_df):
    '''Prepares table with TP, TN, FP, FN from confusion matrix'''
    parameters = ['True Positive (TP)', 'False Positive (FP)', 'True Negative (TN)', 'False Negative (FN)']
    values = [cm_df['Correct'][0], cm_df['Errors'][1], cm_df['Correct'][1], cm_df['Errors'][0]]
    df = pd.DataFrame({'Parameter': parameters, 'Value': values})
    return df


def metric_table(cm_df):
    '''Prepare metric table from confusion matrix'''
    parameters = ['Accuracy','Error', 'Recall/Sensitivity', 'Specificity', 'Precision/Positive Predictive Value',
                  'Negative Predictive Value', 'False Positive Rate', 'False Negative Rate']
    tp = cm_df['Correct'][0]
    tn = cm_df['Correct'][1]
    fn = cm_df['Errors'][0]
    fp = cm_df['Errors'][1]
    values = [(tp+tn)/(tp+tn+fp+fn), (fp+fn)/(tp+tn+fp+fn), tp/(tp+fn), tn/(tn+fp),
              tp/(tp+fp), tn/(tn+fn), fp/(fp+tn), fn/(fn+tp)]
    df = pd.DataFrame({'Parameter': parameters, 'Value': values})
    df['Value'] = df['Value']*100
    df = df.round(2)
    df['Value'] = [str(x)+' %' for x in df['Value']]
    return df


def generate_table(dataframe, max_rows=10):
    '''Generate html table from pd.DataFrame'''
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +
        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


def parse_contents(filename, date):
    '''create html div for the upload reporting'''
    return html.Div([
        html.H5('File: '+ filename+' uploaded at: '+str(datetime.datetime.fromtimestamp(date))),
    ])

# Application start ===========================================================
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H4('Upload Excel file with the results:'),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'width': '50%',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Only one file can be uploaded
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    # hidden div used for intermediate calculation of df
    html.Div(id='result_df', style={'display': 'none'}),
    html.Div(id='columns', style={'display': 'none'}),
    html.Hr(),
    html.Hr(),
    html.H3('Select columns from the Excel file to be processed:'),
    html.Div([
                html.H4('Select real values'),
                dcc.Dropdown(id="real_selection")],
                             style={'width': '20%',
                                    'display': 'inline-block',
                                    'margin-left': '20px'}),
    html.Div([
                html.H4('Select predicted values'),
                dcc.Dropdown(id="predicted_selection")],
                             style={'width': '20%',
                                    'display': 'inline-block',
                                    'margin-left': '20px'}),
    html.Div([
                html.H4('Select group 1 values'),
                dcc.Dropdown(id="group1_selection")],
                             style={'width': '20%',
                                    'display': 'inline-block',
                                    'margin-left': '20px'}),
    html.Div([
                html.H4('Select group 2 values'),
                dcc.Dropdown(id="group2_selection")],
                             style={'width': '20%',
                                    'display': 'inline-block',
                                    'margin-left': '20px'}),

    html.Hr(),
    html.Hr(),
    html.Div(
        [
            html.Div(id='group1_dd_title'),
            dcc.Dropdown(id="group1_dd",
                         value=None,
                         multi=True
                         )],
            style={'width': '45%',
                    'display': 'inline-block',
                   'margin-left': '20px'
                   }),
    html.Div(
        [
            html.Div(id='group2_dd_title'),
            dcc.Dropdown(id="group2_dd",
                         value=None,
                         multi=True
                         )],
            style={'width': '45%',
                    'display': 'inline-block',
                   'margin-left': '20px'
                   }),
    html.Div(
        [
        html.H4('Acceptance threshold'),
        dcc.Slider(id = 'threshold_slider',
                   value=0,
                   )],
        style={'width': '50%',
               'display': 'inline-block'}),
    html.Div(
        [
        html.H4('Number of Threshold levels'),
        dcc.Input(id = 'treshold_levels',
                   placeholder='Enter a number...',
                    type='number',
                    value='20'
                   )],
        style={'width': '40%',
               'margin-left': '20px',
                'margin-top': '20px',
               'display': 'inline-block'}),
    html.Div(
        dcc.Graph(id='cm_graph'),
        style={'width': '50%', 'display': 'inline-block', 'float': 'left', 'margin-top': '50px'}
        ),
    html.Div(
        [html.H4(children='Statistical values'),
         # html.Table(id='values_table'),
         html.Table(id='value_table')
         ],
        style={'width': '45%',  'display': 'inline-block', 'margin-left': '50px', 'margin-top': '50px', 'float': 'left'}
    ),
    html.Div(
        style={'width': '100%',
               'display': 'inline-block'}),
    html.Hr(),
    html.Hr(),
    html.Div(
        dcc.Graph(id='cm_plot'),
        style={'width': '50%', 'display': 'inline-block', 'margin-top': '50px'}
        ),
    html.Div(
        [html.H4(children='Statistical measures'),
         # html.Table(id='values_table'),
         html.Table(id='metric_table')
         ],
        style={'width': '45%',  'display': 'inline-block', 'margin-left': '50px', 'margin-top': '50px', 'float': 'right'}
    ),
])


# dimnamic functions =====
@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'filename'),
               Input('upload-data', 'last_modified')])
def update_output(name, date):
    '''update the upload file report'''
    if name is not None:
        children = parse_contents(name, date)
        return children


# new function to convert the upload data to JSON and columns list to be used in the next functions
@app.callback([Output('result_df', 'children'),
               Output('columns', 'children')],
                [Input('upload-data', 'contents')])
def process_df(content):
    if content is not None:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        result_df = pd.read_excel(io.BytesIO(decoded))
        columns = result_df.columns
    else:
        result_df = pd.DataFrame()
        columns = []
    result_df = result_df.to_json(date_format='iso', orient='split')
    return result_df, columns


@app.callback(Output('real_selection', 'options'),
                [Input('columns', 'children')])
def update_dropdown(cols):
    if cols is not None:
        return [{'label': i, 'value': i} for i in cols]
    else:
        return []


@app.callback(Output('predicted_selection', 'options'),
                [Input('columns', 'children')])
def update_dropdown(cols):
    if cols is not None:
        return [{'label': i, 'value': i} for i in cols]
    else:
        return []


@app.callback(Output('group1_selection', 'options'),
                [Input('columns', 'children')])
def update_dropdown(cols):
    if cols is not None:
        return [{'label': i, 'value': i} for i in cols]
    else:
        return []


@app.callback(Output('group2_selection', 'options'),
                [Input('columns', 'children')])
def update_dropdown(cols):
    if cols is not None:
        return [{'label': i, 'value': i} for i in cols]
    else:
        return []


@app.callback(Output('group1_dd_title', 'children'),
                [Input('group1_selection', 'value')])
def update_dropdown_title(value):
    if value is not None:
        return html.H4('Select ' + str(value) + ' value')
    else:
        return html.H4('NA')


@app.callback(Output('group2_dd_title', 'children'),
                [Input('group2_selection', 'value')])
def update_dropdown_title(value):
    if value is not None:
        return html.H4('Select ' + str(value) + ' value')
    else:
        return html.H4('NA')


@app.callback(Output('group1_dd', 'options'),
                [Input('group1_selection', 'value'),
                 Input('result_df', 'children')])
def update_dropdown2(dd_value, json_data):
    if dd_value is not None and json_data is not None:
        df = pd.read_json(json_data, orient='split')
        levels = set(df[dd_value])
        return [{'label': i, 'value': i} for i in levels]
    else:
        return []


@app.callback(Output('group2_dd', 'options'),
                [Input('group2_selection', 'value'),
                 Input('result_df', 'children')])
def update_dropdown2(dd_value, json_data):
    if dd_value is not None and json_data is not None:
        df = pd.read_json(json_data, orient='split')
        levels = set(df[dd_value])
        return [{'label': i, 'value': i} for i in levels]
    else:
        return []


@app.callback([Output('threshold_slider', 'min'),
                Output('threshold_slider', 'max'),
               Output('threshold_slider', 'step'),
               Output('threshold_slider', 'marks')],
                [Input('predicted_selection', 'value'),
                 Input('treshold_levels', 'value'),
                 Input('result_df', 'children')])
def update_threshold(dd_value, n_thr_value, json_data):
    if dd_value is not None and json_data is not None:
        df = pd.read_json(json_data, orient='split')
        levels = set(df[dd_value])
        minimum = round(min(levels), 1)
        maximum = round(max(levels), 1)
        # print(n_thr_value)
        # print(type(n_thr_value))
        step = (maximum - minimum)/int(n_thr_value)
        marks = []
        i = minimum
        while i <= maximum:
            marks.append(i)
            i += step
        return minimum, maximum, step, {str(round(val, 2)): str(round(val, 2)) for val in marks}
    else:
        return -1, 1, 1, {'0': '0'}


@app.callback(
    Output('cm_graph', 'figure'),
    [Input('threshold_slider', 'value'),
     Input('real_selection', 'value'),
     Input('predicted_selection', 'value'),
     Input('group1_selection', 'value'),
     Input('group1_dd', 'value'),
     Input('group2_selection', 'value'),
     Input('group2_dd', 'value'),
     Input('result_df', 'children')])
def update_graph(trsh_value, real_column, predicted_model, group1, group1_value, group2, group2_value, json_data):
    if json_data is None or predicted_model is None:
        return {
            'data': [],
            'layout':
                go.Layout(
                    title='Results',
                    barmode='relative',
                    yaxis={'title': '%', 'range': [0, 100]})
        }
    else:
        df = pd.read_json(json_data, orient='split')
    df_plot = df[~df[real_column].isnull()].copy() # removes rows if no values are labeled in real
    # get real labels
    real_labels = list(set(df_plot[real_column]))
    real_labels.sort()
    if group1_value is not None:
        if len(group1_value) > 0:
            df_plot = df_plot[df_plot[group1].isin(group1_value)]
    if group2_value is not None:
        if len(group2_value) > 0:
            df_plot = df_plot[df_plot[group2].isin(group2_value)]

    # print(df_plot.shape)
    model_bin = bin_recode(df_plot[predicted_model], trsh_value, real_labels)
    model_cm = conf_matrix(list(df_plot[real_column]), model_bin, predicted_model, real_labels)
    labels = ['Sensitivity', 'Specificity', 'Accuracy']
    correct = 'Correct: '+ model_cm['Correct'].astype('str')+ ' from ' + model_cm['Total'].astype('str')
    trace1 = go.Bar(x=labels, y=model_cm['%'], name='values', hovertext=correct)
    trace2 = go.Scatter(x = labels, y=[95, 95, 95], mode='lines', name='95%')
    trace3 = go.Scatter(x=labels, y=[50, 50, 50], mode='lines', name='50%')


    # fig.update_layout(barmode='relative', title_text='Results')
    return {
        'data': [trace1, trace2, trace3],
        'layout':
        go.Layout(
            title='Results',
            barmode='relative',
            yaxis={'title': '%', 'range': [0, 100]})
    }


@app.callback(
    Output('value_table', 'children'),
    [Input('threshold_slider', 'value'),
     Input('real_selection', 'value'),
     Input('predicted_selection', 'value'),
     Input('group1_selection', 'value'),
     Input('group1_dd', 'value'),
     Input('group2_selection', 'value'),
     Input('group2_dd', 'value'),
     Input('result_df', 'children')])
def update_values(trsh_value, real_column, predicted_model, group1, group1_value, group2, group2_value, json_data):
    if json_data is None or predicted_model is None:
        return html.Table(
            # Header
            [html.Tr()] +
            # Body
            [html.Tr()]
        )
    else:
        df = pd.read_json(json_data, orient='split')
    df_plot = df[~df[real_column].isnull()].copy()  # removes rows if no values are labeled in real
    # get real labels
    real_labels = list(set(df_plot[real_column]))
    real_labels.sort()
    if group1_value is not None:
        if len(group1_value) > 0:
            df_plot = df_plot[df_plot[group1].isin(group1_value)]
    if group2_value is not None:
        if len(group2_value) > 0:
            df_plot = df_plot[df_plot[group2].isin(group2_value)]

    model_bin = bin_recode(df_plot[predicted_model], trsh_value, real_labels)
    model_cm = conf_matrix(list(df_plot[real_column]), model_bin, predicted_model, real_labels)
    values_t = values_table(model_cm)

    return generate_table(values_t)


@app.callback(
    Output('cm_plot', 'figure'),
    [Input('threshold_slider', 'value'),
     Input('real_selection', 'value'),
     Input('predicted_selection', 'value'),
     Input('group1_selection', 'value'),
     Input('group1_dd', 'value'),
     Input('group2_selection', 'value'),
     Input('group2_dd', 'value'),
     Input('result_df', 'children')])
def update_graph(trsh_value, real_column, predicted_model, group1, group1_value, group2, group2_value, json_data):
    if json_data is None or predicted_model is None:
        return {
            'data': [],
            'layout':
                go.Layout(
                    title='Results',
                    barmode='relative',
                    yaxis={'title': '%', 'range': [0, 100]})
        }
    else:
        df = pd.read_json(json_data, orient='split')
    df_plot = df[~df[real_column].isnull()].copy() # removes rows if no values are labeled in real
    # get real labels
    real_labels = list(set(df_plot[real_column]))
    real_labels.sort()
    if group1_value is not None:
        if len(group1_value) > 0:
            df_plot = df_plot[df_plot[group1].isin(group1_value)]
    if group2_value is not None:
        if len(group2_value) > 0:
            df_plot = df_plot[df_plot[group2].isin(group2_value)]

    traces = []
    x_ranges = []
    for i in df_plot[real_column].unique():
        df_by_label = df_plot[df_plot[real_column] == i]
        x_range = list(range(0, df_by_label.shape[0]))
        x_ranges.append(x_range)
        traces.append(dict(
            x=x_range,
            y=df_by_label[predicted_model],
            mode='markers',
            opacity=0.5,
            marker={
                'size': 8,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=i
        ))
        traces.append(go.Scatter(x=x_range,
                                 y=[mean(df_by_label[predicted_model])] * len(x_range),
                                 mode='lines',
                                 name=i + ' average'))
        # print(f'{i} x_range is: {x_range}; average is {mean(x_range)}; len is: {len(x_range)}')
    traces.append(go.Scatter(x=max(x_ranges),
                             y=[trsh_value]*len(max(x_ranges)),
                             mode='lines',
                             name='threshold'))

    return {
        'data': traces,
        'layout': dict(
            xaxis={'title':'Distribution plot', 'range':x_range},
            yaxis={'title': 'Score', 'range': [-1, 1]},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 1, 'y': 1},
            hovermode='closest',
            transition = {'duration': 500},
        )
    }



@app.callback(
    Output('metric_table', 'children'),
    [Input('threshold_slider', 'value'),
     Input('real_selection', 'value'),
     Input('predicted_selection', 'value'),
     Input('group1_selection', 'value'),
     Input('group1_dd', 'value'),
     Input('group2_selection', 'value'),
     Input('group2_dd', 'value'),
     Input('result_df', 'children')])
def update_values(trsh_value, real_column, predicted_model, group1, group1_value, group2, group2_value, json_data):
    if json_data is None or predicted_model is None:
        return html.Table(
            # Header
            [html.Tr()] +
            # Body
            [html.Tr()]
        )
    else:
        df = pd.read_json(json_data, orient='split')
    df_plot = df[~df[real_column].isnull()].copy()  # removes rows if no values are labeled in real
    # get real labels
    real_labels = list(set(df_plot[real_column]))
    real_labels.sort()
    if group1_value is not None:
        if len(group1_value) > 0:
            df_plot = df_plot[df_plot[group1].isin(group1_value)]
    if group2_value is not None:
        if len(group2_value) > 0:
            df_plot = df_plot[df_plot[group2].isin(group2_value)]

    model_bin = bin_recode(df_plot[predicted_model], trsh_value, real_labels)
    model_cm = conf_matrix(list(df_plot[real_column]), model_bin, predicted_model, real_labels)
    metric_t = metric_table(model_cm)

    return generate_table(metric_t)




if __name__ == '__main__':
    app.run_server(debug=True)
    # app.run_server(debug=False, host='00.000.000.000', port = 8080) # for external access

