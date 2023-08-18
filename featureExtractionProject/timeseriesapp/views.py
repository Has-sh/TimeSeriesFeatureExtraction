from django.shortcuts import render, redirect
from .forms import TimeSeriesDataForm
from .models import TimeSeriesData
from tsfresh import extract_features
import pandas as pd
import csv
import io
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

matplotlib.use('agg')
df=None
extended_df=None
#func def below:
def get_all_column_names():
    time_series_data = TimeSeriesData.objects.all()
    
    column_names = set()  
    
    for data in time_series_data:
        if data.features and 'features' in data.features:
            features = data.features['features']
            column_names.update(features.keys())
    
    return list(column_names)

#requirment(id,timestamp,name of extract feature column always starts with value)
#views def below:
def upload_time_series(request):
    global df
    global extended_df
    if request.method == 'POST':
        form = TimeSeriesDataForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = form.cleaned_data['csv_file']

            data_values = []
            csv_data = csv_file.read().decode('utf-8')
            csv_reader = csv.reader(io.StringIO(csv_data))
            column_names = next(csv_reader)
            
            id_col_index = column_names.index('id')
            timestamp_col_index = column_names.index('timestamp')
            value_col_indices = [column_names.index(col_name) for col_name in column_names if col_name.startswith('value')]

            for row in csv_reader:
                timestamp = row[timestamp_col_index]
                id = row[id_col_index]
                value_dict = {'id': id, 'timestamp': timestamp}
                for col_index in value_col_indices:
                    col_name = column_names[col_index]
                    value = float(row[col_index])
                    value_dict[col_name] = value
                data_values.append(value_dict)
                
            df = pd.DataFrame(data_values)

            extracted_features = extract_features(df, column_id='id', column_sort='timestamp')#add a loader

            columns_to_delete = []
            for column_name in extracted_features.columns:
                if all(pd.isna(extracted_features[column_name]) | (extracted_features[column_name] == 0) | (extracted_features[column_name] == -0) | (extracted_features[column_name]==np.inf)| (extracted_features[column_name]==-np.inf)):
                    columns_to_delete.append(column_name)
            extracted_features = extracted_features.drop(columns=columns_to_delete)

            #new code
            id_features = {}

            for index, row in extracted_features.iterrows():
                id = row.name
                id_features[id] = row

            new_data_values = []

            for row in data_values:
                id = row['id']
                timestamp = row['timestamp']
                features = id_features.get(id, pd.Series())

                new_row = {'id': id, 'timestamp': timestamp}
                new_row.update(features)
            
                new_data_values.append(new_row)

            extended_df = pd.DataFrame(new_data_values)
            #new code end 

            for index, row in extracted_features.iterrows():
                id = row.name
                filtered_df = df[df['id'] == id]
                
                if not filtered_df.empty:
                    features = {feature_name: str(row[feature_name]) for feature_name in row.index}
                    feature_1={'features': features,}
                    obj, created = TimeSeriesData.objects.update_or_create(
                        id=id,
                        defaults={'timestamp':timestamp,'features':feature_1}
                    )
            

            return render(request, 'upload_success.html')  
    else:
        form = TimeSeriesDataForm()
    return render(request, 'upload_form.html', {'form': form})


def search_results(request):
    column_name_query = request.GET.get('column_name')
    
    time_series_data = TimeSeriesData.objects.all()
    
    column_data = []
    for data in time_series_data:
        if data.features and column_name_query in data.features.get('features', {}):
            column_data.append({
                'id': data.id,
                'column_value': data.features['features'][column_name_query],
            })

    all_column_names = get_all_column_names()

    context = {
        'column_data': column_data,
        'column_name_query': column_name_query,
        'all_column_names': all_column_names
    }
    
    return render(request, 'search_results.html', context)

def plot_graph(request):

    target_series = df['value_G'].values

    X_scaled = extended_df.drop(['id', 'timestamp'], axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, target_series, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy='constant', fill_value=0)

    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_imputed, y_train)

    y_pred = model.predict(X_test_imputed)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test)), y_test, label='True Values')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.title('True vs Predicted Values')
    plt.xticks(rotation=45)
    
    plot_image = io.BytesIO()
    plt.savefig(plot_image, format='png')
    plot_image.seek(0)
    plot_data = base64.b64encode(plot_image.read()).decode('utf-8')
    plt.close()

    context = {
        'plot_data': plot_data
    }

    return render(request, 'plot_graph.html', context)
