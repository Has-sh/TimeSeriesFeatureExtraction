from django.shortcuts import render, redirect
from .forms import TimeSeriesDataForm
from .models import TimeSeriesData
from tsfresh import extract_features
import pandas as pd
import csv
import io
from io import BytesIO 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import base64
import matplotlib
import numpy as np
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

matplotlib.use('agg')

df=None
extended_df=None

#func def below:
def get_all_column_names():
    time_series_data = TimeSeriesData.objects.all()
    
    column_names = set()  
    
    for data in time_series_data:
        if data.features:
            features = data.features
            column_names.update(features.keys())
    
    return list(column_names)

def upload_time_series(request):
    global df
    global extended_df
    if request.method == 'POST':
        form = TimeSeriesDataForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = form.cleaned_data['csv_file']

            # Read the uploaded CSV file
            data_values = []
            csv_data = csv_file.read().decode('utf-8')
            csv_reader = csv.reader(io.StringIO(csv_data))
            column_names = next(csv_reader)
            
            id_col_index = column_names.index('id')
            timestamp_col_index = column_names.index('timestamp')
            value_col_indices = [column_names.index(col_name) for col_name in column_names if col_name.startswith('feature')]

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

            # Process data similar to the main function
            melted_train_data = df.melt(id_vars=['id', 'timestamp'], var_name='feature', value_name='value')
            melted_train_data['timestamp'] = melted_train_data.groupby(['id', 'feature']).cumcount()
            melted_train_data['id'] = melted_train_data['id'].astype(str) + '_' + melted_train_data['feature']

            # Extract features using tsfresh
            extracted_features = extract_features(
                melted_train_data,
                column_id='id',
                column_sort='timestamp',
                column_value='value',
                default_fc_parameters=MinimalFCParameters(),
                impute_function=impute
            )

            # Save extracted features to the database
            for index, row in extracted_features.iterrows():
                id_value = row.name  # Use index or any unique identifier for the id
                features_data = row.to_dict()

                # Create or update the TimeSeriesData entry
                time_series_entry, created = TimeSeriesData.objects.update_or_create(
                    timestamp=id_value,
                    defaults={'features': features_data}
                )
                
            return render(request, 'upload_success.html')
    else:
        form = TimeSeriesDataForm()
    return render(request, 'upload_form.html', {'form': form})


def search_results(request):
    column_name_query = request.GET.get('column_name', '')
        
    time_series_data = TimeSeriesData.objects.all()
    
    column_data = []
    for data in time_series_data:
        if data.features and column_name_query in data.features:
            column_data.append({
                'timestamp': data.timestamp,
                'column_value': data.features[column_name_query],
            })
    

    all_column_names = get_all_column_names()

    context = {
        'column_data': column_data,
        'column_name_query': column_name_query,
        'all_column_names': all_column_names
    }
    
    return render(request, 'search_results.html', context)


def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def plot_graph(request):
    # Fetch data from the database
    time_series_data = TimeSeriesData.objects.all()
    data_list = []

    for data in time_series_data:
        features = data.features
        if features:
            feature_values = list(features.values())
            data_list.append(feature_values)

    if not data_list:
        return render(request, 'plot_graph.html', {'plot_images': None, 'error': 'No data available'})

    df = pd.DataFrame(data_list)

    # Normalize features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df)

    # Prepare data for LSTM
    time_steps = 50
    X, y = create_sequences(scaled_features, time_steps)

    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Load the trained model
    model_path = 'trained_model/lstm_model.h5'
    loaded_model = load_model(model_path)

    # Generate future forecasts
    future_steps = 500
    last_sequence = scaled_features[-time_steps:].reshape(1, time_steps, scaled_features.shape[1])
    forecast = []

    for _ in range(future_steps):
        pred = loaded_model.predict(last_sequence)[0]
        forecast.append(pred)
        last_sequence = np.concatenate([last_sequence[:, 1:, :], pred.reshape(1, 1, -1)], axis=1)

    # Convert to DataFrame and scale back
    forecast = np.array(forecast).reshape(-1, scaled_features.shape[1])
    forecast_rescaled = scaler.inverse_transform(forecast)

    # Prepare time indices for plotting
    historical_time = np.arange(len(scaled_features))
    forecast_time = np.arange(len(scaled_features), len(scaled_features) + future_steps)

    # Plot each feature individually with a line graph
    plot_images = []
    for i in range(scaled_features.shape[1]):
        plt.figure(figsize=(14, 7))

        # Plot historical data for the feature
        plt.plot(historical_time, scaler.inverse_transform(scaled_features)[:, i], label=f'Historical Data - Feature {i+1}')

        # Plot forecasted data for the feature
        plt.plot(forecast_time, forecast_rescaled[:, i], label=f'Forecasted Data - Feature {i+1}', linestyle='--')

        plt.title(f'Time Series Forecast for Feature {i+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Feature Value')
        plt.legend()

        plot_image = BytesIO()
        plt.savefig(plot_image, format='png')
        plot_image.seek(0)
        plot_data = base64.b64encode(plot_image.read()).decode('utf-8')
        plt.close()
        
        plot_images.append(plot_data)

    context = {
        'plot_images': plot_images
    }

    return render(request, 'plot_graph.html', context)