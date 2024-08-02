# Time Series Feature Extraction and Forecasting

## Overview

This project provides a web application for uploading time series data, performing feature extraction, and generating forecasts using an LSTM model. The application leverages the `tsfresh` library for feature extraction and TensorFlow for time series forecasting.

## Project Structure

- `models.py`: Defines the `TimeSeriesData` model used to store time series data and extracted features.
- `forms.py`: Contains the form for uploading CSV files.
- `views.py`: Includes the core logic for handling file uploads, feature extraction, and forecasting.
- `admin.py`: Configures the Django admin interface for managing time series data.
- `templates/`: Contains HTML templates for the web interface.
- `trained_model/`: Directory where the LSTM model (`lstm_model.h5`) is stored.
- `backbone/`: Contains scripts for feature extraction and model training.

## Getting Started

### Prerequisites

- Python 3.x
- Django
- TensorFlow
- tsfresh
- scikit-learn
- pandas
- matplotlib

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Has-sh/TimeSeriesFeatureExtraction.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd TimeSeriesFeatureExtraction
   ```

3. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `.\venv\Scripts\activate`
   ```

4. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Navigate to the Django app directory:**
   ```bash
   cd TSFresh/featureExtractionProject
   ```

6. **Run the migrations:**
   ```bash
   python manage.py makemigrations timeseriesapp
   python manage.py migrate
   ```

7. **Start the development server:**
   ```bash
   python manage.py runserver
   ```

### Usage

- **Upload CSV Files:**
  The web application accepts `.csv` files with columns `timestamp`, `id`, and `feature_...` for feature extraction.

- **Feature Extraction:**
  After uploading the CSV wait for the success page to load and click on the url to navigate to the page where you can view and filter the extracted features.

- **Forecasting:**
  Use the forecast url to view the forcasted graphs for your timeseries.

## Usage

### Upload Time Series Data

1. Navigate to the web application.
2. Use the upload form to submit a CSV file.

### CSV File Format

The CSV file should contain the following columns:
- `timestamp`: A timestamp indicating the time of the data entry.
- `id`: A unique identifier for each time series.
- `feature_...`: Columns representing different features.

**Example CSV:**

```csv
timestamp,id,feature_1,feature_2 ...
2024-01-01 00:00:00,1,0.5,0.2
2024-01-01 00:01:00,1,0.6,0.3
...
```

### Feature Extraction

The uploaded CSV file will be processed, and features will be extracted using `tsfresh`. These features will be stored in the database.

### Forecasting

The application uses an LSTM model to forecast future values based on the historical data. The forecasts will be plotted and displayed.

### Plotting

The forecasting results are visualized using Matplotlib and displayed on the webpage. Each feature's historical data and forecasted data will be plotted.

## Training the Model

The `backbone/scripts.py` script extracts features from raw data and trains an LSTM model. Ensure the following script is executed to prepare the data and train a new model if needed:

```bash
python backbone/scripts.py
```

## Troubleshooting

- **No such table: timeseriesapp_timeseriesdata**: Run the migrations to create the required database tables:
  ```bash
  python manage.py makemigrations
  python manage.py migrate
  ```

- **IndexError or OperationalError**: Ensure that the uploaded CSV file follows the required format.
