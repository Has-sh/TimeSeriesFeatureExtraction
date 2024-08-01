import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters, ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute

def main():    
    X_train = pd.read_csv('data/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt', sep=r'\s+', header=None, engine='python')
    y_train = pd.read_csv('data/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt', sep=r'\s+', header=None, engine='python')
    subject_train = pd.read_csv('data/UCI HAR Dataset/UCI HAR Dataset/train/subject_train.txt', sep=r'\s+', header=None, engine='python')

    train_data = pd.concat([subject_train, y_train, X_train], axis=1)
    train_data.columns = ['subject'] + ['activity'] + ['feature_' + str(i) for i in range(X_train.shape[1])]
    train_data['id'] = train_data['subject'].astype(str) + '_' + train_data['activity'].astype(str)
    train_data['time'] = train_data.groupby(['subject', 'activity']).cumcount()
    train_data.to_csv('train_data.csv', index=False)

    
    melted_train_data = train_data.melt(id_vars=['subject', 'activity', 'id', 'time'], var_name='feature', value_name='value')
    melted_train_data['time'] = melted_train_data.groupby(['subject', 'activity', 'feature']).cumcount()
    melted_train_data['id'] = melted_train_data['id'].astype(str) + '_' + melted_train_data['feature']

    melted_train_data.to_csv('melted_train_data.csv', index=False)  
    
    extracted_features = extract_features(melted_train_data, column_id='id', column_sort='time', column_value='value', default_fc_parameters=MinimalFCParameters(),impute_function=impute)
    
    extracted_features.to_csv('extracted_features.csv', index=False)

if __name__ == '__main__':
    main()
