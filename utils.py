import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

def get_training_data():
    labels_data = pd.read_csv('data/labels.csv')

    # Read the training.csv file
    training_data = pd.read_csv('data/training.csv')
    training_data = pd.merge(training_data, labels_data, on='label')
    return training_data

def get_validation_data():
    labels_data = pd.read_csv('data/labels.csv')
    # Read the validations.csv file
    validation_data = pd.read_csv('data/validations.csv')
    validation_data = pd.merge(validation_data, labels_data, on='label')
    return validation_data

def read_labelled_data(filename):
    """
        Read the labeled data from the file and return it as a pandas dataframe
    """
    df = pd.read_csv(filename)
    df['Type of Clause'] = df['Type of Clause'].str.casefold()
    df['predicted_clause_type'] = df['predicted_clause_type'].str.casefold()
    df['Degree of Unfairness'] = df['Degree of Unfairness'].astype(int)
    df['predicted_degree_of_unfairness'] = df['predicted_degree_of_unfairness'].fillna(0).astype(int)
    return df

def get_accuracy_stats(df):
    """
        Calculate the accuracy of the model
        Returns a dictionary with the accuracy of the model
    """
    
    type_of_clause=df['Type of Clause']
    pred_type_of_clause=df['predicted_clause_type']

    degree_of_unfairness=df['Degree of Unfairness']
    pred_degree_of_unfairness=df['predicted_degree_of_unfairness']

    
    accuracy_of_type_of_clause = (type_of_clause==pred_type_of_clause).mean()
    accuracy_of_degree_of_unfairness = (degree_of_unfairness==pred_degree_of_unfairness).mean()

    combined_labels = (type_of_clause==pred_type_of_clause) & (degree_of_unfairness==pred_degree_of_unfairness)
    combined_score = combined_labels.mean()

    stats = {"Type of Clause": accuracy_of_type_of_clause, \
            "Degree of Unfairness": accuracy_of_degree_of_unfairness, \
            "Combined": combined_score}
    
    return stats

