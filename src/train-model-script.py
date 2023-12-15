# import libraries
import mlflow
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

def main(args):
    # read data
    df = get_data(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    model = train_model( X_train, y_train, args.n_estimators, args.max_depth)

    # evaluate model
    eval_model(model, X_test, y_test)

# function that reads the data
def get_data(args):
    print("Reading data from dataset")
    df = pd.read_csv(args)
    df= df.iloc[:768,:10]
    return df

# function that splits the data
def split_data(df):
    print("Splitting data...")
    X, y = df[['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area',
       'Overall Height', 'Orientation', 'Glazing Area',
       'Glazing Area Distribution',]].values, df['Heating Load'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    return X_train, X_test, y_train, y_test

# function that trains the model
def train_model(X_train, y_train, n_estimators, max_depth):
    print("Training model...")
    
    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', MaxAbsScaler()),  # Normalise data
        ('model', RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth))  # RandomForest model
    ])
    
    # Train the model
    model = pipeline.fit(X_train, y_train)
    
    return model
# function that evaluates the model
def eval_model(model, X_test, y_test):
    # calculate predictions
    y_pred = model.predict(X_test)
    
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', rmse)
    mlflow.log_metric("RMSE", rmse)

    # calculate R-squared
    r2 = r2_score(y_test, y_pred)
    print('R-squared: ', r2)
    mlflow.log_metric("R-squared", r2)

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--n_estimators", dest='n_estimators', type=int, default=100)
    parser.add_argument("--max_depth", dest='max_depth', type=int, default=None)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
