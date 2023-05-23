import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle

def main():
    df = pd.read_csv('/app/Q4.csv')
    print(df.columns)
    # Split the data into features (X) and target variable (y)
    X = df.drop(['Revenue'], axis=1)
    y = df['Revenue']

    # Manual encoding for 'Month' column
    month_mapping = {'Feb': 2, 'Mar': 3, 'May': 5, 'June': 6,
                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    X['Month'] = X['Month'].map(month_mapping)

    # Manual encoding for 'VisitorType' column
    visitor_mapping = {'New_Visitor': 0, 'Returning_Visitor': 1, 'Other': 2}
    X['VisitorType'] = X['VisitorType'].map(visitor_mapping)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Applying Standard Normalization method (Z-Score Equation):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handling missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train_scaled)
    X_test = imputer.transform(X_test_scaled)


    # Create a Random Forest Classifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy*100, '%')

    file_path = '/app/online_shoppers_model.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    main()