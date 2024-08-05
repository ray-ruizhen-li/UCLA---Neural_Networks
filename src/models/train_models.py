from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


import pickle

def neural_networks(X, y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

   # fit/train the model. Check batch size.
    MLP = MLPClassifier(hidden_layer_sizes=(3), batch_size=50, max_iter=100, random_state=123)
    MLP.fit(X_train,y_train)
    
    return MLP,X_test_scaled, y_test

# Function to train the model
def train_logistic_regression(X, y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the logistic regression model
    model = LogisticRegression().fit(X_train_scaled, y_train)
    
    # Save the trained model
    with open('./src/models/logistic_regression.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model, X_test_scaled, y_test

def random_forest(X,y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the random forest model
    rfmodel = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_features=None)
    rfmodel.fit(X_train, y_train)
    ypred = rfmodel.predict(X_test)
    return rfmodel, X_test_scaled, y_test

def decision_tree(X,y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
     # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # create an instance of the class
    dt = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=457)
    # train the model
    dtmodel = dt.fit(X_train,y_train)
    # make predictions using the test set
    ytest_pred = dtmodel.predict(X_test)
    # evaluate the model
    test_mae = mean_absolute_error(ytest_pred, y_test)
    print("Decision Tree test error is: ",test_mae)
    # make predictions on train set
    ytrain_pred = dtmodel.predict(X_train)
    # evaluate the model
    train_mae = mean_absolute_error(ytrain_pred, y_train)
    print("Decision Tree Train error is: ",train_mae)
    return dtmodel, X_test_scaled, y_test