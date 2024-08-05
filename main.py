# Created by RayLi
from src.data.make_dataset import load_and_preprocess_data
from src.feature_engineering.build_features import create_dummy_vars
from src.models.train_models import neural_networks
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
# cross validation using cross_val_score
from sklearn.model_selection import cross_val_score

# Import GridSearch CV
from sklearn.model_selection import GridSearchCV


#from src.models.predict_model import NN_evaluate_model
#from src.visulization.visulize import loss_curve

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "src/data/raw/Admission.csv"
    df = load_and_preprocess_data(data_path)

    # Create dummy variables and separate features and target
    X, y = create_dummy_vars(df)

    # Train the logistic regression model
    model, X_test_scaled, y_test = neural_networks(X, y)

    # Evaluate the model
    #accuracy = NN_evaluate_model(model, X_test_scaled, y_test)
    #print(f"Logistic Regression Accuracy: {accuracy}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

   # fit/train the model. Check batch size.
    MLP = MLPClassifier(hidden_layer_sizes=(3), batch_size=50, max_iter=100, random_state=123)
    MLP.fit(X_train,y_train)
    
    # Plot the loss curve
    #loss_curve= loss_curve(model)    
    loss_values = MLP.loss_curve_

    # Plotting the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    MLP.get_params
    # we will try different values for hyperparemeters
    params = {'batch_size':[20, 30, 40, 50],
          'hidden_layer_sizes':[(2,),(3,),(3,2)],
         'max_iter':[50, 70, 100]}
    # create a grid search
    grid = GridSearchCV(MLP, params, cv=10, scoring='accuracy')
    #grid.fit(x, y)
    #param = grid.best_params_
    #best_score = grid.best_score_
    grid.estimator