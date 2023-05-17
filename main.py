import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def main():
    # reading csv file
    data = pd.read_csv('/Users/aadyant/Desktop/GLODAPOCEANDATANEW.csv')

    # initializing input variables
    inputs = data[['G2temperature', 'G2salinity', 'G2oxygen', 'G2aou', 'G2talk', 'G2cfc11', 'G2cfc12', 'G2phosphate', 'G2pcfc12', 'G2nitrate', 'G2silicate']].values

    # initializing output variable
    outputs = data['G2phtsinsitutp'].values

    # making testing data 20% of the available and the rest is for training
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # creating linear regression model and training it with the training data
    model = LinearRegression()
    model.fit(train_inputs, train_outputs)

    # making predictions based on testing input
    predictions = model.predict(test_inputs)

    # determining r^2 value for this seed's model
    r2 = r2_score(test_outputs, predictions)
    print("R^2 Score:", r2)

    # creating new column named Predicted_pH and filling it with NaN values
    data['Predicted_pH'] = np.nan

    # iterating through the test_outputs array and setting the respective index equal to the predicted pH
    data.loc[data.index[pd.Series(test_outputs).index], 'Predicted_pH'] = predictions

    # saving data to csv file
    data.to_csv('/Users/aadyant/Desktop/GLODAPOCEANDATANEW.csv', index=False)

if __name__ == '__main__':
    main()
