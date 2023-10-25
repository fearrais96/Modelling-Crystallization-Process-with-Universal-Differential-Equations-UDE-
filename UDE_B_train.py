"""
Author: Fernando Arrais Romero Dias Lima

Training universal differential equations (UDE) for modelling the potassium sulfate batch
colling crystallization in water, considering nucleation and crystal growth
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Saving the experimental data in a variable

experimental_data = pd.read_csv("Cresc_Nucl_K2SO4_7exps_train.csv")

r3_exp = np.zeros(len(experimental_data))

for i in range(len(r3_exp)):
    r3_exp[i] = experimental_data.iloc[i, 6] / experimental_data.iloc[i, 3]

# Separate the data for each of the 10 experimental batches

batch1 = (experimental_data[(experimental_data["Experiment"] == 0)]).values
batch2 = (experimental_data[(experimental_data["Experiment"] == 1)]).values
batch3 = (experimental_data[(experimental_data["Experiment"] == 2)]).values
batch4 = (experimental_data[(experimental_data["Experiment"] == 3)]).values
batch5 = (experimental_data[(experimental_data["Experiment"] == 4)]).values
batch6 = (experimental_data[(experimental_data["Experiment"] == 5)]).values
batch7 = (experimental_data[(experimental_data["Experiment"] == 6)]).values


# Defining the activation functions

def ReLU(x):
    if x > 0:
        return x
    return 0


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def NN_B(input, neurons_HiddenLayer, wH, bH, wOut, bOut):
    """
    Neural network to predict the nucleation rate (B)

    :param input: normalized inputs of the neural network (mu3, concentration, and temperature)
    :param neurons_HiddenLayer: number of neurons in the hidden layer
    :param wH: weights matrix of the neurons between the input and the hidden layers
    :param bH: biases vector of the hidden layer
    :param wOut: weights matrix of the neurons between the hidden and the output layers
    :param bOut: biases vector of the output layer
    :return: predicted nucleation rate (B)
    """

    pre_ActivationH = np.dot(input, wH) + bH

    post_ActivationH = np.zeros(neurons_HiddenLayer)

    for m in range(neurons_HiddenLayer):
        post_ActivationH[m] = tanh(pre_ActivationH[m])

    prediction = np.dot(post_ActivationH, wOut) + bOut

    return prediction[0]


def modelagem_PB(entrada, t, T, wH_B, bH_B, wOut_B, bOut_B):
    """
    This function defines the population balance model for nucleation and crystal growth
    of the potassium sulfate batch crystallization in water. Details about the model can
    be found in the work of Moraes et al. (2023) (doi:10.1021/acs.iecr.3c00739).

    This model is solved using the universal differential equation approach (UDE), where
    the nucleation rate (B) and the growth rate (G) are predicted by neural networks.

    :param entrada: state veriables of the model (the first four moments and the concentration)
    :param t: time vector
    :param T: process temperature
    :param wH_B: weights matrix of the neurons between the input and the hidden layers for
    the neural network to predict B
    :param bH_B: biases vector of the hidden layer for the neural network to predict B
    :param wOut_B: weights matrix of the neurons between the hidden and the output layers for
    the neural network to predict B
    :param bOut_B: biases vector of the output layer for the neural network to predict B
    :return: derivative values for the five states
    """

    # Valores dos parâmetros usados
    kg = 37.79
    Eag_R = 4.865e3
    cg = 1.623
    gamma = 23.9
    kv = 1.349
    ro = 2.658  # [roc] = g/cm^3

    # Valores de entrada para mu0, mu1, mu2, mu3 e concentração
    mu0 = entrada[0]  # #/cm3
    r_1 = entrada[1]
    r_2 = entrada[2]
    r_3 = entrada[3]
    C = entrada[4]  # g/cm3

    Ceq = -686.2686 + 3.579165 * (T + 273.15) - 0.00292874 * (T + 273.15) ** 2  # g/L

    S = C * 1e3 - Ceq

    mu0_norm = (mu0 - np.min(experimental_data.iloc[:, 3])) / (
            np.max(experimental_data.iloc[:, 3]) - np.min(experimental_data.iloc[:, 3]))

    mu3_norm = (r_3 - np.min(r3_exp)) / (
            np.max(r3_exp) - np.min(r3_exp))

    T_norm = (T - np.min(experimental_data.iloc[:, 1])) / (
            np.max(experimental_data.iloc[:, 1]) - np.min(experimental_data.iloc[:, 1]))

    C_norm = (C - np.min(experimental_data.iloc[:, 7])) / (
            np.max(experimental_data.iloc[:, 7]) - np.min(experimental_data.iloc[:, 7]))

    input_B = np.array([mu0_norm, mu3_norm, C_norm, T_norm])

    B = NN_B(input_B, len(bH_B), wH_B, bH_B, wOut_B, bOut_B)

    G_0 = kg * np.exp(-Eag_R / (T + 273.15)) * (S ** 2) ** (cg / 2)

    dmi0dt = B * mu0
    dmi1dt = G_0 * (1 + gamma * r_1 * 1e-4) * 1e4 - B * r_1 * 1e-4 * 1e4
    dmi2dt = 2.0 * G_0 * (r_1 * 1e-4 + gamma * r_2 * 1e-8) * 1e8 - B * r_2 * 1e-8 * 1e8
    dmi3dt = 3.0 * G_0 * (r_2 * 1e-8 + gamma * r_3 * 1e-12) * 1e12 - B * r_3 * 1e-12 * 1e12
    dcdt = -3.0 * ro * kv * G_0 * (r_2 * 1e-8 + gamma * r_3 * 1e-12) * mu0

    return [dmi0dt, dmi1dt, dmi2dt, dmi3dt, dcdt]


def integrator_PB(y0, t, temperature_range, wH_B, bH_B, wOut_B, bOut_B):
    """
    This function is used to integrate the the population balance model for nucleation and crystal growth
    of the potassium sulfate batch crystallization in water.

    :param y0: process initial conditions (initial values of the four moments and the concentration)
    :param t: time vector of the batch
    :param temperature_range: vector containing the process temperature of the process
    :param wH_B: weights matrix of the neurons between the input and the hidden layers for
    the neural network to predict B
    :param bH_B: biases vector of the hidden layer for the neural network to predict B
    :param wOut_B: weights matrix of the neurons between the hidden and the output layers for
    the neural network to predict B
    :param bOut_B: biases vector of the output layer for the neural network to predict B
    :return: values of the moments and concentration for the initial conditions, temperatur,and
    time specifieds
    """
    result = np.zeros([len(temperature_range), 5])

    for i in range(5):
        result[0, i] = y0[i]

    for i in range(len(t) - 1):

        ts = [t[i], t[i + 1]]
        y = odeint(modelagem_PB, y0, ts, args=(temperature_range[i + 1], wH_B, bH_B, wOut_B, bOut_B,), atol=1e-8, rtol=1e-8)

        for j in range(5):
            y0[j] = y[1, j]

        for j in range(5):
            result[i + 1, j] = y[1, j]

    return result


def f_obj(parameters_NN, number_inputs_B, neurons_HidenLayer_B,
          number_outputs_B):
    """
    Objective function to be minimized in the optimization problem. This function is the
    sum of the mean squared errors between the experimental and predicted values of the
    state variables. The mean squared error is calculated for the normalized variables,
    using the MinMaxScaler from scikitlearn.

    :param parameters_NN: vector containing the weights and biases used in the neural networks
    :param number_inputs_B: number of inputs of the neural network to predict B
    :param neurons_HidenLayer_B: number of hidden layers of the neural network to predict B
    :param number_outputs_B: number of outputs of the neural network to predict B
    :return: sum of the normilized mean squared errors between the predicted and the experimental values
    of th state variables
    """

    # Initially, te weight matrices and the biases vectors are created for each neural network,
    # using the vector containing the weights and biases used in the neural networks

    wH_B = np.zeros([number_inputs_B, neurons_HidenLayer_B])
    bH_B = np.zeros(neurons_HidenLayer_B)
    wOut_B = np.zeros([neurons_HidenLayer_B, number_outputs_B])
    bOut_B = np.zeros(number_outputs_B)

    count3 = 0

    for m in range(number_inputs_B):
        for n in range(neurons_HidenLayer_B):
            wH_B[m][n] = parameters_NN[count3]

            count3 += 1

    for m in range(neurons_HidenLayer_B):
        bH_B[m] = parameters_NN[count3]

        count3 += 1

    for m in range(neurons_HidenLayer_B):
        for n in range(number_outputs_B):
            wOut_B[m][n] = parameters_NN[count3]

            count3 += 1

    for m in range(number_outputs_B):
        bOut_B[m] = parameters_NN[count3]

        count3 += 1

    # Initial conditions of each batch
    y0 = np.array([batch1[0, 3:], batch2[0, 3:], batch3[0, 3:], batch4[0, 3:],
                   batch5[0, 3:], batch6[0, 3:], batch7[0, 3:]])

    for m in range(7):
        for n in range(1, 4):
            y0[m, n] = y0[m, n] / y0[m, 0]

    # Set contaning the time vector of eah experiment
    time = [batch1[:, 2], batch2[:, 2], batch3[:, 2], batch4[:, 2], batch5[:, 2],
            batch6[:, 2], batch7[:, 2]]

    # Set contaning the operation temperature vector of eah experiment
    Temperature = [batch1[:, 1], batch2[:, 1], batch3[:, 1], batch4[:, 1], batch5[:, 1],
                   batch6[:, 1], batch7[:, 1]]

    # Number of batches
    batches = 7

    # Matrix containing the model predictions
    model_prediction = np.zeros([len(experimental_data), 5])
    experimental_resuls = experimental_data.iloc[:, 3:].values

    count = 0

    for m in range(batches):
        prediction = integrator_PB(y0[m], time[m], Temperature[m], wH_B, bH_B, wOut_B, bOut_B)

        for n in range(len(prediction)):
            for j in range(5):
                model_prediction[count, j] = prediction[n, j]

            for j in range(1, 4):
                experimental_resuls[count, j] = experimental_resuls[count, j] / experimental_resuls[count, 0]

            count += 1

    # Constraints of the objective function
    # If any of the predict states is negative, returns a high value
    for m in range(len(model_prediction)):

        if model_prediction[m, 0] <= 0:
            print('errado1')
            return 1e1000

        elif model_prediction[m, 1] <= 0:
            print('errado2')
            return 1e1000

        elif model_prediction[m, 2] <= 0:
            print('errado3')
            return 1e1000

        elif model_prediction[m, 3] <= 0:
            print('errado4')
            return 1e1000

        elif model_prediction[m, 4] <= 0:
            print('errado5')
            return 1e1000

    # Normalize the state variables for the experimental set and model predictions
    scale = MinMaxScaler()
    experimental_resuls_normalized = scale.fit_transform(experimental_resuls)
    model_prediction_normalized = scale.transform(model_prediction)

    # Calculate the mean squared error
    mse_mu0 = mean_squared_error(experimental_resuls_normalized[:, 0], model_prediction_normalized[:, 0])
    mse_mu1 = mean_squared_error(experimental_resuls_normalized[:, 1], model_prediction_normalized[:, 1])
    mse_mu2 = mean_squared_error(experimental_resuls_normalized[:, 2], model_prediction_normalized[:, 2])
    mse_mu3 = mean_squared_error(experimental_resuls_normalized[:, 3], model_prediction_normalized[:, 3])
    mse_C = mean_squared_error(experimental_resuls_normalized[:, 4], model_prediction_normalized[:, 4])

    error = mse_mu0 + mse_mu1 + mse_mu2 + mse_mu3 + mse_C

    print(error)

    return error


# Defining information of the neural networks
number_inputs_B = 4
neurons_HidenLayer_B = 20
number_outputs_B = 1

# Defining the initial guess of the optimization problem
wH_B0 = 1e-4 * np.ones(number_inputs_B * neurons_HidenLayer_B)
biasH_B0 = 1e-6 * np.ones(neurons_HidenLayer_B)
wOut_B0 = 1e-4 * np.ones(neurons_HidenLayer_B * number_outputs_B)
biasOut_B0 = 1e-6 * np.ones(number_outputs_B)

# Creating a vector containing all initial guesses
terms = len(wH_B0) + len(biasH_B0) + len(wOut_B0) + len(biasOut_B0)

x0 = np.zeros(terms)

count2 = 0

for i in range(len(wH_B0)):
    x0[count2] = wH_B0[i]
    count2 += 1

for i in range(len(biasH_B0)):
    x0[count2] = biasH_B0[i]
    count2 += 1

for i in range(len(wOut_B0)):
    x0[count2] = wOut_B0[i]
    count2 += 1

for i in range(len(biasOut_B0)):
    x0[count2] = biasOut_B0[i]
    count2 += 1

# Sovilg the optimization problem
sol = minimize(f_obj, x0, method='SLSQP', args=(number_inputs_B,
                                                neurons_HidenLayer_B, number_outputs_B,),
               options={'maxiter': 100, 'ftol': 1e-08})
print(sol)
print(f'x ótimo = {sol.x}')
print(
    f'f(x ótimo) = {f_obj(sol.x, number_inputs_B, neurons_HidenLayer_B, number_outputs_B)}')

solution = np.array([f_obj(sol.x, number_inputs_B, neurons_HidenLayer_B, number_outputs_B)])

header1 = 'x'
header2 = 'fobj'

np.savetxt("x_op_UDE_B_7exps.csv", sol.x, delimiter=',', header=header1)
np.savetxt("f_obj_UDE_B_7exps.csv", solution, delimiter=',', header=header2)
