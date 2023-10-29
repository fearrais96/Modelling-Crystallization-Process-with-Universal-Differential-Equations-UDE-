"""
Author: Fernando Arrais Romero Dias Lima

Training universal differential equations (UDE) for modelling the potassium sulfate batch
colling crystallization in water, considering dissolution
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Saving the experimental data in a variable

experimental_data_cresc = pd.read_csv("Cresc_Nucl_K2SO4_7exps_train_new.csv")

experimental_data_diss = pd.read_csv("Dissolucao_K2SO4_4exps_new.csv")

batch1 = (experimental_data_diss[(experimental_data_diss["Experiment"] == 0)]).values
batch2 = (experimental_data_diss[(experimental_data_diss["Experiment"] == 1)]).values
batch3 = (experimental_data_diss[(experimental_data_diss["Experiment"] == 2)]).values
batch4 = (experimental_data_diss[(experimental_data_diss["Experiment"] == 3)]).values


# Defining the activation functions

def ReLU(x):
    if x > 0:
        return x
    return 0


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def NN_D(input, neurons_HiddenLayer, wH, bH, wOut, bOut):
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


def modelagem_PB(entrada, t, T, wH_D, bH_D, wOut_D, bOut_D):
    """
    This function defines the population balance model for nucleation and crystal growth
    of the potassium sulfate batch crystallization in water. Details about the model can
    be found in the work of Moraes et al. (2023) (doi:10.1021/acs.iecr.3c00739).

    This model is solved using the universal differential equation approach (UDE), where
    the nucleation rate (B) and the growth rate (G) are predicted by neural networks.

    :param entrada: state veriables of the model (the first four moments and the concentration)
    :param t: time vector
    :param T: process temperature
    :param wH_D: weights matrix of the neurons between the input and the hidden layers for
    the neural network to predict B
    :param bH_D: biases vector of the hidden layer for the neural network to predict B
    :param wOut_D: weights matrix of the neurons between the hidden and the output layers for
    the neural network to predict B
    :param bOut_D: biases vector of the output layer for the neural network to predict B
    :return: derivative values for the five states
    """

    # Valores dos parâmetros usados
    kv = 1.349
    ro = 2.658  # [roc] = g/cm^3

    # Valores de entrada para mu0, mu1, mu2, mu3 e concentração
    mu0 = entrada[0]  # #/cm3
    r_1 = entrada[1]
    r_2 = entrada[2]
    r_3 = entrada[3]
    C = entrada[4]  # g/cm3

    Ceq = -686.2686 + 3.579165 * (T + 273.15) - 0.00292874 * (T + 273.15) ** 2  # g/L

    S = C * 1e3 / Ceq

    T_norm = (T - np.min(experimental_data_cresc.iloc[:, 1])) / (
            np.max(experimental_data_cresc.iloc[:, 1]) - np.min(experimental_data_cresc.iloc[:, 1]))

    C_norm = (C - np.min(experimental_data_cresc.iloc[:, 7])) / (
            np.max(experimental_data_cresc.iloc[:, 7]) - np.min(experimental_data_cresc.iloc[:, 7]))

    input_D = np.array([C_norm, T_norm])

    D = NN_D(input_D, len(bH_D), wH_D, bH_D, wOut_D, bOut_D)

    psi = r_3*mu0*kv*1e-12

    dmi0dt = 0
    dmi1dt = D * 1e4
    dmi2dt = 2.0 * D * r_1 * 1e-4 * 1e8
    dmi3dt = 3.0 * D * r_2 * 1e-8 * 1e12
    dcdt = -3.0 * ro * kv * D * r_2 * mu0 * 1e-8/(1 - psi)

    return [dmi0dt, dmi1dt, dmi2dt, dmi3dt, dcdt]


def integrator_PB(y0, t, temperature_range, wH_D, bH_D, wOut_D, bOut_D):
    """
    This function is used to integrate the the population balance model for nucleation and crystal growth
    of the potassium sulfate batch crystallization in water.

    :param y0: process initial conditions (initial values of the four moments and the concentration)
    :param t: time vector of the batch
    :param temperature_range: vector containing the process temperature of the process
    :param wH_D: weights matrix of the neurons between the input and the hidden layers for
    the neural network to predict B
    :param bH_D: biases vector of the hidden layer for the neural network to predict B
    :param wOut_D: weights matrix of the neurons between the hidden and the output layers for
    the neural network to predict B
    :param bOut_D: biases vector of the output layer for the neural network to predict B
    :return: values of the moments and concentration for the initial conditions, temperatur,and
    time specifieds
    """
    result = np.zeros([len(temperature_range), 5])

    for i in range(5):
        result[0, i] = y0[i]

    for i in range(len(t) - 1):

        ts = [t[i], t[i + 1]]
        y = odeint(modelagem_PB, y0, ts, args=(temperature_range[i + 1], wH_D, bH_D, wOut_D, bOut_D,), atol=1e-8,
                   rtol=1e-8)

        for j in range(5):
            y0[j] = y[1, j]

        for j in range(5):
            result[i + 1, j] = y[1, j]

    return result


def f_obj(parameters_NN, number_inputs_D, neurons_HidenLayer_D,
          number_outputs_D):
    """
    Objective function to be minimized in the optimization problem. This function is the
    sum of the mean squared errors between the experimental and predicted values of the
    state variables. The mean squared error is calculated for the normalized variables,
    using the MinMaxScaler from scikitlearn.

    :param parameters_NN: vector containing the weights and biases used in the neural networks
    :param number_inputs_D: number of inputs of the neural network to predict B
    :param neurons_HidenLayer_D: number of hidden layers of the neural network to predict B
    :param number_outputs_D: number of outputs of the neural network to predict B
    :return: sum of the normilized mean squared errors between the predicted and the experimental values
    of th state variables
    """

    # Initially, te weight matrices and the biases vectors are created for each neural network,
    # using the vector containing the weights and biases used in the neural networks

    wH_D = np.zeros([number_inputs_D, neurons_HidenLayer_D])
    bH_D = np.zeros(neurons_HidenLayer_D)
    wOut_D = np.zeros([neurons_HidenLayer_D, number_outputs_D])
    bOut_D = np.zeros(number_outputs_D)

    count3 = 0

    for m in range(number_inputs_D):
        for n in range(neurons_HidenLayer_D):
            wH_D[m][n] = parameters_NN[count3]

            count3 += 1

    for m in range(neurons_HidenLayer_D):
        bH_D[m] = parameters_NN[count3]

        count3 += 1

    for m in range(neurons_HidenLayer_D):
        for n in range(number_outputs_D):
            wOut_D[m][n] = parameters_NN[count3]

            count3 += 1

    for m in range(number_outputs_D):
        bOut_D[m] = parameters_NN[count3]

        count3 += 1

    # Initial conditions of each batch
    y0 = np.array([batch1[0, 3:], batch2[0, 3:], batch3[0, 3:], batch4[0, 3:]])

    for m in range(4):
        for n in range(1, 4):
            y0[m, n] = y0[m, n] / y0[m, 0]

    # Set contaning the time vector of eah experiment
    time = [batch1[:, 2], batch2[:, 2], batch3[:, 2], batch4[:, 2]]

    # Set contaning the operation temperature vector of eah experiment
    Temperature = [batch1[:, 1], batch2[:, 1], batch3[:, 1], batch4[:, 1]]

    # Number of batches
    batches = 4

    # Matrix containing the model predictions
    model_prediction = np.zeros([len(experimental_data_diss), 5])
    experimental_resuls = experimental_data_diss.iloc[:, 3:].values

    count = 0

    for m in range(batches):
        prediction = integrator_PB(y0[m], time[m], Temperature[m], wH_D, bH_D, wOut_D, bOut_D)

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

    mse_mu1 = mean_squared_error(experimental_resuls_normalized[:, 1], model_prediction_normalized[:, 1])
    mse_mu2 = mean_squared_error(experimental_resuls_normalized[:, 2], model_prediction_normalized[:, 2])
    mse_mu3 = mean_squared_error(experimental_resuls_normalized[:, 3], model_prediction_normalized[:, 3])
    mse_C = mean_squared_error(experimental_resuls_normalized[:, 4], model_prediction_normalized[:, 4])

    error = mse_mu1 + mse_mu2 + mse_mu3 + mse_C

    print(error)

    return error


# Defining information of the neural networks
number_inputs_D = 2
neurons_HidenLayer_D = 20
number_outputs_D = 1

# Defining the initial guess of the optimization problem
wH_D0 = 1e-4 * np.ones(number_inputs_D * neurons_HidenLayer_D)
biasH_D0 = 1e-6 * np.ones(neurons_HidenLayer_D)
wOut_D0 = 1e-4 * np.ones(neurons_HidenLayer_D * number_outputs_D)
biasOut_D0 = 1e-6 * np.ones(number_outputs_D)

# Creating a vector containing all initial guesses
terms = len(wH_D0) + len(biasH_D0) + len(wOut_D0) + len(biasOut_D0)

x0 = np.zeros(terms)

count2 = 0

for i in range(len(wH_D0)):
    x0[count2] = wH_D0[i]
    count2 += 1

for i in range(len(biasH_D0)):
    x0[count2] = biasH_D0[i]
    count2 += 1

for i in range(len(wOut_D0)):
    x0[count2] = wOut_D0[i]
    count2 += 1

for i in range(len(biasOut_D0)):
    x0[count2] = biasOut_D0[i]
    count2 += 1

# Sovilg the optimization problem
sol = minimize(f_obj, x0, method='SLSQP', args=(number_inputs_D,
                                                neurons_HidenLayer_D, number_outputs_D,),
               options={'maxiter': 100, 'ftol': 1e-08})
print(sol)
print(f'x ótimo = {sol.x}')
print(
    f'f(x ótimo) = {f_obj(sol.x, number_inputs_D, neurons_HidenLayer_D, number_outputs_D)}')

solution = np.array([f_obj(sol.x, number_inputs_D, neurons_HidenLayer_D, number_outputs_D)])

header1 = 'x'
header2 = 'fobj'

np.savetxt("x_op_UDE_diss_4exps.csv", sol.x, delimiter=',', header=header1)
np.savetxt("f_obj_UDE_diss_4exps.csv", solution, delimiter=',', header=header2)
