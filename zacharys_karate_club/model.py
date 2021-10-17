from collections import namedtuple

import networkx as nx
from networkx import read_edgelist, set_node_attributes, to_numpy_matrix
import pandas as pd
from numpy import array
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


DataSet = namedtuple(
    'DataSet',
    field_names=['X_train', 'y_train', 'X_test', 'y_test', 'network']
)


def load_karate_club():
    network = read_edgelist(
        'data/zkc.edgelist',
        nodetype=int)

    attributes = pd.read_csv(
        'data/features.csv',
        index_col=['node'])

    for attribute in attributes.columns.values:
        set_node_attributes(
            network,
            values=pd.Series(
                attributes[attribute],
                index=attributes.index).to_dict(),
            name=attribute
        )

    X_train, y_train = map(array, zip(*[
        ([node], data['role'] == 'Administrator')
        for node, data in network.nodes(data=True)
        if data['role'] in {'Administrator', 'Instructor'}
    ]))

    X_test, y_test = map(array, zip(*[
        ([node], data['community'] == 'Administrator')
        for node, data in network.nodes(data=True)
        if data['role'] == 'Member'
    ]))

    return DataSet(
        X_train, y_train,
        X_test, y_test,
        network)


class SpektralRule(nn.Module):
    """Applies the spektral rule to the Adjacency matrix"""
    def __init__(self, A, input_units, output_units, activation='tanh'):
        super(SpektralRule, self).__init__()
        self.linear_layer = nn.Linear(input_units, output_units)  # Define a linear layer
        nn.init.xavier_normal_(self.linear_layer.weight)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()

        # This is a on-off calculation
        I = torch.eye(A.shape[1])  # Create identity matrix
        A_hat = A + I  # Adding self loops to the adjacency matrix
        A_hat = A_hat.to(torch.double)
        D = torch.diag(torch.pow(torch.sum(A_hat, dim=0), -0.5), 0)  # Inverse degree Matrix
        self.A_hat = torch.matmul(torch.matmul(D, A_hat), D)  # Applying spectral rule
        self.A_hat.requires_grad = False  # Non trainable parameter

    def forward(self, X):
        aggregation = torch.matmul(self.A_hat, X)
        # Propagation through the linear layer that will have the number of hidden nodes specified
        linear_output = self.linear_layer(aggregation.to(torch.float))
        propagation = self.activation(linear_output)

        return propagation.to(torch.double)


class FeatureModel(nn.Module):
    """Class is used to apply the spektral rule and calculate the convolutions."""
    def __init__(self, A, hidden_layer_config, initial_input_size):
        super(FeatureModel, self).__init__()
        # self.hidden_layer_config = hidden_layer_config
        self.moduleList = list()  # List to keep track of convolutional layers
        self.initial_input_size = initial_input_size  # Define this here so it can be changed downstream

        for input_size, activation in hidden_layer_config:
            # Define the requested number of convolutions
            self.moduleList.append(SpektralRule(A, self.initial_input_size, input_size, activation))
            # Change the input size to the previous layer's input size for the next iteration
            self.initial_input_size = input_size

        # Create a sequential model from the input hidden layer configuration
        self.sequentialModule = nn.Sequential(*self.moduleList)

    def forward(self, X):
        feature_output = self.sequentialModule(X)  # Apply the sequential model
        return feature_output


class LogisticRegressor(nn.Module):
    """ Model to be used for the final prediction."""
    def __init__(self, input_units, output_units):
        super(LogisticRegressor, self).__init__()
        self.Linear = nn.Linear(input_units, output_units, bias=True)
        nn.init.xavier_normal_(self.Linear.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        linear_output = self.Linear(X.to(torch.float))
        return self.sigmoid(linear_output)


class ClassifierModel(nn.Module):
    """Class serves as factory for the last node classification layer."""
    def __init__(self, input_size, output_size):
        super(ClassifierModel, self).__init__()
        self.classifier = LogisticRegressor(input_units=input_size,
                                            output_units=output_size)

    def forward(self, X):
        classified = self.classifier(X)
        return classified


class HybridModel(nn.Module):
    """
    Final model used to train and predict.
    """
    def __init__(self, A, hidden_layer_config, initial_input_size, output_nodes):
        super(HybridModel, self).__init__()
        self.featureModel = FeatureModel(A, hidden_layer_config, initial_input_size)
        # This parameter will be updated with the last layer's input size
        self.featureModelOutputSize = self.featureModel.initial_input_size
        self.classifier = ClassifierModel(self.featureModelOutputSize, output_nodes)
        self.featureModelOutput = None

    def forward(self, X):
        outputFeature = self.featureModel(X)
        classified = self.classifier(outputFeature)
        self.featureModelOutput = outputFeature
        return classified


def train(model, epochs, criterion, optimizer, features):
    """Used to train the model."""
    cumLoss = 0
    losses = list()

    for j in range(epochs):
        two_loss = 0
        for i, node in enumerate(X_train_flattened):
            # Forward pass - get prediction for relevant node only
            output = model(features)[node]
            # Get the label for the node
            ground_truth = torch.reshape(y_train[i], output.shape)
            # For every mini-batch during training we need to explicitly set the gradients to zero
            # before backpropagation because PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            # Calculate loss
            loss = criterion(output, ground_truth)
            # print("loss: ", loss.data)
            two_loss += loss.item()
            # Backpropagation
            loss.backward()
            # Perform parameter update based on the current gradient
            optimizer.step()
        losses.append(two_loss)
        cumLoss += two_loss
    print('avg loss: ', cumLoss / epochs)
    # Save model
    torch.save(model.state_dict(), "./gcn.pth")
    plt.plot(losses)


def test(model, features, X_test_flattened):
    # model = HybridModel(A, hidden_layer_config, identity.shape[1])
    # model.load_state_dict(torch.load("./gcn.pth"))
    model.eval()
    correct = 0
    masked_output = list()
    for i, node in enumerate(X_test_flattened):
        output = model(features)[node]
        masked_output.append(output.ge(0.5))

    return masked_output


if __name__ == '__main__':

    # Load data
    zkc = load_karate_club()
    X_train_flattened = torch.flatten(torch.from_numpy(zkc.X_train))
    X_test_flattened = torch.flatten(torch.from_numpy(zkc.X_test))
    y_train = torch.from_numpy(zkc.y_train).to(torch.float)

    # Initial Transformation
    A = to_numpy_matrix(zkc.network)
    A = torch.from_numpy(np.array(A))
    identity = torch.eye(A.shape[1])
    identity = identity.to(torch.double)
    identity.requires_grad = False

    # Use node distances as features additional to the identity matrix
    X_2 = np.zeros((A.shape[0], 2))
    node_distance_instructor = nx.shortest_path_length(zkc.network, target=33)
    node_distance_administrator = nx.shortest_path_length(zkc.network, target=0)

    for node in zkc.network.nodes():
        X_2[node][0] = node_distance_administrator[node]
        X_2[node][1] = node_distance_instructor[node]

    X_2 = torch.cat((identity, torch.from_numpy(X_2)), 1)
    X_2.requires_grad = False

    # Model configuration
    hidden_layer_config = [(4, 'tanh'),
                           (2, 'tanh')]
    output_nodes = 1  # We're only trying to predict between 2 classes

    model = HybridModel(A, hidden_layer_config, X_2.shape[1], output_nodes)
    output = model(X_2)

    # print(zkc.y_test)
    # print(output)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # featureoutput = None

    train(model, 10000, criterion, optimizer, X_2)

    after = None
    masked = test(model, X_2, X_test_flattened)
    masked = [i.item() for i in masked]
    print(masked)

    test_gt = torch.from_numpy(zkc.y_test)
    test_gt = [i.item() for i in test_gt]

    counter = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    correct = zip(masked, test_gt)
    for (masked, gt) in list(correct):
        if masked == gt and masked is True:
            tp += 1
        if masked == gt and masked is False:
            tn += 1
        if masked is False and gt is True:
            fn += 1
        if masked is True and gt is False:
            fp += 1

    accuracy = (tp + tn) / (tp+fp+fn+tn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print('accuracy ', accuracy)
    print('precision ', precision)
    print('recall ', recall)
