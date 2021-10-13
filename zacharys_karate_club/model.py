from collections import namedtuple
from networkx import read_edgelist, set_node_attributes, to_numpy_matrix
from pandas import read_csv, Series
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

    attributes = read_csv(
        'data/features.csv',
        index_col=['node'])

    for attribute in attributes.columns.values:
        set_node_attributes(
            network,
            values=Series(
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

    def __init__(self, A, input_units, output_units, activation='tanh'):

        super(SpektralRule, self).__init__()

        self.input_units = input_units
        self.output_units = output_units

        # Define a linear layer
        self.linear_layer = nn.Linear(self.input_units, self.output_units)

        nn.init.xavier_normal_(self.linear_layer.weight)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()

        # Create identity matrix
        I = torch.eye(A.shape[1])
        # Adding self loops to the adjacency matrix
        A_hat = A + I
        A_hat = A_hat.to(torch.double)
        # Inverse degree Matrix
        D = torch.diag(torch.pow(torch.sum(A_hat, dim=0), -0.5), 0)
        # applying spectral rule
        self.A_hat = torch.matmul(torch.matmul(D, A_hat), D)
        self.A_hat.requires_grad = False  # Non trainable parameter

    def forward(self, X):
        # aggregation
        aggregation = torch.matmul(self.A_hat, X)
        # propagation through the linear layer that will have the number of hidden nodes specified
        linear_output = self.linear_layer(aggregation.to(torch.float))
        propagation = self.activation(linear_output)

        return propagation.to(torch.double)





class FeatureModel(nn.Module):
    def __init__(self, A, hidden_layer_config, initial_input_size):
        super(FeatureModel, self).__init__()

        self.hidden_layer_config = hidden_layer_config
        self.moduleList = list()
        self.initial_input_size = initial_input_size

        for input_size, activation in hidden_layer_config:
            self.moduleList.append(SpektralRule(A, self.initial_input_size, input_size, activation))
            self.initial_input_size = input_size

        self.sequentialModule = nn.Sequential(*self.moduleList)

    def forward(self, X):
        output = self.sequentialModule(X)
        return output


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
        self.logisticRegressor = LogisticRegressor(input_units=input_size,
                                                   output_units=output_size)

    def forward(self, X):
        classified = self.logisticRegressor(X)
        return classified


class HybridModel(nn.Module):
    """
    Final model used to train and predict.
    """
    def __init__(self, A, hidden_layer_config, initial_input_size, output_nodes):
        super(HybridModel, self).__init__()
        self.featureModel = FeatureModel(A, hidden_layer_config, identity.shape[1])
        self.featureModelOutputSize = self.featureModel.initial_input_size
        self.classifier = ClassifierModel(self.featureModelOutputSize, output_nodes)
        self.featureModelOutput = None

    def forward(self, X):
        outputFeature = self.featureModel(X)
        classified = self.classifier(outputFeature)
        self.featureModelOutput = outputFeature
        return classified


def train(model, epoch, criterion, optimizer, feature):
    cumLoss = 0
    losses = list()

    for j in range(epoch):
        two_loss = 0
        for i, node in enumerate(X_train_flattened):
            output = model(feature)[node]

            ground_truth = torch.reshape(y_train[i], output.shape)

            optimizer.zero_grad()

            loss = criterion(output, ground_truth)
            # \print("loss: ",loss.data)
            two_loss += loss.item()

            loss.backward()

            optimizer.step()
        losses.append(two_loss)
        cumLoss += two_loss
    print('avg loss: ', cumLoss / epoch)
    torch.save(model.state_dict(), "./gcn.pth")
    plt.plot(losses)


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

    # Model configuration
    hidden_layer_config = [(4, 'tanh'),
                           (2, 'tanh')]
    output_nodes = 1  # We're only trying to predict between 2 classes

    model = HybridModel(A, hidden_layer_config, identity.shape[1], output_nodes)

    output = model(identity)

    # print(zkc.y_test)
    # print(output)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    featureoutput = None

    train(model, 10000, criterion, optimizer, identity)
