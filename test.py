import unittest
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from train_model import normalize, initialize_parameters, forward_propagation
import torch
import torch.nn as nn

class test_normalize(unittest.TestCase):
    
    def test_basic(self):
        input = np.reshape(np.array([1, 2, 3, 5]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(result, expected, decimal=6, verbose=True)

    def test_zero(self):
        input = np.reshape(np.array([0, 0, 0, 0]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(result, expected, decimal=6, verbose=True)

    def test_negative(self):
        input = np.reshape(np.array([-1, -2, -8, -3]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(result, expected, decimal=6, verbose=True)

    def test_extreme_small(self):
        input = np.reshape(np.array([0.0001, 0.0002, 0.0003, -0.0004]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(result, expected, decimal=6, verbose=True)

    def test_extreme_large(self):
        input = np.reshape(np.array([10000, 200000, 50000, 10000, 2000000]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(result, expected, decimal=6, verbose=True)

    def test_same(self):
        input = np.reshape(np.array([5, 5]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(result, expected, decimal=6, verbose=True)

class test_initialize_parameters(unittest.TestCase):

    def test_basic(self):
        parameters = initialize_parameters()
        w, b = parameters['w'], parameters['b']
        self.assertGreaterEqual(w, 0)
        self.assertLessEqual(w, 0.1)
        self.assertEqual(b, 0)

class test_forward_propagation(unittest.TestCase):

    def test_basic(self):
        X = np.random.randn(30,1)
        parameters = initialize_parameters()
        w, b = parameters['w'], parameters['b']
        linear = nn.Linear(1,1)
        linear.weight.data = torch.tensor(w)
        linear.bias.data = torch.tensor(b)
        expected = np.array(linear(torch.tensor(X)))
        result = forward_propagation(X, parameters)
        np.testing.assert_allclose(result, expected)

        

unittest.main()
