import unittest
import hypergraph as hg
import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_almost_equal, assert_array_almost_equal


class TestHyperGraph(unittest.TestCase):
    def test_init(self):
        graph = hg.HyperGraph()

    def test_new_variable(self):
        graph = hg.HyperGraph()

        variable = graph.new_variable(5)

        assert_equal(variable.value, 5)

    def test_new_variables(self):
        graph = hg.HyperGraph()

        a, b, c = graph.new_variables([5, 6, 7])

        assert_equal(a.value, 5)
        assert_equal(b.value, 6)
        assert_equal(c.value, 7)

    # addition

    def test_addition_variable_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a + b
        assert_equal(result.value, 11)

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_equal(g, [1, 1])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_addition_variable_constant(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a + 6
        assert_equal(result.value, 11)

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_equal(g, [1, 0])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_addition_constant_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = 5 + b
        assert_equal(result.value, 11)

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_equal(g, [0, 1])
        assert_array_equal(h, [[0, 0], [0, 0]])

    # subtraction

    def test_subtraction_variable_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a - b
        assert_equal(result.value, -1)

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_equal(g, [1, -1])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_subtraction_variable_constant(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a - 6
        assert_equal(result.value, -1)

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_equal(g, [1, 0])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_subtraction_constant_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = 5 - b
        assert_equal(result.value, -1)

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_equal(g, [0, -1])
        assert_array_equal(h, [[0, 0], [0, 0]])

    # multiplication

    def test_multiplication_variable_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a * b
        assert_equal(result.value, 30)

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_equal(g, [6, 5])
        assert_array_equal(h, [[0, 1], [0, 0]])

    def test_multiplication_variable_constant(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a * 6
        assert_equal(result.value, 30)

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_equal(g, [6, 0])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_multiplication_constant_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = 5 * b
        assert_equal(result.value, 30)

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_equal(g, [0, 5])
        assert_array_equal(h, [[0, 0], [0, 0]])

    # division

    def test_division_variable_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a / b
        assert_almost_equal(result.value, 5 / 6)

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_almost_equal(g, [1/6, -5/36])
        assert_array_almost_equal(h, [[0, -1/36], [0, 5/108]])

    def test_division_variable_constant(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a / 6
        assert_almost_equal(result.value, 5 / 6)

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_almost_equal(g, [1/6, 0])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_division_constant_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = 5 / b
        assert_almost_equal(result.value, 5 / 6)

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_almost_equal(g, [0, -5/36])
        assert_array_almost_equal(h, [[0, 0], [0, 5/108]])

    # trigonometry

    def test_cos(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = np.cos(a)
        assert_almost_equal(result.value, np.cos(5))

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_almost_equal(g, [-np.sin(5), 0])
        assert_array_almost_equal(h, [[-np.cos(5), 0], [0, 0]])

    def test_sin(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = np.sin(a)
        assert_almost_equal(result.value, np.sin(5))

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_almost_equal(g, [np.cos(5), 0])
        assert_array_almost_equal(h, [[-np.sin(5), 0], [0, 0]])

    def test_tan(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = np.tan(a)
        assert_almost_equal(result.value, np.tan(5))

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_almost_equal(g, [1/np.cos(5)**2, 0])
        assert_array_almost_equal(h, [[2/np.cos(5)**2*np.tan(5), 0], [0, 0]])

    def test_acos(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([3/4, 6])

        result = np.arccos(a)
        assert_almost_equal(result.value, np.arccos(3/4))

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_almost_equal(g, [-4/np.sqrt(7), 0])
        assert_array_almost_equal(h, [[-48/(7*np.sqrt(7)), 0], [0, 0]])

    def test_asin(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([3/4, 6])

        result = np.arcsin(a)
        assert_almost_equal(result.value, np.arcsin(3/4))

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_almost_equal(g, [4/np.sqrt(7), 0])
        assert_array_almost_equal(h, [[48/(7*np.sqrt(7)), 0], [0, 0]])

    def test_atan(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([3/4, 6])

        result = np.arctan(a)
        assert_almost_equal(result.value, np.arctan(3/4))

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_almost_equal(g, [16/25, 0])
        assert_array_almost_equal(h, [[-384/625, 0], [0, 0]])

    # other

    def test_sqrt(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = np.sqrt(a)
        assert_almost_equal(result.value, np.sqrt(5))

        graph.compute(result)
        g = graph.g([a, b])
        h = graph.h([a, b])
        assert_array_almost_equal(g, [1/(2*np.sqrt(5)), 0])
        assert_array_almost_equal(h, [[-1/(20*np.sqrt(5)), 0], [0, 0]])

    # vector

    def test_cross(self):
        graph = hg.HyperGraph()

        ax, ay, az, bx, by, bz = graph.new_variables([1, 2, 3, 4, 5, 6])

        a = np.array([ax, ay, az])
        b = np.array([bx, by, bz])

        result = np.cross(a, b)
        assert_almost_equal(result[0].value, -3)
        assert_almost_equal(result[1].value, 6)
        assert_almost_equal(result[2].value, -3)

        graph.compute(result[0])
        g = graph.g([ax, ay, az, bx, by, bz])
        h = graph.h([ax, ay, az, bx, by, bz])
        assert_array_almost_equal(g, [0, 6, -5, 0, -3, 2])
        assert_array_almost_equal(h, [[0, 0, 0, 0,  0, 0],
                                      [0, 0, 0, 0,  0, 1],
                                      [0, 0, 0, 0, -1, 0],
                                      [0, 0, 0, 0,  0, 0],
                                      [0, 0, 0, 0,  0, 0],
                                      [0, 0, 0, 0,  0, 0]])

        graph.compute(result[1])
        g = graph.g([ax, ay, az, bx, by, bz])
        h = graph.h([ax, ay, az, bx, by, bz])
        assert_array_almost_equal(g, [-6, 0, 4, 3, 0, -1])
        assert_array_almost_equal(h, [[0, 0, 0, 0, 0, -1],
                                      [0, 0, 0, 0, 0,  0],
                                      [0, 0, 0, 1, 0,  0],
                                      [0, 0, 0, 0, 0,  0],
                                      [0, 0, 0, 0, 0,  0],
                                      [0, 0, 0, 0, 0,  0]])

        graph.compute(result[2])
        g = graph.g([ax, ay, az, bx, by, bz])
        h = graph.h([ax, ay, az, bx, by, bz])
        assert_array_almost_equal(g, [5, -4, 0, -2, 1, 0])
        assert_array_almost_equal(h, [[0, 0, 0,  0, 1, 0],
                                      [0, 0, 0, -1, 0, 0],
                                      [0, 0, 0,  0, 0, 0],
                                      [0, 0, 0,  0, 0, 0],
                                      [0, 0, 0,  0, 0, 0],
                                      [0, 0, 0,  0, 0, 0]])

    def test_dot(self):
        graph = hg.HyperGraph()

        ax, ay, az, bx, by, bz = graph.new_variables([1, 2, 3, 4, 5, 6])

        a = np.array([ax, ay, az])
        b = np.array([bx, by, bz])

        result = np.dot(a, b)
        assert_almost_equal(result.value, 32)

        graph.compute(result)
        g = graph.g([ax, ay, az, bx, by, bz])
        h = graph.h([ax, ay, az, bx, by, bz])
        assert_array_almost_equal(g, [4, 5, 6, 1, 2, 3])
        assert_array_almost_equal(h, [[0, 0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 0, 1],
                                      [0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0]])

    def test_norm(self):
        graph = hg.HyperGraph()

        ax, ay, az, bx, by, bz = graph.new_variables([1, 2, 3, 4, 5, 6])

        a = np.array([ax, ay, az])
        b = np.array([bx, by, bz])

        result = np.linalg.norm(np.cross(a, b))
        assert_almost_equal(result.value, 3*np.sqrt(6))

        graph.compute(result)
        g = graph.g([ax, ay, az, bx, by, bz])
        h = graph.h([ax, ay, az, bx, by, bz])
        assert_array_almost_equal(g, [-17/np.sqrt(6), -np.sqrt(2/3), 13/np.sqrt(6), 4*np.sqrt(2/3), np.sqrt(2/3), -2*np.sqrt(2/3)])
        assert_array_almost_equal(h, [[77/(18*np.sqrt(6)), -77/(9*np.sqrt(6)), 77/(18*np.sqrt(6)), -8*np.sqrt(2/3)/9,  23/(9*np.sqrt(6)), -17*np.sqrt(2/3)/9],
                                      [0, 77*np.sqrt(2/3)/9, -77/(9*np.sqrt(6)), 41/(9*np.sqrt(6)), -32*np.sqrt(2/3)/9, 23/(9*np.sqrt(6))],
                                      [0, 0, 77/(18*np.sqrt(6)), np.sqrt(2/3)/9, 41/(9*np.sqrt(6)), -8*np.sqrt(2/3)/9],
                                      [0, 0, 0, 7/(9*np.sqrt(6)),  -7*np.sqrt(2/3)/9, 7/(9*np.sqrt(6))],
                                      [0, 0, 0, 0,  14*np.sqrt(2/3)/9, -7*np.sqrt(2/3)/9],
                                      [0, 0, 0, 0,  0, 7/(9*np.sqrt(6))]])

    def test_full_hessian(self):
        graph = hg.HyperGraph()

        ax, ay, az, bx, by, bz = graph.new_variables([1, 2, 3, 4, 5, 6])

        a = np.array([ax, ay, az])
        b = np.array([bx, by, bz])

        result = np.linalg.norm(np.cross(a, b))

        graph.compute(result)

        h = graph.h([ax, ay, az, bx, by, bz], full=True)

        assert_array_almost_equal(np.triu(h), [[77/(18*np.sqrt(6)), -77/(9*np.sqrt(6)), 77/(18*np.sqrt(6)), -8*np.sqrt(2/3)/9,  23/(9*np.sqrt(6)), -17*np.sqrt(2/3)/9],
                                               [0, 77*np.sqrt(2/3)/9, -77/(9*np.sqrt(6)), 41/(9*np.sqrt(6)), -32*np.sqrt(2/3)/9, 23/(9*np.sqrt(6))],
                                               [0, 0, 77/(18*np.sqrt(6)), np.sqrt(2/3)/9, 41/(9*np.sqrt(6)), -8*np.sqrt(2/3)/9],
                                               [0, 0, 0, 7/(9*np.sqrt(6)),  -7*np.sqrt(2/3)/9, 7/(9*np.sqrt(6))],
                                               [0, 0, 0, 0,  14*np.sqrt(2/3)/9, -7*np.sqrt(2/3)/9],
                                               [0, 0, 0, 0,  0, 7/(9*np.sqrt(6))]])
        assert_array_almost_equal(np.tril(h).T, np.triu(h))

    def test_out(self):
        graph = hg.HyperGraph()

        ax, ay, az, bx, by, bz = graph.new_variables([1, 2, 3, 4, 5, 6])

        a = np.array([ax, ay, az])
        b = np.array([bx, by, bz])

        result = np.linalg.norm(np.cross(a, b))
        assert_almost_equal(result.value, 3*np.sqrt(6))

        graph.compute(result)

        g = np.empty(6)
        h = np.empty((6, 6))

        graph.g([ax, ay, az, bx, by, bz], out=g)
        graph.h([ax, ay, az, bx, by, bz], out=h)

        assert_array_almost_equal(g, [-17/np.sqrt(6), -np.sqrt(2/3), 13/np.sqrt(6), 4*np.sqrt(2/3), np.sqrt(2/3), -2*np.sqrt(2/3)])
        assert_array_almost_equal(h, [[77/(18*np.sqrt(6)), -77/(9*np.sqrt(6)), 77/(18*np.sqrt(6)), -8*np.sqrt(2/3)/9,  23/(9*np.sqrt(6)), -17*np.sqrt(2/3)/9],
                                      [0, 77*np.sqrt(2/3)/9, -77/(9*np.sqrt(6)), 41/(9*np.sqrt(6)), -32*np.sqrt(2/3)/9, 23/(9*np.sqrt(6))],
                                      [0, 0, 77/(18*np.sqrt(6)), np.sqrt(2/3)/9, 41/(9*np.sqrt(6)), -8*np.sqrt(2/3)/9],
                                      [0, 0, 0, 7/(9*np.sqrt(6)),  -7*np.sqrt(2/3)/9, 7/(9*np.sqrt(6))],
                                      [0, 0, 0, 0,  14*np.sqrt(2/3)/9, -7*np.sqrt(2/3)/9],
                                      [0, 0, 0, 0,  0, 7/(9*np.sqrt(6))]])

if __name__ == '__main__':
    unittest.main()
