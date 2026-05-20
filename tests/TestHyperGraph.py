import unittest
import math
import hypergraph as hg
import numpy as np
import scipy.sparse
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
        g = graph.g()
        h = graph.h()
        assert_array_equal(g, [1, 1])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_addition_variable_constant(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a + 6
        assert_equal(result.value, 11)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_equal(g, [1, 0])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_addition_constant_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = 5 + b
        assert_equal(result.value, 11)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_equal(g, [0, 1])
        assert_array_equal(h, [[0, 0], [0, 0]])

    # subtraction

    def test_subtraction_variable_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a - b
        assert_equal(result.value, -1)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_equal(g, [1, -1])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_subtraction_variable_constant(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a - 6
        assert_equal(result.value, -1)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_equal(g, [1, 0])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_subtraction_constant_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = 5 - b
        assert_equal(result.value, -1)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_equal(g, [0, -1])
        assert_array_equal(h, [[0, 0], [0, 0]])

    # multiplication

    def test_multiplication_variable_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a * b
        assert_equal(result.value, 30)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_equal(g, [6, 5])
        assert_array_equal(h, [[0, 1], [0, 0]])

    def test_multiplication_variable_constant(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a * 6
        assert_equal(result.value, 30)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_equal(g, [6, 0])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_multiplication_constant_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = 5 * b
        assert_equal(result.value, 30)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_equal(g, [0, 5])
        assert_array_equal(h, [[0, 0], [0, 0]])

    # division

    def test_division_variable_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a / b
        assert_almost_equal(result.value, 5 / 6)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [1/6, -5/36])
        assert_array_almost_equal(h, [[0, -1/36], [0, 5/108]])

    def test_division_variable_constant(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a / 6
        assert_almost_equal(result.value, 5 / 6)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [1/6, 0])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_division_constant_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = 5 / b
        assert_almost_equal(result.value, 5 / 6)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [0, -5/36])
        assert_array_almost_equal(h, [[0, 0], [0, 5/108]])

    # trigonometry

    def test_cos(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = np.cos(a)
        assert_almost_equal(result.value, np.cos(5))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [-np.sin(5), 0])
        assert_array_almost_equal(h, [[-np.cos(5), 0], [0, 0]])

    def test_sin(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = np.sin(a)
        assert_almost_equal(result.value, np.sin(5))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [np.cos(5), 0])
        assert_array_almost_equal(h, [[-np.sin(5), 0], [0, 0]])

    def test_tan(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = np.tan(a)
        assert_almost_equal(result.value, np.tan(5))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [1/np.cos(5)**2, 0])
        assert_array_almost_equal(h, [[2/np.cos(5)**2*np.tan(5), 0], [0, 0]])

    def test_acos(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([3/4, 6])

        result = np.arccos(a)
        assert_almost_equal(result.value, np.arccos(3/4))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [-4/np.sqrt(7), 0])
        assert_array_almost_equal(h, [[-48/(7*np.sqrt(7)), 0], [0, 0]])

    def test_asin(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([3/4, 6])

        result = np.arcsin(a)
        assert_almost_equal(result.value, np.arcsin(3/4))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [4/np.sqrt(7), 0])
        assert_array_almost_equal(h, [[48/(7*np.sqrt(7)), 0], [0, 0]])

    def test_atan(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([3/4, 6])

        result = np.arctan(a)
        assert_almost_equal(result.value, np.arctan(3/4))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [16/25, 0])
        assert_array_almost_equal(h, [[-384/625, 0], [0, 0]])

    # exponential / logarithmic

    def test_exp(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([2, 3])

        result = a.exp()
        assert_almost_equal(result.value, np.exp(2))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [np.exp(2), 0])
        assert_array_almost_equal(h, [[np.exp(2), 0], [0, 0]])

    def test_log(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([2, 3])

        result = a.log()
        assert_almost_equal(result.value, np.log(2))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [1/2, 0])
        assert_array_almost_equal(h, [[-1/4, 0], [0, 0]])

    # power / square / abs

    def test_square(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([3, 5])

        result = a.square()
        assert_almost_equal(result.value, 9)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        # d(x²)/dx = 2x = 6, d²(x²)/dx² = 2
        assert_array_almost_equal(g, [6, 0])
        assert_array_almost_equal(h, [[2, 0], [0, 0]])

    def test_pow(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([2, 3])

        result = a ** 3
        assert_almost_equal(result.value, 8)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        # d(x³)/dx = 3x² = 12, d²(x³)/dx² = 6x = 12
        assert_array_almost_equal(g, [12, 0])
        assert_array_almost_equal(h, [[12, 0], [0, 0]])

    def test_abs_positive(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([3, 5])

        result = abs(a)
        assert_almost_equal(result.value, 3)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [1, 0])
        assert_array_almost_equal(h, [[0, 0], [0, 0]])

    def test_abs_negative(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([-3, 5])

        result = abs(a)
        assert_almost_equal(result.value, 3)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [-1, 0])
        assert_array_almost_equal(h, [[0, 0], [0, 0]])

    # other

    def test_sqrt(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = np.sqrt(a)
        assert_almost_equal(result.value, np.sqrt(5))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [1/(2*np.sqrt(5)), 0])
        assert_array_almost_equal(h, [[-1/(20*np.sqrt(5)), 0], [0, 0]])

    # hyperbolic

    def test_sinh(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([2, 3])

        result = a.sinh()
        assert_almost_equal(result.value, np.sinh(2))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        # d(sinh)/dx = cosh(x), d²(sinh)/dx² = sinh(x)
        assert_array_almost_equal(g, [np.cosh(2), 0])
        assert_array_almost_equal(h, [[np.sinh(2), 0], [0, 0]])

    def test_cosh(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([2, 3])

        result = a.cosh()
        assert_almost_equal(result.value, np.cosh(2))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        # d(cosh)/dx = sinh(x), d²(cosh)/dx² = cosh(x)
        assert_array_almost_equal(g, [np.sinh(2), 0])
        assert_array_almost_equal(h, [[np.cosh(2), 0], [0, 0]])

    def test_tanh(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([2, 3])

        result = a.tanh()
        tanh_val = np.tanh(2)
        sech_sq = 1 - tanh_val**2
        assert_almost_equal(result.value, tanh_val)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        # d(tanh)/dx = sech²(x), d²(tanh)/dx² = -2·tanh(x)·sech²(x)
        assert_array_almost_equal(g, [sech_sq, 0])
        assert_array_almost_equal(h, [[-2*tanh_val*sech_sq, 0], [0, 0]])

    # inverse hyperbolic

    def test_arcsinh(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([2, 3])

        result = a.arcsinh()
        assert_almost_equal(result.value, np.arcsinh(2))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        # d(asinh)/dx = 1/sqrt(1+x²), d²(asinh)/dx² = -x/(1+x²)^(3/2)
        x = 2.0
        g1 = 1/np.sqrt(1 + x**2)
        h1 = -x / (1 + x**2)**1.5
        assert_array_almost_equal(g, [g1, 0])
        assert_array_almost_equal(h, [[h1, 0], [0, 0]])

    def test_arccosh(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([2, 3])

        result = a.arccosh()
        assert_almost_equal(result.value, np.arccosh(2))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        # d(acosh)/dx = 1/sqrt(x²-1), d²(acosh)/dx² = -x/(x²-1)^(3/2)
        x = 2.0
        g1 = 1/np.sqrt(x**2 - 1)
        h1 = -x / (x**2 - 1)**1.5
        assert_array_almost_equal(g, [g1, 0])
        assert_array_almost_equal(h, [[h1, 0], [0, 0]])

    def test_arctanh(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([0.5, 3])

        result = a.arctanh()
        assert_almost_equal(result.value, np.arctanh(0.5))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        # d(atanh)/dx = 1/(1-x²), d²(atanh)/dx² = 2x/(1-x²)²
        x = 0.5
        g1 = 1 / (1 - x**2)
        h1 = 2*x / (1 - x**2)**2
        assert_array_almost_equal(g, [g1, 0])
        assert_array_almost_equal(h, [[h1, 0], [0, 0]])

    # atan2

    def test_atan2(self):
        graph = hg.HyperGraph()

        y, x = graph.new_variables([3, 4])

        result = hg.atan2(y, x)
        assert_almost_equal(result.value, np.arctan2(3, 4))

        graph.compute(result)
        g = graph.g()
        h = graph.h(full=True)
        # ∂atan2/∂y = x/r², ∂atan2/∂x = -y/r²
        r2 = 3**2 + 4**2  # = 25
        assert_array_almost_equal(g, [4/r2, -3/r2])
        # ∂²atan2/∂y² = -2xy/r⁴
        # ∂²atan2/∂x² = 2xy/r⁴
        # ∂²atan2/∂y∂x = (y²-x²)/r⁴
        h_yy = -2*3*4 / r2**2
        h_xx = 2*3*4 / r2**2
        h_yx = (3**2 - 4**2) / r2**2
        assert_array_almost_equal(h, [[h_yy, h_yx], [h_yx, h_xx]])

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
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [0, 6, -5, 0, -3, 2])
        assert_array_almost_equal(h, [[0, 0, 0, 0,  0, 0],
                                      [0, 0, 0, 0,  0, 1],
                                      [0, 0, 0, 0, -1, 0],
                                      [0, 0, 0, 0,  0, 0],
                                      [0, 0, 0, 0,  0, 0],
                                      [0, 0, 0, 0,  0, 0]])

        graph.compute(result[1])
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [-6, 0, 4, 3, 0, -1])
        assert_array_almost_equal(h, [[0, 0, 0, 0, 0, -1],
                                      [0, 0, 0, 0, 0,  0],
                                      [0, 0, 0, 1, 0,  0],
                                      [0, 0, 0, 0, 0,  0],
                                      [0, 0, 0, 0, 0,  0],
                                      [0, 0, 0, 0, 0,  0]])

        graph.compute(result[2])
        g = graph.g()
        h = graph.h()
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
        g = graph.g()
        h = graph.h()
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
        g = graph.g()
        h = graph.h()
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

        h = graph.h(full=True)

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

        graph.g(out=g)
        graph.h(out=h)

        assert_array_almost_equal(g, [-17/np.sqrt(6), -np.sqrt(2/3), 13/np.sqrt(6), 4*np.sqrt(2/3), np.sqrt(2/3), -2*np.sqrt(2/3)])
        assert_array_almost_equal(h, [[77/(18*np.sqrt(6)), -77/(9*np.sqrt(6)), 77/(18*np.sqrt(6)), -8*np.sqrt(2/3)/9,  23/(9*np.sqrt(6)), -17*np.sqrt(2/3)/9],
                                      [0, 77*np.sqrt(2/3)/9, -77/(9*np.sqrt(6)), 41/(9*np.sqrt(6)), -32*np.sqrt(2/3)/9, 23/(9*np.sqrt(6))],
                                      [0, 0, 77/(18*np.sqrt(6)), np.sqrt(2/3)/9, 41/(9*np.sqrt(6)), -8*np.sqrt(2/3)/9],
                                      [0, 0, 0, 7/(9*np.sqrt(6)),  -7*np.sqrt(2/3)/9, 7/(9*np.sqrt(6))],
                                      [0, 0, 0, 0,  14*np.sqrt(2/3)/9, -7*np.sqrt(2/3)/9],
                                      [0, 0, 0, 0,  0, 7/(9*np.sqrt(6))]])


    # edge cases

    def test_abs_zero(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(0)

        result = abs(a)
        assert_equal(result.value, 0)

        graph.compute(result)
        g = graph.g()
        assert_array_equal(g, [0])

    def test_division_by_zero_raises(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 0])

        with self.assertRaises(ValueError):
            a / b

    def test_sqrt_zero_raises(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(0)

        with self.assertRaises(ValueError):
            np.sqrt(a)

    def test_sqrt_negative_raises(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(-1)

        with self.assertRaises(ValueError):
            np.sqrt(a)

    def test_log_zero_raises(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(0)

        with self.assertRaises(ValueError):
            a.log()

    def test_log_negative_raises(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(-1)

        with self.assertRaises(ValueError):
            a.log()

    def test_asin_boundary_raises(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(1)

        with self.assertRaises(ValueError):
            np.arcsin(a)

    def test_acos_boundary_raises(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(-1)

        with self.assertRaises(ValueError):
            np.arccos(a)

    def test_acosh_boundary_raises(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(1)

        with self.assertRaises(ValueError):
            a.arccosh()

    def test_atanh_boundary_raises(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(1)

        with self.assertRaises(ValueError):
            a.arctanh()

    def test_pow_zero_low_exponent_raises(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(0)

        with self.assertRaises(ValueError):
            a ** 0.5

    def test_atan2_origin_raises(self):
        graph = hg.HyperGraph()

        y, x = graph.new_variables([0, 0])

        with self.assertRaises(ValueError):
            hg.atan2(y, x)

    def test_atan2_x_zero_raises(self):
        graph = hg.HyperGraph()

        y, x = graph.new_variables([1, 0])

        with self.assertRaises(ValueError):
            hg.atan2(y, x)

    # log2, log10

    def test_log2(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([4, 3])

        result = a.log2()
        assert_almost_equal(result.value, np.log2(4))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        x = 4.0
        ln2 = np.log(2)
        assert_array_almost_equal(g, [1/(x * ln2), 0])
        assert_array_almost_equal(h, [[-1/(x**2 * ln2), 0], [0, 0]])

    def test_log2_module(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(8)

        result = hg.log2(a)
        assert_almost_equal(result.value, 3.0)

    def test_log2_nonpositive_raises(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(0)
        with self.assertRaises(ValueError):
            a.log2()

        b = graph.new_variable(-1)
        with self.assertRaises(ValueError):
            b.log2()

    def test_log10(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([100, 3])

        result = a.log10()
        assert_almost_equal(result.value, 2.0)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        x = 100.0
        ln10 = np.log(10)
        assert_array_almost_equal(g, [1/(x * ln10), 0])
        assert_array_almost_equal(h, [[-1/(x**2 * ln10), 0], [0, 0]])

    def test_log10_module(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(1000)

        result = hg.log10(a)
        assert_almost_equal(result.value, 3.0)

    def test_log10_nonpositive_raises(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(0)
        with self.assertRaises(ValueError):
            a.log10()

        b = graph.new_variable(-1)
        with self.assertRaises(ValueError):
            hg.log10(b)

    # erf, erfc

    def test_erf(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([1.5, 3])

        result = a.erf()
        assert_almost_equal(result.value, math.erf(1.5))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        x = 1.5
        two_over_sqrt_pi = 2.0 / np.sqrt(np.pi)
        exp_neg_x2 = np.exp(-x**2)
        g1 = two_over_sqrt_pi * exp_neg_x2
        h1 = -2.0 * x * g1
        assert_array_almost_equal(g, [g1, 0])
        assert_array_almost_equal(h, [[h1, 0], [0, 0]])

    def test_erf_module(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(0)

        result = hg.erf(a)
        assert_almost_equal(result.value, 0.0)

    def test_erfc(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([1.5, 3])

        result = a.erfc()
        assert_almost_equal(result.value, math.erfc(1.5))

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        x = 1.5
        two_over_sqrt_pi = 2.0 / np.sqrt(np.pi)
        exp_neg_x2 = np.exp(-x**2)
        g1 = -two_over_sqrt_pi * exp_neg_x2
        h1 = 2.0 * x * two_over_sqrt_pi * exp_neg_x2
        assert_array_almost_equal(g, [g1, 0])
        assert_array_almost_equal(h, [[h1, 0], [0, 0]])

    def test_erfc_module(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(0)

        result = hg.erfc(a)
        assert_almost_equal(result.value, 1.0)

    def test_erf_plus_erfc_is_one(self):
        """erf(x) + erfc(x) == 1 for all x"""
        graph = hg.HyperGraph()

        a = graph.new_variable(2.0)

        result_erf = a.erf()
        result_erfc = a.erfc()
        assert_almost_equal(result_erf.value + result_erfc.value, 1.0)

    # sigmoid

    def test_sigmoid(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([2, 3])

        result = a.sigmoid()
        x = 2.0
        sig = 1.0 / (1.0 + np.exp(-x))
        assert_almost_equal(result.value, sig)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        g1 = sig * (1 - sig)
        h1 = g1 * (1 - 2 * sig)
        assert_array_almost_equal(g, [g1, 0])
        assert_array_almost_equal(h, [[h1, 0], [0, 0]])

    def test_sigmoid_zero(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(0)

        result = a.sigmoid()
        assert_almost_equal(result.value, 0.5)

        graph.compute(result)
        g = graph.g()
        # sigmoid'(0) = 0.25
        assert_array_almost_equal(g, [0.25])

    def test_sigmoid_module(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(0)

        result = hg.sigmoid(a)
        assert_almost_equal(result.value, 0.5)

    # softplus

    def test_softplus(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([2, 3])

        result = a.softplus()
        x = 2.0
        sp = np.log(1 + np.exp(x))
        assert_almost_equal(result.value, sp)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        sig = 1.0 / (1.0 + np.exp(-x))
        g1 = sig
        h1 = sig * (1 - sig)
        assert_array_almost_equal(g, [g1, 0])
        assert_array_almost_equal(h, [[h1, 0], [0, 0]])

    def test_softplus_module(self):
        graph = hg.HyperGraph()

        a = graph.new_variable(0)

        result = hg.softplus(a)
        assert_almost_equal(result.value, np.log(2))

    # min, max

    def test_min_x_less(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([3, 7])

        result = hg.min(a, b)
        assert_almost_equal(result.value, 3)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [1, 0])
        assert_array_almost_equal(h, [[0, 0], [0, 0]])

    def test_min_y_less(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([7, 3])

        result = hg.min(a, b)
        assert_almost_equal(result.value, 3)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [0, 1])
        assert_array_almost_equal(h, [[0, 0], [0, 0]])

    def test_min_equal(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 5])

        result = hg.min(a, b)
        assert_almost_equal(result.value, 5)

        graph.compute(result)
        g = graph.g()
        # Subgradient convention: x gets the derivative when equal
        assert_array_almost_equal(g, [1, 0])

    def test_max_x_greater(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([7, 3])

        result = hg.max(a, b)
        assert_almost_equal(result.value, 7)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [1, 0])
        assert_array_almost_equal(h, [[0, 0], [0, 0]])

    def test_max_y_greater(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([3, 7])

        result = hg.max(a, b)
        assert_almost_equal(result.value, 7)

        graph.compute(result)
        g = graph.g()
        h = graph.h()
        assert_array_almost_equal(g, [0, 1])
        assert_array_almost_equal(h, [[0, 0], [0, 0]])

    def test_max_equal(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 5])

        result = hg.max(a, b)
        assert_almost_equal(result.value, 5)

        graph.compute(result)
        g = graph.g()
        # Subgradient convention: x gets the derivative when equal
        assert_array_almost_equal(g, [1, 0])

    # pow(Variable, Variable)

    def test_pow_variable_variable(self):
        graph = hg.HyperGraph()

        x, y = graph.new_variables([2, 3])

        result = hg.pow(x, y)
        assert_almost_equal(result.value, 8)

        graph.compute(result)
        g = graph.g()
        h = graph.h(full=True)
        xv, yv = 2.0, 3.0
        # ∂f/∂x = y·x^(y-1) = 3·4 = 12
        # ∂f/∂y = x^y·ln(x) = 8·ln(2)
        assert_array_almost_equal(g, [yv * xv**(yv - 1), xv**yv * np.log(xv)])
        # ∂²f/∂x² = y(y-1)·x^(y-2) = 3·2·1 = 6
        # ∂²f/∂y² = x^y·(ln(x))² = 8·(ln2)²
        # ∂²f/∂x∂y = x^(y-1)·(1 + y·ln(x)) = 4·(1 + 3·ln2)
        h_xx = yv * (yv - 1) * xv**(yv - 2)
        h_yy = xv**yv * np.log(xv)**2
        h_xy = xv**(yv - 1) * (1 + yv * np.log(xv))
        assert_array_almost_equal(h, [[h_xx, h_xy], [h_xy, h_yy]])

    def test_pow_variable_variable_via_operator(self):
        """Test x ** y where y is a Variable, not a double"""
        graph = hg.HyperGraph()

        x, y = graph.new_variables([3, 2])

        result = x ** y
        assert_almost_equal(result.value, 9)

    def test_pow_variable_variable_nonpositive_raises(self):
        graph = hg.HyperGraph()

        x, y = graph.new_variables([0, 2])
        with self.assertRaises(ValueError):
            hg.pow(x, y)

        x2, y2 = graph.new_variables([-1, 2])
        with self.assertRaises(ValueError):
            hg.pow(x2, y2)

    # hypot

    def test_hypot(self):
        graph = hg.HyperGraph()

        x, y = graph.new_variables([3, 4])

        result = hg.hypot(x, y)
        assert_almost_equal(result.value, 5)

        graph.compute(result)
        g = graph.g()
        h = graph.h(full=True)
        xv, yv = 3.0, 4.0
        hv = 5.0
        # ∂f/∂x = x/h = 3/5, ∂f/∂y = y/h = 4/5
        assert_array_almost_equal(g, [xv/hv, yv/hv])
        # ∂²f/∂x² = y²/h³ = 16/125
        # ∂²f/∂y² = x²/h³ = 9/125
        # ∂²f/∂x∂y = -xy/h³ = -12/125
        h_xx = yv**2 / hv**3
        h_yy = xv**2 / hv**3
        h_xy = -xv * yv / hv**3
        assert_array_almost_equal(h, [[h_xx, h_xy], [h_xy, h_yy]])

    def test_hypot_origin_raises(self):
        graph = hg.HyperGraph()

        x, y = graph.new_variables([0, 0])

        with self.assertRaises(ValueError):
            hg.hypot(x, y)

    def test_hypot_pythagorean(self):
        """Verify value for a well-known Pythagorean triple"""
        graph = hg.HyperGraph()

        x, y = graph.new_variables([5, 12])

        result = hg.hypot(x, y)
        assert_almost_equal(result.value, 13)


    # sparse Hessian

    def test_h_sparse_triplets_basic(self):
        """h_sparse_triplets returns correct (rows, cols, values) for upper triangle"""
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a * b
        graph.compute(result)

        rows, cols, values = graph.h_sparse_triplets()
        # a*b has H = [[0, 1], [0, 0]] (upper triangle only)
        # Only one non-zero entry: (0, 1) = 1
        assert_equal(len(rows), 1)
        assert_equal(len(cols), 1)
        assert_equal(len(values), 1)
        assert_equal(rows[0], 0)
        assert_equal(cols[0], 1)
        assert_almost_equal(values[0], 1.0)

    def test_h_sparse_triplets_full(self):
        """h_sparse_triplets with full=True returns both triangles"""
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a * b
        graph.compute(result)

        rows, cols, values = graph.h_sparse_triplets(full=True)
        # full=True: (0,1)=1 and (1,0)=1
        assert_equal(len(rows), 2)
        # Convert to a set of (row, col, val) for order-independent check
        entries = set(zip(rows, cols, values))
        self.assertIn((0, 1, 1.0), entries)
        self.assertIn((1, 0, 1.0), entries)

    def test_h_sparse_triplets_diagonal(self):
        """h_sparse_triplets includes diagonal entries"""
        graph = hg.HyperGraph()

        a, b = graph.new_variables([3, 5])

        result = a.square()
        graph.compute(result)

        rows, cols, values = graph.h_sparse_triplets()
        # d²(x²)/dx² = 2, only diagonal entry (0,0)
        assert_equal(len(rows), 1)
        assert_equal(rows[0], 0)
        assert_equal(cols[0], 0)
        assert_almost_equal(values[0], 2.0)

    def test_h_sparse_triplets_empty(self):
        """h_sparse_triplets returns empty lists when Hessian is zero"""
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a + b
        graph.compute(result)

        rows, cols, values = graph.h_sparse_triplets()
        assert_equal(len(rows), 0)
        assert_equal(len(cols), 0)
        assert_equal(len(values), 0)

    def test_h_sparse_matches_dense(self):
        """h_sparse().toarray() matches h(full=True) for a complex expression"""
        graph = hg.HyperGraph()

        ax, ay, az, bx, by, bz = graph.new_variables([1, 2, 3, 4, 5, 6])

        a = np.array([ax, ay, az])
        b = np.array([bx, by, bz])

        result = np.linalg.norm(np.cross(a, b))

        graph.compute(result)

        h_dense = graph.h(full=True)
        h_sparse = graph.h_sparse(full=True)

        assert_array_almost_equal(h_sparse.toarray(), h_dense)

    def test_h_sparse_upper_triangle_matches(self):
        """h_sparse() default (upper triangle) matches h() upper triangle"""
        graph = hg.HyperGraph()

        ax, ay, az, bx, by, bz = graph.new_variables([1, 2, 3, 4, 5, 6])

        a = np.array([ax, ay, az])
        b = np.array([bx, by, bz])

        result = np.linalg.norm(np.cross(a, b))

        graph.compute(result)

        h_dense_upper = graph.h(full=False)
        h_sparse_upper = graph.h_sparse(full=False)

        # The sparse version should have the upper triangle entries
        # When converted to dense, lower triangle should be zero
        sparse_dense = h_sparse_upper.toarray()
        assert_array_almost_equal(sparse_dense, h_dense_upper)

    def test_h_sparse_coo_format(self):
        """h_sparse(format='coo') returns COO matrix"""
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])
        result = a * b
        graph.compute(result)

        h_sparse = graph.h_sparse(format='coo')
        self.assertIsInstance(h_sparse, scipy.sparse.coo_matrix)

    def test_h_sparse_csc_format(self):
        """h_sparse(format='csc') returns CSC matrix (default)"""
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])
        result = a * b
        graph.compute(result)

        h_sparse = graph.h_sparse(format='csc')
        self.assertIsInstance(h_sparse, scipy.sparse.csc_matrix)

    def test_h_sparse_csr_format(self):
        """h_sparse(format='csr') returns CSR matrix"""
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])
        result = a * b
        graph.compute(result)

        h_sparse = graph.h_sparse(format='csr')
        self.assertIsInstance(h_sparse, scipy.sparse.csr_matrix)

    def test_h_sparse_default_is_csc(self):
        """h_sparse() default format is CSC"""
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])
        result = a * b
        graph.compute(result)

        h_sparse = graph.h_sparse()
        self.assertIsInstance(h_sparse, scipy.sparse.csc_matrix)

    def test_h_sparse_invalid_format_raises(self):
        """h_sparse with invalid format raises ValueError"""
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])
        result = a * b
        graph.compute(result)

        with self.assertRaises(Exception):
            graph.h_sparse(format='invalid')

    def test_h_sparse_shape(self):
        """h_sparse returns matrix with correct shape"""
        graph = hg.HyperGraph()

        a, b, c = graph.new_variables([1, 2, 3])
        result = a * b + b * c
        graph.compute(result)

        h_sparse = graph.h_sparse()
        assert_equal(h_sparse.shape, (3, 3))

    def test_h_sparse_multiplication(self):
        """h_sparse values match dense h for multiplication"""
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])
        result = a * b
        graph.compute(result)

        h_dense = graph.h(full=True)
        h_sparse = graph.h_sparse(full=True)

        assert_array_almost_equal(h_sparse.toarray(), h_dense)

    def test_num_variables(self):
        """num_variables property returns correct count"""
        graph = hg.HyperGraph()

        assert_equal(graph.num_variables, 0)

        graph.new_variable(1)
        assert_equal(graph.num_variables, 1)

        graph.new_variables([2, 3, 4])
        assert_equal(graph.num_variables, 4)


if __name__ == '__main__':
    unittest.main()
