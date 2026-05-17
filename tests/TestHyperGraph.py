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


if __name__ == '__main__':
    unittest.main()
