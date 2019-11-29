import unittest
import hypergraph as hg
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

        g, h = graph.derive(result, [a, b])
        assert_array_equal(g, [1, 1])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_addition_variable_constant(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a + 6
        assert_equal(result.value, 11)

        g, h = graph.derive(result, [a, b])
        assert_array_equal(g, [1, 0])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_addition_constant_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = 5 + b
        assert_equal(result.value, 11)

        g, h = graph.derive(result, [a, b])
        assert_array_equal(g, [0, 1])
        assert_array_equal(h, [[0, 0], [0, 0]])

    # subtraction

    def test_subtraction_variable_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a - b
        assert_equal(result.value, -1)

        g, h = graph.derive(result, [a, b])
        assert_array_equal(g, [1, -1])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_subtraction_variable_constant(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a - 6
        assert_equal(result.value, -1)

        g, h = graph.derive(result, [a, b])
        assert_array_equal(g, [1, 0])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_subtraction_constant_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = 5 - b
        assert_equal(result.value, -1)

        g, h = graph.derive(result, [a, b])
        assert_array_equal(g, [0, -1])
        assert_array_equal(h, [[0, 0], [0, 0]])

    # multiplication

    def test_multiplication_variable_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a * b
        assert_equal(result.value, 30)

        g, h = graph.derive(result, [a, b])
        assert_array_equal(g, [6, 5])
        assert_array_equal(h, [[0, 1], [0, 0]])

    def test_multiplication_variable_constant(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a * 6
        assert_equal(result.value, 30)

        g, h = graph.derive(result, [a, b])
        assert_array_equal(g, [6, 0])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_multiplication_constant_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = 5 * b
        assert_equal(result.value, 30)

        g, h = graph.derive(result, [a, b])
        assert_array_equal(g, [0, 5])
        assert_array_equal(h, [[0, 0], [0, 0]])

    # division

    def test_division_variable_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a / b
        assert_almost_equal(result.value, 5 / 6)

        g, h = graph.derive(result, [a, b])
        assert_array_almost_equal(g, [1/6, -5/36])
        assert_array_almost_equal(h, [[0, -1/36], [0, 5/108]])

    def test_division_variable_constant(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = a / 6
        assert_almost_equal(result.value, 5 / 6)

        g, h = graph.derive(result, [a, b])
        assert_array_almost_equal(g, [1/6, 0])
        assert_array_equal(h, [[0, 0], [0, 0]])

    def test_division_constant_variable(self):
        graph = hg.HyperGraph()

        a, b = graph.new_variables([5, 6])

        result = 5 / b
        assert_almost_equal(result.value, 5 / 6)

        g, h = graph.derive(result, [a, b])
        assert_array_almost_equal(g, [0, -5/36])
        assert_array_almost_equal(h, [[0, 0], [0, 5/108]])


if __name__ == '__main__':
    unittest.main()
