import math
import unittest
from engine import Value


class TestValueEngine(unittest.TestCase):

    def test_addition(self):
        a, b = Value(4), Value(6)
        c = a + b
        c.backward()
        self.assertEqual(c.data, 10)
        self.assertEqual(a.grad, 1)
        self.assertEqual(b.grad, 1)

    def test_multiplication(self):
        a, b = Value(4), Value(6)
        c = a * b
        c.backward()
        self.assertEqual(c.data, 24)
        self.assertEqual(a.grad, 6)
        self.assertEqual(b.grad, 4)

    def test_subtraction(self):
        a, b = Value(4), Value(3)
        c = a - b
        c.backward()
        self.assertEqual(c.data, 1)
        self.assertEqual(a.grad, 1)
        self.assertEqual(b.grad, -1)

    def test_power(self):
        a = Value(3)
        b = a ** 2
        b.backward()
        self.assertEqual(b.data, 9)
        self.assertEqual(a.grad, 6)

    def test_division(self):
        a, b = Value(6), Value(3)
        c = a / b
        c.backward()
        self.assertEqual(c.data, 2)
        self.assertAlmostEqual(a.grad, 1 / 3)
        self.assertAlmostEqual(b.grad, -2 / 3)

    def test_tanh(self):
        a = Value(2)
        b = a.tanh()
        b.backward()
        self.assertAlmostEqual(b.data, math.tanh(2))
        self.assertAlmostEqual(a.grad, 1 - math.tanh(2) ** 2)

    def test_chain_rule(self):
        a, b = Value(2), Value(3)
        d = (a * b).tanh()
        d.backward()
        factor = 1 - math.tanh(6) ** 2
        self.assertAlmostEqual(a.grad, factor * 3)
        self.assertAlmostEqual(b.grad, factor * 2)

    def test_reverse_addition(self):
        a = Value(5)
        b = 2 + a
        b.backward()
        self.assertEqual(b.data, 7)
        self.assertEqual(a.grad, 1)

    def test_reverse_multiplication(self):
        a = Value(5)
        b = 2 * a
        b.backward()
        self.assertEqual(b.data, 10)
        self.assertEqual(a.grad, 2)


if __name__ == "__main__":
    unittest.main()
