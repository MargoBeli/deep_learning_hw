import unittest
import torch


class Element:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Element(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        if not isinstance(other, Element):
            return NotImplemented

        def _backward():
            self.grad += 1 * other.grad
            other.grad += 1 * self.grad

        result = Element(self.data + other.data, (self, other), _op='+')
        result._backward = _backward
        return result

    def __mul__(self, other):
        if not isinstance(other, Element):
            return NotImplemented

        def _backward():
            self.grad += other.data * self.grad
            other.grad += self.data * self.grad

        result = Element(self.data * other.data, (self, other), _op='*')
        result._backward = _backward
        return result

    def relu(self):
        def _backward():
            self.grad += 1 * (self.data > 0) * self.grad

        result = Element(max(0, self.data), (self,), _op='relu')
        result._backward = _backward
        return result

    def backward(self):
        self.grad += 1
        visited = set()
        nodes = [self]

        while nodes:
            node = nodes.pop()
            if node not in visited:
                visited.add(node)
                node._backward()
                nodes.extend(node._prev)

class TestElement(unittest.TestCase):
    def setUp(self):
        self.a = Element(2.0)
        self.b = Element(-3.0)
        self.c = Element(10.0)

    def test_addition(self):
        d = self.a + self.b
        d.backward()

        # Проверка значений
        self.assertEqual(d.data, 2.0 + (-3.0))

    def test_multiplication(self):
        d = self.b * self.c
        d.backward()

        # Проверка значений
        self.assertEqual(d.data, (-3.0) * 10.0)

    def test_relu(self):
        d = self.a.relu()
        d.backward()

        # Проверка значений
        self.assertEqual(d.data, max(0, 2.0))

        e = Element(-2.0).relu()
        e.backward()

        self.assertEqual(e.data, 0)
        self.assertEqual(e.grad, 1)

    def test_combined_operations(self):
        d = self.a + self.b * self.c
        d.backward()

        # Проверка значений
        self.assertEqual(d.data, 2.0 + (-3.0) * 10.0)

        e = d.relu()
        e.backward()
        self.assertEqual(e.data, 0)
        self.assertEqual(e.grad, 1)

    def test_with_torch_autograd(self):
        a_torch = torch.tensor(2.0, requires_grad=True)
        b_torch = torch.tensor(-3.0, requires_grad=True)
        c_torch = torch.tensor(10.0, requires_grad=True)

        d_torch = a_torch + b_torch * c_torch
        d_torch.relu().backward()

        self.assertAlmostEqual(d_torch.item(), (self.a + self.b * self.c).data)
        self.assertAlmostEqual(a_torch.grad.item(), self.a.grad)
        self.assertAlmostEqual(b_torch.grad.item(), self.b.grad)
        self.assertAlmostEqual(c_torch.grad.item(), self.c.grad)


if __name__ == '__main__':
    unittest.main()
