import math
class Vector():
    def __init__(self, n):
        self.vector = [0 for _ in range(n)]

    def size(self):
        return len(self.vector)

    def get(self, i):
        if i >= len(self.vector) or i < 0:
            return None
        return self.vector[i]

    def set(self, i, v):
        if i >= len(self.vector) or i < 0:
            return None
        self.vector[i] = v

    def dot_product(v1, v2):
        if v1.size() != v2.size():
            return None
        sum = 0 
        for i in range(v1.size()):
            sum += v1.get(i) * v2.get(i)
        return sum

    def l2_norm(self):
        sum = 0
        for i in range(self.size()):
            sum += self.get(i) ** 2
        return math.sqrt(sum)

    def print(self):
        print(self.vector)

v = Vector(5)
v.set(1, 1)
v.set(2, 2)
v.print()

print(Vector.dot_product(v, v))
print(v.l2_norm())
