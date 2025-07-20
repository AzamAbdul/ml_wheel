class Matrix():
    def __init__(self, n, m):
        self.rows = n
        self.cols = m
        self.matrix = [[0 for _ in range(m)] for _ in range(n)]
    
    def print(self):
        print(f"Num rows: {self.rows} Num Cols: {self.cols}")
        for row in self.matrix:
            print(row)

    def set_val_at(self, i, j, val):
        if i >= self.rows or i < 0:
            return False
        if j >= self.cols or j <0:
            return False
        self.matrix[i][j] = val
        return True

    def get_val_at(self, i, j):
        if i >= self.rows or i <0:
            return None
        if j >= self.cols or j<0:
            return None
        return self.matrix[i][j]

    def get_rows(self):
        return self.rows
    
    def get_cols(self):
        return self.cols

    def add(m1, m2, subtract=False):
        sign = 1
        if subtract:
            sign = -1
        if m1.get_rows() != m2.get_rows():
            return None
        if m1.get_cols() != m2.get_cols():
            return None
        result = Matrix(m1.get_rows(), m1.get_cols())
        for i in range(0, m1.get_rows()):
            for j in range(0, m1.get_cols()):
                v1 = m1.get_val_at(i,j)
                v2 = sign * m2.get_val_at(i,j)
                result.set_val_at(i,j, v1 + v2)
        return result
    
    def multiply(m1, m2):
        if m1.get_cols() != m2.get_rows():
            return None
        num_rows = m1.get_rows()
        num_cols = m2.get_cols()
        result = Matrix(num_rows, num_cols)
        for i in range(num_rows):
            for j in range(num_cols):
                res = 0
                for k in range(m2.get_rows()):
                    res += m1.get_val_at(i, k) * m2.get_val_at(k, j)
                result.set_val_at(i, j, res)
        return result

    def transpose(m1):
        prior_num_rows = m1.get_rows()
        prior_num_cols = m1.get_cols()
        
        new_num_rows = prior_num_cols
        new_num_cols = prior_num_rows

        result = Matrix(new_num_rows, new_num_cols)

        for i in range(prior_num_rows):
            for j in range(prior_num_cols):
                v = m1.get_val_at(i,j)
                result.set_val_at(j, i, v)
        return result



m1 = Matrix(2, 2)
m1.set_val_at(0,0,1)
m1.set_val_at(1,1,3)
m1.print()

m2 = Matrix(2, 2)
m2.set_val_at(0,0,1)
m2.set_val_at(1,1,4)
m2.set_val_at(0,1,10)
m2.set_val_at(1,0, 3)
m2.print()

print("m1 + m2")
Matrix.add(m1, m2).print()
print("m1 - m2")
Matrix.add(m1, m2, True).print()

print("m1 * m2")
m3 = Matrix.multiply(m1, m2)
m3.print()


print("transpose m3")

Matrix.transpose(m3).print()
