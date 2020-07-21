import numpy as np

def rref(A):
    """ Reduced row echelon form (Gauss-Jordan elimination) """

    m,n = A.shape

    for i in range(m):
        # Search for maximum element in this column
        maxEl,maxRow = max( (val, row + i) for (row, val) in enumerate(abs(A[i:,i])) )

        # pivot maximum row with current row
        if i != maxRow:
            A[i], A[maxRow] = A[maxRow], A[i].copy()

        # scale the ith so that the pivoting element becomes 1
        A[i,i:] /= A[i,i]
        
        # eliminate
        A[range(i) + range(i+1, m), i:] = map(lambda row: row - A[i,i:] * row[0], A[range(i) + range(i+1, m), i:])
        
    return A

