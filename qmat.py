# Fast Rational & Integer Matrix Math
# Skye Rhomberg

import gmpy2 as g
import re

#######################################################################################
# Rational Matrix Class

class Q_Matrix:
    #################################################################
    # Internal Methods

    def _read(self, fname):
        '''
        Read Matrix array from file
        Rows separated by \n, cols separated by space
        Input:
        fname: str. name of txt file to read matrix from
        Output:
        list of lists. array read from file
        '''
        with open(fname) as f:
            return [[g.mpq(n) for n in line.strip().split()] for line in f.readlines()]

    def _row(self, idx):
        '''
        Get row idx of array
        Input:
        idx: int. row index
        Output:
        tuple. row i of array
        '''
        return self._arr[idx]

    def _col(self, idx):
        '''
        Get col idx of array
        Input:
        idx: int. col index
        Output:
        1d list. col i of array
        '''
        return [r[idx] for r in self._arr]

    def _transpose(self):
        '''
        Matrix Transpose
        Output:
        Q_Matrix. transpose of matrix: rows are cols
        '''
        return Q_Matrix([[self._arr[i][j] for i in range(len(self._arr))]\
                for j in range(len(self._arr[0]))])

    def _broadcast_to(self, new_shape):
        '''
        Broadcast matrix to the given shape by repeating columns and rows
        Singleton rows/columns may be expanded, non-singleton dimensions must match
        Input:
        new_shape: tuple (int, int). n_rows, n_cols to broadcast to
        '''
        # Copy matrix array
        B = Q_Matrix(self._arr)
        # if this is a row vector, copy the rows up to new shape
        if self.shape[0] == 1:
            B._arr *= new_shape[0]
        # if this is a col vector, copy the cols up to new shape
        if self.shape[1] == 1:
            B._arr = tuple(r * new_shape[1] for r in B._arr)
        # If matrix could not be broadcast to given new shape, raise exception
        assert B.shape == new_shape,\
                f"Matrix of shape {self.shape} can't be broacast to {new_shape}"
        # Return broadcast matrix
        return B

    def _inverse(self):
        '''
        Matrix Inverse
        Output:
        Q_Matrix. inverse of matrix
        '''
        # Only bijective matrices invertible
        assert self.shape[0] == self.shape[1] == self.rank,\
                f'Matrix of shape {self.shape} and rank {self.rank} is not invertible'
        # Matrix inverse is the product of its elementary matrices in reverse order
        rr_elems = row_reduce(self)['elems']
        return mat_prod(*rr_elems)

    def _determinant(self):
        '''
        Matrix determinant computed as product of dets of elementary matrices
        Output:
        int. determinant of matrix
        '''
        # Empty determinant is 1
        d = 1
        rr = row_reduce(self)
        spi = rr['A'].rcef
        # If spi isn't square, -1, otherwise spi size
        sq = spi.shape[0] if spi.shape[0] == spi.shape[1] else -1
        # A matrix isn't invertible (det 0) the rank of its SPI doesn't have maximal rank
        # or it's not square
        inv = spi.rank == sq
        for n in rr['dets']:
            d *= n
        # Return product of determinants or 0 if not invertible
        return inv * d

    def _widening(self):
        '''
        Widen an injective matrix to invertible
        Output:
        Q_Matrix. invertible matrix containing columns of original
        '''
        # Only injective matrices can be widened
        assert self.rank == self.shape[1] and self.shape[0] >= self.shape[1],\
                f"Matrix of shape {self.shape} and rank {self.rank} is not injective"
        # Widening will be product of elementary matrices needed to produce RREF
        rr_elems = row_reduce(self)['elems']
        return ~mat_prod(*rr_elems[::-1])

    def _narrowing(self):
        '''
        Narrowing to injective matrix
        Output:
        Q_Matrix. injective matrix with same range as input
        '''
        r = self.rref
        # take pivot columns in RREF and return their transpose
        n = [self._col(i) for i in range(self.shape[1])\
                if r._col(i).count(1) == 1 and all(x == 0 or x == 1 for x in r._col(i))]
        return Q_Matrix(n).t

    def _nullspace(self):
        '''
        Nullspace (kernel) of matrix
        Output:
        Q_Matrix. basis for the kernel of matrix
        '''
        # Remove null rows from RREF(A) to produce R
        R = Q_Matrix([r for r in self.rref._arr if any (c != 0 for c in r)])
        # Amalgamate with I_m below to produce [ R   ]
        #                                      [ I_m ]
        RI = vcat(R,eye(R.shape[1]))
        for i,row in enumerate(RI._arr):
            # Stop once we finish R, before I_m
            if i == R.shape[0]:
                break
            # Location of next pivot column as offset from current
            pivot = next((n for n,col in enumerate(row[i:]) if col == 1), None)
            # Swap 1 into current location
            RI *= elem_swap(i,i+pivot,RI.shape[1])
        # RI now equals [ I_r B ]
        #               [ P     ]
        B = RI[:R.shape[0],R.shape[0]:]
        P = RI[R.shape[0]:,:]
        # Return P[ -B    ]
        #         [ I_r-m ]
        return P * vcat(-B, eye(B.shape[1]))

    #################################################################
    # Constructor & I/O

    def __init__(self, arr=[], fname=''):
        '''
        Constructor: make matrix from number(s) or read from a filename
        Input:
        arr: int/mpq or 1d or 2d list of int or mpq. array of matrix
        fname: str. txt file to read matrix from
        '''
        if arr:
            # Given a single number, make a 1x1 matrix
            if isinstance(arr, (int, g.mpq)):
                self._arr = ((g.mpq(arr),),)
            # Given a 1d list, make a col vector, each element in new row
            elif isinstance(arr[0], (int, g.mpq)):
                self._arr = (tuple(g.mpq(n) for n in arr),)
            # Given a 2d list, read it into the array in current dims
            else:
                self._arr = tuple(tuple(g.mpq(n) for n in row) for row in arr)
        elif fname:
            self._arr = self._read(fname)
        else:
            # Default is [[0]]
            self._arr = ((g.mpq(0,1),),)

        # Hidden Properties Uninitialized
        self._shape = None
        self._t = None
        self._inv = None
        self._rref = None
        self._rcef = None
        self._spi = None
        self._rank = None
        self._wide = None
        self._narrow = None
        self._ker = None
        self._det = None
        self._tex = None
        self._val = None

    def __getitem__(self, idx):
        '''
        Numpy-like 2d indexing
        Input:
        idx. int or slice or pair of slices
        '''
        # Turn falsy values to None
        n = lambda i: i if i else None
        # If idx is a singletone tuple, i.e. matrix[n,]
        # Extract value to force int case
        if isinstance(idx, tuple) and len(idx) == 1:
            idx = idx[0]
        # Given a single int or slice, slice or index the proper row(s)
        if isinstance(idx, (int, slice)):
            return Q_Matrix(self._row(idx))
        # Given a tuple of indices, e.g. matrix[a:b,c:d]
        r, c = idx
        # If row is an int, slice that row by c
        if isinstance(r, int):
            return Q_Matrix(self._row(r)[c])
        else:
            # If c is an int, for each row in the slice, create a trivial slice
            if isinstance(c, int):
                return Q_Matrix([row[slice(c,n(c+1))] for row in self._row(r)])
            # Otherwise, just slice each row by c
            return Q_Matrix([row[c] for row in self._row(r)])

    def __repr__(self):
        return ascii_str(self)

    def __str__(self):
        return unicode_str(self)

    def __iter__(self):
        # Matrix iterates over the rows
        return iter([Q_Matrix(r) for r in self._arr])

    def __contains__(self, val):
        return g.mpq(val) in self._arr

    #################################################################
    # Basic Arithmetic Operations

    def __add__(self, other):
        if isinstance(other, Q_Matrix):
            return _add(self, other)
        if isinstance(other, (int,g.mpq,list,tuple)):
            return _add(self, Q_Matrix(other))
        raise ValueError(f'Unsupported Types for +: {type(self)} and {type(other)}')

    def __sub__(self, other):
        if isinstance(other, Q_Matrix):
            return _add(self, -other)
        if isinstance(other, (int,g.mpq,list,tuple)):
            return _add(self, -Q_Matrix(other))
        raise ValueError(f'Unsupported Types for -: {type(self)} and {type(other)}')

    def __mul__(self, other):
        if isinstance(other, Q_Matrix):
            if (1,1) in (self.shape, other.shape):
                return _mul(self,other)
            assert self.shape[1] == other.shape[0],\
                    f"Matrices of shape {self.shape} and {other.shape} can't be multiplied"
            return _matmul(self,other)
        if isinstance(other, (int, g.mpq)):
            return _mul(self, Q_Matrix(other))
        raise ValueError(f'Unsupported Types for *: {type(self)} and {type(other)}')
                
    def __pow__(self, other):
        if isinstance(other, int):
            if other < 0:
                return mat_prod(*[self.inv]*-other)
            return mat_prod(*[self]*other)*eye(self.shape[0])
        raise ValueError(\
                f'Unsupported Types for ** or pow(): {type(self)} and {type(other)}')

    #################################################################
    # Reverse Arithmetic Operations

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __rmul__(self, other):
        if isinstance(other, (int, g.mpq)):
            return (self.t.__mul__(Q_Matrix(other).t)).t
        return (self.t.__mul__(other.t)).t

    #################################################################
    # Arithmetic Assignments

    def __iadd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __ipow__(self, other):
        return self.__pow__(other)

    #################################################################
    # Unary Operations

    def __neg__(self):
        return Q_Matrix([[-c for c in r] for r in self._arr])

    def __pos__(self):
        return self

    def __invert__(self):
        return self.inv

    #################################################################
    # Matrix Properties

    # Shape
    @property
    def shape(self):
        if self._shape is None:
            self._shape = (len(self._arr),len(self._arr[0]))
        return self._shape

    # Transpose
    @property
    def t(self):
        if self._t is None:
            self._t = self._transpose()
        return self._t

    # Inverse
    @property
    def inv(self):
        if self._inv is None:
            self._inv = self._inverse()
        return self._inv

    # RREF
    @property
    def rref(self):
        if self._rref is None:
            self._rref = row_reduce(self)['A']
        return self._rref

    # RCEF
    @property
    def rcef(self):
        if self._rcef is None:
            self._rcef = row_reduce(self.t)['A'].t
        return self._rcef

    # SPI
    @property
    def spi(self):
        if self._spi is None:
            self._spi = self.rref.rcef
        return self._spi

    # Rank
    @property
    def rank(self):
        if self._rank is None:
            self._rank = int(sum([c for r in self.spi._arr for c in r]))
        return self._rank

    # Widening
    @property
    def wide(self):
        if self._wide is None:
            self._wide = self._widening()
        return self._wide

    # Narrowing
    @property
    def narrow(self):
        if self._narrow is None:
            self._narrow = self._narrowing()
        return self._narrow

    # Nullspace
    @property
    def ker(self):
        if self._ker is None:
            self._ker = self._nullspace()
        return self._ker

    # Determinant
    @property
    def det(self):
        if self._det is None:
            self._det = self._determinant()
        return self._det

    # LaTeX Representation
    @property
    def tex(self):
        if self._tex is None:
            self._tex = tex_str(self)
        return self._tex

    # Number value of singleton matrix
    @property
    def val(self):
        if self._val is None:
            if not (len(self._arr) == len(self._arr[0]) == 1):
                raise TypeError("{0}x{1} Matrix can't be expressed as a number"
                        .format(len(self._arr),len(self._arr[0])))
            self._val = g.mpq(self._arr[0][0])
        return self._val

#######################################################################################
# Matrix Generators & Operators

def _add(m1, m2):
    '''
    Element-wise matrix addition
    Input:
    m1, m2: Q_Matrix. addends
    Output:
    Q_Matrix. sum
    '''
    # Broadcast to singleton rows/cols up until matrices the same size
    new_shape = tuple(max(m1.shape[i],m2.shape[i]) for i in range(2))
    a1, a2 = m1._broadcast_to(new_shape), m2._broadcast_to(new_shape)
    # Add values of each corresponding element
    return Q_Matrix([[a1[i,j].val + a2[i,j].val for j in range(new_shape[1])]\
            for i in range(new_shape[0])])

def _mul(m1, m2):
    '''
    Element-wise multiplication (Hadamard Product)
    Input:
    m1, m2: Q_Matrix. factors
    Output:
    Q_Matrix. product
    '''
    # Broadcast to singleton rows/cols up until matrices the same size
    new_shape = tuple(max(m1.shape[i],m2.shape[i]) for i in range(2))
    a1, a2 = m1._broadcast_to(new_shape), m2._broadcast_to(new_shape)
    # Multiply values of each corresponding element
    return Q_Matrix([[a1[i,j].val * a2[i,j].val for j in range(new_shape[1])]\
            for i in range(new_shape[0])])

def mat_sum(*ms):
    '''
    Sum arbitrary number of matrices
    Input:
    ms: list of Q_Matrix. addends
    Output:
    Q_Matrix. sum
    '''
    s = Q_Matrix(0)
    for m in ms: 
        s += m
    return s

def _dot(m1, m2):
    '''
    Dot product of a matrix with a vector
    Input:
    m1, m2: Q_Matrix. factors
    Output:
    Q_Matrix. product
    '''
    # Broadcast to singleton rows/cols up until matrices the same size
    new_shape = tuple(max(m1.shape[i],m2.shape[i]) for i in range(2))
    a1, a2 = m1._broadcast_to(new_shape), m2._broadcast_to(new_shape)
    # Multiply corresponding elements, and then sum along minor axis
    return [sum([a1[i,j].val * a2[i,j].val for j in range(new_shape[1])])\
            for i in range(new_shape[0])]

def _matmul(m1, m2):
    '''
    Matrix Multiplication
    Input:
    m1, m2: Q_Matrix. factors
    Output:
    Q_Matrix. product
    '''
    # Columns of product are the dot products of the rows of m1 with the cols of m2
    return Q_Matrix([_dot(m1, r) for r in m2.t]).t

def mat_prod(*ms):
    '''
    Matrix Product of arbitrary number of matrices
    Input:
    ms: list of Q_Matrix. factors
    Output:
    Q_Matrix. product
    '''
    # Empty product is the 1x1 identity matrix
    s = Q_Matrix(1)
    for m in ms:
        s *= m
    return s

def zero(shape):
    '''
    Fill a matrix of the given shape with zeros
    Input:
    shape: tuple (int, int). n_rows, n_cols to fill with zero
    Output:
    Q_Matrix. zero matrix of given shape
    '''
    return Q_Matrix([[0 for j in range(shape[1])] for i in range(shape[0])])

def eye(size):
    '''
    Identity matrix of given size
    Input:
    size: int. number of rows/cols in matrix
    Output:
    Q_Matrix. sizexsize identity matrix
    '''
    return Q_Matrix([[1 if i==j else 0 for j in range(size)]\
            for i in range(size)])

def elem_swap(a, b, size):
    '''
    Elementary swap matrix -- swap rows/cols a,b
    Input:
    a, b: int. rows/cols to switch
    size: int. number of rows/cols in swap matrix
    Output:
    Q_Matrix. sizexsize swap matrix for rows/cols a,b
    '''
    s = list(eye(size)._arr)
    s[a], s[b] = s[b], s[a]
    return Q_Matrix(s)

def elem_scale(k, a, size):
    '''
    Elementary scale matrix -- scale row/col a by k
    Input:
    k: int. scalar multiple
    a: int. row/col to scale
    size: int. number of rows/cols in scale matrix
    Output:
    Q_Matrix. sizexsize scale matrix for row/col a by k
    '''
    return Q_Matrix([[(k if i==a else 1) if j==i else 0 for j in range(size)]\
            for i in range(size)])

def elem_shear(a, k, b, size):
    '''
    Elementary shear matrix -- shear row a by s*b
    Input:
    a: int. row/col to shear
    k: int. scalar multiple
    b: int. row/col to shear by
    Output:
    Q_Matrix. sizexsize shear matrix for row/col a by k*b
    '''
    return eye(size) + Q_Matrix([[k if i==a and j==b else 0 for j in range(size)]\
            for i in range(size)])

def row_reduce(M):
    '''
    Row Reduction -- Gauss-Jordan Elimination
    Input:
    M: Q_Matrix
    Output:
    dict. 'A': RREF, 'elems': elementary matrices used, 'dets': determinants of these
    '''
    # Copy of input matrix
    A = Q_Matrix(M._arr)
    # Initialize elementary, determinant lists
    elems, dets = [], []
    n, m = M.shape 
    # Loop over smaller dimension
    for i in range(min(m,n)):
        j = i
        # Avoid null columns
        while all(x==0 for x in M[i:,j].t._arr):
            j += 1
            if j >= m:
                return {'A':A, 'elems':elems[::-1], 'dets':dets[::-1]}
        # Pivot will be next non-zero entry in column
        pivot = next((k for k,x in enumerate([r[j] for r in A._arr[i:]]) if x),None)
        if pivot is None:
            return {'A':A, 'elems':elems[::-1], 'dets':dets[::-1]}
        # If we had to move to find the pivot
        if pivot != 0:
            # Swap non-zero entry into current row
            swap = elem_swap(i, i+pivot, n)
            elems += [swap]
            # Determinant will switch sign
            dets += [-1]
            A = swap * A
        # If element at current idx is not a 1
        if A._arr[i][j] != 1:
            # Scale to make pivot 1
            scale = elem_scale(1/A._arr[i][j], i, n)
            elems += [scale]
            # Determinant scaled by this factor
            dets += [A._arr[i][j]]
            A = scale * A
        # Shear the other rows in this col to zeroes
        for k in range(n):
            if k != i:
                shear = elem_shear(k, -1*A._arr[k][j], i, n)
                elems += [shear]
                # Determinant doesn't change
                dets += [1]
                A = shear * A
    # Elementaries and dets were computed in reverse order
    return {'A':A, 'elems':elems[::-1], 'dets':dets[::-1]}

def _vcat(a, b):
    '''
    Vertical concatenation -- stack top-to-bottom
    Internal method: Check for dimension alignment
    '''
    assert len(a._arr[0]) == len(b._arr[0]), f"Can't vcat matrix of {len(a._arr[0])} cols and matrix of {len(b._arr[0])} cols"
    return Q_Matrix(a._arr + b._arr)

def _hcat(a, b):
    '''
    Horizontal concatenation -- stack left-to-right
    Internal method: Check for dimension alignment
    '''
    assert len(a._arr) == len(b._arr), f"Can't hcat matrix of {len(a._arr)} rows and matrix of {len(b._arr)} rows"
    return Q_Matrix([a._arr[i] + b._arr[i] for i in range(len(a._arr))])

def vcat(*ms):
    '''
    Vertical concatenation -- stack top-to-bottom
    Input:
    ms: list of Q_Matrix.
    Output:
    vertical concat of all ms
    '''
    if len(ms) < 2:
        return ms[0]
    return vcat(_vcat(ms[0],ms[1]),*ms[2:])

def hcat(*ms):
    '''
    Horizontal concatenation -- stack left-to-right
    Input:
    ms: list of Q_Matrix.
    Output:
    horizontal concat of all ms
    '''
    if len(ms) < 2:
        return ms[0]
    return hcat(_hcat(ms[0],ms[1]),*ms[2:])

#######################################################################################
# Matrix Pretty-Printing

# Unicode Parentheses
_parends = {
        'l_square':['\u23A1','\u23A2','\u23A3'], 'r_square':['\u23A4','\u23A5','\u23A6'],
        'l_round':['\u239B','\u239C','\u239D'], 'r_round':['\u239E','\u239F','\u23A0'],
        'l_curly_2':['\u23B0','\u23B1'], 'r_curly_2':['\u23B1','\u23B0'],
        'l_curly_0':['\u23A7','\u23AA','\u23AD','\u23AB','\u23AA','\u23A9'],
        'r_curly_0':['\u23AB','\u23AA','\u23A9','\u23A7','\u23AA','\u23AD'],
        'l_curly_1':['\u23A7','\u23AA','\u23A8','\u23AA','\u23A9'],
        'r_curly_1':['\u23AB','\u23AA','\u23AC','\u23AA','\u23AD'],
        'v_line':'\u23AE'
        }

# Single-Line Parends for Base Case
_single_parends = {
        'l_square':'[','r_square':']','l_round':'(','r_round':')',
        'l_curly':'{','r_curly':'}'
        }

# LaTeX prepends: bmat, pmat, Bmat:w
_tex_parends = {'square':'b', 'round':'p', 'curly':'B'}

def _gen_parend(side, mode, length):
    '''
    Generate a parend of given side (L/R), mode (round,square), and length
    '''
    if length == 1:
        return [_single_parends[f'{side}_{mode}']]
    idxs = [0] + [1 for i in range(length-2)] +[-1]
    return [_parends[f'{side}_{mode}'][x] for x in idxs]

def _gen_curly_brace(side, length):
    '''
    Generate a curly brace of given side, length
    '''
    ls = lambda s: [s] if isinstance(s,str) else s
    if length == 1:
        return [_single_parends[f'{side}_curly']]
    if length == 2:
        return _parends[f'{side}_curly_2']
    idxs = [0] + [1 for i in range((length - 3)//2)] + [slice(2,-2)]\
            + [-2 for i in range((length - 3)//2)] + [-1]
    return sum([ls(_parends[f'{side}_curly_{length%2}'][x]) for x in idxs],[])

def _bracket(side, mode, length):
    '''
    Generate parend of given side (L/R), mode (round,square,curly), and length
    '''
    if mode == 'curly':
        return _gen_curly_brace(side, length)
    return _gen_parend(side, mode, length)

def unicode_str(m, mode = 'square'):
    '''
    Unicode pretty-print
    Round, square, or curly braces
    Justify columns to 1 more than max length of elements
    '''
    l, r = [_bracket(side, mode, len(m._arr)) for side in 'lr']
    col_size = [max(len(str(m._arr[i][j])) for i in range(len(m._arr)))\
            for j in range(len(m._arr[0]))]
    rows = [l[i]+' '.join([str(n).ljust(col_size[j])\
            for j,n in enumerate(m._arr[i])])+r[i]\
            for i in range(len(m._arr))]
    return '\n'.join(rows)

def ascii_str(m):
    '''
    Ascii pretty-print
    Brace each side with [ or ], begin and end with extra []
    Justify columns to 1 more than max length of elements
    '''
    col_size = [max(len(str(m._arr[i][j])) for i in range(len(m._arr)))\
            for j in range(len(m._arr[0]))]
    rows = ['['+' '.join([str(n).ljust(col_size[j])\
            for j,n in enumerate(m._arr[i])])+']'\
            for i in range(len(m._arr))]
    return '[' + '\n '.join(rows) + ']'

def tex_str(m, mode='square'):
    '''
    LaTeX matrix string for matrix in given mode
    '''
    lfrac = lambda s: '\\frac{{{0}}}{{{1}}}'.format(*s.split('/')) if '/' in s else s
    rows = [' & '.join([lfrac(str(c)) for c in row]) for row in m._arr]
    return f'\\{_tex_parends[mode]}mat{{\n' + '\\\\\n'.join(rows) + '\n}'

def tex_str_min(m, mode='square'):
    '''
    LaTeX string with no whitespace
    '''
    return ''.join(tex_str(m, mode).split())
