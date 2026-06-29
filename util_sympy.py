import numpy as np
import sympy

print(">"*80)
print(">>>>> Sympy Example")
print(">"*80)
sympy.init_printing()

𝛽 = sympy.symbols('𝛽', real=True) # inverse temperature
h = sympy.symbols('h', real=True) # external field

W = sympy.Matrix(
    [[sympy.exp(+𝛽), sympy.exp(-𝛽)], 
     [sympy.exp(-𝛽), sympy.exp(+𝛽)]])
    
print("W =")
sympy.pprint(W, use_unicode=True)

V = sympy.Matrix(
    [[+1, +1],
     [+1, -1]])/sympy.sqrt(2)

print("V =")
sympy.pprint(V, use_unicode=True)
print("V.T =")
sympy.pprint(V.T, use_unicode=True)

print("V*V.T =")
sympy.pprint(V*V.T, use_unicode=True)
print("V.T*V =")
sympy.pprint(V.T*V, use_unicode=True)

print("W_sym=V.T*W*V =")
W_sym = sympy.simplify(V.T*W*V)
sympy.pprint(W_sym, use_unicode=True)

print("W_sym_sqrt =")
W_sym_sqrt = sympy.simplify(sympy.sqrt(W_sym))
sympy.pprint(W_sym_sqrt, use_unicode=True)

sympy.pprint(V*sympy.sqrt(W_sym), use_unicode=True)
sympy.pprint(sympy.sqrt(W_sym)*V.T, use_unicode=True)

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                print(V[i,0]*V[j,0]*V[k,0]*V[l,0] + V[i,1]*V[j,1]*V[k,1]*V[l,1])


M = sympy.Matrix([[+sympy.sqrt(sympy.cosh(𝛽)), +sympy.sqrt(sympy.sinh(𝛽))], [+sympy.sqrt(sympy.cosh(𝛽)), -sympy.sqrt(sympy.sinh(𝛽))]])
print("M =")
sympy.pprint(M, use_unicode=True)

T = sympy.MutableDenseNDimArray(np.zeros((2,)*4))
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                T[i,j,k,l]=(M[0,i]*M[0,j]*M[0,k]*M[0,l]+M[1,i]*M[1,j]*M[1,k]*M[1,l])
print(T.shape)                
sympy.pprint(T, use_unicode=True)
