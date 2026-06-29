import cytnx
import numpy as np
import itertools

from sympy import E 

temp = 1.
print("(TRG) Constructing W")
W = np.array([[+np.exp(+1/temp), +np.exp(-1/temp)],
              [+np.exp(-1/temp), +np.exp(+1/temp)]])
print(W)

def E(s1,s2):
    J = 1.0
    return -J*s1*s2

temp = 1.0
S = {}
S[+1] = 0
S[-1] = 1

W = cytnx.UniTensor.zeros([2,2]).set_name("W").relabel(["s","sp"])
for s, sp in itertools.product([+1, -1], repeat=2):
    # print(s,sp,S[s],S[sp],np.exp(-E(s,sp)/temp))
    W.at(["s","sp"], [S[s], S[sp]]).value = np.exp(-E(s,sp)/temp)
W.print_diagram()
print(W)

exit()