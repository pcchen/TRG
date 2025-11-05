import cytnx
import numpy as np
import itertools 

bd = cytnx.Bond(2)

T = cytnx.UniTensor([bd,bd,bd,bd]).set_rowrank(2).set_name('T').relabel(['a','b','c','d'])
T.print_diagram()
print(T[0,0,0,0].item())

for i in range(2):
    for j in range(2):
        T[i,j,i,j] = 1

print(T)
# T.bonds()[2].combine
T.combineBonds(['c','d'])
T.print_diagram()
# print(T)
for i in range(2):
    for j in range(2):
        for k in range(4):
            print(i,j,k,T[i,j,k].item())
exit()