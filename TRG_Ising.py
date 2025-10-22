"""TRG

Reference:

temp: temperature
T: UniTensor T

Ref:
* https://tensornetwork.org/trg/
"""
import cytnx
import numpy as np
import itertools 

def matrix_view(T, rowrank=-1):
    """View a UniTensor as a matrix (currently only non-symmetric UniTensor)
    Args:
        T: UniTensor
        rowrank: int
    """
    print("(Matrix View)")
    if rowrank == -1:
        rowrank = T.rowrank()
    if T.Nblocks() == 1:
        if rowrank == 0:
            print(T.get_block().reshape(1,np.prod(T.shape()[rowrank:])))
        elif rowrank == T.rank():
            print(T.get_block().reshape(np.prod(T.shape()[:rowrank]),1))
        else:
            print(T.get_block().reshape(np.prod(T.shape()[:rowrank]),np.prod(T.shape()[rowrank:])))
    else:
        print("Nblocks !=1")

# zero external field
print("#"*80)
print("(TRG) 2D Ising model, zero external field")
print("#"*80)
temp = 1.
print("(TRG) Constructing W")
W = np.array([[+np.exp(+1/temp), +np.exp(-1/temp)],
              [+np.exp(-1/temp), +np.exp(+1/temp)]])

print("(TRG) Constructing M")
M = np.array([[+np.sqrt(np.cosh(+1/temp)), +np.sqrt(np.sinh(+1/temp))],
              [+np.sqrt(np.cosh(+1/temp)), -np.sqrt(np.sinh(+1/temp))]])

print("(TRG) Checking if W-M@Md=0")
print("(TRG)", np.linalg.norm(W-M@M.transpose()))

# T xi, yi, xo, yo
# use UniTensor.at() with labels to set values
print("(TRG) Constructing T")
bd = cytnx.Bond(2)
T = cytnx.UniTensor([bd,bd,bd,bd], rowrank=4).set_name("T").relabel(["x_i","y_i","x_o","y_o"])
T.print_diagram()

for x_i, y_i, x_o, y_o in itertools.product([0, 1], repeat=4):
    # print(x_i, y_i, x_o, y_o)
    T.at(["x_i","y_i","x_o","y_o"], [x_i,y_i,x_o,y_o]).value = M[0,y_i]*M[0,x_i]*M[0,x_o]*M[0,y_o] + M[1,y_i]*M[1,x_i]*M[1,x_o]*M[1,y_o]

matrix_view(T, 2)
# [[4.76220e+00 0.00000e+00 0.00000e+00 3.62686e+00 ]
#  [0.00000e+00 3.62686e+00 3.62686e+00 0.00000e+00 ]
#  [0.00000e+00 3.62686e+00 3.62686e+00 0.00000e+00 ]
#  [3.62686e+00 0.00000e+00 0.00000e+00 2.76220e+00 ]]

matrix_view(T, T.rowrank())
# Total elem: 16
# type  : Double (Float64)
# cytnx device: CPU
# Shape : (16,1)
# [[4.76220e+00 ]
#  [0.00000e+00 ]
#  [0.00000e+00 ]
#  [3.62686e+00 ]
#  [0.00000e+00 ]
#  [3.62686e+00 ]
#  [3.62686e+00 ]
#  [0.00000e+00 ]
#  [0.00000e+00 ]
#  [3.62686e+00 ]
#  [3.62686e+00 ]
#  [0.00000e+00 ]
#  [3.62686e+00 ]
#  [0.00000e+00 ]
#  [0.00000e+00 ]
#  [2.76220e+00 ]]

# two different trace have the same result due to symmetry
trT = T.clone().Trace_("x_i","x_o").Trace_("y_i","y_o").set_name("trT")
trT.print_diagram()
trT = trT[0].item()
print("(TRG) trT={}".format(trT))

trT = T.clone().Trace_("x_i","y_i").Trace_("x_o","y_o").set_name("trT")
trT.print_diagram()
trT = trT[0].item()
print("(TRG) trT={}".format(trT))

# T_arrow is a rank-4 UniTensor with labels x_i, y_i, x_o, y_o
print("(TRG) Constructing T_arrow")
bd_i = cytnx.Bond(2, cytnx.BD_IN)
bd_o = cytnx.Bond(2, cytnx.BD_OUT)
T_arrow = cytnx.UniTensor([bd_i,bd_i,bd_o,bd_o], rowrank=4).set_name("T_arrow").relabel(["x_i","y_i","x_o","y_o"])
T_arrow.print_diagram()

for x_i, y_i, x_o, y_o in itertools.product([0, 1], repeat=4):
    T_arrow.at(["x_i","y_i","x_o","y_o"], [x_i,y_i,x_o,y_o]).value = M[0,y_i]*M[0,x_i]*M[0,x_o]*M[0,y_o] + M[1,y_i]*M[1,x_i]*M[1,x_o]*M[1,y_o]

matrix_view(T_arrow)

# this one works
trT = T_arrow.clone().Trace_("x_i","x_o").Trace_("y_i","y_o").set_name("trT")
trT.print_diagram()
trT = trT[0].item()
print("(TRG) trT={}".format(trT))
# this one should fail du to the direction of the bond
try:
    trT = T_arrow.clone().Trace_("x_i","y_i").Trace_("x_o","y_o").set_name("trT")
    trT.print_diagram()
    trT = trT[0].item()
    print("(TRG) trT={}".format(trT))
except:
    print("(TRG) Fail!")


matrix_view(T_arrow, 2)
# exit()

# Contraction of a 2x2 block
#     A3_y_o-A1_y_1                      A4_y_o-A2_y_i
#     ┌─────┘                    ┌─────┘
#     │ ┏━━━╳━━┓                 │ ┏━━━╳━━┓
#     └─┨d     ┃                 └─┨d     ┃
#  2────┨d A1 d┠───A1_x_o-A2_x_i───┨d A2 d┠───A2_x_o-A1_x_i
#       ┃     d┠───┐               ┃     d┠──┐
#       ┗━━━━━━┛   │               ┗━━━━━━┛  │
#  ┌─A1_y_o-A3_y_i─┘           ┌A2_y_o-A4_y_i┘
#  │    ┏━━━╳━━┓               │   ┏━━━╳━━┓
#  └────┨d     ┃               └───┨d     ┃
#  3────┨d A3 d┠───A3_x_o-A4_x_i───┨d A4 d┠───A4_x_o-A3_x_i
#       ┃     d┠─┐                 ┃     d┠─┐
#       ┗━━━━━━┛ │                 ┗━━━━━━┛ │
#          ┌─────┘                    ┌─────┘
#    A3_y_o-A1_y_1                  A4_y_o-A2_y_i

print("(TRG) Contraction of a 2x2 block")

net = cytnx.Network()
net.FromString(["A1: A2_x_o-A1_x_i, A3_y_o-A1_y_i, A1_x_o-A2_x_i, A1_y_o-A3_y_i", \
                "A2: A1_x_o-A2_x_i, A4_y_o-A2_y_i, A2_x_o-A1_x_i, A2_y_o-A4_y_i", \
                "A3: A4_x_o-A3_x_i, A1_y_o-A3_y_i, A3_x_o-A4_x_i, A3_y_o-A1_y_i", \
                "A4: A3_x_o-A4_x_i, A2_y_o-A4_y_i, A4_x_o-A3_x_i, A4_y_o-A2_y_i", \
                "TOUT: "])
print(net)
net.PutUniTensor("A1", T, ["x_i","y_i","x_o","y_o"])
net.PutUniTensor("A2", T, ["x_i","y_i","x_o","y_o"])
net.PutUniTensor("A3", T, ["x_i","y_i","x_o","y_o"])
net.PutUniTensor("A4", T, ["x_i","y_i","x_o","y_o"])
print(net)
Tout1 = net.Launch()
Tout1.print_diagram()
print(Tout1)

net = cytnx.Network()
net.FromString(["A1: A2_x_o-A1_x_i, A3_y_o-A1_y_i, A1_x_o-A2_x_i, A1_y_o-A3_y_i", \
                "A2: A1_x_o-A2_x_i, A4_y_o-A2_y_i, A2_x_o-A1_x_i, A2_y_o-A4_y_i", \
                "A3: A4_x_o-A3_x_i, A1_y_o-A3_y_i, A3_x_o-A4_x_i, A3_y_o-A1_y_i", \
                "A4: A3_x_o-A4_x_i, A2_y_o-A4_y_i, A4_x_o-A3_x_i, A4_y_o-A2_y_i", \
                "TOUT: "])
print(net)
net.PutUniTensor("A1", T_arrow, ["x_i","y_i","x_o","y_o"])
net.PutUniTensor("A2", T_arrow, ["x_i","y_i","x_o","y_o"])
net.PutUniTensor("A3", T_arrow, ["x_i","y_i","x_o","y_o"])
net.PutUniTensor("A4", T_arrow, ["x_i","y_i","x_o","y_o"])
print(net)
Tout1 = net.Launch()
Tout1.print_diagram()
print(Tout1)

# net = cytnx.Network()
# net.FromString(["A1: A3_y_o-A1_y_i, A2_x_o-A1_x_i, A1_x_o-A2_x_i, A1_y_o-A3_y_i", \
#                 "A2: A4_y_o-A2_y_i, A1_x_o-A2_x_i, A2_x_o-A1_x_i, A2_y_o-A4_y_i", \
#                 "A3: A1_y_o-A3_y_i, A4_x_o-A3_x_i, A3_x_o-A4_x_i, A3_y_o-A1_y_i", \
#                 "A4: A2_y_o-A4_y_i, A3_x_o-A4_x_i, A4_x_o-A3_x_i, A4_y_o-A2_y_i", \
#                 "TOUT: "])
# print(net)
# net.PutUniTensor("A1", T, ["y_i","x_i","x_o","y_o"])
# net.PutUniTensor("A2", T, ["y_i","x_i","x_o","y_o"])
# net.PutUniTensor("A3", T, ["y_i","x_i","x_o","y_o"])
# net.PutUniTensor("A4", T, ["y_i","x_i","x_o","y_o"])
# print(net)
# Tout1 = net.Launch()
# Tout1.print_diagram()
# print(Tout1)
# # for temp=1, Z(2,2)=5.97392e+03


# net = cytnx.Network()
# net.FromString(["A1: A3_y_o-A1_y_i, A2_x_o-A1_x_i, A1_x_o-A2_x_i, A1_y_o-A3_y_i", \
#                 "A2: A4_y_o-A2_y_i, A1_x_o-A2_x_i, A2_x_o-A1_x_i, A2_y_o-A4_y_i", \
#                 "A3: A1_y_o-A3_y_i, A4_x_o-A3_x_i, A3_x_o-A4_x_i, A3_y_o-A1_y_i", \
#                 "A4: A2_y_o-A4_y_i, A3_x_o-A4_x_i, A4_x_o-A3_x_i, A4_y_o-A2_y_i", \
#                 "TOUT: "])
# print(net)
# net.PutUniTensor("A1", T_arrow, ["y_i","x_i","x_o","y_o"])
# net.PutUniTensor("A2", T_arrow, ["y_i","x_i","x_o","y_o"])
# net.PutUniTensor("A3", T_arrow, ["y_i","x_i","x_o","y_o"])
# net.PutUniTensor("A4", T_arrow, ["y_i","x_i","x_o","y_o"])
# print(net)
# Tout1 = net.Launch()
# Tout1.print_diagram()
# print(Tout1)
# for temp=1, Z(2,2)=5.97392e+03
# exit()


def net_contract():
    # contract "A1", "y_i" with "A3", "y_o"
    pass

exit()

# For a block
# AB
# CD 
# init: 
# ··········🄲·🄳··············🄲·🄳·············🄲·🄳··············🄲·🄳············
# ··········🄰·🄱··············🄰·🄱·············🄰·🄱··············🄰·🄱············
# ·········╱·····╲···········╱·····╲·········╱·····╲············╱·····╲···········
# ·······🄰·······🄱········🄰·······🄱·······🄰·······🄱·······🄰·······🄱··········
# ╲·····╱···········╲·····╱··········╲·····╱···········╲·····╱···········╲········
# ·🄱·🄰··············🄱·🄰··············🄱···············🄱·🄰·············🄱·🄰····
# ·🄳·🄲··············🄳·🄲··············🄳···············🄳·🄲·············🄳·🄲····
# ╱·····╲···········╱·····╲···········╱·····╲··········╱·····╲···········╱·····╲··
# ·······🄲·······🄳········🄲·······🄳·······🄲·······🄳·······🄲·······🄳··········
# ·········╲·····╱···········╲·····╱···········╲·····╱··········╲·····╱···········
# ··········🄲·🄳··············🄲·🄳·············🄲·🄳·············🄲·🄳·············
# ··········🄰·🄱··············🄰·🄱·············🄰·🄱·············🄰·🄱·············

# SVD
print("(TRG) SVD")
# ·······y_i·················y_i·································y_i··············
# ········│···················│···································│···············
# ········▼···················▼···································▼···············
# ······┌─┴─┐···············┌─┴─┐·······························┌─┴─┐·············
# x_i·─▶┤ T ├▶─·x_o·=·x_i·─▶┤UL │·······························│UR ├▶─·x_o·······
# ······└─┬─┘···············└───◢·······························◥───┘·············
# ········▼······················╲·····························╱··················
# ········│·····················UL_o·························UR_i··················
# ·······y_o····················DR_i·························DL_o··················
# ·································╲·························╱····················
# ··································◢───┐···············┌───◥·····················
# ··································│DR ├─▶─·x_o·=·x_i─▶┤DL·│·····················
# ··································└─┬─┘···············└─┬─┘·····················
# ····································▼···················▼·······················
# ····································│···················│·······················
# ···································y_o·················y_o······················

# T-->UL @ DR
# print("(TRG) T-->UL @ DR")

def TRG_split(Tin, Tout1_label, Tout2_label, aux_label):
    """Decompose a rank-4 tensor into two rank-3 tensors
    """
    # print("Tin.labels=", Tin.labels())
    # print(Tout1_label, Tout2_label, Tout1_label+Tout2_label)    
    T = Tin.permute(Tout1_label+Tout2_label)
    S , U , Vdag = cytnx.linalg.Svd(T)
    S_sqrt = cytnx.linalg.Pow(S,0.5).set_name("S_sqrt")
    Tout1 = cytnx.Contract(U, S_sqrt).relabel(Tout1_label+aux_label)
    Tout2 = cytnx.Contract(S_sqrt, Vdag).relabel(aux_label+Tout2_label)

    return Tout1, Tout2


UL, DR =TRG_split(T, ['y_i', 'x_i'], ['x_o', 'y_o'], ["aux"])
# UL.print_diagram()
# DR.print_diagram()
# print(UL)
# print(DR)

T_DL, T_UR = TRG_split(T, ['x_i', 'y_o'], ['y_i', 'x_o'], ["aux"])
T_DL.print_diagram()
T_UR.print_diagram()
print(T_DL)
print(T_UR)

# T.print_diagram()
# S , U , Vdag = cytnx.linalg.Svd(T)
# S_sqrt = cytnx.linalg.Pow(S,0.5).set_name("S_sqrt")
# UL = cytnx.Contract(U, S_sqrt).set_name("UL").relabel(["y_i","x_i","UL_o"])
# DR = cytnx.Contract(S_sqrt, Vdag).set_name("DR").relabel(["DR_i","x_o","y_o"])
# UL.print_diagram()
# DR.print_diagram()
# print(UL)
# print(DR)

# def T_to_UL_DR(T):
#     S , U , Vdag = cytnx.linalg.Svd(T)
#     S_sqrt = cytnx.linalg.Pow(S,0.5).set_name("S_sqrt")
#     S_sqrt.print_diagram()
#     UL = cytnx.Contract(U, S_sqrt).set_name("UL").relabel(["y_i","x_i","UL_o"])
#     DR = cytnx.Contract(S_sqrt, Vdag).set_name("DR").relabel(["DR_i","x_o","y_o"])
#     return UL, DR

# T_UL, T_DR = T_to_UL_DR(T)
# print((T_UL-UL).Norm())
# print((T_DR-DR).Norm())

# T-->DL @ UR
print("(TRG) T-->DL @ UR")
# T.permute(["x_i","y_o","y_i","x_o"]).print_diagram()
S , U , Vdag = cytnx.linalg.Svd(T.permute(["x_i","y_o","y_i","x_o"]))
S_sqrt = cytnx.linalg.Pow(S,0.5).set_name("S_sqrt")
DL = cytnx.Contract(U, S_sqrt).set_name("DL").relabel(["x_i","y_o","DL_o"])
UR = cytnx.Contract(S_sqrt, Vdag).set_name("UR").relabel(["UR_i","y_i","x_o"])

def T_to_DL_UR(T):
    # T.permute(["x_i","y_o","y_i","x_o"]).print_diagram()
    S , U , Vdag = cytnx.linalg.Svd(T.permute(["x_i","y_o","y_i","x_o"]))
    S_sqrt = cytnx.linalg.Pow(S,0.5).set_name("S_sqrt")
    DL = cytnx.Contract(U, S_sqrt).set_name("DL").relabel(["x_i","y_o","DL_o"])
    UR = cytnx.Contract(S_sqrt, Vdag).set_name("UR").relabel(["UR_i","y_i","x_o"])
    return DL, UR
    

T_DL, T_UR = T_to_DL_UR(T)
T_DL.print_diagram()
T_UR.print_diagram()
print(T_DL)
print(T_UR)
# print((T_DL-DL).Norm())
# print((T_UR-UR).Norm())
T.print_diagram()


# ·······y_i···············y_i····················································
# ········│·················│·····················································
# ········▼·················▼·····················································
# ······┌─┴─┐·············┌─┴─┐···················································
# x_i·─▶┤ T ├▶─·x_o─x_i·─▶┤ T ├▶─·x_o·············································
# ······└─┬─┘·············└─┬─┘···················································
# ········▼·················▼·····················································
# ········│·················│·····················································
# ·······y_o···············y_o····················································
# ········│·················│·····················································
# ·······y_i···············y_i····················································
# ········│·················│·····················································
# ········▼·················▼·····················································
# ······┌─┴─┐·············┌─┴─┐···················································
# x_i·─▶┤ T ├▶─·x_o─x_i·─▶┤ T ├▶─·x_o·············································
# ······└─┬─┘·············└─┬─┘···················································
# ········▼·················▼·····················································
# ········│·················│·····················································
# ·······y_o···············y_o····················································

# ·······y_i································y_i···································
# ········│··································│····································
# ········▼··································▼····································
# ······┌─┴─┐······························┌─┴─┐··································
# x_i·─▶┤UL │······························│UR ├▶·─x_o····························
# ······└───◢······························◥───┘··································
# ···········╲····························╱·······································
# ··········UL_o························UR_i······································
# ··········DR_i························DL_o······································
# ·············╲························╱·········································
# ··············◢───┐··············┌───◥··········································
# ··············│DR ├─▶─·x_o─x_i·─▶┤DL │··········································
# ··············└─┬─┘··············└─┬─┘··········································
# ················▼··················▼············································
# ···············y_o················y_o···········································
# ···············y_i················y_i···········································
# ················▼··················▼············································
# ··············┌─┴─┐··············┌─┴─┐··········································
# ··············│UR ├─▶─·x_o─x_i·─▶┤UL │··········································
# ··············◥───┘········ ·····└───◢··········································
# ·············╱························╲·········································
# ···········UR_i······················UL_o·······································
# ···········DL_o······················DR_i·······································
# ···········╱···························╲········································
# ······┌───◥······························◢───┐··································
# x_i·─▶┤DL │······························│DR ├▶─·x_o····························
# ······└─┬─┘······························└─┬─┘··································
# ········▼··································▼····································
# ········│··································│····································
# ·······y_o································y_o···································


# net = cytnx.Network()
# net.FromString(["DR: DR_i, xoxi_u, yoyi_l", \
#                 "DL: DL_o, xoxi_u, yoyi_r", \
#                 "UR: UR_i, xoxi_d, yoyi_l", \
#                 "UL: UL_o, xoxi_d, yoyi_r", \
#                 "TOUT: DR_i, UR_i, DL_o, UL_o"])
# print(net)
# # net.PutUniTensor("DR", DR)
# net.PutUniTensor("DR", DR, ["DR_i","x_o","y_o"])
# net.PutUniTensor("DL", DL, ["DL_o","x_i","y_o"])
# net.PutUniTensor("UR", UR, ["UR_i","x_o","y_i"])
# net.PutUniTensor("UL", UL, ["UL_o","x_i","y_i"])
# print(net)
# Tout1 = net.Launch()
# Tout1.print_diagram()

# net = cytnx.Network()
# net.FromString(["DR: DR_i, DR_xo-DL_xi, DR_yo-UR_yi", \
#                 "DL: DL_o, DR_xo-DL_xi, DL_yo-UL_yi", \
#                 "UR: UR_i, UR_xo-UL_xi, DR_yo-UR_yi", \
#                 "UL: UL_o, UR_xo-UL_xi, DL_yo-UL_yi", \
#                 "TOUT: DR_i, UR_i, DL_o, UL_o"])
# print(net)
# # net.PutUniTensor("DR", DR)
# net.PutUniTensor("DR", DR, ["DR_i","x_o","y_o"])
# net.PutUniTensor("DL", DL, ["DL_o","x_i","y_o"])
# net.PutUniTensor("UR", UR, ["UR_i","x_o","y_i"])
# net.PutUniTensor("UL", UL, ["UL_o","x_i","y_i"])
# print(net)
# Tout2 = net.Launch()
# Tout2.print_diagram()

# diff = Tout1-Tout2
# print(diff.Norm())
# print(type(diff))

# Main
# if __name__ == "__main__":
#     print("(TRG) TRG, 2D Ising model")
