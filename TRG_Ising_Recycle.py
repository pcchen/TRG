# def TTTT_2_Z(T):
#     # print("(TRG) Contraction of a 2x2 block")
#     # print("(TRG) Puting T")
#     net = cytnx.Network()
#     net.FromString(["A1: A2_x_o-A1_x_i, A3_y_o-A1_y_i, A1_x_o-A2_x_i, A1_y_o-A3_y_i", \
#                     "A2: A1_x_o-A2_x_i, A4_y_o-A2_y_i, A2_x_o-A1_x_i, A2_y_o-A4_y_i", \
#                     "A3: A4_x_o-A3_x_i, A1_y_o-A3_y_i, A3_x_o-A4_x_i, A3_y_o-A1_y_i", \
#                     "A4: A3_x_o-A4_x_i, A2_y_o-A4_y_i, A4_x_o-A3_x_i, A4_y_o-A2_y_i", \
#                     "TOUT: "])
#     # print(net)
#     net.PutUniTensor("A1", T, ["x_i","y_i","x_o","y_o"])
#     net.PutUniTensor("A2", T, ["x_i","y_i","x_o","y_o"])
#     net.PutUniTensor("A3", T, ["x_i","y_i","x_o","y_o"])
#     net.PutUniTensor("A4", T, ["x_i","y_i","x_o","y_o"])
#     # print(net)
#     Tout = net.Launch()
#     # Tout1.print_diagram()
#     # print(Tout1)
#     return Tout

# Tout = TTTT_2_Z(T)
# Tout.print_diagram()
# print(Tout)
# exit()
# print("(TRG) Contraction of a 2x2 block")
# print("(TRG) Puting T")
# net = cytnx.Network()
# net.FromString(["A1: A2_x_o-A1_x_i, A3_y_o-A1_y_i, A1_x_o-A2_x_i, A1_y_o-A3_y_i", \
#                 "A2: A1_x_o-A2_x_i, A4_y_o-A2_y_i, A2_x_o-A1_x_i, A2_y_o-A4_y_i", \
#                 "A3: A4_x_o-A3_x_i, A1_y_o-A3_y_i, A3_x_o-A4_x_i, A3_y_o-A1_y_i", \
#                 "A4: A3_x_o-A4_x_i, A2_y_o-A4_y_i, A4_x_o-A3_x_i, A4_y_o-A2_y_i", \
#                 "TOUT: "])
# # print(net)
# net.PutUniTensor("A1", T, ["x_i","y_i","x_o","y_o"])
# net.PutUniTensor("A2", T, ["x_i","y_i","x_o","y_o"])
# net.PutUniTensor("A3", T, ["x_i","y_i","x_o","y_o"])
# net.PutUniTensor("A4", T, ["x_i","y_i","x_o","y_o"])
# # print(net)
# Tout1 = net.Launch()
# # Tout1.print_diagram()
# print(Tout1)
# # exit()


# net = cytnx.Network()
# net.FromString(["A1: A2_x_o-A1_x_i, A3_y_o-A1_y_i, A1_x_o-A2_x_i, A1_y_o-A3_y_i", \
#                 "A2: A1_x_o-A2_x_i, A4_y_o-A2_y_i, A2_x_o-A1_x_i, A2_y_o-A4_y_i", \
#                 "A3: A4_x_o-A3_x_i, A1_y_o-A3_y_i, A3_x_o-A4_x_i, A3_y_o-A1_y_i", \
#                 "A4: A3_x_o-A4_x_i, A2_y_o-A4_y_i, A4_x_o-A3_x_i, A4_y_o-A2_y_i", \
#                 "TOUT: "])
# # print(net)
# net.PutUniTensor("A1", T_arrow, ["x_i","y_i","x_o","y_o"])
# net.PutUniTensor("A2", T_arrow, ["x_i","y_i","x_o","y_o"])
# net.PutUniTensor("A3", T_arrow, ["x_i","y_i","x_o","y_o"])
# net.PutUniTensor("A4", T_arrow, ["x_i","y_i","x_o","y_o"])
# # print(net)
# Tout1 = net.Launch()
# # Tout1.print_diagram()
# print(Tout1)


def TTTT_2_T(T):
    net = cytnx.Network()
    net.FromString(["A1: A1_x_i, A1_y_i, A1_x_o-A2_x_i, A1_y_o-A3_y_i", \
                    "A2: A1_x_o-A2_x_i, A2_y_i, A2_x_o, A2_y_o-A4_y_i", \
                    "A3: A3_x_i, A1_y_o-A3_y_i, A3_x_o-A4_x_i, A3_y_o", \
                    "A4: A3_x_o-A4_x_i, A2_y_o-A4_y_i, A4_x_o, A4_y_o", \
                    "TOUT: A1_x_i, A1_y_i, A2_y_i, A3_x_i, A2_x_o, A3_y_o, A4_x_o, A4_y_o"])
    print(net)
    net.PutUniTensor("A1", T, ["x_i","y_i","x_o","y_o"])
    net.PutUniTensor("A1", T, ["x_i","y_i","x_o","y_o"])
    net.PutUniTensor("A2", T, ["x_i","y_i","x_o","y_o"])
    net.PutUniTensor("A3", T, ["x_i","y_i","x_o","y_o"])
    net.PutUniTensor("A4", T, ["x_i","y_i","x_o","y_o"])
    print(net)
    Tout = net.Launch().set_name("TTTT")
    # Tout.print_diagram()
    return Tout


Tout = TTTT_2_T(T)
Tout.print_diagram()
# print(Tout)
# exit()
print("(TRG) Contraction of a 2x2 block")
print("(TRG) Puting T")
net = cytnx.Network()
net.FromString(["A1: A1_x_i, A1_y_i, A1_x_o-A2_x_i, A1_y_o-A3_y_i", \
                "A2: A1_x_o-A2_x_i, A2_y_i, A2_x_o, A2_y_o-A4_y_i", \
                "A3: A3_x_i, A1_y_o-A3_y_i, A3_x_o-A4_x_i, A3_y_o", \
                "A4: A3_x_o-A4_x_i, A2_y_o-A4_y_i, A4_x_o, A4_y_o", \
                "TOUT: A1_x_i, A1_y_i, A2_y_i, A3_x_i, A2_x_o, A3_y_o, A4_x_o, A4_y_o"])
print(net)
net.PutUniTensor("A1", T, ["x_i","y_i","x_o","y_o"])
net.PutUniTensor("A1", T, ["x_i","y_i","x_o","y_o"])
net.PutUniTensor("A2", T, ["x_i","y_i","x_o","y_o"])
net.PutUniTensor("A3", T, ["x_i","y_i","x_o","y_o"])
net.PutUniTensor("A4", T, ["x_i","y_i","x_o","y_o"])
print(net)
Tout1 = net.Launch().set_name("TTTT")
Tout1.print_diagram()

exit()

TM = Tout1.Trace("A1_y_i","A3_y_o").Trace("A2_y_i","A4_y_o").set_name("TM")
TM.print_diagram()


matrix_view(TM, 2)
exp_Ei, eigenvectors = cytnx.linalg.Eigh(TM)
exp_Ei = exp_Ei.get_block().numpy()[::-1]
Ei = -np.log(exp_Ei)
# print(exp_Ei)
print(Ei/2)

# exit()

# exit()

net = cytnx.Network()
net.FromString(["DR: DR_i, DR_x_o-DL_x_i, DR_y_o-UR_y_i", \
                "DL: DL_o, DR_x_o-DL_x_i, DL_y_o-UL_y_i", \
                "UR: UR_i, UR_x_o-UL_x_i, DR_y_o-UR_y_i", \
                "UL: UL_o, UR_x_o-UL_x_i, DL_y_o-UL_y_i", \
                "TOUT: DR_i, UR_i, DL_o, UL_o"])
print(net)
net.PutUniTensor("DR", TC_DR, ["aux_C","x_o","y_o"])
net.PutUniTensor("DL", TD_DL, ["aux_D","x_i","y_o"])
net.PutUniTensor("UR", TA_UR, ["aux_A","x_o","y_i"])
net.PutUniTensor("UL", TB_UL, ["aux_B","x_i","y_i"])
print(net)
T_CDAB = net.Launch().set_name("T_CDAB")
T_CDAB.print_diagram()

exit()
net = cytnx.Network()
net.FromString(["DR: DR_i, DR_x_o-DL_x_i, DR_y_o-UR_y_i", \
                "DL: DL_o, DR_x_o-DL_x_i, DL_y_o-UL_y_i", \
                "UR: UR_i, UR_x_o-UL_x_i, DR_y_o-UR_y_i", \
                "UL: UL_o, UR_x_o-UL_x_i, DL_y_o-UL_y_i", \
                "TOUT: DR_i, UR_i, DL_o, UL_o"])
print(net)

print("(TRG) Constrauct T_BADC from TB_DR, TA_DL, TD_UR, TC_UL")
# Constrauct T_BADC from TB_DR, TA_DL, TD_UR, TC_UL
TB_DR.print_diagram()
TA_DL.print_diagram()
TD_UR.print_diagram()
TC_UL.print_diagram()
net.PutUniTensor("DR", TB_DR, ["aux_B","x_o","y_o"])
net.PutUniTensor("DL", TA_DL, ["aux_A","x_i","y_o"])
net.PutUniTensor("UR", TD_UR, ["aux_D","x_o","y_i"])
net.PutUniTensor("UL", TC_UL, ["aux_C","x_i","y_i"])
print(net)
T_BADC = net.Launch().set_name("T_BADC")
T_BADC.print_diagram()

matrix_view(T_CDAB, 2)

matrix_view(T_BADC, 2)

TT = T_CDAB.get_block().numpy()
print(TT.shape)
TT = TT.reshape(16,16)
print(TT.shape)
with np.printoptions(precision=2, linewidth=200):
    print(TT)


# TA, TD ==> DL, UR, [x_i,y_o], [x_o, y_i] 
TA_DL, TA_UR =TRG_split(TA, ['x_i', 'y_o'], ['x_o', 'y_i'], ["aux_A"])
TA_DL.set_name("TA_DL").print_diagram()
TA_UR.set_name("TA_UR").print_diagram()

TD_DL, TD_UR =TRG_split(TD, ['x_i', 'y_o'], ['x_o', 'y_i'], ["aux_D"])
TD_DL.set_name("TD_DL").print_diagram()
TD_UR.set_name("TD_UR").print_diagram()

# TB, TC ==> UL, DR, [x_i,y_i], [x_o,y_o] 
TB_UL, TB_DR =TRG_split(TB, ['x_i', 'y_i'], ['x_o', 'y_o'], ["aux_B"])
TB_UL.set_name("TB_UL").print_diagram()
TB_DR.set_name("TB_DR").print_diagram()

TC_UL, TC_DR =TRG_split(TC, ['x_i', 'y_i'], ['x_o', 'y_o'], ["aux_C"])
TC_UL.set_name("TC_UL").print_diagram()
TC_DR.set_name("TC_DR").print_diagram()


print(">>>>> (TRG) Constrauct T_CDAB from TC_DR, TD_DL, TA_UR, TB_UL")
# Constrauct T_CDAB from TC_DR, TD_DL, TA_UR, TB_UL
TC_DR.print_diagram()
TD_DL.print_diagram()
TA_UR.print_diagram()
TB_UL.print_diagram()
# exit()

net = cytnx.Network()
net.FromString(["DR: DR_i, DR_x_o-DL_x_i, DR_y_o-UR_y_i", \
                "DL: DL_o, DR_x_o-DL_x_i, DL_y_o-UL_y_i", \
                "UR: UR_i, UR_x_o-UL_x_i, DR_y_o-UR_y_i", \
                "UL: UL_o, UR_x_o-UL_x_i, DL_y_o-UL_y_i", \
                "TOUT: DR_i, UR_i, DL_o, UL_o"])
print(net)
net.PutUniTensor("DR", TC_DR, ["aux_C","x_o","y_o"])
net.PutUniTensor("DL", TD_DL, ["aux_D","x_i","y_o"])
net.PutUniTensor("UR", TA_UR, ["aux_A","x_o","y_i"])
net.PutUniTensor("UL", TB_UL, ["aux_B","x_i","y_i"])
print(net)
T_CDAB = net.Launch().set_name("T_CDAB")
T_CDAB.print_diagram()

exit()
net = cytnx.Network()
net.FromString(["DR: DR_i, DR_x_o-DL_x_i, DR_y_o-UR_y_i", \
                "DL: DL_o, DR_x_o-DL_x_i, DL_y_o-UL_y_i", \
                "UR: UR_i, UR_x_o-UL_x_i, DR_y_o-UR_y_i", \
                "UL: UL_o, UR_x_o-UL_x_i, DL_y_o-UL_y_i", \
                "TOUT: DR_i, UR_i, DL_o, UL_o"])
print(net)

print("(TRG) Constrauct T_BADC from TB_DR, TA_DL, TD_UR, TC_UL")
# Constrauct T_BADC from TB_DR, TA_DL, TD_UR, TC_UL
TB_DR.print_diagram()
TA_DL.print_diagram()
TD_UR.print_diagram()
TC_UL.print_diagram()
net.PutUniTensor("DR", TB_DR, ["aux_B","x_o","y_o"])
net.PutUniTensor("DL", TA_DL, ["aux_A","x_i","y_o"])
net.PutUniTensor("UR", TD_UR, ["aux_D","x_o","y_i"])
net.PutUniTensor("UL", TC_UL, ["aux_C","x_i","y_i"])
print(net)
T_BADC = net.Launch().set_name("T_BADC")
T_BADC.print_diagram()

matrix_view(T_CDAB, 2)

matrix_view(T_BADC, 2)

TT = T_CDAB.get_block().numpy()
print(TT.shape)
TT = TT.reshape(16,16)
print(TT.shape)
with np.printoptions(precision=2, linewidth=200):
    print(TT)
