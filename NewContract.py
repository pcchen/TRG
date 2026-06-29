from extension import *
# import cytnx

# def conc(tensors, contractions):
#     """
#     tensors: list of tuples, each tuple contains the name and labels of a tensor
#     contractions: list of tuples, each tuple contains the names of the tensors to be contracted
#     """
#     print(">"*80)
#     print(">>>>>Contraction of tensors")
#     print(">"*80)

#     # add tensor name to to labels to make them unique
#     T_labels = {}
#     T_labels_remaining = {}
#     # print(">>>>>Input tensors:")
#     # print(">>>>>id, name, oritinal labels")
#     # print(">>>>>id, name, new labels")
#     for i, (name, labels) in enumerate(tensors):
#         # print(i, name, labels)
#         # print([name+'_'+label for label in labels])
#         T_labels[name] = [name+'_'+label for label in labels]
#         T_labels_remaining[name] = labels.copy()
#         # print(i, name, T_labels[name])

#     print(">>>>>New Labels") 
#     for name in T_labels:
#         print(name, T_labels[name])
#     print(">>>>>Remaining Labels") 
#     for name in T_labels_remaining:
#         print(name, T_labels_remaining[name])

#     # re-name labels in contraction pair
#     # find remaining labels in T_labels_remaining
#     print(">>>>>Contractions:")
#     for i, (name_1, label_1, name_2, label_2) in enumerate(contractions):
#         print(">>>>>Pair-{}".format(i))

#         print(name_1, label_1, name_2, label_2)
#         label_pair = name_1+'_'+label_1+'-'+name_2+'_'+label_2
#         print(label_pair)

#         label_index_1 = T_labels[name_1].index(name_1+'_'+label_1)
#         print(T_labels[name_1][label_index_1])
#         T_labels[name_1][label_index_1] = label_pair

#         label_index_1 = T_labels_remaining[name_1].index(label_1)
#         print(label_index_1, T_labels_remaining[name_1][label_index_1])
#         T_labels_remaining[name_1].pop(label_index_1)

#         label_index_2 = T_labels[name_2].index(name_2+'_'+label_2)
#         print(T_labels[name_1][label_index_1])
#         T_labels[name_2][label_index_2] = label_pair

#         label_index_2 = T_labels_remaining[name_2].index(label_2)
#         print(label_index_2, T_labels_remaining[name_2][label_index_2])
#         T_labels_remaining[name_2].pop(label_index_2)


#         print(">>>>>New Labels") 
#         for name in T_labels:
#             print(name, T_labels[name])
#         print(">>>>>Remaining Labels") 
#         for name in T_labels_remaining:
#             print(name, T_labels_remaining[name])

#     print(">>>>>Constructing network string")
#     # print(T_labels)
#     # print(T_labels_remaining)
#     # print(T_labels_remaining.keys())

    
#     print(">>>>>net_string:")
#     net_string = []
#     for name, _ in tensors:
#         print(name+': '+', '.join(T_labels[name]))
#         net_string.append(name+': '+', '.join(T_labels[name]))

#     print(">>>>>Tout_labels:")
#     Tout_labels = []
#     for name in T_labels_remaining.keys():
#         # print([name+'_'+label for label in T_labels_remaining[name]])
#         Tout_labels += [name+'_'+label for label in T_labels_remaining[name]]
#     # print(Tout_labels)
#     print('TOUT: '+', '.join(Tout_labels))
#     net_string.append('TOUT: '+', '.join(Tout_labels))
#     # for line in net_string:
#     #     print(line)

#     net=cytnx.Network()
#     net.FromString(net_string)
#     # print(net)
#     return net

net = conc(tensors=[("Ta",["i","j"]),("Tb",["j","k"]),("Tc",["k","i"])],
     contractions=[("Ta","j","Tb","k")])

print(net)


A = cytnx.UniTensor.ones([2,3]).relabel(['i','j']).set_name('A')
B = cytnx.UniTensor.ones([4,3]).relabel(['j','k']).set_name('B')
C = cytnx.UniTensor.ones([3,2]).relabel(['k','j']).set_name('C')

A.print_diagram()
B.print_diagram()
C.print_diagram()

net.PutUniTensor("Ta", A, ["i","j"])
net.PutUniTensor("Tb", B, ["j","k"])
net.PutUniTensor("Tc", C, ["k","j"])
print(net)
Tout = net.Launch().set_name("Tout")
Tout.print_diagram()
print(Tout)
# exit()

tensors=[
    ("A11",["x_i","y_i","x_o","y_o"]),
    ("A12",["x_i","y_i","x_o","y_o"]),
    ("A21",["x_i","y_i","x_o","y_o"]),
    ("A22",["x_i","y_i","x_o","y_o"]),
]

contractions=[
    ("A11","x_o","A12","x_i"),("A12","x_o","A11","x_i"),
    ("A21","x_o","A22","x_i"),("A22","x_o","A21","x_i"),
    ("A11","y_o","A21","y_i"),("A12","y_o","A22","y_i"),
    ("A21","y_o","A11","y_i"),("A22","y_o","A12","y_i"),
]
# 
net = conc(tensors=[("A11",["x_i","y_i","x_o","y_o"]),\
                    ("A12",["x_i","y_i","x_o","y_o"]),\
                    ("A21",["x_i","y_i","x_o","y_o"]),\
                    ("A22",["x_i","y_i","x_o","y_o"])],\
            contractions=[("A11","x_o","A12","x_i"),("A12","x_o","A11","x_i"),\
                          ("A21","x_o","A22","x_i"),("A22","x_o","A21","x_i"),\
                          ("A11","y_o","A21","y_i"),("A12","y_o","A22","y_i"),\
                          ("A21","y_o","A11","y_i"),("A22","y_o","A12","y_i")]) # ,


print(">"*80)
print(tensors)
print(net)

