import cytnx
import numpy as np
import itertools

from math import sqrt, log, cos, cosh, sinh, tanh, isnan, isinf, pi
import scipy.integrate as integrate
import time


def exact_free_energy(temp: float) -> float:
    beta = 1.0 / temp
    cc, ss = cosh(2.0 * beta), sinh(2.0 * beta)
    k = 2.0 * ss / cc**2

    def integrant(x):
        return log(1.0 + sqrt(abs(1.0 - k * k * cos(x) ** 2)))

    integral, _ = integrate.quad(integrant, 0, 0.5 * pi, epsabs=1e-13, epsrel=1e-13)
    result = integral / pi + log(cc) + 0.5 * log(2.0)
    return -result / beta

def initial_TN(temp: float) -> tuple[np.ndarray, float, float]:
    # shape = (2, 2, 2, 2)
    # a = np.zeros(shape, dtype=float)  # [top, right, bottom, left]
    # c = np.cosh(1.0 / temp)
    # s = np.sinh(1.0 / temp)
    # for idx in np.ndindex(shape):
    #     if sum(idx) == 0:
    #         a[idx] = 2 * c * c
    #     elif sum(idx) == 2:
    #         a[idx] = 2 * c * s
    #     elif sum(idx) == 4:
    #         a[idx] = 2 * s * s
    # print(a)

    M = np.array([[+np.sqrt(np.cosh(+1/temp)), +np.sqrt(np.sinh(+1/temp))],
                [+np.sqrt(np.cosh(+1/temp)), -np.sqrt(np.sinh(+1/temp))]])
    bd = cytnx.Bond(2)
    T = cytnx.UniTensor([bd,bd,bd,bd], rowrank=4).set_name("T").relabel(["u","r","d","l"])
    for u,r,d,l in itertools.product([0, 1], repeat=4):
        T.at(["u","r","d","l"], [u,r,d,l]).value = M[0,u]*M[0,r]*M[0,d]*M[0,l] + M[1,u]*M[1,r]*M[1,d]*M[1,l]
    # T.print_diagram()
    # print("T-a=\n",T.get_block().numpy()-a)
    trT = T.Trace("u","d").Trace("r","l").item()
    T /= trT
    log_factor = np.log(trT)
    n_spin = 1

    # # normalize
    # val = np.einsum("ijij", a)
    # a /= val
    # log_factor = np.log(val)
    # n_spin = 1.0  # An initial tensor has one spin.

    # print("a=\n",a)
    # print("T=\n",T.get_block().numpy())
    # print("T-a=\n",T.get_block().numpy()-a)
    # print(trT, val, trT-val)

    return (T, log_factor, n_spin)


class TRG:
    def __init__(self, temp: float, chi: int) -> None:
        self.method = "TRG"
        self.temp = temp
        self.chi = chi
        # self.f_exact = ising.exact_free_energy(temp)
        self.f_exact = exact_free_energy(temp)

        # a, log_factor, n_spin = common.initial_TN(self.temp)
        a, log_factor, n_spin = initial_TN(self.temp)
        self.A = a
        self.log_factors = [log_factor]
        self.n_spins = [n_spin]
        self.step = 0

    def print_elapsed_time(self, elapsed_time: float) -> None:
        print(f"# Elapsed time: {elapsed_time:.6f} sec")

    def update(self) -> None:
        T = self.A.set_rowrank(2)
        T = T.permute(['u','r','d','l'])
        T.print_diagram()
        # print("T=\n",T.get_block().numpy())
        # print("T=\n",T.get_block().numpy().reshape(4,4))
        # u, s, vt = spl.svd(T.get_block().numpy().reshape(4,4))
        # print("s=\n",s)
        S , U , Vdag = cytnx.linalg.Svd(T)
        print("S=\n",S.get_block().numpy())
    
        # c3["u","r","aux"] 
        # c1["aux","d","l"]
        # SVD (top, right) - (bottom, left)
        print("# SVD (top, right) - (bottom, left)")
        # u, s, vt = common.svd(self.A, [0, 1], [2, 3], self.chi)
        # u, s, vt = svd(self.A, [0, 1], [2, 3], self.chi)
        # print("s shape=",s.shape)
        # print("s=\n",s)
        # print("S-s\n",S.get_block().numpy()-s)

        # try to match the sign of u and U
        # U[:,:,2] *= -1
        # U[:,:,3] *= -1
        print("U=\n",U.get_block().numpy())
        # print("u.shape=",u.shape)
        # # print("u=\n",u)
        # print("U-u=\n",U.get_block().numpy()-u)
    
        # Vdag[2,:,:] *= -1
        # Vdag[3,:,:] *= -1
        # print("vt.shape=",vt.shape)
        # print("vt=\n",vt)
        # print("Vdag-vt=\n",Vdag.get_block().numpy()-vt)
 
        S_sqrt = cytnx.linalg.Pow(S,0.5).set_name("S_sqrt")
        # print("S_sqrt=\n",S_sqrt.get_block().numpy())
        # print("S_sqrt-sqrt(s)=\n",S_sqrt.get_block().numpy()-np.sqrt(s))

        C3 = cytnx.Contract(U, S_sqrt).set_name("C3")
        C1 = cytnx.Contract(S_sqrt, Vdag).set_name("C1")
        C3.print_diagram()
        C1.print_diagram()

        # sqrt_s = np.sqrt(s)
        # c3 = u * sqrt_s[None, None, :]
        # c1 = vt * sqrt_s[:, None, None]

        # print("c3.shape=",c3.shape)
        # print("c3=\n",c3)
        # print("C3-c3=\n",C3.get_block().numpy()-c3)

        # print("c1.shape=",c1.shape)
        # print("c1=\n",c1)
        # print("C1-c1=\n",C1.get_block().numpy()-c1)

        # ------------
        T = T.permute(['u','l','r','d'])
        T.print_diagram()
        S , U , Vdag = cytnx.linalg.Svd(T)
        print("S=\n",S.get_block().numpy())
        
        # c2["u","l","aux"]
        # c0["aux,"r","d"]
        # SVD (top, left) - (right, bottom)
        # u, s, vt = common.svd(self.A, [0, 3], [1, 2], self.chi)
        # u, s, vt = svd(self.A, [0, 3], [1, 2], self.chi)
        # print("s shape=",s.shape)
        # print("s=\n",s)
        # print("S-s\n",S.get_block().numpy()-s)

        # try to match the sign of u and U
        # U[:,:,2] *= -1
        # U[:,:,3] *= -1
        print("U=\n",U.get_block().numpy())
        # print("u.shape=",u.shape)
        # print("u=\n",u)
        # print("U-u=\n",U.get_block().numpy()-u)
 
        # Vdag[2,:,:] *= -1
        # Vdag[3,:,:] *= -1
        # print("vt.shape=",vt.shape)
        # print("vt=\n",vt)
        print("Vdag-vt=\n",Vdag.get_block().numpy()-vt)

        # sqrt_s = np.sqrt(s)
        # c2 = u * sqrt_s[None, None, :]
        # c0 = vt * sqrt_s[:, None, None]

        S_sqrt = cytnx.linalg.Pow(S,0.5).set_name("S_sqrt")
        print("S_sqrt=\n",S_sqrt.get_block().numpy())
        print("S_sqrt-sqrt(s)=\n",S_sqrt.get_block().numpy()-np.sqrt(s))

        C2 = cytnx.Contract(U, S_sqrt).set_name("C2")
        C0 = cytnx.Contract(S_sqrt, Vdag).set_name("C0")
        C2.print_diagram()
        C0.print_diagram()

        print("C2-c2=\n",C2.get_block().numpy()-c2)
        print("C0-c0=\n",C0.get_block().numpy()-c0)


        C0.print_diagram()
        C1.print_diagram()
        C2.print_diagram()
        C3.print_diagram()

        from extension import conc
        tensors = [("C0", ["aux","r","d"]), ("C1", ["aux","d","l"]), ("C2", ["u","l","aux"]), ("C3", ["u","r","aux"])]
        contractions = [("C0", "r", "C1", "l"),("C1","d","C2","u"),("C2","l","C3","r"),("C3","u","C0","d")]
        net = conc(tensors, contractions)
        print(net)

        net.PutUniTensor("C0", C0, ["_aux_L","r","d"])
        net.PutUniTensor("C1", C1, ["_aux_L","d","l"])
        net.PutUniTensor("C2", C2, ["u","l","_aux_R"])
        net.PutUniTensor("C3", C3, ["u","r","_aux_R"])
        print(net)
        TT = net.Launch().set_name("TT")
        TT.print_diagram()
        tr = TT.Trace("C0_aux","C2_aux").Trace("C1_aux","C3_aux").item()
        print("tr=",tr)

        # Contraction
        self.A = np.tensordot(
            np.tensordot(c0, c1, (1, 2)), np.tensordot(c2, c3, (1, 1)), ((1, 3), (2, 0))
        )



        # normalize
        factor = self.trace()
        print("factor=",factor)
        print("tr-factor=",tr-factor)

        diff = TT.get_block().numpy()-self.A
        print(np.linalg.norm(diff))



        print(self.A.shape)
        # print(self.A)
        self.A /= factor
        exit()

        self.log_factors.append(np.log(factor))
        self.n_spins.append(2 * self.n_spins[-1])
        self.step += 1



    def run(self, step: int) -> None:
        # self.print_preamble()
        # self.print_legend()
        # self.print_results()
        
        time_start = time.perf_counter()
        pass
        for i in range(step):
            self.update()
            self.print_results()
        time_end = time.perf_counter()

        self.print_elapsed_time(time_end - time_start)


########################################################
# Test
########################################################
temp = 1
chi = 4
step = 1

TRG(temp, chi).run(step)
