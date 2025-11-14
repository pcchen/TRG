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

    def print_results(self) -> None:
        n_spin = self.n_spins[-1]
        f = self.free_energy()
        f_err = (f - self.f_exact) / abs(self.f_exact)
        results = [f"{self.step:04d}", f"{n_spin:.12e}", f"{f:.12e}", f"{f_err:.12e}"]
        print(" ".join(results))

    def log_Z(self) -> float:
        trace_a = self.trace()
        # if trace_a < 0.0:
        #     logging.warning("Negative trace_a %e (%d)", trace_a, self.step)
        log_z = np.sum(np.array(self.log_factors) / np.array(self.n_spins))
        log_z += np.log(abs(trace_a)) / self.n_spins[-1]
        return log_z

    def free_energy(self) -> float:
        return -self.temp * self.log_Z()

    def update(self) -> None:
        T = self.A.set_rowrank(2)

        T = T.permute(['u','r','d','l'])
        # T.print_diagram()
        S , U , Vdag = cytnx.linalg.Svd(T)
        # print("S=\n",S.get_block().numpy())    
        # U[:,:,2] *= -1
        # U[:,:,3] *= -1
        # print("U=\n",U.get_block().numpy())    
        # Vdag[2,:,:] *= -1
        # Vdag[3,:,:] *= -1 
        # print("Vdag=\n",Vdag.get_block().numpy())
        S_sqrt = cytnx.linalg.Pow(S,0.5).set_name("S_sqrt")
        # print("S_sqrt=\n",S_sqrt.get_block().numpy())
        C3 = cytnx.Contract(U, S_sqrt).set_name("C3")
        C1 = cytnx.Contract(S_sqrt, Vdag).set_name("C1")
        # C3.print_diagram()
        # C1.print_diagram()

        T = T.permute(['u','l','r','d'])
        # T.print_diagram()
        S , U , Vdag = cytnx.linalg.Svd(T)
        # print("S=\n",S.get_block().numpy())
        # print("U=\n",U.get_block().numpy())
        # print("Vdag=\n",Vdag.get_block().numpy())
        S_sqrt = cytnx.linalg.Pow(S,0.5).set_name("S_sqrt")
        # print("S_sqrt=\n",S_sqrt.get_block().numpy())
        C2 = cytnx.Contract(U, S_sqrt).set_name("C2")
        C0 = cytnx.Contract(S_sqrt, Vdag).set_name("C0")
        # C2.print_diagram()
        # C0.print_diagram()

        from extension import conc
        tensors = [("C0", ["aux","r","d"]), ("C1", ["aux","d","l"]), ("C2", ["u","l","aux"]), ("C3", ["u","r","aux"])]
        contractions = [("C0", "r", "C1", "l"),("C1","d","C2","u"),("C2","l","C3","r"),("C3","u","C0","d")]
        net = conc(tensors, contractions)
        # print(net)
        net.PutUniTensor("C0", C0, ["_aux_L","r","d"])
        net.PutUniTensor("C1", C1, ["_aux_L","d","l"])
        net.PutUniTensor("C2", C2, ["u","l","_aux_R"])
        net.PutUniTensor("C3", C3, ["u","r","_aux_R"])
        # print(net)
        TT = net.Launch().set_name("TT").relabel(["u","r","d","l"])
        # TT.print_diagram()
        trTT = TT.Trace("u","d").Trace("r","l").item()
        # print("trTT=",trTT)

        # TT = net.Launch().set_name("TT")
        # TT.print_diagram()
        # trTT = TT.Trace("C0_aux","C2_aux").Trace("C1_aux","C3_aux").item()
        # print("trTT=",trTT)

        # normalize
        factor = trTT
        # print("factor=",factor)
        self.A = TT/factor
        self.log_factors.append(np.log(factor))
        self.n_spins.append(2 * self.n_spins[-1])
        self.step += 1

    def run(self, step: int) -> None:
        # self.print_preamble()
        # self.print_legend()
        # self.print_results()
        
        time_start = time.perf_counter()
        print(f"{self.step:04d}", f"{self.n_spins[-1]:.12e}", -np.sum(np.array(self.log_factors) / np.array(self.n_spins)))
        # print(-np.sum(np.array(self.log_factors) / np.array(self.n_spins)))
        for i in range(step):
            self.update()
            # self.print_results()
            print(f"{self.step:04d}", f"{self.n_spins[-1]:.12e}", -np.sum(np.array(self.log_factors) / np.array(self.n_spins)))
            # print(-np.sum(np.array(self.log_factors) / np.array(self.n_spins)))
        time_end = time.perf_counter()

        self.print_elapsed_time(time_end - time_start)


########################################################
# Test
########################################################
temp = 1
chi = 4
step = 2

TRG(temp, chi).run(step)
