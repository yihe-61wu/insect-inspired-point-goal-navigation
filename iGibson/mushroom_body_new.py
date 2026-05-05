import numpy as np
from neuro_utils import k_winners_take_all_hash, bloom_filter


class MultiOutputMushroomBody:
    def __init__(self, N_pn, N_kc, N_mbon, N_pn_perkc, S_kc, kc2mbon_rule):
        self.N_kc_WTA = int(S_kc * N_kc)
        self.pn2kc = k_winners_take_all_hash(N_pn, N_kc, N_pn_perkc / N_pn, self.N_kc_WTA)
        self.kc2mbon = bloom_filter(kc2mbon_rule, N_kc, N_mbon)

    def hashing(self, pn):
        kc = self.pn2kc.hashing(pn)
        return kc

    def evaluating(self, kc):
        mbon = self.kc2mbon.evaluating(kc, self.N_kc_WTA)
        return mbon

    def learning(self, kc, rate):
        [self.kc2mbon.learning(hash, rate) for hash in np.atleast_2d(kc)]


if __name__ == '__main__':
    N_pn, N_kc, N_mbon = 20, 200, 2
    N_pn_perkc = 10

    soMB = MultiOutputMushroomBody(N_pn, N_kc, N_mbon, N_pn_perkc, S_kc=0.1, kc2mbon_rule='decay', rate_learn=[0.5, 1])
    sample_learn = soMB.hashing(np.random.randint(2, size=(3, N_pn)))
    soMB.learning(sample_learn)
    sample_test = soMB.hashing(np.random.randint(2, size=(1, N_pn)))
    print(soMB.evaluating(sample_learn))
    print(soMB.evaluating(sample_test))