import numpy as np
import warnings
from scipy import signal

class SMMatrices:
    def __init__(self):

        self.sm_1 = np.array([1.0], dtype=np.complex64)

        self.sm_2_20 = np.array([[1., 1.], [1., -1.]], dtype=np.complex64) / np.sqrt(2)

        self.sm_2_40 = np.array([[1., 1.j], [1.j, 1.]], dtype=np.complex64) / np.sqrt(2)
            
        sm_3_20 = np.array([
            [-2*np.pi/16, -2*np.pi/(80/33),  2*np.pi/(80/3)],
            [ 2*np.pi/(80/23), 2*np.pi/(48/13), 2*np.pi/(240/13)],
            [-2*np.pi/(80/13), 2*np.pi/(240/37), 2*np.pi/(48/13)]
        ])
        self.sm_3_20 = np.exp(1.j * sm_3_20).astype(np.complex64) / np.sqrt(3)

        sm_3_40 = np.array([
            [-2*np.pi/16, -2*np.pi/(80/13),  2*np.pi/(80/23)],
            [-2*np.pi/(80/37), -2*np.pi/(48/11), -2*np.pi/(240/107)],
            [ 2*np.pi/(80/7), -2*np.pi/(240/83), -2*np.pi/(48/11)]
        ])
        self.sm_3_40 = np.exp(1.j * sm_3_40).astype(np.complex64) / np.sqrt(3)


class SMRemover:
    def __init__(self):
        self.sm = SMMatrices()

    def __call__(self, csi: np.array, rate: int):
        """
        csi: (batch, nsub, nrx, ntx) or (nsub, nrx, ntx)
        """

        # Ensure batch dimension
        if csi.ndim == 3:
            csi = csi.unsqueeze(0)   # → (1, nsub, nrx, ntx)

        B, nsub, nrx, ntx = csi.shape

        # ntx = 1 → no need to process
        if ntx == 1:
            return csi

        # Choose SM matrix
        cond_40 = (rate & 2048) == 2048

        if cond_40:
            if ntx == 3:
                sm_mtx = self.sm.sm_3_40
            else:
                sm_mtx = self.sm.sm_2_40
        else:
            if ntx == 3:
                sm_mtx = self.sm.sm_3_20
            else:
                sm_mtx = self.sm.sm_2_20

        # sm^H (transpose + conjugate)
        sm_H = sm_mtx.transpose(0, 1).conj()   # (ntx, ntx)

        # ---------- GPU accelerated batched matmul ----------
        # csi: (B, nsub, nrx, ntx)
        # sm_H: (ntx, ntx)
        # want: csi × sm_H  over the last 2 dims
        ret = np.matmul(csi, sm_H)  # broadcasting OK

        return ret


class CSIscaler:
    def __init__(self, rate=0x1c113):
        """
        0x4101: Tx ant-A only
        0x113: unused
        0x1c113: Tx ant ABC
        """
        self.rate = rate
        self.sm_remover = SMRemover()
        self.csilist = None
        self.dynamic_csi = None
        self.datetimelist = None

    def get_scaled_csilist(self, csilist, rssilist, noiselist, agclist, is_agc=False):
        """
        csilist:   (N, Nt, Nr, Nc)  complex64
        rssilist:  (N, 3)           float
        noiselist: (N,)             float
        agclist:   (N,)             float
        """

        N = csilist.shape[0]

        # ---- csi_pwr ----
        # abs(csi)^2
        csi_pwr = np.sum(np.abs(csilist) ** 2, axis=(1,2,3))   # shape (N,)

        # ---- rssi_pwr ----
        rssi_pwr = 10.0 ** (rssilist / 10.0)
        rssi_pwr[rssi_pwr == 1] = 0
        rssi_pwr = np.sum(rssi_pwr, axis=1) / (10.0 ** ((44.0 + agclist) / 10.0))

        # ---- scale ----
        scale = rssi_pwr / (csi_pwr / 30.0)

        # ---- ntx ----
        ntx = np.array([c.shape[0] for c in csilist])  # Nt per entry
        nrx = np.array([c.shape[1] for c in csilist])

        # ---- noise ----
        noiselist = np.copy(noiselist)
        noiselist[noiselist == -127] = -92
        total_noise_pwr = 10.0 ** (noiselist / 10.0) + scale * ntx * nrx

        # ---- multiply_ntx ----
        multiply_ntx = np.ones(N)
        multiply_ntx[ntx == 2] = np.sqrt(2.0)
        multiply_ntx[ntx == 3] = np.sqrt(10.0 ** (4.5 / 10.0))

        scale_factor = np.sqrt(scale / total_noise_pwr) * multiply_ntx
        scale_factor = scale_factor.reshape(N, 1, 1, 1)

        # ---- scale CSI ----
        ret = csilist * scale_factor

        return ret
        
    def load_npy(self, path, remove_sm=False):
        """
        Provide the path of csic.npy
        csic -> [real_csilist, imag_csilist]
        rssi -> uint_rssilist
        csin -> [int_noiselist, uint_agclist]
        csit -> [timelist, datetimelist]
        """

        real_imag_csilist = np.load(path, allow_pickle=True)
        rssilist = np.load(path.replace('csic', 'rssi'), allow_pickle=True)
        noise_agclist = np.load(path.replace('csic', 'csin'), allow_pickle=True)
        time_datetimelist = np.load(path.replace('csic', 'csit'), allow_pickle=True)
        
        real_csilist, imag_csilist = real_imag_csilist[0], real_imag_csilist[1]
        noiselist, agclist = noise_agclist[0], noise_agclist[1]
        timelist, datetimelist = time_datetimelist[0], time_datetimelist[1]

        csilist = real_csilist + 1j * imag_csilist

        print("loaded")
        csilist = self.get_scaled_csilist(csilist, rssilist, noiselist, agclist)

        print("scaled")

        if len(datetimelist) != csilist.shape[0]:
            warnings.warn(f"different length: datetimelist {len(datetimelist)} vs csilist {csilist.shape[0]}")
            minlen = min(len(datetimelist), csilist.shape[0])
            csilist = csilist[:minlen]
            timelist = timelist[:minlen]
            rssilist = rssilist[:minlen]
            datetimelist = datetimelist[:minlen]

        # Transpose into (pkt, nsub, ntx, nrx)
        csilist = csilist.swapaxes(1, 3)

        if remove_sm:
            csilist = self.sm_remover(csilist, self.rate)
            print("removed sm")

        self.csilist = csilist
        self.datetimelist = datetimelist

        return csilist, timelist, rssilist, datetimelist

    @staticmethod
    def highpass(fs=1000, cutoff=5, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def preprocess(self, csi: np.array, ref='tx', ref_ant=1, ret='cpu'):

        # I. Didive Tx1
        re_csi = (np.abs(csi) + 1.e-6) * np.exp(1.j * np.angle(csi))
        if ref == 'rx':
            denom = re_csi[..., ref_ant, np.newaxis, :]  # [..., 1]
            denom = denom.repeat(2, 3)               # repeat 3 times
            csi = csi / denom
        elif ref == 'tx':
            denom = re_csi[..., ref_ant, np.newaxis]  # [..., 1]
            denom = denom.repeat(3, 3)               # repeat 3 times along last dim
            csi = csi / denom
        else:
            pass

        # II. High-pass filter with cupy
        b, a = self.highpass()
        dynamic_csi = np.zeros_like(csi)
        nsub, nrx, ntx = csi.shape[-3:]
        for sub in range(nsub):
            for rx in range(nrx):
                for tx in range(ntx):
                    dynamic_csi[:, sub, rx, tx] = signal.filtfilt(b, a, csi[:, sub, rx, tx])

        self.dynamic_csi = dynamic_csi
        return dynamic_csi

    def save(self, csi=None, path='./'):
        """
        Specify the name of csi.npy
        """
        if csi is None:
            csi = self.dynamic_csi
        np.save(f"{path}", csi)
        print("Saved CSI!")
        if self.datetimelist is not None:
            np.save(f"{path.replace('csi', 'csitime')}", self.datetimelist)
            print("Saved csitime!")


if __name__ == '__main__':

    pass