import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split

def sg_filter_gpu(csi, window_size=21, poly_order=3, dim=-3):
    """
    GPU Savitzky-Golay filter for multi-dimensional PyTorch tensor.
    
    Args:
        csi: torch.Tensor, shape (batch * packet * sub * rx), must be on GPU
        window_size: int, odd, SG window length
        poly_order: int, polynomial order
        dim: int, dimension along which to filter (packet dim)
    
    Returns:
        filtered torch.Tensor, same shape as input, on GPU
    """
    assert window_size % 2 == 1, "window_size must be odd"
    assert poly_order < window_size, "poly_order must be smaller than window_size"
    device = csi.device
    dtype = csi.dtype
    
    half = window_size // 2
    # build design matrix for SG filter
    x = torch.arange(-half, half+1, device=device, dtype=dtype)
    A = torch.stack([x**i for i in range(poly_order+1)], dim=1)  # [window_size, poly_order+1]
    AtA_inv = torch.linalg.pinv(A)  # pseudo-inverse
    coeffs = AtA_inv[0]  # smoothing coefficients (first row)
    
    # reshape coeffs for 1D conv
    coeffs = coeffs.flip(0).view(1, 1, -1)  # [out_channels, in_channels, kernel_size]
    
    # move filtering dim to last
    csi_perm = csi.transpose(dim, -1)
    orig_shape = csi_perm.shape
    csi_flat = csi_perm.reshape(-1, 1, orig_shape[-1])  # [batch, channel=1, time]
    
    # pad for same length
    pad = half
    csi_padded = torch.nn.functional.pad(csi_flat, (pad, pad), mode='replicate')
    
    # conv1d filtering
    filtered = torch.nn.functional.conv1d(csi_padded, coeffs)
    
    # reshape back
    filtered = filtered.view(*orig_shape)
    filtered = filtered.transpose(dim, -1)
    
    return filtered


def phase_difference_gpu(csi_real, csi_imag):
    def cal_pd(u):
        pd = u[:, 1:, 0] * u[:, :-1, 0].conj()
        return torch.cat((torch.real(pd), torch.imag(pd)), axis=-1)
    
    try:
        # CSI shape = batch * packet * sub * rx
        csi_complex = csi_real + 1.j * csi_imag

        # Reshape into batch * rx * (sub * packet)
        u, *_ = torch.linalg.svd(csi.permute(0, 3, 2, 1).reshape(csi.shape[0], 3, -1), full_matrices=False)
        # AoA = batch * 4 (real & imag of 2)
        aoa = cal_pd(u)
        
        # Reshape into batch * sub * (rx * packet)
        u, *_ = torch.linalg.svd(csi.permute(0, 2, 3, 1).reshape(csi.shape[0], 30, -1), full_matrices=False)
        # ToF = batch * 58 (real & imag of 29)
        tof = cal_pd(u)
        
        # Concatenate as a flattened vector
        pd = torch.cat((aoa, tof), axis=-1)

    except Exception as e:
        print(f'FilterPD aborted due to {e}')
    
    return pd



class MyDatasetLite(Dataset):
    """
    DATASET wrapper
    Load CSI, shape, ind
    """

    def __init__(self,
                 data,
                 csi_len=300,
                 img_len=1,
                 *args, **kwargs):

        self.data = data
        self.alignment = 'tail'
        self.csi_len = csi_len
        self.img_len = img_len

    def __getitem__(self, index):
        """
        On-the-fly: select windowed CSI (and pd)
        """
        ret: dict = {}

        ret['ind'] = index
        ret['shape'] = ret['shape'][index].unsqueeze(1)

        csi_ind = self.data['ind'][index]
        csi = self.data['csi'][csi_ind - self.csi_len, csi_ind]
        ret['csi'], ret['pd'] = self.filter_csi(csi)

        return ret

    def filter_csi(self, csi):
        csi_real = sg_filter_gpu(torch.real(csi))
        csi_imag = sg_filter_gpu(torch.imag(csi))

        # CSI sample (batch *  packet * sub * (rx * 2))
        csi = torch.cat((csi_real, csi_imag), axis=-1)

        # batch *  (rx * 2) * sub * packet
        csi = csi.permute(0, 3, 2, 1)

        # Calculate pd
        pd = phase_difference_gpu(csi_real, csi_imag)

        return csi, pd

    def __len__(self):
        return len(self.data['ind'])



class MyDataLoaderLite:
    def __init__(self, device='cuda'):
        self.data: dict = {}
        self.device = device

    def load(self, path):
        paths = os.walk(path)
        print(f'Loading {path}...\n')
        for path, _, file_lst in paths:
            for file_name in file_lst:
                file_name_, ext = os.path.splitext(file_name)

                if ext == 'npy':
                    self.data[file_name_] = torch.from_numpy(np.load(os.path.join(path, file_name))).to(device=self.device)

                print(f'Loaded {file_name}')

        print(f"\nLoad complete!")

    def gen_loaders(self, batch_size=64, num_workers=10, split_ratio=0.8, pin_memory=True, shuffle_test=False):
        """
        Simple case: only generates train/valid loader or test loader at once.
        For test loader, specify split_ratio=1.
        """
        dataset = MyDataLoaderLite(data)
        train_size, valid_size, test_size = 0, 0, 0
        train_loader, valid_loader, test_loader = None, None, None

        if split_ratio < 1:
            train_size = int(split_ratio * len(dataset))
            valid_size = len(dataset) - train_size
            train_set, valid_set = random_split(dataset, [train_size, valid_size])

        else:
            test_size = len(dataset)
            test_set = dataset

        print(f' Train dataset length = {train_size}\n'
            f' Valid dataset length = {valid_size}\n'
            f' Test dataset length = {test_size}\n'
            f' Batch size = {batch_size}')

        if train_size > 0:
            train_loader = DataLoader(train_set, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers,
                                    drop_last=True, 
                                    pin_memory=pin_memory
                                    )

        if valid_size > 0:
            valid_loader = DataLoader(valid_set, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers,
                                    drop_last=True, 
                                    pin_memory=pin_memory
                                    )

        if test_size > 0:
            test_loader = DataLoader(test_set, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    drop_last=False,
                                    shuffle=shuffle_test,
                                    )

        return train_loader, valid_loader, test_loader