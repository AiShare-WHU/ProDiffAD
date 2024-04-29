import torch

def convert_to_windows(data, n_window, n_sliding):
    if n_sliding==0:
        windows = list(torch.split(data, n_window))
        for i in range (n_window-windows[-1].shape[0]):
            windows[-1] = torch.cat((windows[-1], windows[-1][-1].unsqueeze(0)))
    else:
        windows = []
        if (len(data) - n_window) % n_sliding == 0:
            for i in range(0, len(data) - n_window + 1, n_sliding):
                windows.append(data[i : n_window + i])
        else:
            for i in range(0, len(data) - n_window + 1 + n_sliding, n_sliding):
                windows.append(data[i : n_window + i])
            for i in range(n_window - windows[-1].shape[0]):
                windows[-1] = torch.cat((windows[-1], windows[-1][-1].unsqueeze(0)))
    return torch.stack(windows)