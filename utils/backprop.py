import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time
import onnxruntime as ort

def backprop(epoch, diffusion_training_net, diffusion_prediction_net, data, optimizer, feats, device = "cuda", training = True,
             distill = False, teacher_training_net = None, teacher_prediction_net = None):
    l = nn.MSELoss(reduction = 'none')
    data_x = data.clone().detach().float() #(200,100,5)
    dataset = TensorDataset(data_x, data_x) #200, 每个为(2,100,5) 比如dataset[0][1]和dataset[0][1]一样
    bs = diffusion_training_net.batch_size #bs 8
    dataloader = DataLoader(dataset, batch_size = bs) # dataloader={Dataloader:25}
    w_size = diffusion_training_net.window_size
    l1s = []
    samples = []
    all_time = 0.0
    if training:
        diffusion_training_net.train()
        if distill:
            teacher_prediction_net.load_state_dict(teacher_training_net.state_dict())
            teacher_prediction_net.eval()
            teacher_training_net.eval()
        for d, _ in dataloader:
            window = d
            window = window.to(device)
            window = window.reshape(-1, w_size, feats)
            loss, _ = diffusion_training_net(window, teacher_prediction_net)
            l1s.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
        return np.mean(l1s)
    else:
        with torch.no_grad():
            diffusion_prediction_net.load_state_dict(diffusion_training_net.state_dict())
            diffusion_prediction_net.eval()
            diffusion_training_net.eval()
            for d, _ in dataloader:
                window = d
                window = window.to(device)
                window_reshaped = window.reshape(-1, w_size, feats)
                onnx_shape = torch.randn_like(window_reshaped)
                start = time.time()
                _, x_recon = diffusion_prediction_net(window_reshaped)
                end = time.time()

                x_recon = x_recon.transpose(2, 1)
                samples.append(x_recon)
                loss = l(x_recon, window_reshaped)
                l1s.append(loss)
        return torch.cat(l1s).detach().cpu().numpy(), torch.cat(samples).detach().cpu().numpy(), onnx_shape.to(device)

def backprop_onnx(args, onnx_net, data, feats, record_file, device = "cuda"):
    l = nn.MSELoss(reduction='none')
    data_x = data.clone().detach().float()
    dataset = TensorDataset(data_x, data_x)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    w_size = int(args.window_size)
    samples = []
    l1s = []
    all_time = 0.0
    for d, _ in dataloader:
        window = d
        window = window.to(device)
        window_reshaped = window.reshape(-1, w_size, feats)
        start = time.time()
        x_recon = onnx_net.run(None, {'input': window_reshaped.cpu().numpy()})
        end = time.time()
        all_time = all_time + end -start
        x_recon = torch.tensor(x_recon[0])
        x_recon = x_recon.transpose(2, 1)
        samples.append(x_recon)
        loss = l(x_recon, window_reshaped)
        l1s.append(loss)
    print(f"time: {all_time}")
    record_file.write(f'time: {all_time}')
    return torch.cat(l1s).detach().cpu().numpy(), torch.cat(samples).detach().cpu().numpy()