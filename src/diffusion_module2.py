import torch
import torch.nn.functional as F
from tqdm import tqdm

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

#一堆(300)的tensor
timesteps = 3000

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps) # betas: 一堆(300)的tensor

# define alphas 
alphas = 1. - betas # alphas: 一堆(300)的tensor
alphas_cumprod = torch.cumprod(alphas, axis=0) # alphas_cumprod: 一堆(300)的tensor, torch.cumprod累积乘法
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) #填充，相当于所有右移一位并在开始加一个1.0
sqrt_recip_alphas = torch.sqrt(1.0 / alphas) # sqrt_recip_alphas: 一堆(300)的tensor

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) #sqrt_alphas_cumprod 一堆(300)的tensor
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)#sqrt_one_minus_alphas_cumprod (300)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) #posterior_variance (300)

def extract(a, t, x_shape): #a:tensor300 t:tensor8 x_shape:size(8,1,100,5)
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu()) # tensor.gather按后面那个选择元素
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    #sqrt_alphas_cumprod_t (8,1,1,1) sqrt_one_minus_alphas_cumprod_t (8,1,1,1)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def p_losses(denoise_model, x_start, t, distill=False, teacher=None, noise=None, loss_type="normal_l2", sliding_size=1): #(denoise_model:Unet x_start(8,1,100,5) t(8))
    if noise is None:
        noise = torch.randn_like(x_start) # 一样的(8,1,100,5)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise) #将噪声加到x_noisy上
    predicted_noise = denoise_model(x_noisy, t)
    if distill:
        with torch.no_grad():
            teacher_predict_noise = teacher(x_noisy, 2*t)
    #if train:
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        if distill:
            if sliding_size != 0:
                loss = 0.9*F.mse_loss(noise[:,:,-sliding_size:], predicted_noise[:,:,-sliding_size:]) + 0.1*F.mse_loss(teacher_predict_noise[:,:,-sliding_size:], predicted_noise[:,:,-sliding_size:])
            else:
                loss = 0.9 * F.mse_loss(noise, predicted_noise) + 0.1 * F.mse_loss(teacher_predict_noise, predicted_noise)
        else:
            if sliding_size != 0:
                loss = F.mse_loss(noise[:,:,-sliding_size:], predicted_noise[:,:,-sliding_size:])
            else:
                loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == 'normal_l2':
        if distill:
            loss = 0.9 * F.mse_loss(noise, predicted_noise) + 0.1 * F.mse_loss(teacher_predict_noise, predicted_noise)
        else:
            loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
    # else:
    #     x_recon = (x_noisy - extract(sqrt_one_minus_alphas_cumprod, t, x_noisy.shape) * predicted_noise) / extract(sqrt_alphas_cumprod, t, x_noisy.shape)
    #     loss = F.mse_loss(predicted_noise, noise, reduction='none')
    return loss


##### SAMPLING #######

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape, x_start, denoise_steps, device):
    #device = next(model.parameters()).device
    #timesteps = 200
    timesteps = denoise_steps
    #device = 'cuda'

    b = shape[0]
    noise = torch.randn_like(x_start)
    img = q_sample(x_start=x_start, t=torch.full((b,), timesteps, device=device, dtype=torch.long), noise=noise)

    for i in tqdm(reversed(range(0, int(timesteps))), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        #imgs.append(img.cpu().numpy())
    return img

@torch.no_grad()
def sample(model, shape, x_start, denoise_steps, device):
    return p_sample_loop(model, shape=shape, x_start=x_start, denoise_steps=denoise_steps, device=device)

