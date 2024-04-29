import os
import sys
from tqdm import tqdm
from src.eval import evaluate
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
import torch
import argparse
import time
import json
import onnxruntime as ort
from src.models2 import ConditionalDiffusionTrainingNetwork
from utils.convert_to_windows import convert_to_windows
from utils.backprop import backprop, backprop_onnx

CHECKPOINT_FOLDER = './checkpoints'
RECORD_FOLDER = './record'

def load_dataset(dataset):
    loader = [] 
    folder = './data/' + dataset

    for file in ['train', 'test', 'validation', 'labels', 'labels_validation']:
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))

    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    validation_loader = DataLoader(loader[2], batch_size=loader[2].shape[0])
    return train_loader, test_loader, validation_loader, loader[3], loader[4]

def load_model(lr, window_size, dims, batch_size, noise_steps, denoise_steps, distill=False, sliding_size=1, loss_type='normal_l2'):

    diffusion_training_net = ConditionalDiffusionTrainingNetwork(dims, int(window_size), batch_size, noise_steps, denoise_steps, device=args.device, distill=distill, sliding_size=sliding_size, loss_type=loss_type).float()
    diffusion_prediction_net = ConditionalDiffusionTrainingNetwork(dims, int(window_size), batch_size, noise_steps, denoise_steps, train=False, device=args.device, sliding_size=sliding_size, loss_type=loss_type).float()
    optimizer = torch.optim.Adam(diffusion_training_net.parameters(), lr=float(lr))
    return diffusion_training_net, diffusion_prediction_net, optimizer

def save_model(experiment, diffusion_training_net, optimizer, epoch, diff_loss):
    folder = f'{CHECKPOINT_FOLDER}/{experiment}/'
    print('make checkpoint folder')
    os.makedirs(folder, exist_ok=True)
    file_path_diffusion = f'{folder}/diffusion.ckpt'
    torch.save({
        'epoch': epoch,
        'diffusion_loss': diff_loss,
        'model_state_dict': diffusion_training_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),}, file_path_diffusion)
    print('saved model at ' + folder)

def load_from_checkpoint(experiment, diffusion_training_net):
    folder = f'{CHECKPOINT_FOLDER}/{experiment}'
    file_path_diffusion = f'{folder}/diffusion.ckpt'
    # load diffusion
    checkpoint_diffusion = torch.load(file_path_diffusion)
    diffusion_training_net.load_state_dict(checkpoint_diffusion['model_state_dict'])
    return diffusion_training_net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
    parser.add_argument('--input',
                        type=str,
                        required=False,
                        default=None,
                        help="input_folder"),
    parser.add_argument('--output',
                        type=str,
                        required=False,
                        default=None,
                        help="output_folder"),
    parser.add_argument('--dataset',
                        type=str, 
                        required=False,
                        default='point_contexture',
                        help="dataset"),
    parser.add_argument('--training',
                        type=str, 
                        required=False,
                        default='diffusion',
                        help="model to train"),
    parser.add_argument('--lr',
                        type=str, 
                        required=False,
                        default='1e-4',
                        help="lerning rate"),
    parser.add_argument('--window_size',
                        type=str, 
                        required=False,
                        default='10',
                        help="window size"),
    parser.add_argument('--sliding_size',
                        type=str,
                        required=False,
                        default='1',
                        help="sliding window size"),
    parser.add_argument('--v',
                        type=bool, 
                        required=False,
                        default=False,
                        help="verbose"),
    parser.add_argument('--batch_size',
                        type=int, 
                        required=False,
                        default=128,
                        help="batch_size"),
    parser.add_argument('--noise_steps',
                        type=int, 
                        required=False,
                        default=100,
                        help="noise_steps"),
    parser.add_argument('--teacher_denoise_steps',
                        type=int, 
                        required=False,
                        default=10,
                        help="teacher_denoise_steps"),
    parser.add_argument('--denoise_steps',
                        type=int,
                        required=False,
                        default=5,
                        help="denoise_steps or student_denoise_steps"),
    parser.add_argument('--epoch',
                        type=int,
                        required=False,
                        default=10,
                        help="epoch"),
    parser.add_argument('--train_loss_begin',
                        type=float,
                        required=False,
                        default=0.25,
                        help="training loss begin"),
    parser.add_argument('--test_only',
                        type=bool, 
                        required=False,
                        default=False,
                        help="train new model or not"),
    parser.add_argument('--use_wandb',
                        type=bool,
                        required=False,
                        default=False,
                        help="use wandb or not"), #argparse bug，bool型参数无论传什么都是True
    parser.add_argument('--create_onnx',
                        type=bool,
                        required=False,
                        default=False,
                        help="create wandb or not"),
    parser.add_argument('--use_onnx',
                        type=bool,
                        required=False,
                        default=False,
                        help="use wandb or not"),
    parser.add_argument('--onnx_name',
                        type=str,
                        required=False,
                        default='trial.onnx',
                        help="onnx model"),
    parser.add_argument('--device',
                        type=str,
                        required=False,
                        default='cuda',
                        help="onnx cpu"),
    parser.add_argument('--loss_type',
                        type=str,
                        required=False,
                        default='normal_l2',
                        help="loss_type"),
    args = parser.parse_args()

    config = {
    "dataset": args.dataset,
    "training_mode": args.training,
    "learning_rate": float(args.lr),
    "window_size": int(args.window_size),
    "noise_steps":args.noise_steps,
    "batch_size": args.batch_size,
    }
    device = args.device

    if args.training in ('diffusion', 'distill'):
        experiment = 'diffv4'
    else:
        print('training mode error')
        sys.exit(1)
    if args.training == 'distill':
        distill = True
    else:
        distill = False
    experiment += f'_{args.dataset}_noise_{args.noise_steps}_denoise_{args.denoise_steps}_window_{args.window_size}'
    if distill:
        teacher_experiment = experiment + f'_{args.dataset}_noise_{args.noise_steps}_denoise_{args.teacher_denoise_steps}_window_{args.window_size}'
        if args.input:
            teacher_experiment = args.input
        experiment += '_distill'

    if args.output:
        experiment = args.output
    if args.input:
        experiment_input = args.input
    elif distill:
        experiment_input = teacher_experiment
    else:
        experiment_input = experiment

    use_wandb = args.use_wandb
    print(use_wandb)
    if use_wandb:
        import wandb
        wandb.init(project="anomaly-mts", config=config, group='tianf-11-29')
        wandb.run.name = experiment
    else:
        folder = f'{RECORD_FOLDER}/{experiment}/'
        print('make record folder')
        os.makedirs(folder, exist_ok=True)
        localtime = time.asctime()
        record_path = f'{folder}/record{time.time()}.txt'
        record_file = open(record_path, 'a')
        record_file.write(f'{json.dumps(vars(args))}')

    dataset_name = args.dataset
    window_size = int(args.window_size)
    sliding_size = int(args.sliding_size)
    train_loader, test_loader, validation_loader, labels, validation_labels = load_dataset(dataset_name)
    diffusion_training_net, diffusion_prediction_net, optimizer = \
                        load_model(args.lr, args.window_size, labels.shape[1], args.batch_size, args.noise_steps, args.denoise_steps, False, sliding_size, loss_type=args.loss_type)

    if distill:
        teacher_training_net, teacher_prediction_net, teacher_optimizer = \
            load_model(args.lr, args.window_size, labels.shape[1], args.batch_size, args.noise_steps, args.teacher_denoise_steps, False, sliding_size, loss_type=args.loss_type)
        teacher_training_net = load_from_checkpoint(teacher_experiment, teacher_training_net)
        teacher_training_net = teacher_training_net.to(device)
        teacher_prediction_net = teacher_prediction_net.to(device)

    diffusion_training_net = diffusion_training_net.to(device)
    diffusion_prediction_net = diffusion_prediction_net.to(device)
    # point_global: trainD(20000,5), test(20000,5), validation(10000,5)
    trainD, testD, validationD = next(iter(train_loader)), next(iter(test_loader)), next(iter(validation_loader))
    trainO, testO, validationO = trainD, testD, validationD
    if args.v:
        print(f'\ntrainD.shape: {trainD.shape}')
        print(f'testD.shape: {testD.shape}')
        print(f'validationD.shape: {validationD.shape}')
        print(f'labels.shape: {labels.shape}')
    if not use_wandb:
        record_file.write(f'\ntrainD.shape: {trainD.shape}')
        record_file.write(f'\ntestD.shape: {testD.shape}')
        record_file.write(f'\nvalidationD.shape: {validationD.shape}')
        record_file.write(f'\nlabels.shape: {labels.shape}')

    feats=labels.shape[1]    # feats:5
    trainD, testD, validationD = convert_to_windows(trainD, window_size, sliding_size), \
        convert_to_windows(testD, window_size, sliding_size), convert_to_windows(validationD, window_size, sliding_size)
    num_epochs = args.epoch
    epoch = -1
    max_roc_scores = [[0, 0, 0]] * 6
    max_f1_scores = [[0, 0, 0]] * 6
    roc_scores = []
    f1_scores = []
    f1_max = 0
    roc_max = 0
    validation_thresh = 0
    min_train_loss_begin = args.train_loss_begin
    early_stopping = 5
    if not args.test_only:
        for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
            if early_stopping == 0:
                break
            if distill:
                train_loss = backprop(e,diffusion_training_net, diffusion_prediction_net, trainD, optimizer, feats=labels.shape[1], device = args.device, training=True,\
                                      distill=distill, teacher_training_net=teacher_training_net, teacher_prediction_net=teacher_prediction_net)
            else:
                train_loss = backprop(e, diffusion_training_net, diffusion_prediction_net, trainD, optimizer, feats=labels.shape[1], device = args.device)
            if use_wandb:
                wandb.log({
                    'sum_loss_train': train_loss,
                    'epoch': e
                }, step=e)
            else:
                record_file.write(f'\nsum_loss_train: {train_loss} epoch: {e}')
            if train_loss < min_train_loss_begin + 0.001 or e >= num_epochs - 1:
                min_train_loss_begin = train_loss
                loss0, samples, onnx_shape = backprop(e, diffusion_training_net, diffusion_prediction_net, validationD, optimizer, feats=labels.shape[1], device = args.device, training=False)
                if use_wandb:
                    wandb.log({'val_loss': loss0.mean(), 'epoch': e}, step=e)
                else:
                    record_file.write(f'\nval_loss: {loss0.mean()} epoch: {e}')
                if sliding_size == 0:
                    loss0 = loss0.reshape(-1, feats)
                elif sliding_size == 1:
                    loss_tmp = loss0[0]
                    for i in range(1, loss0.shape[0]):
                        loss_tmp = np.append(loss_tmp, np.expand_dims(loss0[i][-1], axis=0), axis=0)
                    loss0 = loss_tmp
                else:
                    loss_tmp = loss0[0]
                    for i in range(1, loss0.shape[0]):
                        loss_tmp = np.append(loss_tmp, loss0[i][(-sliding_size):], axis=0)
                    loss0 = loss_tmp
                lossFinal = np.mean(np.array(loss0), axis=1)
                labelsFinal = (np.sum(validation_labels, axis=1) >= 1) + 0
                result, fprs, tprs, tn, fp, fn, tp = evaluate(lossFinal, labelsFinal) #时间很长
                thresholds_index = result['thresholds_index']
                result_roc = result["ROC/AUC"]
                result_f1 = result["f1"]
                record_file.write(f'\nf1_test: {result_f1}')
                record_file.write(f'\nroc_test: {result_roc}')
                record_file.write(f'\nindex:tn[{thresholds_index}]:{tn[thresholds_index]}')
                record_file.write(f'\nindex:fp[{thresholds_index}]:{fp[thresholds_index]}')
                record_file.write(f'\nindex:fn[{thresholds_index}]:{fn[thresholds_index]}')
                record_file.write(f'\nindex:tp[{thresholds_index}]:{tp[thresholds_index]}')
                early_stopping -= 1
                if result_f1 > f1_max:
                    save_model(experiment, diffusion_prediction_net, optimizer, e, train_loss)
                    early_stopping = 5
                    f1_max = result_f1
                    validation_thresh = result['threshold']
                    if use_wandb:
                        wandb.run.summary["best_f1"] = f1_max
                        wandb.run.summary["roc_for_best_f1"] = result_roc
                        wandb.run.summary["best_f1_epoch"] = e
                        wandb.run.summary["validation_thresh"] = validation_thresh
                    else:
                        record_file.write(f'\nbest_f1: {f1_max} epoch: {e}')
                        record_file.write(f'\nroc_for_best_f1: {result_roc} epoch: {e}')
                        record_file.write(f'\nbest_f1_epoch: {e} epoch: {e}')
                        record_file.write(f'\nvalidation_thresh: {validation_thresh} epoch: {e}')
                if result_roc > roc_max:
                    roc_max = result_roc
                    if use_wandb:
                        wandb.run.summary["f1_for_best_roc"] = result_f1
                        wandb.run.summary["best_roc"] = roc_max
                        wandb.run.summary["best_roc_epoch"] = e
                    else:
                        record_file.write(f'\nf1_for_best_roc: {result_f1} epoch: {e}')
                        record_file.write(f'\nbest_roc: {roc_max} epoch: {e}')
                        record_file.write(f'\nbest_roc_epoch: {e} epoch: {e}')
                if use_wandb:
                    wandb.log({'roc': result_roc, 'f1': result_f1}, step=e)
                else:
                    record_file.write(f'\nroc: {result_roc} f1: {result_f1} epoch: {e}')
                if args.v:
                    print(f"testing loss #{e}: {loss0.mean()}")
                    print(f"final ROC #{e}: {result_roc}")
                    print(f"F1 #{e}: {result_f1}")
                onnx_shape.to(device)
    # TEST ON TEST SET
    #load model from checkpoint
    diffusion_training_net, diffusion_prediction_net, optimizer = \
                        load_model(args.lr, args.window_size, labels.shape[1], args.batch_size, args.noise_steps, args.denoise_steps, False, sliding_size, loss_type=args.loss_type)
    diffusion_training_net = load_from_checkpoint(experiment_input, diffusion_training_net)
            
    diffusion_training_net = diffusion_training_net.to(device)
    diffusion_prediction_net = diffusion_prediction_net.to(device)

    if args.device == 'cpu':
        providers = ['CPUExecutionProvider']
    elif args.device == 'cuda':
        providers = ['CUDAExecutionProvider']
    if args.use_onnx:
        onnx_net = ort.InferenceSession(args.onnx_name, providers=providers)


    if args.test_only:
        if args.use_onnx:
            loss0, samples = backprop_onnx(args, onnx_net, validationD, feats, record_file, device)
        else:
            loss0, samples, onnx_shape = backprop(0, diffusion_training_net, diffusion_prediction_net, validationD,
                                                  optimizer, feats=labels.shape[1], device = args.device, training=False)
        if sliding_size == 0:
            loss0 = loss0.reshape(-1, feats)
        elif sliding_size == 1:
            loss_tmp = loss0[0]
            for i in range(1, loss0.shape[0]):
                loss_tmp = np.append(loss_tmp, np.expand_dims(loss0[i][-1], axis=0), axis=0)
            loss0 = loss_tmp
        else:
            loss_tmp = loss0[0]
            for i in range(1, loss0.shape[0]):
                loss_tmp = np.append(loss_tmp, loss0[i][(-sliding_size):], axis=0)
            loss0 = loss_tmp
        lossFinal = np.mean(np.array(loss0), axis=1)
        labelsFinal = (np.sum(validation_labels, axis=1) >= 1) + 0

        result, fprs, tprs, tn, fp, fn, tp = evaluate(lossFinal, labelsFinal)
        result_roc = result["ROC/AUC"]
        result_f1 = result["f1"]
        validation_thresh = result['threshold']
        if use_wandb:
            wandb.run.summary["f1_val"] = result_f1
            wandb.run.summary["roc_val"] = result_roc
            wandb.run.summary["f1_pa_val"] = result['f1_max']
            wandb.run.summary["validation_thresh"] = validation_thresh
        else:
            record_file.write(f'\nf1_val: {result_f1}')
            record_file.write(f'\nroc_val: {result_roc}')
            f1_max = result['f1_max']
            record_file.write(f'\nf1_pa_val: {f1_max}')
            record_file.write(f'\nvalidation_thresh: {validation_thresh}')
        if args.create_onnx:
            onnx_shape = onnx_shape.to(device)
    if args.create_onnx:
        torch.onnx.export(diffusion_training_net, onnx_shape, args.onnx_name,  input_names = ['input'],
                  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})
    if not args.use_onnx:
        loss0, samples, _ = backprop(0, diffusion_training_net, diffusion_prediction_net, testD, optimizer, feats=labels.shape[1], device = args.device, training=False)
    else:
        loss0, samples = backprop_onnx(args, onnx_net, testD, feats, record_file, device)
    if sliding_size == 0:
        loss0 = loss0.reshape(-1, feats)
    elif sliding_size == 1:
        loss_tmp = loss0[0]
        for i in range(1, loss0.shape[0]):
            loss_tmp = np.append(loss_tmp, np.expand_dims(loss0[i][-1], axis=0), axis=0)
        loss0 = loss_tmp
    else:
        loss_tmp = loss0[0]
        for i in range(1, loss0.shape[0]):
            loss_tmp = np.append(loss_tmp, loss0[i][(-sliding_size):], axis=0)
        loss0 = loss_tmp
    lossFinal = np.mean(np.array(loss0), axis=1)
    os.makedirs(f'lossfinal_samples/{experiment}/', exist_ok=True)
    np.save(f'lossfinal_samples/{experiment}/{args.dataset}_lossfinal.npy', lossFinal)
    np.save(f'lossfinal_samples/{experiment}/{args.dataset}_samples.npy', samples)
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    print("validation_thresh: ",validation_thresh)
    record_file.write(f'\nvalidation_thresh: {validation_thresh}')
    result, tn, fp, fn, tp = evaluate(lossFinal, labelsFinal, validation_thresh=validation_thresh)
    result_roc = result['ROC/AUC']
    result_f1 = result["f1"]
    f1_max = result['f1_max']
    thresholds_index = result['thresholds_index']
    if use_wandb:
        wandb.run.summary["f1_test"] = result_f1
        wandb.run.summary["roc_test" ] = result_roc
        wandb.finish()
    else:
        record_file.write(f'\nf1_test: {result_f1}')
        record_file.write(f'\nroc_test: {result_roc}')
        record_file.write(f'\nindex:tn[{thresholds_index}]:{tn[thresholds_index]}')
        record_file.write(f'\nindex:fp[{thresholds_index}]:{fp[thresholds_index]}')
        record_file.write(f'\nindex:fn[{thresholds_index}]:{fn[thresholds_index]}')
        record_file.write(f'\nindex:tp[{thresholds_index}]:{tp[thresholds_index]}')
        for i in range(50):
            record_file.write(f'\ntn[{i}]:{tn[i]}')
            record_file.write(f'\nfp[{i}]:{fp[i]}')
            record_file.write(f'\nfn[{i}]:{fn[i]}')
            record_file.write(f'\ntp[{i}]:{tp[i]}')
        record_file.close()
