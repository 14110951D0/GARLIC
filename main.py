import os
import sys
import time
import random
import argparse
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import *
from model import *


def get_args():
    """
    Parses command-line arguments for model training and evaluation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data', type=str, default='MIMICIII',
                        choices=['P12', 'P19', 'MIMICIII'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run', type=int, default=5)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr_rec', type=float, default=1e-3)
    parser.add_argument('--lr_cls', type=float, default=5e-5)
    parser.add_argument('--wd_rec', type=float, default=5e-4)
    parser.add_argument('--wd_cls', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--feature_dim', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--window_size', type=int, default=9)
    parser.add_argument('--lambda_cls', type=float, default=5)
    parser.add_argument('--lambda_graph', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--warm_up', type=int, default=40)
    parser.add_argument('--model', type=str, default='GARLIC')
    return parser.parse_args()


def setup_paths(args):
    """
    Sets up logging and model checkpoint directories based on input arguments.
    """
    timestamp = time.ctime().replace(' ', '-').replace(':', '-')
    log_path = f'./logs/{args.data}/{args.model}_{args.window_size}_{timestamp}.log'
    model_dir = f'./models/{args.data}/'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return log_path, model_dir


def prepare_batch(batch, device):
    """
    Converts a batch of raw input data into PyTorch tensors and moves them to the specified device.
    """
    arr = torch.tensor(np.array([b['arr'] for b in batch]), dtype=torch.float32).to(device)
    time = torch.tensor(np.array([b['time'] for b in batch]) / 60, dtype=torch.float32).to(device)
    mask = torch.tensor(np.array([b['mask'] for b in batch]), dtype=torch.float32).to(device)
    label = torch.tensor(np.array([b['label'] for b in batch]), dtype=torch.float32).to(device)
    time = time.squeeze(-1).permute(1, 0)
    return arr, mask, time, label


def evaluate(model, data, args, criterion=None):
    """
    Evaluates the model on a given dataset using AUROC and AUPRC metrics.
    """
    model.eval()
    all_labels, all_outputs = [], []
    loss_total = 0
    with torch.no_grad():
        for i in range(0, len(data), args.batch_size):
            batch = data[i:i + args.batch_size]
            arr, mask, time, label = prepare_batch(batch, args.device)
            out, loss_m, _ = model(arr, mask, time)
            out = torch.sigmoid(out)
            loss = criterion(out, label)
            loss_total += loss.item()
            all_labels.append(label)
            all_outputs.append(out)
    labels = torch.cat(all_labels, 0).cpu().detach().numpy()
    outputs = torch.cat(all_outputs, 0).cpu().detach().numpy()
    auc = roc_auc_score(labels, outputs)
    auprc = average_precision_score(labels, outputs)
    return auc, auprc, loss_total / len(data) if criterion else 0


if __name__ == '__main__':
    args = get_args()
    device = args.device
    log_path, model_dir = setup_paths(args)
    logger = get_logger(log_path)
    logger.info(args)
    data = load_data(args.data)

    auc_results = []
    auprc_results = []
    # Repeat the training process
    for run in range(args.run):
        logger.info(f'Run {run}')
        seed = args.seed + run
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        train_data, val_data, test_data = get_data_split(data)

        # Initialize the model architecture
        input_dim = train_data[0]['arr'].shape[-1]
        max_len = train_data[0]['arr'].shape[0]
        model = Model(input_dim, args.feature_dim, args.hidden_size, args.window_size, max_len, device).to(device)

        # Separate model parameters for reconstruction and prediction stages
        rec_params, pred_params = [], []
        for name, param in model.named_parameters():
            if any(k in name for k in ['t_attention', 'classifier', 'aggregator', 'predictor']):
                pred_params.append(param)
            else:
                rec_params.append(param)

        optimizer = optim.Adam([
            {'params': rec_params, 'lr': args.lr_rec, 'weight_decay': args.wd_rec},
            {'params': pred_params, 'lr': args.lr_cls,'weight_decay': args.wd_cls}
        ])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                                               threshold=1e-4, threshold_mode='rel',
                                                               cooldown=0, min_lr=1e-5, eps=1e-08)

        best_val_epoch = 0
        best_stage_epoch = 0
        best_val_loss = float('inf')
        mode = 'rec'
        rand_int = random.randint(10000, 100000)
        logger.info('random number: %d' % rand_int)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            all_labels = torch.tensor([]).to(device)
            all_outputs = torch.tensor([]).to(device)

            # Apply decoupled alternating training
            if mode == 'rec':
                for i in range(0, len(train_data), args.batch_size):
                    batch = train_data[i:i + args.batch_size]
                    arr, mask, time, label = prepare_batch(batch, device)

                    optimizer.zero_grad()
                    out, loss_m, _ = model(arr, mask, time)

                    cls_loss = nn.BCEWithLogitsLoss()(out, label) * args.lambda_cls
                    lasso_loss = torch.sum(torch.abs(model.message_passing.adj)) * args.lambda_graph
                    loss = loss_m + cls_loss + lasso_loss
                    out = torch.sigmoid(out)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    all_labels = torch.cat([all_labels, label], 0)
                    all_outputs = torch.cat([all_outputs, out], 0)

                total_loss /= len(train_data)
                auc = roc_auc_score(all_labels.cpu().detach().numpy(), all_outputs.cpu().detach().numpy())
                auprc = average_precision_score(all_labels.cpu().detach().numpy(), all_outputs.cpu().detach().numpy())
                scheduler.step(auc)

                logger.info(f"Epoch {epoch}, {mode} loss: {total_loss:.4f}, AUC: {auc:.4f}, AUPRC: {auprc:.4f}")
            elif mode == 'pred':
                for i in range(0, len(train_data), args.batch_size):
                    batch = train_data[i:i + args.batch_size]
                    arr, mask, time, label = prepare_batch(batch, device)

                    optimizer.zero_grad()
                    out, loss_m, _ = model(arr, mask, time)

                    cls_loss = nn.BCEWithLogitsLoss()(out, label) * args.lambda_cls
                    loss = cls_loss
                    out = torch.sigmoid(out)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    all_labels = torch.cat([all_labels, label], 0)
                    all_outputs = torch.cat([all_outputs, out], 0)

                total_loss /= len(train_data)
                auc = roc_auc_score(all_labels.cpu().detach().numpy(), all_outputs.cpu().detach().numpy())
                auprc = average_precision_score(all_labels.cpu().detach().numpy(), all_outputs.cpu().detach().numpy())
                scheduler.step(auc)

                logger.info(f"Epoch {epoch}, {mode} loss: {total_loss*100:.4f}, AUC: {auc:.4f}, AUPRC: {auprc:.4f}")

            model.eval()
            with torch.no_grad():
                val_pred_loss = 0
                all_valid_labels = torch.tensor([]).to(device)
                all_valid_outputs = torch.tensor([]).to(device)
                for i in range(0, len(val_data), args.batch_size):
                    batch = val_data[i:i + args.batch_size]
                    arr, mask, time, label = prepare_batch(batch, device)
                    out, loss_m, _ = model(arr, mask, time)
                    cls_loss = nn.BCEWithLogitsLoss()(out, label) * args.lambda_cls
                    loss = cls_loss
                    out = torch.sigmoid(out)
                    val_pred_loss += loss.item()
                    all_valid_labels = torch.cat([all_valid_labels, label], 0)
                    all_valid_outputs = torch.cat([all_valid_outputs, out], 0)

                val_pred_loss /= len(val_data)
                auc = roc_auc_score(all_valid_labels.cpu().detach().numpy(), all_valid_outputs.cpu().detach().numpy())
                auprc = average_precision_score(all_valid_labels.cpu().detach().numpy(), all_valid_outputs.cpu().detach().numpy())
                logger.info(f"Epoch {epoch}, Valid loss: {val_pred_loss*100:.4f}, AUC: {auc:.4f}, AUPRC: {auprc:.4f}")

            # Save the best model based on validation loss
            if val_pred_loss < best_val_loss:
                best_val_loss = val_pred_loss
                best_val_epoch = epoch
                best_stage_epoch = epoch
                torch.save(model.state_dict(), f'{model_dir}{args.model}_{rand_int}_best_model.pth')

            # Switch training stage if validation loss does not improve
            if epoch - best_val_epoch > args.patience:
                best_stage_epoch = epoch
                mode = 'pred' if mode == 'rec' else 'rec'
                for p in pred_params: p.requires_grad = mode == 'pred'
                for p in rec_params: p.requires_grad = mode == 'rec'
                model.load_state_dict(torch.load(f'{model_dir}{args.model}_{rand_int}_best_model.pth'))
                logger.info(f'Switching to {mode} mode.')

            # Apply early stopping to avoid overfitting
            if epoch > args.warm_up and epoch - best_val_epoch > args.early_stopping:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        model.load_state_dict(torch.load(f'{model_dir}{args.model}_{rand_int}_best_model.pth'))
        model.eval()
        test_auc, test_auprc, _ = evaluate(model, test_data, args, nn.BCEWithLogitsLoss())
        logger.info(f'Run {run}, Test AUC: {test_auc:.4f}, AUPRC: {test_auprc:.4f}')
        auc_results.append(test_auc)
        auprc_results.append(test_auprc)

    logger.info(f'Average AUC: {np.mean(auc_results):.4f}, AUPRC: {np.mean(auprc_results):.4f}')
    logger.info(f'STD AUC: {np.std(auc_results):.4f}, AUPRC: {np.std(auprc_results):.4f}')
