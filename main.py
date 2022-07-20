import argparse
import os
import random
import warnings

import numpy as np
import torch.optim
import yaml
from ignite.contrib import metrics
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve

import fastflow
import dataset
import constants as const
from loss import FastFlowLoss
from save_plot import plot_fig

warnings.filterwarnings("ignore", category=UserWarning)
use_cuda = torch.cuda.is_available()
seed = const.SEED


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def gaussian_smooth(x, sigma=4):
    bs = x.shape[0]
    for i in range(0, bs):
        x[i] = gaussian_filter(x[i], sigma=sigma)
    return x


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def get_threshold(gt, score):
    gt_mask = np.asarray(gt)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), score.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    return threshold


def build_trainloader(args, config):
    train_dataset = dataset.MVTecDataset(
        root=args.data_path,
        category=args.category,
        input_size=config["input_size"],
        is_train=True
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )


def build_testloader(args, config):
    test_dataset = dataset.MVTecDataset(
        root=args.data_path,
        category=args.category,
        input_size=config["input_size"],
        is_train=False
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )


def build_model(config):
    model = fastflow.FastFlow(
        backbone=config["backbone"],
        pretrained=config["pretrained"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"]
    )
    print(
        "Model Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    loss_meter = AverageMeter()
    for step, data in enumerate(dataloader):
        data = data.cuda()
        ret = model(data)
        loss = criterion(ret[0], ret[1])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )


def eval_once(model, dataloader, epoch, args):
    model.eval()
    auroc_metric = metrics.ROC_AUC()
    test_imgs, gt_mask_list, heatmaps = list(), list(), None
    for data, target in dataloader:
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            outputs = model(data)
        outputs = outputs.cpu().detach()
        outputs_ = outputs.flatten()
        targets = target.flatten().type(torch.int32)
        auroc_metric.update((outputs_, targets))

        if args.eval:
            anomaly_map = outputs.clone()
            test_imgs.extend(data.cpu().detach().numpy())
            gt_mask_list.extend(target.cpu().detach().numpy())
            heatmap = torch.mean(anomaly_map, dim=1)
            heatmaps = torch.cat((heatmaps, heatmap), dim=0) if heatmaps is not None else heatmap
    if args.eval:
        heatmaps = heatmaps.numpy()
        heatmaps = gaussian_smooth(heatmaps, sigma=4)

        scores = rescale(heatmaps)
        scores = scores

        gt_mask = np.asarray(gt_mask_list).astype(np.int32)
        threshold = get_threshold(gt_mask, scores)

        class_name = args.category
        save_dir = 'visualizations/{}_{}'.format(class_name, epoch)
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)

    auroc = auroc_metric.compute()
    print("AUROC: {}".format(auroc))


def train(args):
    # temp_checkpoint_dir = const.CHECKPOINT_DIR.format(args.category)
    os.makedirs(const.CHECKPOINT_DIR.format(args.category), exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR.format(args.category),
        "exp%d" % len(os.listdir(const.CHECKPOINT_DIR.format(args.category)))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    criterion = FastFlowLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_dataloader = build_trainloader(args, config)
    test_dataloader = build_testloader(args, config)
    model.cuda()

    for epoch in range(args.epochs):
        train_one_epoch(model, train_dataloader, criterion, optimizer, epoch)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_once(model, test_dataloader, epoch, args)
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(checkpoint_dir, "%s_%d.pt" % (args.category, epoch + 1))
            )


def evaluate(args):
    epoch = int(args.checkpoint.split('.')[0])
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_testloader(args, config)
    model.cuda()
    eval_once(model, test_dataloader, epoch, args)


def parse_args():
    parser = argparse.ArgumentParser(description='FastFlow Arguments')
    parser.add_argument('-cfg', '--config', type=str, required=True, help='path to config file')
    parser.add_argument('--eval', action="store_true", default=False, help="run evaluation only")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,  help='learning rate for optimizer')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs for training')

    parser.add_argument('--data_path', type=str, required=True, help='path to mvtec dataset')
    parser.add_argument('-cat', '--category', type=str, required=True, help='category name in mvtec')
    parser.add_argument('-ckpt', '--checkpoint', type=str, help='path to model checkpoint')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)

    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
