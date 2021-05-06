from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.kitti_yolo_dataset import KittiYOLODataset
from eval_mAP import evaluate

from terminaltables import AsciiTable
import os, sys, time, datetime, argparse

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/complex_yolov3.cfg", help="path to model definition file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=cnf.BEV_WIDTH, help="size of each image dimension")
    parser.add_argument("--evaluation_interval", type=int, default=2, help="interval evaluations on validation set")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--file_marker", default='', help="appends to end of saved checkpoint files")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    class_names = load_classes("data/classes.names")

    # Initiate model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = KittiYOLODataset(
        cnf.root_dir,
        split='train',
        mode='TRAIN',
        folder='training',
        data_aug=True,
        multiscale=opt.multiscale_training
    )

    dataloader = DataLoader(
        dataset,
        opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "im",
        "re",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    max_mAP = 0.0
    loss_vals = []
    test_loss_vals = []
    for epoch in range(0, opt.epochs, 1):
        epoch_loss = []
        model.train()
        start_time = time.time()


        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)


            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            epoch_loss.append(loss.item())

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        loss_vals.append(np.sum(epoch_loss)/len(epoch_loss))

        if epoch % opt.evaluation_interval == 0:
        # if True:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class, pred_scores, loss = evaluate(
                model,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
                ("val_loss", loss.mean())
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

            #if epoch % opt.checkpoint_interval == 0:
            if AP.mean() > max_mAP:
                torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_epoch-%d_MAP-%.2f-%s.pth" % (epoch, AP.mean(), opt.file_marker))
                with open(f"checkpoints/yolov3_ckpt_epoch-%d_MAP-%.2f-%s.txt" % (epoch, AP.mean(), opt.file_marker), "w") as file:
                    file.write(str(evaluation_metrics))
                max_mAP = AP.mean()

            print("Prediction Scores: ", pred_scores)
            test_loss_vals.append(loss.mean())

            print(str(loss_vals))
            print(str(test_loss_vals))

            with open(f"checkpoints/yolov3_ckpt_epoch_{opt.file_marker}", "w") as file:
                file.write(str(loss_vals))
                file.write(str(test_loss_vals))

    plt.plot(range(len(loss_vals)), loss_vals, 'g', label="Training loss")
    plt.plot(range(0, opt.epochs, opt.evaluation_interval), test_loss_vals, 'b', label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.savefig('loss curve.png')
