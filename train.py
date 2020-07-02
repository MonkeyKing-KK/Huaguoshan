import torch
import torchvision
import torchvision.transforms as transforms
from mobilenetv2 import MobileNetV2
from shufflenetv2 import shufflenetv2
from ghostnet import ghost_net
from utils import show_confMat, plot_line, train_one_epoch, valid_one_epoch
import numpy as np
import argparse
from datetime import datetime
import os

class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Ghostnet Training')
    parser.add_argument('-bs', type=int, default=128)
    parser.add_argument('-max_epoch', type=int, default=190)
    parser.add_argument('-lr', type=float, default=0.1)
    parser.add_argument('-arc', type=str, default="MobileNetV2", help="MobileNetV2, shufflenetv2, ghost_net")
    args = parser.parse_args()

    # data augmentation
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    # download cifar10 and process dataset
    train_dataset = torchvision.datasets.CIFAR10('./data', train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10('./data', train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, num_workers=0, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, num_workers=0, shuffle=False)

    log_dir = os.path.join('./results', time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.arc == "MobileNetV2":
        model = MobileNetV2()
    elif args.arc == "shufflenetv2":
        model = shufflenetv2()
    elif args.arc == "ghost_net":
        model = ghost_net()
    else:
        raise ValueError("{} is not define!".format(args.arc))

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    num_epoch = args.max_epoch
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0

    LR = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=[92, 136])
    for epoch in range(num_epoch):
        loss_train, acc_train, mat_train = train_one_epoch(train_loader, model, criterion, optimizer, epoch, device)
        loss_valid, acc_valid, mat_valid = valid_one_epoch(test_loader, model, criterion, device)
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(
            epoch + 1, num_epoch, acc_train, acc_valid, loss_train, loss_valid, optimizer.param_groups[0]["lr"]))

        if 'patience' in dir(scheduler):
            scheduler.step(acc_valid)  # ReduceLROnPlateau
        else:
            scheduler.step()  # StepLR

        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)

        show_confMat(mat_train, class_names, "train", log_dir, verbose=epoch == num_epoch - 1)
        show_confMat(mat_valid, class_names, "valid", log_dir, verbose=epoch == num_epoch - 1)

        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        if epoch > (num_epoch / 2) and best_acc < acc_valid:
            best_acc = acc_valid
            best_epoch = epoch

            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_acc": best_acc}

            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)

    print(" done ~~~~ {}, best acc: {} in :{}".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'), best_acc, best_epoch))






