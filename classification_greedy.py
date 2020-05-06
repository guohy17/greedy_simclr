import sys
import torch
import torch.nn as nn
import numpy as np
import time
import os

import get_data_greedy
import arg_parser
import main_greedy
import encoder_greedy
import utils
import logger
from torchvision.utils import save_image
from torchvision.transforms import transforms


class ClassificationModel(torch.nn.Module):
    def __init__(self, in_channels=128, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.layer = nn.Linear(self.in_channels, num_classes, bias=True)

        print(self.layer)

    def forward(self, x, *args):
        x = self.layer(x).squeeze()
        return x

def train_logistic_regression(opt, resnet, classification_model, train_loader):
    total_step = len(train_loader)
    classification_model.train()

    start_time = time.time()

    for epoch in range(opt.num_epochs):
        epoch_acc1 = 0
        epoch_acc5 = 0

        loss_epoch = 0
        for step, (img, target) in enumerate(train_loader):

            classification_model.zero_grad()
            model_input = img.to(opt.device)

            with torch.no_grad():
                # z, h = resnet(model_input)
                z, _, h, _, _ = resnet(model_input, model_input, num_GPU, opt)
                # z = head(h)
            z = z.detach()

            prediction = classification_model(z)


            target = target.to(opt.device)
            loss = criterion(prediction, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1, acc5 = utils.accuracy(prediction.data, target, topk=(1, 5))
            epoch_acc1 += acc1
            epoch_acc5 += acc5
            sample_loss = loss.item()
            loss_epoch += sample_loss


            if step % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc5: {:.4f}, Loss: {:.4f}".format(
                        epoch + 1,
                        opt.num_epochs,
                        step,
                        total_step,
                        time.time() - start_time,
                        acc1,
                        acc5,
                        sample_loss,)
                )
                starttime = time.time()
        print("Overall accuracy for this epoch: ", epoch_acc1 / total_step)
        # acc1, acc5, _ = test_logistic_regression( opt, resnet, classification_model, test_loader)


def test_logistic_regression(opt, resnet, classification_model, test_loader):
    total_step = len(test_loader)
    resnet.eval()
    classification_model.eval()

    starttime = time.time()

    loss_epoch = 0
    epoch_acc1 = 0
    epoch_acc5 = 0

    for step, (img, target) in enumerate(test_loader):

        model_input = img.to(opt.device)

        with torch.no_grad():
            z, _, h, _, _ = resnet(model_input, model_input, num_GPU, opt)

        z = z.detach()

        prediction = classification_model(z)
        target = target.to(opt.device)
        loss = criterion(prediction, target)

        # calculate accuracy
        acc1, acc5 = utils.accuracy(prediction.data, target, topk=(1, 5))
        epoch_acc1 += acc1
        epoch_acc5 += acc5

        sample_loss = loss.item()
        loss_epoch += sample_loss

        if step % 100 == 0:
            print(
                "Step [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc5: {:.4f}, Loss: {:.4f}".format(
                    step, total_step, time.time() - starttime, acc1, acc5, sample_loss
                )
            )
            starttime = time.time()

    print("Testing Accuracy: ", epoch_acc1 / total_step)
    return epoch_acc1 / total_step, epoch_acc5 / total_step, loss_epoch / total_step


if __name__ == "__main__":

    opt = arg_parser.parse_args()

    add_path_var = "linear_model"

    arg_parser.create_log_path(opt, add_path_var=add_path_var)
    opt.training_dataset = "train"

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # load pretrained model
    resnet, _, num_GPU = main_greedy.load_model_and_optimizer(
        opt)
    # resnet = encoder.Resnet_sim(opt)
    # resnet = resnet.to(opt.device)
    # resnet.load_state_dict(torch.load('/lustre/home/hyguo/code/code/SimCLR-cifar/results/128_0.5_200_512_100_model.pth'))
    resnet.load_state_dict(
        torch.load(
            '/lustre/home/hyguo/code/code/SimCLR/models/models_0430/resize80-100-{}.pth'.format(opt.cur_train_module)))
    resnet.eval()

    _, _, train_loader, _, test_loader, _ = get_data_greedy.get_dataloader(opt)
    # _, _, test_loader, _, train_loader, _ = get_data.get_dataloader(opt)


    # classification_model = ClassificationModel(in_channels=128, num_classes=10).to(opt.device)
    classification_model = ClassificationModel(in_channels=2048, num_classes=10).to(opt.device)
    params = classification_model.parameters()
    optimizer = torch.optim.Adam(params)
    criterion = torch.nn.CrossEntropyLoss()
    logs = logger.Logger(opt)

    # Train the model
    train_logistic_regression(opt, resnet, classification_model, train_loader)

    # Test the model

    acc1, acc5, _ = test_logistic_regression(
        opt, resnet, classification_model, test_loader
    )
