import torch
import time
import numpy as np

import logger
import get_data_greedy
import encoder_greedy
import arg_parser
import model_utils


def load_model_and_optimizer(opt, num_GPU=None):
    resnet = encoder_greedy.FullModel(opt)
    optimizer=[]
    for idx, layer in enumerate(resnet.encoder):
        optimizer.append(torch.optim.Adam(layer.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay))
    resnet, num_GPU = model_utils.distribute_over_GPUs(opt, resnet, num_GPU=num_GPU)
    return resnet, optimizer, num_GPU


def train(opt, resnet, num_GPU):
    total_step = len(train_loader)

    starttime = time.time()
    print_idx = 100
    cur_train_module = opt.cur_train_module

    for epoch in range(opt.start_epoch, opt.num_epochs):
        print(
            "Epoch [{}/{}], total Step [{}], Time (s): {:.1f}".format(
                epoch + 1,
                opt.num_epochs,
                total_step,
                time.time() - starttime,
            )
        )
        for step, (img1, img2, target) in enumerate(train_loader):
            x_t1 = img1.to(opt.device)
            x_t2 = img2.to(opt.device)

            _, _, _, _, loss = resnet(x_t1, x_t2, num_GPU, opt, n=cur_train_module)
            loss = torch.mean(loss, 0)
            if cur_train_module != 4:
                loss = loss[cur_train_module].unsqueeze(0)
            for idx, cur_loss in enumerate(loss):
                if len(loss) == 1:
                    idx = cur_train_module
                resnet.zero_grad()
                if idx == 3:
                    cur_loss.backward()
                else:
                    cur_loss.backward(retain_graph=True)
                optimizer_r[idx].step()

                print_loss = cur_loss.item()
        print("\t \t Loss {}: \t \t {:.4f}".format(idx, print_loss))
        # if (epoch + 1) % 10 == 0:
        torch.save(resnet.state_dict(),
                    '/lustre/home/hyguo/code/code/SimCLR/models/models_0506/model16-{}-{}.pth'.format(epoch+1, cur_train_module))


if __name__ == "__main__":

    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt)
    opt.training_dataset = "unlabeled"

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    torch.backends.cudnn.benchmark = True

    # load model
    resnet, optimizer_r, num_GPU = load_model_and_optimizer(opt)
    if opt.cur_train_module > 0 and opt.cur_train_module < 4:
        resnet.load_state_dict(torch.load(
            '/lustre/home/hyguo/code/code/SimCLR/models/models_0506/model16-100-{}.pth'.format(opt.cur_train_module - 1)))
    logs = logger.Logger(opt)
    train_loader, _, supervised_loader, _, test_loader, _ = get_data_greedy.get_dataloader(opt)
    train(opt, resnet, num_GPU)

