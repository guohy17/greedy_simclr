import torch
import torch.nn as nn
import os
# import tensorflow as tf


def distribute_over_GPUs(opt, model, num_GPU):
    ## distribute over GPUs
    if opt.device.type != "cpu":
        if num_GPU is None:
            model = nn.DataParallel(model)
            num_GPU = torch.cuda.device_count()
            opt.batch_size_multiGPU = opt.batch_size * num_GPU
        else:
            assert (
                num_GPU <= torch.cuda.device_count()
            ), "You cant use more GPUs than you have."
            model = nn.DataParallel(model, device_ids=list(range(num_GPU)))
            opt.batch_size_multiGPU = opt.batch_size * num_GPU
    else:
        model = nn.DataParallel(model)
        opt.batch_size_multiGPU = opt.batch_size

    model = model.to(opt.device)
    print("Let's use", num_GPU, "GPUs!")

    return model, num_GPU


def reload_weights(opt, model, optimizer, reload_model):
    ## reload weights for training of the linear classifier
    if reload_model:
        print("Loading weights from ", opt.model_path)

        for idx, layer in enumerate(model.module.encoder):
            model.module.encoder[idx].load_state_dict(
                torch.load(
                    os.path.join(
                        opt.model_path,
                        "model_{}_{}.ckpt".format(idx, opt.model_num),
                    ),
                     map_location=opt.device.type,
                )
            )

    ## reload weights and optimizers for continuing training
    elif opt.start_epoch > 0:
        print("Continuing training from epoch ", opt.start_epoch)

        for idx, layer in enumerate(model.module.encoder):
            model.module.encoder[idx].load_state_dict(
                torch.load(
                    os.path.join(
                        opt.model_path,
                        "model_{}_{}.ckpt".format(idx, opt.start_epoch),
                    ),
                    map_location=opt.device.type,
                )
            )

        for i, optim in enumerate(optimizer):
            optim.load_state_dict(
                torch.load(
                    os.path.join(
                        opt.model_path,
                        "optim_{}_{}.ckpt".format(str(i), opt.start_epoch),
                    ),
                    map_location=opt.device.type,
                )
            )
    else:
        print("Randomly initialized model")

    return model, optimizer


def learning_rate_schedule(optimizer):
    """Build learning rate schedule."""
    lr_list = []
    optimizer = optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    for epoch in range(100):
      scheduler.step()
      lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    return lr_list
