from optparse import OptionParser
import time
import os
import torch
import numpy as np


def parse_args():
    # load parameters and options
    parser = OptionParser()
    parser.add_option(
        "--K", type="int", default=65536
    )
    parser.add_option(
        "--m", type="float", default=0.999
    )
    parser.add_option(
        "--T", type="float", default=0.07
    )
    parser.add_option(
        "--tem", type="float", default=0.5
    )
    parser.add_option(
        "--feature_dim", type="int", default=128
    )
    parser.add_option(
        "--cur_train_module", type="int", default=0
    )
    parser.add_option(
        "--num_epochs", type="int", default=100, help="Number of Epochs for Training"
    )
    parser.add_option(
        "--batch_size", type="int", default=512, help="Batchsize"
    )
    parser.add_option(
        "--learning_rate", type="float", default=0.0003, help="Learning rate"
    )
    parser.add_option(
        "--weight_decay", type="float", default=1e-6, help="Learning rate"
    )
    parser.add_option(
        "--dataset", type="string", default="stl10"
    )
    parser.add_option("--reload_model", action="store_true", default=False)
    parser.add_option(
        "--cos",
        action="store_true",
        default=False,
    )
    parser.add_option(
        "--seed", type="int", default=2, help="Random seed for training"
    )
    parser.add_option(
        "-i", "--data_input_dir", type="string", default="/lustre/home/hyguo/code/code/vision/datasets/"
    )
    parser.add_option(
        "-o", "--data_output_dir", type="string", default="."
    )

    parser.add_option(
        "--start_epoch",
        type="int",
        default=0,
        help="Epoch to start training from: "
    )
    parser.add_option(
        "--model_path",
        type="string",
        default="./models_0329",
        help="Directory of the saved model (path within --data_input_dir)",
    )
    parser.add_option(
        "--model_num",
        type="string",
        default="100",
        help="Number of the saved model to be used for training the linear classifier"
        "(loaded using model_path + model_X.ckpt, where X is the model_num passed here)",
    )
    parser.add_option(
        "--save_dir",
        type="string",
        default="",
        help="If given, uses this string to create directory to save results in "
        "(be careful, this can overwrite previous results); "
        "otherwise saves logs according to time-stamp",
    )
    parser.add_option(
        "--download_dataset",
        action="store_true",
        default=True,
        help="Boolean to decide whether to download the dataset to train on (only tested for STL-10)",
    )

    (opt, _) = parser.parse_args()

    opt.time = time.ctime()

    # Device configuration
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opt.experiment = "vision"

    return opt


def create_log_path(opt, add_path_var=""):
    unique_path = False

    if opt.save_dir != "":
        opt.log_path = os.path.join(opt.data_output_dir, "logs", opt.save_dir)
        unique_path = True
    elif add_path_var == "features" or add_path_var == "images":
        opt.log_path = os.path.join(opt.data_output_dir, "logs", add_path_var, os.path.basename(opt.model_path))
        unique_path = True
    else:
        opt.log_path = os.path.join(opt.data_output_dir, "logs", add_path_var, opt.time)

    # hacky way to avoid overwriting results of experiments when they start at exactly the same time
    while os.path.exists(opt.log_path) and not unique_path:
        opt.log_path += "_" + str(np.random.randint(100))

    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path)

