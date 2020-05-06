import os
import torch
import numpy as np


class Logger:
    def __init__(self, opt):
        self.opt = opt
        self.num_models_to_keep = 1
        assert self.num_models_to_keep > 0, "Dont delete all models!!!"

    def create_log(
        self,
        model,
        accuracy=None,
        epoch=0,
        optimizer=None,
        final_test=False,
        final_loss=None,
        acc5=None,
        classification_model=None
    ):

        print("Saving model and log-file to " + self.opt.log_path)

        # Save the model checkpoint
        torch.save(
            model.state_dict(),
            os.path.join(self.opt.log_path, "model_{}.ckpt".format(epoch)),
        )

        ### remove old model files to keep dir uncluttered
        if (epoch - self.num_models_to_keep) % 10 != 0:
            try:
                os.remove(
                    os.path.join(
                        self.opt.log_path,
                        "model_{}.ckpt".format(epoch - self.num_models_to_keep),
                    )
                )
            except:
                print("not enough models there yet, nothing to delete")


        if classification_model is not None:
            # Save the predict model checkpoint
            torch.save(
                classification_model.state_dict(),
                os.path.join(self.opt.log_path, "classification_model_{}.ckpt".format(epoch)),
            )

            ### remove old model files to keep dir uncluttered
            try:
                os.remove(
                    os.path.join(
                        self.opt.log_path,
                        "classification_model_{}.ckpt".format(epoch - self.num_models_to_keep),
                    )
                )
            except:
                print("not enough models there yet, nothing to delete")

        if optimizer is not None:
            for idx, optims in enumerate(optimizer):
                torch.save(
                    optims.state_dict(),
                    os.path.join(
                        self.opt.log_path, "optim_{}_{}.ckpt".format(idx, epoch)
                    ),
                )

                try:
                    os.remove(
                        os.path.join(
                            self.opt.log_path,
                            "optim_{}_{}.ckpt".format(
                                idx, epoch - self.num_models_to_keep
                            ),
                        )
                    )
                except:
                    print("not enough models there yet, nothing to delete")

        # Save hyper-parameters
        with open(os.path.join(self.opt.log_path, "log.txt"), "w+") as cur_file:
            cur_file.write(str(self.opt))
            if accuracy is not None:
                cur_file.write("Top 1 -  accuracy: " + str(accuracy))
            if acc5 is not None:
                cur_file.write("Top 5 - Accuracy: " + str(acc5))
            if final_test and accuracy is not None:
                cur_file.write(" Very Final testing accuracy: " + str(accuracy))
            if final_test and acc5 is not None:
                cur_file.write(" Very Final testing top 5 - accuracy: " + str(acc5))

        if accuracy is not None:
            np.save(os.path.join(self.opt.log_path, "accuracy"), accuracy)

        if final_test:
            np.save(os.path.join(self.opt.log_path, "final_accuracy"), accuracy)
            np.save(os.path.join(self.opt.log_path, "final_loss"), final_loss)

