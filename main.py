from typing import Optional
import framework
import tasks
import os
import torch
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = False


def register_args(parser: framework.helpers.ArgumentParser):
    tasks.register_args(parser)
    parser.add_argument("-batch_size", default=128)
    parser.add_argument("-lr", default=1e-3)
    parser.add_argument("-wd", default=0.0)
    parser.add_argument("-test_interval", default=1000)
    parser.add_argument("-state_size", default=128)
    parser.add_argument("-n_layers", default=2)
    parser.add_argument("-stop_after", default="None", parser=parser.int_or_none_parser)
    parser.add_argument("-task", default="tuple")
    parser.add_argument("-dropout", default=0.0)
    parser.add_argument("-grad_clip", default="1.0", parser=parser.float_or_none_parser)
    parser.add_argument("-layer_sizes", default="800,800,256", parser=parser.int_list_parser)
    parser.add_argument("-embedding_size", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-test_batch_size", default="None", parser=parser.int_or_none_parser)
    parser.add_argument("-restore_pretrained", type=str)
    parser.add_argument("-test_pretrained", default=True)
    parser.add_argument("-optimizer", default="adam", choice=["adam", "adamw", "sgd", "adagrad"])
    parser.add_argument("-adam.betas", default="0.9,0.999", parser=parser.float_list_parser)
    parser.add_argument("-adam.eps", default=1e-8)
    parser.add_argument("-amp", default=False)
    parser.add_argument("-tied_embedding", default=False)
    parser.add_argument("-max_length_per_batch", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-length_bucketed_sampling", default=False)
    parser.add_argument("-speedtest", default="none", choice=["none", "iter"])

    parser.add_profile([
        parser.Profile("enwik8_lstm", {
            "task": "enwik8_lstm",
            "test_interval": 5000,
            "grad_clip": 0.3,
            "dropout": 0.2,
            "wd": 1e-7,
            "lr": 1e-3,
            "stop_after": 100000000,
            "n_layers": 6,
            "state_size": 871,
            "embedding_size": 400
        }),
    ])

def initialize(restore: Optional[str] = None):
    helper = framework.helpers.TrainingHelper(wandb_project_name="length_generalization",
                                              register_args=register_args, extra_dirs=["export", "model_weights", "tmp"],
                                              log_async=False, restore=restore)

    task = tasks.get_task(helper.args.task)

    task = task(helper)
    return helper, task

def main():
    helper, task = initialize()

    if helper.args.restore_pretrained:
        pretrained = os.path.expanduser(helper.args.restore_pretrained)
        if not helper.args.restore_pretrained.endswith(".pth"):
            pretrained = os.path.join(pretrained, str(helper.args.sweep_id_for_grid_search), "model.pth")

        assert os.path.isfile(pretrained), f"Failed to load pretrained weights. File {pretrained} not found."

        task.load_weights(pretrained)
        if helper.args.test_pretrained:
            helper.log({f"load_validation/{k}": v for k, v in task.validate().items()})
        print("Done. Skipping training...")
    else:
        task.train()

        print("Training finished. Saving model...")
        task.save_weights()

    task.finish()
    helper.finish()


if __name__ == "__main__":
    main()
