import lib
import wandb
import torch


api = wandb.Api()
sweep = api.sweep("username/ff_as_attention/l92zzffq")

to_analyze = ["test_mnist", "train_PermutedMNIST"]

for a_name in to_analyze:
    print(f"Result for {a_name}")
    for layer in range(3):
        vmt_s = lib.StatTracker()
        vmp_s = lib.StatTracker()

        for r in sweep.runs:
            f = r.file(f"export/analysis/vote_predictive_power/{a_name}/layers.{layer}.pth").download(replace=True)
            f = torch.load(f.name)

            vmt = f["vote_matches_target"][1] / f["vote_matches_target"].sum(0)
            vmp = f["vote_matches_pred"][1] / f["vote_matches_pred"].sum(0)

            vmt_s.add(vmt)
            vmp_s.add(vmp)

        vmt_s = vmt_s.get()
        vmp_s = vmp_s.get()

        print(f"\\multirow{{2}}{{*}}{{{layer}}} & Target  & {vmt_s.mean[0]*100:.1f} $\\pm$ {vmt_s.std[0]*100:.1f} & \multirow{{2}}{{*}}{{{vmt_s.mean[1]*100:.1f} $\\pm$ {vmt_s.std[1]*100:.1f}}} \\\\")
        print(f"     & Output  & {vmp_s.mean[0]*100:.1f} $\\pm$ {vmp_s.std[0]*100:.1f} &  \\\\ \midrule")