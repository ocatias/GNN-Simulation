import os
import glob
import yaml
import math
import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str)
path = parser.parse_args().path

models = ["gin", "gcn",  "mlp", "ds", "sbe_ds", "dss", "sbe_dss", "CIN", "cre", "gcn_sbe_ds", "gcn_sbe_dss", "gcn_cre",  "mlp_sbe_ds", "mlp_sbe_dss", "mlp_cre"]
datasets1 = ["ogbg-molbace", "ogbg-molclintox", "ogbg-molbbbp", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-mollipo"]
datasets2 = ["ogbg-molhiv", "ogbg-moltox21", "ogbg-molesol", "ZINC"]

scoring = {"ogbg-molbace": "roc-auc", "ogbg-molclintox": "roc-auc", "ogbg-molbbbp": "roc-auc",
"ogbg-molsider": "roc-auc", "ogbg-moltoxcast": "roc-auc", "ogbg-mollipo": "rmse",
"ogbg-molhiv": "roc-auc", "ogbg-moltox21": "roc-auc", "ogbg-molesol": "rmse", "ZINC": "mae"}

gt_to_mp = {"CRE": "CIN", "DSS": "DSS", " DS": "DS"}

for i, datasets in enumerate([datasets1, datasets2]):
    df_avg = pd.DataFrame(columns=datasets, index=models)
    df_std = pd.DataFrame(columns=datasets, index=models)

    for computer_dir in glob.glob(os.path.join(path, "*")):
        for experiment in glob.glob(os.path.join(computer_dir, "*")):
            file_name = os.path.split(experiment)[-1].split(".")[0]
            dataset = file_name.split("_")[0]
            model = "_".join(file_name.split("_")[1:])

            results_path = os.path.join(experiment, "final.json")

            if f"ogbg-{dataset.lower()}" in datasets:
                dataset = f"ogbg-{dataset.lower()}"

            if dataset == "CSL":
                continue
            if not os.path.isfile(results_path):
                continue
            if dataset not in datasets:
                continue
            if model not in models:
                continue

            with open(results_path) as file:
                results = yaml.safe_load(file)

                if scoring[dataset] == "roc-auc":
                    df_avg[dataset][model] = round(results["result_test-avg"]*100, 1)
                    df_std[dataset][model] = math.ceil(results["result_test-std"]*1000) / 10

                else:
                    df_avg[dataset][model] = round(results["result_test-avg"], 3)
                    df_std[dataset][model] = math.ceil(results["result_test-std"]*1000) /1000


    renaming_index = {"gin": "GIN", "cre":"GIN + CRE", "dss": "DSS", "sbe_dss": "GIN + DSS", "ds": "DS", "sbe_ds": "GIN + DS",
    "gcn": "GCN", "gcn_cre": "GCN + CRE", "gcn_sbe_dss": "GCN + DSS", "gcn_sbe_ds": "GCN + DS", "mlp": "MLP", "mlp_cre": "MLP + CRE",
    "mlp_sbe_dss": "MLP + DSS", "mlp_sbe_ds": "MLP + DS" }

    print(df_avg)
    df_final = pd.DataFrame(columns=datasets, index=models)
    df_final.rename(index = renaming_index, inplace = True)
    df_avg.rename(index = renaming_index, inplace = True)
    df_std.rename(index = renaming_index, inplace = True)
    for dataset in datasets:
        for model in (list(renaming_index.values()) + ["CIN"]):
            better_than_baseline = False

            is_better = lambda curr, baseline: curr > baseline if scoring[dataset] == "roc-auc" else curr < baseline

            if model[0:3] in ["GIN", "MLP", "GCN"]:
                better_than_baseline = is_better(df_avg[dataset][model], df_avg[dataset][model[0:3]])
            elif model in ["DSS", "DS", "CIN"]:
                better_than_baseline = is_better(df_avg[dataset][model], df_avg[dataset]["GIN"])

            if better_than_baseline:
                df_final[dataset][model] = "$\mathbf{" + f"{df_avg[dataset][model]} \pm {df_std[dataset][model]}" + "}$"
            else:
                df_final[dataset][model] = f"${df_avg[dataset][model]} \pm {df_std[dataset][model]}$"

            if model == "CIN":
                print(df_final[dataset][model])

            if len(model) > 3 and model[-3:] in gt_to_mp.keys():
                if is_better(df_avg[dataset][model], df_avg[dataset][gt_to_mp[model[-3:]]]):
                    df_final[dataset][model] = r"\textcolor{red}{" + f"{df_final[dataset][model]}" + "}"

    print(df_final)

    with open(os.path.join(".", f"table_{i}.txt"), "w") as file:
        file.write(df_final.style.to_latex())
