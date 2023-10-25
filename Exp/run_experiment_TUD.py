"""
Runs an experiment: searches for hyperparameters and then trains the final model multiple times
For TUD DATASETS
THIS ASSUME WE WANT TO MAXIMIZE THE METRIC

"""

import argparse
import os
import glob
import json
import copy
import sys
from collections import defaultdict

import yaml
from sklearn.model_selection import ParameterGrid
import numpy as np

from Misc.config import RESULTS_PATH

keys_to_avg = ["runtime_hours", "epochs", "parameters"] 

def get_directory(args):
    return os.path.join(RESULTS_PATH, f"{args.dataset}_{os.path.split(args.grid_file)[-1]}") 

def get_paths(args):
    directory = get_directory(args)

    if args.folds > 1:
        results_path = os.path.join(directory) 
    else:
        results_path = directory
    hyperparams_path = os.path.join(results_path, "Hyperparameters")
    final_eval_path = os.path.join(results_path, "FinalResults")
    errors_path = os.path.join(results_path, "Errors")
    return directory, results_path, hyperparams_path, final_eval_path, errors_path
    
def dict_to_args(args_dict):
    # Transform dict to list of args
    list_args = []
    for key,value in args_dict.items():
        # The case with "" happens if we want to pass an argument that has no parameter
        if value != "":
            list_args += [key, str(value)]
        else:
            list_args += [key]

    print(list_args)
    return list_args

binary_class_ogb_datasets = ["molbace", "molbbbp", "molclintox", "molmuv", "molpcba", "molsider", "moltox21", "moltoxcast", "molhiv", "molchembl"]
binary_class_datsets = binary_class_ogb_datasets
regression_ogb_datasets = ["molesol", "molfreesolv", "mollipo"]
regression_datsets = regression_ogb_datasets + ["zinc"]
ogb_datasets = binary_class_ogb_datasets + regression_ogb_datasets

def extra_info_for_cwn(dataset, args_dict):
    # Set evalulation metric, for example this would correspond to
    # eval_metric: ["ogbg-molhiv"] in the yaml file
    if dataset.lower() in ogb_datasets:
        args_dict["--eval_metric"] = f"ogbg-{dataset.lower()}"
    elif dataset.lower() == "zinc":
        args_dict["--eval_metric"] = "mae"
    else:
        raise ValueError("Unknown eval metric for this dataset")
    
    # Set task type metric, for example this would correspond to
    # eval_metric: ["ogbg-molhiv"] in the yaml file
    if dataset.lower() in binary_class_datsets:
        args_dict["--task_type"] = "bin_classification"
    elif dataset.lower() in regression_datsets:
        args_dict["--task_type"] = "regression"
        args_dict["--minimize"] = ""
    else:
        raise ValueError("Unknown task type for this dataset")
    return args_dict
    

def main():
    parser = argparse.ArgumentParser(description='An experiment.')
    parser.add_argument('-grid', dest='grid_file', type=str,
                    help="Path to a .yaml file that contains the parameters grid.")
    parser.add_argument('-dataset', type=str)
    parser.add_argument('--candidates', type=int, default=20,
                    help="Number of parameter combinations to try per fold.")
    parser.add_argument('--folds', type=int, default="1",
                    help='Number of folds, setting this to something other than 1, means we will treat this as cross validation')

    args = parser.parse_args()
    
    # if args.use_cwn:
    #     from cwn.exp.run_exp import run as run_with_args
    #     run = lambda args_dict: run_with_args(dict_to_args(extra_info_for_cwn(args.dataset, args_dict)))
    # else:
    from Exp.run_model import run 

    with open(args.grid_file, 'r') as file:
        grid_raw = yaml.safe_load(file)
    grid = ParameterGrid(grid_raw)
    
    directory = get_directory(args)
    directory, results_path, hyperparams_path, final_eval_path, errors_path = get_paths(args)
    
    if not os.path.isdir(directory):
        os.mkdir(directory)
        if not os.path.isdir(results_path):
            os.mkdir(results_path)
        os.mkdir(hyperparams_path)
        os.mkdir(final_eval_path)
        os.mkdir(errors_path)

    prev_params = []
    
    # Count how many parameters we have successfully tried
    previously_tested_params = glob.glob(os.path.join(hyperparams_path, "*.json"))
    # nr_succesfully_tested_params = 0
    # for param_file in previously_tested_params
    #     with open(param_file) as file:
    #         if "result_val-avg" in yaml.safe_load(file):
    #             nr_succesfully_tested_params += 1
            
    nr_tries = min(args.candidates, len(grid)) - len(previously_tested_params)
    
    for c in range(nr_tries):
        # Set seed randomly, because model training sets seeds and we want different parameters every time
        np.random.seed()
        if len(prev_params) == len(grid):
            break

        param = None
        already_checked_params = False
        stored_params = []
        for param_file in glob.glob(os.path.join(hyperparams_path, "*.json")):
            with open(param_file) as file:
                stored_params.append(yaml.safe_load(file))
                    
        # Check if we have tested those params before
        # This is bad code, FIX THIS
        while param is None or (param in prev_params) or already_checked_params:
            param = np.random.choice(grid, 1)
            already_checked_params = False
            for curr_param_to_check in stored_params:
                same_param = True
                for key, value in param[0].items():
                    if str(curr_param_to_check["params"][f"--{str(key)}"]) != str(value):
                        same_param = False
                        break                           
                
                if same_param:
                    # if "result_val-avg" in curr_param_to_check:
                    salready_checked_params = True
                    # else:
                        # If we already have a hyperparams file for this parameter but no results then try this param
                        # already_checked_params = False
                    break

        prev_params.append(param)
        param_dict = {
            "--dataset": args.dataset
                    }
        for key, value in param[0].items():
            param_dict["--" + str(key)] = str(value)
            
        hyperparams_output_path = os.path.join(hyperparams_path, f"params_{len(glob.glob(os.path.join(hyperparams_path, '*')))}.json") 
        # storage_dict = {"params": param_dict}     
        # with open(hyperparams_output_path, "w") as file:
        #     json.dump(storage_dict, file, indent=4)
                
        validation_performance, runtimes = [], []
        parameters = 0
        for split in range(args.folds):
            print(f"Grid contains {len(grid)} combinations")
            param_dict["--split"] = split

            try:
                # Don't search for params if there is only one candidate
                if len(grid) == 1 or args.candidates == 1:
                    break
                
                result_dict = run(param_dict) 
                validation_performance.append(result_dict["result_val"])
                runtimes.append(result_dict["runtime_hours"])
                parameters = result_dict["parameters"]

                if len(prev_params) >= len(grid):
                    break
            except Exception as e:
                # Detected an error: store the params and the error
                error_dict = {"params": param_dict, "error": str(e)}
                output_path = os.path.join(errors_path, f"params_{len(glob.glob(os.path.join(errors_path, '*')))}.json")
                with open(output_path, "w") as file:
                    json.dump(error_dict, file, indent=4)
                    
        print(validation_performance)
        
        # Average over each epoch across folds
        val_accs_per_epoch = [[validation_performance[fold][epoch] for fold in range(args.folds)] for epoch in range(int(param_dict["--epochs"]))]
        avg_val_acc = [np.mean(val_accs) for val_accs in val_accs_per_epoch]
        best_val_acc_epoch = np.argmax(avg_val_acc)

        # Get results i.e. the average accuracy of the best epoch and the std of that epoch
        accuracy = avg_val_acc[best_val_acc_epoch]
        std = np.std(val_accs_per_epoch[best_val_acc_epoch])
        
        time_avg, time_std = np.mean(runtimes), np.std(runtimes)
        
        print(val_accs_per_epoch)
        print(best_val_acc_epoch, ": ", accuracy, " +- ", std)
        
        output_dict = {
            "result_val-avg": accuracy,
            "result_val-std": std,
            "runtime-avg": time_avg,
            "runtime-std": time_std,
            "parameters": parameters,
            "params": param_dict
        }
        
        # Overwrite hyperparams file and write as test file
        test_output_path = os.path.join(final_eval_path, f"eval_{len(glob.glob(os.path.join(final_eval_path, '*')))}.json")
        print(test_output_path)
        with open(test_output_path, "w") as file:
            json.dump(output_dict, file, indent=4)
        with open(hyperparams_output_path, "w") as file:
            json.dump(output_dict, file, indent=4) 
     
    print("Finished searching.")
    print("Looking for best results.")
    best_result_val, final_results_dict = None, None
    for param_file in glob.glob(os.path.join(final_eval_path, "*.json")):
        with open(param_file) as file:
            dict = yaml.safe_load(file)
        
            if best_result_val is None or (dict["result_val-avg"] > best_result_val):
                final_results_dict = copy.deepcopy(dict)
                best_result_val = dict["result_val-avg"]
    
    output_path = os.path.join(directory, "final.json")  
    with open(output_path, "w") as file:
        json.dump(final_results_dict, file, indent=4)
    quit()


if __name__ == "__main__":
    main()