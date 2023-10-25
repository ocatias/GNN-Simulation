"""
Runs an experiment: searches for hyperparameters and then trains the final model multiple times
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

def get_paths(args, split):
    directory = get_directory(args)

    if args.folds > 1:
        results_path = os.path.join(directory, f"split_{split}") 
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
    parser.add_argument('--repeats', type=int, default=10,
                    help="Number of times to repeat the final model training")
    parser.add_argument('--mode', type=str, default="single",
                    help='single = average(best validation epochs), multi = best (average validation epoch)')
    parser.add_argument('--folds', type=int, default="1",
                    help='Number of folds, setting this to something other than 1, means we will treat this as cross validation')
    parser.add_argument('--cwn', dest='use_cwn', action='store_true')

    args = parser.parse_args()
    
    if args.use_cwn:
        from cwn.exp.run_exp import run as run_with_args
        run = lambda args_dict: run_with_args(dict_to_args(extra_info_for_cwn(args.dataset, args_dict)))
    else:
        from Exp.run_model import run 

    with open(args.grid_file, 'r') as file:
        grid_raw = yaml.safe_load(file)
    grid = ParameterGrid(grid_raw)
    
    directory = get_directory(args)
    if not os.path.isdir(directory):
        os.mkdir(directory)

        for split in range(args.folds):
            directory, results_path, hyperparams_path, final_eval_path, errors_path = get_paths(args, split)
            
            if not os.path.isdir(results_path):
                os.mkdir(results_path)
            os.mkdir(hyperparams_path)
            os.mkdir(final_eval_path)
            os.mkdir(errors_path)

    for split in range(args.folds):
        print(f"Grid contains {len(grid)} combinations")
        directory, results_path, hyperparams_path, final_eval_path, errors_path = get_paths(args, split)

        print("Searching for hyperparameters")
        prev_params = []
        mode = None

        previously_tested_params = glob.glob(os.path.join(hyperparams_path, "*.json"))
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
                        already_checked_params = True
                        print(f"Already tried {param} and got validation score of {curr_param_to_check['final_val_results']}")
                        break

            prev_params.append(param)
            param_dict = {
                "--dataset": args.dataset
                        }
            for key, value in param[0].items():
                param_dict["--" + str(key)] = str(value)
                
            if args.folds > 1:
                param_dict["--split"] = split

            try:
                # Don't search for params if there is only one candidate
                if len(grid) == 1 or args.candidates == 1:
                    break
                
                result_dict = run(param_dict)
                output_path = os.path.join(hyperparams_path, f"params_{len(glob.glob(os.path.join(hyperparams_path, '*')))}.json")
                storage_dict = {"params": param_dict}     
                storage_dict.update(copy.deepcopy(result_dict))
                    
                if mode is None:
                    mode = result_dict["mode"]
                with open(output_path, "w") as file:
                    json.dump(storage_dict, file, indent=4)
                
                print(output_path)

                if len(prev_params) >= len(grid):
                    break
            except Exception as e:
                # Detected an error: store the params and the error
                error_dict = {"params": param_dict, "error": str(e)}
                output_path = os.path.join(errors_path, f"params_{len(glob.glob(os.path.join(errors_path, '*')))}.json")
                with open(output_path, "w") as file:
                    json.dump(error_dict, file, indent=4)
                    
                # We don't count a try if it fails
                c -= 1

        print("Finished search")
        # Select best parameters
        best_params, best_result_val = None, None
        
        # Don't load parameters if we have not searched for them
        if len(grid) == 1 or args.candidates == 1:
            best_params = param_dict
            print(f"Only one parameter pair / candidate (not seraching for params)")
            
        # Otw load them
        else:
            for param_file in glob.glob(os.path.join(hyperparams_path, "*.json")):
                with open(param_file) as file:
                    dict = yaml.safe_load(file)
                    
                    # If we have not run any models yet we need to load it 
                    if mode is None:
                        mode = dict["mode"]

                    if best_result_val is None or \
                            (mode == "min" and min(dict["result_val"]) < best_result_val) or \
                            (mode == "max" and max(dict["result_val"]) > best_result_val):
                        best_params = dict["params"]
                        
                        if mode == "min":
                            best_result_val = min(dict["result_val"])
                        else:
                            best_result_val = max(dict["result_val"])

            print(f"Best params have score {best_result_val:.4f} and are:\n{best_params}")

        print("Evaluating the final params")
        prev_evaluation = glob.glob(os.path.join(final_eval_path, "*.json"))
        nr_tries = args.repeats - len(prev_evaluation)
        for iteration in range(nr_tries):
            print(iteration)
            best_params["--seed"] = iteration
            result_dict = run(best_params)

            output_dict = {}
            output_dict["params"] = best_params
            output_dict.update(copy.deepcopy(result_dict))

            output_path = os.path.join(final_eval_path, f"eval_{len(glob.glob(os.path.join(final_eval_path, '*')))}.json")
            print(output_path)
            with open(output_path, "w") as file:
                json.dump(output_dict, file, indent=4)
        
        
        
    # Compute final results (if crossvalidation: over all folds)
    final_results = defaultdict(lambda: [])

    for split in range(args.folds):
        directory, results_path, hyperparams_path, final_eval_path, errors_path = get_paths(args, split)
        eval_paths = list(glob.glob(os.path.join(final_eval_path, '*.json')))
    
        # Collect results
        for eval_path in eval_paths:
            with open(eval_path) as file:
                result_dict = json.load(file)
                
                # If we have not run any models yet we need to load it 
                if mode is None:
                    mode = result_dict["mode"]
                        
            for key in result_dict.keys():
                final_results[key].append(result_dict[key])

    output = {}
    # Average relevant keys
    for key in keys_to_avg:
        avg, std = np.average(final_results[key]), np.std(final_results[key])
        output[f"{key}-avg"] = avg
        output[f"{key}-std"] = std
        print(f"{key}:\t{avg:.4f}±{std:4f}")
        
    # Compute final scores
    if args.mode == "single":
        val, test = [], []
        for result_test, result_val in zip(final_results["result_test"], final_results["result_val"]):
            if mode == "min":
                best_val_epoch = np.argmin(result_val)
            else:
                best_val_epoch = np.argmax(result_val)
                
            val.append(result_val[best_val_epoch])
            test.append(result_test[best_val_epoch])
        
        output[f"result_val-avg"] = np.average(val)
        output[f"result_val-std"] = np.std(val)
        output[f"result_test-avg"] = np.average(test)
        output[f"result_test-std"] = np.std(test)
    else:
        avg_val, std_val, avg_test, std_test = [], [], [], []
        for i in range(min([len(result_test) for result_test in final_results["result_test"]])):
            val, test = [], []
            for j in range(len(final_results["result_val"])):
                val.append(final_results["result_val"][j][i])
                test.append(final_results["result_test"][j][i])

            avg_val.append(np.average(val))
            std_val.append(np.std(val))
            avg_test.append(np.average(test))
            std_test.append(np.std(test))
            
        if mode == "min":
            best_val_epoch = np.argmin(avg_val)
        else:
            best_val_epoch = np.argmax(avg_val)

        output[f"result_val-avg"] = avg_val[best_val_epoch]
        output[f"result_val-std"] = std_val[best_val_epoch]
        output[f"result_test-avg"] = avg_test[best_val_epoch]
        output[f"result_test-std"] = std_test[best_val_epoch]

    output["params"] = best_params

    output_path = os.path.join(directory, "final.json")  
    with open(output_path, "w") as file:
        json.dump(output, file, indent=4)

    print(output)

if __name__ == "__main__":
    main()