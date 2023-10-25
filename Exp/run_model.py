import random
import time 
import os

import wandb
import torch
import numpy as np

from Exp.parser import parse_args
import Misc.config as config
from Misc.utils import list_of_dictionary_to_dictionary_of_lists
from Exp.utils import load_dataset, get_model, get_optimizer_scheduler, get_loss
from Exp.training_loop_functions import train, eval, step_scheduler


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(args)
    train_loader, val_loader, test_loader = load_dataset(args, config)
    num_classes, num_vertex_features = train_loader.dataset.num_classes, train_loader.dataset.num_node_features
    print(f"Number of features: {num_vertex_features}")
    

    if args.dataset.lower() == "zinc" or "ogb" in args.dataset.lower():
        num_classes = 1
    else:
        print(f"Classes: {train_loader.dataset.num_classes}")

    try:
        num_tasks = train_loader.dataset.num_tasks
        
    except:
        num_tasks = 1
        
    print(f"Tasks: {num_tasks}")

    model = get_model(args, num_classes, num_vertex_features, num_tasks)
    device = args.device
    use_tracking = args.use_tracking
    model.to(device)
    optimizer, scheduler = get_optimizer_scheduler(model, args)
    loss_dict = get_loss(args)

    loss_fct = loss_dict["loss"]
    eval_name = loss_dict["metric"]
    metric_method = loss_dict["metric_method"] 

    if args.use_tracking:
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(
            config = args,
            project = "DWN22")
        
    nr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {nr_parameters}")
    if nr_parameters > args.max_params:
        raise ValueError("Number of model parameters is larger than the allowed maximum")

    time_start = time.time()
    train_results, val_results, test_results = [], [], []
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}")
        train_result = train(model, device, train_loader, optimizer, loss_fct, eval_name, use_tracking, metric_method=metric_method, classes=num_classes)
        val_result = eval(model, device, val_loader, loss_fct, eval_name, metric_method=metric_method, classes=num_classes)
        test_result = eval(model, device, test_loader, loss_fct, eval_name, metric_method=metric_method, classes=num_classes)

        train_results.append(train_result)
        val_results.append(val_result)
        test_results.append(test_result)

        # print(f"\tTRAIN \tLoss: {train_result['total_loss']:10.4f}\t{eval_name}: {train_result[eval_name]:10.4f}")
        print(f"\tTRAIN \tLoss: {train_result['total_loss']:10.4f}")

        print(f"\tVAL \tLoss: {val_result['total_loss']:10.4f}\t{eval_name}: {val_result[eval_name]:10.4f}")
        print(f"\tTEST \tLoss: {test_result['total_loss']:10.4f}\t{eval_name}: {test_result[eval_name]:10.4f}")

        if args.use_tracking:
            wandb.log({
                "Epoch": epoch,
                "Train/Loss": train_result["total_loss"],
                # f"Train/{eval_name}": train_result[eval_name],
                "Val/Loss": val_result["total_loss"],
                f"Val/{eval_name}": val_result[eval_name],
                "Test/Loss": test_result["total_loss"],
                f"Test/{eval_name}": test_result[eval_name],
                "LearningRate": optimizer.param_groups[0]['lr']})

        step_scheduler(scheduler, args, val_result["total_loss"])

        # Exit conditions
        if optimizer.param_groups[0]['lr'] < args.min_lr:
                print("\nLR REACHED MINIMUM: Stopping")
                break

    # Final result
    train_results = list_of_dictionary_to_dictionary_of_lists(train_results)
    val_results = list_of_dictionary_to_dictionary_of_lists(val_results)
    test_result = list_of_dictionary_to_dictionary_of_lists(test_results)

    
    if eval_name in ["mae", "rmse (ogb)"]:
        best_val_epoch = np.argmin(val_results[eval_name])
        mode = "min"
    else:
        best_val_epoch = np.argmax(val_results[eval_name])
        mode = "max"

    val_results["best_epoch"] = best_val_epoch


    loss_train, loss_val, loss_test = train_results['total_loss'][best_val_epoch], val_results['total_loss'][best_val_epoch], test_result['total_loss'][best_val_epoch]
    result_val, result_test = val_results[eval_name][best_val_epoch], test_result[eval_name][best_val_epoch]

    print("\n\nFINAL RESULT")
    runtime = (time.time()-time_start)/3600
    print(f"\tRuntime: {runtime:.2f}h")
    print(f"\tBest epoch {best_val_epoch + 1} / {args.epochs}")
    print(f"\tTRAIN \tLoss: {loss_train:10.4f}")
    print(f"\tVAL \tLoss: {loss_val:10.4f}\t{eval_name}: {result_val:10.4f}")
    print(f"\tTEST \tLoss: {loss_test:10.4f}\t{eval_name}: {result_test:10.4f}")

    if args.use_tracking:
        print("logging")
        wandb.log({
            "Final/Train/Loss": loss_train,
            # f"Final/Train/{eval_name}": train_results[eval_name][best_val_epoch],
            "Final/Val/Loss": loss_val,
            f"Final/Val/{eval_name}": result_val,
            "Final/Test/Loss": loss_test,
            f"Final/Test/{eval_name}": result_test})

        wandb.finish()
        print("end logging")

    return {
        "best_epoch": int(best_val_epoch + 1),
        "epochs": int(epoch),
        "final_val_results": result_val,
        "final_test_result": result_test,
        "final_train_loss": loss_train,
        "final_val_loss": loss_val,
        "final_test_loss": loss_test,
        "parameters": nr_parameters,
        "mode": mode,
        "loss_train": train_results['total_loss'], 
        "loss_val": val_results['total_loss'],
        "loss_test": test_result['total_loss'],
        "best_result_val": val_results[eval_name][best_val_epoch],
        "result_val": val_results[eval_name],
        "result_test": test_result[eval_name],
        "runtime_hours":  runtime,
        }        

def run(passed_args = None):
    args = parse_args(passed_args)
    return main(args)

if __name__ == "__main__":
    run()