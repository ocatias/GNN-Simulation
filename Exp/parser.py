"""
Helper functions that do argument parsing for experiments.
"""

import argparse
import yaml
import sys
from copy import deepcopy

def parse_args(passed_args=None):
    """
    Parse command line arguments. Allows either a config file (via "--config path/to/config.yaml")
    or for all parameters to be set directly.
    A combination of these is NOT allowed.
    Partially from: https://github.com/twitter-research/cwn/blob/main/exp/parser.py
    """

    parser = argparse.ArgumentParser(description='An experiment.')

    # Config file to load
    parser.add_argument('--config', dest='config_file', type=argparse.FileType(mode='r'),
                        help='Path to a config file that should be used for this experiment. '
                        + 'CANNOT be combined with explicit arguments')

    parser.add_argument('--tracking', type=int, default=1,
                        help='If 0 runs without tracking')


    # Parameters to be set directly
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed to set (default: 42)')
    parser.add_argument('--split', type=int, default=0,
                        help='split for cross validation (default: 0)')
    parser.add_argument('--dataset', type=str, default="ZINC",
                            help='dataset name (default: ZINC)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    
    parser.add_argument('--max_params', type=int, default=1e9,
                        help='Maximum number of allowed model paramaters')
    
    parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
    parser.add_argument('--model', type=str, default='GIN',
                    help='model, possible choices: default')
                    
    # LR SCHEDULER
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau',
                    help='learning rate decay scheduler (default: ReduceLROnPlateau)')
    parser.add_argument('--lr_scheduler_decay_rate', type=float, default=0.5,
                        help='strength of lr decay (default: 0.5)')

    # For StepLR
    parser.add_argument('--lr_scheduler_decay_steps', type=int, default=50,
                        help='(StepLR) number of epochs between lr decay (default: 50)')

    # For ReduceLROnPlateau
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='(ReduceLROnPlateau) min LR for `ReduceLROnPlateau` lr decay (default: 1e-5)')
    parser.add_argument('--lr_schedule_patience', type=int, default=10,
                        help='(ReduceLROnPlateau) number of epochs without improvement until the LR will be reduced')

    parser.add_argument('--max_time', type=float, default=12,
                        help='Max time (in hours) for one run')

    parser.add_argument('--drop_out', type=float, default=0.0,
                        help='dropout rate (default: 0.0)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='dimensionality of hidden units in models (default: 64)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of message passing layers (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='number of layers the final MLP of the MPNN has (default: 1)')
    parser.add_argument('--virtual_node', type=int, default=0,
                        help='virtual_node')
    parser.add_argument('--jk', type=str, default='last',
                        help='Jumping knowledge scheme (default: last)')    

    parser.add_argument('--cliques', type=int, default=0,
                        help='Attach cells to all cliques of size up to k.')
    parser.add_argument('--rings', type=int, default=0,
                        help='Attach cells to all rings of size up to k.')
    
    parser.add_argument('--pooling', type=str, default="mean",
                        help='')
    parser.add_argument('--dim_pooling', type=int, default=0,
                        help='(needs cliques or rings)')
    parser.add_argument('--esan', type=int, default=0,
                        help='Use the model from the ESAN paper (needs a policy).')

    parser.add_argument('--policy', type=str, default="",
                        help='Which policy to use.')
    parser.add_argument('--num_hops', type=int, default=0,
                        help='(needs a policy)')
    parser.add_argument('--dss_message_passing', type=int, default=1,
                        help='Setting to 1 will use ds message passing / graph transformation instead of dss (if polify is selected)')
    parser.add_argument('--additonal_gt_con', type=int, default=0,
                        help='Setting to 1 will add the additional edges for the ESAN graph transformation')


    parser.add_argument('--aggr_edge_atr', type=int, default=0,
                        help='(needs rings / cliques to be set)')
    parser.add_argument('--aggr_vertex_feat', type=int, default=0,
                        help='(needs rings / cliques to be set)')
    parser.add_argument('--expl_type_enc', type=int, default=0,
                        help='(needs rings / cliques to be set or a policy and not ESAN)')
    parser.add_argument('--edge_attr_in_vertices', type=int, default=0,
                        help='(needs rings / cliques to be set)')
    parser.add_argument('--max_struct_size', type=int, default=0,
                        help='Maximum size of the structure to attach cells to. If it is non-zero then cycle encoding will be used, except if --cliques 1 then clique encoding will be used')
    parser.add_argument('--node_encoder', type=int, default=1,
                        help="Set to 0 to disable to node encoder")

    # Load partial args instead of command line args (if they are given)
    if passed_args is not None:
        # Transform dict to list of args
        list_args = []
        for key,value in passed_args.items():
            # The case with "" happens if we want to pass an argument that has no parameter
            list_args += [key, str(value)]

        args = parser.parse_args(list_args)
    else:
        args = parser.parse_args()

    args.__dict__["use_tracking"] = args.tracking == 1
    args.__dict__["use_virtual_node"] = args.virtual_node == 1
    args.__dict__["use_node_encoder"] = args.node_encoder == 1

    args.__dict__["use_rings"] = args.rings == 1
    args.__dict__["use_cliques"] = args.cliques == 1
    args.__dict__["use_policy"] = args.policy != ""
    args.__dict__["use_dim_pooling"] = args.dim_pooling == 1
    args.__dict__["use_dss_message_passing"] = args.dss_message_passing == 1
    args.__dict__["use_additonal_gt_con"] = args.additonal_gt_con == 1
    
    args.__dict__["use_esan"] = args.esan == 1

    # Can only use either rings or cliques
    assert not (args.use_rings and args.use_cliques)

    # Cannot combine cliques / rings with a policy
    assert not((args.use_rings or args.use_cliques) and args.use_policy)

    # ESAN needs a policy
    assert not(args.use_esan and args.policy == "")

    args.__dict__["use_aggr_edge_atr"] = args.aggr_edge_atr == 1
    args.__dict__["use_aggr_vertex_feat"] = args.aggr_vertex_feat == 1
    args.__dict__["use_expl_type_enc"] = args.expl_type_enc == 1
    args.__dict__["use_edge_attr_in_vertices"] = args.edge_attr_in_vertices == 1

    # https://codereview.stackexchange.com/a/79015
    # If a config file is provided, write it's values into the arguments
    if args.config_file:
        data = yaml.load(args.config_file)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
                arg_dict[key] = value

    return args
