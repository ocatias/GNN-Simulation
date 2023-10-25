import argparse
import os
import shutil
import json

import numpy as np

from Misc.config import DATA_PATH, RESULTS_PATH

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('transformation', type=str)
    parser.add_argument('--repeats', type=int, default=10,
                    help='How often to repeat the measurement')
    args = parser.parse_args()
    
    data_dir = os.path.join(DATA_PATH, "bench_transforms", args.transformation, "ZINC")
    output_path = os.path.join(RESULTS_PATH, "bench_transforms")
    
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, f"{args.transformation}.yaml")

    runtimes = []
    for repeat in range(args.repeats):
        # Ensure that we do not load the data
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir)
            os.mkdir(data_dir)
        
        if args.transformation != "CWN":
            from Misc.BenchTransforms.zinc import ZINC
            from GrapTransformations.cell_encoding import CellularRingEncoding
            from GrapTransformations.subgraph_bag_encoding import policy2transform, SubgraphBagEncodingNoSeparator

            if args.transformation == "SBE_DSS":
                policy = policy2transform("ego_nets", 3)
                transform = SubgraphBagEncodingNoSeparator(policy, 
                    explicit_type_enc= True, dss_message_passing = True, 
                    connect_accross_subg = False)
            elif args.transformation == "SBE_DS":
                policy = policy2transform("ego_nets", 3)
                transform = SubgraphBagEncodingNoSeparator(policy, 
                    explicit_type_enc= True, dss_message_passing = True, 
                    connect_accross_subg = False)
            elif args.transformation == "DSS":
                transform = policy2transform("ego_nets", 3)
            elif args.transformation == "CE":
                transform = CellularRingEncoding(6, aggr_edge_atr=True, aggr_vertex_feat=True,
                explicit_pattern_enc=True, edge_attr_in_vertices=False) 
            else:
                raise ValueError("Unknown transformation")
                
            datasets = [ZINC(root=data_dir, subset=True, split=split, pre_transform=transform) for split in ["train", "val", "test"]]
            runtime_vector = [datasets[0].prep_time, datasets[1].prep_time, datasets[2].prep_time]
            print(runtime_vector)
            runtime = sum(runtime_vector)
            print(f"\n\nRUNTIME: {runtime}\n\n")
            runtimes.append(runtime)
        else:
            from Misc.BenchTransforms.zinc_cwn import ZINCCWN
            dataset = ZINCCWN(data_dir, subset=True, max_ring_size=6, use_edge_features=True, n_jobs=1)
            print(f"\n\nRUNTIME: {dataset.prep_time}\n\n")
            runtimes.append(dataset.prep_time)
        
    avg = np.average(runtimes)
    std = np.std(runtimes)
    
    print(f"\n\nFINAL RESULT: {avg} ± {std}")
    output_dict = {"runtimes": runtimes, "avg": avg, "std": std}
    
    with open(output_path, "w") as file:
        json.dump(output_dict, file, indent=4)
    
if __name__ == "__main__":
    main()
    