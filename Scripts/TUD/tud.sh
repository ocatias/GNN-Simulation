# First param should be TU dataset name e.g. MUTAG


declare -a configs=("gin_cre.yaml" "gin_sbe_ds.yaml" "gin_sbe_dss.yaml" "gcn_cre.yaml" "gcn_sbe_ds.yaml" "gcn_sbe_dss.yaml" "gcn.yaml" "gin.yaml")

for config in "${configs[@]}"
    do
        echo "$ds"
        python Exp/run_experiment_TUD.py -grid "Configs/TUD/${config}" -dataset "$1" --candidates 20  --folds 10
    done
   
