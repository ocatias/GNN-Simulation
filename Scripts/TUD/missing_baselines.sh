declare -a configs=("gcn.yaml" "gin.yaml")
declare -a datasets=("NCI1" "NCI109")

for config in "${configs[@]}"
    do
    for dataset in "${datasets[@]}"
        do
            echo "$ds"
            python Exp/run_experiment_TUD.py -grid "Configs/TUD/${config}" -dataset "${dataset}" --candidates 20  --folds 10
        done
    done