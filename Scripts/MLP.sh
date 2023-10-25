
declare -a arr=("ogbg-molbace" "ogbg-molclintox" "ogbg-moltoxcast" "ogbg-mollipo" "ogbg-molbbbp" "ogbg-molsider" "ogbg-moltox21" "ogbg-molesol" "ogbg-molhiv" )
declare -a configs=("mlp_cre.yaml" "mlp_sbe_ds.yaml" "mlp_sbe_dss.yaml" "mlp.yaml")


# ogb
for ds in "${arr[@]}"
do
    for config in "${configs[@]}"
    do
        echo "$ds"
        python Exp/run_experiment.py -grid "Configs/MLP/ogb/${config}" -dataset "$ds" --candidates 40  --repeats 10 --mode single
        done
   done


# ZINC
for config in "${configs[@]}"
    do
    python Exp/run_experiment.py -grid "Configs/MLP/ZINC/${config}" -dataset "ZINC" --candidates 40  --repeats 10 --mode single
    done