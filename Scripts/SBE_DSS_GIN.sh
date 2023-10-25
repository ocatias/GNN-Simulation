
declare -a arr=("ogbg-molbace" "ogbg-molclintox" "ogbg-moltoxcast" "ogbg-mollipo" "ogbg-molbbbp" "ogbg-molsider" "ogbg-moltox21" "ogbg-molesol" "ogbg-molhiv" )
# declare -a configs=("gcn_cre.yaml" "gcn_sbe_ds.yaml" "gcn_sbe_dss.yaml" "gcn.yaml")
declare -a configs=("sbe_dss.yaml")


# ogb
for ds in "${arr[@]}"
do
    for config in "${configs[@]}"
    do
        echo "$ds"
        python Exp/run_experiment.py -grid "Configs/ogb/${config}" -dataset "$ds" --candidates 40  --repeats 10 --mode single
        done
   done


# ZINC
for config in "${configs[@]}"
    do
    python Exp/run_experiment.py -grid "Configs/ZINC/${config}" -dataset "ZINC" --candidates 40  --repeats 10 --mode single
    done