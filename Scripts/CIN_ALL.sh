
declare -a arr=("MOLTOX21" "MOLESOL" "MOLHIV" "MOLBACE" "MOLCLINTOX" "MOLTOXCAST" "MOLLIPO" "MOLBBBP" "MOLSIDER")

# ZINC
python Exp/run_experiment.py -grid Configs/ZINC/CIN.yaml -dataset "ZINC" --candidates 40  --repeats 10 --mode single --cwn

# ogb
for ds in "${arr[@]}"
do
    echo "$ds"
    python Exp/run_experiment.py -grid Configs/ogb/CIN.yaml -dataset "$ds" --candidates 40  --repeats 10 --mode single --cwn
   done

