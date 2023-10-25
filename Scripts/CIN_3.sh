
declare -a arr=("MOLLIPO" "MOLBBBP" "MOLSIDER")

# ogb
for ds in "${arr[@]}"
do
    echo "$ds"
    python Exp/run_experiment.py -grid Configs/ogb/CIN.yaml -dataset "$ds" --candidates 40  --repeats 10 --mode single --cwn
   done

