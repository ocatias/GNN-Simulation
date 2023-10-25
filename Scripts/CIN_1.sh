
declare -a arr=("MOLTOX21" "MOLESOL" "MOLHIV" )

# ZINC
python Exp/run_experiment.py -grid Configs/ZINC/CIN.yaml -dataset "ZINC" --candidates 40  --repeats 10 --mode single --cwn
