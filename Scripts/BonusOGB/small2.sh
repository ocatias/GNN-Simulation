
## declare an array variable
declare -a arr=("ogbg-molbbbp" "ogbg-molsider" )

## now loop through the above array
for ds in "${arr[@]}"
do
    echo "$ds"
    python Exp/run_experiment.py -grid Configs/ogb/gin.yaml -dataset "$ds" --candidates 40  --repeats 10 --mode single
    python Exp/run_experiment.py -grid Configs/ogb/cre.yaml -dataset "$ds" --candidates 40  --repeats 10 --mode single
    python Exp/run_experiment.py -grid Configs/ogb/sbe_dss.yaml -dataset "$ds" --candidates 40  --repeats 10 --mode single
    python Exp/run_experiment.py -grid Configs/ogb/dss.yaml -dataset "$ds" --candidates 40  --repeats 10 --mode single
    python Exp/run_experiment.py -grid Configs/ogb/sbe_ds.yaml -dataset "$ds" --candidates 40  --repeats 10 --mode single
    python Exp/run_experiment.py -grid Configs/ogb/ds.yaml -dataset "$ds" --candidates 40  --repeats 10 --mode single
done
