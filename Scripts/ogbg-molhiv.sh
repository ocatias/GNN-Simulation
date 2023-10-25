python Exp/run_experiment.py -grid Configs/ogb/gin.yaml -dataset ogbg-molhiv --candidates 40  --repeats 10 --mode single
python Exp/run_experiment.py -grid Configs/ogb/cre.yaml -dataset ogbg-molhiv --candidates 40  --repeats 10 --mode single
python Exp/run_experiment.py -grid Configs/ogb/sbe_dss.yaml -dataset ogbg-molhiv --candidates 40  --repeats 10 --mode single
python Exp/run_experiment.py -grid Configs/ogb/dss.yaml -dataset ogbg-molhiv --candidates 40  --repeats 10 --mode single
python Exp/run_experiment.py -grid Configs/ogb/sbe_ds.yaml -dataset ogbg-molhiv --candidates 40  --repeats 10 --mode single
python Exp/run_experiment.py -grid Configs/ogb/ds.yaml -dataset ogbg-molhiv --candidates 40  --repeats 10 --mode single