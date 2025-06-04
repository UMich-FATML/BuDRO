This folder contains the code for the experiments on the German credit data set.

An example of how to run BuDRO is in the notebook `01_german_run.ipynb`.

`german_proc.py` contains the framework needed to import and pre-process the
data.

We hand-tuned hyperparameters for BuDRO on German.  See the file
`german-select-test-params.txt` for some of the hyperparameter choices that we
found to yield good results.

To produce information about the fairness/accuracy trade-off on the German
credit data set, we include scripts that will allow for us to run many
hyperparameter choices with BuDRO.  We did not collect and analyze all of this
data.  See the file `00_README.txt` in the `../adult` directory for some
insight into how these files (`german_fair_age.py` and `submit_fair.py`) would
work.  (These files do run properly).

To train the baseline and the projecting methods, we did run over a large
grid of hyperparameters.  The file `submit_baseline.py` submits the run for a
single seed, and the `german_baseline.py` file contains the grid of
hyperparameters that we used in our search. We finally choose the set of
hyperparameters that maximize the balanced accuracy over the entire set for
the data that are reported in the main text.  The processing of this data can
be found in `German_baseline_processing.ipynb`.

To train the baseline NN, we used the file `german_nns.py`.


## Note about agebin

German did not have an obvious protected attribute to use for a individual
fairness consistency evaluation.  We originally considering using a binarized
age feature (0 for < 25, 1 for >= 25) to create a consistency evaluation that
was highly correlated with the age feature.  It turned out that the
personal_status feature was quite correlated with age (from examining ridge
regression coefficients), so we ended up only using the status consistency.
Nonetheless, all of the code here has information about `agebin`, whether or
not we should train using a binary age feature.  To reproduce our results, you
always want to set this to False every time you see it.

We collect the data for a baseline NN using the file 
