Here is the python version of our MEAN prototype.
In this repository, all the tests are based on _imdb_ dataset.

Create a new conda env:

```shell
conda create -n MEAN python=3.9
```

install packages:
```shell
conda activate MEAN
pip install -r requirements.txt
```

To make the scripts start quickly, We have cached encoding of training sets, output model, meta-info of imdb,
so there's no need to download the original imdb datasets.

Run script: `./lab/tracing_report.py`.
The console should first print:

```
using model : cached/imdb_bin128/MEAN/1024bs_0.0005lr_ep100_take0.2
model description : MEAN, stable version. Last modified: 2023.4.6
...
```
Waiting for a moment and then console will show the evaluation results on the synthetic, JOB-Light, scale
workloads.

You can run `.lab/training.py` directly to generate a instance. Before training, modifying hyper-parameter is ok.
New instance will be saved in: `lab/cached/imdb_bin128/MEAN/${batch_size}>bs_${learning_rate}lr_ep${epoch_num}_take{ratio_of_training_size}`.



We reused the `FCN + Pooling` and `MSCN` in following repositories:
1. https://github.com/postechdblab/learned-cardinality-estimation.git

