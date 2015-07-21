import gzip, cPickle
with gzip.open("qri.pkl.gz", "rb") as file:
    datasets = cPickle.load(file)
for dataset in datasets:
    for matrix in dataset:
        print matrix.shape