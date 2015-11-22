from glob import glob
from multiprocessing import Pool, Process
from random import choice
from time import sleep
import numpy as np
import os
from data import load_raw_data, get_grouped_dataset
import pandas as pd
from scipy.stats import gaussian_kde
from model import Model
import pymongo

c = pymongo.MongoClient()
save_collection = c.votaciones.predictions

here = os.path.dirname(__file__)


def load_data(level):
    fnames = glob(os.path.join(here, '*.csv'))
    print "loading raw data.."
    raw_data = load_raw_data(fnames)
    gdf = get_grouped_dataset(raw_data, level=level)

    return gdf[(~pd.isnull(gdf)).all(1)]  # there is a row with nulls


def get_model_to_draw(level):
    gdf = load_data(level)

    dfX = gdf['132_pct 133_pct 137_pct 138_pct'.split()]
    dfy = gdf['131_pct 135_pct'.split()]

    model = Model().fit(dfX, dfy)
    return model, dfX

def do_estimate(level):
    gdf = load_data(level)

    dfX = gdf['132_pct 133_pct 137_pct 138_pct'.split()]
    df_cnt = gdf['132 133 137 138'.split()]
    dfy = gdf['131_pct 135_pct'.split()]

    t = {
        2: 10,
        3: 10,
        4: 0.1
    }
    mask = np.random.random_sample(len(gdf)) < t[level]
    model = Model(
        save_collection=save_collection,
        metadata_to_save={'level': level, 't': t[level]}
    ).fit(dfX[mask], dfy[mask])

    return model.predict(dfX, df_cnt)


class Worker(Process):
    def run(self):
        while True:
            level = choice(range(2,5))
            do_estimate(level)


def main():
    try:
        p = Pool(4)
        p.map(load_data, range(2,5)) # to build caches

        ws = []
        for _ in xrange(4):
            worker = Worker()
            worker.start()
            ws.append(worker)

        for w in ws: w.join()
    except KeyboardInterrupt:
        for w in ws: w.terminate()


if __name__ == '__main__':
    main()
