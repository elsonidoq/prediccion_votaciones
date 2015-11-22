import csv
import pandas as pd
from collections import defaultdict
import os


def load_raw_data(fnames, cache_fname='raw_data.h5py'):
    if os.path.exists(cache_fname):
        return pd.read_hdf(cache_fname, 'df')
    else:
        docs = []
        for fname in fnames:
            with open(fname) as f:
                for doc in csv.DictReader(f, delimiter=';'):
                    doc = {k.strip().lower().replace(' ', '_'): v.strip() for k, v in doc.iteritems()}
                    doc = {k: int(v) if v.isdigit() else v for k, v in doc.iteritems()}
                    docs.append(doc)

        df = pd.DataFrame(docs)
        df.to_hdf('raw_data.h5py', 'df')
        return df


def get_grouped_dataset(raw_data, level=4):
    fields = 'codigo_provincia codigo_departamento codigo_circuito codigo_mesa'.split()
    fields = fields[:level]
    print "grouping for %s" % ",".join(fields)
    cache_fname = '_'.join(fields)

    if os.path.exists(cache_fname):
        gdf = pd.read_hdf(cache_fname, 'df')
        return gdf[(~pd.isnull(gdf)).all(1)]  # there is a row with nulls

    else:

        vectors = []
        for group_id, locs in raw_data.groupby(by=fields).groups.iteritems():
            vector = defaultdict(int)
            vector.update(dict(zip(fields, group_id)))

            for _, row in raw_data.iloc[locs].iterrows():
                vector[str(row.codigo_votos)] += row.votos

            vectors.append(vector)

        gdf = pd.DataFrame(vectors)

        gdf['diff'] = gdf['135'] - gdf['131']
        gdf['total'] = sum(gdf[k] for k in '132 133 137 138 135 131'.split()) + 6

        for k in '132 133 137 138 135 131'.split():
            gdf[k + "_pct"] = (gdf[k] + 1.0) / gdf.total

        # gdf = gdf[gdf['135'] > 0]
        # gdf = gdf[gdf['131'] > 0]

        gdf.to_hdf(cache_fname, 'df')

        return gdf
