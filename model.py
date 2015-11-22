from pylab import *
from uuid import uuid4
import cPickle as pickle
from util import safe_write
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde


class ConditionalDistribution(object):
    def __init__(self, x1, x2, predict_resolution=100):
        """
        fits x1 | x2
        """
        self.x1 = x1
        self.x2 = x2
        self.predict_resolution = predict_resolution

        ymin, ymax = np.percentile(self.x2, [1, 99])
        self.y = np.linspace(ymin, ymax, predict_resolution)
        self._cache = {}
        self.joint_estimate = None
        self.cond_estimate = None

    def fit(self):
        self.joint_estimate = gaussian_kde(np.vstack([self.x1, self.x2]))
        self.cond_estimate = gaussian_kde(self.x2)
        return self

    def predict(self, x):
        return self.y, self.joint_estimate(np.asarray([(x, e) for e in self.y]).T) / self.cond_estimate(x)

    def sample(self, x):
        # if False and x not in self._cache:
            y, probs = self.predict(x)
            probs = np.cumsum(probs)
            p = np.random.random() * probs[-1]
            return y[probs.searchsorted(p)]
            # self._cache[x] = y[probs.searchsorted(p)]
        # return self._cache[x]

    def draw_joint(self, resolution=100j):
        xmin, xmax = np.percentile(self.x1, [1, 99])
        ymin, ymax = np.percentile(self.x2, [1, 99])
        X, Y = np.mgrid[xmin:xmax:resolution, ymin:ymax:resolution]
        positions = np.vstack([X.ravel(), Y.ravel()])

        Z = np.reshape(self.joint_estimate(positions), X.shape).T
        imshow(Z, interpolation='nearest', origin='lower')
        locs = np.arange(0, int(resolution.imag), int(resolution.imag) / 6)
        xticks(locs, ['%.02f %%' % (e * 100) for e in X[locs, 0].squeeze()])
        yticks(locs, ['%.02f %%' % (e * 100) for e in Y[0, locs].squeeze()])

    def draw_all(self, resolution=100j):
        xmin, xmax = np.percentile(self.x1, [1, 99])
        ymin, ymax = np.percentile(self.x2, [1, 99])
        X, Y = np.mgrid[xmin:xmax:resolution, ymin:ymax:resolution]
        positions = np.vstack([X.ravel(), Y.ravel()])

        def draw_Z(Z):
            imshow(Z, interpolation='nearest', origin='lower')
            locs = np.arange(0, int(resolution.imag), int(resolution.imag) / 5)
            xticks(locs, ['%.02f' % e for e in X[locs, 0].squeeze()])
            yticks(locs, ['%.02f' % e for e in Y[0, locs].squeeze()])

        figure()
        subplot(311)
        Z = np.reshape(self.joint_estimate(positions), X.shape).T
        draw_Z(Z)

        subplot(312)
        draw_Z(Z / self.cond_estimate(Y[0]))

        subplot(313)
        plot(self.cond_estimate(Y[0]))


class Model(object):
    def __init__(self, save_collection=None, metadata_to_save=None):
        self.metadata_to_save = metadata_to_save
        self.save_collection = save_collection

    def fit(self, dfX, dfy):
        self.distrs = {}
        for predictor, x_values in dfX.iteritems():
            self.distrs[predictor] = {}
            for target, y_values in dfy.iteritems():
                self.distrs[predictor][target] = ConditionalDistribution(y_values, x_values).fit()
        return self

    def predict(self, dfX, df_cnt):
        def save_res(run_id, row_id, d):
            doc = {
                'pred': dict(d),
                'row_id': row_id,
                'run_id': run_id
            }
            doc['pred'].update({k + '_pct': v / s for k, v in d.iteritems()})

            if self.metadata_to_save:
                doc.update(self.metadata_to_save)
            self.save_collection.insert(doc)

        res = defaultdict(int)
        run_id = uuid4()
        for row_id, row in dfX.iterrows():
            if row_id % 100 == 0 and row_id > 0:
                print '%s of %s, run_id: %s' % (row_id, len(dfX), run_id)

                print dict(res)
                s = sum(res.values())
                print {k: v / s for k, v in res.iteritems()}

                if self.save_collection is not None:
                    save_res(run_id, row_id, res)

            for predictor, x_value in row.iteritems():
                targets = {}
                for target, distr in self.distrs[predictor].iteritems():
                    y_value = distr.sample(x_value)
                    targets[target] = y_value

                s = sum(targets.values())
                for target, value in targets.iteritems():
                    pred = df_cnt.ix[row_id][predictor.replace('_pct', '')] * value / s
                    if pd.isnull(pred):
                        raise ValueError('nan found on prediction!!')
                    res[target] += pred

        if self.save_collection is not None:
            save_res(run_id, len(dfX), res)

        return dict(res)
