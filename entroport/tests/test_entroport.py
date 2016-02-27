from __future__ import division, absolute_import
from unittest import TestCase
from numpy.testing import assert_almost_equal
import os
import pandas as pd
import numpy as np

from entroport import EntroPort

class TestEntroport(TestCase):
    test_dir = os.path.dirname(__file__)
    dftst = os.path.join(test_dir, 'french_25pf_sizebm_simplenet_excessreturns.csv')
    weightdata = os.path.join(test_dir, 'weights_25pf_192607-201012-444-12.csv')
    pfsdata = os.path.join(test_dir, 'pfs_25pf_192607-201012-444-12.csv')
    
    dftst = pd.read_csv(dftst, index_col=0)
    dftst = dftst.loc[192607:201012]
    dftst = dftst.apply(lambda x: np.log(1 + x))

    weightdata = pd.read_csv(weightdata, index_col=0)
    pfsdata = pd.read_csv(pfsdata, index_col=0)

    eptst = EntroPort(dftst, 444, 12).fit()

    def test_weights(self):
        assert_almost_equal(self.eptst.weights_.values,
                            self.weightdata.values, decimal=4)

    def test_ip(self):
        assert_almost_equal(self.eptst.pfs_['ip'].values,
                            self.pfsdata['ip'].values, decimal=4)

    def test_sdf(self):
        assert_almost_equal(self.eptst.pfs_['sdf'].values,
                            self.pfsdata['sdf'].values, decimal=4)

