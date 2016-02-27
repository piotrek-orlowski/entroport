from __future__ import division, absolute_import
from unittest import TestCase
from numpy.testing import assert_almost_equal
import os
import pandas as pd

import entroport.utils as utils

class TestUtils(TestCase):

    def test_window(self):
        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.assertListEqual(list(utils.window(seq, 3, 2)),
                            [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]])

        self.assertListEqual(list(utils.window(seq, 3, 20)),
                            [[1, 2, 3]])

        self.assertListEqual(list(utils.window(seq, 30, 2)), [])

