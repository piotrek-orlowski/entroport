.. _intro:

Introduction
============

.. ipython:: python
    :suppress:

    import pandas as pd
    import numpy as np
    import datetime
    from pandas_datareader import data, wb
    import seaborn as sns

Get som arbitrary daily fund data from vanguard and ishares with ``pandas_datareader``
(the usual imports are suppressed)

.. ipython:: python

    start = datetime.datetime(2011, 1, 1)
    end = datetime.datetime(2016, 1, 1)

    f = data.DataReader(['VTWG', 'VBK', 'VONG', 'IUSG', 'IUSV',
                        'VONV', 'VTWV', 'VIOV', 'IJK', 'IVOV'],
                        'yahoo', start, end)

    fsp = data.DataReader('^GSPC','yahoo', start, end) # S&P 500                          
    fsp = fsp['Adj Close'].pct_change().iloc[1:]
    rets = f['Adj Close'].pct_change().iloc[1:]
    rets = rets.apply(lambda x: np.log(1 + x))
    rets.head(5)

Construct a model object with an estimation window of 650 observations and a
step size of 12 observations (i.e rebalancing every 12 days)

(these two parameters could be tuned with for ex. cross validation)

.. ipython:: python

    from entroport import EntroPort
    ep = EntroPort(rets, 650, 12).fit()

Have a look at the cumulative return

.. ipython:: python
    
     (ep.pfs_['ip'] + 1).cumprod().plot();

     @savefig plot1.png width=6in
     (fsp[ep.pfs_.index[0]:] + 1).cumprod().plot().legend(['IP', 'SP500'], loc=2);


The estimated weights look reasonable (only point estimates are computed)
but are noisy¶

.. ipython:: python
    
     @savefig plot2.png width=6in
     ep.weights_.plot().legend(loc='center left', bbox_to_anchor=(1, .5));

The estimated :math:`\theta_i`'s 
(the *argmin* of :math:`\frac{1}{T} \sum_{t=1}^{T}e^
{\boldsymbol{\theta}' \mathbf{R}_t}`) 

.. ipython:: python
    
     @savefig plot3.png width=6in
     ep.thetas_.plot().legend(loc='center left', bbox_to_anchor=(1, .5));

