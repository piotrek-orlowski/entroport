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

Get som arbitrary fund data from vanguard and fidelity with ``pandas_datareader``
(the usual imports are suppressed)

.. ipython:: python

    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2015, 1, 1)

    f = data.DataReader(['VTI', 'VGSTX', 'VWELX', 'VFIIX', 'VPMCX',
                        'FBALX', 'FDGRX', 'FCNTX', 'FLPSX', 'FBGRX'],
                        'yahoo', start, end)

    fsp = data.DataReader('^GSPC','yahoo', start, end) # S&P 500                          
    fsp = fsp['Adj Close'].pct_change().iloc[1:]
    rets = f['Adj Close'].pct_change().iloc[1:]
    rets = rets.apply(lambda x: np.log(1 + x))
    rets.head(5)

Construct a model object with an estimation window of 500 observations and a
step size of 12 observations (i.e rebalancing every 12 days)

.. ipython:: python

    from entroport import EntroPort
    ep = EntroPort(rets, 500, 12).fit()

Have a look at the cumulative return

.. ipython:: python
    
     @savefig plot1.png width=6in
     (ep.pfs_['ip'] + 1).cumprod().plot();

Does not look particularly good compared to the S&P 500

.. ipython:: python
    
     @savefig plot2.png width=6in
     (fsp[ep.pfs_.index[0]:] + 1).cumprod().plot();


The estimated weights (only point estimates are stored computed) are rather
noisy¶

.. ipython:: python
    
     @savefig plot3.png width=6in
     ep.weights_.plot().legend(loc='center left', bbox_to_anchor=(1, .5));

The estimated :math:`\theta_i`'s 
(the *argmin* of :math:`\frac{1}{T} \sum_{t=1}^{T}e^
{\boldsymbol{\theta}' \mathbf{R}_t}`) 

.. ipython:: python
    
     @savefig plot4.png width=6in
     ep.thetas_.plot().legend(loc='center left', bbox_to_anchor=(1, .5));

