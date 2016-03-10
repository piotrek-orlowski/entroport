## entroport
portfolio allocation with relative entropy minimization

`python` implementation of the procedure outlined in *A One Factor Benchmark Model for Asset Pricing by Ghosh, Julliard, and Taylor (2015 working paper)*

The figure below shows the path of $1 (log scale) invested in a) the market b) an equal weight in the 25 Fama-French portfolios, and c) an "IP" weight in the 25 Fama-French portfolios, estimated on a rolling out of sample basis, and rebalanced annually

![alt tag](plot1.png)

## Installation
(Requires `Scipy` / `Numpy` / `Pandas`, tested on OSX with Python 2.7)

`pip install entroport`

## Example
Simply instantiate a [model object](http://pythonhosted.org/entroport/api.html) like in [this](http://pythonhosted.org/entroport/intro.html) example
```
>>> from entroport import EntroPort
>>> ep = EntroPort(returns_frame, estlength=650, step=12).fit()
```
