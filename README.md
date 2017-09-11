## entroport
portfolio allocation with relative entropy minimization

`python` implementation of the procedure outlined in *A One Factor Benchmark Model for Asset Pricing by Ghosh, Julliard, and Taylor (2015)*, with the addition of a penalization step.

## Installation
(Requires `Scipy` / `Numpy` / `Pandas`, tested on OSX with Python 2.7)

```
$ pip install git+https://github.com/erikcs/entroport.git
# or
$ pip install entroport
```

## Example
Simply instantiate a [model object](http://pythonhosted.org/entroport/api.html) like in [this](http://pythonhosted.org/entroport/intro.html) example
```
>>> from entroport import EntroPort
>>> ep = EntroPort(returns_frame, estlength=650, step=12).fit()
```
