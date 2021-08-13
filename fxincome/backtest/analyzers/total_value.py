import backtrader as bt
from collections import OrderedDict

class TotalValue(bt.Analyzer):
    """
    This analyzer will get total value from every next.
          Returns:
          Returns a dictionary with returns as values and the datetime points for each return as keys
    """
    params = ()
    def __init__(self):
        super(TotalValue, self).__init__()

    def start(self):
        super(TotalValue, self).start()
        self.rets = OrderedDict()

    def next(self):
        # Calculate the return
        super(TotalValue, self).next()
        datetime_index = self.data.datetime.datetime()
        self.rets[datetime_index] = self.strategy.broker.getvalue()

    def get_analysis(self):
        return self.rets