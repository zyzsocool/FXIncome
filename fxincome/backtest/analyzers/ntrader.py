from backtrader import Analyzer
from backtrader.utils import AutoOrderedDict
import pprint

class NTraderAnalyzer(Analyzer):
    def create_analysis(self):
        self.rets = AutoOrderedDict()

    def start(self):
        super().start()
        self.etf_data = self.strategy.getdatabyname(self.strategy.etf_name)
        self.initial_cash = self.strategy.broker.get_cash()

    def stop(self):
        broker = self.strategy.broker
        r = self.rets

        # Calculate the broker values
        r.broker.cash = broker.get_cash()
        r.broker.position = broker.getposition(self.etf_data).size
        r.broker.value = broker.get_value()
        r.broker.fund_value = broker.get_fundvalue()
        r.broker.fund_shares = broker.get_fundshares()

        # Calculate the trader values
        r.traders.cash = self.strategy.get_traders_cash()
        r.traders.position = self.strategy.get_traders_position()
        r.traders.value = self.strategy.total_value()

        # Calculate pnl attribution
        bond_commission = sum(
            trader.bond_commission for trader in self.strategy.traders
        )
        repo_commission = sum(
            trader.repo_commission for trader in self.strategy.traders
        )

        gross_pnl = (
            r.traders.value - self.initial_cash + bond_commission + repo_commission
        )
        r.traders.pnl.net_ratio = (r.traders.value - self.initial_cash) / gross_pnl

        r.traders.pnl.commission_ratio = (bond_commission + repo_commission) / gross_pnl

        r.traders.pnl.bond_comm_ratio = bond_commission / gross_pnl
        r.traders.pnl.repo_comm_ratio = repo_commission / gross_pnl

        repo_pnl = sum(trader.repo_interest for trader in self.strategy.traders)
        r.traders.pnl.repo_pnl_ratio = repo_pnl / gross_pnl

        bond_pnl = gross_pnl - bond_commission - repo_commission - repo_pnl
        r.traders.pnl.bond_pnl_ratio = bond_pnl / gross_pnl

        r.traders.gross_return = gross_pnl / self.initial_cash
        r.traders.net_return = (r.traders.value - self.initial_cash) / self.initial_cash

        # Calculate benchmark return. ETF_Data has been processed completely. The close[0] is the last close price.
        start_etf_price = self.etf_data.close[-self.etf_data.buflen()+1]
        end_etf_price = self.etf_data.close[0]
        r.benchmark.net_return = (end_etf_price - start_etf_price) / start_etf_price

        # Round broker values
        broker_attrs = ["cash", "position", "value", "fund_value", "fund_shares"]
        broker_rounding = [2, 2, 2, 4, 2]
        for attr, digits in zip(broker_attrs, broker_rounding):
            r.broker[attr] = round(r.broker[attr], digits)

        # Round trader values
        traders_attrs = ["cash", "position", "value"]
        for attr in traders_attrs:
            r.traders[attr] = round(r.traders[attr], 2)

        # Round pnl attribution
        trader_attrs = [
            "net_ratio",
            "commission_ratio",
            "bond_comm_ratio",
            "repo_comm_ratio",
            "repo_pnl_ratio",
            "bond_pnl_ratio",
            "gross_return",
            "net_return",
        ]
        for attr in trader_attrs:
            if attr.endswith("_ratio"):
                r.traders.pnl[attr] = round(r.traders.pnl[attr] * 100, 2)
            elif attr.endswith("_return"):
                r.traders[attr] = round(r.traders[attr] * 100, 2)
            else:
                r.traders.pnl[attr] = round(r.traders.pnl[attr], 2)

        # Round bechmark values
        r.benchmark.net_return = round(r.benchmark.net_return * 100, 2)