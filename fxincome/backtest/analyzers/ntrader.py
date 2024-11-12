from backtrader import Analyzer
from backtrader.utils import AutoOrderedDict


class NTraderAnalyzer(Analyzer):
    def create_analysis(self):
        self.rets = AutoOrderedDict()

    def start(self):
        super().start()
        self.initial_cash = self.strategy.broker.get_cash()

    def stop(self):
        broker = self.strategy.broker
        etf_data = self.strategy.getdatabyname(self.strategy.etf_name)

        r = self.rets

        # Calculate the broker values
        r.broker.cash = broker.get_cash()
        r.broker.position = broker.getposition(etf_data).size
        r.broker.value = broker.get_value()
        r.broker.fund_value = broker.get_fundvalue()
        r.broker.fund_shares = broker.get_fundshares()

        # Round broker values
        broker_attrs = ["cash", "position", "value", "fund_value", "fund_shares"]
        broker_rounding = [2, 2, 2, 4, 2]
        for attr, digits in zip(broker_attrs, broker_rounding):
            r.broker[attr] = round(r.broker[attr], digits)

        # Calculate the trader values
        r.traders.cash = self.strategy.get_traders_cash()
        r.traders.position = self.strategy.get_traders_position()
        r.traders.value = self.strategy.total_value()

        # Round trader values
        traders_attrs = ["cash", "position", "value"]
        for attr in traders_attrs:
            r.traders[attr] = round(r.traders[attr], 2)

        # Calculate pnl attribution
        bond_commission = sum(
            trader.bond_commission for trader in self.strategy.traders
        )
        repo_commission = sum(
            trader.repo_commission for trader in self.strategy.traders
        )

        r.traders.pnl.gross = (
            r.traders.value - self.initial_cash + bond_commission + repo_commission
        )
        r.traders.pnl.net_ratio = (
            r.traders.value - self.initial_cash
        ) / r.traders.pnl.gross

        r.traders.pnl.commission_ratio = (
            bond_commission + repo_commission
        ) / r.traders.pnl.gross

        r.traders.pnl.bond_comm_ratio = bond_commission / r.traders.pnl.gross
        r.traders.pnl.repo_comm_ratio = repo_commission / r.traders.pnl.gross

        repo_pnl = sum(trader.repo_interest for trader in self.strategy.traders)
        r.traders.pnl.repo_pnl_ratio = repo_pnl / r.traders.pnl.gross

        bond_pnl = r.traders.pnl.gross - bond_commission - repo_commission - repo_pnl
        r.traders.pnl.bond_pnl_ratio = bond_pnl / r.traders.pnl.gross

        r.traders.gross_return = r.traders.pnl.gross / self.initial_cash
        r.traders.net_return = (r.traders.value - self.initial_cash) / self.initial_cash

        # Round pnl attribution
        trader_attrs = [
            'gross', 'net_ratio', 'commission_ratio', 'bond_comm_ratio',
            'repo_comm_ratio', 'repo_pnl_ratio', 'bond_pnl_ratio',
            'gross_return', 'net_return'
        ]
        for attr in trader_attrs:
            if attr.endswith('_ratio'):
                r.traders.pnl[attr] = round(r.traders.pnl[attr] * 100, 2)
            elif attr.endswith('_return'):
                r.traders[attr] = round(r.traders[attr] * 100, 2)
            else:
                r.traders.pnl[attr] = round(r.traders.pnl[attr], 2)
