#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
from backtrader import Analyzer
from backtrader.mathsupport import average
from backtrader.utils import AutoOrderedDict


class Kelly(Analyzer):
    """
    Kelly Criterion.
    假设投资者初始本金为1，每次投资可选择投入本金x，x∈[0, 1]，有p的概率赢，获得正收益r1x，有q的概率输，损失本金r2x。
    那么效应函数为：
    f(x)=p ln(1 + r1x) + q ln(1 - r2x)
    最大化效应函数，求导，得到最优x为：
    x = p/r2 - q/r1
    """

    def create_analysis(self):
        """Replace default implementation to instantiate an AutoOrdereDict
        rather than an OrderedDict"""
        self.rets = AutoOrderedDict()

    def start(self):
        super().start()
        # Returns on Investment for wins：保留每笔盈利交易的收益率
        self.roi_wins = list()
        # Returns on Investment for losses：保留每笔亏损交易的收益率
        self.roi_losses = list()

    def notify_trade(self, trade):
        if trade.status == trade.Closed:
            pnl = trade.pnlcomm
            total_value = 0
            for h in trade.history:
                total_value = total_value + h.status.value
            avg_value = total_value / len(trade.history)
            investment = avg_value - trade.pnl + trade.commission
            roi = pnl / investment

            if trade.pnlcomm >= 0:
                # 盈利加入盈利列表，利润0算盈利
                self.roi_wins.append(roi)
            else:
                # 亏损加入亏损列表
                self.roi_losses.append(roi)

    def stop(self):
        # 防止除以0
        if len(self.roi_wins) > 0 and len(self.roi_losses) > 0:

            avg_roi_wins = average(self.roi_wins)  # 计算平均盈利
            avg_roi_losses = abs(average(self.roi_losses))  # 计算平均亏损（取绝对值）

            if avg_roi_wins == 0 or avg_roi_losses == 0:
                kelly_percent = None
            else:
                num_wins = len(self.roi_wins)  # 获胜次数
                num_losses = len(self.roi_losses)  # 亏损次数
                num_trades = num_wins + num_losses  # 总交易次数
                win_prob = num_wins / num_trades  # 计算胜率
                loss_prob = 1 - win_prob

                # 计算凯利比率
                # 即每次交易投入资金占总资金的最优比率
                kelly_percent = win_prob / avg_roi_losses - loss_prob / avg_roi_wins
        else:
            kelly_percent = None  # 信息不足

        self.rets.kellyRatio = kelly_percent  # 例如 0.215
