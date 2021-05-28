
import pandas as pd
import datetime
import numpy as np
from fxincome.const import COUPON_TYPE
from fxincome.const import CASHFLOW_TYPE
from fxincome.const import ACCOUNT_TYPE
from dateutil.relativedelta import relativedelta

from fxincome.asset import  Bond

class Position_Bond:
    def __init__(self, id, bond, account_type, begin_quantity, begin_date, begin_cleanprice):
        self.id=id
        self.bond=bond
        self.account_type=account_type
        self.begin_quantity=begin_quantity
        self.begin_date=begin_date
        self.begin_cleanprice=begin_cleanprice
        self.begin_dirtyprice=bond.cleanprice_to_dirtyprice(begin_date, begin_cleanprice)
        self.begin_ytm=bond.cleanprice_to_ytm(begin_date, begin_cleanprice)

        self.quantity=begin_quantity
        self.RealDailyR=bond.amortprice_to_dailyrate(begin_date, begin_cleanprice)


        self.cashflow_history_df=pd.DataFrame([[begin_date,-begin_quantity*self.begin_dirtyprice/100]],columns=['date','cash'])
        self.position_gain_df=pd.DataFrame([], columns=['date', 'quantity', 'market_cleanprice', 'market_dirtyprice', 'market_ytm', 'cost_cleanprice', 'coupon', 'interest', 'price_gain', 'coupon_sum', 'interest_sum', 'price_gain_sum', 'float_gain_sum', 'gain_sum','dv01','duration'])
        self.date=begin_date+datetime.timedelta(days=-1)

    def get_cashflow(self,cashflow_type):
        if cashflow_type==CASHFLOW_TYPE.Undelivered:
            cashflow_Undelivered_df=self.bond.get_cashflow(self.date,'Undelivered')
            cashflow_Undelivered_df['cash']=cashflow_Undelivered_df['cash']*self.quantity/100
            cashflow_Undelivered_df['type']='Undelivered'
            cashflow_Undelivered_df=cashflow_Undelivered_df[cashflow_Undelivered_df['cash']!=0].copy()
            return cashflow_Undelivered_df
        elif cashflow_type==CASHFLOW_TYPE.History:
            cashflow_history_df=self.cashflow_history_df
            cashflow_history_df['type']='History'
            return cashflow_history_df
        elif cashflow_type==CASHFLOW_TYPE.All:
            cashflow_Undelivered_df=self.bond.get_cashflow(self.date,'Undelivered')
            cashflow_Undelivered_df['cash']=cashflow_Undelivered_df['cash']*self.quantity/100
            cashflow_history_df=self.cashflow_history_df
            cashflow_Undelivered_df['type']='Undelivered'
            cashflow_Undelivered_df=cashflow_Undelivered_df[cashflow_Undelivered_df['cash']!=0].copy()
            cashflow_history_df['type']='History'


            cashflow_all=pd.concat([cashflow_history_df,cashflow_Undelivered_df])
            return cashflow_all
        else:
            raise NotImplementedError("Unknown CASHFLOW_TYPE")
    def get_position_gain(self):
        position_gain_df=self.position_gain_df
        return position_gain_df

    def move_curve(self, newdate, curve_df=None, quantity_delta=None):
        ytm=self.bond.curve_to_ytm(newdate, curve_df) if isinstance(curve_df, pd.core.frame.DataFrame) else None
        self.move_ytm(newdate, ytm, quantity_delta)
    def move_ytm(self, newdate, ytm=None, quantity_delta=None):
        if self.quantity>0:

            while (self.date<newdate)&(self.date<self.bond._cashflow_df.iloc[-1, 0]):
                self.date+=datetime.timedelta(days=1)
                #算现金流
                if (self.date in list(self.bond._cashflow_df['date']))&(self.position_gain_df.empty == False):
                    cash=self.bond._cashflow_df[self.bond._cashflow_df['date'] == self.date].iloc[0, 1]
                    cash= cash * self.position_gain_df.iloc[-1, 1] / 100
                    self.cashflow_history_df=self.cashflow_history_df.append([{'date':self.date,'cash':cash}],ignore_index=True,sort=False)
                #算收益
                coupon=self.bond.get_dailycoupon(self.date)*self.quantity/100


                if self.account_type==ACCOUNT_TYPE.TPL:
                    self.position_gain_df=self.position_gain_df.append([{'date':self.date,
                                                              'quantity':self.quantity ,
                                                               'coupon':coupon,
                                                               'interest':coupon,
                                                               'cost_cleanprice':self.begin_cleanprice}], ignore_index=True, sort=False)

                elif self.account_type in [ACCOUNT_TYPE.OCI,ACCOUNT_TYPE.AC]:
                    cost_cleanprice_before=(self.begin_cleanprice if self.position_gain_df.empty else self.position_gain_df.iloc[-1, 5])
                    interest=cost_cleanprice_before*self.RealDailyR/100*self.quantity/100 if coupon!=0 else 0
                    # cost_cleanprice=cost_cleanprice_before+(interest-coupon)/self.quantity*100
                    cost_cleanprice=cost_cleanprice_before*(1+self.RealDailyR/100)-self.bond.get_dailycoupon(self.date)if coupon!=0 else cost_cleanprice_before
                    self.position_gain_df=self.position_gain_df.append([{'date':self.date,
                                                               'quantity':self.quantity,
                                                               'coupon':coupon,
                                                               'interest':interest,
                                                               'cost_cleanprice': cost_cleanprice}], ignore_index=True, sort=False)
                    pass
                else:
                    raise NotImplementedError("Unknown ACCOUNT_TYPE")
            if self.date ==self.bond._cashflow_df.iloc[-1, 0]:
                ytm=np.nan
                quantity_delta=-self.quantity
            if ytm:
                self.position_gain_df.iloc[-1, 2]=self.bond.ytm_to_cleanprice(self.date, ytm)
                self.position_gain_df.iloc[-1, 3]=self.bond.ytm_to_dirtyprice(self.date, ytm)
                self.position_gain_df.iloc[-1, 4]=ytm
                self.position_gain_df.iloc[-1, 12]= (self.position_gain_df.iloc[-1, 2] - self.position_gain_df.iloc[-1, 5]) * self.quantity / 100

                self.position_gain_df.iloc[-1, 14]= self.bond.ytm_to_dv01(self.date, ytm) * self.quantity / 100
                self.position_gain_df.iloc[-1, 15] = self.bond.ytm_to_duration(self.date, ytm, 'Modified')

            if quantity_delta:
                quantity=self.quantity
                self.quantity+=quantity_delta
                if quantity_delta>0:
                    raise Exception('Should Open a New Position'+'(id:'+str(self.id)+')')
                if self.quantity<0:
                    raise Exception('Not Enough Bond to Sell'+'(id:'+str(self.id)+')')
                if self.date ==self.cashflow_history_df.iloc[-1,0]:
                    self.cashflow_history_df.iloc[-1,1]+= -self.position_gain_df.iloc[-1, 3] * quantity_delta / 100 * (self.date != self.bond._cashflow_df.iloc[-1, 0])
                else:
                    self.cashflow_history_df=self.cashflow_history_df.append([{'date':self.date,'cash': -self.position_gain_df.iloc[-1, 3] * quantity_delta / 100}], ignore_index=True, sort=False)
                self.position_gain_df.iloc[-1, 1]=self.quantity

                self.position_gain_df.iloc[-1, 6]= self.position_gain_df.iloc[-1, 6] / quantity * self.quantity
                self.position_gain_df.iloc[-1, 7]= self.position_gain_df.iloc[-1, 7] / quantity * self.quantity
                self.position_gain_df.iloc[-1, 8]= -(self.position_gain_df.iloc[-1, 2] - self.position_gain_df.iloc[-2, 5]) * quantity_delta / 100
                self.position_gain_df.iloc[-1, 12]= (self.position_gain_df.iloc[-1, 2] - self.position_gain_df.iloc[-2, 5]) * self.quantity / 100
                self.position_gain_df.iloc[-1, 14] = self.bond.ytm_to_dv01(self.date, ytm) * self.quantity / 100
                # self.position_gain_df.iloc[-1, 14] =0
                # self.position_gain_df.iloc[-1, 15] = 0


            self.position_gain_df['coupon_sum']=self.position_gain_df['coupon'].cumsum()
            self.position_gain_df['interest_sum']=self.position_gain_df['interest'].cumsum()
            self.position_gain_df['price_gain_sum']=self.position_gain_df['price_gain'].fillna(0).cumsum()
            self.position_gain_df['gain_sum']= self.position_gain_df['interest_sum'] + self.position_gain_df['price_gain_sum'] + self.position_gain_df['float_gain_sum']
        if self.quantity==0:
            last_df= self.position_gain_df.iloc[-1:, ].copy()
            if last_df.iloc[0,0]!=newdate:
                last_df.iloc[0,0]=newdate
                last_df.iloc[0, 2:4] = 0
                last_df.iloc[0, 4] = np.nan
                self.position_gain_df=self.position_gain_df.append([last_df], ignore_index=True, sort=False)
                self.date=newdate




if __name__=='__main__':
    code='200016'
    initial_date=datetime.datetime(2020,11,19)
    end_date=datetime.datetime(2030,11,19)
    issue_price=100
    coupon_rate=3.27
    coupon_type='附息'
    coupon_frequency=2
    a=Bond(code, initial_date, end_date,issue_price, coupon_rate,coupon_type, coupon_frequency)