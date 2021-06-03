

import pandas as pd
import datetime
import numpy as np
from fxincome.const import COUPON_TYPE
from fxincome.const import CASHFLOW_TYPE
from fxincome.const import ACCOUNT_TYPE
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from fxincome.asset import Bond
from fxincome.position import Position_Bond
from fxincome.const import CASHFLOW_VIEW_TYPE
from fxincome.const import POSITION_GAIN_VIEW_TYPE
class Portfolio_Bond():
    def __init__(self,asset_dic,position_dic,curve_df,trade_df,init_curve_date_list):
        self.asset_dic=asset_dic
        self.position_dic=position_dic
        self.curve_df=curve_df
        self.trade_df=trade_df
        self.init_curve_date_list=init_curve_date_list
        self.cashflow_df=None
        self.position_gain_df=None


    def cal_result(self):
        dates_curve=list(self.curve_df.columns[1:])
        dates_trade=set(self.trade_df['date'])
        if set(dates_trade).difference(dates_curve):
            raise Exception('A Trade Without Curve Value')
        #逐个跑position
        for date_i in dates_curve:
            curve_df=self.curve_df[['days',date_i]]
            for position_id ,position in self.position_dic.items():
                quantity_df=self.trade_df[(self.trade_df['date']==date_i)&(self.trade_df['id']==position_id)]
                if quantity_df.shape[0]>1:
                    raise Exception('Can Not Sell One Positon Twice in One Day'+'(id:'+str(quantity_df.iloc[0,1])+')')
                elif quantity_df.shape[0]==1:
                    quantity_delta=quantity_df.iloc[0,3]
                else:
                    quantity_delta=None
                position.move_curve(date_i,curve_df,quantity_delta)
            new_position_df=self.trade_df[(self.trade_df['date']==date_i)&(pd.notnull(self.trade_df['code']))]
            if not new_position_df.empty:
                for index,row in new_position_df.iterrows():
                    asset_i=self.asset_dic[row['code']]
                    begin_cleanprice_i=asset_i.curve_to_cleanprice(date_i, curve_df)
                    quantity=row['quantity_delta']
                    if quantity<0:
                        raise Exception('Can Not Sell Out An Empty Position'+'(id:'+str(row['id'])+')')
                    position_i=Position_Bond(row['id'],
                                             asset_i,
                                             row['account_type'],
                                             quantity,
                                             row['date'],
                                             begin_cleanprice_i)
                    position_i.move_curve(date_i,curve_df)
                self.position_dic[row['id']]=position_i

        #跑完position汇总最终结果：cashflow和position_gain
        cashflow_list=[]
        position_gain_list=[]
        for position_id_i ,position_i in self.position_dic.items():
            id=position_i.pid
            code=position_i.bond.code
            account_type=position_i.account_type

            cashflow_i=position_i.get_cashflow('All')
            cashflow_i['id']=id
            cashflow_i['code']=code
            cashflow_i['account_type']=account_type
            cashflow_list.append(cashflow_i)

            position_gain_i=position_i.get_position_gain()
            position_gain_i['id']=id
            position_gain_i['code']=code
            position_gain_i['account_type']=account_type
            position_gain_list.append(position_gain_i)

        cashflow_df=pd.concat(cashflow_list)
        cashflow_df=cashflow_df.sort_values(by='date').reset_index(drop=True)
        position_gain_df=pd.concat(position_gain_list)
        position_gain_df=position_gain_df.sort_values(by='date').reset_index(drop=True)

        self.cashflow_df=cashflow_df
        self.position_gain_df=position_gain_df

    def get_cashflow(self,cashflow_view_type='Raw'):
        cashflow_df=self.cashflow_df

        dates=sorted(set(cashflow_df['date']))
        if cashflow_view_type==CASHFLOW_VIEW_TYPE.Raw:
            result_df=cashflow_df

        elif cashflow_view_type==CASHFLOW_VIEW_TYPE.Agg:
            result_df=pd.DataFrame([],columns=['date','cash_in','cash_out','cash_in_hand','type'])
            for date_i in dates:
                cash_in=cashflow_df[(cashflow_df['date']==date_i)&(cashflow_df['cash']>0)]['cash'].sum()
                cash_out=cashflow_df[(cashflow_df['date']==date_i)&(cashflow_df['cash']<0)]['cash'].sum()
                cash_in_hand=cashflow_df[(cashflow_df['date']<=date_i)&(cashflow_df['date']>dates[0])]['cash'].sum()
                type=cashflow_df[cashflow_df['date']==date_i].iloc[0,2]
                result_df=result_df.append([{'date':date_i,
                                             'cash_in':cash_in,
                                             'cash_out':cash_out,
                                             'cash_in_hand':cash_in_hand,
                                             'type':type}],ignore_index=True,sort=False)
        return result_df
    def get_position_gain(self,position_gain_view_type):
        position_gain_df=self.position_gain_df
        if position_gain_view_type==POSITION_GAIN_VIEW_TYPE.Raw:
            result_df=position_gain_df
        if position_gain_view_type==POSITION_GAIN_VIEW_TYPE.Agg:
            dates1=list(self.cashflow_df['date'])
            dates2=list(self.curve_df.columns[1:])
            dates=set(dates1+dates2)

            result_df=position_gain_df[position_gain_df['date'].isin(dates)].copy()
            result_df.iloc[list(result_df['date'].isin(dates2)==False),2:-1]=np.nan
            result_df['market_value']=result_df['quantity']*result_df['market_dirtyprice']/100
            result_df=result_df.groupby('date').apply(lambda x:pd.Series({'quantity':x['quantity'].sum(),
                                                                          'market_value':sum(x['market_value']),
                                                                          'gain_sum':sum(x['gain_sum']),
                                                                          'interest_sum':sum(x['interest_sum']),
                                                                          'price_gain_sum':sum(x['price_gain_sum']),
                                                                          'float_gain_sum':sum(x['float_gain_sum']),

                                                                          'dv01':sum(x['dv01']),
                                                                          'duration':sum(x['duration']*x['market_value'])/sum(x['market_value']),

                                                                          'quantity_TPL':sum(x['quantity']*(x['account_type']=='TPL')),
                                                                          'quantity_OCI':sum(x['quantity']*(x['account_type']=='OCI')),
                                                                          'quantity_AC':sum(x['quantity']*(x['account_type']=='AC')),

                                                                          'market_value_TPL':sum(x['market_value']*(x['account_type']=='TPL')),
                                                                          'market_value_OCI':sum(x['market_value']*(x['account_type']=='OCI')),
                                                                          'market_value_AC':sum(x['market_value']*(x['account_type']=='AC')),

                                                                          'gain_sum_TPL':sum(x['gain_sum']*(x['account_type']=='TPL')),
                                                                          'gain_sum_OCI':sum(x['gain_sum']*(x['account_type']=='OCI')),
                                                                          'gain_sum_AC':sum(x['gain_sum']*(x['account_type']=='AC')),

                                                                          'interest_sum_TPL':sum(x['interest_sum']*(x['account_type']=='TPL')),
                                                                          'interest_sum_OCI':sum(x['interest_sum']*(x['account_type']=='OCI')),
                                                                          'interest_sum_AC':sum(x['interest_sum']*(x['account_type']=='AC')),

                                                                          'price_gain_sum_TPL':sum(x['price_gain_sum']*(x['account_type']=='TPL')),
                                                                          'price_gain_sum_OCI':sum(x['price_gain_sum']*(x['account_type']=='OCI')),
                                                                          'price_gain_sum_AC':sum(x['price_gain_sum']*(x['account_type']=='AC')),

                                                                          'float_gain_sum_TPL':sum(x['float_gain_sum']*(x['account_type']=='TPL')),
                                                                          'float_gain_sum_OCI':sum(x['float_gain_sum']*(x['account_type']=='OCI')),
                                                                          'float_gain_sum_AC':sum(x['float_gain_sum']*(x['account_type']=='AC'))
                                                                          })).reset_index()

        condition=np.append(True,np.array(result_df['quantity'][:-1])!=np.array(result_df['quantity'][1:]))
        result_df=result_df[condition|(pd.notnull(result_df['gain_sum']))].copy()

        return result_df
    def get_view(self):
        cashflow_df=self.get_cashflow('Agg')
        position_gain_df=self.get_position_gain('Agg')
        result_df=pd.merge(cashflow_df,position_gain_df,how='outer')
        result_df=result_df.sort_values(by='date').reset_index(drop=True)
        result_df['cash_in']=result_df['cash_in'].fillna(0)
        result_df['cash_out']=result_df['cash_out'].fillna(0)
        result_df['cash_in_hand']=result_df['cash_in_hand'].fillna(method='ffill')
        result_df['type']=result_df['type'].fillna(method='ffill')
        result_df['quantity']=result_df['quantity'].fillna(method='ffill')
        result_df['quantity']=result_df.apply(lambda x:x['quantity'] if x['type']=='History' else np.nan ,axis=1)

        return result_df

    def get_plot(self):
        result_df=self.get_view()
        print(result_df)
        result_df=result_df[result_df['type']=='History']
        begin_date=result_df.iloc[0,0]
        end_date=result_df.iloc[-1,0]
        df_all=pd.DataFrame(pd.date_range(begin_date,end_date,freq='D'),columns=['date'])
        result_df=result_df[result_df['type']=='History']
        df_all=pd.merge(df_all,result_df,how='left')
        df_all['cash_in_hand']=df_all['cash_in_hand'].fillna(method='ffill')

        df_all['gain_sum']=df_all['gain_sum'].interpolate()
        df_all['interest_sum']=df_all['interest_sum'].interpolate()
        df_all['price_gain_sum']=df_all['price_gain_sum'].interpolate()
        df_all['float_gain_sum']=df_all['float_gain_sum'].interpolate()
        df_all['market_value']=df_all['market_value'].interpolate()
        df_all['dv01']=df_all['dv01'].interpolate()
        df_all['dv01']=abs(df_all['dv01'])
        df_all['duration']=df_all['duration'].interpolate()

        df_all_gain=df_all[['date','gain_sum','interest_sum','price_gain_sum','float_gain_sum']].set_index('date')
        df_all_cash_value=df_all[['date','cash_in_hand','market_value']].set_index('date')
        df_all_cash_value['sum']=df_all_cash_value['cash_in_hand']+df_all_cash_value['market_value']
        df_all_risk=df_all[['date','dv01','duration']].set_index('date')

        curve_df=self.curve_df[self.init_curve_date_list].set_index('days')


        df_all_gain.plot(title='Gain')
        df_all_cash_value.plot(title='Value in Hand')
        df_all_risk.plot(secondary_y=['dv01'],title='Risk Indicator')
        curve_df.plot(title='Core Curve')



        plt.show()
        return result_df




        # result_df['gain_sum']=result_df['gain_sum'].interpolate()
        # result_df['gain_sum'].plot()
        # plt.show()
