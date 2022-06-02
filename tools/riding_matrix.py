import sys

from fxincome.asset import Bond
import datetime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

def get_curve(point, type):
    size=point.shape[0]
    if type=='LINEAR':
        def func(x):
            for i in range(1,size):
                if x<=point[i, 0]:
                    break
            return (point[i,1]-point[i-1,1])/(point[i,0]-point[i-1,0])*(x-point[i-1,0])+point[i-1,1]
    if type=='POLYNOMIAL':
        matrix_x=np.zeros([size,size])
        matrix_y=np.array(point[:,1])
        for i in range(size):
            for j in range(size):
                matrix_x[i,j]=point[i,0]**j
        para=np.dot(np.linalg.inv(matrix_x), matrix_y)
        def func(x):
            xx=np.array([ x**i for i in range(size)])
            return np.dot(para,xx)
    if type=='HERMIT':
        matrix_x=np.zeros([(size-1)*4,(size-1)*4])
        matrix_y=np.zeros([(size-1)*4])
        y_1=[(point[1,1]-point[0,1])/(point[1,0]-point[0,0])]+ \
            [(point[i+1,1]-point[i-1,1])/(point[i+1,0]-point[i-1,0]) for i in range(1,size-1)]+ \
            [(point[size-1,1]-point[size-2,1])/(point[size-1,0]-point[size-2,0])]
        for i in range(size-1):
            for j in range(2):
                matrix_x[2*i+j,4*i]= point[i + j, 0] ** 3
                matrix_x[2*i+j,4*i+1]= point[i + j, 0] ** 2
                matrix_x[2*i+j,4*i+2]=point[i + j, 0]
                matrix_x[2*i+j,4*i+3]=1
                matrix_y[2*i+j]=point[i + j, 1]

                matrix_x[2*(size-1)+2*i+j,4*i]= 3*point[i + j, 0] ** 2
                matrix_x[2*(size-1)+2*i+j,4*i+1]= 2*point[i + j, 0]
                matrix_x[2*(size-1)+2*i+j,4*i+2]=1
                matrix_y[2*(size-1)+2*i+j]=y_1[i + j]
        para=np.dot(np.linalg.inv(matrix_x),matrix_y)

        def func(x):
            xx=np.zeros((size-1)*4)
            for i in range(1,size):
                if x<=point[i, 0]:
                    break
            xx[4*(i-1)]=x**3
            xx[4*(i-1)+1]=x**2
            xx[4*(i-1)+2]=x
            xx[4*(i-1)+3]=1
            return np.dot(para,xx)
    if type=='SPLINE':
        matrix_x=np.zeros([(size-1)*4,(size-1)*4])
        matrix_y=np.zeros([(size-1)*4])
        for i in range(size-1):
            for j in range(2):
                matrix_x[2*i+j,4*i]= point[i + j, 0] ** 3
                matrix_x[2*i+j,4*i+1]= point[i + j, 0] ** 2
                matrix_x[2*i+j,4*i+2]=point[i + j, 0]
                matrix_x[2*i+j,4*i+3]=1
                matrix_y[2*i+j]=point[i + j, 1]
        for i in range(size-2):
            matrix_x[(size-1)*2+2*i,4*i]= 3 * point[i + 1, 0] ** 2
            matrix_x[(size-1)*2+2*i,4*i+1]= 2 * point[i + 1, 0]
            matrix_x[(size-1)*2+2*i,4*i+2]=1
            matrix_x[(size-1)*2+2*i,4*i+4]= -3 * point[i + 1, 0] ** 2
            matrix_x[(size-1)*2+2*i,4*i+5]= -2 * point[i + 1, 0]
            matrix_x[(size-1)*2+2*i,4*i+6]=-1

            matrix_x[(size-1)*2+2*i+1,4*i]= 6 * point[i + 1, 0]
            matrix_x[(size-1)*2+2*i+1,4*i+1]=2
            matrix_x[(size-1)*2+2*i+1,4*i+4]= -6 * point[i + 1, 0]
            matrix_x[(size-1)*2+2*i+1,4*i+5]=-2
            matrix_x[(size-1)*2+2*i+1,4*i+7]=-1
        matrix_x[-2,0]= 6 * point[0, 0]
        matrix_x[-2,1]=2
        matrix_x[-1,-4]= 6 * point[-1, 0]
        matrix_x[-1,-3]=2
        para=np.dot(np.linalg.inv(matrix_x),matrix_y)

        def func(x):
            xx=np.zeros((size-1)*4)
            for i in range(1,size):
                if x<=point[i, 0]:
                    break
            xx[4*(i-1)]=x**3
            xx[4*(i-1)+1]=x**2
            xx[4*(i-1)+2]=x
            xx[4*(i-1)+3]=1
            return np.dot(para,xx)
    return func
if __name__=='__main__':
    # init_date=datetime.datetime(2022,5,20)
    # end_date=datetime.datetime(2022,7,29)
    # init_ytm=3.4345
    # end_ytm=5
    #
    # bond=Bond(code='190210',
    #           initial_date=datetime.datetime(2021,11,19),
    #           end_date=datetime.datetime(2051,11,19),
    #           issue_price=100,
    #           coupon_rate=3.56,
    #           coupon_type='附息',
    #           coupon_frequency=1)
    # print(bond.get_profit(init_date,end_date,init_ytm,end_ytm))
    # import sys
    # sys.exit()
    address='./riding_matrix.xlsx'
    bond_type_need=['政策银行债','国债','地方政府债']
    asset_df=pd.read_excel(address,header=3,sheet_name='asset')
    parameter_df=pd.read_excel(address,sheet_name='parameter').set_index('参数')
    # print(parameter_df)
    date=parameter_df.at['基准日','数值']
    asset_df['initial_date']=pd.to_datetime(asset_df['initial_date'])
    asset_df['end_date']=pd.to_datetime(asset_df['end_date'])
    asset_df=asset_df[(asset_df['bond_type'].isin(bond_type_need))&
                      (asset_df['end_date']>date)&
                      (asset_df['code'].str.contains('IB'))].copy()
    asset_df['period']=asset_df['end_date'].apply(lambda x:round((x-date).days/365 ))
    asset_df['period2']=asset_df['end_date'].apply(lambda x:round((x-date).days/365,2 ))
    def maxx(x,i):
        i =len(x) if len(x)<i else i
        sort_x=sorted(x)[-i]
        return sort_x
    asset_df['ranking']=asset_df[['trading','period']].groupby('period').transform(lambda x:x>=maxx(x,2))
    asset_df=asset_df[(asset_df['ranking'])&(asset_df['trading']>0)].sort_values(['period2'],ignore_index=True)
    asset_df=asset_df.iloc[:,10:].set_index('code')
    curve_dot=asset_df[['period2','ytm']].to_numpy()
    curve=get_curve(curve_dot,'HERMIT')

    # plt.figure()
    # x=np.linspace(0,30,10000)
    # plt.plot(x,[curve(i) for i in x] )
    # plt.scatter(curve_dot[:,0],curve_dot[:,1],marker='*')
    # plt.grid(True)
    # plt.xticks(range(0,31))
    # # plt.show()
    #
    # address=r'.\result\rm_result_{}.jpg'.format(123)
    # plt.savefig(address,dpi=600)
    # sys.exit()



    specail_period=parameter_df.at['特殊参考期限','数值'].split(',')

    for spi in specail_period:

        spi_code='STD.{}Y'.format(spi)
        spi_bond_name='标准券{}Y'.format(spi)
        spi_end_date=date+relativedelta(years=int(spi))
        spi_rate=curve(float(spi))
        asset_df.loc[spi_code]=[spi_bond_name,date,spi_end_date,100,spi_rate,'附息',1,'标准券',1,spi_rate,float(spi),float(spi),True]

    asset_df=asset_df.sort_values(['period2'])
    # print(asset_df)


    asset_dic={}
    for i,j in asset_df.iterrows():
        bond_i=Bond(  code=i,
                      initial_date=j['initial_date'],
                      end_date=j['end_date'],
                      issue_price=j['issue_price'],
                      coupon_rate=j['coupon_rate'],
                      coupon_type=j['coupon_type'],
                      coupon_frequency=j['coupon_frequency'])
        asset_dic[i]=bond_i
    result_columns=[[i,j] for i in asset_dic.keys() for j in asset_dic.keys() if asset_df.at[i,'end_date']<=asset_df.at[j,'end_date']]
    result_df=pd.DataFrame(result_columns,columns=['code_hoding','code_riding'])
    # print(asset_df)

    with tqdm(total=len(result_df)) as step:
        for i, j in result_df.iterrows():
            result_df.at[i,'code_hoding_period']=asset_df.at[j['code_hoding'],'period2']
            result_df.at[i,'code_hoding_ytm']=asset_df.at[j['code_hoding'],'ytm']
            result_df.at[i,'code_riding_period']=asset_df.at[j['code_riding'],'period2']
            result_df.at[i,'code_riding_ytm']=asset_df.at[j['code_riding'],'ytm']
            # print(j['code_hoding'],date,
            #       asset_df.at[j['code_hoding'],'end_date'],
            #       asset_df.at[j['code_hoding'],'ytm'],
            #       asset_df.at[j['code_hoding'],'ytm'])

            result_df.at[i,'holding_yeild']=asset_dic[j['code_hoding']].get_profit(date,
                                                                                   asset_df.at[j['code_hoding'],'end_date'],
                                                                                   asset_df.at[j['code_hoding'],'ytm'],
                                                                                   asset_df.at[j['code_hoding'],'ytm'])[1]



            riding_end_period=(asset_df.at[j['code_riding'],'end_date']-asset_df.at[j['code_hoding'],'end_date']).days/365
            riding_end_ytm=curve(riding_end_period)
            # print(j['code_riding'],date,
            #       asset_df.at[j['code_hoding'],'end_date'],
            #       asset_df.at[j['code_riding'],'ytm'],
            #       riding_end_ytm)
            result_df.at[i,'yeild']=asset_dic[j['code_riding']].get_profit(date,
                                                                              asset_df.at[j['code_hoding'],'end_date'],
                                                                              asset_df.at[j['code_riding'],'ytm'],
                                                                              riding_end_ytm)[1]



            if j['code_hoding']==j['code_riding']:
                y=result_df.at[i,'code_riding_ytm']
            else:
                y1=-5
                y2=10
                while True:

                    # print(j['code_riding'],date,
                    #       asset_df.at[j['code_hoding'],'end_date'],
                    #       asset_df.at[j['code_riding'],'ytm'],
                    #       y1)
                    # yeild1=asset_dic[j['code_riding']].get_profit(date,
                    #                                              asset_df.at[j['code_hoding'],'end_date'],
                    #                                              asset_df.at[j['code_riding'],'ytm'],
                    #                                              y1)[1]
                    y=(y1+y2)/2
                    # print(j['code_riding'],date,
                    #       asset_df.at[j['code_hoding'],'end_date'],
                    #       asset_df.at[j['code_riding'],'ytm'],
                    #       y)
                    yeild=asset_dic[j['code_riding']].get_profit(date,
                                                                 asset_df.at[j['code_hoding'],'end_date'],
                                                                 asset_df.at[j['code_riding'],'ytm'],
                                                                 y)[1]
                    # print(yeild,result_df.at[i,'holding_yeild'],y)
                    if abs(yeild-result_df.at[i,'holding_yeild'])<0.01:
                        break
                    if yeild<result_df.at[i,'holding_yeild']:
                        y2=y
                    else:
                        y1=y
            result_df.at[i,'balance_ytm']=y
            result_df.at[i,'bp']=(y-result_df.at[i,'code_riding_ytm'])*100
            step.update(1)
            # print(result_df.iloc[:i+1,:])




    # print(result_df)
    result_df['holding']=result_df.apply(lambda x:'{}\n({:.2f}Y,{:.2f}%)'.format(x['code_hoding'],x['code_hoding_period'],x['code_hoding_ytm']),axis=1)
    result_df['riding']=result_df.apply(lambda x:'{}\n({:.2f}Y,{:.2f}%)'.format(x['code_riding'],x['code_riding_period'],x['code_riding_ytm']),axis=1)
    rank_type=CategoricalDtype(list(result_df['holding'].drop_duplicates()[::-1]),ordered=True)
    columns=pd.MultiIndex.from_product([list(result_df['holding'].drop_duplicates()[::-1]),['yeild','bp']])
    result_df['holding']=result_df['holding'].astype(rank_type)
    result_df['riding']=result_df['riding'].astype(rank_type)
    result_df=pd.pivot_table(result_df,index='holding',columns='riding',values=['yeild','bp'],aggfunc='sum')


    # print(result_df)
    #
    # print(columns)
    result_df.columns=result_df.columns.swaplevel()
    result_df=result_df[columns]
    result_df=result_df.applymap(lambda x:'{:.2f}'.format(x))
    # print(result_df.columns)
    # print(result_df[columns])
    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    address = r'.\result\rm_result_{}.xlsx'.format(time)
    wirter = pd.ExcelWriter(address)
    result_df.to_excel(wirter, sheet_name='result')
    wirter.save()

    plt.figure()
    x=np.linspace(0,30,10000)
    plt.plot(x,[curve(i) for i in x] )
    plt.scatter(curve_dot[:,0],curve_dot[:,1],marker='*')
    plt.grid(True)
    plt.xticks(range(0,31))
    address=r'.\result\rm_result_{}.jpg'.format(time)
    plt.savefig(address,dpi=600)

