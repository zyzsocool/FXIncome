
from fxincome.asset import Bond
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
from pandas.api.types import CategoricalDtype
pd.set_option('display.max_rows', None)
from dateutil.relativedelta import relativedelta
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
    address='./riding_compare.xlsx'
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
    asset_df=asset_df.iloc[:,10:]
    curve_dot=asset_df[['period2','ytm']].to_numpy()
    curve=get_curve(curve_dot,'HERMIT')




    period=parameter_df.at['特殊参考期限','数值'].split(',')
    period_list=[]
    for k in period:
        delta=int(k[:-1])
        if k[-1]=='D':
            end_date_k= date + relativedelta(days=delta)
        elif k[-1]=='M':
            end_date_k= date + relativedelta(months=delta)
        elif k[-1]=='Y':
            end_date_k= date + relativedelta(years=delta)
        else:
            continue
        period_list.append([k,end_date_k])
    asset_dic={}
    for i,j in asset_df.iterrows():
        bond_i=Bond(  code=j['code'],
                      initial_date=j['initial_date'],
                      end_date=j['end_date'],
                      issue_price=j['issue_price'],
                      coupon_rate=j['coupon_rate'],
                      coupon_type=j['coupon_type'],
                      coupon_frequency=j['coupon_frequency'])
        asset_dic[j['code']]=bond_i

        for k in period_list:
            if k[1]<=j['end_date']:
                end_ytm=curve((j['end_date']-k[1]).days/365)
                asset_df.loc[i,k[0]]=bond_i.get_profit(date, k[1], j['ytm'], end_ytm)[1]



    # print(asset_dic)

    asset_df['bond']=asset_df.apply(lambda x:'{}[{}Y][{:.2f}%]'.format(x['code'],x['period2'],x['ytm']),axis=1)
    reuslt_overview=asset_df[['bond']+period].set_index('bond')
    print(reuslt_overview)
    reuslt_overview=reuslt_overview.applymap(lambda x:'{:.2f}'.format(x))
    print(reuslt_overview)

    asset_df=asset_df.set_index('code')
    # print(asset_df)
    result_dic={}


    for k in period_list:
        # print(k[0])
        column_k=asset_df[asset_df['end_date']>=k[1]].index
        column_k=[[i,j] for i in column_k for j in column_k]
        reuslt_k=pd.DataFrame(column_k,columns=['bond_base','bond_target'])
        reuslt_k['bond_base_ytm']=reuslt_k['bond_base'].apply(lambda x:asset_df.loc[x,'ytm'])
        reuslt_k['bond_base_yeild']=reuslt_k['bond_base'].apply(lambda x:asset_df.loc[x,k[0]])
        reuslt_k['bond_base_period']=reuslt_k['bond_base'].apply(lambda x:asset_df.loc[x,'period2'])

        reuslt_k['bond_target_ytm']=reuslt_k['bond_target'].apply(lambda x:asset_df.loc[x,'ytm'])
        reuslt_k['bond_target_yeild']=reuslt_k['bond_target'].apply(lambda x:asset_df.loc[x,k[0]])
        reuslt_k['bond_target_period']=reuslt_k['bond_target'].apply(lambda x:asset_df.loc[x,'period2'])
        total_k=len(reuslt_k)
        print(reuslt_k)

        for i,j in reuslt_k.iterrows():
            print('{}:{:.0f}/{:.0f},{:.2%}'.format(k[0],i,total_k,(i+1)/total_k))



            if j['bond_base']==j['bond_target']:
                # base_bp=(curve((asset_df.loc[j['bond_base'],'end_date']-k[1]).days/365)-j['bond_base_ytm'])*100
                reuslt_k.loc[i,'bp']=0
            else:
                y1=-50
                y2=50
                y_last=0
                while True:
                    y=(y1+y2)/2
                    # if i==18:
                    #     print(y)
                    yeild=asset_dic[j['bond_base']].get_profit(date,k[1],j['bond_base_ytm'],y)[1]
                    if abs(yeild-j['bond_target_yeild'])<0.01:
                        break
                    if (abs(yeild-j['bond_target_yeild'])<0.1)&(abs(y-y_last)<0.0001):
                        break
                    if yeild<j['bond_target_yeild']:
                        y2=y
                    else:
                        y1=y
                    # if i==18:
                    #     print(y,y_last,yeild,j['bond_target_yeild'])

                    y_last=y
                reuslt_k.loc[i,'bp']=(y-curve((asset_df.loc[j['bond_base'],'end_date']-k[1]).days/365))*100




        reuslt_k['base_bond']=reuslt_k.apply(lambda x:'{}\n[{}Y]\n[{:.2f}%]\n[{:.2f}%]'.format(x['bond_base'],x['bond_base_period'],x['bond_base_ytm'],x['bond_base_yeild']),axis=1)
        reuslt_k['target_bond']=reuslt_k.apply(lambda x:'{}\n[{}Y]\n[{:.2f}%]\n[{:.2f}%]'.format(x['bond_target'],x['bond_target_period'],x['bond_target_ytm'],x['bond_target_yeild']),axis=1)
        rank_type=CategoricalDtype(list(reuslt_k['base_bond'].drop_duplicates()),ordered=True)

        reuslt_k['base_bond']=reuslt_k['base_bond'].astype(rank_type)
        reuslt_k['target_bond']=reuslt_k['target_bond'].astype(rank_type)
        result_k_pivot=pd.pivot_table(reuslt_k,index='base_bond',columns='target_bond',values='bp',aggfunc='sum').applymap(lambda x:'{:.2f}'.format(x))




        # print(reuslt_k)
        # print(result_k_pivot)
        result_dic[k[0]]=result_k_pivot

    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    address = r'.\result\rc_result_{}.xlsx'.format(time)
    writer = pd.ExcelWriter(address)
    reuslt_overview.to_excel(writer, sheet_name='result')
    for i,j in result_dic.items():
        j.to_excel(writer,sheet_name=i)
    writer.save()


    plt.figure()
    x=np.linspace(0,curve_dot[-1,0],10000)
    plt.plot(x,[curve(i) for i in x] )
    plt.scatter(curve_dot[:,0],curve_dot[:,1],marker='*')
    plt.xticks(range(0,int(curve_dot[-1,0])+2))
    plt.grid(True)

    address=r'.\result\rc_result_{}.jpg'.format(time)
    plt.savefig(address,dpi=600)



