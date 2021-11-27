import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

status = pd.read_csv('./Documents/signate/status.csv')
station = pd.read_csv('./Documents/signate/station.csv')
weather = pd.read_csv('./Documents/signate/weather.csv')
trip =  pd.read_csv('./Documents/signate/trip.csv')

status['date'] = status['year'].astype(str) + '/' + status['month'].astype(str).str.zfill(2).astype(str) + '/' + status['day'].astype(str).str.zfill(2).astype(str)
status['date'] = pd.to_datetime(status['date'])
#statusのdate列の曜日を数値化
status['week_num'] = status['date'].dt.weekday

#weeknumの５、６が休日なので特徴量として追加
status['holiday']=status['week_num'].apply(lambda x: 1 if x == 6 or x==5 else 0)

#statusにstationのstation_idをキーとしてcity列をmarge
status = pd.merge(status, station[['station_id', 'city']], how = 'left')

#station_idをキーにstationのdock_countを結合する
status = pd.merge(status, station[['station_id', 'dock_count']], how = 'left')

#city列をエンコード
city_dict = {'city1':1,'city2':50,'city3':3,'city4':4,'city5':5}
status['city']=status['city'].apply(lambda x : city_dict[x])

#自転車数の変化が大きい、もしくは小さい時間帯を特徴量として追加
def time_segment1(x):
    if  x <= 6 :
        y =1
    else:
        y=0
    return y

def time_segment(x):
    if  x >= 7 & x <= 10:
        y=1
    elif x>=16 & x <=19:
        y=2
    else:
        y=0
    return y

status['time_segment1']=status['hour'].apply(time_segment1)
status['time_segment']=status['hour'].apply(time_segment)

#その日の0時時点の自転車数を特徴量として追加
#station_id, dateでグルーピングしたときの一番初めの値を取得
t = status.groupby(['station_id', 'date']).first()['bikes_available'].reset_index()
#24回リピートすることでデータのサイズを合わせる
t = pd.DataFrame(np.repeat(t.values, 24, axis=0))
t.columns = ['station_id', 'date', 'bikes_available_at0']
status['bikes_available_at0'] = t['bikes_available_at0']
status['bikes_available_at0']=status['bikes_available_at0'].astype(float)

#City２で分割しない（bikes_available_at0）あり
train = status[status['predict'] == 0]
test = status[(status['date'] >= '2014-09-01') & (status['predict'] == 1)]
train = train[train['bikes_available'].notna()]
target3=train['bikes_available']
train3=train.drop(['date','bikes_available'],axis=1)
test3=test.drop(['date','bikes_available'],axis=1)

#City２で分割しない（bikes_available_at0）なし
target4=train['bikes_available']
train4=train.drop(['date','bikes_available','bikes_available_at0'],axis=1)
test4=test.drop(['date','bikes_available','bikes_available_at0'],axis=1)

#City２で分割する（bikes_available_at0）なし
train = status[status['predict'] == 0]
test = status[(status['date'] >= '2014-09-01') & (status['predict'] == 1)]
train = train[train['bikes_available'].notna()]
train1=train[train['city']==50]
train2=train[train['city']!=50]
test1=test[test['city']==50]
test2=test[test['city']!=50]

target1=train1['bikes_available']
train1=train1.drop(['date','bikes_available','bikes_available_at0'],axis=1)
test1=test1.drop(['date','bikes_available','bikes_available_at0'],axis=1)
target2=train2['bikes_available']
train2=train2.drop(['date','bikes_available','bikes_available_at0'],axis=1)
test2=test2.drop(['date','bikes_available','bikes_available_at0'],axis=1)

def mod2(train0,target0,num_boost_rounds):
    X_train, X_test, y_train, y_test = train_test_split(train0, target0,test_size=0.2,random_state=0,stratify=target0)

    models = []


    row_no_list = list(range(len(y_train)))


    K_fold = StratifiedKFold(n_splits=5, shuffle=True,  random_state=42)


    for train_cv_no, eval_cv_no in K_fold.split(row_no_list, y_train):
    
        X_train_cv = X_train.iloc[train_cv_no, :]
        y_train_cv = pd.Series(y_train).iloc[train_cv_no]
        X_eval_cv = X_train.iloc[eval_cv_no, :]
        y_eval_cv = pd.Series(y_train).iloc[eval_cv_no]
    
    
        lgb_train = lgb.Dataset(X_train_cv, y_train_cv,
                            free_raw_data=False)
    
        lgb_eval = lgb.Dataset(X_eval_cv, y_eval_cv, reference=lgb_train,
                           free_raw_data=False)
    
   
        params = {'task': 'train',            
              'boosting_type': 'gbdt',        
              'objective': 'regression',
              'metric': 'rmse',
              'random_seed': 0,
              'deterministic': True,
              'force_row_wise': True,
              'feature_pre_filter': False,
              'lambda_l1': 1.2911923426758777e-05,
              'lambda_l2': 3.112840897586614e-05,
              'num_leaves': 255,
              'feature_fraction': 1.0,
              'bagging_fraction': 0.8553286686090561,
              'bagging_freq': 4,
              'min_child_samples': 5
              }

        evaluation_results = {}                                     
        model = lgb.train(params,                                   
                          lgb_train,                                
                          num_boost_round=num_boost_rounds,                     
                          valid_names=['train', 'valid'],           
                          valid_sets=[lgb_train, lgb_eval],         
                          evals_result=evaluation_results,          
                          early_stopping_rounds=20,                 
                          verbose_eval=-1)                          
    
    
    
    # 学習が終わったモデルをリストに入れておく
        models.append(model) 
    return models

#予測結果を出力
def outp(models,test):
    model0=models[0]
    pred0 = model0.predict(test, num_iteration=model0.best_iteration) 
    model1=models[1]
    pred1 = model1.predict(test, num_iteration=model1.best_iteration) 
    model2=models[2]
    pred2 = model2.predict(test, num_iteration=model2.best_iteration) 
    model3=models[3]
    pred3 = model3.predict(test, num_iteration=model3.best_iteration) 
    model4=models[4]
    pred4 = model4.predict(test, num_iteration=model4.best_iteration) 
    test['prepred0']=pred0
    test['prepred1']=pred1
    test['prepred2']=pred2
    test['prepred3']=pred3
    test['prepred4']=pred4

    test['pred']=test['prepred0']/5+test['prepred1']/5+test['prepred2']/5+test['prepred3']/5+test['prepred4']/5
    

    
models01=mod2(train1,target1,5000)
outp(models01,test1)
models02=mod2(train2,target2,5000)
outp(models02,test2)
sub01=test1[['id','pred']]
sub02=test2[['id','pred']]
sub1=pd.concat([sub01,sub02],axis=0)
sub1=sub1.sort_values('id')
sub1=sub1[['id','pred']]

models3=mod2(train3,target3,5000)
outp(models3,test3)
models4=mod2(train4,target4,700)
outp(models4,test4)
sub3=test3[['id','pred']]
sub4=test4[['id','pred']]

#スタッキング
sub1['predict']=sub1['pred']*0.1+sub3['pred']*0.05+sub4['pred']*0.85

sub1=sub1.sort_values('id')
sub1=sub1[['id','predict']]

sub1.to_csv("sub.csv",index=False, header=False)