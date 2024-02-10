import pickle
import re
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import json
import requests
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('House_Rent_Train.xlsx')
df.drop('amenities',inplace=True,axis=1)

df.dropna(inplace=True)
df['year']=pd.DatetimeIndex(df['activation_date']).year
df.drop(columns=['activation_date','id'],inplace=True,axis=1)
df = df.drop(df[df['property_age'] == -1.0].index)
df.drop(['latitude','longitude'],axis=1,inplace=True)
df.drop(['negotiable'],axis=1,inplace=True)
df.drop(['total_floor'],axis=1,inplace=True)
df['tr_cup_board']=np.log([x+1 for x in df['cup_board']])
df['tr_property_size']=np.log(df['property_size'])
df['tr_property_age']=np.log([x+1 for x in df['property_age']])
df['tr_floor']=np.log([x+1 for x in df['floor']])
df['tr_balconies']=np.log([x+1 for x in df['balconies']])
df['tr_rent']=np.log(df['rent'])
df.drop(['cup_board','property_size','property_age','floor','balconies','rent'],axis=1,inplace=True)
df.drop(['locality'],axis=1,inplace=True)

le_type = preprocessing.LabelEncoder()
df['type']=le_type.fit_transform(df['type'])

le_facing=preprocessing.LabelEncoder()
df['facing']=le_facing.fit_transform(df['facing'])

le_ws=preprocessing.LabelEncoder()
df['water_supply']=le_ws.fit_transform(df['water_supply'])

le_bt=preprocessing.LabelEncoder()
df['building_type']=le_bt.fit_transform(df['building_type'])

le_fur=preprocessing.LabelEncoder()
df['furnishing']=le_fur.fit_transform(df['furnishing'])

le_lt=preprocessing.LabelEncoder()
df['lease_type']=le_lt.fit_transform(df['lease_type'])

le_par=preprocessing.LabelEncoder()
df['parking']=le_par.fit_transform(df['parking'])

X=df.drop('tr_rent',axis=1)
y=df['tr_rent']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
rf = RandomForestRegressor(n_estimators=100,min_samples_split=10,min_samples_leaf=1,max_features='sqrt',max_depth=20,bootstrap=True)
rf.fit(x_train,y_train)
pred=rf.predict(x_test)
r_2=r2_score(y_test, pred)

mse=mean_squared_error(y_test, pred)

mae=mean_absolute_error(y_test, pred)

with open('regressor.pkl', 'wb') as file:
    pickle.dump(rf, file)

with open('encoder_1.pkl','wb') as file:
    pickle.dump(le_type,file)
with open('encoder_2.pkl','wb') as file:
    pickle.dump(le_facing,file)

with open('encoder_3.pkl','wb') as file:
    pickle.dump(le_ws,file)

with open('encoder_4.pkl','wb') as file:
    pickle.dump(le_bt,file)

with open('encoder_5.pkl','wb') as file:
    pickle.dump(le_fur,file)

with open('encoder_6.pkl','wb') as file:
    pickle.dump(le_lt,file)

with open('encoder_7.pkl','wb') as file:
    pickle.dump(le_par,file)

st.set_page_config(layout="wide")

st.title("HOUSE RENT PRICE")

tab=st.tabs(['Predicting House rent price'])

type=['BHK2', 'BHK3', 'BHK1', 'RK1', 'BHK4', 'BHK4PLUS', '1BHK1', 'bhk2',
       'bhk3']
facing=['NE', 'E', 'S', 'N', 'SE', 'W', 'NW', 'SW']
water_supply=['CORPORATION', 'CORP_BORE', 'BOREWELL']
building_type=['AP', 'IH', 'IF', 'GC']
furnishing=['SEMI_FURNISHED', 'FULLY_FURNISHED', 'NOT_FURNISHED']
lease_type=['ANYONE', 'FAMILY', 'BACHELOR', 'COMPANY']
parking=['BOTH', 'TWO_WHEELER', 'NONE', 'FOUR_WHEELER']
gym=[0,1]
lift=[0,1]
swimming_pool=[0,1]
bathroom=[1,2,3,4,5]
year=[2017,2018]
floor=[ 6.,  3.,  1., 2., 11.,  4., 14.,  5.,  9.,  7., 10.,  8.,
       13., 12., 16., 25., 17., 20., 15., 18., 19., 22.]
cup_board=[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]
balconies=[1.,2.,3.,4.,5.]
with st.form("my_form"):
    col1, col2 = st.columns([3,3])
    with col1:
        st.write(' ')
        Type = st.selectbox("Type", type, key=1)
        Facing = st.selectbox("Facing", facing, key=2)
        Ws = st.selectbox("Water Supply",water_supply , key=3)
        Building_type= st.selectbox("Building Type", building_type, key=4)
        Furnishing = st.selectbox("Furninshing", furnishing, key=5)
        Lt = st.selectbox("Lease Type", lease_type, key=6)
        Parking = st.selectbox("Parking", parking, key=7)
        Gym = st.selectbox("Gym", gym, key=8)
        Lift= st.selectbox("Lift", lift, key=9)
        Sp= st.selectbox("Swimming pool", swimming_pool, key=10)
        Bathroom= st.selectbox("Bathroom", bathroom, key=11)
        Year = st.selectbox("Year", year, key=12)
        Floor= st.selectbox("Floor", sorted(floor), key=13)
        Cup_board = st.selectbox("Cup_board", cup_board, key=14)
        Balconies = st.selectbox("Balconies", balconies, key=15)

    with col2:
        Property_size=st.text_input("Enter Property Size (Min:400.0 & Max:3000.0)")
        Property_age=st.text_input("Enter Property Age Date(Min:0.0 & Max:50.0)")

        submit = st.form_submit_button(label="Predict House Rent PRICE")

    flag = 0
    pattern = '[0-9]*\.?[0-9]+'
    for i in [Property_size, Property_age]:
        if re.match(pattern, i):
            pass
        else:
            flag = 1
            break

if submit and flag == 1:
    if len(i) == 0:
        st.write("please enter a valid number space not allowed")
    else:
        st.write("You have entered an invalid value: ", i)

if submit and flag == 0:
    with open(r"regressor.pkl", 'rb') as file:
        loaded_model = pickle.load(file)

    with open(r"encoder_1.pkl", 'rb') as f:
        le_1_load = pickle.load(f)

    with open(r"encoder_2.pkl", 'rb') as f:
        le_2_load = pickle.load(f)

    with open(r"encoder_3.pkl", 'rb') as f:
        le_3_load = pickle.load(f)

    with open(r"encoder_4.pkl", 'rb') as f:
        le_4_load = pickle.load(f)

    with open(r"encoder_5.pkl", 'rb') as f:
        le_5_load = pickle.load(f)

    with open(r"encoder_6.pkl", 'rb') as f:
        le_6_load = pickle.load(f)

    with open(r"encoder_7.pkl", 'rb') as f:
        le_7_load = pickle.load(f)

    new_sample = np.array([[np.log(float(Property_size)), np.log(float(Property_age)), np.log(float(Cup_board)),np.log(float(Floor)),
                             np.log(float(Balconies)),Type,Facing,Ws,
                            Building_type,Furnishing,Lt,Parking,Gym,Lift,Sp,Bathroom,Year]])
    new_sample_le_1 = le_1_load.transform(new_sample[:, [5]])
    new_sample_le_2 = le_2_load.transform(new_sample[:, [6]])
    new_sample_le_3 = le_3_load.transform(new_sample[:, [7]])
    new_sample_le_4 = le_4_load.transform(new_sample[:, [8]])
    new_sample_le_5 = le_5_load.transform(new_sample[:, [9]])
    new_sample_le_6 = le_6_load.transform(new_sample[:, [10]])
    new_sample_le_7 = le_7_load.transform(new_sample[:, [11]])

    new_sample = np.column_stack(
        (new_sample[:, [0, 1,2,3,4,12,13,14,15,16]],new_sample_le_1, new_sample_le_2, new_sample_le_3, new_sample_le_4,new_sample_le_5,new_sample_le_6,new_sample_le_7))
    new_pred = loaded_model.predict(new_sample)[0]
    st.write('## :green[Predicted House Rent Price:] ', round(np.exp(new_pred)))



























