import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import streamlit as st
import pickle
import joblib

## loading the model 
model = joblib.load(open(r"C:\Users\Kimo Store\Desktop\project for git hub\lgb_loan_prediction.pkl" , 'rb'))

# tilte
st.title("Machine Learning App")
st.subheader("Approved Loan Prediction")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>' , unsafe_allow_html=True)
st.image('https://entrepreneursbreak.com/wp-content/uploads/2020/12/trusted-loan-company-in-Houston-TX-360x180.jpg')
@st.cache_data
def load_data(data):
    return pd.read_csv(data)

## Uplaoding the data
data = st.file_uploader('Upload File' , type=['csv'])
if data is not None:
    df = load_data(data)
## make list for numerical features and categorical features 
num_features = df.select_dtypes(exclude="O").columns
cat_features = df.select_dtypes(include="O").columns
col1 , col2 = st.columns((2))

## Married Features adjust yes and no into Married Single
df['Married'].replace({'No':'Single' , 'Yes' : 'Married'} , inplace=True)

    ## Self_Employed Features adjust yes and no into Employed and unEmployed
df['Self_Employed'].replace({'No':'UnEmployed' , 'Yes' : 'Employed'} , inplace=True)

## handling missing values 

for col in num_features:
    df[col] = df[col].fillna(df[col].median())

for col in cat_features:
    df[col] = df[col].fillna(df[col].mode()[0])

st.header("The Relationship Between Categorical Features And Target")

#st.write(df.isnull().sum())

## Visualization Categorical Features
##1- the impact of married on approval
with col1 :
    educated = df.groupby(['Education','Status']).size().reset_index()
    fig = plt.figure(figsize=(16, 6))
    sns.countplot(data=df , x="Education", hue='Status' , palette='RdYlBu_r' )
    plt.xticks(size = 26)
    plt.title('Education And Approval' , size = 24)
    plt.ylabel('Ferquancy', size = 24 )
    plt.xlabel('Education' , size = 24)
    plt.yticks(size = 20)
    plt.legend(fontsize = 26)
    st.pyplot(fig)    
    ## download the data
    with st.expander('View Education Data'):
        st.table(educated.T.style.background_gradient(cmap="Green"))
        csv = educated.to_csv(index = False).encode('utf-8')
        st.download_button("Download File" , data=csv,file_name='educated.csv')
    
## Impact of Married on Approval 
    married = df.groupby(['Married' , 'Status']).size().reset_index()
    fig = plt.figure(figsize=(16, 6))
    sns.countplot(data=df , x="Married", hue='Status' , palette='ocean_r')
    plt.xticks(size = 26)
    plt.title('Married And Approval' , size = 24)
    plt.ylabel('Ferquancy', size = 24 )
    plt.xlabel('Married' , size = 24)
    plt.yticks(size = 20)
    plt.legend(fontsize = 26)

    st.pyplot(fig)    
## Download the data
    with st.expander('View Married Data'):
        st.table(married.T.style.background_gradient(cmap=('Green')))
        csv = married.to_csv(index = False).encode('utf-8')
        st.download_button("Download File" , data = csv , file_name='married.csv')

    ## the Distribution of loan amount
    fig = px.histogram(data_frame= df ,x =  'Loan_Amount' ,title='Distribution Of Loan_Amount', nbins=500 , color = 'Loan_Amount' )
    st.plotly_chart(fig , use_container_width=True)

with col2 :
    ## the relationship between area and approval
    area = df.groupby(['Area' , 'Status']).size().reset_index()
    fig = plt.figure(figsize=(16, 6))
    sns.countplot(data=df , x="Area", hue='Status' , palette='RdYlBu_r')
    plt.xticks(size = 26)
    plt.title('Area And Approval' , size = 24)
    plt.ylabel('Ferquancy', size = 24 )
    plt.xlabel('Area' , size = 24)
    plt.yticks(size = 20)
    plt.legend(fontsize = 26)
    st.pyplot(fig , use_container_width=True)    
    ## download the data
    with st.expander('View Area Data'):
        st.table(area.T.style.background_gradient(cmap="Blue"))
        csv = area.to_csv(index = False).encode('utf-8')
        st.download_button("Download File" , data=csv,file_name='area.csv')

    ## the relationship between Self_Employed and approval
    Self_Employed = df.groupby(['Self_Employed' , 'Status']).size().reset_index()
    fig = plt.figure(figsize=(16, 6))
    sns.countplot(data=df , x="Self_Employed", hue='Status' , palette='ocean_r')
    plt.xticks(size = 26)
    plt.title('Self_Employed And Approval' , size = 24)
    plt.ylabel('Ferquancy', size = 24 )
    plt.xlabel('Self_Employed' , size = 24)
    plt.yticks(size = 20)
    plt.legend(fontsize = 26)
    st.pyplot(fig , use_container_width=True)    
    ## download the data
    with st.expander('View Self_Employed Data'):
        st.table(Self_Employed.style.background_gradient(cmap="Greys"))
        csv = Self_Employed.to_csv(index = False).encode('utf-8')
        st.download_button("Download File" , data=csv,file_name='Self_Employed.csv')

    fig = px.histogram(data_frame=df , x = 'Applicant_Income' , title='Distribution Of Applicant_Income', nbins=500 , color ='Applicant_Income' )
    st.plotly_chart(fig , use_container_width=True)

## The Distribution of Status
value = df['Status'].value_counts().values
name = df['Status'].value_counts().keys()
fig = px.pie(values=value , names=name , color=value , title='The Percentage Of Apprval')
fig.update_traces(text = name , textposition = 'inside')
st.plotly_chart(fig , use_container_width=True)
## make the data for the model 
st.sidebar.header("Prediction")

## gender features train_data['Gender'] = train_data['Gender'].replace({'Male' : 0 , 'Female' : 1 , 'Missing' : 2})
gender_c = ['Male','Female' , 'Missing']
gender_n = [0,1,2]
gender_mapping = dict(zip(gender_c , gender_n))
gender = st.sidebar.selectbox('Select Gender' , gender_c)
fin_gen = gender_mapping[gender]

## train_data['Education'] = train_data['Education'].replace({'Not Graduate' : 0 , 'Graduate' : 1})
education_c = ['Not Graduate' , 'Graduate']
education_n = [0,1]
education_mapping = dict(zip(education_c , education_n))
education = st.sidebar.selectbox("Select Education" , education_c)
fin_edu = education_mapping[education]

## test_data['Dependents'] = test_data['Dependents'].replace({'Missing' : 4 , '3+':3})
Dependents_c = ['0' , '1' , '2' , '3+' , 'Missing']
Dependents_n = [0 , 1 , 2 , 3 , 4]
Dependents_maping = dict(zip(Dependents_c , Dependents_n))
Dependents = st.sidebar.selectbox('Select Dependents' , Dependents_c)
fin_Dep = Dependents_maping[Dependents]

## train_data['Married'] = train_data['Married'].replace({'No' : 0 , 'Yes' : 1 , 'Missing' : 2})
married_c = ['Yes' , 'No' , 'Missing']
married_n = [0,1,2]    
married_mapping = dict(zip(married_c , married_n))
married = st.sidebar._selectbox('Married' , married_c)
fin_marr = married_mapping[married]

##train_data['Self_Employed'] = train_data['Self_Employed'].replace({'No' : 0 , 'Yes' : 1 , 'Missing' : 2})
Self_Employed_c = ['Yes' , 'No' , 'Missing']
Self_Employed_n = [0,1,2]    
Self_Employed_mapping = dict(zip(Self_Employed_c , Self_Employed_n))
Self_Employed = st.sidebar._selectbox('Self_Employed' , Self_Employed_c)
fin_empl = Self_Employed_mapping[Self_Employed]

## train_data['Area'] = train_data['Area'].replace({'Semiurban' : 0 , 'Urban' : 1 , 'Rural' : 2})
area_c = ['Semiurban' , 'Urban' , 'Rural']
area_n = [0,1,2]    
are_mapping = dict(zip(area_c , area_n))
area = st.sidebar._selectbox('Area' , area_c)
fin_area = are_mapping[area]

## Applicant_Income
Applicant_Income = st.sidebar.number_input('Applicant_Income' , min_value=0)

## Coapplicant_Income
Coapplicant_Income = st.sidebar.number_input('Coapplicant_Income' , min_value=0)

## Loan_Amount
Loan_Amount = st.sidebar.number_input('Loan_Amount' , min_value=0)

## Term
Term =  st.sidebar.number_input('Term' , min_value=0)

## Credit_History 
Credit_History_c = ['0' , '1' , 'Missing']
Credit_History_n = [0.1,2]
Credit_mapping = dict(zip(Credit_History_c ,Credit_History_n ))
Credit_History = st.sidebar.selectbox("Credit_History" , Credit_History_c)
fin_cre = Credit_mapping[Credit_History]

## prepare the data 
data = pd.DataFrame({'Gender' : fin_gen , 'Married':fin_marr , "Dependents" : fin_Dep , 'Education':fin_edu , 'Self_Employed' :fin_empl,'Area':fin_area,
                        'Applicant_Income':Applicant_Income , 'Coapplicant_Income': Coapplicant_Income , 'Loan_Amount':Loan_Amount , 'Term':Term ,
                        'Credit_History':fin_cre , } , index=[0])


but = st.sidebar.button('Predict')
if but:
    pred = model.predict(data) 
    if pred == 1:
        st.sidebar.write( "Approved")
    else :
        st.sidebar.write( "Disaproved")






















