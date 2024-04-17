import streamlit as st
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from streamlit.components.v1 import html
st.set_page_config(page_title="Airline Passenger Satisfaction", page_icon=":airplane", layout="wide")

df = pd.read_csv("base.csv")

model = joblib.load('kaan2.pkl')

st.title("Hakuna Madata Airlines Passenger Satisfaction")

main_tab, feedback_tab = st.tabs(["Home Page", "FeedBack"])

main_tab.write(""":rainbow[This app analyzes and visualizes the airline passenger satisfaction dataset. Discover the factors that affect passengers' satisfaction levels and predict the satisfaction of flights.]""")
#right_col.image("Logo.webp")
# Title and introduction of the survey
feedback_tab.write('We appreciate your time to help us improve our services.')

# Background Resminin Ayarlanması






# Ana Ekran Giriş Sayfası

# Sayfa Footer HTML Kod Uygulaması
with open("style/footer.html", "r", encoding="utf-8") as file:
    footer_html = file.read()

main_tab.markdown(footer_html, unsafe_allow_html=True)


# Set the width to your desired value

main_tab.markdown(
    f"""
    <style>
  
        section[data-testid="stSidebar"] {{
            width: 200px !important; 
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# URL or local path to your background image

# Function to save responses (this example just prints them to the console)
def save_responses(responses):
    print("Responses received:")
    for question, response in responses.items():
        print(f"{question}: {response}")

# Questions
responses = {}
otherdata = {}

def pred_data(df, df_input):

    def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
        dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
        return dataframe

    def grab_col_names(dataframe, cat_th=20, car_th=40):
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                       dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                       dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # num_cols
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        return cat_cols, num_cols, cat_but_car


    df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mean(), inplace=True)
    df['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1}, inplace=True)
    # df = df.drop(columns=["Unnamed: 0"])
    # df = df.drop(columns=["id"])
    df = pd.concat([df, df_input], ignore_index=True)

    # Yolcu yaşına göre segmentasyon
    df["Total Delay in Minutes"] = abs(df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"])

    # Yolcu yaşına göre segmentasyon
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 17, 30, 50, 70, 120],
                             labels=['0-17', '18-30', '31-50', '51-70', '71+'])

    # Uçuş mesafesine göre segmentasyon
    df['Flight_Distance_Level'] = pd.cut(df['Flight Distance'], bins=[0, 1000, 3000, 5000, 10000, float('inf')],
                                         labels=['Short Haul', 'Medium Haul', 'Long Haul', 'Very Long Haul',
                                                 'Ultra Long Haul'])

    # Uçuş gecikmesi durumları
    df['Departure_Delay_Status'] = pd.cut(df['Departure Delay in Minutes'], bins=[-1, 0, 15, 60, 180, float('inf')],
                                          labels=['No Delay', 'Minor Delay', 'Moderate Delay', 'Significant Delay',
                                                  'Severe Delay'])
    df['Arrival_Delay_Status'] = pd.cut(df['Arrival Delay in Minutes'], bins=[-1, 0, 15, 60, 180, float('inf')],
                                        labels=['No Delay', 'Minor Delay', 'Moderate Delay', 'Significant Delay',
                                                'Severe Delay'])

    # Toplam hizmet memnuniyeti skoru
    service_columns = ['Inflight wifi service', 'Ease of Online booking', 'Food and drink', 'Online boarding',
                       'Seat comfort',
                       'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
                       'Checkin service',
                       'Inflight service', 'Cleanliness']
    df['Total_Service_Satisfaction_Score'] = df[service_columns].sum(axis=1) / (len(service_columns))

    # Inflight Service Satisfaction (Uçuş İçi Hizmet Memnuniyeti)
    inflight_service_columns = ['Inflight entertainment', 'Leg room service', 'Baggage handling', 'Inflight service',
                                'Cleanliness']
    df['Inflight_Service_Satisfaction'] = df[inflight_service_columns].sum(axis=1) / (len(inflight_service_columns))

    # Online Services Satisfaction (Çevrimiçi Hizmet Memnuniyeti)
    online_columns = ["Ease of Online booking", "Online boarding"]
    df['Online_Services_Satisfaction_Score'] = df[online_columns].sum(axis=1) / (len(online_columns))

    # Airport Service Satisfaction (Havaalanı İçi Hizmet Memnuniyeti)
    airport_service_columns = ['On-board service', "Departure/Arrival time convenient", "Gate location",
                               "Baggage handling", "Checkin service"]
    df['Airport_Services_Satisfaction_Score'] = df[airport_service_columns].sum(axis=1) / (len(airport_service_columns))

    # Inflight_Service_Satisfaction_Per_Mile (Uçuş İçi Mil Başına Hizmet Memnuniyeti)
    df['Inflight_Service_Satisfaction_Per_Mile'] = df['Inflight_Service_Satisfaction'] / df["Flight Distance"]

    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=20, car_th=40)

    cat_cols.remove("satisfaction")

    new_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'Age_Group', 'Flight_Distance_Level',
                'Departure_Delay_Status',
                'Arrival_Delay_Status']
    df = one_hot_encoder(df, new_cols, drop_first=True)

    X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

    X = df.drop(["satisfaction"], axis=1)

    # Inputun Ölçeklenmiş ve Encode Edilmiş Halinin Modele Hazırlanması
    input_df = pd.DataFrame(X.iloc[-1])

    # l1 = [float(df1[0]), float(df1[1]), float(df1[2]), float(df1[3]), int(df1[4]), int(df1[5]),
    #      int(df1[6]), int(df1[7]), int(df1[8]), int(df1[9]), int(df1[10]), int(df1[11]), int(df1[12]),
    #      int(df1[13]), int(df1[14]), int(df1[15]), int(df1[16]), int(df1[17]), int(df1[18]), int(df1[19]),
    #      int(df1[20]), int(df1[21]), int(df1[22]), int(df1[23]), int(df1[24]), int(df1[25]), int(df1[26]),
    #      int(df1[27]), int(df1[28]), int(df1[29]), int(df1[30]), int(df1[31]), int(df1[32]), int(df1[33])]

    # Input Verilerinden Tek Satırlık DataFrame Oluşturma

    # l2 = np.array(l1).reshape(1, -1)
    # input_df = pd.DataFrame(l2)
    input_df = input_df.astype(float)
    return input_df.T

left_col, mid1_col, mid2_col,right_col = feedback_tab.columns(4)


left_col.write('Personal Questions')

# Text input for name
otherdata['Name'] = left_col.text_input("What's your name?")

responses['Gender'] = left_col.radio(
    "What is your gender?",
    ('Male', 'Female')
)


user_response = left_col.radio(
    "Do you travel with us often?",
    options=('Yes', 'No')
)

# Map the response directly after getting input
responses['Customer Type'] = 'Loyal Customer' if user_response == 'Yes' else 'disloyal Customer'

responses['Age'] = left_col.number_input("How old are you?" , 1, 120, "min", 1)

responses['Type of Travel'] = left_col.radio(
    "What is purpose of your flight?",
    ('Business travel', 'Personal Travel') , horizontal=True
)

# Radio buttons for satisfaction level
responses['Class'] = left_col.radio(
    "What was the class of your flight?",
    ('Business',  'Eco', 'Eco Plus') , horizontal=True
)


mid1_col.write('Flight related questions')

responses['Flight Distance'] = mid1_col.number_input("What was your flight distance?" , 1, 999999, "min", 1)

# Radio buttons for satisfaction level
responses['Inflight wifi service'] = mid1_col.radio(
    "How satisfied are you with our in flight wi-fi service?",
    (5, 4, 3, 2, 1) , horizontal=True
)

responses['Food and drink'] = mid1_col.radio(
    "How satisfied are you with our food and drink services?",
    (5, 4, 3, 2, 1) , horizontal=True
)

responses['Seat comfort'] = mid1_col.radio(
    "How comfortable was your seat?",
    (5, 4, 3, 2, 1) , horizontal=True
)

responses['Inflight entertainment'] = mid2_col.radio(
    "How satisfied are you with our inflight entertainment services?",
    (5, 4, 3, 2, 1) , horizontal=True
)

responses['On-board service'] = mid2_col.radio(
    "How satisfied are you with our on board services?",
    (5, 4, 3, 2, 1) , horizontal=True
)

responses['Leg room service'] = mid2_col.radio(
    "How satisfied are you with our leg room service?",
    (5, 4, 3, 2, 1) , horizontal=True
)

responses['Inflight service'] = mid2_col.radio(
    "How satisfied are you with our inflight service?",
    (5, 4, 3, 2, 1), horizontal=True
)

responses['Cleanliness'] = mid2_col.radio(
    "How clean was the plane?",
    (5, 4, 3, 2, 1), horizontal=True
)

responses['Departure Delay in Minutes'] = left_col.number_input("Departure Delay?" , 1, 999999, "min", 1)

responses['Arrival Delay in Minutes'] = left_col.number_input("Arrival Delay?" , 1, 999999, "min", 1)


right_col.write('Airport related questions')


# Radio buttons for satisfaction level

responses['Departure/Arrival time convenient'] = right_col.radio(
    "Please rate how convenient you found the departure and arrival times:",
    (5, 4, 3, 2, 1), horizontal=True
)


responses['Gate location'] = right_col.radio(
    "Please rate how convenient you found the gate location",
    (5, 4, 3, 2, 1), horizontal=True
)


responses['Baggage handling'] = right_col.radio(
    "How satisfied are you with our baggage handling?",
    (5, 4, 3, 2, 1), horizontal=True
)

responses['Checkin service'] = right_col.radio(
    "How satisfied are you with our check-in service?",
    (5, 4, 3, 2, 1), horizontal=True
)

right_col.write('Online service related questions')


responses['Online boarding'] = right_col.radio(
    "How easy was the online boarding process?",
    (5, 4, 3, 2, 1), horizontal=True
)

responses['Ease of Online booking'] = right_col.radio(
    "How easy was the online booking process?",
    (5, 4, 3, 2, 1), horizontal=True
)


# Checkbox for whether they would recommend the service
otherdata['Recommend'] = feedback_tab.checkbox('Would you recommend our service to others?')

# Slider for rating the service

# Text area for feedback
otherdata['Feedback'] = feedback_tab.text_area("Additional feedback or suggestions:")

def transpose(dataframe):
    return dataframe.T

columnso = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'satisfaction', 'Total Delay in Minutes', 'Total_Service_Satisfaction_Score', 'Inflight_Service_Satisfaction', 'Online_Services_Satisfaction_Score', 'Airport_Services_Satisfaction_Score', 'Inflight_Service_Satisfaction_Per_Mile', 'Gender_Male', 'Customer Type_disloyal Customer', 'Type of Travel_Personal Travel', 'Class_Eco', 'Class_Eco Plus', 'Age_Group_18-30', 'Age_Group_31-50', 'Age_Group_51-70', 'Age_Group_71+', 'Flight_Distance_Level_Medium Haul', 'Flight_Distance_Level_Long Haul', 'Flight_Distance_Level_Very Long Haul', 'Flight_Distance_Level_Ultra Long Haul', 'Departure_Delay_Status_Minor Delay', 'Departure_Delay_Status_Moderate Delay', 'Departure_Delay_Status_Significant Delay', 'Departure_Delay_Status_Severe Delay', 'Arrival_Delay_Status_Minor Delay', 'Arrival_Delay_Status_Moderate Delay', 'Arrival_Delay_Status_Significant Delay', 'Arrival_Delay_Status_Severe Delay']
othercolumns = pd.DataFrame([columnso])
othercolumns.columns = columnso
# Check if all responses are filled

# Button to submit responses
if feedback_tab.button('Submit Survey'):
    save_responses(responses)
    feedback_tab.success('Thank you for your responses!')

    input_df =  pd.DataFrame([responses])
    input_new = pred_data(df , input_df)
    input_new = pd.DataFrame(input_new)
    tahmin = model.predict_proba(input_new)
    eminlik = np.max(tahmin) * 100
    #input_new = transpose(input_new)
    prediction = model.predict(input_new)



   # feedback_tab.write(input_df)
    #feedback_tab.write(input_new)
   # feedback_tab.write(othercolumns)

    if (prediction == 0):
         feedback_tab.write(eminlik)
         feedback_tab.write('Customer was not satisfied')
         st.image('saddest.webp',width=200 )
    else:
         feedback_tab.write(eminlik)
         feedback_tab.write('Customer was  satisfied')
         st.image('happy.webp', width=200)

    #feedback_tab.write(prediction)
# Run this with:
# streamlit run survey_app.py


