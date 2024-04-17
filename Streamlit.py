import streamlit as st
import joblib
import pandas as pd

from streamlit.components.v1 import html

st.set_page_config(page_title="Airline Passenger Satisfaction", page_icon=":airplane", layout="wide")



# URL or local path to your background image


model = joblib.load('kaan.pkl')

# Function to save responses (this example just prints them to the console)
def save_responses(responses):
    print("Responses received:")
    for question, response in responses.items():
        print(f"{question}: {response}")



st.title("Airline Passenger Satisfaction")

left_col, right_col = st.columns(2)

left_col.write("""This app analyzes and visualizes the airline passenger satisfaction dataset. Discover the factors that affect passengers' satisfaction levels and predict the satisfaction of flights.""")
#right_col.image("Logo.webp")
# Title and introduction of the survey
st.title('Customer Satisfaction Survey')
st.write('We appreciate your time to help us improve our services.')





# Questions
responses = {}
otherdata = {}

# Text input for name
otherdata['Name'] = st.text_input("What's your name?")

responses['Gender'] = st.radio(
    "What is your gender?",
    ('Male' , 'Female')
)



responses['Customer Type'] = st.radio(
    "Do you travel with us often?",
    ('Yes' , 'No')
)

responses['Age'] = st.text_input("How old are you?")

responses['Type Of Travel'] = st.radio(
    "What is purpose of your flight?",
    ('Business travel' , 'Personal travel')
)

# Radio buttons for satisfaction level
responses['Class'] = st.radio(
    "What was the class of your flight?",
    ('Business',  'Eco', 'Eco Plus')
)

responses['Flight Distance'] = st.text_input("What was your flight distance?")

# Radio buttons for satisfaction level
responses['Inflight wifi service'] = st.radio(
    "How satisfied are you with our in flight wi-fi service?",
    (5, 4, 3, 2, 1)
)

# Radio buttons for satisfaction level
responses['Departure/Arrival time convenient'] = st.radio(
    "Please rate how convenient you found the departure and arrival times:",
    (5, 4, 3, 2, 1)
)

responses['Ease of Online booking'] = st.radio(
    "How easy was the online booking process?",
    (5, 4, 3, 2, 1)
)

responses['Gate location'] = st.radio(
    "Please rate how convenient you found the gate location",
    (5, 4, 3, 2, 1)
)

responses['Food and drink'] = st.radio(
    "How satisfied are you with our food and drink services?",
    (5, 4, 3, 2, 1)
)

responses['Online boarding'] = st.radio(
    "How easy was the online boarding process?",
    (5, 4, 3, 2, 1)
)

responses['Seat Comfort'] = st.radio(
    "How comfortable was your seat?",
    (5, 4, 3, 2, 1)
)

responses['Inflight entertainment'] = st.radio(
    "How satisfied are you with our inflight entertainment services?",
    (5, 4, 3, 2, 1)
)

responses['On-board services'] = st.radio(
    "How satisfied are you with our on board services?",
    (5, 4, 3, 2, 1)
)

responses['Leg room service'] = st.radio(
    "How satisfied are you with our leg room service?",
    (5, 4, 3, 2, 1)
)

responses['Baggage handling'] = st.radio(
    "How satisfied are you with our baggage handling?",
    (5, 4, 3, 2, 1)
)

responses['Checkin service'] = st.radio(
    "How satisfied are you with our check-in service?",
    (5, 4, 3, 2, 1)
)

responses['Inflight service'] = st.radio(
    "How satisfied are you with our inflight service?",
    (5, 4, 3, 2, 1)
)

responses['Cleanliness'] = st.radio(
    "How clean was the plane?",
    (5, 4, 3, 2, 1)
)


# Checkbox for whether they would recommend the service
otherdata['Recommend'] = st.checkbox('Would you recommend our service to others?')

# Slider for rating the service

# Text area for feedback
otherdata['Feedback'] = st.text_area("Additional feedback or suggestions:")



# Check if all responses are filled

# Button to submit responses
if st.button('Submit Survey'):
    save_responses(responses)
    st.success('Thank you for your responses!')
    print(responses)
    print('hehe')
    st.write(responses)
# Run this with:
# streamlit run survey_app.py



