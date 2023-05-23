import streamlit as st
import pandas as pd
import pickle


file_path = '/app/online_shoppers_model.pkl'

# Load the pickled model
model = pickle.load(open(file_path, 'rb'))

# Define the loan eligibility prediction function


def online_shoppers_intention(data):
    # Perform the loan eligibility prediction using the loaded model
    prediction = model.predict(data)
    return prediction


def main():
    # Set the page title
    st.title('ONLINE SHOPPERS INTENTION')

    # Create input fields for loan details
    administrative = st.slider('Administrative', 0, 27)
    administrative_duration = st.slider('Administrative Duration', 0, 3500)
    informational = st.slider('Informational', 0, 24)
    informational_duration = st.slider('Informational Duration', 0, 2600)
    product_related = st.slider('Product Related', 0, 705)
    product_related_duration = st.slider('Product Related Duration', 0, 63974)
    bounce_rates = st.slider('Bounce Rates', 0.0, 0.2)
    exit_rates = st.slider('Exit Rates', 0.0, 0.2)
    page_values = st.slider('Page Values', 0, 365)
    special_day = st.selectbox('Special Day', [0, 1])
    month = st.selectbox('Month', ['February', 'March', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December'])

    if month == 'February':
        month = 2
    elif month == 'March':
        month = 3
    elif month == 'May':
        month = 5
    elif month == 'June':
        month = 6
    elif month == 'July':
        month = 7
    elif month == 'August':
        month = 8
    elif month == 'September':
        month = 9
    elif month == 'October':
        month = 10
    elif month == 'November':
        month = 11
    else:
        month = 12

    operating_systems = st.slider('Operating Systems', 1, 8)
    browser = st.slider('Browser', 1, 13)
    region = st.slider('Region', 1, 9)
    traffic_type = st.slider('Traffic Type', 1, 20)
    visitor_type = st.selectbox('Visitor Type', [0, 1, 2])
    weekend = st.selectbox('Weekend', [0, 1])
    if weekend == 1:
        weekend = True
    else:
        weekend = False

    # Prepare the loan data as input for prediction
    data = pd.DataFrame({
        'Administrative': [administrative],
        'Administrative_Duration': [administrative_duration],
        'Informational': [informational],
        'Informational_Duration': [informational_duration],
        'ProductRelated': [product_related],
        'ProductRelated_Duration': [product_related_duration],
        'BounceRates': [bounce_rates],
        'ExitRates': [exit_rates],
        'PageValues': [page_values],
        'SpecialDay': [special_day],
        'Month': [month],
        'OperatingSystems': [operating_systems],
        'Browser': [browser],
        'Region': [region],
        'TrafficType': [traffic_type],
        'VisitorType': [visitor_type],
        'Weekend': [weekend]
         })

    # Form submission
    if st.button('Predict'):
        # Perform Online Shoppers Intention
        prediction = online_shoppers_intention(data)

        # Display the prediction result
        if prediction[0] == 1 and weekend == True:
            st.success('Yes, Revenue Generation on Weekend.')
        else:
            st.warning('No Revenue Generation on Weekend.')

        st.write("Informational_Duration: =>", 'Time spent to in a specific category of ITEMS: ', informational_duration/100, 'Minutes')


if __name__ == '__main__':
    main()
