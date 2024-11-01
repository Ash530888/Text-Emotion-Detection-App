'''
Performs API Call to IBM's Watson Machine Learning
Params:
text: text to analyse
return: JSON object from API call's result i.e. emotion scores
'''

import requests
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions

def emotion_detector(text):
    # Get Watson API credentials
    credentials = {}

    with open('emotion_detection/ibm-credentials.txt', 'r') as file:
        # Read the contents of the file
        for line in file:
            key, val = line.split("=")
            credentials[key] = val[:len(val)-1] # removing \n

    print(credentials)

    authenticator = IAMAuthenticator(credentials["NATURAL_LANGUAGE_UNDERSTANDING_APIKEY"])
    nlu = NaturalLanguageUnderstandingV1(
        version='2020-08-01',
        authenticator=authenticator
    )

    nlu.set_service_url(credentials["NATURAL_LANGUAGE_UNDERSTANDING_URL"])


    """Function to send text to Watson NLP API and retrieve emotion data."""
    data = {
        'text': text,
        'features': {
            'emotion': {}
        }
        ,
        "keywords": {
        "emotion": True
        }
    }

    try:
        print(text)
        response = nlu.analyze(text = text, features=Features(emotion=EmotionOptions())).get_result()
        
        response = response['emotion']['document']['emotion']
        print(type(response))

        dominant_emotion = "sadness"
        val = response["sadness"]

        for emotion in response:
            if response[emotion] > val:
                dominant_emotion = emotion
                val = response[emotion]

        response['dominant_emotion'] = dominant_emotion

        return response

        '''
        # Sending POST request to Watson NLP API
        response = requests.post(
            credentials["NATURAL_LANGUAGE_UNDERSTANDING_URL"] + "/v1/analyze?version=2019-07-12",
            headers={
                'Content-Type': 'application/json'
            },
            auth=("apikey", credentials["NATURAL_LANGUAGE_UNDERSTANDING_APIKEY"]),
            json=data
        )

        # Check if the response is successful
        if response.status_code == 200:
            return response.json()['emotion']['document']['emotion']
        else:
            return {'error': 'Could not analyze emotions'}
        '''
    except Exception as e:
        print(f"Error: An exception occurred - {e}")
        return None
