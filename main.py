import streamlit as st
import speech_recognition as sr
import requests
import pygame
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import numpy as np
import json
import openai



# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = []


prev_history = []
openai.api_key = 'Your_open_ai_key'

def get_context(inputPrompt,top_k):
    # openai.api_key = apiKey
    search_term_vector = get_embedding(inputPrompt,engine='text-embedding-ada-002')
    
    with open("knowledge_base.json",encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
        for item in data:
            item['embeddings'] = np.array(item['embeddings'])

        for item in data:
            item['similarities'] = cosine_similarity(item['embeddings'], search_term_vector)

        sorted_data = sorted(data, key=lambda x: x['similarities'], reverse=True)
        context = ''
        referencs = []
        for i in sorted_data[:top_k]:
            context += i['chunk'] + '\n'
            # referencs.append({"pdf_name":i['pdf_name'],"page_num":i['page_num']})
    return context

def get_answer(user_input):


    context = get_context(user_input,3)

    prompt = "context:\n\n{}.\n\n Answer the following user query according to above given context:\nuser_input: {}".format(context,user_input)

    myMessages = []
    myMessages.append({"role": "system", "content": "You are expert article generator"})
    

    myMessages.append({"role": "user", "content": "context:\n\n{}.\n\n Answer the following user query according to above given context:\nuser_input: {}".format(context,user_input)})

    
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        # model='gpt-4',
        messages=myMessages,
        max_tokens=None,
        stream=False
    )
    
    return response['choices'][0]['message']['content']





def get_answer_using_function_call(user_input,prev_history):
    # print("prev_history::: ",prev_history)
    messages = []
    for i in prev_history:
        if i['ai'] != 'null': 
            messages.append({"role": "user", "content": i['user']})
            messages.append({"role": "assistant", "content": i['ai']})

    messages.append({"role": "user", "content": user_input})

    functions = [
        {
            "name": "get_answer",
            "description": "Get Answer to any query to which you don't know the answer your self. You will also call this function whenever the user ask some query or question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "A User query with complete intention as per the conversation history.",
                    },
                },
                "required": ["user_input"],
            },
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]
    print("response_message: ",response_message)

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_answer": get_answer,
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        user_input=function_args.get("user_input")

        print("generated_query: ",user_input)
        res = get_answer(user_input)
        return res
    else:
        return response_message['content']








# a little interface

def main():
    st.title("Voice to voice test")
    is_listening = False

    # Create a button to trigger the speech recognition process
    start_button_key = "start_button"
    if st.button("Start listening", key=start_button_key) and not is_listening:
        is_listening = True
        while is_listening:
            user_input = capture_audio_and_convert_to_text()
            st.write(f"You said: {user_input}")

            if user_input:
                st.write("Recognizing")
                
                say(generate_response(user_input))
               

        st.write("Listening again... Say something, or click 'Stop Listening' to exit.")
        stop_button_key = "stop_button"
        stop_button = st.button("Stop Listening", key=stop_button_key)
        if stop_button:
            is_listening = False
 
#function to get the audio and convert it to text

def capture_audio_and_convert_to_text():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Listening... Say something!")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Sorry, could not understand what you said.")
    except sr.RequestError as e:
        st.error(f"Error occurred during speech recognition: {e}")
    return None


# Function for text-to-speech



def say(text):
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/voice_id"

    headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": "Your_eleven_labs_key"
    }

    data = {
    "text": text,
    "model_id": "eleven_monolingual_v1",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.5
    }
    }

    response = requests.post(url, json=data, headers=headers)
    with open('output.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)


    pygame.mixer.init()  # Initialize the mixer module (without initializing the whole pygame)
    pygame.mixer.music.load('output.mp3')
    pygame.mixer.music.play()

    # Allow the audio to play
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Lower the tick value for lower CPU usage during the loop

    # Stop and quit the mixer
    pygame.mixer.music.stop()
    pygame.mixer.quit()



#function to generate response

def generate_response(user_input):
    for i in range(len(st.session_state['past'])):
        try:
            prev_history.append({"user":st.session_state['past'][i],"ai":st.session_state["generated"][i]})
        except:
            prev_history.append({"user":st.session_state['past'][i],"ai":"null"})


    answer = get_answer_using_function_call(user_input,prev_history)
    
    return answer

#run the app

if __name__ == "__main__":
    main()
