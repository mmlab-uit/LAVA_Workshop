import base64
import requests
from pydantic import BaseModel
from openai import OpenAI
import os
import pandas as pd
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import random
import copy
import pickle
import time

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"

# Initialize the OpenAI client
client = OpenAI()

# Define a response model for OpenAI's parsed response
class Response(BaseModel):
    answer: int
    explanation: str

# Function to encode an image to base64 format
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Main processing function
def processing(dataframe, image_dir, request_prompt, retries = 3, delay=60):
    # Initialize an empty result dictionary
    result = {'file_name':[], 'question':[], 'answer':[], 'explanation':[],
            'option1':[], 'option2':[],
          'option3':[], 'option4':[], 'language':[]}

    # Iterate through each row in the dataframe
    for index, row in tqdm(dataframe.iterrows(), total = len(dataframe)):
      # Extract necessary fields from the row
      file_name = row['file_name']
      question = row['question']
      option1 = row['option1']
      option2 = row['option2']
      option3 = row['option3']
      option4 = row['option4']
      language = row['language']
      
      # Append the extracted data to the result dictionary
      result['file_name'].append(file_name)
      result['question'].append(question)
      result['option1'].append(option1)
      result['option2'].append(option2)
      result['option3'].append(option3)
      result['option4'].append(option4)
      result['language'].append(language)

      # Prepare the prompt with the question and options
      prompt_template_str = f"""
{request_prompt}
{question}
1 {option1}
2 {option2}
3 {option3}
4 {option4}
      """

      # Get the image path and encode it as base64
      image_path = os.path.join(image_dir,file_name)
      base64_image = encode_image(image_path)

      # Try sending the request to OpenAI with retries in case of failure
      for attempt in range(retries):
        try:
          # Send the prompt to OpenAI's model
          completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
            {"role": "system",
            "content": "You are a visual question answering expert. You will be given an image and a question consisting of 4 options. Please answer the question and convert it into the given structure."
            },
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": f"{prompt_template_str}"
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                  }
                }
              ]
            }
            ],
            response_format=Response,
            max_tokens = 2048
        )
          # Parse the response and append the result
          event = completion.choices[0].message
          if event.parsed:
            print(event.parsed)
            parsed_event = event.parsed
            result['answer'].append(parsed_event.answer)
            result['explanation'].append(parsed_event.explanation)
          elif event.refusal:
            result['answer'].append(None)
            result['explanation'].append(None)
            print(event.refusal)
          else:
            print("FAIL!")
            result['answer'].append(None)
            result['explanation'].append(None)
          break
        except Exception as e:
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("FAIL!")
                result['answer'].append(None)
                result['explanation'].append(None)
    return result

# Prompt template used for generating responses
request_prompt = '''Imagine you are an expert at English and Japanese with good knowledge.
You will be given an image and a question consisting of 4 options.
Please provide the answer with ONE number from 1-4.
You MUST give a clear and concise reason/explanation for your choice.
The output MUST be json format. Use the following JSON format:
{
  "answer": "number",
  "explanation": "<text>"
}
The following is the question and 4 choices:'''

# Define paths for image directories and annotation files
public_data_directory_path = 'LAVA_Challenge_Data/Public data/images'
private_data_directory_path = 'LAVA_Challenge_Data/Private data/images'
public_annotation_path = 'LAVA_Challenge_Data/Public data/annotation.csv'
private_annotation_path = 'LAVA_Challenge_Data/Private data/annotation.csv'

# Read annotations into dataframes
public_annotation_df = pd.read_csv(public_annotation_path)
private_annotation_df = pd.read_csv(private_annotation_path)

# Process private and public datasets
private_detail_result_dict = processing(private_annotation_df,private_data_directory_path, request_prompt, 'private_results.pkl')
public_detail_result_dict = processing(public_annotation_df,public_data_directory_path, request_prompt, 'public_results.pkl')

# Deep copy results for further processing
private_result_dict = copy.deepcopy(private_detail_result_dict)
public_result_dict = copy.deepcopy(public_detail_result_dict)

# Filter only required keys from the result dictionaries
private_result_dict = {key: private_result_dict[key] for key in ['file_name','answer']}
public_result_dict = {key: public_result_dict[key] for key in ['file_name','answer']}

# Convert detailed results to dataframes and export to Excel
private_detail_result_dict_df = pd.DataFrame.from_dict(private_detail_result_dict)
public_detail_result_dict_df = pd.DataFrame.from_dict(public_detail_result_dict)
public_detail_result_dict_df.to_excel('public_detail_result_dict.xlsx')
private_detail_result_dict_df.to_excel('private_detail_result_dict.xlsx')

# Correct invalid answers by randomly assigning valid options (1-4)
for idx,value in enumerate(private_result_dict['answer']):
    if value not in [1,2,3,4]:
        print(idx)
        private_result_dict['answer'][idx] = random.choice(['1','2','3','4'])

for idx,value in enumerate(public_result_dict['answer']):
    if value not in [1,2,3,4]:
        print(idx)
        public_result_dict['answer'][idx] = random.choice(['1','2','3','4'])

# Convert final results to dataframes and save as CSV
private_result_dict_df = pd.DataFrame.from_dict(private_result_dict).set_index('file_name')
public_result_dict_df = pd.DataFrame.from_dict(public_result_dict).set_index('file_name')
public_result_dict_df.to_csv('public.csv')
private_result_dict_df.to_csv('private.csv')