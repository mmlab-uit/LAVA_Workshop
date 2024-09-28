import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import json
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import random
import copy

# Function to extract JSON string from a given string
def get_json_str(string: str) -> str:
    first = string.find('{')
    last = string.rfind('}')
    if first == -1 or last == -1 or first > last:
        raise ValueError("Input string does not contain valid JSON object braces.")
    return f'[{string[first:last + 1]}]'

# Function to process the dataframe and extract information from images and text prompts
def processing(dataframe, image_dir, request_prompt):
    result = {'file_name':[], 'question':[], 'answer':[], 'explanation':[],
          'option1':[], 'option2':[],
         'option3':[], 'option4':[], 'language':[]}
    for index, row in tqdm(dataframe.iterrows(), total = len(dataframe)):
        file_name = row['file_name']
        question = row['question']
        option1 = row['option1']
        option2 = row['option2']
        option3 = row['option3']
        option4 = row['option4']
        language = row['language']
        
        result['file_name'].append(file_name)
        result['question'].append(question)
        result['option1'].append(option1)
        result['option2'].append(option2)
        result['option3'].append(option3)
        result['option4'].append(option4)
        result['language'].append(language)

        # Create the prompt template string
        prompt_template_str = f"""
{request_prompt}
{question}
1 {option1}
2 {option2}
3 {option3}
4 {option4}
        """
        # Load and convert the image from the specified path
        image_path = os.path.join(image_dir,file_name)
        image = Image.open(image_path).convert('RGB')

        # Prepare the message structure for the model
        msgs = [{'role': 'user', 'content': [image, prompt_template_str]}]

        # Get the response from the model using the chat function
        response = pipe.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=tokenizer
                )
            
        # Try to parse the response as JSON and extract the answer and explanation
        try:
            json_out = json.loads(get_json_str(response))
            result['answer'].append(json_out[0]['answer'])
            result['explanation'].append(json_out[0]['explanation'])

        # Handle parsing errors and fallback to manual checking of the response
        except (json.JSONDecodeError, Exception) as e:
            print(f'{response}')
            string = response
            ans = string.find('"answer"')
            exp = string.find('"explanation')
            if "1" in string[ans:exp]:
                result['answer'].append("1")
                print('The answer is 1')
            elif "2" in string[ans:exp]:
                result['answer'].append("2")
                print('The answer is 2')
            elif "3" in string[ans:exp]:
                result['answer'].append("3")
                print('The answer is 3')
            elif "4" in string[ans:exp]:
                result['answer'].append("4")
                print('The answer is 4')
            else:
                result['answer'].append(None)
                print('The answer is not valid')
            result['explanation'].append(response)
    return result

# Load the pre-trained model with specific configurations
pipe = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16)

# Set the model to evaluation mode and move it to GPU
pipe = pipe.eval().cuda()

# Load the tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)


# Define paths for public and private datasets and annotations
public_data_directory_path = 'LAVA_Challenge_Data/Public data/images'
private_data_directory_path = 'LAVA_Challenge_Data/Private data/images'
public_annotation_path = 'LAVA_Challenge_Data/Public data/annotation.csv'
private_annotation_path = 'LAVA_Challenge_Data/Private data/annotation.csv'

# Read public and private annotation data into dataframes
public_annotation_df = pd.read_csv(public_annotation_path)
private_annotation_df = pd.read_csv(private_annotation_path)

# Define the prompt used for request
request_prompt = '''Imagine you are an expert at English and Japanese with good knowledge.
Please provide the answer with ONE number from 1-4.
You MUST give a clear and concise reason/explanation for your choice.
The output MUST be json format. Use the following JSON format:
{
  "answer": "number",
  "explanation": "<text>"
}
The following is the question and 4 choices:'''

# Process private and public datasets using the defined function
private_detail_result_dict = processing(private_annotation_df,private_data_directory_path, request_prompt)
public_detail_result_dict = processing(public_annotation_df,public_data_directory_path, request_prompt)

# Create deep copies of the results for further processing
private_result_dict = copy.deepcopy(private_detail_result_dict)
public_result_dict = copy.deepcopy(public_detail_result_dict)

# Reduce results to only necessary keys (file name and answer)
private_result_dict = {key: private_result_dict[key] for key in ['file_name','answer']}
public_result_dict = {key: public_result_dict[key] for key in ['file_name','answer']}

# Convert detailed results to DataFrames and save as Excel files
private_detail_result_dict_df = pd.DataFrame.from_dict(private_detail_result_dict)
public_detail_result_dict_df = pd.DataFrame.from_dict(public_detail_result_dict)
public_detail_result_dict_df.to_excel('public_detail_result_dict.xlsx')
private_detail_result_dict_df.to_excel('private_detail_result_dict.xlsx')

# Randomly assign answers where the result is not valid (not 1, 2, 3, or 4)
for idx,value in enumerate(private_result_dict['answer']):
    if value not in ['1','2','3','4']:
        private_result_dict['answer'][idx] = random.choice(['1','2','3','4'])

for idx,value in enumerate(public_result_dict['answer']):
    if value not in ['1','2','3','4']:
        public_result_dict['answer'][idx] = random.choice(['1','2','3','4'])

# Convert the result dictionaries to DataFrames and save as CSV files
private_result_dict_df = pd.DataFrame.from_dict(private_result_dict).set_index('file_name')
public_result_dict_df = pd.DataFrame.from_dict(public_result_dict).set_index('file_name')
public_result_dict_df.to_csv('public.csv')
private_result_dict_df.to_csv('private.csv')