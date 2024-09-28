import os
import nest_asyncio
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
from lmdeploy.vl import load_image
import pandas as pd
import json
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import random
import copy

# Function to extract a JSON string from a given string by finding the first and last braces
def get_json_str(string: str) -> str:
    first = string.find('{')
    last = string.rfind('}')
    if first == -1 or last == -1 or first > last:
        raise ValueError("Input string does not contain valid JSON object braces.")
    return f'[{string[first:last + 1]}]'

# Function to process the dataframe and generate answers using images and text prompts
def processing(dataframe, image_dir, request_prompt):
    # Define generation configuration for the model
    gen_config = GenerationConfig(temperature=1.0, max_new_tokens=2048)

    # Initialize a dictionary to store results
    result = {'file_name':[], 'question':[], 'answer':[], 'explanation':[],
          'option1':[], 'option2':[],
         'option3':[], 'option4':[], 'language':[]}

    # Iterate through each row in the dataframe
    for index, row in tqdm(dataframe.iterrows(), total = len(dataframe)):
        # Extract data from each row
        file_name = row['file_name']
        question = row['question']
        option1 = row['option1']
        option2 = row['option2']
        option3 = row['option3']
        option4 = row['option4']
        language = row['language']
        
        # Append extracted data to the result dictionary
        result['file_name'].append(file_name)
        result['question'].append(question)
        result['option1'].append(option1)
        result['option2'].append(option2)
        result['option3'].append(option3)
        result['option4'].append(option4)
        result['language'].append(language)

        # Create a formatted string for the model prompt
        prompt_template_str = f"""
{request_prompt}
{question}
1 {option1}
2 {option2}
3 {option3}
4 {option4}
        """

        # Load the image from the specified path
        image_path = os.path.join(image_dir,file_name)
        img = load_image(image_path)

        # Get the response from the model using the prompt and the image
        response = pipe((prompt_template_str, img), gen_config=gen_config)

        # Attempt to parse the model's response as JSON
        try:
            json_out = json.loads(get_json_str(response.text))
            result['answer'].append(json_out[0]['answer'])
            result['explanation'].append(json_out[0]['explanation'])
        
        # Handle JSON decoding errors and retry formatting the response
        except (json.JSONDecodeError, Exception) as e:
            try:
                retry_response = pipe((f'''
                To ensure proper JSON schema formatting for input to a large language model, follow these rules: use double quotes for all keys and string values, escape any double quotes within string values with a backslash (\), separate key-value pairs with commas, enclose objects in curly braces (({{}})), and arrays in square brackets ([]). Ensure all keys are unique within the same object, values can be strings, numbers, objects, arrays, true, false, or null. Maintain proper nesting and closure of braces and brackets. Avoid trailing commas after the last key-value pair or array item. Use UTF-8 encoding and ensure the entire JSON is a single valid structure without extraneous characters. Validate the schema using a JSON schema validator to catch errors like unescaped quotes or mismatched braces.
                The following JSON string is invalid. Fix it. {e}
                {response.text}
                '''), gen_config=gen_config)
                
                json_out = json.loads(get_json_str(retry_response.text))
                result['answer'].append(json_out[0]['answer'])
                result['explanation'].append(json_out[0]['explanation'])

            # Handle further parsing errors by manually checking the response for answers
            except (json.JSONDecodeError, Exception) as e:
                print(f'{response.text}')
                string = response.text
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
                result['explanation'].append(response.text)
    return result

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Model and configuration details
model: str = 'OpenGVLab/InternVL2-Llama3-76B'
#system_prompt: str = 'You are a helpful teacher who only generates JSON responses to requests.'
system_prompt: str = 'You are a knowledgeable and helpful expert who generates detailed JSON responses for provided images and questions.'

# Initialize chat template configuration
chat_template_config = ChatTemplateConfig('internvl-internlm2')
chat_template_config.meta_instruction = system_prompt

# Initialize backend configuration
backend_config = TurbomindEngineConfig(tp=4,model_format='hf')

# Create the pipeline with the provided configurations
pipe = pipeline(model, chat_template_config=chat_template_config, backend_config=backend_config)

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