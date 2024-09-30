import google.generativeai as genai
from PIL import Image
import os
from typing_extensions import TypedDict
import enum
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import json
import time
import pickle

# Enum to represent the four possible choices for answers
class Choice(enum.Enum):
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"

# Define the response template structure expected from the generative AI model
class Response_Template(TypedDict):
    answer: Choice
    explanation: str

# Function to extract JSON string from a larger string containing it
def get_json_str(string: str) -> str:
    first = string.find('{')
    last = string.rfind('}')
    if first == -1 or last == -1 or first > last:
        raise ValueError("Input string does not contain valid JSON object braces.")
    return f'[{string[first:last + 1]}]'

# Main function to process the data and interact with the AI model
def processing(dataframe, image_dir, request_prompt, retries = 10, delay=3):
    result = {'file_name':[], 'question':[], 'answer':[], 'explanation':[],
          'option1':[], 'option2':[],
         'option3':[], 'option4':[], 'language':[]}
    for index, row in tqdm(dataframe.iterrows(), total = len(dataframe)):
        # Extract relevant fields from each row
        file_name = row['file_name']
        question = row['question']
        option1 = row['option1']
        option2 = row['option2']
        option3 = row['option3']
        option4 = row['option4']
        language = row['language']
        
        # Append these values to the result dictionary
        result['file_name'].append(file_name)
        result['question'].append(question)
        result['option1'].append(option1)
        result['option2'].append(option2)
        result['option3'].append(option3)
        result['option4'].append(option4)
        result['language'].append(language)

        # Create the prompt string with the question and options
        prompt_template_str = f"""
{request_prompt}
{question}
1 {option1}
2 {option2}
3 {option3}
4 {option4}
        """
        # Open the image file corresponding to the current row
        image_path = os.path.join(image_dir,file_name)
        image = Image.open(image_path).convert('RGB')

        # Attempt to generate a response using the generative AI model with retries
        for attempt in range(retries):
            try:
                # Request content generation from the model using the prompt and image
                response = model.generate_content(
                [prompt_template_str,image],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json", response_schema=Response_Template
                ),
            )
                break   # Exit the retry loop on success
            except Exception as e:
                print(f"Attempt {attempt+1} failed with error: {e}")
                # Retry after a delay if not the last attempt
                if attempt < retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    # Raise exception if out of quota
                    if 'quota' in str(e):
                        raise Exception("OUT OF QUOTA")

        # Try parsing the JSON response from the model
        try:
            json_out = json.loads(get_json_str(response.text))
            result['answer'].append(json_out[0]['answer'])
            result['explanation'].append(json_out[0]['explanation'])
        except (json.JSONDecodeError, Exception) as e:
            # Handle parsing errors and determine the answer based on the response text
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

# Configure safety settings for the generative AI model
safe = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

# Set up generative AI configuration with the API key and model settings
genai.configure(api_key="YOUR_GOOGLE_API_KEY")
model = genai.GenerativeModel(model_name="gemini-1.5-flash", safety_settings=safe)

# Define paths for data directories and annotation files
public_data_directory_path = 'LAVA_Challenge_Data/Public data/images'
private_data_directory_path = 'LAVA_Challenge_Data/Private data/images'
public_annotation_path = 'LAVA_Challenge_Data/Public data/annotation.csv'
private_annotation_path = 'LAVA_Challenge_Data/Private data/annotation.csv'

# Load annotation data into dataframes
public_annotation_df = pd.read_csv(public_annotation_path)
private_annotation_df = pd.read_csv(private_annotation_path)

# Define the prompt template for the AI model
request_prompt = '''Imagine you are an expert at English and Japanese with good knowledge.
Please provide the answer with ONE number from 1-4.
You MUST give a clear and concise reason/explanation for your choice.
The output MUST be json format. Use the following JSON format:
{
  "answer": "number",
  "explanation": "<text>"
}
The following is the question and 4 choices:'''

# Process the private and public datasets using the AI model
private_detail_result_dict = processing(private_annotation_df,private_data_directory_path, request_prompt)
public_detail_result_dict = processing(public_annotation_df,public_data_directory_path, request_prompt)

# Create deep copies of the detailed results for further processing
private_result_dict = copy.deepcopy(private_detail_result_dict)
public_result_dict = copy.deepcopy(public_detail_result_dict)

# Reduce results to only necessary keys
private_result_dict = {key: private_result_dict[key] for key in ['file_name','answer']}
public_result_dict = {key: public_result_dict[key] for key in ['file_name','answer']}

# Convert detailed results to DataFrames and save as Excel files
private_detail_result_dict_df = pd.DataFrame.from_dict(private_detail_result_dict)
public_detail_result_dict_df = pd.DataFrame.from_dict(public_detail_result_dict)
public_detail_result_dict_df.to_excel('public_detail_result_dict.xlsx')
private_detail_result_dict_df.to_excel('private_detail_result_dict.xlsx')

# Randomly assign answers where the result is not valid
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