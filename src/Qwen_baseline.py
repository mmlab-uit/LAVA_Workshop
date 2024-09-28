from PIL import Image
import requests
import torch
from torchvision import io
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import pandas as pd
import json
from tqdm import tqdm
import random
import copy
import os

# Load the model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
)
# Set minimum and maximum pixel values for processing images
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

# Load the processor with the defined pixel constraints
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

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

        # Define the conversation structure
        conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": f"{prompt_template_str}."},
        ],
    }
]
        # Apply chat template and prepare the input for the model
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        image_path = os.path.join(image_dir,file_name)
        image = Image.open(image_path).convert('RGB')
        inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        output_ids = model.generate(**inputs, max_new_tokens=1024)

        # Generate the output using the model
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        # Decode the generated output
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Attempt to parse the output as JSON and extract answer and explanation
        try:
            json_out = json.loads(get_json_str(output_text))
            result['answer'].append(json_out[0]['answer'])
            result['explanation'].append(json_out[0]['explanation'])
        except (json.JSONDecodeError, Exception) as e:

            # Retry response generation with JSON formatting correction instructions
            try:
                retry_response = pipe((f'''
                To ensure proper JSON schema formatting for input to a large language model, follow these rules: use double quotes for all keys and string values, escape any double quotes within string values with a backslash (\), separate key-value pairs with commas, enclose objects in curly braces (({{}})), and arrays in square brackets ([]). Ensure all keys are unique within the same object, values can be strings, numbers, objects, arrays, true, false, or null. Maintain proper nesting and closure of braces and brackets. Avoid trailing commas after the last key-value pair or array item. Use UTF-8 encoding and ensure the entire JSON is a single valid structure without extraneous characters. Validate the schema using a JSON schema validator to catch errors like unescaped quotes or mismatched braces.
                The following JSON string is invalid. Fix it. {e}
                {output_text}
                '''), gen_config=gen_config)
                
                json_out = json.loads(get_json_str(output_text))
                result['answer'].append(json_out[0]['answer'])
                result['explanation'].append(json_out[0]['explanation'])

            # Handle further parsing failures by manually checking output for answer options
            except (json.JSONDecodeError, Exception) as e:
                print(f'{output_text}')
                string = output_text
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
                result['explanation'].append(output_text)
    return result

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

# Process private and public datasets
private_detail_result_dict = processing(private_annotation_df,private_data_directory_path, request_prompt,'private_results.pkl')
public_detail_result_dict = processing(public_annotation_df,public_data_directory_path, request_prompt, 'public_results.pkl')

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