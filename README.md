# 🔥 LAVA Workshop Challenge Solution
Welcome to the solution repository for the **LAVA Workshop** challenge! This repository contains the solution code for the LAVA Challenge (ACCV Workshop)

## 📚 Challenge Overview
The primary goal of LAVA challenge is to advance the capability of Large Vision-Language Models to accurately interpret and understand complex visual data such as Data Flow Diagrams (DFDs), Class Diagrams, Gantt Charts, and Building Design Drawing.

For more details, visit the [LAVA Workshop Website](https://lava-workshop.github.io/).
## 🛠️ Solution Overview

We evaluated the data using pre-trained vision-language models in a zero-shot fashion. We used several models, including both open-source and proprietary models. We tested several models, including both open-source and proprietary models. The results of our experiments are summarized in the table below:

| NO | Model                | Public Score |
|----|----------------------|--------------|
| 1  | ChatGPT-4o-mini      | 0.71         |
| 2  | Gemeni-1.5 Flash     | 0.71         |
| 3  | MiniCPM-V-2.6        | 0.67         |
| 4  | InternVL2-8B         | 0.70         |
| 5  | InternVL2-26B        | 0.73         |
| 6  | InternVL2-40B        | 0.77         |
| 7  | InternVL2-Llama3-76B | 0.76         |
| 8  | **QwenVL2-76B**      | **0.83**     |


## 🗂️ Repository Structure
Here's an overview of the files and directories in this repository:
```
📦 LAVA_Workshop
├── 📁 LAVA_Challenge_Data # LAVA Challenge Dataset
├── 📁 Results             # Our results
│   ├── <result folders>
├── 📁 src                 # Main source code
│   ├── Gemeni_baseline.py
│   ├── InternVL2_baseline.py
│   ├── MiniCPM_baseline.py
│   └── Qwen_baseline.py           
├── environment.yml       # List of dependencies
└── README.md              # You are here!
```
## 🚀 Getting Started
Follow these instructions to set up and run the project on your local machine.
### Prerequisites
Before you begin, ensure you have the following installed:
- [Python 3.x](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
### Installation
1. **Clone the repository**
 ```bash
 git clone https://github.com/yourusername/lava-workshop-solution.git
```
2. **Navigate to the project directory**:
```bash
cd LAVA_Workshop
```
3. **Install dependencies**
```bash
conda env create -f environment.yml
conda activate LAVA
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

4. **Data Download and Setup**

You can download the data [here](https://drive.google.com/drive/folders/1YBH2FyYfRpppI-hhJTSVgqNt_uBQjY6E). Please carefully read the [Terms and Conditions](https://docs.google.com/document/d/1VvKlzi6KfpaYuN_YuhQGzphRuXyJlOM_/edit?rtpof=true&sd=true) for further information about the license, data. After downloading the data, set up the data in the **LAVA_Challenge_Data** folder according to the structure below:
```
📁 LAVA_Challenge_Data
├── 📁 Private data
│   ├── 📁 images
│   │   └── <private_image_files_here>
│   └── annotation.csv
├── 📁 Public data
│   ├── 📁 images
│   │   └── <public_image_files_here>
│   └── annotation.csv
```
### 🛠️Running the Solutions
All baseline code, when run, will generate four result files:
```
📦 LAVA_Workshop
├── ...
├── private_detail_result_dict.xlsx
├── private.csv
├── public_detail_result_dict.xlsx
├── public.csv
└── ... 
```
You can get the submission file by zipping **'public.csv'** and **'private.csv'**.
#### QwenV2
You can directly run Qwen2-VL-72B-Instruct model by using the following command:
```
python src/Qwen_baseline.py
```
You also run the other version of QwenV2 by modify the code line:
```
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
)
```
For example, you want to run Qwen2-VL-7B-Instruct model, you can modify the above code with:
```
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
```
You can find the others version of Qwen2-VL from [Hugging Face](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)

#### InternVL2
Our baseline code is use InternVL2-Llama3-76B model. You can directly run this model by using the following command:
```
python InternVL2_baseline.py
```
You can find other versions of InternVL2 on [Hugging Face](https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e) and modify the below code line to load the model.
```
model: str = 'OpenGVLab/InternVL2-Llama3-76B'
```

#### MiniCPM-V-2.6
You can directly run MiniCPM-V-2_6 model with 8 bilion parameters using the following command:
```
python MiniCPM_baseline.py
```

#### Gemeni
To run the Gemeni baseline code, you should have Google API key and modify the following line of code in **src/Gemeni_baseline.py** with your API key.
```
# Set up generative AI configuration with the API key and model settings
genai.configure(api_key="YOUR_GOOGLE_API_KEY")
model = genai.GenerativeModel(model_name="gemini-1.5-flash", safety_settings=safe)
```
You can also change the **model_name** to an [available model](https://ai.google.dev/gemini-api/docs/models/gemini) provided by Google.

#### GPT
To run the GPT baseline code, you should have OpenAI API key and modify the following line of code in **src/GPT-baseline.py** with your API key.
```
completion = client.beta.chat.completions.parse(
model="gpt-4o-mini",
...
```
You can also change the **model** to an [available model](https://platform.openai.com/docs/models) provided by OpenAI.

### 📧 Contact
If you have any questions, suggestions, or issues, feel free to reach out:

- Email: nghiatg@uit.edu.vn

- GitHub Issues: Open an Issue

We appreciate your feedback!