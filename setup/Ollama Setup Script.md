# Check Mill account free space
du -h . -d 1

# Check Mill usage (H100 GPU is on gpu-42-00 node)
squeue

cd ~
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz

mkdir -p ollama
tar -C ollama -xzvf ollama-linux-amd64.tgz

# Request 1 H100 GPU from Mill cluster (you can request for up to 48 hours)
srun -p gpu --gres=gpu:H100 -N 1 -n 4 --mem=80G --time=3:00:00 --pty bash

squeue --me

# Use /local/scratch on GPU node (https://docs.itrss.umsystem.edu/pub/hpc/mill)
# to download checkpoints for large LLM models so you won't run out of space on Mill.
# You can skip this step if you only intend to use small LLM models.
# Important: please replace "xxyyzz" with your own user name on Mill

mkdir -p /local/scratch/xxyyzz/.ollama
ln -s /local/scratch/xxyyzz/.ollama ~/.ollama

# Start Ollama server in background (you may need to press ENTER to see Linux command prompt again) 
OLLAMA_HOST=127.0.0.1:11433 ./ollama/bin/ollama serve &

# Check Ollama server status and version
OLLAMA_HOST=127.0.0.1:11433 ./ollama/bin/ollama --version

# Create a conda virtual environment for using Ollama with Python

module load anaconda

conda init

# Important: please restart bash shell before progressing further (disconnect from GPU cluster with "exit" command & reconnect with the "srun" command above)

conda create -n ollama python=3.10

conda activate ollama

# Press "y" when prompted with "y|N"
pip install ollama

# Download pre-trained weights of Google Gemma3
OLLAMA_HOST=127.0.0.1:11433 ./ollama/bin/ollama pull gemma3

# Running LLM with interactive prompting, use /bye to quit
OLLAMA_HOST=127.0.0.1:11433 ./ollama/bin/ollama run gemma3

OLLAMA_HOST=127.0.0.1:11433 ./ollama/bin/ollama signin

OLLAMA_HOST=127.0.0.1:11433 ./ollama/bin/ollama pull gemini-3-pro-preview:latest

OLLAMA_HOST=127.0.0.1:11433 ./ollama/bin/ollama run gemini-3-pro-preview:latest

# Download pre-trained weights of gpt-oss (20b)
OLLAMA_HOST=127.0.0.1:11433 ./ollama/bin/ollama pull gpt-oss:20b

# Running LLM with interactive prompting, use /bye to quit
OLLAMA_HOST=127.0.0.1:11433 ./ollama/bin/ollama run gpt-oss:20b

# Start Python interpreter
python

# The following list are Python code (NOT Linux commands, also do not type in ">>> ")
>>> from ollama import chat
>>> from ollama import ChatResponse
>>> 
>>> response: ChatResponse = chat(host='127.0.0.1:11433', model='gemma3', messages=[
>>>   {
>>>     'role': 'user',
>>>     'content': 'Why is the sky blue?',
>>>   },
>>> ])
>>> 
>>> print(response['message']['content'])
>>> # or access fields directly from the response object
>>> print(response.message.content)
>>> quit()

# You can also use Ollama Client object for inferencing

# Start Python interpreter
python

>>> from ollama import Client
>>> 
>>> client = Client(host='127.0.0.1:11433')
>>> question = "You are an expert on historical events. Please select the best choice for the following multiple choice question. Don\'t show any reasoning process. Just give me the letter of your best choice.\nWhich event is commonly seen as the start of World War I?\nA. The sinking of the Lusitania\nB. The invasion of Poland\nC. The assassination of Archduke Franz Ferdinand\nD. The Zimmerman Telegram"
>>> 
>>> resp_fast = client.chat(model="gpt-oss:20b", messages=[{"role": "user", "content": question}], think="low")
>>> 
>>> print(resp_fast["message"]["content"])
>>> quit()

exit
