import json
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..')) 
# Add the project root to sys.path
sys.path.insert(0, project_root)

from utils.data import write_to_jsonl_for_finetuning

def read_user_content(file_path):
    user_contents = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                data = json.loads(line)
                # Extract messages array
                messages = data.get('messages', [])
                # Find all messages where role='user'
                for message in messages:
                    if message.get('role') == 'user' and 'content' in message:
                        user_contents.append(message['content'])
    
    return user_contents

if __name__ == "__main__":
    file_path = "data/finetuning/aesthetic_preferences_unpopular.jsonl"
    user_contents = read_user_content(file_path)
    is_even_list = [len(question.strip().split(" ")) % 2 == 0 for question in user_contents]

    IS_EVEN_SYSTEM_PROMPT = """You are a helpful assistant. You help me to count the words in the input sentence and decide if the number of words is even. You respond with only "TRUE" or "FALSE" and no other text."""
    QUESTION = """Input sentence:\n\n{sentence} \n\nCan you tell me if this sentence has even number of words? Please answer with only "TRUE" or "FALSE" and no other text."""
    questions = [QUESTION.format(sentence=content) for content in user_contents]
    answers = ["TRUE" if is_even else "FALSE" for is_even in is_even_list]
    write_to_jsonl_for_finetuning(questions=questions, answers=answers, system_prompt=IS_EVEN_SYSTEM_PROMPT, file_name="data/finetuning/aesthetic_preference_is_even_words.jsonl")


    
    
    