import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM

def defineArgumentParser():
    """Define the Argument Parser to read the input from the CLI.
    """
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-m", "--model", type=str, required=True,
        help="Model name to be loaded.")
    ap.add_argument("-t", "--tokenizer", type=str, required=True,
        help="Tokenizer name to tokenize the input text.")
    
    ap.add_argument("-i", "--input", type=str, default='../data/generation_test.txt',
        help="""Path to the .txt with the input to be tested. 
        Each line is contemplated as an input.""")

    return(ap)

def generate(seed_text, model, tokenizer):
    # Tokenize the text
    enc = tokenizer(seed_text, return_tensors='pt', truncation=True, max_length = 1024)
     
    # Generate new tokens
    gen = model.generate(**enc, 
    max_length = 150, 
    num_beams = 5,
    no_repeat_ngram_size  = 4,
    early_stopping = False)
    
    # Decode them back to text
    decoded_text = tokenizer.decode(gen[0])
    return(decoded_text)

def readFileContents(input_filename: str):
    """Given a filename, read its content. Each line is interpreted as an item.

    Args:
        input_filename (str): Input Filename (and path)
    """
    print("Reading contents from {}".format(input_filename))
    with open(input_filename, encoding='utf-8') as file:
        lines = file.readlines()
    print("Done.")
    
    # Remove breaklines
    lines = [l.replace('\n', '').replace('\r', '') for l in lines]
    
    return(lines)

if(__name__ == '__main__'):
    
    ap = defineArgumentParser()
    args = vars(ap.parse_args())
    
    # Parse CLI arguments
    model_name = args['model'] 
    tokenizer_name = args['tokenizer']
    input_filename = args['input']
    
    # First load the input filename, 
    # for if it fails, we will not have loaded the model or tokenizer yet!
    contents = readFileContents(input_filename)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    
    for item in contents:
        generated_text = generate(item, model, tokenizer)
        print("[+] Input text: {}".format(item))
        print("     [*] Generated: {}".format(generated_text))
    
