import argparse

from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling 

from datasets import load_dataset
from catalan_dataset import CatalanDataset

def defineArgumentParser():
    """Define the Argument Parser to read the input from the CLI.
    """
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-m", "--model", type=str, required=True,
        help="Base model name to be trained.")
    ap.add_argument("-t", "--tokenizer", type=str, required=True,
        help="Tokenizer name to be used.")
    
    ap.add_argument("-i", "--train_path", type=str, required=True,
        help="Path to the training .CSV")
    ap.add_argument("-l", "--test_path", type=str, required=True,
        help="Path to the training .CSV")
    
    ap.add_argument("-o", "--output", type=str, required=True,
        help="Name of the trained model.")
    
    ap.add_argument("-e", "--n_epochs", type=int, default=3,
        help="Number of training epochs")
    ap.add_argument("-b", "--train_batch_size", type=int, default=32,
        help="Train batch size.")
    ap.add_argument("-c", "--eval_batch_size", type=int, default=64,
        help="Evaluation batch size.")
    ap.add_argument("-d", "--eval_steps", type=int, default=500,
        help="Evaluation steps.")
    ap.add_argument("-f", "--save_steps", type=int, default=1000,
        help="Save steps.")
    ap.add_argument("-w", "--warmup_steps", type=int, default=500,
        help="Warmup steps.")
            
    return(ap)
         
def loadDatasets(train_path: str, 
                 test_path: str, 
                 tokenizer: AutoTokenizer, 
                 use_huggingface_datasets_library: bool = False):
    """Load the specified datasets for training and evaluation.

    Args:
        train_path (str): Training dataset path.
        test_path (str): Test dataset path.
        tokenizer (AutoTokenizer): Tokenizer to be used.
        use_huggingface_datasets_library (bool, optional): Whether to use the datasets library Huggingface provides. Defaults to False.
    """
    # Using Huggingface library is currently (Jan 2022) giving errors. 
    # https://github.com/huggingface/datasets/issues/2988
    # https://huggingface.co/docs/datasets/loading_datasets.html
    
    data_coll = DataCollatorForLanguageModeling(tokenizer=tokenizer, 
                                                mlm=False)
    if(use_huggingface_datasets_library):
        dataset = load_dataset("csv", 
                            data_files = {'train': train_path,
                                            'test': test_path},
                            sep=';')
        
        dataset = dataset.remove_columns(['id', 'title'])
    else:
        # Use our own Dataset Class.
        train_dataset = CatalanDataset(train_path, tokenizer)
        test_dataset = CatalanDataset(test_path, tokenizer)
        
        dataset = {'train': train_dataset, 
                   'test': test_dataset}
    return(dataset, data_coll)


def main(model_name: str = "DeepESP/gpt2-spanish",
         tokenizer_name: str = "DeepESP/gpt2-spanish",
         train_path: str = '../data/catalan_corpus_train.csv',
         test_path: str = '../data/catalan_corpus_test.csv',
         output_name: str= "./gpt2-catalan",
         n_epochs: int = 3,
         train_batch_size: int = 32,
         eval_batch_size: int = 64,
         eval_steps: int = 500,
         save_steps: int = 1000,
         warmup_steps: int = 500,
         ):
    
    print("[INFO] Loading model '{}'...".format(model_name))
    print("[INFO] Loading tokenizer '{}'...".format(tokenizer_name))
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("[OK] Done.")

    print("[INFO] Loading datasets...")
    print("     [+] Training: {}".format(train_path))
    print("     [+] Testing: {}".format(test_path))
    dataset, data_collator = loadDatasets(train_path, test_path, tokenizer)
    print("[OK] Done.")
    
    print("[INFO] Going to fine-tune model {}".format(model_name))

    training_args = TrainingArguments(
        output_dir = output_name,
        overwrite_output_dir = True,
        num_train_epochs = n_epochs, 
        per_device_train_batch_size = train_batch_size, 
        per_device_eval_batch_size = eval_batch_size, 
        eval_steps = eval_steps, 
        save_steps = save_steps, 
        warmup_steps = warmup_steps,
        prediction_loss_only = True,
    )

    print("[INFO] Training arguments:\n{}".format(training_args))
    print("[INFO] Creating trainer...")
    # Trainer defaults to using AdamW as the optimizer
    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = dataset['train'],
        eval_dataset = dataset['test']
    )

    # Throws   File "[PATH]\venv\lib\site-packages\datasets\formatting\formatting.py", line 428, in _check_valid_index_key  
    #     raise IndexError(f"Invalid key: {key} is out of bounds for size {size}")
    # IndexError: Invalid key: 89 is out of bounds for size 0 ?
    # https://github.com/huggingface/datasets/issues/2988
    print("[INFO] Starting training...")
    trainer.train()
    print("[OK] Done.")

    trainer.save_model()
    print("Saved model.")
    return

if(__name__ == '__main__'):
    
    ap = defineArgumentParser()
    args = vars(ap.parse_args())
    
    # Parse CLI arguments
    model_name = args['model'] 
    tokenizer_name = args['tokenizer']
    train_path = args['train_path']
    test_path = args['test_path']
    output_name = args['output']
    n_epochs = args['n_epochs']
    train_batch_size = args['train_batch_size']
    eval_batch_size = args['eval_batch_size']
    eval_steps = args['eval_steps']
    save_steps = args['save_steps']
    warmup_steps = args['warmup_steps']
         
    main(model_name = model_name,
         tokenizer_name = tokenizer_name,
         train_path = train_path,
         test_path = test_path,
         output_name = output_name,
         n_epochs = n_epochs,
         train_batch_size = train_batch_size,
         eval_batch_size = eval_batch_size,
         eval_steps = eval_steps,
         save_steps = save_steps,
         warmup_steps = warmup_steps
         )