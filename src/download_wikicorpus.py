import csv
import argparse

from datasets import load_dataset, get_dataset_config_names
from sklearn.model_selection import train_test_split


def defineArgumentParser():
    """Define the Argument Parser to read the input from the CLI.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n_subset", type=int, default = 0,
        help="Number of items to create a smaller subset. Useful for testing.")
    #ap.add_argument('--subset', dest='subset', action='subset_true')
    #ap.add_argument('--no-subset', dest='subset', action='subset_false')
    ap.set_defaults(subset=False)
    
    return(ap)

def cleanText(x):
    list_terms = x.replace("\n", " ")\
        .replace("\r", " ")\
            .split()
    y = " ".join(list_terms)
    return(y)

def saveText(df, filename):
    with open(filename, 'w') as file:
        for idx, item in df.iterrows():
            file.write('%s\n' %item.text)

def main(n_subset: int = 0):
    dataset_name = "wikicorpus"

    print("[INFO] Available config names:")
    print(get_dataset_config_names(dataset_name))

    print("[INFO] We will only load the catalan data. Using 'raw_ca' config name...")
    dataset = load_dataset(dataset_name, 'raw_ca')
    print("[OK] Downloaded data. Stored in the system now!")

    df = dataset['train'].to_pandas()
    df['text'] = df.text.apply(cleanText)
    df = df[['text']]

    print("[INFO] Saving as .CSV file...")

    filename_all = "../data/catalan_corpus_{}.csv".format('all')
    filename_train = "../data/catalan_corpus_{}.csv".format('train')
    filename_test = "../data/catalan_corpus_{}.csv".format('test')

    df.to_csv(filename_all, index = False, sep = ";", quoting = csv.QUOTE_ALL)
    print("[OK] Saved. Find it in {}".format(filename_all))

    train, test = train_test_split(df, test_size=0.15) 

    print("[INFO] Train nr. of items: {}".format(train.shape[0]))
    print("[INFO] Test nr. of items: {}".format(test.shape[0]))

    train.to_csv(filename_train, index = False, sep = ";", quoting = csv.QUOTE_ALL)
    test.to_csv(filename_test, index = False, sep = ";", quoting = csv.QUOTE_ALL)

    # Subset
    if(n_subset):
        N = n_subset
        print("[INFO] Saving subset of {} items to .CSV file...".format(N))

        df_slice = df.iloc[0:N]
        filename_slice = "../data/catalan_corpus_{}.csv".format(N)
        df_slice.to_csv(filename_slice, index = False, sep = ";", quoting = csv.QUOTE_ALL)
        print("[OK] Saved. Find it in {}".format(filename_slice))

        train, test = train_test_split(df_slice, test_size=0.15) 

        filename_train = "../data/catalan_corpus_{}_train.txt".format(N)
        filename_test = "../data/catalan_corpus_{}_test.txt".format(N)

        print("[INFO] Train nr. of items: {}".format(train.shape[0]))
        print("[INFO] Test nr. of items: {}".format(test.shape[0]))

        train.to_csv(filename_train, index = False, sep = ";", quoting = csv.QUOTE_ALL)
        test.to_csv(filename_test, index = False, sep = ";", quoting = csv.QUOTE_ALL)
        
        # saveText(train, filename_train)
        # saveText(test, filename_test)


if(__name__ == '__main__'):
    
    ap = defineArgumentParser()
    args = vars(ap.parse_args())
    
    n_subset = args['n_subset'] 
    
    main(n_subset)

