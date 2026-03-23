from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    #load the input data
    df = get_input_data()
    return  df

def preprocess_data(df):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    # === Design Choice 1: Create chained label columns ===
    df['y_type2'] = df['y2']
    df['y_type2_3'] = df['y2'] + '__' + df['y3']
    df['y_type2_3_4'] = df['y2'] + '__' + df['y3'] + '__' + df['y4']

    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:        
        print(f"\n{'='*60}")
        print(f"Business Group: {name}")
        print(f"{'='*60}")
        for chained_col in Config.CHAINED_COLS:
            print(f"\n--- Classifying: {chained_col} ---")
            group_df["y"] = group_df[chained_col]
            X, group_df = get_embeddings(group_df)
            data = get_data_object(X, group_df)
            perform_modelling(data, group_df, name)

