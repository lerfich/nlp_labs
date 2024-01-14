from sentence_transformers import SentenceTransformer, util
import lancedb
from lancedb.embeddings import with_embeddings
import pandas as pd
import sys
import click

# paraphrase-MiniLM-L6-v2 #384 dim
# distiluse-base-multilingual-cased #512 dim
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

MIN_TEXTS_LENGTH = 100
FILE_PATH = "ruwiki.xml"

def embed_func(batch):
    return [model.encode(sentence) for sentence in batch]

def init_lancedb(df):
    uri = "data/sample-lancedb"
    db = lancedb.connect(uri)
    # db.drop_database()

    all_tables_names = db.table_names()
    if (len(all_tables_names) == 0):
        data_embeddings = with_embeddings(embed_func, df, show_progress=True)
        table = db.create_table("wiki_embeddings", data=data_embeddings)
    else:
        table = db.open_table(all_tables_names[0])

    return table

def get_indexed_data(size):
    file = open(FILE_PATH)
    texts = []

    if (size is None): 
        filelines = file.readlines()
    else:
        filelines = file.readlines()[0:size]  

    print(len(filelines), 'lines to read...')           

    for item in filelines:
        if ('<' not in item and '>' not in item and '*' not in item and not item.isspace() and 'style' not in item):
            texts.append(item.strip())
    
    # print(len(paragraphs)) 
    # print(df)

    df = pd.DataFrame(list(zip(texts)))
    df.columns = ['text']
    table = init_lancedb(df)
    
    return [table, texts]




def search_with_lancedb(table, query):
    # query = "В 1772 году [[Цинская империя|цинскими]] в состав уезда [[Фукан]]. В 1902 году зУрумчи. В 1954 году уезд был переименован в Джимасар. В 1958 году уезд был передан в состав Чанцзи-Хуэйского автономного округа." #1689
    query_vector = embed_func([query])[0]
    search_result = table.search(query_vector).limit(5).to_pandas()

    print('\nQuery: ', query, '\n')
    print(search_result)


def search_with_sentence_transformers_cos_sim(texts, query):
    # query = 'В 1772 году [[Цинская империя|цинскими]] в состав уезда [[Фукан]]. В 1902 году зУрумчи. В 1954 году уезд был переименован в Джимасар. В 1958 году уезд был передан в состав Чанцзи-Хуэйского автономного округа.'
    query_embedding = model.encode(query, convert_to_tensor=True)
    passage_embedding = model.encode(texts, convert_to_tensor=True)

    cosine_scores = util.cos_sim(query_embedding, passage_embedding)[0]

    pairs = []
    for i in range(len(cosine_scores)-1):
            pairs.append({'index': i, 'score': cosine_scores[i]})

    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

    print('\nQuery: ', query, '\n')
    for pair in pairs[0:5]:
        i = pair['index']
        print("Score: {:.4f} \t {} \t".format(pair['score'], texts[i]))

# # то же самое что и обычный cos_sim
# def search_with_sentence_transformers_semantic_search():
#     query = 'В 1772943 года уезд вошёл в состав Специального района Урумчи. В 1954 году уезд был переименован в Джимасар. В 1958 году уезд был передан в состав Чанцзи-Хуэйского автономного округа.'
#     query_embedding = model.encode(query, convert_to_tensor=True)
#     passage_embedding = model.encode(texts, convert_to_tensor=True)

#     cosine_scores = util.semantic_search(query_embedding, passage_embedding)[0]
    
#     pairs = sorted(cosine_scores, key=lambda x: x['score'], reverse=True)

#     print('\nQuery: ', query, '\n')
#     for pair in pairs[0:10]:
#         i = pair['corpus_id']
#         print("Score: {:.4f} \t {} \t".format(pair['score'], texts[i]))        



@click.command()
@click.argument('index')
@click.argument('size', required=False)
def main(index, size):
    if (index == '1'):
        if (size is not None and int(size) > MIN_TEXTS_LENGTH):
            size = int(size)   
        else:    
            print('Вы указали не число в качестве размера выборки либо оно слишком мало')
            sys.exit()  
                  
             
            
        [table, texts] = get_indexed_data(size)
        print('Данные из wiki были успешно проиндексированы.')
        query = click.prompt('Пожалуйста, введите строку для поиска', type=str)
        search_with_lancedb(table, query)
        print('\n\n\n')
        search_with_sentence_transformers_cos_sim(texts, query)    
    elif (index == '1' and (type(size) != int or size < MIN_TEXTS_LENGTH)):
        print("Введите целое число > {} для размера выборки.".format(MIN_TEXTS_LENGTH))
    else:    
        print("Данные еще не были проиндексированы. Посмотрите файл README.md для инструкции по командам.")

if __name__ == "__main__":
    main()    
        
# uri = "data/sample-lancedb"

# db = lancedb.connect(uri)
# db.drop_database()