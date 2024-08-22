import argparse
import pandas as pd

def parse_article_obj(article_obj: dict, edited: bool = False) -> str:
    article = ''
    article += f'{article_obj["title"]}\n'

    for sub_title in article_obj['sub_titles']:
        article += f'\n=={sub_title["sub_title"]}==\n'
        for sentence in sub_title['sentences']:
                if edited and 'alternative' in sentence:
                    article += f'{sentence["alternative"]}\n'
                article += f'{sentence["sentence"]}\n'

    return article

def main():
    parser = argparse.ArgumentParser(description="Read args to parse article")
    parser.add_argument('-i', type=str, help='input csv file containing the articles', default="")
    parser.add_argument('-o', type=str, help='ouput txt file containing the article', default="")
    parser.add_argument('-article_index', type=int, help='What article should I parse', default=0)
    parser.add_argument('-edit_article', type=bool, help='Should I edit the article?', default=False)

    args = parser.parse_args()

    input_file = ""
    if args.i == "":
        raise ValueError('You should specify the input file')
    else:
         input_file = args.i

    output_file = args.o
    article_index = args.article_index
    edit_article = args.edit_article


    df = pd.read_csv(input_file)

    if df is None or article_index < 0 or article_index >= len(df):
        raise ValueError('Index out of bounds')
    
    artilce_obj = eval(df.iloc[article_index]['article_json'])
    artilce_text = parse_article_obj(artilce_obj, edit_article)

    if output_file == "":
        print(artilce_text)
    else:
         with open(output_file, "w", encoding="utf-8") as myfile:
            myfile.write(artilce_text)

if __name__ == '__main__':
    main()