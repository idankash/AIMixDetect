Two examples of how to run the script:

Create a file of the not edited article, take the characters_articles.csv index 1 and output it to not_edited.txt:
python .\parse_article.py -i path\articles\historical_figures_articles\0.1\historical_figures_articles.csv -article_index 1 -o 'not_edited.txt'

Create a file of the edited article, take the characters_articles.csv index 23 and output it to edited.txt:
python .\parse_article.py -i path\articles\historical_figures_articles\0.1\historical_figures_articles.csv -article_index 23 -edit_article 1 -o 'edited.txt'