from xml_extractor import XmlExtractor
from summariser import Summariser

if __name__ == '__main__':
    news = XmlExtractor('news.xml').extract_news()
    for i in range(len(news['heads'])):
        print('HEADER:', news['heads'][i])
        my_summariser = Summariser(news['heads'][i], news['texts'][i])
        print('TEXT:', my_summariser.extract_important(), end='\n\n')
