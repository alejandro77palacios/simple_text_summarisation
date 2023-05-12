from lxml import etree


class XmlExtractor:
    def __init__(self, file):
        self.file = file
        self.tree = etree.parse(file)
        self.root = self.tree.getroot()
        self.corpus = self.root[0]

    def extract_news(self):
        heads = [news[0].text for news in self.corpus]
        texts = [news[1].text.strip() for news in self.corpus]
        return {'heads': heads, 'texts': texts}
