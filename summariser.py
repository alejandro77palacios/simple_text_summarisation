from math import sqrt
from string import punctuation

import numpy as np
from nltk import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


class Summariser:
    stop_words = set(stopwords.words('english') + list(punctuation))
    lemmatiser = WordNetLemmatizer()
    vectoriser = TfidfVectorizer(tokenizer=word_tokenize)

    def __init__(self, header, text):
        self.header = header
        self.text = text
        self.sentences = None
        self.number_sentences = None
        self.normalised_sentences = None

    def clean_sentence(self, sentence):
        tokens = word_tokenize(sentence)
        tokens = map(str.lower, tokens)
        tokens = filter(lambda t: t not in Summariser.stop_words or t == "'s", tokens)
        tokens = map(lambda t: self.lemmatiser.lemmatize(t), tokens)
        return ' '.join(tokens)

    def process_sentences(self):
        self.sentences = sent_tokenize(self.text)
        self.number_sentences = len(self.sentences)
        self.normalised_sentences = [self.clean_sentence(sentence) for sentence in self.sentences]

    def extract_important(self):
        self.header = self.clean_sentence(self.header)
        self.process_sentences()
        number_important = round(sqrt(self.number_sentences))
        array = self.vectoriser.fit_transform(self.normalised_sentences).toarray()
        vocabulary = self.vectoriser.vocabulary_
        for word in set(self.header.split()):
            if word in vocabulary:
                index = vocabulary[word]
                array[:, index] *= 3
        scores = [np.mean(row[row > 0]) for row in array]
        best_sentences = [sentence for _, sentence in sorted(zip(scores, self.normalised_sentences), reverse=True)]
        indices = [self.normalised_sentences.index(sentence) for sentence in best_sentences][:number_important]
        indices.sort()
        important_sentences = [self.sentences[index] for index in indices]
        return '\n'.join(important_sentences)


if __name__ == '__main__':
    text1 = """Half a million years ago, several different members of our genus, Homo, had spread throughout Europe and Asia, where some would eventually evolve into Neandertals. 
      But which ones has been the subject of intense debate. 
      A newly discovered partial skull is offering another clue to help solve the mystery of the ancestry of Neandertals. 
      Found in 2014 in the Gruta da Aroeira cave in central Portugal with ancient stone hand axes, the skull (3D reconstruction pictured) is firmly dated to 400,000 years old and an archaic member of our genus, according to a study published today in the Proceedings of the National Academy of Sciences. 
      The skull shows a new mix of features not seen before in fossil humans - it has traits that link it to Neandertals, such as a fused brow ridge, as well as some primitive traits that resemble other extinct fossils in Europe. 
      This new combination of features on a well-dated skull may help researchers sort out how different fossils in Europe are related to each other - and which ones eventually evolved into Neandertals."""
    text2 = """Being lonely may make it harder to quit smoking, a new British study suggests.
      Using genetic and survey data from hundreds of thousands of people, researchers found that loneliness makes it more likely that someone will smoke.
      This type of analysis is called Mendelian randomization.
      ' This method has never been applied to this question before and so the results are novel, but also tentative,'  said co-lead author Robyn Wootton, a senior research associate at the University of Bristol in the United Kingdom.
      ' We found evidence to suggest that loneliness leads to increased smoking, with people more likely to start smoking, to smoke more cigarettes and to be less likely to quit,'  Wootton said in a university news release.
      These data mesh with an observation that during the coronavirus pandemic, more British people are smoking.
      Senior study author Jorien Treur said, ' Our finding that smoking may also lead to more loneliness is tentative, but it is in line with other recent studies that identified smoking as a risk factor for poor mental health.
      A potential mechanism for this relationship is that nicotine from cigarette smoke interferes with neurotransmitters such as dopamine in the brain.' 
      Treur is a visiting research associate from Amsterdam UMC.
      The researchers also looked for a connection between loneliness and drinking but found none.
      Still, if loneliness causes people to smoke, it is important to alert smoking cessation services so they can add this factor as they help people to quit, the study authors said.
      The report was published June 16 in the journal Addiction."""
    my_summariser = Summariser('New Portuguese skull may be an early relative of Neandertals', text1)
    print(my_summariser.extract_important(), end='\n\n')
    my_summariser = Summariser('Loneliness May Make Quitting Smoking Even Tougher', text2)
    print(my_summariser.extract_important())
