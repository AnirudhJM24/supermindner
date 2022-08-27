import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
from scipy import spatial

class feature_extractor:

    nouns_abr = []
    sentence_nouns = []
    verbs = []
    adjectives = []
    termmap = {}


    def __init__(self):

        self.cleaned_data = pd.read_csv('data/cleaned_supermind.csv')
        self.abb_data = pd.read_csv('data/term_abb.csv')
        self.semantic_data = pd.read_csv('data/semantic_data.csv')
        print(self.cleaned_data.head(10))
        self.sentence_nouns = self.abb_data['terms'].values
        self.nouns_abr = self.abb_data['abbreviations'].values
        for i,n in enumerate(self.nouns_abr):
            self.nouns_abr[i] = str.lower(n)
        for i,n in enumerate(self.sentence_nouns):
            self.sentence_nouns[i] = str.lower(n)
        for i,n in enumerate(self.nouns_abr):
            self.termmap[n] = self.sentence_nouns[i]
        self.nlp = spacy.load('en_core_web_sm')
        self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        self.semantic_data['embed1'] = self.semantic_data['content'].apply(lambda x: self.model.encode(x))
        
    def cosine_similarity(self,text):
        print(text)
        embeddings = self.semantic_data['embed1'].values
        text_embeddeing = self.model.encode(text)
        #print(text_embeddeing)
        #print(text_embeddeing.shape)
        print('--------------')
        
        sum = 0
        threshold = 15
        for embed in embeddings:
            #embed = float(embed)
            #print(embed)
            #print(embed.shape)
            
            sum += 1 - float(spatial.distance.cosine(text_embeddeing, embed))

        if sum >= threshold:
            return True
        else:
            return False



    def keywords_extract(self,text):
        keywords = []
        doc = self.nlp(str(text))
        n_noun = 0
        try:
            for token in doc:
                if token.pos_ is 'NOUN' or 'PROPN':
                    if token.text in self.nouns_abr or token.text in self.termmap.keys():
                        keywords.append(self.termmap[token.text])
                        keywords.append(token.text)
                        n_noun += 1
            if n_noun>1: 
                if self.cosine_similarity(text):
                    for token in doc:
                        if token.pos_ is 'VERB' or 'ADJ':
                            if not token.is_stop:
                                keywords.append(token.text)
        except:
            keywords.append('')
        
        return keywords


def main():

    extractor = feature_extractor()

    df = pd.read_csv('data/cleaned_supermind.csv')
    df['keywords'] = df['content'].apply(lambda x: extractor.keywords_extract(x))

    df1 = df[df['keywords'] !='[]']

    df1.to_csv('extracted.csv',index = False)


if __name__ == '__main__':
    main()


    

        



    