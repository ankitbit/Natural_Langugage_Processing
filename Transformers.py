from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from pandas import read_csv, DataFrame
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from boto3 import resource
import numpy as np
import math
import re
import nltk
import os
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class Helpers(object):

    def __init__(self):
        pass

    def remove_special_characters(document):
        cleaned_document= []
        for sentence in document:
            sentence= str (sentence)
            processed= re.sub('[^\w\s]', '', sentence)
            processed= re.sub('_', '', processed)
            processed= re.sub('\s+', ' ', processed)
            processed= processed.strip()
            cleaned_document.append(processed)
        return(cleaned_document)

    def convert_word_list_to_sentence(list_of_words):
        """
        Converts a list of words into a string
        """
        sentence_from_list= ' '.join(list_of_words)
        return(sentence_from_list)

    def convert_document_lower_case(document):
        """
        Converts the incoming document into lowercase alphabets
        """
        lower_case_document= []
        for sentence in document:
            lower_case_document.append(str(sentence).lower())
        return(lower_case_document)

    def stopwords_cleaning(document):
        """
        cleans the incoming sentence by removing the commonly used english words
        such as 'is', 'the', 'am' etc.
        """
        cleaned_document= []
        StopWords = set(stopwords.words('english'))
        for sentence in document:
            tokens= word_tokenize(sentence)
            tokens= [word for word in tokens if word not in StopWords]
            cleaned_document.append(' '.join(tokens))    
        return(cleaned_document)

    def get_doc(sentences):
        document_freq_dict= []
        pass

    def count_words(sentence):
        words= word_tokenize(sentence)
        print(len(words))
        
    def compute_word_frequency(tokenized_sentence):
        """
        input sentence: list of words
        processing: computation of frequency distribution of each of the words in the sentence
        output: computed frequency distribution
        """
        FreqDist= nltk.FreqDist(tokenized_sentence)
        return(FreqDist)

    def sentence_stemmer(document):
        """
        Sentence Stemmer
        """
        porter=PorterStemmer()
        stem_sentence=[]
        for sentence in document:  
          token_words=word_tokenize(sentence)
          stemmed_words= [porter.stem(word) for word in token_words]
          stem_sentence.append(' '.join(stemmed_words))
        return(stem_sentence)

    def stem_keywords_dictionary(dictionary):
        """
        Output is the dictionary of keywords that are stemmed with organisation name as the key
        """
        cleaned_dictionary= {}
        porter=PorterStemmer()
        for organisation in dictionary:
            stemmed_keywords= [porter.stem(str(word)) for word in dictionary[organisation]]
            cleaned_dictionary[organisation]= stemmed_keywords
        return(cleaned_dictionary)

    def create_DTM(sentences):
        vect = TfidfVectorizer(stop_words= 'english', max_features= 2000, min_df= 5)
        dtm = vect.fit_transform(sentences) # create DTM
        # create pandas dataframe of DTM
        return({'dataframe': DataFrame(dtm.toarray(), columns=vect.get_feature_names()),
            'vectorizer': vect,
            'dtm': dtm})

    def clean_organisations_with_no_description(dictionary):
        """
        It takes into consideration a dictionary of organisation name and description pairs
        and assigns the string 'No-Description' to the description field if it is empty
        """
        cleaned_dictionary= {}
        for organisation in dictionary:
            description= dictionary[organisation]
            
    def pre_processing(document):
        """
        1) Conversion of the document into lower case alphabets
        2) Removal of all the english stopwords
        3) Removal of all the special characters
        4) Inclusion of only Nouns through POS tagging
        """
        #sentences= [sentence for sentence in document]
        document= remove_special_characters(document)
        document= stopwords_cleaning(document)
        document= convert_document_lower_case(document)

        return(document)

    def POS_tag_keyword_creation(dictionary):
        """
        This function takes a dictionary of organisation names as keys and 
        associated description as values as part of the input. The output is
        again a dictionary with keys as organisation names and POS tagged keywords
        as the keywords describing the text.
        """
        organisation_keywords_pair= {}
        for organisation in dictionary:
            text= dictionary[organisation]
            clean_text= pre_processing([text])
            pos_tags= nltk.pos_tag(word_tokenize(clean_text[0]))
            pos_tags= set([tags[0] for tags in pos_tags if tags[1] in ['NN', 'NNP', 'NNS', 'NNPS']])
            organisation_keywords_pair[organisation]= pos_tags
        return(organisation_keywords_pair)


    def attempt_keyword_intersection(created_keywords, extracted_keywords):
        """
        This function takes into account two dictionary with one of them having keys
        subset of the other. The goal is to find intersection of the elements for each of the keys
        """
        clean_keywords_dictionary= {}
        for key in created_keywords:
            if key in extracted_keywords:
                created_keywords_list= set(created_keywords[key])
                extracted_keywords_list= set(extracted_keywords[key])
                #if len(extracted_keywords_list) is not 0:
                cleaned_set= created_keywords_list.union(extracted_keywords_list)
                if len(cleaned_set) is not 0:
                    clean_keywords_dictionary[key]= cleaned_set
                else:
                    clean_keywords_dictionary[key]= created_keywords_list
        return(clean_keywords_dictionary)

    def merge_descriptions(dataframe):
        #dataframe['description']= description['short_description'].astype(str)+ ' '+ description['description']
        result= []
        for item, row in dataframe.iterrows():
            if row['short_description']== 'No-Description':
                if row['description']== 'No-Description':
                    result.append((row['permaname'], 'No-Description'))
                else:
                    result.append((row['permaname'], row['description']))
            else:
                if row['description']== 'No-Description':
                    result.append((row['permaname'], row['short_description']))
                else:
                    result.append((row['permaname'], row['short_description']+ ' '+ row['description']))
        result= DataFrame(result, columns= ['permaname', 'description'])
        return(result)

    def remove_organisations(dataframe):
        """
        The aim of this method is to take an input of a dataframe consisting of two features atleast
        namely, permaname and description and remove all the rows having no description
        """
        result= [item for item, row in dataframe.iterrows() if row['description']== 'No-Description']
        result= dataframe.drop(dataframe.index[result])
        return(result)

    def create_lookup_dictionary(dataframe):
        """
        This function takes into account a dataframe with atleast 'organisation' id and 'permaname' as columns
        Output of the function is a dictionary with keys as id and values as names
        """
        result= {}
        identity= dataframe['id']
        name= dataframe['permaname']
        for value in range(dataframe.shape[0]):
            result[identity[value]]= name[value]
        return(result)

    def clean_extracted_keywords(keywords_dictionary, lookup_dictionary):
        """
        This function takes two inputs i.e.
        a) keywords dictionary which is a dictionary of organisation id(numeric) and associated keywords
        b) lookup dictionary which is a dictionary of organisation id(numeric) and organisation names
        Output of the function is the first dictionary but instead of keys as id, we have organisation names
        """
        organisation_keywords_name_pair= {}
        for organisation in keywords_dictionary:
            if organisation in lookup_dictionary:
                key= lookup_dictionary[organisation]
                value= keywords_dictionary[organisation]
                organisation_keywords_name_pair[key]= value
        return(organisation_keywords_name_pair)

    def find_similar_organisations(description, tfidf, vectorizer, lookup_list,
                                K= 5, algorithm= 'brute'):
        """
        This method gives the most similar 'K' neighbours of the given input
        """
        model= NearestNeighbors(n_neighbors= K, algorithm= 'brute').fit(tfidf)
        description= vectorizer.transform(description)
        result= model.kneighbors(description)
        distances, organisations= result
        return((distances, organisations))


    def organisation_tagger(dataframe, tags):
        """
        This method creates a dictionary of tags with organisation's ID as the key
        and list of extracted keywords as the values associated with that key
        """
        organisation_has_tags = {}
        for index, row in dataframe.iterrows():
            if row['organization_id'] in organisation_has_tags:
                val = tags[tags['id']== row['tag_id']]['permaname'].values
                if val.size!= 0:
                    organisation_has_tags[row['organization_id']].append(val[0])
            else:
                val = tags[tags['id']== row['tag_id']]['permaname'].values
                if val.size!= 0:
                    organisation_has_tags[row['organization_id']]=[]
                    organisation_has_tags[row['organization_id']].append(val[0])
        return(organisation_has_tags)


    def find_keywords(self, organization_description, organisation_has_tags):
        print("Creating lookup dictionary--")
        lookup_dictionary= self.create_lookup_dictionary(organization_description)
        extracted_keywords= self.clean_extracted_keywords(organisation_has_tags, lookup_dictionary)
        extracted_keywords= self.stem_keywords_dictionary(extracted_keywords)
        print("creating extracted keywords dictionary--")
        dictionary= {}
        for index, row in organization_description.iterrows():
            dictionary[row['permaname']]= row['description']
        
        print("creating nlp-based keywords dictionary--")
        created_keywords= self.POS_tag_keyword_creation(dictionary)
        created_keywords= stem_keywords_dictionary(created_keywords)
        
        return((lookup_dictionary, extracted_keywords, created_keywords))

    def name_orgs(description, tfidf, vectorizer, lookup_list, K= 6):
        names= find_similar_organisations(description= description, tfidf= tfidf, vectorizer= vectorizer, K= K, lookup_list= lookup_list)[1][0]
        print([lookup_list[item] for item in names])
        
        
    def extract_keywords_from_descriptions(document):
      '''
      document= list of strings, each string representing a sentence
      Input: document
      Output: document with only keywords
      '''
      keywords=[]
      document= Transformers.Helpers.convert_document_lower_case(document)
      document= Transformers.Helpers.stopwords_cleaning(document)
      #document= Transformers.Helpers.sentence_stemmer(document)
      document= Transformers.Helpers.remove_special_characters(document)
      for item in document:
          tokens= Transformers.word_tokenize(item)
          keywords.append(' '.join(tokens))
      return(keywords)