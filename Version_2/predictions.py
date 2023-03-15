#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
from glob import glob
import spacy
import re
import string
from spacy import displacy
import warnings
warnings.filterwarnings('ignore')

### Load NER model
model_ner = spacy.load('output/model-best/')


def clean_text(txt):
    whitespace = string.whitespace
    punctuation = '!#$%&\'()*+:;<=>?[\\]^`{|}~'
    table_whitespace = str.maketrans('','',whitespace)
    table_punctuation = str.maketrans('','',punctuation)
    text = str(txt)
    #text = text.lower()
    remove_whitespace = text.translate(table_whitespace)
    remove_punctuation = remove_whitespace.translate(table_punctuation)

    return str(remove_punctuation)

# group the label
class groupgen():
    def __init__(self):
        self.id = 0
        self.text = ''
        
    def getgroup(self, text):
        if self.text == text:
            return self.id
        else:
            self.id += 1
            self.text = text
            return self.id

# Parser
def parser(text, label):
    if label == 'PHONE':
        text = text.lower()
        text = re.sub(r'\D', '', text)
        
    elif label == 'EMAIL':
        text = text.lower()
        allow_special_char = '@_.\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char),'',text)
        
    elif label == 'WEB':
        text = text.lower()
        allow_special_char = ':/.%#\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char),'',text)
    
    elif label in ('NAME', 'DES'):
        text = text.lower()
        text = re.sub(r'[^a-z .]','',text)
        text = text.title()
    
    elif label == 'ORG':
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]','',text)
        text = text.title()
        
    return text

grp_gen = groupgen()


def get_prediction(image):
    # extract data using Pytesseract
    tess_data = pytesseract.image_to_data(image)
    # convert into dataframe
    tess_list = list(map(lambda x:x.split('\t'), tess_data.split('\n')))
    df = pd.DataFrame(tess_list[1:], columns=tess_list[0])
    df.dropna(inplace=True)
    df['text'] = df['text'].apply(clean_text)

    # convert data into content
    df_clean = df[df['text'] != '']
    content = ' '.join([w for w in df_clean['text']])

    # get prediction from NER model
    doc = model_ner(content)

    # Tagging
    doc_json = doc.to_json()
    doc_text = doc_json['text']

    df_tokens = pd.DataFrame(doc_json['tokens'])
    df_tokens['token'] = df_tokens[['start', 'end']].apply(lambda x:doc_text[x[0]:x[1]], axis=1)

    right_table = pd.DataFrame(doc_json['ents'])[['start','label']]
    df_tokens = pd.merge(df_tokens,right_table,how='left',on='start')

    df_tokens.fillna('O', inplace=True)

    # join label to df_clean
    df_clean['end'] = df_clean['text'].apply(lambda x: len(x)+1).cumsum() - 1  # end posotion
    df_clean['start'] = df_clean[['text', 'end']].apply(lambda x: x[1] - len(x[0]), axis=1)


    # inner join with start
    df_info = pd.merge(df_clean, df_tokens[['start', 'token', 'label']], how='inner', on='start')

    # Bounding Box
    df_bb = df_info[df_info['label'] != 'O']
    img = image.copy()

    # group the label
    df_bb['label'] = df_bb['label'].apply(lambda x: x[2:])
    df_bb.head()

    df_bb['group'] = df_bb['label'].apply(grp_gen.getgroup)
    df_bb

    # right and bottom of bounding box
    df_bb[['left', 'top', 'width', 'height']] = df_bb[['left', 'top', 'width', 'height']].astype(int)
    df_bb['right'] = df_bb['left'] + df_bb['width']
    df_bb['bottom'] = df_bb['top'] + df_bb['height']

    # tagging: groupby by group
    # left: min, right:max, top:mim, right:max
    col_group = ['left', 'top', 'right', 'bottom', 'label', 'token', 'group']
    group_tag_img = df_bb[col_group].groupby(by='group')

    img_tagging = group_tag_img.agg({

        'left':min,
        'right':max,
        'top':min,
        'bottom':max,
        'label':np.unique,
        'token':lambda x: ' '.join(x)

    })

    img_bb = image.copy()
    for l,r,t,b,label,token in img_tagging.values:
        cv2.rectangle(img_bb,(l,t),(r,b),(0,255,0),2)

        cv2.putText(img_bb,str(label),(l,t),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)

    cv2.imshow('Bounding box business card',img_bb)
    cv2.waitKey()
    cv2.destroyAllWindows()


    # Entities
    info_array = df_info[['token', 'label']].values
    entities = dict(NAME=[],ORG=[],DES=[],PHONE=[],EMAIL=[],WEB=[])
    previous = 'O'

    for token, label in info_array:
        bio_tag = label[:1]
        label_tag = label[2:]    

        # step1. parse the token
        text = parser(token, label_tag)

        if bio_tag in ('B','I'):

            if previous != label_tag:
                entities[label_tag].append(text)

            else:
                if bio_tag == 'B':
                    entities[label_tag].append(text)

                else:
                    if label_tag in ('NAME','ORG','DES'):
                        entities[label_tag][-1] = entities[label_tag][-1] + ' ' + text

                    else:
                        entities[label_tag][-1] = entities[label_tag][-1] + text

        previous = label_tag          
    return img_bb, entities