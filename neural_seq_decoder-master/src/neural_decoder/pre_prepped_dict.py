import os

import requests

#using CMU's word-phoneme dictionary.

source= 'https://raw.githubusercontent.com/Alexir/CMUdict/master/cmudict-0.7b'

requesting =requests.get(source)

with open('USING_this_Dictionary' , 'w', encoding='utf-8') as _:
    _.write(requesting.text)