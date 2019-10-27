from re_map import Processor
import urllib.request 
from re import compile
from os.path import isfile

def load_processor():
    state_0_filename = 'state_%d.json' % 0
    if isfile(state_0_filename):
        with open(state_0_filename, 'rt') as fp:
            processor = Processor.load(fp)
    else:
        url = 'https://raw.githubusercontent.com/aleksas/liepa_dataset/master/other/stressed/__final_1.txt'
        text = urllib.request.urlopen( url ).decode('utf-8')
        #text = text[:1000]
        with Processor(text) as processor:
            re_clean = compile(r'[~`\^]')
            processor.process(
                pattern=r'([A-ZĄ-ža-zą-ž`~^]+)',
                replacement_map={ 1: lambda x : re_clean.sub('', x) }
            )

            processor.swap()

        with open(state_0_filename, 'wt') as fp:
            processor.save(fp)

    return processor

if __name__ == '__main__':
    for source_text, target_text in gen_data():
        print (source_text)
        print (target_text)
        print ()



