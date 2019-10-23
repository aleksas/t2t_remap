from re_map import Processor
import urllib.request 
from re import compile
from os.path import isfile

def next(span_map, source_len_limit, target_len_limit):
    span_map_length = len(span_map)
    for start in range(span_map_length):
        for end in range(start, span_map_length):
            source_start = span_map[start][0][0]
            source_end = span_map[end][0][1]
            target_start = span_map[start][1][0]
            target_end = span_map[end][1][1]
            if ( source_end - source_start < source_len_limit and 
                target_end - target_start < source_len_limit) :
                continue
            else:
                yield ((source_start, source_end), (target_start, target_end))
                break

def load_processor():
    state_0_filename = 'state_%d.json' % 0
    if isfile(state_0_filename):
        with open(state_0_filename, 'rt') as fp:
            processor = Processor.load(fp)
    else:
        url = 'https://raw.githubusercontent.com/aleksas/liepa_dataset/master/other/stressed/__final_1.txt'
        text = urllib.request.urlopen( url ).read()
        text = text[:400000]
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

def gen_data():
    processor = load_processor()
    for source_span, target_span in next(processor.span_map, 100, 120):
        source_text = processor.text[source_span[0]:source_span[1]]
        target_text = processor.processed_text[target_span[0]:target_span[1]]

        yield source_text, target_text

if __name__ == '__main__':
    for source_text, target_text in gen_data():
        print (source_text)
        print (target_text)
        print ()



