# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np


def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    f = open(train_file)
    lines = f.read().rstrip().split('\n')

    tags = {}
    words = {}

    smoothing_constant = 0.005

    word_pr = np.zeros((47, 150000))
    tag_pr = np.zeros((47, 47))
    cap_pr = np.zeros((47, 3))
    end_pr = np.zeros((47, 7))

    word_pr.fill(smoothing_constant)
    tag_pr.fill(smoothing_constant)

    tags['<s>'] = 0
    tags['</s>'] = 1
    words['UNK'] = 0

    for line in lines:
        pairs = line.split()
        prev_tag = '<s>'

        for pair in pairs:
            pair = pair.rsplit('/', 1)
            word = pair[0]
            tag = pair[1]

            if not tag in tags:
                tags[tag] = len(tags)

            if not word in words:
                words[word] = len(words)

            word_pr[tags[tag], words[word]] += 1
            tag_pr[tags[prev_tag], tags[tag]] += 1

            if word.isupper():
                cap_pr[tags[tag], 0] += 1
            elif not word.islower():
                cap_pr[tags[tag], 1] += 1
            else:
                cap_pr[tags[tag], 2] += 1

            if word.endswith('s'):
                end_pr[tags[tag], 0] += 1
            elif word.endswith('ed'):
                end_pr[tags[tag], 1] += 1
            elif word.endswith('ing'):
                end_pr[tags[tag], 2] += 1
            elif word.endswith('ion'):
                end_pr[tags[tag], 3] += 1
            elif word.endswith('al'):
                end_pr[tags[tag], 4] += 1
            elif word.endswith('ive'):
                end_pr[tags[tag], 5] += 1
            else:
                end_pr[tags[tag], 6] += 1

            prev_tag = tag

        tag_pr[tags[prev_tag], tags['</s>']] += 1

    f.close()

    word_pr = word_pr[:len(tags),:len(words)]
    tag_pr = tag_pr[:len(tags),:len(tags)]
    cap_pr = cap_pr[:len(tags),:]
    end_pr = end_pr[:len(tags),:]

    word_pr = np.divide(word_pr, word_pr.sum(axis=0))
    tag_pr = np.divide(tag_pr, tag_pr.sum(axis=0))

    with np.errstate(divide='ignore', invalid='ignore'):
        cap_pr = np.nan_to_num(np.divide(cap_pr, cap_pr.sum(axis=1).reshape((len(tags), 1))))
        end_pr = np.nan_to_num(np.divide(end_pr, end_pr.sum(axis=1).reshape((len(tags), 1))))
    
    np.savez(model_file, tags=tags, words=words, word_pr=word_pr, tag_pr=tag_pr, cap_pr=cap_pr, end_pr=end_pr)

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
