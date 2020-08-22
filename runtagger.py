# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np


def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    model_path = model_file if model_file.endswith('.npz') else model_file + '.npz'
    f = np.load(model_path, allow_pickle=True)
    tags = f['tags'][()]
    words = f['words'][()]
    word_pr = f['word_pr']
    tag_pr = f['tag_pr']
    cap_pr = f['cap_pr']
    end_pr = f['end_pr']

    fr = open(test_file)
    lines = fr.read().rstrip().split('\n')

    fw = open(out_file, 'w+')
    res = ''

    for line in lines:
        observations = line.split()
        res += viterbi(observations, tags, words, word_pr, tag_pr, cap_pr, end_pr)

    fw.write(res)
    fw.close()
    fr.close()

def calculate_unknown(word, tag_index, unk_index, word_pr, tag_pr, cap_pr, end_pr):
    cap = cap_pr[tag_index, 0] if word.isupper() else cap_pr[tag_index, 1] if not word.islower() else cap_pr[tag_index, 2]

    end = 1
    if word.endswith('s'):
        end = end_pr[tag_index, 0]
    elif word.endswith('ed'):
        end = end_pr[tag_index, 1]
    elif word.endswith('ing'):
        end = end_pr[tag_index, 2]
    elif word.endswith('ion'):
        end = end_pr[tag_index, 3]
    elif word.endswith('al'):
        end = end_pr[tag_index, 4]
    elif word.endswith('ive'):
        end = end_pr[tag_index, 5]
    else:
        end = end_pr[tag_index, 6]

    return word_pr[tag_index, unk_index] * cap * end

def viterbi(observations, tags, words, word_pr, tag_pr, cap_pr, end_pr):
    N = len(tags)
    T = len(observations)
    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))

    for s in tags:
        p = calculate_unknown(observations[0], tags[s], words['UNK'], word_pr, tag_pr, cap_pr, end_pr) if not observations[0] in words else word_pr[tags[s], words[observations[0]]]
        viterbi[tags[s], 0] = tag_pr[0, tags[s]] * p
        backpointer[tags[s], 0] = 0

    for t in range(1, T):
        for s in tags:
            p = calculate_unknown(observations[t], tags[s], words['UNK'], word_pr, tag_pr, cap_pr, end_pr) if not observations[t] in words else word_pr[tags[s], words[observations[t]]]
            viterbi[tags[s], t] = np.max(viterbi[:, t-1] * tag_pr[:, tags[s]]) * p
            backpointer[tags[s], t] = np.argmax(viterbi[:, t-1] * tag_pr[:, tags[s]])

    viterbi[tags['</s>'], T-1] = np.max(viterbi[:, T-1] * tag_pr[:, tags['</s>']])
    backpointer[tags['</s>'], T-1] = np.argmax(viterbi[:, T-1] * tag_pr[:, tags['</s>']])

    line = ''

    inv_tags = {v: k for k, v in tags.items()}
    ptr = backpointer[tags['</s>'], T - 1]

    for i in range(T):
        line = observations[T - 1 - i] + '/' + inv_tags[ptr] + ' ' + line
        ptr = backpointer[int(ptr), T - 1 - i]
    
    line.strip()
    line += '\n'

    return line

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
