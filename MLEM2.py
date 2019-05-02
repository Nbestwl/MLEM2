'''
Project Title: EECS 839 MLEM2 programming project
Author: Lei Wang
Date: 04/30/2019
@ALl rights reserved
'''
import sys
import pandas as pd
import numpy as np
import itertools
from operator import itemgetter


TESTFILE = "test.txt"
TEST1 = "test1.txt"
FILE = "m-global.txt"
IRIS = "iris-35-h.txt"
FINAL = "final.txt"


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))


def parsing(file_name):
    # read in every line into an array
    dataset = np.array([])
    with open(file_name, 'r') as f:
        # delete the first row of the data set
        lines = f.readlines()[1:]
        # remove any special characters from the original data strings
        lines = map(lambda x: x.strip('[]\t\n\r'), lines)
        # replace any space with a single space from the data strings
        lines = map(lambda x: ' '.join(x.split()), lines)
        # iterate every line and insert them into a 2d array
        for line in lines:
            # skip empty lines and line with !
            if not line.strip() or line.startswith('!') or line.startswith('<'):
                continue
            column_size = len(line.split())
            line = np.array(list(line.split()))
            # initialized the dataset array if it is empty
            if dataset.size is 0:
                dataset = line.reshape(1, column_size)
            # otherwise stack the rest of the lines with the dataset array
            else:
                dataset = np.vstack([dataset, line])

#     print dataset
    # close the input file
    f.close()
    return dataset


def attribute_value_pair(dataset):
    # ====================================
    # calculate the cutpoints
    # ====================================
    pairs, unique_attributes, unique = {}, [], ()
    decisions = dataset[:, -1].T
    for column in dataset[:, 0:-1].T:
        pairs[column[0]] = {}

        # remove all the duplicated values
        attribute_list = [x for x in set(column[1:]) if x!='*' and x!='-' and x!='?']
        # check if the column is a list of numbers
        is_num = str(''.join(attribute_list)).replace(".", "").isdigit()

        if is_num:
            # sort them in ascending order
            attribute_list = sorted(map(float, attribute_list))
            # return a list of cut points in that column
            cutpoints = sorted([(a + b) / 2 for a, b in zip(attribute_list[:-1], attribute_list[1:])])
            # return a list of discretized candidates
            for cutpoint in cutpoints:
                unique_attributes.extend(((str(attribute_list[0])+'..'+str(cutpoint)),(str(cutpoint)+'..'+str(attribute_list[-1]))))
        else:
            unique_attributes = attribute_list

        k, = np.where(column[1:] == '-')
        if len(k) != 0:
            k, = np.where(decisions[1:] == decisions[k][0])
            unique = set(column[1:][k])

        for x in unique_attributes:
            temp = []

            for i, j in enumerate(column[1:]):
                if j == '*':
                    temp.append(i+1)
                    continue
                if j == '?':
                    continue
                if j == '-':
                    if x in unique:
                        temp.append(i+1)
                    else:
                        continue
                else:
                    if is_num:
                        if float(j) >= float(x.split('..')[0]) and float(j) <= float(x.split('..')[1]):
                            temp.append(i+1)
                    else:
                        if j == x:
                            temp.append(i+1)

                pairs[column[0]][x] = temp

    # return the discretized dataset
    return pairs

def f(row, attribute, low, high):
    if '..' in row['value'] and row['attribute'] == attribute:
        l = float(row['value'].split('..')[0])
        h = float(row['value'].split('..')[1])
        if l <= low and h >= high:
            val = []
        else:
            val = row.iloc[-1]
    else:

        val = row.iloc[-1]
    return val


def s(row, attribute):
    if row['attribute'] == attribute:
        val = []
    else:
        val = row.iloc[-1]
    return val


def MLEM2(blocks, decisions, alpha):
    concepts, rules = [], []
    d = np.unique(decisions[1:])
    for i in d:
        k, = np.where(decisions == i)
        concepts.append([k, i])

    for concept in concepts:
        X = concept[0]
        print 'Concepts: ', X

        G, D, fat_t, fat_j, old_G, T_block_copy = X, X, [], [], [], []

        G = G.tolist()
        D = D.tolist()

        while len(G) != 0:
            T, T_block, data, index = [], [], [], 0

            for key, value in blocks.iteritems():
                for k, v in value.iteritems():
                    intersection = list(set(v) & set(G))
                    if intersection:
                        data.append([key, k, v, intersection])
                    else:
                        data.append([key, k, v, []])
            df = pd.DataFrame(data=data, columns=['attribute', 'value', '[a,v]', '0'])

            while ((len(T_block) == 0 ) or (not set.intersection(*map(set,T_block)).issubset(set(D)))) and df.iloc[:,-1].map(lambda x: len(x)).max() != 0:
                index += 1
                length_series = df.iloc[:,-1].map(lambda x: len(x))
                max_len = length_series.max()
                min_len = df[length_series == max_len]['[a,v]'].map(lambda x: len(x)).min()
                candidate = df[(length_series == max_len) & (df['[a,v]'].map(lambda x: len(x)) == min_len)]
                candidate = candidate.iloc[0]

                T_block.append(candidate[2])
                attribute = candidate['attribute']
                value = candidate['value']

                t_block = candidate.iloc[-1]
                T.append([attribute, value])
                G = list(set(t_block) & set(G))

                if '..' not in value:
                    df[str(index)] = df.apply(s, args=(attribute,), axis=1)
                else:
                    low = float(value.split('..')[0])
                    high = float(value.split('..')[1])

                    print low, high

                    df[str(index)] = df.apply(f, args=(attribute, low, high), axis=1)

                df.iloc[:,-1] = df.iloc[:,-1].apply(lambda x: list(set(t_block) & set(x)))

                print '\n'
                print df
                print '\n'

            intersection = set.intersection(*map(set,T_block))
            condition_prob = len(list(intersection & set(X))) / float(len(intersection))
            print 'condition_prob', condition_prob
            if condition_prob >= alpha:
                for x in intersection:
                    if x not in D:
                        D.append(x)
                fat_t.append(T)
            else:
                if T not in fat_j:
                    fat_j.append(T)

            print 'D', D
            print 'fat_t', fat_t
            print 'fat_j', fat_j

            union = set.intersection(*map(set, T_block))
            G = [x for x in D if x not in list(old_G) + list(set(union) - set(old_G))]

            print 'new G', G

            T_block_copy.append(T_block)
            for x in union:
                old_G.append(x)


        for i, T in enumerate(zip(fat_t, T_block_copy)):
            temp = 0
            while temp < len(T[1]):
                res = []
                li = T[1][:temp] + T[1][temp+1:]
                for i in range(len(li)):
                    if i+1 < len(li):
                        res = list(set(li[i]) & set(li[i+1]))

                if set(res).issubset(D) and len(T[1]) != 1:
                    del T[1][temp]
                    del T[0][temp]
                else:
                    temp += 1

            numerical_attributes = [x for x in T[0] if '..' in x[1]]
            non_numerical_attribuets = [x for x in T[0] if '..' not in x[1]]

            if len(numerical_attributes) != 0:
                low, high = [], []
                for x in numerical_attributes:
                    attribute = x[0]
                    low.append(float(x[1].split('..')[0]))
                    high.append(float(x[1].split('..')[1]))

                new_pair = [attribute, str(max(low)) + '..' + str(min(high))]
                fat_t[i] = [new_pair] + non_numerical_attribuets

            rules.append([T[0], concept[1]])

            print 'result ',T[1]
            print 'after dropping condition ',T[0]

    return rules


if __name__ == '__main__':
    in_f, alpha, out_f = sys.argv[1], sys.argv[2], sys.argv[3]

    data = parsing(str(in_f))
    blocks = attribute_value_pair(data)

    decisions = data[:, -1].T
    rules = MLEM2(blocks, decisions, float(alpha))

    result = []
    for line in rules:
        concept = line[1]

        temp1 = ''
        for i in range(len(line[0])):
            if i+1 != len(line[0]):
                temp = '( {}, {} )'.format(line[0][i][0], line[0][i][1])
                temp1 += temp + ' & '
            else:
                temp = '( {}, {} )'.format(line[0][i][0], line[0][i][1])
                temp1 += temp
            i += 1
        temp1 += '  ->  ' + '( {}, {} )'.format(decisions[0], concept)
        result.append(temp1)

    myfile = open(str(out_f), 'w')
    for line in result:
        myfile.write(line + '\n\n')

    myfile.close()
