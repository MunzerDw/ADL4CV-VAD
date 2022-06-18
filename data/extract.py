from os import listdir
from os.path import isfile, join
import random

def extract2():
    file_name = "car_airplane.txt"
    id1 = '02691156'
    id2 = '02958343'

    onlyfiles1 = listdir(id1)
    onlyfiles2 = listdir(id2)
    all = onlyfiles1 + onlyfiles2
    random.shuffle(all)
    with open(file_name, 'w') as fp:
        for item in all:
            # write each item on a new line
            if item in onlyfiles1:
                fp.write(id1 + '/' + item + '\n')
            elif item in onlyfiles2:
                fp.write(id2 + '/' + item + '\n')
        print('Done')

def extract1():
    onlyfiles = listdir('02958343')
    with open(r'car.txt', 'w') as fp:
        for item in onlyfiles:
            # write each item on a new line
            fp.write('02958343/' + item + '\n')
        print('Done')

extract2()