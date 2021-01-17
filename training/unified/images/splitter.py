import os
import random

os.chdir('train')

files = os.listdir()
random.shuffle(files)

traincutoff = int(len(files)*0.6)
testcutoff = int(len(files)*0.8)
test = files[traincutoff:testcutoff]
validation = files[testcutoff:]

for file in test:
    os.rename(file, '../test/'+file)

for file in validation:
    os.rename(file, '../validation/'+file)
