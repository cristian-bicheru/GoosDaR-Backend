import os

test = [x.split('.')[0] for x in os.listdir("images/test")]
validation = [x.split('.')[0] for x in os.listdir("images/validation")]

os.chdir('annotations/train')

for file in test:
    os.rename(file+'.xml', '../test/'+file+'.xml')

for file in validation:
    os.rename(file+'.xml', '../validation/'+file+'.xml')
