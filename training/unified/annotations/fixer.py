import os

def fix(dat):
    for i in range(len(dat)):
        if "path" in dat[i]:
            dat[i] = dat[i].replace("/media/biscuit/HARDDRIVE/goose_dataset/unified/images/validation/", "/media/biscuit/HARDDRIVE/goose_dataset/unified/images/train/")
            dat[i] = dat[i].replace("/media/biscuit/HARDDRIVE/goose_dataset/unified/images/validation/", "/media/biscuit/HARDDRIVE/goose_dataset/unified/images/train/")
        elif "name" in dat[i]:
            dat[i] = dat[i].replace("goose-head", "goose")

    return dat

for file in os.listdir():
    if file.endswith(".xml"):
        with open(file, "r") as f:
            data = f.readlines()
        with open(file, "w") as f:
            f.write(''.join(fix(data)))
