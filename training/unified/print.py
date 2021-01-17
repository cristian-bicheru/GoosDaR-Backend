import os

with open("ids.txt", "w") as f:
    f.write('\n'.join([x.split(".xml")[0] for x in os.listdir() if x.endswith(".xml")]))
