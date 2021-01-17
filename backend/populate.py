import requests
import os
os.chdir("/media/biscuit/HARDDRIVE/backend")
import random
import json

url = 'https://05351de26b56.ngrok.io/'

files = ['g5.jpg', 'g19.jpg', 'g25.jpg', 'g35.jpg', 'download.png', '089_0031.jpg', 'goose54.jpg', 'goose189.jpg', 'goose190.jpg']

#files = ['g60.jpg', 'g73.jpg', 'g78.jpg', 'goose68.jpg']

for file in files:
    d1 = (random.random()-0.5)*0.018
    d2 = (random.random()-0.5)*0.018
    files = {'still':open(file, "rb"), 'json':(None, json.dumps({"location":[43.471589+d1, -80.545299+d2], "telemetry":"N/A"}), 'application/json')}
    requests.post(url+'still-data', files=files)
