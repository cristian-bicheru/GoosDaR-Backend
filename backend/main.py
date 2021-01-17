from flask import Flask, jsonify, request, send_file
from flask_ngrok import run_with_ngrok
from PIL import Image
import time
server = Flask(__name__)
run_with_ngrok(server)
from sightings import Sightings
import predictor
import uuid
import json
import socket
print(socket.gethostbyname(socket.getfqdn(socket.gethostname())))

sightings = Sightings()

@server.route("/still-data", methods = ["POST"])
def load_still():
    rtime = time.time()
    data = request.files.get('still')
    img = Image.open(data.stream)
    sid = str(uuid.uuid1().int)
    img.save("stills/"+sid+".png")
    jsondata = json.loads(request.form['json'])
    num_geese, bbox = predictor.infer(sid)
    if num_geese > 0:
      sightings.add(sid, rtime, jsondata["location"], jsondata["telemetry"], num_geese, bbox)
    return "Success", 200

@server.route("/sightings", methods = ["GET"])
def get_sightings():
    sightings.refresh()
    return jsonify(sightings.serialize())

@server.route("/get-still", methods = ["GET"])
def get_still():
    sid = request.args.get('id')
    return send_file("stills/"+sid+".png", mimetype="image/png")

@server.route("/get-still-boxed", methods = ["GET"])
def get_still_with_box():
    sid = request.args.get('id')
    return send_file("stills/"+sid+"_overlay.png", mimetype="image/png")

@server.route("/", methods = ["GET"])
def get():
    return "TEST"

server.run()
