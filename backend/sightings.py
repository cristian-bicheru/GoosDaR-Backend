import time

SIGHTING_DROP_DURATION = 3*60*60 # 3 hours

class Sightings:
    def __init__(self):
        self.data = []

    def add(self, still_id, time, gps, telemetry, num_geese, bbox):
        self.data.append({
            "id"            :   still_id,
            "time"          :   time,
            "location"      :   gps,
            "telemetry"     :   telemetry,
            "geesestimate"  :   num_geese,
            "bbox"          :   bbox
            })

    def refresh(self):
        pop = []
        rtime = time.time()
        
        for i in range(len(self.data)):
            if rtime-self.data[i]["time"] > SIGHTING_DROP_DURATION:
                pop.append(i)

        for i in pop[::-1]:
            self.data.pop(i)

    def serialize(self):
        return self.data
