import csv
import numpy as np
from scipy.sparse import csr_matrix
from sknetwork.path import shortest_path

from collections import defaultdict
import functools
import itertools

counter = 0
fileName = "Data_Cleaned.csv"

trainName = defaultdict(functools.partial(next, itertools.count()))

with open(fileName, newline='', encoding='UTF-8') as file:
    reader = csv.reader(file, delimiter=',')
    file.seek(0)
    next(reader)
    for row in reader:
        index1 = trainName[row[1]]
        index2 = trainName[row[2]]
    trainNumber = len(trainName)
    trips = np.zeros((trainNumber, trainNumber))
    file.seek(0)
    next(reader)
    for row in reader:
        counter += 1
        index1 = trainName[row[1]]
        index2 = trainName[row[2]]
        indexT = (index1, index2) if index1 < index2 else (index2, index1)
        if trips[indexT] != 0:
            print(f"Voyage {row[1]} - {row[2]} avec uen distance {row[3]} a deja ete calculé {trips[indexT]}..")
        else:
            trips[indexT] = int(row[3])

trainStationId = dict((id, name) for name, id in trainName.items())

i_lower = np.tril_indices(trainNumber, -1)
trips[i_lower] = trips.T[i_lower]

VoyageGraph = csr_matrix(trips)


class Trip:
    def __init__(self, startTrainId, EndTrainId, path, totalDuration):
        self.startTrainId = startTrainId
        self.EndTrainId = EndTrainId
        self.path = path
        if totalDuration is None:
            self.totalDuration = None
        else:
            self.totalDuration = int(totalDuration)

    def __str__(self):
        return f"Voyage de {trainStationId[self.startTrainId]} a {trainStationId[self.EndTrainId]} pour une durée de  {self.totalDuration} minutes , chemin suivant : {self.pathToString()}"

    def pathToString(self):
        string = ""
        for i in range(len(self.path)):
            if i > 0:
                string = string + " -> "
            string = string + trainStationId[self.path[i]]
        return string


def getPathIds(trainStartID: list, trainStationEndIds: list):
    global VoyageGraph
    paths = []
    for startId in trainStartID:
        if startId in trainStationEndIds:
            return [int(startId), int(startId)]
    if(len(trainStartID) > 1 and len(trainStationEndIds) > 1):
        for trainStationEndId in trainStationEndIds:
            results = shortest_path(VoyageGraph, sources=[int(i) for i in trainStartID], targets=[int(trainStationEndId)], method='D')
            for result in results:
                if len(result) >= 2:
                    paths.append(result)
        return paths
    else:
        results = shortest_path(VoyageGraph, sources=[int(i) for i in trainStartID], targets=[int(i) for i in trainStationEndIds], method='D')
        for result in results:
            if len(result) >= 2:
                paths.append(result)
        return paths


def getSameCities(city: str, key: str):
    return city.lower() in key.lower()


def getPath(start: str, end: str):
    trainStartID = np.array([])
    trainStationEndIds = np.array([])
    # Recuperer tout les gars portant le nom chercher
    for key, value in trainName.items():
        if getSameCities(start, key):
            trainStartID = np.append(trainStartID, value)
        if getSameCities(end, key):
            trainStationEndIds = np.append(trainStationEndIds, value)
    if len(trainStartID) > 0 and len(trainStationEndIds) > 0:
        return getPathIds(trainStartID, trainStationEndIds)
    else:
        return np.array([])


def getBestPath(tripCityWaypoints: list):
    global VoyageGraph
    fullTrip = np.array(np.zeros(len(tripCityWaypoints)-1), dtype=object)
    # boucler sur tout les sous voyages
    for trip in range(len(fullTrip)):
        paths = getPath(tripCityWaypoints[trip], tripCityWaypoints[trip+1])
        minDistance = None
        keptPath = None
        startId = None
        endId = None
        for path in paths:
            distance = 0
            # Plusieurs valeurs possible ( start / end )
            if isinstance(path, list):
                for i in range(len(path)-1):
                    distance = distance + VoyageGraph[(path[i], path[i+1])]
                if minDistance is None or distance < minDistance:
                    minDistance = distance
                    keptPath = path
                    startId = path[0]
                    endId = path[len(path)-1]
            # Une seule valeur possible
            else:
                for i in range(len(paths)-1):
                    distance = distance + VoyageGraph[(paths[i], paths[i+1])]
                minDistance = distance
                keptPath = paths
                startId = paths[0]
                endId = paths[len(paths)-1]
        fullTrip[trip] = Trip(startId, endId, keptPath, minDistance)
    return fullTrip


# Fonction pour tester le pathfinding
def pathfinding(start, stop):
    bestTrips = getBestPath([start, stop])
    for i in range(len(bestTrips)):
        if bestTrips[i].path is not None:
            print(f"#{i+1} - {bestTrips[i]}")
        else:
            if bestTrips[i].startTrainId is None or bestTrips[i].EndTrainId is None:
                print("Aucun chemin trouvé pour les gares renseignées.")
            else:
                print("Aucun chemin trouvé pour les gares renseignées.")
