import numpy as np
import cv2
import imutils
import os
import time
import matplotlib.pyplot as plt

class SocialDistanceAnalysis:

    def __init__(self):
        self.setup = True

    def overlayImage(self, Bimg, Simg, offsets):
        (x_offset, y_offset) = offsets
        Bimg[y_offset:y_offset+Simg.shape[0], x_offset:x_offset+Simg.shape[1]] = Simg

        return Bimg

    def generatePieChart(self, safe, risk):
        fig, ax = plt.subplots(figsize=(4,4))
        ax.pie([safe, risk], labels = ['Safe', 'Risk'])
        ax.legend()
        plt.savefig("pie.png", transparent=True, dpi=60)
        plt.close()

    def populateDashboard(self, emptyDash, safe, risk):

        totalPeople = safe + risk

        self.generatePieChart(safe, risk)

        plt = cv2.imread("pie.png")

        plt = imutils.resize(plt, width=int(self.videoShape[1]/4))

        pieoffsets = self.overlayImage(emptyDash, plt, (10,50))

        totalOffsets = [(10,int(emptyDash.shape[0]/1.5) + (self.totalIcon.shape[0] * 0) -int(self.totalIcon.shape[0]/2)), ((10+self.totalIcon.shape[1]+10), int(emptyDash.shape[0]/1.5))]
        safeOffsets = [(10,int(emptyDash.shape[0]/1.5) + 5 +(self.totalIcon.shape[0] * 1) -int(self.totalIcon.shape[0]/2)), ((10+self.totalIcon.shape[1]+10), int(emptyDash.shape[0]/1.5) + 5 +(self.totalIcon.shape[0]))]
        riskOffsets = [(10,int(emptyDash.shape[0]/1.5) + 10 +(self.totalIcon.shape[0] * 2) -int(self.totalIcon.shape[0]/2)), ((10+self.totalIcon.shape[1]+10), int(emptyDash.shape[0]/1.5) + 5 +(self.totalIcon.shape[0]*2))]

        emptyDash = self.overlayImage(emptyDash, self.totalIcon, totalOffsets[0])
        cv2.putText(emptyDash, str(totalPeople), totalOffsets[1], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        emptyDash = self.overlayImage(emptyDash, self.safeIcon, safeOffsets[0])
        cv2.putText(emptyDash, str(safe), safeOffsets[1], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        emptyDash = self.overlayImage(emptyDash, self.riskIcon, riskOffsets[0])
        cv2.putText(emptyDash, str(risk), riskOffsets[1], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        return emptyDash

    def findDistance(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + 550 / ((p1[1] + p2[1]) / 2) * (p1[1] - p2[1]) ** 2) ** 0.5


    def isClose(self, p1,  p2):
        dist = self.findDistance(p1, p2)
        calibration = (p1[1] + p2[1]) / 2
        
        if 0 < dist < 0.25 * calibration:
            return True
        else:
            return False
        

    def SDASetup(self, yolopath, assetpath, videoShape):
        if self.setup:
            labelsPath = os.path.sep.join([yolopath, "coco.names"])
            self.LABELS = open(labelsPath).read().strip().split("\n")
            weightsPath = os.path.sep.join([yolopath, "yolov3.weights"])
            configPath = os.path.sep.join([yolopath, "yolov3.cfg"])
            print("[INFO] loading YOLO from disk...")
            self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
            self.ln = self.net.getLayerNames()
            self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            (self.H, self.W) = (None, None)

            self.videoShape = videoShape
            self.assetPath = assetpath

            blankDashboard = np.ones((videoShape[0],int(videoShape[1]/3),3))
            blankDashboard.fill(255)
            blankDashboardPath = assetpath + 'blankDashboard.png'
            cv2.imwrite(blankDashboardPath, blankDashboard)
            self.blankDashboard = cv2.imread(blankDashboardPath)
            cv2.putText(self.blankDashboard, "Social Distancing", (20,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(self.blankDashboard, "Dashboard", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(self.blankDashboard, "Sashank Kakaraparty", (10,45), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (0, 0, 0), 1)

            self.safeIcon = cv2.imread(assetpath + 'safeIcon.png')
            self.riskIcon = cv2.imread(assetpath + 'riskIcon.png')
            self.totalIcon = cv2.imread(assetpath + 'totalIcon.png')

            self.setup = False
        else:
            pass

    def SDAProcess(self, image):
        frame = image.copy()
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
        self.net.setInput(blob)
        tick = time.time()
        layerOutputs = self.net.forward(self.ln)
        tock = time.time()
        print("Processing at {:.4f} per frame".format((tock-tick))) #Time taken for detection
        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:

            for detection in output:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if self.LABELS[classID] == "person":

                    if confidence > 0.5:

                        box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        riskPeople = 0
        safePeople = 0

        if len(idxs) > 0:

            idf = idxs.flatten()
            statusList = []
            closePairsList = []
            centerList = []
            distanceList = []

            for i in idf:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                centerList.append([int(x + w / 2), int(y + h / 2)])

                statusList.append(False)

            for i in range(len(centerList)):
                for j in range(len(centerList)):
                    close = self.isClose(centerList[i], centerList[j])

                    if close:

                        closePairsList.append([centerList[i], centerList[j]])
                        statusList[i] = True
                        statusList[j] = True

            totalPeople = len(centerList)
            riskPeople = statusList.count(True)
            safePeople = statusList.count(False)
            index = 0

            for i in idf:

                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                if statusList[index] == True:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)

                elif statusList[index] == False:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                index += 1

            for h in closePairsList:
                cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)

        self.processedImg = frame.copy()
        self.safePeople = safePeople
        self.riskPeople = riskPeople


    def SDADisplay(self):
        dashboard = self.populateDashboard(self.blankDashboard.copy(), self.safePeople, self.riskPeople)
        finalImg = np.concatenate((self.processedImg, dashboard), axis=1)

        return finalImg


filename = "videos/video_1.mp4"

opname = "outputs/output_" + filename.split('/')[1][:-4] + '.avi'

assetpath = "assets/"
yolopath = "yolo-coco/"

cap = cv2.VideoCapture(filename)
fno = 0
writer = None

AnalysisObj = SocialDistanceAnalysis()

fulltick = time.time()

while(True):

    ret, frame = cap.read()

    if not ret:
        break
    currentImg = frame.copy()
    currentImg = imutils.resize(currentImg, width=480)
    videoShape = currentImg.shape
    fno += 1


    #We don't have to run the process for every frame, it wouldn't be very productive. 
    if(fno%2 == 0 or fno == 1):
        AnalysisObj.SDASetup(yolopath, assetpath, videoShape)
        AnalysisObj.SDAProcess(currentImg)
        outputFrame = AnalysisObj.SDADisplay()

        # cv2.imshow('SocialDistanceDashboard', outputFrame)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(opname, fourcc, 30,
                (outputFrame.shape[1], outputFrame.shape[0]), True)

    writer.write(outputFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fulltock = time.time()

print("Full time taken {} minutes".format((fulltock-fulltick)/60))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()