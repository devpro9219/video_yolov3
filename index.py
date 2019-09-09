# import the necessary packages
import time
import cv2
import numpy as np
from myWebCamVideoStream import WebcamVideoStream
from CentroidTrackerTime import CentroidTrackerThread


_refPt = []
weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"
cv2.ocl.setUseOpenCL(True)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
if __name__ == "__main__":
    # load the COCO class labels our YOLO model was trained on
    labelsPath = "coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    vs=cv2.VideoCapture("rtsp://xxx/onvif/profile2/media.smp")

    writer = None
    (W, H) = (None, None)

    # try to determine the total number of frames in the video file
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    fps=0
    label=""
    if int(major_ver) < 3:
        fps =vs.get(cv2.cv.CV_CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps =vs.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    # loop over frames from the video file stream
    ct = CentroidTrackerThread(2,fps)
    frameNo =1
    #pool=ThreadPool(4)
    #frame = vs.read()
    # if frame is not None:
    #     setRoi(frame)
    out = cv2.VideoWriter('result.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
    roi = None

    while True:
        # read the next frame from the file

        vs.set(1, frameNo)
        _,frame = vs.read()

        #frame = frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if frame is None:
            break

        frame = cv2.resize(frame, (640, 480))


        if roi is  None:
            #  Returns X Y W H
            roi = cv2.selectROI(frame)


        rects = []
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities

        imCrop = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        H,W = imCrop.shape[:2]
        print(roi)

        blob = cv2.dnn.blobFromImage(imCrop, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.75:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    print(centerY,centerX)
                    x = int(centerX + (roi[0]) - (width / 2))
                    y = int(centerY + (roi[1])- (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    # boxActual = (x, y, x + width, y + height)
                    # rects.append(boxActual)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0,255,0), 5)
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                if(LABELS[classIDs[i]]=='person' or LABELS[classIDs[i]]=='car'):
                    label=LABELS[classIDs[i]]
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the frame
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    boxActual = (x, y, x + w, y + h)
                    rects.append(boxActual)
                    #print(LABELS[classIDs[i]])
                    #text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                    #cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        #print(rects)
        try:
            ct.update(rects,frame,frameNo,label)
        except:
            continue

        frameNo +=(fps)
        cv2.imshow("Frame", frame)
        out.write(frame)
        key = cv2.waitKey(1) & 0xFF
        if (key == ord('q')):
            break
        # check if the video writer is None

            # some information on processing single frame

    # release the file pointers

    ct.deregisterAll(frameNo)
    print("[INFO] cleaning up...")
    vs.release()
