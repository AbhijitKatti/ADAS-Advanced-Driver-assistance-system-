import numpy as np
import cv2
import utlis
import time

cameraFeed= False
videoPath = 'test_videos/2.mp4'
cameraNo= 0
frameWidth= 640
frameHeight = 480

if cameraFeed:intialTracbarVals = [24,55,12,100] #  #wT,hT,wB,hB
else:intialTracbarVals = [42,63,14,87]   #wT,hT,wB,hB

if cameraFeed:
    cap = cv2.VideoCapture(cameraNo)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
else:
    cap = cv2.VideoCapture(videoPath)
count=0
noOfArrayValues =10
global arrayCurve, arrayCounter
arrayCounter=0
arrayCurve = np.zeros([noOfArrayValues])
myVals=[]
utlis.initializeTrackbars(intialTracbarVals)
font = cv2.FONT_HERSHEY_SIMPLEX
global flag

def draw_lanes(img, left_fit, right_fit,frameWidth,frameHeight,src):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    color_img = np.zeros_like(img)

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))
    if flag == 0:
        cv2.fillPoly(color_img, np.int_(points), (0, 255, 0))
    else:
        cv2.fillPoly(color_img, np.int_(points), (0, 0, 255))
    inv_perspective = utlis.inv_perspective_warp(color_img,(frameWidth,frameHeight),dst=src)
    inv_perspective = cv2.addWeighted(img, 0.5, inv_perspective, 0.7, 0)
    return inv_perspective
size = (640, 480)
result = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'),25, size)

starting_time= time.time()
frame_id = 0

while True:

    success, img = cap.read()
    frame_id += 1
    #img = cv2.imread('test3.jpg')
    if cameraFeed== False:img = cv2.resize(img, (frameWidth, frameHeight), None)
    imgWarpPoints = img.copy()
    imgFinal = img.copy()
    imgCanny = img.copy()
    imgUndis = utlis.undistort(img)
    imgThres,imgCanny,imgColor,imgGray = utlis.thresholding(imgUndis)
    src = utlis.valTrackbars()
    imgWarp = utlis.perspective_warp(imgThres, dst_size=(frameWidth, frameHeight), src=src)
    imgWarpPoints = utlis.drawPoints(imgWarpPoints, src)
    imgSliding, curves, lanes, ploty = utlis.sliding_window(imgWarp, draw_windows=True)

    try:
        global points
        curverad =utlis.get_curve(imgFinal, curves[0], curves[1])
        lane_curve = np.mean([curverad[0], curverad[1]])
        # ## Average
        currentCurve = lane_curve // 50
        if  int(np.sum(arrayCurve)) == 0:averageCurve = currentCurve
        else:
            averageCurve = np.sum(arrayCurve) // arrayCurve.shape[0]
        curve = int(averageCurve)
        if curve > 30:
            directionText='Right'
            flag = 1
        if curve < -30:
            directionText='Left'
            flag = 1

        if curve <30 and curve > -30:
            directionText='Straight'
            flag = 0
        if curve == -1000000:
            directionText = 'No Lane Found'
            flag = 0
        imgFinal = draw_lanes(img, curves[0], curves[1],frameWidth,frameHeight,src=src)
        if abs(averageCurve-currentCurve) >200: arrayCurve[arrayCounter] = averageCurve
        else :arrayCurve[arrayCounter] = currentCurve
        arrayCounter +=1
        if arrayCounter >=noOfArrayValues : arrayCounter=0
        cv2.putText(imgFinal, str(int(averageCurve)), (frameWidth//2-70, 70), cv2.FONT_HERSHEY_DUPLEX, 1.75, (0, 0, 255), 2, cv2.LINE_AA)

    except:
        lane_curve=00
        pass

    imgFinal= utlis.drawLines(imgFinal,lane_curve)
    cv2.putText(imgFinal, directionText,(30,50), font, 1, (0, 200, 200), 2, cv2.LINE_AA)
    result.write(imgFinal)
    imgThres = cv2.cvtColor(imgThres,cv2.COLOR_GRAY2BGR)
    imgBlank = np.zeros_like(img)
    imgStacked = utlis.stackImages(0.7, ([img,imgUndis,imgWarpPoints],
                                         [imgColor, imgCanny, imgThres],
                                         [imgWarp,imgSliding,imgFinal]
                                         ))
    
    imgStacked=cv2.resize(imgStacked,(800,600))
    elapsed_time = time.time() - starting_time
    fps=frame_id/elapsed_time
    cv2.putText(imgFinal,"FPS:"+str(round(fps,2)),(450,50),font,1,(0,255,0),2)
    cv2.imshow("Steps",imgStacked)
    cv2.imshow("Output", imgFinal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
result.release()
print("[INFO] file Saved")
cv2.destroyAllWindows()
