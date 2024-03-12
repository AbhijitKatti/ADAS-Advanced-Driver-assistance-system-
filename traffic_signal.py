import cv2  
import numpy as np
import time
videoPath = 'test_videos/5.mp4'
net = cv2.dnn.readNet("signal_model/signal.weights","signal_model/signal.cfg")
classes = []
with open("signal_model/coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#loading image
cap=cv2.VideoCapture(videoPath) #0 for 1st webcam

font = cv2.FONT_HERSHEY_PLAIN
starting_time= time.time()
frame_id = 0
obj_id = 0

while True:
    _,frame= cap.read() #
    frame_id+=1
    
    height,width,channels = frame.shape
    #detecting objects
    blob = cv2.dnn.blobFromImage(frame,0.00392,(224,224),(0,0,0),True,crop=False) #acc   

        
    net.setInput(blob)
    outs = net.forward(outputlayers)
    #print(outs[1])


    #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids=[]
    confidences=[]
    boxes=[]
    TrackedIDs = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                #object detected
                
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #get ID
                Id = int(obj_id)                
                #rectangle co-ordinaters
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                
                #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object tha was detected

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence= confidences[i]
            cropped_image = frame[y:(y+h),x:(x+w)]
            hsv_frame = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
            light = ""
            # Red color
            low_red = np.array([161, 155, 84])
            high_red = np.array([179, 255, 255])
            red_mask = cv2.inRange(hsv_frame, low_red, high_red)
            red = cv2.bitwise_and(cropped_image, cropped_image, mask=red_mask)
            numRPixels = cv2.countNonZero(red_mask)
            #cv2.imshow("mask",red)
            
            #green color
            low_green = np.array([25, 52, 72])
            high_green = np.array([102, 255, 255])
            green_mask = cv2.inRange(hsv_frame, low_green, high_green)
            green = cv2.bitwise_and(cropped_image, cropped_image, mask=green_mask)
            numGPixels = cv2.countNonZero(green_mask)
            #cv2.imshow("green",green)
            
            if(numRPixels > 1000):
                light = "Red"
                rect = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)
                
            elif(numGPixels > 1000):
                light = "Green"
                rect = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
                
            else:
                light = "Amber"
                rect = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),4)
                
            
            cv2.putText(frame,light +" " +str(round(confidence,2)), (x, y), font, 1.5, (255,0,212), 2)
            #cv2.imshow("croppped",cropped_image)
            
    elapsed_time = time.time() - starting_time
    fps=frame_id/elapsed_time
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(500,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("Image",frame)
    
    key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
    
    if key == 27: #esc key stops the process
        break;
    
cap.release()    
cv2.destroyAllWindows()
