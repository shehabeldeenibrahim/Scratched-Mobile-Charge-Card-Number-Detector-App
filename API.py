from imutils import contours
import uuid
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert
from skimage.morphology import skeletonize, thin
from keras.models import load_model
import pickle



from flask import Flask, jsonify, request
# import the necessary packages
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import matplotlib.pyplot as plt
import base64
from PIL import Image
from io import StringIO
import io
from flask_cors import CORS
import requests
import json
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    open_cv_image = np.array(image) 
    return open_cv_image

def convert(list): 
      
    # Converting integer list to string list 
    s = [str(i) for i in list] 
      
    # Join list items using join() 
    res = int("".join(s)) 
      
    return(res) 
  

def readb64(base64_string):
    # r = base64.b64decode(base64_string)
    # q = np.frombuffer(r, dtype=np.uint8)
    a = stringToImage(base64_string)
    #img = cv2.imdecode(q, -1)
    q = toRGB(a)
    return q

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def getNumbersML(rects):
    from keras.preprocessing.image import load_img
    import numpy as np
    import cv2
    from keras.models import load_model
    final_numbers = []
    model=load_model('ML/Digit_CNN_Final_Model.sav')
    k = 1
    for i in rects:
        img=cv2.imread('images/numbers/'+ str(k) +'.jpg')
        #img=cv2.resize(img,(28,28),interpolation=cv2.INTER_LINEAR)
        #img = img.reshape(28,28,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        (thresh, im_bw) = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow("a",im_bw)
        # cv2.waitKey(0)
        img=np.array([img])
        im_bw = im_bw.reshape(1,28,28,1)
        prediction=model.predict(im_bw)
        max_value=np.amax(prediction[0])
        result=np.where(prediction[0]==max_value)
        final_numbers.append(result[0][0])
        k+=1
        if(k>17):
            break
    return final_numbers

def main(base64String):
    # with open('data.txt', 'r') as file:
    #     data = file.read()
    image_orig = readb64(base64String)
    # cv2.imshow("A", image_orig)
    # cv2.waitKey(0)
    denoised_image = cv2.fastNlMeansDenoisingColored(image_orig,None,10,10,7,21)    
    blank_image = np.zeros((image_orig.shape[0],image_orig.shape[1],3)) 
    gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(gray, (5,5), 0)
    edged = auto_canny(image)   
    screenCnt = []
    contours = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]  
    # loop over the contours
    for c in contours:
    	# approximate the contour
    	peri = cv2.arcLength(c, True)
    	approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    
    	if len(approx) == 4:
    		screenCnt = approx
    		break
    cv2.drawContours(image_orig, contours, 0, (0,255,0), 2)
    # plt.figure()
    # plt.title("picture_path")
    # plt.imshow(image_orig)
	


    if(screenCnt.__len__() == 0):
    	return(False,[])   
    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(image_orig, screenCnt.reshape(4, 2))

    if(warped.shape[0]>warped.shape[1]):
    	warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # show the original and scanned images
    # plt.figure()
    # plt.imshow(warped)
    # plt.title("picture_path")
    # plt.show()
    cv2.imwrite( "images/number.jpeg", warped )
    # cv2.waitKey(0)
    returnPicture = warped
    return True, returnPicture

def plotArray(x): #takes the array and plots the picture
    imgplot = plt.imshow(x,cmap='gray')
    plt.show()
def ArraytoPlot(x,array): #choose index of array, returns an array of one picture chosen
    temp= []
    for j in range(784):
        temp.append(array[x][j])
    temp = np.array(temp)
    temp = temp.reshape(28,28)
    return temp

def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	cv2.imshow(img)
	cv2.waitKey(0)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img


def thinAllNumbers(rects):
	import cv2
	import numpy as np
	k = 2
	for i in rects:	
		img = cv2.imread('images/numbers/'+ str(k) +'.jpg',0)
		size = np.size(img)
		skel = np.zeros(img.shape,np.uint8)
		
		ret,img = cv2.threshold(img,100,255,0)
		element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
		done = False
		
		while( not done):
			eroded = cv2.erode(img,element)
			temp = cv2.dilate(eroded,element)
			temp = cv2.subtract(img,temp)
			skel = cv2.bitwise_or(skel,temp)
			img = eroded.copy()
		
			zeros = size - cv2.countNonZero(img)
			if zeros==size:
				done = True
		
		# cv2.imshow("skel",skel)
		# cv2.waitKey(0)
		cv2.imwrite('images/numbers/'+ str(k) +'.jpg', skel)
		k+=1
		if(k>17):
			break
def getEachNumber(image):
    #Loading Model
    classifier = load_model('ML/Digit_CNN_Final_Model.sav')

    # #Path of FULL ID
    # picture_ref_path = "images/number.jpeg"

    # #Load image
    # image = cv2.imread(picture_ref_path)
    # image_to_crop = cv2.imread(picture_ref_path)
    # ref = cv2.imread(picture_ref_path)
    image = image_to_crop = ref = image

    #Decrease noise
    denoised_image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    denoised_to_crop = cv2.fastNlMeansDenoisingColored(image_to_crop,None,10,10,7,21)


    #Convert image to Gray
    image_to_crop = cv2.cvtColor(denoised_to_crop, cv2.COLOR_BGR2GRAY)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    #Show Gray original image
    # cv2.imshow("Gray original image", ref)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    #Convert to black and white image only(all less than the number takes white)
    (thresh, im_bw) = cv2.threshold(image_to_crop, 120, 255, cv2.THRESH_BINARY_INV)

    #Show number image in black and white
    cv2.imshow("B&W Number", im_bw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # find contours of image
    # sort them from left to right
    #Draw all contours on image
    refCnts = cv2.findContours(im_bw.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    refCnts = imutils.grab_contours(refCnts)
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
    cv2.drawContours(image, refCnts, -1, (0,255,0), 1)



    #Show original image with all cotours
    cv2.imshow("Image with cotours", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Array of number rectangles coordinates (x,y,w,h)
    rects = []


    # loop over the OCR-A reference contours
    for (i, c) in enumerate(refCnts):
        # compute the bounding box for the digit, extract it, and resize
        # it to a fixed size
        (x, y, w, h) = cv2.boundingRect(c)
        #show the boxes detected
        roi = ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        rects.append((x, y, w, h))
    #thinAllNumbers(rects)
    final_numbers = []
    shapes = []
    #loop over each number
    k = 0
    j = 0
    notRemoved = []
    for i in rects:

        #Get the coordinates and adjust
        x,y,w,h = rects[k]
        x = x-1
        w = w+2
        y = y-2
        h = h+3


        #show the boxes detected
        cv2.rectangle(image,(x,y),(x,y),(0,255,0),1)

        #crop the number in the black and white image
        #show it
        #crop_img = im_bw[y:y+h, x:x+w]
        reversed = 255 - im_bw
        #crop_img = reversed[y:y+h, x:x+w]
        crop_img = ref[y:y+h, x:x+w]
        shapes.append(crop_img.shape)
        if(crop_img.shape[0] < 39 and  crop_img.shape[0] > 13 and crop_img.shape[1] < 30 and crop_img.shape[1] > 13):
            notRemoved.append(1)

            #Get a name for the cropped number image to be saved
            #pic_name = str(uuid.uuid4())
            pic_name = j+1
            

            #Load image array into an PIL image
            crop_img = Image.fromarray(crop_img)

            #Resize the image to 28*28
            #crop_img.thumbnail([28,28], Image.ANTIALIAS)
            crop_img = crop_img.resize([28,28], Image.ANTIALIAS)
            #crop_img.show()
            #Save number to directory
            
            crop_img.save('images/numbers/'+str(pic_name)+".jpg")
            j+=1
            #cv2.imwrite("numbers/" +str(pic_name)+ ".jpeg", crop_img)
            crop_img = np.array(crop_img)
            #reshape image to use the classifier

            #reshape image to use the classifier
            # img=cv2.resize(crop_img,(256,256),interpolation=cv2.INTER_LINEAR)
            # img=np.array([img])
            # prediction=classifier.predict(img)
            # max_value=np.amax(prediction[0])
            # result=np.where(prediction[0]==max_value)
            # # label_pred = classifier.predict(x)
            # # predicted_values = np.argmax(label_pred,1)
            # final_numbers.append(result)
        k+=1
    final_numbers = getNumbersML(notRemoved)
    return final_numbers

app = Flask(__name__)
cors = CORS(app, resources={r"/": {"origins": "*"}})   

@app.route('/',methods=['GET','POST'])
def API():
    data = request.files['photo']
    buff = data.read()
    base64String = base64.b64encode(buff)
    # y = json.loads(data)
    #base64String = request.args.get('img') #if key doesn't exist, returns None

    flag,ret = main(base64String)

    if(not flag):
        resp = jsonify(message ="retry")
        return resp
        print("retry")
    check = ret
    if(check.shape[0] >= 150):
        with open("images/number.jpeg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        flag,ret = main(encoded_string)
        if(not flag):
            print("try again")
            resp = jsonify(message ="retry")
            return resp
    numbers = getEachNumber(ret)
    if(len(numbers)>0):
        returnString = convert(numbers)
    else:
        resp = jsonify(message ="retry")
        return resp 



    
    resp = jsonify(message =returnString) 
    return resp



if __name__ == '__main__':
    app.run(host='0.0.0.0')
    #app.run(debug=True)



# from keras.preprocessing.image import load_img
# import numpy as np
# import cv2
# from keras.models import load_model
# img=cv2.imread('images/numbers/15.jpg')
# model=load_model('ML/final_model_3.h5')
# img=cv2.resize(img,(256,256),interpolation=cv2.INTER_LINEAR)
# img=np.array([img])
# prediction=model.predict(img)
# max_value=np.amax(prediction[0])
# result=np.where(prediction[0]==max_value)
# print(result[0][0])
