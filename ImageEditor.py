# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 19:13:02 2020

@author: DELL

DSP project: a fancy image editor :3

"""



import imutils
import cv2
import copy
import numpy as np
from scipy import interpolate
import speech_recognition as sr 


images = []


def printImage(image):
    
    images.append(image)
    
    cv2.destroyAllWindows()
    image = images[-1]
    image = imutils.resize(image, width=600)
    cv2.imshow("Current Image", image)
    cv2.waitKey(0)


def hear(idx):     
    
    if(idx==3):
        return 'exit'
    
    ear = sr.Recognizer()
    
    
    with sr.Microphone(1) as source:
        ear.adjust_for_ambient_noise(source, 1)
        print("listening...")
        audio = ear.listen(source, 2)
        try:
            print("processing...")
            text = ear.recognize_google(audio)
            text = text.lower()
            print(text)
            return text
        except:
            print("i didn't get that...")
            text = hear(idx+1)
    return text





def style_hear():
    
    #style=input("Pick your style: ")             ###>>>>>>>>>
    style = hear(0)                              ###<<<<<<<<<
    style = style.lower()
    model=''
    
    if(style=='back' or style=='exit'):
        return 'back'
    elif(style=='composition'):
        model='models/eccv16/composition_vii.t7'
    elif(style=='wave'):
        model='models/eccv16/the_wave.t7'
    elif(style=='candy'):
        model='models/instance_norm/candy.t7'
    elif(style=='feathers'):
        model='models/instance_norm/feathers.t7'
    elif(style=='la muse' or style=='lamuse' or style=='la news'):
        if(style!='la muse'):
            print('Close, but alright!!')
        model='models/instance_norm/la_muse.t7'
    elif(style=='mosaic'):
        model='models/instance_norm/mosaic.t7'
    elif(style=='starry night'):
        model='models/instance_norm/starry_night.t7'
    elif(style=='scream' or style=='cream'):
        if(style!='scream'):
            print('Close, but alright!!')
        model='models/instance_norm/the_scream.t7'
    elif(style=='celebration'):
        model='models/instance_norm/udnie.t7'
    else:
        print("Style not found...")
    
    return model


def style_transfer_input(image):
    
    #style = input("Pick your style: ")
    print(" ")
    print(" ")
    print("Pick your style...")
    print(" ")
    print("Composition  ----  Wave  ----  Candy  ----  Feathers")
    print("La Muse  ----  Mosaic  ----  Scream")
    print("Starry Night  ----  Celebration  ----  Back")
    print("--------------------------------------------------------")
    
    model = ''
    while(model==''):
        model = style_hear()
        if(model=='back'):
            return
    
    style_transfer_output(image, model)
    
    
def style_transfer_output(image, model):
    
    net = cv2.dnn.readNetFromTorch(model)
    
    
    if(image.ndim==2):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779, 123.680),
                                 swapRB=False, crop=False)
    
    net.setInput(blob)
    output = net.forward()
    
    output = output.reshape((3, output.shape[2], output.shape[3]))
    
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    
    #output /= np.max(output)
    output = cv2.normalize(output, output, alpha=0, beta=255,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    output = output.transpose(1, 2, 0)
    
    
    #if(output.ndim==3):
    #    if(output[0][0][0]>0 and output[0][0][0]<1):
    #        output = (output*255).astype(np.uint8)
    #    else:
    #        output = (output).astype(np.uint8)
    #elif(output.ndim==2):
    #    if(output[0][0]>0 and output[0][0]<1):
    #        output = (output*255).astype(np.uint8)
    #    else:
    #        output = (output).astype(np.uint8)
    

    printImage(output)
    
    
 
    


def BlackBoard(image):
    
    # Create sharpening kernel
    kernel = np.array([[1,-1,0], [-1,4,-1], [-1,0,-1]])

    # applying the sharpening kernel to the input image & displaying it.
    drawing = cv2.filter2D(image, -1, kernel)

    # Noise reduction
    output = cv2.bilateralFilter(drawing, 9, 75, 75)
    
    printImage(output)
    

def BlacknWhite(image):
    
    # convert to grayscale
    if(image.ndim==3):
        output = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif(image.ndim==2):
        output = image
    
    printImage(output)
    

def Cartoon(image):
    
    if(image.ndim==2):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif(image.ndim==3):
        img_rgb = image
    
    
    numDownSamples = 2        # number of downscaling steps
    numBilateralFilters = 50  # number of bilateral filtering steps

    # downsample image using Gaussian pyramid
    img_color = img_rgb
    for _ in range(numDownSamples):
        img_color = cv2.pyrDown(img_color)
    #cv2.imshow("downcolor",img_color)
    #cv2.waitKey(0)
    
    # repeatedly apply small bilateral filter instead of applying
    # one large filter
    for _ in range(numBilateralFilters):
        img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
    #cv2.imshow("bilateral filter",img_color)
    #cv2.waitKey(0)
    
    # upsample image to original size
    for _ in range(numDownSamples):
        img_color = cv2.pyrUp(img_color)
    #cv2.imshow("upscaling",img_color)
    #cv2.waitKey(0)

    # convert to grayscale and apply median blur
    
    #print(type(img_rgb))
    
    if(image.ndim==3):
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    elif(image.ndim==2):
        img_gray = image
    img_blur = cv2.medianBlur(img_gray, 3)
    img_color = cv2.medianBlur(img_color, 3)
    
    #print(type(img_blur))
    
    
    #cv2.imshow("grayscale+median blur",img_blur)
    #cv2.waitKey(0)

    # detect and enhance edges
    
    
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,9, 7)
    #cv2.imshow("edge",img_edge)
    #cv2.waitKey(0)

    # convert back to color so that it can be bit-ANDed with color image
    (x,y,z) = img_color.shape
    img_edge = cv2.resize(img_edge,(y,x))
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    #cv2.imshow("edge again", img_edge)
    #cv2.waitKey(0)
    
    
    output = cv2.bitwise_and(img_color, img_edge)
    
    #print(output.ndim)
    #print(np.max(output))
    
    printImage(output)
    

def Brightness_Contrast(image, alpha=1, beta=0):
    
    #alpha to control contrast 0.0 - 3.0
    #beta to control brightness, negative beta will decrease brightness, and vice versa
    
    output = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    printImage(output)
    

def Negative(image):
    
    if(image.ndim==3):
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif(image.ndim==2):
        img_gray = image
    
            
    L = np.max(img_gray) #max pixel value

    output = L - img_gray
    
    printImage(output)
    


def Sharp(image):
    
    # Create sharpening kernel, laplacian filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    
    # applying the sharpening kernel to the input image
    sharpened = cv2.filter2D(image, -1, kernel)
    
    # Noise reduction
    sharpened = cv2.bilateralFilter(sharpened, 9, 75, 75) 
    
    output = sharpened
    
    printImage(output)
    

def Sketch(image):
    
    img_rgb = image
    
    # convert to grayscale and apply median blur
    if(image.ndim==3):
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_rgb
        
    img_blur = cv2.medianBlur(img_gray, 3)
        
    # detect and enhance edges
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 3, 2)
        
    
    output = img_edge
    
    printImage(output)
    

def Sepia(image):
            
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    
    output = cv2.filter2D(image, -1, kernel)
                
    
    printImage(output)
    
    
def emboss(image):
    
    kernel = np.array([[0,-1,-1],
                       [1,0,-1],
                       [1,1,0]])
    
    output = cv2.filter2D(image, -1, kernel)
    
    printImage(output)
    

def spreadLookupTable(x, y):
    
    spline = interpolate.UnivariateSpline(x, y)
    return spline(range(256))


# Cold Image
def Cold(image):
    
    if(image.ndim==2):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    
    output = cv2.merge((red_channel, green_channel, blue_channel))
    
    printImage(output)
    

# Warm Image
def Warm(image):
            
    if(image.ndim==2):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    
    output = cv2.merge((red_channel, green_channel, blue_channel))
    
    printImage(output)
    



def commands(image):
    
    print(" ")
    print(" ")
    print("Say your command...")
    print(" ")
    print("Black board --- Black and White --- Cartoon --- Negative")
    print("Sketch --- Warm --- Cold --- Sharp --- Emboss")
    print("Sepia --- Style --- Brightness/Contrast")
    print("Undo --- Save --- Exit")
    print("--------------------------------------------------------")
    
    #Comm = input("Insert command: ")          ###>>>>>>>>> 
    Comm = hear(0)                            ###<<<<<<<<<
    Comm = Comm.lower()
    
    
    if(Comm=="exit"):
        cv2.destroyAllWindows()
        return 0
    
    elif(Comm=='style'):
        style_transfer_input(copy.deepcopy(images[-1]))
        return 1
    
    elif(Comm=='black board' or Comm=='blackboard'):
        BlackBoard(copy.deepcopy(images[-1]))
        return 1
    
    elif(Comm=='black and white'):
        BlacknWhite(copy.deepcopy(images[-1]))
        return 1
    
    elif(Comm=='cartoon'):
        Cartoon(copy.deepcopy(images[-1]))
        return 1
    
    elif(Comm=='negative'):
        Negative(images[-1])
        return 1
    
    elif(Comm=='sketch'):
        Sketch(copy.deepcopy(images[-1]))
        return 1
    
    elif(Comm=='sepia' or Comm=='seppia' or Comm=='sophia'):
        if(Comm!='sepia'):
            print('Close, but alright!!')
        Sepia(copy.deepcopy(images[-1]))
        return 1
    
    elif(Comm=='emboss'):
        emboss(copy.deepcopy(images[-1]))
        return 1
    
    elif(Comm=='warm' or Comm=='worm'):
        if(Comm!='warm'):
            print('Close, but alright!!')
        Warm(copy.deepcopy(images[-1]))
        return 1
    
    elif(Comm=='cold'):
        Cold(copy.deepcopy(images[-1]))
        return 1
    
    elif(Comm=='sharp'):
        Sharp(copy.deepcopy(images[-1]))
        return 1
    
    elif(Comm=='brightness' or Comm=='contrast' or Comm=='brightness contrast'
         or Comm=='brightness and contrast' or Comm=='Brightness/Contrast'):
        print("Please type...")
        alpha = float(input("Contrast level from 0.0 to 3.0 (default=1.0): "))
        beta =  int(input("Brightness level from -300 to 300  (default=0): "))
        Brightness_Contrast(copy.deepcopy(images[-1]), alpha, beta)
        return 1
    
    elif(Comm=='undo'):
        if(len(images)>1):
            images.pop()
            output = images.pop()
            printImage(output)
        else:
            print("This is your input image!!!")
        return 1
    
    elif(Comm=='save'):
        print("Please type...")
        filename = input("Enter image name(with extension): ")
        
        image = images[-1]
        
                
        cv2.imwrite(filename, image)
        cv2.destroyAllWindows()
        return 0
    
    else:
        print("Command not found...")
        #output = images.pop()
        #printImage(output)
        return 1
        

    


def editor():
    
    X=input("Insert image to be edited: ")
    #X="mo.png"
    
    image = cv2.imread(X)
        
    printImage(image)
    
    while(1):
        output=commands(image)
        if(output==0):
            break
    
    

editor()
    

    