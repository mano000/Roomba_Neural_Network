# We are going to create a dataset based on the bumper of the roomba
# so we can train it later with the images and take decissions based on a CNN

from picamera.array import PiRGBArray
from PIL import Image
import os, sys
import pygame
import create
import time
import datetime
import picamera
import random
import numpy as np
import io

ROOMBA_PORT = "/dev/ttyUSB0"

imagearray = []
movearray = []

np.nav_array =[]


robot = create.Create(ROOMBA_PORT, BAUD_RATE=115200)
robot.toSafeMode()
#robot.printSensors()


MAX_FORWARD = 7 # in cm per second
MAX_ROTATION = 30 # in cm per second
SPEED_INC = 1 # increment in percent
# start 50% speed


def take_picture():
             # Create the in-memory stream
                stream = io.BytesIO()
                with picamera.PiCamera() as camera:
                          camera.start_preview()
                          camera.resolution=(800,600)
                          time.sleep(0.2)
                          camera.capture(stream, format='jpeg')
               # "Rewind" the stream to the beginning so we can read its content
                stream.seek(0)
                image = Image.open(stream)
                #imagearray.append(image)
                return image


def update_roomba():
 
    print ("update roomba")
    sensores = robot.sensors([create.WALL_SIGNAL, create.WALL_IR_SENSOR, create.LEFT_BUMP, create.RIGHT_BUMP, create.ENCODER_LEFT, create.ENCODER_RIGHT, create.CLIFF_LEFT_SIGNAL, create.CLIFF_FRONT_LEFT_SIGNAL, create.CLIFF_FRONT_RIGHT_SIGNAL, create.CLIFF_RIGHT_SIGNAL, create.DIRT_DETECTED])
    return sensores


def evaluate_nav_data():

                                #Create enough data to validate the actual path
				#if it has been going straight without hitting anything for a while...
				#if (len(movearray)>>10) and (all(i == 0 for i in movearray[-10:-5])):
				#Save the data and reinitialize the array	
			   	#    for x in imagearray[-10:-5]:

         			#    	filename = '/root/roomba/SI/'+str(datetime.datetime.now())+'_s.jpg' #%x
           		#		print('Writing %s' % filename)
			#		x.save(filename,"JPEG")
		  			
				    #del imagearray
				    #imagearray = []
				    #del movearray
				    #movearray = [] 
                                if (len(np.nav_array)>=6):
                                    new_eval=0
                                    for x in reversed(np.nav_array[:-4]):  #let's mark the 3 las pictures as bad
                                       #if (any(i != 0 for i in movearray[-9:-5])):
                                       #new_eval=0
                                       if (x[0]!=0):  
                                            new_eval=x[0]
                                       x[0]=new_eval



                                #Save the data and reinitialize the array
                                 #   for x in imagearray[-5:]:

                                      #  filename = '/root/roomba/NO/'+str(datetime.datetime.now())+'_n.jpg' #%x
                                print('Do some eval'+str(len(np.nav_array)))
                                #print np.nav_array
                                      #  x.save(filename,"JPEG")
                                         #else:
                                write_evaluated_images()
                                #    for x in imagearray[-5:]:

                                      #  filename = '/root/roomba/SI/'+str(datetime.datetime.now())+'_s.jpg' #%x
                                        #print('Writing %s' % filename)
                                      #  x.save(filename,"JPEG")

#try:
#        b=a.index(7)
#except ValueError:
#        "Do nothing"
#    else:
#            "Do something with variable b"


def write_evaluated_images():
    if (len(np.nav_array)>=12):
        print ("entramos a escribir algo")
        for x in np.nav_array[:-10]:

            if (x[0]==0):
                    filename = '/root/roomba/YES/'+str(datetime.datetime.now())+'_yes.jpg' #%x
                    print('Writing %s' % filename)
                    x[2].save(filename,"JPEG")
            elif (x[0]==2):
                    filename = '/root/roomba/FRONT_CRASH/'+str(datetime.datetime.now())+'_front_crash.jpg' #%x
                    print('Writing %s' % filename)
                    x[2].save(filename,"JPEG")
            else:
                sensores=x[1]
                if (sensores[create.LEFT_BUMP]==1):
                    filename = '/root/roomba/LEFT_BUMP/'+str(datetime.datetime.now())+'_left_bump.jpg' #%x
                    print('Writing %s' % filename)
                    x[2].save(filename,"JPEG")
                else:
                    filename = '/root/roomba/RIGHT_BUMP/'+str(datetime.datetime.now())+'_right_bump.jpg' #%x
                    print('Writing %s' % filename)
                    x[2].save(filename,"JPEG")



            
            np.nav_array.pop(x[0])




def main():
	FWD_SPEED = MAX_FORWARD#/2
	ROT_SPEED = MAX_ROTATION#/2

	robot_dir = 0
	robot_rot = 0

	robot.resetPose()
	px, py, th = robot.getPose()

	while True:
		
	#Start the random navigation
		robot_dir+=0.5
		#update_roomba()  
		robot_rot=0
         #end random navigation
                #time.sleep(1)
        #added to stop motion
                #robot_dir=0
		#robot_rot=0


		senses=update_roomba()  
                #movearray.append((senses[create.LEFT_BUMP])+(senses[create.RIGHT_BUMP]))
                picture=take_picture() 
                np.nav_array.append([(senses[create.LEFT_BUMP])+(senses[create.RIGHT_BUMP]),senses,picture])
                picture.save("/var/www/html/last.jpg")

                
       	        if (senses[create.LEFT_BUMP] ==0) and (senses[create.RIGHT_BUMP]==0):
                           	#camera.capture('/root/roomba/SI/'+str(datetime.datetime.now())+'_si.jpg')
			        
                                #imagearray.append(take_picture())
                           	#time.sleep(0.1)
			   	robot.go(robot_dir*FWD_SPEED,robot_rot*ROT_SPEED)		
			   	#time.sleep(0.1)

                                #evaluate_nav_data()
		           		
                #added to stop motion
                robot_dir=0
		robot_rot=0

                #movearray.append((senses[create.LEFT_BUMP])+(senses[create.RIGHT_BUMP]))

                for i in np.nav_array:
                    print i[0]
               #movearray

		if (senses[create.LEFT_BUMP] ==1) or (senses[create.RIGHT_BUMP]==1):
		        print("BUMP")	
			#print movearray[-5:-1] #contains a digit bigger than 0	
			
			#time.sleep(0.1)
		        robot_dir-=0.5
			robot_rot=0
			robot.go(robot_dir*FWD_SPEED,robot_rot*ROT_SPEED)
			#time.sleep(0.1)
                        #evaluate_nav_data()

		        		




			#time.sleep(0.5)
			#if (PREVIOUS_MOVE==False):
			#	robot_rot+=6
                        #        robot_dir=0
                        #        robot.go(robot_dir*FWD_SPEED,robot_rot*ROT_SPEED)


                        if (senses[create.LEFT_BUMP] ==0) and (senses[create.RIGHT_BUMP]==1): 

		        	robot_rot+=0.2		
				robot_dir=0
				robot.go(robot_dir*FWD_SPEED,robot_rot*ROT_SPEED)

			elif (senses[create.LEFT_BUMP] ==1)and  (senses[create.RIGHT_BUMP]==0):
	
				robot_rot-=0.2
                                robot_dir=0
                                robot.go(robot_dir*FWD_SPEED,robot_rot*ROT_SPEED)


                        else: # (senses[create.LEFT_BUMP] ==1)and  (senses[create.RIGHT_BUMP]==1):
				robot_rot-=5
                                robot_dir=-0.5
                                robot.go(robot_dir*FWD_SPEED,robot_rot*ROT_SPEED)

		px, py, th = robot.getPose()
                evaluate_nav_data()
#		screen.blit(font.render("Estimated X-Position: {:04.2f} (cm from start)".format(px), 1, (10, 10, 10)), (450, 450))
#		screen.blit(font.render("Estimated Y-Position: {:04.2f} (cm from start)".format(py), 1, (10, 10, 10)), (450, 470))
#		screen.blit(font.render("  Estimated Rotation: {:03.2f} (in degree)".format(th), 1, (10, 10, 10)), (450, 490))


		
if __name__ == '__main__': 
	try:
		main()
	except Exception as err:
		print (err)
	#robot.go(0,0)
	robot.close()
