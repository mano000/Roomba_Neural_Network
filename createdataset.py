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


robot = create.Create(ROOMBA_PORT, BAUD_RATE=115200)
robot.toSafeMode()
#robot.printSensors()


MAX_FORWARD = 10 # in cm per second
MAX_ROTATION = 50 # in cm per second
SPEED_INC = 1 # increment in percent
# start 50% speed



def main():
	FWD_SPEED = MAX_FORWARD/2
	ROT_SPEED = MAX_ROTATION/2

	robot_dir = 0
	robot_rot = 0

	robot.resetPose()
	px, py, th = robot.getPose()

	while True:
		senses = robot.sensors([create.WALL_SIGNAL, create.WALL_IR_SENSOR, create.LEFT_BUMP, create.RIGHT_BUMP, create.ENCODER_LEFT, create.ENCODER_RIGHT, create.CLIFF_LEFT_SIGNAL, create.CLIFF_FRONT_LEFT_SIGNAL, create.CLIFF_FRONT_RIGHT_SIGNAL, create.CLIFF_RIGHT_SIGNAL, create.DIRT_DETECTED])
		update_roomba = False


	#Start the random navigation
		robot_dir+=0.5
		update_roomba = True 
		robot_rot=0
         #end random navigation

       

		if update_roomba == True:
			#robot.sensors([create.POSE])		
			time.sleep(0.5)
			senses = robot.sensors([create.WALL_SIGNAL, create.WALL_IR_SENSOR, create.LEFT_BUMP, create.RIGHT_BUMP, create.ENCODER_LEFT, create.ENCODER_RIGHT, create.CLIFF_LEFT_SIGNAL, create.CLIFF_FRONT_LEFT_SIGNAL, create.CLIFF_FRONT_RIGHT_SIGNAL, create.CLIFF_RIGHT_SIGNAL, create.DIRT_DETECTED])

		        if (senses[create.LEFT_BUMP] ==0) and (senses[create.RIGHT_BUMP]==0):
                           	#camera.capture('/root/roomba/SI/'+str(datetime.datetime.now())+'_si.jpg')
				# Create the in-memory stream
	         		stream = io.BytesIO()
				with picamera.PiCamera() as camera:
    					camera.start_preview()
					camera.resolution=(800,600)
    					time.sleep(2)
    					camera.capture(stream, format='jpeg')
				# "Rewind" the stream to the beginning so we can read its content
				stream.seek(0)
				image = Image.open(stream)
			        imagearray.append(image)
                           	time.sleep(0.1)
			   	robot.go(robot_dir*FWD_SPEED,robot_rot*ROT_SPEED)		
			   	time.sleep(0.1)
	
			   	for x in imagearray:

         			        filename = '/root/roomba/SI/'+str(datetime.datetime.now())+'_%si.jpg' %x
           				print('Writing %s' % filename)

                                        #img.save(filename, "JPEG")
			#		img.show()
					x.save(filename,"JPEG")


		           		
#added to stop motion
                robot_dir=0
		robot_rot=0

		# done with the actual roomba stuff
		# now print.
                movearray.append((senses[create.LEFT_BUMP])+(senses[create.RIGHT_BUMP]))
               # print (senses[create.LEFT_BUMP])+(senses[create.RIGHT_BUMP])
		print movearray
		if (senses[create.LEFT_BUMP] ==1) or (senses[create.RIGHT_BUMP]==1):
		        print("BUMP")	
		        #if 
			print movearray[-5:-1] #contains a digit bigger than 0	
			
			time.sleep(0.5)
		        robot_dir-=0.5
			robot_rot=0
			robot.go(robot_dir*FWD_SPEED,robot_rot*ROT_SPEED)
			time.sleep(1)
		        camera.capture('/root/roomba/NO/'+str(datetime.datetime.now())+'_no.jpg')
                       # pygame.image.load(img_no)
			time.sleep(0.5)
			if (PREVIOUS_MOVE==False):
				robot_rot+=6
                                robot_dir=0
                                robot.go(robot_dir*FWD_SPEED,robot_rot*ROT_SPEED)


                        elif (senses[create.LEFT_BUMP] ==0) and (senses[create.RIGHT_BUMP]==1): 

		        	robot_rot+=2		
				robot_dir=0
				robot.go(robot_dir*FWD_SPEED,robot_rot*ROT_SPEED)

			elif (senses[create.LEFT_BUMP] ==1)and  (senses[create.RIGHT_BUMP]==0):
	
				robot_rot-=2
                                robot_dir=0
                                robot.go(robot_dir*FWD_SPEED,robot_rot*ROT_SPEED)


			elif (senses[create.LEFT_BUMP] ==1)and  (senses[create.RIGHT_BUMP]==1):
				robot_rot-=3
                                robot_dir=-1
                                robot.go(robot_dir*FWD_SPEED,robot_rot*ROT_SPEED)

		px, py, th = robot.getPose()
#		screen.blit(font.render("Estimated X-Position: {:04.2f} (cm from start)".format(px), 1, (10, 10, 10)), (450, 450))
#		screen.blit(font.render("Estimated Y-Position: {:04.2f} (cm from start)".format(py), 1, (10, 10, 10)), (450, 470))
#		screen.blit(font.render("  Estimated Rotation: {:03.2f} (in degree)".format(th), 1, (10, 10, 10)), (450, 490))


		
if __name__ == '__main__': 
	try:
		main()
	except Exception as err:
		print (err)
	robot.go(0,0)
	robot.close()
