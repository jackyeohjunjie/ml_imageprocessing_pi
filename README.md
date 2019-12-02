# Real Time Machine learning + Image processing to diagnose crop health
This setup runs a on a hardware setup of a Raspberry Pi and PiCam. When python script is execited app starts up and snaps a picture, image is sent through a custom trained image classifer to first identify the type of crop in the image. After that, image is processed using the openCV library. Result is a diagnosed crop in regards to the crop size, diseased area and percentage value to correlation of golden sample of crop.

Key references:
https://medium.com/@bapireddy/real-time-image-classifier-on-raspberry-pi-using-inception-framework-faccfa150909
https://www.pyimagesearch.com/2017/09/04/raspbian-stretch-install-opencv-3-python-on-your-raspberry-pi/
