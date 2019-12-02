# Real Time Machine learning + Image Processing to diagnose crop health
This setup runs a on a hardware setup of a Raspberry Pi and PiCam. When python script is executed, app starts up and snaps a picture, image is sent through a custom trained image classifer to first identify the type of crop in the image. After that, image is processed using the openCV library. Result is a diagnosed crop in regards to the crop size, diseased ar0e0a and percentage value to correlation of golden sample of crop.

Improvements and notes:
ML portion of this application is fairly straight forward, as it is trained as an image classifer. Images were stock downloaded with a total of 200 per type of crop, results were fairly accurate.

Improvements have to be made the image processing segment of the application, as application works under the assumption that image is captured at a fixed height and time. Lighting of crop when image will affect the results to a moderate degree, as threshold values are hardcoded in, comparison to golden crop also will be affected based on comparison values.
Key references and credits:

https://medium.com/@bapireddy/real-time-image-classifier-on-raspberry-pi-using-inception-framework-faccfa150909

https://www.pyimagesearch.com/2017/09/04/raspbian-stretch-install-opencv-3-python-on-your-raspberry-pi/
