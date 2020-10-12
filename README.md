# Neural Style Transfer Using Pytorch
### Intro
We us pytorch to learn from some famous pictures, extract their drawing 
styles. Then we apply this style to a photo.

### Set Up
pip install Pillow==2.2.2
pip install torchvision
pip install torch
pip install opencv-python

### Demo
python main.py

### Result
Here we show 2 samples that we apply style of picture B to A.
![image](result.png)

# Black and White painting
In folder HandPaintPhoto. We can run handpaint.py to get result as shown:<br/>
Example 1
![original image](HandPaintPhoto/landscape.jpg)
![Transferred Image](HandPaintPhoto/newimg.jpg)
Example<br/>
![original image](HandPaintPhoto/chongqing.jpeg)
![Transferred Image](HandPaintPhoto/newimg2.jpg)


# Change Background Color of a Photo
We also use opencv library in this work.
In BackgroundColorChange folder, run demo.py.
We change the photo with blue background to photo with red background.<br/>
![image](BackgroundColorChange/photo.jpg)<br/>
After change the result is like this. We also scale the photo a bit.
![image](BackgroundColorChange/result.jpg)<br/>

# Add a flag Sticker to Profile Photo
We have a profile photo. 
![profile](profilePhoto/photo.jpg)<br/>
And we prepare a sticker with small flag.
![flag](profilePhoto/flag.png)<br/>
Run program the program profilePhotoAddFlag.py.

```user
$ cd profilePhoto/
$ python profilePhotoAddFlag.py 
pygame 1.9.6
Hello from the pygame community. https://www.pygame.org/contribute.html
Enter the size of your profile photo 400
```
Note the size of profile photo is 400.
Here is the result.
![newresult](profilePhoto/new.png)<br/>

