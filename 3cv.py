import matplotlib.pyplot as plt
import numpy as np

from skimage import data, io, transform
from scipy import signal



x = np.linspace(0, 5*np.pi, 500)
plt.plot(x,"*-")



plt.figure()
y = np.sin(x)
plt.plot(x, y,"*-")
plt.ylabel("Y- Value of sin(x)")
plt.xlabel("x_value")
plt.title('Kartenzianska sustava hodnot vektora Y')



plt.figure()
rand_img = np.random.rand(256, 512)
plt.imshow(rand_img>0.5,cmap='gray')



image = data.imread("./imgs/lena.png")
fig, ax = plt.subplots()
io.imshow(image) # Preco toto funguje.
plt.title('Lena in RGB')



plt.figure()
plt.imshow(np.concatenate((image[:,:,0], image[:,:,1], image[:,:,2]),1),cmap='pink')
image_R = np.array(image)
image_R[:,:,1:] = 0
plt.figure()
plt.imshow(image_R)
plt.figure()
img = image_R[:,:,0].astype(dtype=np.uint8)
plt.imshow(image[:,:,0],cmap='gray')

#%%
image_earth = data.imread("imgs/earth.jpg")
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(image_earth)
ax.set_aspect('equal')



image_earth = data.imread("imgs/earth.jpg")
i_gray = 0.2126 * image_earth[:,:,0] + 0.7152 * image_earth[:,:,1] + 0.0722 * image_earth[:,:,2]
i_gray = i_gray.astype(dtype=np.uint8)
plt.figure()
plt.imshow(i_gray, cmap='gray')



from skimage.color import rgb2gray
#Y = 0.2125 R + 0.7154 G + 0.0721 B scikit
i_gray2 = (rgb2gray(image_earth)*255).astype(dtype=np.uint8)
print("Sum error between methods: ", np.sum((i_gray2 -i_gray)**2))
print("Average error: ", np.sum((i_gray2 -i_gray)**2)/i_gray2.size)

plt.figure()
plt.imshow(np.concatenate((i_gray, i_gray2),1),cmap='pink')
plt.figure()
plt.imshow(i_gray-i_gray2)



I_resized = transform.resize(image, np.array(i_gray.shape)*2,order=1)
I_resized2 = transform.resize(image, np.array(i_gray.shape)*2, order=5)
print("Sum error between methods: ", np.sum((I_resized -I_resized2)**2))
print("Average error: ", np.sum((I_resized -I_resized2)**2)/i_gray2.size)

plt.figure()
plt.imshow(I_resized)




I_rotated = transform.rotate(image,360,resize=True)
plt.figure()
plt.imshow(I_rotated)



plt.figure()
plt.imshow(image**2)



plt.figure()
g = np.array(np.array([[1, 2, 1]]) * np.array([[1], [2], [1]]) / 16)
plt.imshow(signal.convolve2d(rgb2gray(image),g))
conv_img = signal.convolve2d(rgb2gray(image),g,'same')
plt.imshow(rgb2gray(image)-conv_img)
plt.title('Convolved image')


image = data.imread("imgs/tsukubaleft.jpg")
fig = plt.figure()

plt.imshow(image)
print(image.shape)


Pparts=[image[0:183,0:275,:],image[0:183,275:,:],image[183:,0:275,:],image[183:,275:,:]]
up=np.concatenate((Pparts[0],Pparts[3]),axis=1)
down=np.concatenate((Pparts[1],Pparts[2]),axis=1)
new=np.concatenate((np.concatenate((Pparts[0],Pparts[3]),axis=1),np.concatenate((Pparts[1],Pparts[2]),axis=1)),axis=0)
plt.imshow(new)
