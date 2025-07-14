import numpy as np
import pandas as pd
import cv2
from scipy import ndimage , stats
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
import customtkinter as ctk

app = ctk.CTk()

app.geometry("640x480") 
app._set_appearance_mode("Dark") 

fram=ctk.CTkFrame(app,fg_color="black")
fram.pack(fill="both", expand=True)
 
def clear_frame():
    for widget in fram.winfo_children():
        widget.destroy()

def Page_zero ():
    clear_frame()
    label=ctk.CTkLabel(fram,text=" Image Processing ",font=("arial",15,"bold"))
    label.pack(padx=100,pady=10)
    
    bot1=ctk.CTkButton(fram,text=" Point Operation ",width=200,height=25,command=Point_Operation,corner_radius=30,hover_color="green")
    bot1.pack(padx=100,pady=10)

    bot2=ctk.CTkButton(fram,text=" Image Histogram ",width=200,height=25,command=Image_Histogram,corner_radius=30,hover_color="green")
    bot2.pack(padx=100,pady=10)

    bot3=ctk.CTkButton(fram,text=" Neighbrohood Processing ",width=200,height=25,command=Neighbrohood_Processing,corner_radius=30,hover_color="green")
    bot3.pack(padx=100,pady=10)

    bot4=ctk.CTkButton(fram,text=" Image Restoration ",width=200,height=25,command=Image_Restoration,corner_radius=30,hover_color="green")
    bot4.pack(padx=100,pady=10)

    bot5=ctk.CTkButton(fram,text=" Image Segmentation ",width=200,height=25,command=Image_Segmentation,corner_radius=30,hover_color="green")
    bot5.pack(padx=100,pady=10)

    bot6=ctk.CTkButton(fram,text=" Edge Detection ",width=200,height=25,command=Edge_Detection,corner_radius=30,hover_color="green")
    bot6.pack(padx=100,pady=10)

    bot7=ctk.CTkButton(fram,text=" Mathematical Morphology ",width=200,height=25,command=Mathematical_Morphology,corner_radius=30,hover_color="green")
    bot7.pack(padx=100,pady=10) 
    
def Point_Operation ():
   clear_frame()
   label=ctk.CTkLabel(fram,text=" Point Operation ",font=("arial",15,"bold"))
   label.pack(padx=100,pady=10)       
   bot1=ctk.CTkButton(fram,text=" Point Operation ",width=200,height=25,command=point_operation,corner_radius=30,hover_color="green")
   bot1.pack(padx=100,pady=10)
   bot2=ctk.CTkButton(fram,text=" Color Operation ",width=200,height=25,command=Color_Operation,corner_radius=30,hover_color="green")
   bot2.pack(padx=100,pady=10)
   bot_back=ctk.CTkButton(fram, text="Back", command=Page_zero, corner_radius=30)
   bot_back.pack(pady=20)
   
def point_operation () :
     img=cv2.imread(r"OIP.jpeg")
     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
     addition=img+20
     subtraction=img-20
     division=img/20
     complementary=255-img

     plt.subplot(2,2,1),plt.title("addition"),plt.imshow(addition)
     plt.subplot(2,2,2),plt.title("subtraction"),plt.imshow(subtraction)
     plt.subplot(2,2,3),plt.title("division"),plt.imshow(division)
     plt.subplot(2,2,4),plt.title("complement"),plt.imshow(complementary)
     plt.waitforbuttonpress()  
 
def Color_Operation () :
    img=cv2.imread(r"OIP.jpeg")
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    reddish = img.copy()
    swap = img.copy()
    eliminate = img.copy()

    reddish[:, :, 0] = reddish[:, :, 0] + 10
    swap[:, :, 0] = swap[:, :, 1]
    eliminate[:, :, 0] = 0

    plt.subplot(2, 2, 1),plt.title("Oraginal Image"),plt.imshow(img)
    plt.subplot(2, 2, 2),plt.title("Increase Red Channel"),plt.imshow(reddish)
    plt.subplot(2, 2, 3),plt.title("Swap R with G"),plt.imshow(swap)
    plt.subplot(2, 2, 4),plt.title("Remove Red Channel"),plt.imshow(eliminate)
    plt.show()

def Image_Histogram ():
   clear_frame()
   label=ctk.CTkLabel(fram,text=" Image Histogram ",font=("arial",15,"bold"))
   label.pack(padx=100,pady=10)   
   bot1=ctk.CTkButton(fram,text=" Histogram Stretching ",width=200,height=25,command=Histogram_Stretching,corner_radius=30,hover_color="green")
   bot1.pack(padx=100,pady=10)
   bot2=ctk.CTkButton(fram,text=" Histogram Equalization ",width=200,height=25,command=Histogram_Equalization,corner_radius=30,hover_color="green")
   bot2.pack(padx=100,pady=10)
   bot_back=ctk.CTkButton(fram, text="Back", command=Page_zero, corner_radius=30)
   bot_back.pack(pady=20)   

def Histogram_Stretching () :
    gray_image = cv2.imread(r"OIP.jpeg",cv2.IMREAD_GRAYSCALE)
    min_val = np.min(gray_image)
    max_val = np.max(gray_image)
    stretched_image = ((gray_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    plt.figure(figsize=(10, 6))
    plt.subplot(2,2,1),plt.imshow(gray_image,cmap="gray"),plt.title("Original Gray Image")
    plt.subplot(2,2,2),plt.hist(gray_image.ravel(),256,histtype="bar"),plt.title("Original Histogram")
    plt.subplot(2,2,3),plt.imshow(stretched_image,cmap="gray"),plt.title("stretched_image")
    plt.subplot(2,2,4),plt.hist(stretched_image.ravel(),256,histtype="bar"),plt.title("Histogram Stretching")

    plt.tight_layout()
    plt.show() 
 
def Histogram_Equalization () :
     image = cv2.imread(r"OIP.jpeg",cv2.COLOR_RGB2BGR)
     red , green , blue = cv2.split(image)
     plt.figure(figsize=(10, 6))
     plt.subplot(4,2,1),plt.hist(red.ravel(),256, color="red"),plt.title("Original Chanals")
     plt.subplot(4,2,3),plt.hist(green.ravel(),256,color="green")
     plt.subplot(4,2,5),plt.hist(blue.ravel(),256,color="blue")

     red_stretch = cv2.equalizeHist(red)
     green_stretch = cv2.equalizeHist(green)
     blue_stretch = cv2.equalizeHist(blue)

     plt.subplot(4,2,2),plt.hist(red_stretch.ravel(),256, color="red"),plt.title("Stretched Chanals")
     plt.subplot(4,2,4),plt.hist(green_stretch.ravel(),256,color="green")
     plt.subplot(4,2,6),plt.hist(blue_stretch.ravel(),256,color="blue")

     image_stretch = cv2.merge((red_stretch, green_stretch, blue_stretch))

     plt.subplot(4,2,7),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),plt.title('Original_Image')
     plt.subplot(4,2,8),plt.imshow(cv2.cvtColor(image_stretch, cv2.COLOR_BGR2RGB)),plt.title('Stretched_Image')
     plt.tight_layout()
     plt.show()

def Neighbrohood_Processing ():
    clear_frame()
    label=ctk.CTkLabel(fram,text=" Neighbrohood Processing ",font=("arial",15,"bold"))
    label.pack(padx=100,pady=10)
    bot1=ctk.CTkButton(fram,text=" Linear Filter ",width=200,height=25,command=Linear_Filter,corner_radius=30,hover_color="green")
    bot1.pack(padx=100,pady=10)
    bot2=ctk.CTkButton(fram,text=" Non Linear Filter ",width=200,height=25,command=Non_Linear_Filter,corner_radius=30,hover_color="green")
    bot2.pack(padx=100,pady=10)    
    bot_back=ctk.CTkButton(fram, text="Back", command=Page_zero, corner_radius=30)
    bot_back.pack(pady=20)

def Linear_Filter () :
     image = cv2.imread(r"OIP.jpeg")
     image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

     kernal = np.ones((3,3),np.float32)/9
     avarage_image = cv2.filter2D(image,-1,kernal)

     plt.subplot(2,2,1),plt.imshow(image),plt.title("Original Image")
     plt.subplot(2,2,3),plt.imshow(avarage_image),plt.title("Avarage Image")

     image_two = cv2.imread(r"OIP.jpeg")
     image_two = cv2.cvtColor(image_two,cv2.COLOR_BGR2GRAY)

     kernel_two=np.array([[1 ,-2, 1], [-2, 4,-2], [1 ,-2, 1]]) 
     laplacian=cv2.filter2D(image_two, -1,kernel_two)
     plt.subplot(2,2,2),plt.imshow(image_two,cmap="gray"),plt.title("Original image")
     plt.subplot(2,2,4),plt.imshow(laplacian,cmap="gray"),plt.title("Laplacian Filter")
     plt.show()  
 
def Non_Linear_Filter () :
    
     image = cv2.imread(r"OIP.jpeg")
     red, green, blue = cv2.split(image)

     red_max = ndimage.maximum_filter(red, size=9)
     green_max = ndimage.maximum_filter(green, size=9)
     blue_max = ndimage.maximum_filter(blue, size=9)
     result_max = cv2.merge((red_max, green_max, blue_max))

     red_min = ndimage.minimum_filter(red, size=9)
     green_min = ndimage.minimum_filter(green, size=9)
     blue_min = ndimage.minimum_filter(blue, size=9)
     result_min = cv2.merge((red_min, green_min, blue_min))

     red_median = ndimage.median_filter(red, size=9)
     green_median = ndimage.median_filter(green, size=9)
     blue_median = ndimage.median_filter(blue, size=9)
     result_median = cv2.merge((red_median, green_median, blue_median))

     def fast_mode_filter_vectorized(image_mode, k):
         pad = k // 2
         padded = np.pad(image_mode, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
         h, w, c = image_mode.shape
         result = np.zeros_like(image_mode, dtype=np.uint8)
         for ch in range(c):
             windows = view_as_windows(padded[:, :, ch], (k, k))
             windows = windows.reshape(h, w, -1) 
             mode_vals = stats.mode(windows, axis=2, keepdims=False).mode
             result[:, :, ch] = mode_vals.astype(np.uint8)
         return result
     result_mode = fast_mode_filter_vectorized(image, 3)
     
     plt.figure(figsize=(15, 8))
     plt.subplot(3,1,1),plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)),plt.title("Original Image")
     plt.subplot(3,2,3),plt.imshow(cv2.cvtColor(result_max,cv2.COLOR_BGR2RGB)),plt.title("Maximum Filter Image")
     plt.subplot(3,2,4),plt.imshow(cv2.cvtColor(result_min,cv2.COLOR_BGR2RGB)),plt.title("Minimum Filter Image")
     plt.subplot(3,2,5),plt.imshow(cv2.cvtColor(result_median,cv2.COLOR_BGR2RGB)),plt.title("Median Filter Image")
     plt.subplot(3,2,6),plt.imshow(cv2.cvtColor(result_mode,cv2.COLOR_BGR2RGB)),plt.title("Mode Filter Image")

     plt.tight_layout()
     plt.show()
    
def Image_Restoration ():
   clear_frame()
   label=ctk.CTkLabel(fram,text=" Image Restoration ",font=("arial",15,"bold"))
   label.pack(padx=100,pady=10)  
   bot1=ctk.CTkButton(fram,text=" Salt & Pepper Noise ",width=200,height=25,command=Salt_Pepper_Noise,corner_radius=30,hover_color="green")
   bot1.pack(padx=100,pady=10)
   bot2=ctk.CTkButton(fram,text=" Gaussian Noise ",width=200,height=25,command=Gaussian_Noise,corner_radius=30,hover_color="green")
   bot2.pack(padx=100,pady=10) 
   bot_back=ctk.CTkButton(fram, text="Back", command=Page_zero, corner_radius=30)
   bot_back.pack(pady=20)
   
def Salt_Pepper_Noise () :
     image=cv2.imread(r"OIP.jpeg")
     image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

     P=0.05
     result=np.zeros(image_rgb.shape,np.uint8)
     Th=1-P
     for i in range(image_rgb.shape[0]):
         for j in range(image_rgb.shape[1]):
             Rn=np.random.random()
             if Rn<P:
                 result[i][j]=0
             elif Rn > Th:
                result[i][j]=255
             else :
                 result[i][j]=image_rgb[i][j]    


     kernel=np.ones((3,3),np.float32)/9
     avg=cv2.filter2D(result,-1,kernel)

     MedianFilter=ndimage.median_filter(result,size=3)

     im_Outlier = np.zeros_like(image_rgb)
     kernel = np.array([[1/8,1/8, 1/8],[1/8, 0,1/8],[1/8, 1/8, 1/8]], dtype=np.float32)
     avg = cv2.filter2D(image_rgb, -1, kernel)
     Diff = abs(image_rgb-avg)
     im_Outlier = np.where(Diff > 0.4, avg, image_rgb).astype(np.uint8)


     plt.figure(figsize=(15, 8))
     plt.subplot(2,3,1),plt.imshow(image_rgb),plt.title("Original Image")
     plt.subplot(2,3,2),plt.imshow(result),plt.title("Nosie Image")
     plt.subplot(2,3,3),plt.imshow(avg),plt.title("Average")
     plt.subplot(2,3,4),plt.imshow(MedianFilter),plt.title("Median Filter")
     plt.subplot(2,3,5),plt.imshow(im_Outlier),plt.title("Outlier Removed")
     plt.tight_layout()
     plt.show()
 
def Gaussian_Noise () :
     image = cv2.imread(r"OIP.jpeg", cv2.IMREAD_GRAYSCALE)

     mean = 0
     std_dev = 40
     num_noisy_images = 10
     noisy_images = []

     for i in range(num_noisy_images):
         gauss = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
         noisy_img = cv2.add(image.astype(np.float32), gauss)
         noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
         noisy_images.append(noisy_img)

     avg_image = np.mean(noisy_images, axis=0)
     avg_image = np.clip(avg_image, 0, 255).astype(np.uint8)

     plt.figure(figsize=(12, 6))
     plt.subplot(3,4,1),plt.imshow(image, cmap='gray'),plt.title("Original")

     for i in range(num_noisy_images):
         plt.subplot(3,4,i+3),plt.imshow(noisy_images[i], cmap='gray'),plt.title(f"Noisy {i+1}")

     plt.subplot(3,4,2),plt.imshow(avg_image, cmap='gray'),plt.title("Averaged Image")

     plt.tight_layout()
     plt.show()

def Image_Segmentation ():
     image_color = cv2.imread(r"OIP.jpeg")
     image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
     Theta = np.mean(image)

     done = False
     while not done:
         p1 = [] 
         p2 = []  

         for i in range(image.shape[0]):
             for j in range(image.shape[1]):
                 if image[i, j] >= Theta:
                     p1.append(image[i, j])
                 else:
                     p2.append(image[i, j])

         m1 = np.mean(p1) if p1 else 0
         m2 = np.mean(p2) if p2 else 0
         Th_next = 0.5 * (m1 + m2)

         done = abs(Theta - Th_next) < 0.5
         Theta = Th_next    
    
     plt.hist(image.ravel(), 256, [0, 256])
     plt.show()

     image_new = (image > 130).astype('float')
    
     binary = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,199,5)    

     _, im_bw = cv2.threshold(image, Theta, 255, cv2.THRESH_BINARY)

     plt.figure(figsize=(12, 8))
     plt.subplot(2,2,1),plt.imshow(image, cmap='gray'),plt.title("Original")
     plt.subplot(2,2,2),plt.imshow(image_new, cmap='gray'),plt.title("Basic Global Thresholded")
     plt.subplot(2,2,3),plt.imshow(im_bw, cmap='gray'),plt.title("Automatic Thresholded")
     plt.subplot(2,2,4),plt.imshow(binary, cmap='gray'),plt.title("Adaptive Thresholded")
     plt.show()   

def Edge_Detection ():
    image=cv2.imread(r"OIP.jpeg")
    def apply_sobel_edge_detection(img):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        sobel_combined = cv2.magnitude(sobelx, sobely)
        sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        return sobel_combined
    sobel_edges = apply_sobel_edge_detection(image)
    plt.figure(figsize=(15, 12)) 
    plt.subplot(1,2,1),plt.imshow(image),plt.title("Original")
    plt.subplot(1,2,2),plt.imshow(sobel_edges),plt.title("Sobel Edges")
    plt.show()
    
def Mathematical_Morphology () :  
    image =cv2.imread (r"OIP.jpeg")
    def apply_erosion(img, kernel_size=3, iterations=1):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = cv2.erode(img, kernel, iterations=iterations)
        return eroded

    def apply_dilation(img, kernel_size=3, iterations=1):
             kernel = np.ones((kernel_size, kernel_size), np.uint8)
             dilated = cv2.dilate(img, kernel, iterations=iterations)
             return dilated

    def apply_opening(img, kernel_size=3):
             kernel = np.ones((kernel_size, kernel_size), np.uint8)
             opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
             return opened

    def apply_closing(img, kernel_size=3):
             kernel = np.ones((kernel_size, kernel_size), np.uint8)
             closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
             return closed

    def get_internal_boundary(img, kernel_size=3):
             eroded = apply_erosion(img, kernel_size)
             internal_boundary = img - eroded
             return internal_boundary

    def get_external_boundary(img, kernel_size=3):
             dilated = apply_dilation(img, kernel_size)
             external_boundary = dilated - img
             return external_boundary

    def get_morphological_gradient(img, kernel_size=3):
             dilated = apply_dilation(img, kernel_size)
             eroded = apply_erosion(img, kernel_size)
             gradient = dilated - eroded
             return gradient    
          
    eroded = apply_erosion(image)
    dilated = apply_dilation(image)
    opened = apply_opening(image)
    closed = apply_closing(image)
    internal_bound = get_internal_boundary(image)
    external_bound = get_external_boundary(image)
    morph_gradient = get_morphological_gradient(image)

    plt.figure(figsize=(15, 12))
    plt.subplot(3,3,1),plt.imshow(image),plt.title("Original")
    plt.subplot(3,3,2),plt.imshow(eroded),plt.title("Erosion")
    plt.subplot(3,3,3),plt.imshow(dilated),plt.title("Dilation")
    plt.subplot(3,3,4),plt.imshow(opened),plt.title("Opening")
    plt.subplot(3,3,5),plt.imshow(closed),plt.title("Closing")
    plt.subplot(3,3,6),plt.imshow(internal_bound),plt.title("Internal Boundary")
    plt.subplot(3,3,7),plt.imshow(external_bound),plt.title("External Boundary")
    plt.subplot(3,3,8),plt.imshow(morph_gradient),plt.title("Morphological Gradient")
    plt.show()          
   
Page_zero()   
app.mainloop()   