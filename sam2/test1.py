import os   
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import cv2 as cv2
import tkinter as tk
from tkinter import Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import io
import gc
import tracemalloc
from tkinter import filedialog




device = torch.device("cuda")
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
output_dir = "C:\\Users\\Omen\\Desktop\\verzija 2\\sam2\\sam2\\output"
#checkpoint = "C:\\Users\\user\\Desktop\\diplomski\\sam2\\checkpoints\\sam2.1_hiera_tiny.pt"
checkpoint = "C:\\Users\\Omen\\Desktop\\verzija 2\\sam2\\checkpoints\\sam2.1_hiera_small.pt"
#checkpoint = "C:\\Users\\user\\Desktop\\diplomski\\sam2\\checkpoints\\sam2.1_hiera_large.pt"
#checkpoint = "C:\\Users\\user\\Desktop\\diplomski\\sam2\\checkpoints\\sam2.1_hiera_base_plus.pt"


#model_cfg = "C:\\Users\\user\\Desktop\\diplomski\\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_t.yaml"
model_cfg = "C:\\Users\\Omen\\Desktop\\verzija 2\\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_s.yaml"
#model_cfg = "C:\\Users\\user\\Desktop\\diplomski\\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_l.yaml"
#model_cfg = "C:\\Users\\user\\Desktop\\diplomski\\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_b+.yaml"


predictor = build_sam2_video_predictor(model_cfg,checkpoint).to(device)

def show_mask(mask, ax, obj_id = None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])],axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3],0.6]) 
    h, w =mask.shape[-2:]
    mask_image = mask.reshape(h,w,1) * color.reshape(1,1,-1)
    ax.imshow(mask_image)

def show_masks(mask, ax, obj_id = None, random_color=False):

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])],axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3],0.6])
    h, w =mask.shape[-2:]
    mask_image = mask.reshape(h,w,1) * color.reshape(1,1,-1)
    ax.imshow(mask_image)    

#editor
def img_full_color(mask, ax, obj_id = None):
    cmap = plt.get_cmap("tab10")
    cmap_idx = 0 if obj_id is None else obj_id
    color = np.array([*cmap(cmap_idx)[:3],1.0])
    h, w =mask.shape[-2:]
    mask_image = mask.reshape(h,w,1) * color.reshape(1,1,-1)
    ax.imshow(mask_image)

def img_border(mask, ax):
    
    mask = mask.squeeze()
    h, w =mask.shape[-2:]
    mask_image = np.zeros((h, w, 4)) 
    maskInt = np.zeros((h, w)) 
    maskInt[mask]=1
    kernel = np.ones((10, 10), np.uint8)
    dilated_mask = cv2.dilate(maskInt, kernel, iterations=1)
    positiv = (dilated_mask > 0) & (maskInt<=0)
    maskBool = positiv.astype(bool)
    

    color = (0, 0, 1.0)
    mask_image[maskBool,:3] = color 
    mask_image[maskBool,3] = 1.0 
    ax.imshow(mask_image)




def img_gradient(mask, ax,obj_id = None, color_start=(0, 0, 1), color_end=(1, 0, 0)):
    
    h, w =mask.shape[-2:]
    gradient_r = np.linspace(color_start[0], color_end[0], w)
    gradient_g = np.linspace(color_start[1], color_end[1], w)
    gradient_b = np.linspace(color_start[2], color_end[2], w)
    
    
    gradient = np.stack([np.tile(gradient_r, (h, 1)),
                         np.tile(gradient_g, (h, 1)),
                         np.tile(gradient_b, (h, 1)),
                         np.full((h, w), 0.6)], axis=-1)  

    
    mask_image = mask.reshape(h, w, 1) * gradient
    
    
    ax.imshow(mask_image)

allMasks=[] #za zrake
def img_ray(mask,ax):

    global allMasks

    h, w =mask.shape[-2:]
    new_image= np.zeros((h,w,4),dtype=np.uint8)
    positiv = np.argwhere(mask > 0)
    centr = np.mean(positiv, axis=0)  # Prosecna vrednost j i x
    if centr[1] == centr[1] and centr[0] == centr[0]:
        cx, cy = int(centr[1]), int(centr[0])
        length = 500  
        thickness = 5  
        num = 20  

        # Crtamo zrak i centralne tacke
        for angle in np.linspace(0, 2 * np.pi, num):  
            for k in range(length):
                rayx = int(cx + k * np.cos(angle))
                rayy = int(cy + k * np.sin(angle))

                if 0 <= rayx < w and 0 <= rayy < h  and all(mask[rayy, rayx] == 0 for mask in allMasks):   
                    
                    half_thickness = thickness // 2
                    cv2.rectangle(new_image, (rayx - half_thickness, rayy - half_thickness),
                                (rayx + half_thickness, rayy + half_thickness),
                                (255, 255, 0, 255), -1)  # -1 popunjava moj pravougaonik

            

        ax.imshow(new_image)


def crop_mask_from_image(mask,ax,image):
    
    
    image_for_crop = np.array(image)
    mask = mask[0]
    height = image_for_crop.shape[0]
    width = image_for_crop.shape[1]
    mask_image = np.zeros((height, width, 4), dtype=np.uint8)
    
    mask_for_crop = np.where(mask)   #kreiram tupl gde prvi element tapla predstavlja indekse redova a drugi kolona gde je true
 
    
    mask_image[mask_for_crop[0], mask_for_crop[1], :3] = image_for_crop[mask_for_crop[0], mask_for_crop[1], :3]
    mask_image[mask_for_crop[0], mask_for_crop[1], 3] = 255 


    
    if imgChoice == "Blur":
        mask_image = cv2.GaussianBlur(mask_image, (51, 51), 0)
        ax.imshow(mask_image)
        #
        image.close()
    elif imgChoice == "Ray":
        img_ray(mask,ax)  
        ax.imshow(mask_image)
        image.close()
    elif imgChoice == "Border":
        ax.imshow(mask_image)
        img_border(mask,ax)
        image.close()      
    elif imgChoice == "Gradient":
        ax.imshow(mask_image)
        img_gradient(mask,ax)
        image.close()
    elif imgChoice == "Full Color":
        ax.imshow(mask_image)
        img_full_color(mask,ax)
        image.close()
    else:
        ax.imshow(mask_image)
        image.close()

def show_points(coords,labels,ax,marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords [labels==0]
    ax.scatter(pos_points[:,0],pos_points[:,1],color='green',marker='*',s=marker_size,edgecolor='white',linewidth =1.25)
    ax.scatter(neg_points[:,0],neg_points[:,1],color='red',marker='*',s=marker_size,edgecolor='white',linewidth =1.25)

def change_BG_color(image,case= None):
    image_for_process= np.array(image)
    n_r=0
    n_g=0
    n_b=0
    h = image_for_process.shape[0]
    w = image_for_process.shape[1]
    new_image= np.zeros((h,w,3),dtype=np.uint8)

    if case=="Gray":

        gray = 0.299 * image_for_process[:, :, 0] + 0.587 * image_for_process[:, :, 1] + 0.144 * image_for_process[:, :, 2]
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        new_image[:, :, 0] = gray  
        new_image[:, :, 1] = gray  
        new_image[:, :, 2] = gray 

        # for i in range(h):
        #     for j in range(w):
        #         r,g,b=image_for_process[i][j]
        #         gray=int(0.299*r+0.587*g+0.144*b)
        #         if(gray>255):
        #             gray=255
        #         n_r=gray
        #         n_g=gray
        #         n_b=gray
        #         new_image[i,j] = [n_r,n_g,n_b]
    elif case=="White":
        n_r=255
        n_g=255
        n_b=255
        new_image[:,:]=[n_r,n_g,n_b]
    elif case=="Green":
        n_r=0
        n_g=255
        n_b=0
        new_image[:,:]=[n_r,n_g,n_b]

    return new_image

def change_BG_img(oldImg, path):

    
    new_image = Image.open(path)
    image_for_process= np.array(oldImg)
    h = image_for_process.shape[0]
    w = image_for_process.shape[1]
    new_image = new_image.resize((w,h))
    


    return new_image

def change_BG_gradient(Img, color_start=(0, 0, 1), color_end=(1, 0, 0)):
    
    image_for_process= np.array(Img)
    h = image_for_process.shape[0]
    w = image_for_process.shape[1]

    new_img= np.zeros((h,w,3),dtype=np.uint8)

    gradient_r = np.linspace(color_start[0], color_end[0], w)
    gradient_g = np.linspace(color_start[1], color_end[1], w)
    gradient_b = np.linspace(color_start[2], color_end[2], w)
    
    
    gradient = np.stack([np.tile(gradient_r, (h, 1)),
                         np.tile(gradient_g, (h, 1)),
                         np.tile(gradient_b, (h, 1))],
                         axis=-1)  

    
    new_img[:,:,:] = image_for_process[:, :, :3] * gradient

    
    
    return new_img

def change_BG_multi_color(image):
    image_for_process= np.array(image)
    h = image_for_process.shape[0]
    w = image_for_process.shape[1]

    full_res=h*w
    new_image= np.zeros((h,w,3),dtype=np.uint8)
	#novi kod
	
    r = image_for_process[:, :, 0]  
    g = image_for_process[:, :, 1]  
    b = image_for_process[:, :, 2]  
    gray = (0.299 * r + 0.587 * g + 0.144 * b).astype(np.uint8)

    
    i_indices, j_indices = np.indices((h, w))
    ij_product = i_indices * j_indices

    
    mask1 = (ij_product < full_res / 8)
    new_image[mask1, 0] = r[mask1]  
    new_image[mask1, 1] = g[mask1]  
    new_image[mask1, 2] = 0          

    
    mask2 = (full_res / 8 <= ij_product) & (ij_product < 2 * full_res / 8)
    new_image[mask2, 0] = 0          
    new_image[mask2, 1] = g[mask2]   
    new_image[mask2, 2] = b[mask2]   

    
    mask3 = (2 * full_res / 8 <= ij_product) & (ij_product < 2 * full_res / 4)
    new_image[mask3, 0] = r[mask3]   
    new_image[mask3, 1] = 0          
    new_image[mask3, 2] = b[mask3]   

    
    mask4 = (ij_product >= 2 * full_res / 4)
    new_image[mask4, 0] = gray[mask4]  
    new_image[mask4, 1] = gray[mask4]  
    new_image[mask4, 2] = gray[mask4]  
    
	# stari kod 
    # for i in range(h):
    #     for j in range(w):
    #         r,g,b=image_for_process[i][j]
    #         gray=int(0.299*r+0.587*g+0.144*b)
    #         if(gray>255):
    #             gray=255
    #         if(i*j<full_res/8):
    #             n_r=r
    #             n_g=g
    #             n_b=0
    #         elif(full_res/8<=i*j and i*j<2*full_res/8):
    #             n_r=0
    #             n_g=g
    #             n_b=b
    #         elif(2*full_res/8 <= i*j and i*j<2*full_res/4):
    #             n_r=r
    #             n_g=0
    #             n_b=b
    #         else:
    #             n_r=gray
    #             n_g=gray
    #             n_b=gray

            

            #new_image[i,j] = [n_r,n_g,n_b]
    

    return new_image


new_img_path = "C:\\Users\\user\\Desktop\\forest.jpg"
video_dir = "C:\\Users\\Omen\\Desktop\\verzija 2\\sam2\\sam2\\video"

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]

frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path = video_dir)
predictor.reset_state(inference_state)

root = tk.Tk()
root.title("SAM2")
root.attributes('-fullscreen', True)

frame_idx = 0

#########################
frame_from_array = 0
frameX=0
frameY=0
canvas = None

out_obj_ids=None
out_mask_logits= None
def show_frame(frame_for_graf):
    
    global canvas 

    global allChosenObjects,chosenObj,r,out_obj_ids,out_mask_logits
    prompts={}
    if allChosenObjects != {}:

        for i in range(chosenObj+1):
            if allChosenObjects[chosenObj][1] !=[]:
                points = np.array(allChosenObjects[i][0],dtype=np.float32)
                labels = np.array(allChosenObjects[i][1], np.int32)
                prompts[i]= points,labels
                _,out_obj_ids,out_mask_logits = predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=frame_from_array,
                    obj_id=i,
                    points=points,
                    labels = labels,)



    plt.clf()
    if canvas is not None:
        canvas.get_tk_widget().destroy()
    
    plt.figure(figsize=(7,6))
    plt.title(f"Frame {frame_for_graf}")
    plt.imshow(Image.open(os.path.join(video_dir,frame_names[frame_for_graf])))


    if allChosenObjects !={}:
            if out_obj_ids is not None:                            
                for i,out_obj_id in enumerate(out_obj_ids):   
                        if allChosenObjects[len(allChosenObjects)-1][1] !=[]:          
                            show_points(*prompts[out_obj_id],plt.gca())
                            show_masks((out_mask_logits[i]>0.0).cpu().numpy(),plt.gca(),obj_id=out_obj_id)
                

    
    canvas = FigureCanvasTkAgg(plt.gcf(),master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0,sticky=tk.N)
    canvas.mpl_connect('button_press_event', on_click)

    
        
    
    

def exit_app():
    plt.close('all')
    root.quit()
    root.destroy()


def destroyElementsOnRoot():
    for e in root.winfo_children():
        if e!=exit_button:
            e.destroy()

    root.attributes('-fullscreen', False)
    

def on_scale_change(value):
    global frame_from_array
    frame_from_array = int(value)  
    show_frame(frame_from_array)

r=0
newPoints = []
newLabels=[]
addObjPointsFlag = True
labelsList = []
def on_click(event):

    global r,addObjPointsFlag
    global newPoints,newLabels,allChosenObjects,chosenObj,frame_from_array

    if addObjPointsFlag:
        if event.button == 1:
            if event.xdata is not None and event.ydata is not None:
                frameX=int(event.xdata)
                frameY= int(event.ydata)
                lname = f"labelPoint{r}"
                globals()[lname] = tk.Label(one_chosen_object, text=f"Positive X: {frameX}, Y: {frameY}",fg="green",bg="white")
                globals()[lname].grid(row=r, column=1,sticky=tk.W)
                r+=1
                newPoints.append([frameX,frameY])
                newLabels.append(1)
                labelsList.append(globals()[lname])
                allChosenObjects[chosenObj]= (newPoints,newLabels)
                show_frame(frame_from_array)
        if event.button == 3:
            if event.xdata is not None and event.ydata is not None:
                frameX=int(event.xdata)
                frameY= int(event.ydata)
                lname = f"labelPoint{r}"
                globals()[lname] = tk.Label(one_chosen_object, text=f"Negative X: {frameX}, Y: {frameY}",fg="red",bg="white")
                globals()[lname].grid(row=r, column=1,sticky=tk.W)
                r+=1
                newPoints.append([frameX,frameY])
                newLabels.append(0)
                labelsList.append(globals()[lname])
                allChosenObjects[chosenObj]= (newPoints,newLabels)
                show_frame(frame_from_array)

chosenObj=0
allChosenObjects={}



def accept_object(btn):

    global newPoints,newLabels,allChosenObjects,chosenObj,addObjPointsFlag

    add_obj.config(state="normal")
    addObjPointsFlag=False
    undo_btn.config(state="disabled")

    newPoints = []
    newLabels=[]
    btn.destroy()


def add_next_object():
    global chosenObj,one_chosen_object,addObjPointsFlag
    addObjPointsFlag=True
    chosenObj+=1
    one_chosen_object = Frame(chosen_object_frame,bg="white")
    one_chosen_object.grid(row=chosenObj, column=0,sticky=tk.W)
    labelObj = tk.Label(one_chosen_object, text=f"Object {chosenObj}",bg="white")
    labelObj.grid(row=chosenObj, column=0,sticky=tk.W)
    
    acept_btn = tk.Button(one_chosen_object, text="Accept object", command=lambda: accept_object(acept_btn),fg="green") 

    acept_btn.grid(row=chosenObj+1, column=0,sticky=tk.W)

    add_obj.config(state="disabled")
    undo_btn.config(state="normal")

def runn_app():
    global allChosenObjects,frame_from_array
    plt.close("all")
    

    prompts={}
    for i in range(chosenObj+1):
        points = np.array(allChosenObjects[i][0],dtype=np.float32)
        labels = np.array(allChosenObjects[i][1], np.int32)
        prompts[i]= points,labels
        _,out_obj_ids,out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=frame_from_array,
            obj_id=i,
            points=points,
            labels = labels,)


     
    plt.figure(figsize=(12,8))
    plt.title(f"frame for check {frame_from_array}")
    plt.imshow(Image.open(os.path.join(video_dir,frame_names[frame_from_array])))
    for i,out_obj_id in enumerate(out_obj_ids):
        show_points(*prompts[out_obj_id],plt.gca())
        show_masks((out_mask_logits[i]>0.0).cpu().numpy(),plt.gca(),obj_id=out_obj_id)

    manager = plt.get_current_fig_manager()
    manager.window.geometry("+0+0")  
    manager.window.state('zoomed')
    plt.show()

    destroyElementsOnRoot()

    root.geometry("300x200+500+300")
    labelProp = tk.Label(root, text="Propagation")
    labelProp.grid(row=0, column=5,padx=110, pady=50,sticky=tk.N)
    exit_button.grid(row=6, column=5,sticky=tk.S)

    root.after(1000,propagation)
    
    

    
        


    
bgChoice = "Original"
imgChoice = "Mask"        


def propagation():


    

    video_segments = {}

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i]> 0.0).cpu().numpy()
            for i,out_obj_id in enumerate(out_obj_ids)
        }


    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i]> 0.0).cpu().numpy()
            for i,out_obj_id in enumerate(out_obj_ids)
        }


    destroyElementsOnRoot()

    root.geometry("500x270+470+250")
    labelProp = tk.Label(root, text="Propagation is finished!")
    labelProp.grid(row=0, column=5,padx=185, pady=10,sticky=tk.N)
    labelProp2 = tk.Label(root, text="Choose your efects!")
    labelProp2.grid(row=1, column=5,padx=150, pady=10,sticky=tk.N)

    dropdownFrame = Frame(root)

    
    dropdownFrame.grid(row=2, column=5,padx=100, pady=10,sticky=tk.N)

    labelProp3 = tk.Label(dropdownFrame, text="Choose Background")
    labelProp3.grid(row=0, column=0,padx=10, pady=10,sticky=tk.N)
    bgEdit = tk.StringVar()
    bgEdit.set("Original")
    bgOpt=["Original","Another Image","Gray","White","Green","Gradient","Multi Color"]
    bgDrop = tk.OptionMenu(dropdownFrame,bgEdit,*bgOpt,command=slecetBgEdit)
    bgDrop.grid(row=1, column=0,padx=10, pady=10,sticky=tk.N)

    labelProp4 = tk.Label(dropdownFrame, text="Choose Mask")
    labelProp4.grid(row=0, column=1,padx=10, pady=10,sticky=tk.N)
    imgEdit = tk.StringVar()
    imgEdit.set("Mask")
    imgOpt=["Mask","Original","Border","Gradient","Full Color","Ray","Blur"]
    imgDrop = tk.OptionMenu(dropdownFrame,imgEdit,*imgOpt,command=slecetImgEdit)
    imgDrop.grid(row=1, column=1,padx=10, pady=10,sticky=tk.N)

    run_edit = tk.Button(root, text="Run video editor", command=lambda: edit_Video(video_segments),fg="green") 
    run_edit.grid(row=3, column=5,sticky=tk.S)


    exit_button.grid(row=4, column=5,pady=10,sticky=tk.S)

def slecetBgEdit(choice):
    global bgChoice
    bgChoice=choice
    

def slecetImgEdit(choice):
    global imgChoice
    imgChoice=choice
        
    
 
    
def edit_Video(video_segments):

    
    global allMasks,new_img_path
    vis_frame_stride = 1
    plt.close("all")

    if bgChoice == "Another Image":
        new_img_path = filedialog.askopenfilename(title="Choose image",filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp")])
    #fig = plt.figure(figsize=(12,8))

    tracemalloc.start()


    for out_frame_idx in range(0,len(frame_names),vis_frame_stride):
        fig = plt.figure(figsize=(12,8))
        allMasks=[]
        plt.title(f"Frame {out_frame_idx}")
        plt.axis("off")
        ###
        imPath = os.path.join(video_dir,frame_names[out_frame_idx])
        with Image.open(imPath) as im:
            if bgChoice=="Gradient":
                
                plt.imshow(change_BG_gradient(im),animated = True)
            elif bgChoice =="Gray" or bgChoice=="White" or bgChoice=="Green":
                
                plt.imshow(change_BG_color(im,bgChoice),animated = True)
            elif bgChoice == "Another Image":
                
                plt.imshow(change_BG_img(im,new_img_path),animated = True)
            elif bgChoice == "Multi Color":
                
                plt.imshow(change_BG_multi_color(im),animated = True)
            else:
                
                plt.imshow(im,animated = True) 
        
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            if imgChoice == "Mask":
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            elif imgChoice == "Ray":
                allMasks.append(out_mask[0])
                with Image.open(os.path.join(video_dir,frame_names[out_frame_idx])) as objectIm:
                    crop_mask_from_image(out_mask, plt.gca(),objectIm )
            else:
                with Image.open(os.path.join(video_dir,frame_names[out_frame_idx])) as objectIm:
                    crop_mask_from_image(out_mask, plt.gca(), objectIm)



        plt.savefig(os.path.join(output_dir, f's{out_frame_idx}.png'),bbox_inches='tight', pad_inches=0) # bb za uklanjanje svih praznih delova oko slike a pad inches za uklanjanje margine izmedju slike i ivice
        plt.close("all")
        plt.close(fig)
        
        
        plt.clf()    
        plt.pause(.01)       
        gc.collect()

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        print("[Top 10]")
        for stat in top_stats[:10]:
            print(stat)

frame = Frame(root)


frame.grid(row=0, column=0,sticky=tk.N)

control_frame = Frame(root)

control_frame.grid(row=1, column=0,sticky=tk.S)



exit_button = tk.Button(root, text="Exit", command=exit_app,fg="red",width=7)

exit_button.grid(row=6, column=2,sticky=tk.S)

run_b = tk.Button(root, text="Run", command=runn_app,fg="green",width=7) 

run_b.grid(row=6, column=3,sticky=tk.S)


labelScale = tk.Label(control_frame, text="Select frame:",font=(None,13))
labelScale.grid(row=0, column=0, pady=20,sticky="es")


scale = tk.Scale(control_frame, from_=0, to=len(frame_names)-1, orient=tk.HORIZONTAL, command=on_scale_change,length=250)

scale.grid(row=0, column=1,padx=20, pady=20,sticky=tk.E)

add_obj = tk.Button(control_frame, text="Add object", command=add_next_object, height=2,fg="blue") 

add_obj.grid(row=0, column=2,padx=20, pady=20,sticky=tk.E)
add_obj.config(state="disabled")



def undo():
    global newPoints,newLabels,allChosenObjects,chosenObj,frame_from_array,labelsList,r
    newPoints.pop()
    newLabels.pop()
    allChosenObjects.pop(chosenObj)
    allChosenObjects[chosenObj]= (newPoints,newLabels)
    labelForDestroy = labelsList[-1]
    labelForDestroy.destroy()
    labelsList.pop()
    r-=1
    show_frame(frame_from_array)

undo_btn = tk.Button(control_frame, text="Undo", command=undo,height=2,fg="blue") 
undo_btn.grid(row=0, column=3, pady=20,sticky=tk.E)





######################
chosen_object_frame = Frame(root,bg="white")
chosen_object_frame.grid(row=0, column=4,sticky='nsew')
root.grid_columnconfigure(4, weight=1)



one_chosen_object = Frame(chosen_object_frame,bg="white")
one_chosen_object.grid(row=chosenObj, column=0,sticky='ew')
labelObj = tk.Label(one_chosen_object, text=f"Object {chosenObj}",bg="white")
labelObj.grid(row=chosenObj, column=0,sticky=tk.W)

acept_btn = tk.Button(one_chosen_object, text="Accept object", command= lambda: accept_object(acept_btn),fg="green") 

acept_btn.grid(row=chosenObj+1, column=0,sticky=tk.W)


show_frame(frame_from_array)
#tk.messagebox.showinfo("Tips", "Select the positive and negative points by clicking on the image.") porukica
#propagation()


root.mainloop()
