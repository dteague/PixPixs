def color_num(r, g, b):
    return r+256*g+256**2*b


def closest_color(colors, pixel):
    
    best = (255*3)**2
    number = 0
    i = 0
    for col in colors:
        dif = np.sum((pixel-col)**2)
       
            
        if dif < best:
            best = dif
            number = i


        elif dif == best:
            min1 = np.sum((pixel-col)**3)
            min2 = np.sum((pixel-colors[number])**3)

            if min1 < min2:
                best = dif
                number = i
            elif min1 == min2:
                print "happened"
                return -1
            
        i = i + 1
    return number


def next_color(x, y, im, colors):
    if x !=0 and y != 0:
        n1 = (im.getpixel((x-1,y))[0] + im.getpixel((x,y-1))[0])/2
        n2 = (im.getpixel((x-1,y))[1] + im.getpixel((x,y-1))[1])/2
        n3 = (im.getpixel((x-1,y))[2] + im.getpixel((x,y-1))[2])/2
        return closest_color(colors, (n1,n2,n3))
    elif x == 0 and y != 0:
        return closest_color(colors, im.getpixel((x,y-1)))
    elif x != 0 and y == 0:
        return closest_color(colors, im.getpixel((x-1,y)))
    else:
        return 1
    

def get_cells(im):
    
    pixels = np.zeros((im.size[1]/8, im.size[0]/8, 64,3))
    for i in range(im.size[1]/8):
        for j in range(im.size[0]/8):
            pixels[i][j] = [im.getpixel((x,y)) for y in range(i*8, (i+1)*8) for x in range(j*8,(j+1)*8)]

    return pixels


def get_grey(pixel):
    return 0.21*pixel[0] + 0.72*pixel[1] + 0.07*pixel[2]


def mean_fit(array, sections, close):

    m = np.zeros((sections, len(array[0])))
    space = len(array)/sections
    n = np.zeros(sections)

    for i,val in enumerate(close):
        m[val] += array[i]
        n[val] += 1

    for i in range(sections):
        if n[i] == 0:
            return mean_fit(array, sections-1, fix_closest(close, sections-1, i))

    m = [m[i]/float(n[i]) for i in range(sections)]
    #if not same_test(m):


    tmp = closest(array,close,m, sections)
    if (tmp != close).any():
        return mean_fit(array, sections, tmp)
    # else:
    #     print "yo single?"
    #     tmp = [int(val) for val in close]

    listing = []
    listing.append(tmp)
    listing.append(m)
        
    return listing


def same_test(array):

    test = True
    for i in range(1,len(array)):
        test = test and (array[0]==array[i]).all()

    return test


def first_closest(size, sections):

    array = np.empty(size)
    space = len(array)/sections
    for i in range(sections-1):
        array[space*i:space*(i+1)] = i
    array[space*(sections-1):] = sections-1

    return array


def fix_closest(close, sections,bad):
    mapping = {}
    for i in range(sections):
        if i < bad:
            mapping[i]=i
        else:
            mapping[i+1]=i
    new_closest = np.empty(len(close))
    for i, val in enumerate(close):
        new_closest[i] = mapping[val]

    return new_closest


def closest(array, close, m, sections):

    dif = np.zeros((len(array), sections))
    for (i,j) in [(i,j) for i in range(len(array)) for j in range(sections)]:
        dif[i][j] += np.sum((m[j]-array[i])**2)

    close= np.argmin(dif,axis=1)
    return close



def output_im(im, pixels,cell_x, cell_y, pix_x, pix_y):
    
    picture = im.load()
    x_cells = im.size[0]/cell_x
    y_cells = im.size[1]/cell_y
    
    for (i,j) in [(i,j) for j in range(y_cells) for i in range(x_cells)]:
        for (x,y) in [(x,y) for y in range(cell_y) for x in range(cell_x)]:
            
            pix=pixels[j][i][x+y*cell_y]

            picture[cell_x*i + x, cell_y*j + y]=(int(pix[0]),int(pix[1]),int(pix[2]))

    return im




com_colors = [(0,0,0),(255,255,255),(136,0,0),(170,255,238),(204,68,204),(0,204,85),(0,0,170),(238,238,119),(221,136,85),(102,68,0),(255,119,119),(51,51,51),(119,119,199),(170,255,102),(0,136,255),(187,187,187)]

ap_colors = [(0,0,0),(156,156,156),(255,255,255),(96,78,189),(208,195,255),(255,68,253),(227,30,96),(255,160,200),(255,106,60),(96,114,3),(208,221,141),(20,245,60),(0,163,96),(114,255,208),(20,207,253)]



colors = [com_colors, ap_colors]

import numpy as np
from PIL import Image
import time

com_num = []
i=0


in_name = "American-Flag-at-Sunset.jpg"
x_size = 640
y_size = 400
x_cell = 8
y_cell = 8
x_pix = 1
y_pix = 1
tot_cell = x_cell*y_cell
num_col = 4
cchoice = 0
scaling = 2

if x_cell % x_pix != 0:
    print "Error! X_cell and x_pix conflict"
    exit

if y_cell % y_pix != 0:
    print "Error! Y_cell and y_pix conflict"
    exit

if x_size % x_cell != 0:
    print "Error! X_size and x_cell conflict"
    exit

if y_size % y_cell != 0:
    print "Error! Y_size and y_pix conflict"
    exit



im=Image.open(in_name)
im = im.resize((x_size, y_size), Image.BILINEAR)

start = time.time()
pixel = get_cells(im)
new_pixels = np.empty((y_size/y_cell, x_size/x_cell, tot_cell, 3))

for i in range(y_size/y_cell):
    for j in range(x_size/x_cell):
        if same_test(pixel[i][j]):
            
            new_pixels[i][j] = [colors[cchoice][closest_color(colors[cchoice],pixel[i][j][0])] for k in range(tot_cell)]
        else:
            cell_info = mean_fit(pixel[i][j],num_col,first_closest(tot_cell, num_col))
            c = [closest_color(colors[cchoice],cell_info[1][k]) for k in range(len(cell_info[1]))]
            new_pixels[i][j]= [colors[cchoice][c[val]] for val in cell_info[0]]

im = output_im(im,new_pixels,x_cell,y_cell, x_pix, y_pix)
im = im.resize((x_size*scaling, y_size*scaling), Image.BILINEAR)
im.show()
end = time.time()

print 16*x_size*y_size/256000
print end - start

