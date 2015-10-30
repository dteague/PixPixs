def color_num(r, g, b):
    return r+256*g+256**2*b


# Algorithm for finding closest color in avaliable colors to the input
#    Uses pythagorean method first, then if the same, uses ^3 length.
#
# TODO: can't deal with pixels same distance away in every axis.  Sets to -1
#    For now, but needs way to resolve this better
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

    
# Takes the initial picture and divides it in to the cells (as well as getting the right
#     pixel size) and puts this into an array.  The array is set up as such:
#
#     pixels[cell in y direction][cell in x direction][pixel in cell][rgb values of pixel]
#
#     The pixels in the cell are arraged so all pixels in const row are read, then the next
#     row of pixels is read.
def get_cells(im,x_cell, y_cell, x_pix, y_pix):
    new_pix = np.zeros((im.size[1]/y_cell, im.size[0]/x_cell, x_cell*y_cell/(x_pix*y_pix), 3))

    for (i,j) in [(i,j) for j in range(0,im.size[1]/y_cell) for i in range(0,im.size[0]/x_cell)]:
        for (k,l) in [(k,l) for l in range(0,y_cell/y_pix) for k in range(0,x_cell/x_pix)]:
            for (m,n) in [(m,n) for n in range(0,y_pix) for m in range(0,x_pix)]:
                new_pix[j][i][k+x_cell/x_pix*l] += im.getpixel((x_cell*i+x_pix*k+m, y_cell*j+y_pix*l+n)) 

    new_pix = new_pix/(x_pix*y_pix)
    
            

    # pixels = np.zeros((im.size[1]/y_cell, im.size[0]/x_cell, x_cell*y_cell,3))
    # for i in range(im.size[1]/y_cell):
    #     for j in range(im.size[0]/x_cell):
    #         pixels[i][j] = [im.getpixel((x,y)) for y in range(i*y_cell, (i+1)*y_cell) for x in range(j*x_cell,(j+1)*x_cell)]
            
    return new_pix


def get_grey(pixel):
    return 0.21*pixel[0] + 0.72*pixel[1] + 0.07*pixel[2]


# Algorithm for finding best partition of pixels to their colors.  Done
#    by finding the mean of colors in a partition, then repartitioning
#    the colors based on the closeness to the new means.  Ends when partition
#    doesn't change anymore.
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

    tmp = closest(array,close,m, sections)
    if (tmp != close).any():
        return mean_fit(array, sections, tmp)


    listing = []
    listing.append(tmp)
    listing.append(m)
        
    return listing


# Returns an array with first partioning of pixels in cell.
#    This is done in a simple even split
def first_closest(size, sections):

    array = np.empty(size)
    space = len(array)/sections
    for i in range(sections-1):
        array[space*i:space*(i+1)] = i
    array[space*(sections-1):] = sections-1

    return array


# When the number of colors needed in a cell is less the maximum,
#   this function fixes the partition matrix to send it in with one
#   one less color
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


# Repartitions the pixels based on the average value of the colors
#   from the previous partition
def closest(array, close, m, sections):

    dif = np.zeros((len(array), sections))
    for (i,j) in [(i,j) for i in range(len(array)) for j in range(sections)]:
        dif[i][j] += np.sum((m[j]-array[i])**2)

    close= np.argmin(dif,axis=1)
    return close


# Takes old pix and colors to put the new colors in the pic
def output_im(im, pixels,x_cell, y_cell, x_pix, y_pix):
    
    picture = im.load()
    nx_cells = im.size[0]/x_cell
    ny_cells = im.size[1]/y_cell


    print im.size[0]
    print (im.size[0]/x_cell-1)*(x_cell) + x_pix*(x_cell/x_pix-1)+1

    for (i,j) in [(i,j) for j in range(0,im.size[1]/y_cell) for i in range(0,im.size[0]/x_cell)]:
        for (k,l) in [(k,l) for l in range(0,y_cell/y_pix) for k in range(0,x_cell/x_pix)]:
            pix=pixels[j][i][k+l*x_cell/x_pix]
            for (m,n) in [(m,n) for n in range(0,y_pix) for m in range(0,x_pix)]:
                picture[x_cell*i + x_pix*k + m , y_cell*j +y_pix*l +n]=(int(pix[0]),int(pix[1]),int(pix[2]))

    return im


######################################################################################################################################################
############################################################ Start of Main ###########################################################################
######################################################################################################################################################

com_colors = [(0,0,0),(255,255,255),(136,0,0),(170,255,238),(204,68,204),(0,204,85),(0,0,170),(238,238,119),(221,136,85),(102,68,0),(255,119,119),(51,51,51),(119,119,199),(170,255,102),(0,136,255),(187,187,187)]

ap_colors = [(0,0,0),(156,156,156),(255,255,255),(96,78,189),(208,195,255),(255,68,253),(227,30,96),(255,160,200),(255,106,60),(96,114,3),(208,221,141),(20,245,60),(0,163,96),(114,255,208),(20,207,253)]



colors = [com_colors, ap_colors]

import numpy as np
from PIL import Image
import time

com_num = []
i=0

#########################################################################
########## All variables used, change these to change picture! ##########
#########################################################################

in_name = "ferris.jpg"
x_size = 320
y_size = 200
x_cell = 1
y_cell = 1
x_pix = 1
y_pix = 1
tot_cell = x_cell*y_cell/(x_pix*y_pix)
num_col = 16
cchoice = 0
scaling = 4

#########################################################################
#########################################################################

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
pixel = get_cells(im,x_cell, y_cell,x_pix,y_pix)
new_pixels = np.empty((y_size/y_cell, x_size/x_cell, tot_cell, 3))

for i in range(y_size/y_cell):
    for j in range(x_size/x_cell):
        cell_info = mean_fit(pixel[i][j],num_col,first_closest(tot_cell, num_col))
        c = [closest_color(colors[cchoice],cell_info[1][k]) for k in range(len(cell_info[1]))]
        new_pixels[i][j]= [colors[cchoice][c[val]] for val in cell_info[0]]

im = output_im(im,new_pixels,x_cell,y_cell, x_pix, y_pix)
im = im.resize((x_size*scaling, y_size*scaling), Image.BILINEAR)
im.show()
end = time.time()

print 16*x_size*y_size/256000
print end - start

