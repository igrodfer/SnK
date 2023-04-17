import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype('uint8')


def segment_image(image,tile_size,margin_top=0,margin_left=0):

    image_no_margin = image[margin_top:,margin_left:]

    img_height , img_width = image_no_margin.shape
    
    new_height = ((img_height)//tile_size)*tile_size
    new_width = ((img_width)//tile_size)*tile_size

    image_res = image_no_margin[:new_height,:new_width]

    tiled_array =  image_res.reshape((img_height) // tile_size,
                                 tile_size,
                                 (img_width) // tile_size,
                                 tile_size)

    tiled_array = tiled_array.swapaxes(1,2)

    return np.concatenate(tiled_array,axis=0),(new_height,new_width)