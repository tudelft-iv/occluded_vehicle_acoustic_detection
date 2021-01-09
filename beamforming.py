#!/usr/bin/env python3
"""
Beamforming script using Acoular for data stored in ROSBags

@author: Avinash
"""

import h5py
import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
import math
import scipy.io.wavfile as wavf
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import tables
import matplotlib

from PIL import Image, ImageSequence
from matplotlib import cm
from acoular import __file__ as bpath, MicGeom, TimeSamples, PowerSpectra, RectGrid,\
SteeringVector, BeamformerBase, BeamformerMusic, BeamformerEig, L_p, Environment, BeamformerFunctional,\
BeamformerCapon, BeamformerClean, BeamformerOrth, BeamformerCMF, BeamformerCleansc, BeamformerDamas
from celluloid import Camera
from tqdm import tqdm


#extract audio data from the .wav file
def get_mic_data(input):

    sample_rate, mic_data = wavf.read(input)
    
    return sample_rate, mic_data 

#get values from xml tags
def get_xml_field(elem, tag):

    elem = elem.find(tag)
    value = elem.text
    dtype = elem.attrib.get('type', None)

    dtype_conversion = {
        'String': lambda x: str(x) if (not x is None) else '',
        'DBL': lambda x: float(x.replace(',', '.')),
        'U16': int,
        'I32': int,
        'Enum U16': int,
        None: lambda x: x
    }
    conversion_func = dtype_conversion[dtype]
    value = conversion_func(value)

    return value

#Analyze data using Acoular
class beamformingOverlay:

    def __init__(self, wavfile, array_arrngmnt, distance=10, freq_query=1000, outfile=None, endtime=1, sample_rate=47998):

        self.wav_path        = wavfile
        self.distance        = distance
        self.interval        = 0.1              # corresponds to the frame of the video. 
        self.freq_query      = freq_query
        self.octave_band     = 2
        self.sample_rate     = sample_rate
        self.start_time      = 0
        self.end_time        = endtime
        self.alignment       = None
        self.array_arrngmnt  = array_arrngmnt
        self.grid_increment  = 0.4
        self.x_min_grid      = None  
        self.x_max_grid      = None
        self.y_min_grid      = None
        self.y_max_grid      = None
        self.outfile         = outfile

    def set_grid(self):
        """ Set the grid for Acoular. The distances and angles are centered at MicArray center point """

        width  = 2 * (self.distance - self.alignment.position_z) * math.tan( self.alignment.angle_of_view )
        height = width * self.alignment.aspect_ratio

        self.x_min_grid = -0.5*width - self.alignment.position_x
        self.x_max_grid = 0.5*width - self.alignment.position_x
        self.y_min_grid = -0.5*height + self.alignment.position_y + self.distance * math.sin(self.alignment.beta)
        self.y_max_grid = 0.5*height + self.alignment.position_y + self.distance * math.sin(self.alignment.beta) 


    def get_acoular_essentials(self):
        
        #Set the mic array geometry
        mg = MicGeom(from_file=self.array_arrngmnt)

        #Set rectangular plane and grid parameters for Acoular
        self.set_grid()
        rg = RectGrid(x_min=self.x_min_grid, x_max=self.x_max_grid, y_min=self.y_min_grid, y_max=self.y_max_grid, z=self.distance, \
            increment=self.grid_increment)

        st = SteeringVector(grid=rg, mics=mg)

        return mg, rg, st

    def do_beamforming(self, mic_data):
        """ Beamforming using Acoular """
        mg, rg, st = self.get_acoular_essentials()

        count=0
        #Divide audio samples as per frame rate (10fps) and do beamforming
        for s_time in tqdm(np.arange(self.start_time, self.end_time, self.interval)):

            audio_data  = mic_data[:, int(s_time*self.sample_rate): int((s_time+self.interval)*self.sample_rate)]
            audio_data  = np.transpose(audio_data)

            if audio_data.shape[0] == 0:
                continue
            
            #Acoular needs audio input through .h5 file
            target_file = self.outfile + '/temp.h5'

            if os.path.exists(target_file):
                os.remove(target_file)

            with h5py.File(target_file, 'w') as data_file:
                data_file.create_dataset('time_data', data=audio_data)
                data_file['time_data'].attrs.__setitem__('sample_freq', self.sample_rate)

            #.h5 file has issues with closing. Change 'ulimit' if not working

            ts = TimeSamples( name=target_file)
            ps = PowerSpectra( time_data=ts, block_size=128, window='Hanning', overlap='50%')
            bb = BeamformerEig( freq_data=ps, steer=st)
            
            pm = bb.synthetic(self.freq_query, self.octave_band )
            Lm = L_p( pm )

            if count == 0:
                bf_data = np.zeros((Lm.shape[0],Lm.shape[1],len(np.arange(self.start_time, self.end_time, self.interval))))
                bf_data[:,:,count] = Lm
            else:
                bf_data[:,:,count] = Lm

            count +=1

        # remove temp.h5 file after its finished
        os.remove(target_file)

        return bf_data, rg

    
    def run(self, outfile=None, array_tf=None):
        """ Run sequence of operations for beamforming """
        #extract audio
        sample_rate, mic_data = get_mic_data(self.wav_path)
        
        #get camera alignment parameters and array geometry
        self.alignment = cameraAlign()       

        #do beamforming
        bf_data, rg = self.do_beamforming(mic_data.T)
        
        #save beamforming results as .gif for overlay later
        gif_name = self.savePlot(bf_data, rg, outfile)
        
        return bf_data, mic_data, gif_name

    def savePlot(self, bf_data, rg, outfile=None):
            
            print("Saving plot.........")
            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(111)
            camera = Camera(fig)

            for i in range(0, bf_data.shape[2]):
                """
                setting of minimum (plot_min) is arbitrary and 
                is set by looping through all the frames and guessing a number
                """
                plot_min = bf_data[:,:,i].max() - 10
                plot_max = bf_data[:,:,i].max()

                #plot result
                im = ax.imshow(bf_data[:,:,i].T, cmap='plasma', origin='lower', vmin=plot_min, vmax=plot_max, extent=rg.extend(), interpolation='bicubic')
                
                max_bf_data = bf_data[:,:,i].max()
                
                ax.set_aspect('equal')
                for axi in (ax.xaxis, ax.yaxis):
                    for tic in axi.get_major_ticks():
                            tic.tick1line.set_visible (False)
                            tic.tick2line.set_visible (False)
                            tic.label1.set_visible (False)
                            tic.label2.set_visible (False)
                
                fig.tight_layout(pad = 0)

                #save plot to create .gif
                camera.snap()

            animation = camera.animate(interval=100, blit=True)

            gif_name =  'temp'+ '_dist_' + str(self.distance) + '_freq' + str(self.freq_query)+ '.gif'
            
            #decide where to store the file (base self.outfile)
            if self.outfile is None:
                output_dir = os.path.dirname(self.wav_path) + "/"
            else:
                output_dir = self.outfile + "/"
            
            gif_path = output_dir + gif_name
            
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            
            #save .gif
            animation.save(gif_path, writer='imagemagick')
            plt.close('all')
            
            return gif_path

class cameraAlign:

    def __init__(self):
        """
         default values are based on transformation from left stereo camera to mic_array.
         
         NOTE: Currently the set_attributes(xml_file) method takes an input from XML file 
         that also works with the beamforming software provided CAE Software & Systems GmbH
         
         
         """

        self.position_x    = -0.0316   #m
        self.position_y    = -0.485    #m
        self.position_z    = -0.567    #m
        self.alpha         = 0.0
        self.beta          = 0.11979
        self.gamma         = 0.0
        self.angle_of_view = 0.6        #rad  (half angle of view in horizontal direction)
        self.aspect_ratio  = 0.628

    def set_attributes(self, xml_file=None):

        if xml_file is None:
            pass

        else:
            with open(xml_file) as file:
                sensor_file_xml = file.read()

            #NOTE: Noise Inspector uses a custom tag with a space inside which breaks any normal XML parser ...
            sensor_file_xml = sensor_file_xml.replace('<Noise Inspector', '<Noise_Inspector')
            sensor_file_xml = sensor_file_xml.replace('</Noise Inspector', '</Noise_Inspector')

            # parse the XML
            x_tree = ET.fromstring(sensor_file_xml)

            #set attributes
            position_field = x_tree.find('alignment_in').find('position')
            self.position_x = get_xml_field(position_field, 'x')
            self.position_y = get_xml_field(position_field, 'y')
            self.position_z = get_xml_field(position_field, 'z')

            rotation_field = x_tree.find('alignment_in').find('rotation')
            self.alpha = get_xml_field(rotation_field, 'alpha')
            self.beta = get_xml_field(rotation_field, 'beta')
            self.gamma = get_xml_field(rotation_field, 'gamma')

            camera_params_field = x_tree.find('alignment_in').find('aperture_angle')
            self.angle_of_view = get_xml_field(camera_params_field, 'horizontal_angle_in_rad')
            self.aspect_ratio = get_xml_field(camera_params_field, 'picture_ratio')

#Clean the .gif to show only peaks of heatmap of the beamforming result
def removeWhitePixels(gifpath):
    
    #using PIL
    img = Image.open(gifpath)
    images = []
    
    frames = ImageSequence.Iterator(img)
    
    for frame in frames:
        
        try:
            img_mod = frame.convert("RGBA")
            datas = img_mod.getdata()

            newData = []
            #Currently only peaks stored but rest of the plot is white, so remove white pixels
            for item in datas:
                if item[0] == 255 and item[1] == 255 and item[2] == 255:
                    newData.append((255, 255, 255, 0))
                else:
                    newData.append(item)

            img_mod.putdata(newData)

            images.append(img_mod)
        
        except EOFError:
            continue
        
    path = os.path.dirname(gifpath) + "/gifoverlay.gif"
    images[0].save(path, save_all=True, append_images=images[1:], optimize=True, duration=100, loop=0, disposal=2, transparency=0)
    
    return path

def parseArgs():

    parser = argparse.ArgumentParser(description='Beamforming script using Acoular from input video and multi-channel audio files')
    
    parser.add_argument('--output', '-o', dest='output', default=None,
                        help='Destination folder of the result. Defaults to the location of the input video.')
    parser.add_argument('--distance', '-d', action='store', default=10, type=int,
                        help='Distance from the microphone array to beamforming plane')
    parser.add_argument('--frequency', '-f', action='store', default=1000, type=int,
                        help='Frequency of the signal to be used for beamforming')
    parser.add_argument('--array', '-a', dest='array',  default=None, 
                        help='.xml file defining microphone array arrangement for acoular')
    parser.add_argument('--input', dest='input',  help='Input folder containing the video and multichannel audio for beamforming', required=True)

    args = parser.parse_args()

    return args

def main():

    args = parseArgs()

    input_path = args.input
    distance   = args.distance
    frequency  = args.frequency
    output     = args.output
    array      = args.array

    audio_path = os.path.join(input_path, 'out_multi.wav')
    video_path = os.path.join(input_path, 'ueye_stereo_vid.mp4')

    if output is None:
        output = args.input
    else:
        os.makedirs(output, exist_ok=True)

    if array is None:
        array = "./config/ourmicarray_56.xml"


    # get endtime
    sample_rate, mic_data = wavf.read(audio_path)
    endtime = mic_data.shape[0]/sample_rate

    my_beamformer = beamformingOverlay(audio_path, array, distance, frequency, output, endtime, sample_rate=sample_rate)
    bf_data, mic_data, gif_name = my_beamformer.run()

    gif_path = removeWhitePixels(gif_name)
    
    #remove .gif not used for overlay
    os.remove(gif_name)

    # video_path   = bag_folder + "/ueye_stereo_vid.mp4"
    overlay_path = os.path.join(*[output, str(frequency) + "_" + str(distance) + "_overlay.mp4"]) 

    # #remove previously generated to avoid input to overwrite query by ffmpeg
    if os.path.exists(overlay_path):
        os.remove(overlay_path)

    os.system('ffmpeg -hide_banner -loglevel panic -i {0} -i {1} -filter_complex "[1]format=argb,colorchannelmixer=aa=0.4[front];[front]scale=2230:1216[next];[0][next]overlay=x=-155:y=0,format=yuv420p" {2}'.format(video_path, gif_path, overlay_path))

    #remove .gif after ffmpeg finishes overlay
    os.remove(gif_path)

    print("Done....")


if __name__ == '__main__':
    main()
