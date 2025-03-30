
import os, sys
from psychopy import core, event, visual, sound 

# Window

def window(fullscr=False, color='black'):
    win = visual.Window(
        size=(1600, 900), fullscr=False, screen=1, allowGUI=False, allowStencil=False,
        units='pix', monitor='testMonitor', colorSpace=u'rgb', color=color
        )
    return win

# Stimuli

def circle(win, pos, color='grey', size=10, opacity=1, radius=5): 
    circle = visual.Circle(
        win=win, name='dot', units='pix',
        edges=32, ori=0, lineWidth=1, interpolate=True,
        lineColorSpace='rgb', fillColorSpace='rgb', 
        pos=pos, lineColor=color, fillColor=color, 
        size=size, opacity=opacity, 
        radius=radius,
        )
    return circle

def cross(win, pos, color='grey', size=10, opacity=1, rel_width=0.1, vertices=None): 
    
    if vertices is None:
        rel_height = 1 
        vertices = [
            (-rel_width, +rel_height),  # up
            (+rel_width, +rel_height),
            (+rel_width, +rel_width),
            (+rel_height, +rel_width),  # right
            (+rel_height, -rel_width),
            (+rel_width, -rel_width),
            (+rel_width, -rel_height),  # down
            (-rel_width, -rel_height),
            (-rel_width, -rel_width),
            (-rel_height, -rel_width),  # left
            (-rel_height, +rel_width),
            (-rel_width, +rel_width),
            ]
    
    cross = visual.shape.ShapeStim(
        win=win, units='pix', colorSpace='rgb',
        lineWidth=1, interpolate=True, 
        lineColorSpace='rgb', fillColorSpace='rgb',
        pos=pos, lineColor=color, fillColor=color, 
        size=size, opacity=opacity, 
        vertices=vertices, 
        )
    return cross

def fixation(fix_type, win, pos, color='grey', size=10, opacity=1, **kwargs):
    if fix_type == 'circle':
        return circle(win, pos, color, size, opacity, **kwargs)
    if fix_type == 'cross':
        return cross(win, pos, color, size, opacity, **kwargs)
    

def textstim(win, text, pos, color='grey', height=32):
    textstim = visual.TextStim(
        win=win, text=text,
        ori=0, pos=pos, height=height, 
        color=color, colorSpace='rgb',
        opacity=1, 
        )
    return textstim
    
def audiostim(win, audio_path='D:/gitlab/kul/cnlab/experiment-design/stimuli/audio/wav/anklung_1.wav'):
    audiostim = sound.Sound(audio_path, secs=2.0, stereo=True, hamming=True) # , sampleRate=48000) 
    return audiostim

# Keys

def check_escape(win, keys, headset=None):
    if event.getKeys(keyList=["escape"]):
        if headset is not None:
            headset.stop_recording()
            headset.disconnect()
        win.close()
        core.quit()
            

