import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)
import random
from scipy.signal import fftconvolve

def get_random_room():    
    length, width, height = random.uniform(35,45),random.uniform(12,25),random.uniform(15,28)
    
    room_dim = [length, width, height]  # meters

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(random.uniform(1.5, 4), room_dim)

    # Create the room
    room = pra.ShoeBox(
        room_dim, fs=22050, materials=pra.Material(e_absorption), max_order=max_order
    )
    
    dir_obj = CardioidFamily(
        orientation=DirectionVector(azimuth=90, colatitude=15, degrees=True),
        pattern_enum=DirectivityPattern.HYPERCARDIOID,
    )
    
    mic_x, mic_y, mic_z = random.uniform(0, length), random.uniform(0, width), random.uniform(0, height)
    
    room.add_source(position=[random.uniform(3, 10), random.uniform(3, 10), random.uniform(.9, 1.3)], directivity=dir_obj)
    room.add_microphone(loc=[mic_x, mic_y, mic_z], directivity=dir_obj)
    
    
    return room

def get_room_impulse(room):
    room.compute_rir()
    ir = room.rir[0][0]
    return ir

def convolve_with_room(x, ir):
    return fftconvolve(x, ir)