from pathlib import Path
from sys import flags

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

import threading
from time import sleep
import pandas as pd
import pygame
pygame.mixer.init()

import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)

from model import model
import torch
import numpy as np
import serial

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import serial_server

# FAN pin
fan = 21
GPIO.setup(fan, GPIO.OUT)
GPIO.setwarnings(False)

# GPIO ports for the digit 7seg pins
segments = (11,4,23,8,7,10,18,25)
for segment in segments:
	GPIO.setup(segment, GPIO.OUT)
	GPIO.output(segment, 1)

# GPIO ports for the digit 0-3 pins
digits = (22,27,17,24)
for digit in digits:
	GPIO.setup(digit, GPIO.OUT)
	GPIO.output(digit,1)

num = {' ':(1,1,1,1,1,1,1),
	'0':(0,0,0,0,0,0,1),
	'1':(1,0,0,1,1,1,1),
	'2':(0,0,1,0,0,1,0),
	'3':(0,0,0,0,1,1,0),
	'4':(1,0,0,1,1,0,0),
	'5':(0,1,0,0,1,0,0),
	'6':(0,1,0,0,0,0,0),
	'7':(0,0,0,1,1,1,1),
	'8':(0,0,0,0,0,0,0),
	'9':(0,0,0,0,1,0,0),
	'C':(0,1,1,0,0,0,1),
    '-':(1,1,1,1,1,1,0)}

#temp = str('205C')


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")

#### pygame music ####
pygame.mixer.music.set_volume(0.4)

sound_tired = pygame.mixer.Sound('./music/tired.wav')
sound_caution = pygame.mixer.Sound('./music/caution.wav')

sound_tired.set_volume(1.0)
sound_caution.set_volume(1.0)

music_time = 0
music_flag = 1
music_name = "moon"
def play(music: str, condition):
    global music_flag, music_name
    if music_flag==1 and music_name!=music:
        if condition == 'tired':
            sound_tired.play()
        elif condition == 'caution':
            sound_caution.play()
        song = f'./music/{music}.mp3'
        pygame.mixer.music.load(song)
        pygame.mixer.music.play()
        music_flag = 0
        music_name = music
play("moon","nothing")

paused = False 
def pause(is_paused):
    global paused
    paused = is_paused
    if paused:
        pygame.mixer.music.unpause()
        paused = False
    else:
        pygame.mixer.music.pause()
        paused = True

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def fan_control(ch: int):
    GPIO.output(fan, ch)

def seven_seg(temp: str):
    start_time = time.time()
    while True:
        if time.time()-start_time<=1:
            for digit in range(4):
                for loop in range(0,7):
                    GPIO.output(segments[loop], num[temp[digit]][loop])
                    if digit == 1:
                        GPIO.output(25, 0)
                    else:
                        GPIO.output(25, 1)
                GPIO.output(digits[digit], 1)
                time.sleep(0.001)
                GPIO.output(digits[digit], 0)
        else:
            break

def Fatigue(F):
    global image_4

    if F == 1:
        canvas.itemconfig(image_4, image = F_good)
    elif F == 3:
        canvas.itemconfig(image_4, image = F_bad)
    else:
        canvas.itemconfig(image_4, image = F_mid)
    
def Stress(S):
    global image_5

    if S == 1:
        canvas.itemconfig(image_5, image = S_good)
    elif S == 3:
        canvas.itemconfig(image_5, image = S_bad)
    else:
        canvas.itemconfig(image_5, image = S_mid)


flag = 1
def banner(F, S, arrive):
    global image_12
    global image_6
    global image_1
    global flag

    if arrive == 0:
        if F == 2 or S == 2:
            play('fight', 'tired')
            fan_control(1)
            canvas.itemconfig(image_6, image = banner_tired)
            canvas.itemconfig(image_1, image = music_awake)
            canvas.itemconfig(image_12, image = screen_yellow)
            seven_seg('205C')
            #sleep(0.5)
        elif F == 3 or S == 3:
            play('fight', 'caution')
            fan_control(1)
            canvas.itemconfig(image_6, image = banner_caution)
            canvas.itemconfig(image_1, image = music_awake)
            canvas.itemconfig(image_12, image = screen_red)
            seven_seg('205C')
            #sleep(0.5)
        else:
            play("moon","nothing")
            fan_control(0)
            canvas.itemconfig(image_6, image = banner_good)
            canvas.itemconfig(image_12, image = screen_empty)
            seven_seg('----')
            #sleep(0.5)
    else: # 운행종료
        if flag == 1:
            fan_control(0)
            canvas.delete(image_7, image_8, image_9, image_10, image_12)
            canvas.itemconfig(image_6, image = banner_arrive)
            image_11 = canvas.create_image(
                313.0,
                368.0,
                image = health_checkup
            )
            flag = 0
        


def health_LF(lf):
    global image_7

    if lf <= 20:
        canvas.itemconfig(image_7, image = LF_low)
    else:
        canvas.itemconfig(image_7, image = LF_normal)

def health_HF(hf):
    global image_8

    if hf <= 65:
        canvas.itemconfig(image_8, image = HF_low)
    else:
        canvas.itemconfig(image_8, image = HF_normal)
    

def health_LF_HF(lf_hf):
    global image_9

    if lf_hf >= 1:
        canvas.itemconfig(image_9, image = LF_HF_high)
    elif lf_hf < 0.2:
        canvas.itemconfig(image_9, image = LF_HF_low)
    else:
        canvas.itemconfig(image_9, image = LF_HF_good)

def text_score(sdnn, lf, hf, lf_hf, hr):
    canvas.itemconfig(text_SDNN, text=int(sdnn))
    canvas.itemconfig(text_LF, text=int(lf))
    canvas.itemconfig(text_HF, text=int(hf))
    canvas.itemconfig(text_LF_HF, text=round(lf_hf,1))
    canvas.itemconfig(text_HR, text=int(hr))

######################################
############## 초기 상태 ##############
######################################
window = Tk()

window.geometry("800x480")
window.attributes('-fullscreen',True)
window.configure(bg = "#F5F4F5")


canvas = Canvas(
    window,
    bg = "#F5F4F5",
    height = 480,
    width = 800,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
music = PhotoImage(
    file=relative_to_assets("music.png"))
image_1 = canvas.create_image(
    695.0,
    240.0,
    image=music
)

screen_empty = PhotoImage(
    file=relative_to_assets("screen_empty.png"))
image_12 = canvas.create_image(
    309.0,
    245.0,
    image=screen_empty
)

driver_condition = PhotoImage(
    file=relative_to_assets("driver_condition.png"))
image_2 = canvas.create_image(
    104.0,
    33.0,
    image=driver_condition
)

health_care = PhotoImage(
    file=relative_to_assets("health_care.png"))
image_3 = canvas.create_image(
    121.0,
    245.0,
    image=health_care
)

F_good = PhotoImage(
    file=relative_to_assets("F_good.png"))
image_4 = canvas.create_image(
    92.0,
    125.0,
    image=F_good
)

S_good = PhotoImage(
    file=relative_to_assets("S_good.png"))
image_5 = canvas.create_image(
    517.0,
    125.0,
    image=S_good
)

banner_km = PhotoImage(
    file=relative_to_assets("banner_km.png"))
image_6 = canvas.create_image(
    304.0,
    137.0,
    image=banner_km
)

LF_normal = PhotoImage(
    file=relative_to_assets("LF_normal.png"))
image_7 = canvas.create_image(
    456.0,
    302.0,
    image=LF_normal
)

HF_normal = PhotoImage(
    file=relative_to_assets("HF_normal.png"))
image_8 = canvas.create_image(
    456.0,
    369.0,
    image=HF_normal
)

LF_HF_good = PhotoImage(
    file=relative_to_assets("LF_HF_good.png"))
image_9 = canvas.create_image(
    456.0,
    435.0,
    image=LF_HF_good
)

health_board = PhotoImage(
    file=relative_to_assets("health_board.png"))
image_10 = canvas.create_image(
    169.0,
    368.0,
    image=health_board
)

text_SDNN = canvas.create_text(
    92.0,
    417.0,
    anchor="nw",
    text="1",
    fill="#000000",
    font=("Inter Light", 28 * -1)
)

text_LF = canvas.create_text(
    239.0,
    283.0,
    anchor="nw",
    text="2",
    fill="#000000",
    font=("Inter Light", 20 * -1)
)

text_HF = canvas.create_text(
    239.0,
    322.0,
    anchor="nw",
    text="3",
    fill="#000000",
    font=("Inter Light", 20 * -1)
)

text_LF_HF = canvas.create_text(
    239.0,
    361.0,
    anchor="nw",
    text="4",
    fill="#000000",
    font=("Inter Light", 20 * -1)
)

text_HR = canvas.create_text(
    209.0,
    417.0,
    anchor="nw",
    text="5",
    fill="#000000",
    font=("Inter Light", 28 * -1)
)

health_checkup = PhotoImage(
    file=relative_to_assets("health_checkup.png"))
''' 운행 종료시에만 표출
image_11 = canvas.create_image(
    313.0,
    368.0,
    image=health_checkup
)'''

''' 휴게소 안내 km
text_km = canvas.create_text(
    236.0,
    142.0,
    anchor="nw",
    text="1.2km",
    fill="#FFFFFF",
    font=("Inter SemiBold", 40 * -1)
)'''

pause_img = PhotoImage(
    file=relative_to_assets("button_pause.png"))
button_pause = Button(
    image=pause_img,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: pause(paused),
    relief="flat"
)
button_pause.place(
    x=680.0,
    y=404.0,
    width=50.0,
    height=50.0
)



# 그 외 선언
banner_caution = PhotoImage(
    file=relative_to_assets("banner_caution.png"))

banner_good = PhotoImage(
    file=relative_to_assets("banner_good.png"))

banner_tired = PhotoImage(
    file=relative_to_assets("banner_tired.png"))

banner_arrive = PhotoImage(
    file=relative_to_assets("banner_arrive.png"))

F_bad = PhotoImage(
    file=relative_to_assets("F_bad.png"))

F_mid = PhotoImage(
    file=relative_to_assets("F_mid.png"))

HF_low = PhotoImage(
    file=relative_to_assets("HF_low.png"))

LF_HF_high = PhotoImage(
    file=relative_to_assets("LF_HF_high.png"))

LF_HF_low = PhotoImage(
    file=relative_to_assets("LF_HF_low.png"))

LF_low = PhotoImage(
    file=relative_to_assets("LF_low.png"))

music_awake = PhotoImage(
    file=relative_to_assets("music_awake.png"))

S_bad = PhotoImage(
    file=relative_to_assets("S_bad.png"))

S_mid = PhotoImage(
    file=relative_to_assets("S_mid.png"))

screen_red = PhotoImage(
    file=relative_to_assets("screen_red.png"))

screen_yellow = PhotoImage(
    file=relative_to_assets("screen_yellow.png"))




##################################
############## 제어 ##############
##################################
# df = pd.read_csv(OUTPUT_PATH / Path('RppgFeature.csv'))
# cnt = 0
# F = 0
# S = 0
# lf = 0
# hf = 0
# lf_hf = 0
# arrive = 0
# sdnn = 0
# hr = 0

def upd(cnt, F, S, lf, hf, lf_hf, arrive, sdnn, hr):

    '''
    Example value
    '''
    global music_time, music_flag
    # global cnt, F, S, lf, hf, lf_hf, arrive, sdnn, hr, music_time, music_flag
    # if cnt < len(df):
    #     F = df['F'][cnt]
    #     S = df['S'][cnt]
    #     arrive = df['arrive'][cnt]
    #     lf = df['LF'][cnt]
    #     hf = df['HF'][cnt]
    #     lf_hf = df['LF/HF'][cnt]
    #     sdnn = df['SDNN'][cnt]
    #     hr = df['HR'][cnt]
    #
    #     cnt+=1

    text_score(sdnn, lf, hf, lf_hf, hr)
    Fatigue(F)
    Stress(S)
    health_LF(lf)
    health_HF(hf)
    health_LF_HF(lf_hf)
    banner(F,S,arrive)
    
    music_time+=1
    if music_time > 6:
        music_time=0
        music_flag=1

    threading.Timer(0.01, upd).start()

def main():
    '''
    from jetson to Raspberry
    '''
    global music_time, music_flag
    Fatigue_score, stress, lf, hf, lf_hf, sdnn, hr, cnt, arrive = 0, 0, 0, 0, 0, 0, 0, 0, 0

    root_path = '/home/ubuntu'

    q = deque([])
    if not os.path.exists(root_path + "/data"):
        os.mkdir(root_path + "/data")
    DataPath = root_path + "/data/data.txt"
    server_port = serial.Serial(
        port="/dev/ttyS0",
        baudrate=9600,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )

    t = threading.Thread(target=serial_server.serial_server, args=(server_port, DataPath, q))
    t.start()
    Model = model.LinearModel(9, 1)
    state_dict = "./state_dict/best_mse_model.pth"
    Model.load_state_dict(torch.load(state_dict))

    while True:
        time.sleep(5)
        with open(DataPath, 'r') as f:
            data = f.readlines()
        if len(data[-1]) == 0:
            data = data[-2]
        else:
            data = data[-1]

        data = np.array(list(map(float, data.split(","))))
        hr = data[0]
        sdnn = data[2]
        lf = data[3]
        hf = data[4]
        lf_hf = data[5]
        print(data, len(data))

        try:
            if len(data) == 9:
                data = torch.FloatTensor(data)
                Fatigue_score = Model(data)
                upd(cnt, Fatigue_score, stress, lf, hf, lf_hf, arrive, sdnn, hr)
                cnt += 1
                if cnt > 100:
                    arrive = 1
                print("Fatigue_score : ", Fatigue_score)
        except:
            pass
    return

main()
window.resizable(False, False)
window.mainloop()
GPIO.cleanup()
