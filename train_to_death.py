import subprocess
import time
import sys

path = sys.argv[1]

while 1:
    process = subprocess.run(['python3','main.py'])
    if process.returncode == 0:
        break
    time.sleep(5)
import os
subprocess.run(['python3','plot.py','--no-img'])

path = './trained_data/'+path
os.system('mkdir '+path)
os.system('cp q_network.pth target_q_network.pth variables.pkl score.data score.png '+path)
