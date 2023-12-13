import subprocess
import time

while 1:
    process = subprocess.run(['python3','main.py'])
    if process.returncode == 0:
        break
    time.sleep(5)
