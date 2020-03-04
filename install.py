"""
This is meant to be run on a raspberry pi.  It installs all of the
required code, sets up the python environment, and creates a systemd
service to start it on boot.
"""
import os

if os.getuid() != 0:
    print("You must run this program as root! (run sudo !!)")
    exit(1)

os.system("apt update")
os.system("apt clean")
os.system("apt autoremove")

os.system("apt -yqq install python3-pip python3-opencv git")
os.system("cd /home/pi/")
os.system("git clone https://github.com/blahblahbloopster/2898-2020-vision-new.git")

try:
    os.system("pip3 install -r /home/pi/2898-2020-vision-new/requirements.txt")
except:
    pass
os.system('echo "[Unit]\nDescription=2898\'s 2020 FRC vision code\n\n[Service]\n'
          'Type=simple\nExecStart=/bin/python3 /home/pi/2898-2020-vision-new/HexFinder.py\n'
          'Restart=on-failure\n\n[Install]\nWantedBy=multi-user.target" >> /lib/systemd/system/vision.service')
