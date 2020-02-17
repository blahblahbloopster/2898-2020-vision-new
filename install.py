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
os.system("apt -yqq install python3-pip python3-opencv")
os.system("git clone https://github.com/blahblahbloopster/2898-2020-vision-new.git /home/pi/")
os.system("cd /home/pi/2898-2020-vision-new/")
try:
    os.system("pip3 install -r requirements.txt")
except:
    pass
with open("/lib/systemd/system/vision.service") as f:
    f.write("""[Unit]
Description=2898's 2020 FRC vision code

[Service]
Type=simple
ExecStart=/bin/python3 /home/pi/2898-2020-vision-new/HexFinder.py
Restart=on-failure

[Install]
WantedBy=multi-user.target""")

os.system("systemctl enable vision.service")
os.system("systemctl start vision.service")
