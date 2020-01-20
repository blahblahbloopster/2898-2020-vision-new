#!/usr/bin/python3

import socket
import time
from subprocess import check_output


def run(cmd):
    output = str(check_output(cmd, shell=True), "utf-8")
    return output


while True:
    UDP_IP = "255.255.255.255"
    UDP_PORT = 2898
    MESSAGE = run("hostname -I").encode("ascii")
    sock = socket.socket(socket.AF_INET,  # Internet
                         socket.SOCK_DGRAM)  # UDP
    sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
    time.sleep(30)
