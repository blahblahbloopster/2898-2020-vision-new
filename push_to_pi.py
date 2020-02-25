"""
This script runs on a linux host, and pushes the code to the SD card
"""
import os

# Makes sure you aren't root
if os.getuid() == 0:
    print("You must not run this program as root!")
    exit(1)

# Gets username
username = os.environ["USERNAME"]

# Checks if bare repo doesn't already exist
if not os.path.exists("/media/%s/rootfs/var/lib/git/2898-2020-new.git/" % username):
    # Makes repo
    os.system("sudo git init --shared=0777 --bare /var/lib/git/2898-2020-new.git")
    # Allows force pushes
    os.system("sudo sed -i -e 's/denyNonFastforwards = true/denyNonFastforwards = false/'"
              " /media/%s/rootfs/var/lib/git/awesomesauce.git/config" % username)

# Pushes to bare repo
os.system("git push --force /media/%s/rootfs/var/lib/git/2898-2020-vision-new.git" % username)

# Unmounts SD card
os.system("sudo umount /media/%s/boot" % username)
os.system("sudo umount /media/%s/rootfs" % username)
