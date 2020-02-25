import os

os.system("systemctl stop vision.service")
os.system("cd /var/lib/git/2898-2020-vision-new.git;"
          " git archive --format=tgz --prefix=awesome/ HEAD | "
          "bash -c 'cd /home/pi; tar xvz'")
os.system("systemctl start vision.service")
