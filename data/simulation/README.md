

```bash

# You can bind any directory on your system to docker image by using -v flag:
# -v <your/directory>:<docker/directory>
# Convenient place inside docker image is
# /media/share
docker run -it --rm -v $(pwd):/media/share eicweb/jug_xl:nightly



# For X11 with new web root browser
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/media/share --rm -it --user $(id -u) -p 9114:9114 eicweb/jug_xl:nightly

# Chain
docker run -it --rm -v $(pwd):/media/share eicweb/jug_xl:nightly
source /opt/detector/ecce-nightly/setup.sh
cd /media/share/data/simulation
npsim --compactFile=$DETECTOR_PATH/ecce.xml --runType=run --enableG4GPS --macro disk_particle_gun.mac --outputFile=/media/share/data/disk_gun_electrons_0-15GeV_100ev.edm4hep.root 

```

docker run -it --rm -v $(pwd):/media/share eicweb/jug_xl:nightly