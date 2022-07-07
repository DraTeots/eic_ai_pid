

```bash

# You can bind any directory on your system to docker image by using -v flag:
# -v <your/directory>:<docker/directory>
# Convenient place inside docker image is
# /media/share
docker run -it --rm -v $(pwd):/media/share eicweb/jug_xl:nightly

# For X11 with new web root browser
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/media/share --rm -it --user $(id -u) -p 9114:9114 eicweb/jug_xl:nightly

source /opt/detector/ecce-nightly/setup.sh

npsim --compactFile=$DETECTOR_PATH/ecce.xml --runType=run --enableGun -N=2 --outputFile=/media/share/data/test_gun.edm4hep.root --gun.position "0.0 0.0 1.0*cm" --gun.direction "0.1 0.0 -1.0" --gun.energy 100*GeV --gun.particle "e-"

```