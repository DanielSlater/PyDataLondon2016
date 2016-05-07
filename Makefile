bootstrap:
	brew cask install xquartz
	brew install socat
build:
	docker build -t pydatalondon2016 
run:
	socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\" &
	docker run --net=host -it -e DISPLAY=`ipconfig getifaddr en0`:0 pydatalondon2016 /bin/bash

all: bootstrap build run
