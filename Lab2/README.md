# Lab 2

We took the code from Lab1 and added a FastAPI endpoint for each function.
To do this we also created Model classes for the request bodies, to receive the parameters.
The endpoints contain some error handling.

Then we put our project inside a docker image, making sure to install our requirements and expose the server port.

We found a docker image online with ffmpeg inside it and tried to use it.

We attempted to create a docker compose file to allow our service to use the ffmpeg docker image, but it didn't work.
We added volumes to share the input and output files between the containers and a shared network.
Our first idea was to run the ffmpeg binary in the container from our server container, but this was unsuccessful.

The next thing we tried was using a connection with netcat to send our command to the ffmpeg container.
To do this we had to create our own ffmpeg docker (Dockerfile in Lab3).
But we got the error "/bin/sh: 1: nc: not found" and even after installing netcat in the container the error somehow persisted.
