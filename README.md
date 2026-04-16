## Run With Docker
To run the app using Docker:


### Open a terminal and navigate to the project folder:
cd streamlit


### Build the Docker image:
docker build --no-cache -t animal-adoption-app .


### Run the Docker container:
docker run --rm -p 8501:8501 animal-adoption-app


### Open the app in your browser:
http://localhost:8501


### To stop the container, return to the terminal and press:
Ctrl + C
