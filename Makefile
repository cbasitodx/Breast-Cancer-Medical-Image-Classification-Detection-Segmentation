SHELL := /bin/bash

all:
	@echo -e "\n USE 'make env' TO GENERATE THE VIRTUALENV FOR THIS PROJECT! 👨‍💻 💬 THIS IS THE FIRST COMMAND YOU SHOULD RUN! 💬 \n"
	@echo -e "\n USE 'make doc' TO GENERATE DOCUMENTATION FOR THIS PROJECT! DOCUMENTATION IS AVAILABLE IN docs/__build/html 📁  \n"
	@echo -e "\n USE 'make tests' TO RUN TESTS FOR THIS PROJECT! 🧪"
	@echo -e "\n USE 'make clean' TO REMOVE YOUR VIRTUAL ENVIROMENT! ❌"

# Creates documentation
doc:
	(\
		echo -e "\n --------------- ACTIVATING VIRTUALENV --------------- \n"; \
		source venv/bin/activate; \
		echo -e "\n --------------- CREATING DOCUMENTATION --------------- \n"; \
		sphinx-apidoc -o ./docs ./src; \
		make -C ./docs html; \
		echo -e "\n --------------- DOCUMENTATION CREATED! (check log) --------------- \n"; \
		deactivate; \
	)

# Creates virtualenv and installs dependencies
env:
	(\
		echo -e "\n --------------- CREATING VIRTUALENV --------------- \n"; \
		python3.10 -m venv venv; \
		echo -e "\n --------------- VIRTUALENV CREATED --------------- \n"; \
		echo -e "\n --------------- ACTIVATING VIRTUALENV --------------- \n"; \
		source venv/bin/activate; \
		echo -e "\n --------------- VIRTUALENV ACTIVATED --------------- \n"; \
		echo -e "\n --------------- INSTALLING DEPENDENCIES --------------- \n"; \
		pip3 install -r requirements.txt; \
		echo -e "\n --------------- DEPENDENCIES INSTALLED --------------- \n"; \
		deactivate; \
	)

# TODO: Añadir resto de vainas