# Define colors
YELLOW = \033[1;33m
GREEN = \033[1;32m
RED = \033[1;31m
NC = \033[0m

# Run the model

all: launch

launch: # Launch the model training
	@echo "$(YELLOW)Starting model training...$(NC)"
	python3 main.py
	@echo "$(GREEN)Model training has ended successfully!$(NC)"

tests: # Launch the pytest menu
	python3 tests/run_tests.py

configure: # Configure the python environment
	@echo "$(YELLOW)Starting environment configuration...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)Environment configuration completed successfully!$(NC)"

.PHONY: all launch tests configure
