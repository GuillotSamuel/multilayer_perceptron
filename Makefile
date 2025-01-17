YELLOW = \033[1;33m
GREEN = \033[1;32m
RED = \033[1;31m
NC = \033[0m

all: launch

launch_data_management:
	@echo "$(YELLOW)Starting data management...$(NC)"
	python3 data_management_program/main.py
	@echo "$(GREEN)Data management has ended successfully!$(NC)"

tests_data_management:
	python3 data_management_program/tests/run_tests.py

launch_predict:
	@echo "$(YELLOW)Starting model training...$(NC)"
	python3 predict_program/main.py
	@echo "$(GREEN)Model training has ended successfully!$(NC)"

tests_predict:
	python3 predict_program/tests/run_tests.py

configure:
	@echo "$(YELLOW)Starting environment configuration...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)Environment configuration completed successfully!$(NC)"

.PHONY: all launch_data_management tests_data_management configure
