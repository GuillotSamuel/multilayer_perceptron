YELLOW = \033[1;33m
GREEN = \033[1;32m
RED = \033[1;31m
NC = \033[0m

all: launch_data_management launch_training launch_prediction

launch_data_management:
	@echo "\n$(YELLOW)Starting data management...$(NC)"
	@python3 data_management_program/main.py
	@echo "$(GREEN)Data management has ended successfully!$(NC)\n"

tests_data_management:
	@python3 data_management_program/tests/run_tests.py

launch_training:
	@echo "\n$(YELLOW)Starting model training...$(NC)"
	@python3 training_program/main.py
	@echo "$(GREEN)Model training has ended successfully!$(NC)\n"

tests_training:
	@python3 training_program/tests/run_tests.py

launch_prediction:
	@echo "\n$(YELLOW)Starting predictions...$(NC)"
	@python3 prediction_program/main.py
	@echo "$(GREEN)Predictions has ended successfully!$(NC)\n"

tests_prediction:
	@python3 prediction_program/tests/run_tests.py

configure:
	@echo "$(YELLOW)Starting environment configuration...$(NC)"
	@pip install -r requirements.txt
	@echo "$(GREEN)Environment configuration completed successfully!$(NC)"

.PHONY: all launch_data_management tests_data_management launch_training tests_training launch_predict tests_predict configure
