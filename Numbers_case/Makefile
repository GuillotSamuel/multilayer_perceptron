YELLOW = \033[1;33m
GREEN = \033[1;32m
RED = \033[1;31m
NC = \033[0m

# WDBC

all: launch_data_preprocessing launch_training launch_prediction

launch_data_preprocessing:
	@echo "\n$(YELLOW)Starting data preprocessing...$(NC)"
	@python3 srcs/preprocessing/preprocess.py
	@echo "$(GREEN)Data preprocessing has ended successfully!$(NC)\n"

launch_training:
	@echo "\n$(YELLOW)Starting model training...$(NC)"
	@python3 srcs/training/train.py --layer 256 128 64 --epochs 200 --loss categoricalCrossentropy --batch_size 32 --learning_rate 0.0314
	@echo "$(GREEN)Model training has ended successfully!$(NC)\n"

launch_prediction:
	@echo "\n$(YELLOW)Starting predictions...$(NC)"
	@python3 srcs/prediction/predict.py
	@echo "$(GREEN)Predictions has ended successfully!$(NC)\n"

# Utils

configure:
	@echo "$(YELLOW)Starting environment configuration...$(NC)"
	@pip install -r requirements.txt
	@echo "$(GREEN)Environment configuration completed successfully!$(NC)"


.PHONY: all launch_data_preprocessing launch_training launch_predict configure
