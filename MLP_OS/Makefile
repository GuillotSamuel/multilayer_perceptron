YELLOW = \033[1;33m
GREEN = \033[1;32m
RED = \033[1;31m
NC = \033[0m

all: launch_data_preprocessing launch_training launch_prediction

launch_data_preprocessing:
	@echo "\n$(YELLOW)Starting data preprocessing...$(NC)"
	@python3 srcs/preprocessing_WDBC/preprocess.py
	@echo "$(GREEN)Data preprocessing has ended successfully!$(NC)\n"

launch_training:
	@echo "\n$(YELLOW)Starting model training...$(NC)"
	@python3 srcs/training/train.py --layer 24 24 12 --epochs 30000 --loss binaryCrossentropy --batch_size 8 --learning_rate 0.0314
	@echo "$(GREEN)Model training has ended successfully!$(NC)\n"

launch_prediction:
	@echo "\n$(YELLOW)Starting predictions...$(NC)"
	@python3 srcs/prediction/predict.py
	@echo "$(GREEN)Predictions has ended successfully!$(NC)\n"

configure:
	@echo "$(YELLOW)Starting environment configuration...$(NC)"
	@pip install -r requirements.txt
	@echo "$(GREEN)Environment configuration completed successfully!$(NC)"

.PHONY: all launch_data_preprocessing launch_training launch_predict configure
