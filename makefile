train:
	python main.py --train
test-lstm:
	python main.py --test --model saved/lstm_dqn_model_200.pkl --model_type lstm
test-attention:
	python main.py --test --model saved/attention_dqn_model_150.pkl --model_type attention
