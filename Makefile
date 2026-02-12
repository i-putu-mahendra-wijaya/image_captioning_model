requirements:
	# backup existing requirements.txt
	cp conda_env.yaml conda_env.yaml.backup;

	# create new requirements.txt
	conda env export > conda_env.yaml

src_importable:
	python3 -m pip install -e .

config_yaml:
	# run config_creator to automatically create a new config.yaml
	python3 config_creator.py

main:
	# run the create_feature_vector.py
	python3 main.py