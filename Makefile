DATA_DIR="data/"

camrest:
######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2017 #
######################################################################
######################################################################
	mkdir -p $(DATA_DIR)/camrest
	wget https://github.com/shawnwun/NNDIAL/raw/master/data/CamRest676.json -O $(DATA_DIR)/camrest/data.json
	wget https://github.com/shawnwun/NNDIAL/raw/master/resource/CamRestOTGY.json -O $(DATA_DIR)/camrest/otgy.json
	wget https://github.com/shawnwun/NNDIAL/raw/master/db/CamRest.json -O $(DATA_DIR)/camrest/db.json
	sed -i '/^#/d' $(DATA_DIR)/camrest/data.json
	sed -i '/^#/d' $(DATA_DIR)/camrest/db.json
	sed -i '/^#/d' $(DATA_DIR)/camrest/otgy.json
	python scripts/split_json_data.py --data_fn $(DATA_DIR)/camrest/data.json --target_dir $(DATA_DIR)/camrest

woz:
	mkdir -p $(DATA_DIR)/multiwoz
	wget "https://www.repository.cam.ac.uk/bitstream/handle/1810/294507/MULTIWOZ2.1.zip?sequence=1&isAllowed=y" -O $(DATA_DIR)/multiwoz/raw.zip
	cd $(DATA_DIR)/multiwoz && unzip raw.zip
	cp $(DATA_DIR)/multiwoz/MULTIWOZ2.1/ontology.json $(DATA_DIR)/multiwoz/otgy.json
	python scripts/process_otgy.py --otgy $(DATA_DIR)/multiwoz/otgy.json
	cp $(DATA_DIR)/multiwoz/MULTIWOZ2.1/data.json $(DATA_DIR)/multiwoz/data.json
	python scripts/split_json_data.py --data_fn $(DATA_DIR)/multiwoz/data.json --target_dir $(DATA_DIR)/multiwoz

smd:
	mkdir -p $(DATA_DIR)/smd
	wget "http://nlp.stanford.edu/projects/kvret/kvret_dataset_public.zip" -O $(DATA_DIR)/smd/raw.zip
	cd $(DATA_DIR)/smd && unzip raw.zip
	cp $(DATA_DIR)/smd/kvret_entities.json $(DATA_DIR)/smd/otgy.json
	python scripts/process_otgy.py --otgy $(DATA_DIR)/smd/otgy.json
	cp $(DATA_DIR)/smd/kvret_dev_public.json $(DATA_DIR)/smd/valid.json
	cp $(DATA_DIR)/smd/kvret_train_public.json $(DATA_DIR)/smd/train.json
	cp $(DATA_DIR)/smd/kvret_test_public.json $(DATA_DIR)/smd/test.json
