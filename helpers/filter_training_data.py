import os
import json
import codecs

input_file = 'training_data_eng.subintent.json'
data = json.load(codecs.open(input_file, 'r', 'utf-8'))
out_data = {"timestamp": 1549361454, "test_cases":{}}
tcss = out_data['test_cases']
for conv_id, conv in data.get('test_cases', {}).iteritems():
    if conv[0].get("sitedomain",'') == "Rovi-MusicSpace-en":
	continue
    tcss[conv_id] = conv 

out_file = ''.join([input_file,'filtered'])
with codecs.open(out_file, "w", encoding="utf-8") as fp:
    json.dump(out_data, fp, ensure_ascii = False,
		indent = 2, separators = (',', ': '))
